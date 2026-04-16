from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.app.services.deploy_config_service import load_deploy_config_bundle
from backend.app.services.inference_service import run_inference_pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_PATH = (
    Path("/home/zennakamura/MasterResearch/HyperSIGMA")
    / "HyperspectralDetection"
    / "Hyperion_WaterLabel_20111222"
    / "data"
    / "processed"
    / "hyperion"
    / "hyperion_stack_crop_f16.tif"
)
DEFAULT_OUTPUT_JSON = PROJECT_ROOT / "experiments" / "eval_v3" / "gui_smoke_test_result.json"
REQUIRED_OUTPUT_KEYS = [
    "probability_map_png",
    "probability_overlay_png",
    "spatial_attention_overlay_png",
    "spectral_attention_png",
    "metadata_json",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smoke-test the deploy-config-driven GUI inference pipeline.")
    parser.add_argument(
        "--deploy_config",
        type=str,
        default="configs/deploy/hypersigma_v3_calibrated.json",
        help="Deploy config JSON to use.",
    )
    parser.add_argument("--input_path", type=str, default=str(DEFAULT_INPUT_PATH), help="HSI cube used for the smoke test.")
    parser.add_argument("--header_path", type=str, default=None, help="Optional header/wavelength file.")
    parser.add_argument("--sensor", type=str, default="hyperion", help="Sensor name.")
    parser.add_argument("--device", type=str, default="cuda", help="Inference device.")
    parser.add_argument("--row_start", type=int, default=128)
    parser.add_argument("--row_stop", type=int, default=192)
    parser.add_argument("--col_start", type=int, default=2368)
    parser.add_argument("--col_stop", type=int, default=2432)
    parser.add_argument("--output_json", type=str, default=str(DEFAULT_OUTPUT_JSON), help="Summary JSON path.")
    return parser


def _assert_exists(path_str: str | None, label: str) -> None:
    if not path_str:
        raise FileNotFoundError(f"{label} is missing")
    if not Path(path_str).exists():
        raise FileNotFoundError(f"{label} does not exist: {path_str}")


def main() -> None:
    args = build_parser().parse_args()
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    deploy_bundle = load_deploy_config_bundle(args.deploy_config)
    resolved = deploy_bundle["resolved"]

    _assert_exists(deploy_bundle["deploy_config_path"], "deploy_config")
    _assert_exists(resolved.get("model_checkpoint"), "model_checkpoint")
    _assert_exists(resolved.get("temperature_json"), "temperature_json")
    _assert_exists(resolved.get("manifest_path"), "manifest_path")
    _assert_exists(resolved.get("dataset_path"), "dataset_path")
    _assert_exists(args.input_path, "input_path")

    def _run(device: str):
        return run_inference_pipeline(
            input_path=args.input_path,
            header_path=args.header_path,
            sensor=args.sensor,
            device=device,
            deploy_config_path=deploy_bundle["deploy_config_path"],
            model_checkpoint=None,
            spat_checkpoint=None,
            spec_checkpoint=None,
            temperature=None,
            temperature_json=None,
            decision_threshold=None,
            manifest_path=None,
            patch_dataset_path=None,
            split_policy=None,
            model_type=None,
            patch_size=None,
            stride=None,
            row_start=args.row_start,
            row_stop=args.row_stop,
            col_start=args.col_start,
            col_stop=args.col_stop,
            xmin=None,
            ymin=None,
            xmax=None,
            ymax=None,
        )

    requested_device = args.device
    executed_device = requested_device
    fallback_reason = None
    result = _run(requested_device)
    stderr_blob = f"{result.get('stdout', '')}\n{result.get('stderr', '')}"
    if (
        not result["ok"]
        and requested_device == "cuda"
        and "No CUDA GPUs are available" in stderr_blob
    ):
        executed_device = "cpu"
        fallback_reason = "cuda_unavailable"
        result = _run(executed_device)

    missing_outputs = [
        key for key in REQUIRED_OUTPUT_KEYS
        if not Path(result["files"].get(key, "")).exists()
    ]

    summary = {
        "ok": result["ok"] and not missing_outputs,
        "requested_device": requested_device,
        "executed_device": result.get("executed_device", executed_device),
        "api_requested_device": result.get("requested_device"),
        "api_executed_device": result.get("executed_device"),
        "fallback_reason": fallback_reason,
        "deploy_config_path": deploy_bundle["deploy_config_path"],
        "resolved_model_checkpoint": result.get("resolved_model_checkpoint"),
        "resolved_temperature_json": result.get("resolved_temperature_json"),
        "resolved_threshold": result.get("resolved_threshold"),
        "resolved_manifest": result.get("resolved_manifest"),
        "resolved_dataset": result.get("resolved_dataset"),
        "resolved_run_name": result.get("resolved_run_name"),
        "output_dir": result.get("output_dir"),
        "required_outputs": {
            key: result["files"].get(key)
            for key in REQUIRED_OUTPUT_KEYS
        },
        "missing_outputs": missing_outputs,
        "model_provenance": result.get("model_provenance"),
        "api_result_json": result["files"].get("api_result_json"),
        "metadata_json": result["files"].get("metadata_json"),
    }
    output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if not summary["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
