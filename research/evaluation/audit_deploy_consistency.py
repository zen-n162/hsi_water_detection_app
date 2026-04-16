from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from backend.app.services.deploy_config_service import load_deploy_config_bundle

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_AUDIT_JSON = PROJECT_ROOT / "experiments" / "eval_v3" / "deploy_consistency_audit.json"
DEFAULT_ACCEPTANCE_JSON = PROJECT_ROOT / "experiments" / "eval_v3" / "gui_acceptance_result.json"
DEFAULT_SMOKE_JSON = PROJECT_ROOT / "experiments" / "eval_v3" / "gui_smoke_test_result.json"
DEFAULT_PIPELINE_SUMMARY = PROJECT_ROOT / "experiments" / "eval_v3" / "pipeline_summary.json"
DEFAULT_BEST_RUN_SELECTION = PROJECT_ROOT / "experiments" / "eval_v3" / "best_run_selection.json"
DEFAULT_FRONTEND_APP = PROJECT_ROOT / "frontend" / "src" / "App.tsx"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit deploy-config consistency across GUI/backend/metadata artifacts.")
    parser.add_argument("--deploy_config", type=str, default="configs/deploy/hypersigma_v3_calibrated.json")
    parser.add_argument("--smoke_result_json", type=str, default=str(DEFAULT_SMOKE_JSON))
    parser.add_argument("--pipeline_summary_json", type=str, default=str(DEFAULT_PIPELINE_SUMMARY))
    parser.add_argument("--best_run_selection_json", type=str, default=str(DEFAULT_BEST_RUN_SELECTION))
    parser.add_argument("--frontend_app_tsx", type=str, default=str(DEFAULT_FRONTEND_APP))
    parser.add_argument("--output_json", type=str, default=str(DEFAULT_AUDIT_JSON))
    parser.add_argument("--acceptance_output_json", type=str, default=str(DEFAULT_ACCEPTANCE_JSON))
    return parser


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _path_exists(path_str: str | None) -> bool:
    return bool(path_str) and Path(path_str).exists()


def _make_check(name: str, ok: bool, detail: Any) -> dict[str, Any]:
    return {"name": name, "ok": bool(ok), "detail": detail}


def _float_matches(left: Any, right: Any, tol: float = 1e-12) -> bool:
    try:
        return abs(float(left) - float(right)) <= tol
    except Exception:
        return False


def main() -> None:
    args = build_parser().parse_args()
    deploy_bundle = load_deploy_config_bundle(args.deploy_config)
    deploy = deploy_bundle["resolved"]
    smoke = _load_json(args.smoke_result_json)
    api_result = _load_json(smoke["api_result_json"])
    metadata = _load_json(smoke["metadata_json"])
    pipeline_summary = _load_json(args.pipeline_summary_json)
    best_run_selection = _load_json(args.best_run_selection_json)
    frontend_text = Path(args.frontend_app_tsx).read_text(encoding="utf-8")

    threshold_json_path = (
        Path(deploy["model_checkpoint"]).resolve().parent / "best_val_threshold_calibrated.json"
    )
    threshold_json = _load_json(threshold_json_path)

    pipeline_run_name = Path(pipeline_summary["hypersigma"]["run_dir"]).name
    selection_checkpoint_parent = Path(best_run_selection["best_hypersigma_run"]["checkpoint_path"]).resolve().parent.name
    selection_run_name = best_run_selection["best_hypersigma_run"]["run_name"]

    checks = [
        _make_check("model_checkpoint_exists", _path_exists(deploy.get("model_checkpoint")), deploy.get("model_checkpoint")),
        _make_check("temperature_json_exists", _path_exists(deploy.get("temperature_json")), deploy.get("temperature_json")),
        _make_check("manifest_exists", _path_exists(deploy.get("manifest_path")), deploy.get("manifest_path")),
        _make_check("dataset_exists", _path_exists(deploy.get("dataset_path")), deploy.get("dataset_path")),
        _make_check(
            "threshold_matches_best_val_threshold",
            _float_matches(deploy.get("threshold"), threshold_json.get("best_threshold")),
            {
                "deploy_threshold": deploy.get("threshold"),
                "best_threshold_json": str(threshold_json_path),
                "best_threshold": threshold_json.get("best_threshold"),
            },
        ),
        _make_check(
            "run_name_matches_pipeline_summary",
            deploy.get("run_name") == pipeline_run_name,
            {"deploy_run_name": deploy.get("run_name"), "pipeline_run_name": pipeline_run_name},
        ),
        _make_check(
            "run_name_matches_best_run_selection_checkpoint_parent",
            deploy.get("run_name") == selection_checkpoint_parent,
            {"deploy_run_name": deploy.get("run_name"), "selection_checkpoint_parent": selection_checkpoint_parent},
        ),
        _make_check(
            "best_run_selection_label_consistent",
            str(selection_run_name).startswith(str(deploy.get("run_name"))),
            {"deploy_run_name": deploy.get("run_name"), "selection_run_name": selection_run_name},
        ),
        _make_check(
            "api_response_matches_deploy_checkpoint",
            api_result.get("resolved_model_checkpoint") == deploy.get("model_checkpoint"),
            {
                "api_resolved_model_checkpoint": api_result.get("resolved_model_checkpoint"),
                "deploy_model_checkpoint": deploy.get("model_checkpoint"),
            },
        ),
        _make_check(
            "api_response_matches_deploy_temperature_json",
            api_result.get("resolved_temperature_json") == deploy.get("temperature_json"),
            {
                "api_resolved_temperature_json": api_result.get("resolved_temperature_json"),
                "deploy_temperature_json": deploy.get("temperature_json"),
            },
        ),
        _make_check(
            "api_response_matches_deploy_threshold",
            _float_matches(api_result.get("resolved_threshold"), deploy.get("threshold")),
            {
                "api_resolved_threshold": api_result.get("resolved_threshold"),
                "deploy_threshold": deploy.get("threshold"),
            },
        ),
        _make_check(
            "api_response_matches_deploy_manifest",
            api_result.get("resolved_manifest") == deploy.get("manifest_path"),
            {
                "api_resolved_manifest": api_result.get("resolved_manifest"),
                "deploy_manifest": deploy.get("manifest_path"),
            },
        ),
        _make_check(
            "api_response_matches_deploy_dataset",
            api_result.get("resolved_dataset") == deploy.get("dataset_path"),
            {
                "api_resolved_dataset": api_result.get("resolved_dataset"),
                "deploy_dataset": deploy.get("dataset_path"),
            },
        ),
        _make_check(
            "api_response_matches_deploy_run_name",
            api_result.get("resolved_run_name") == deploy.get("run_name"),
            {
                "api_resolved_run_name": api_result.get("resolved_run_name"),
                "deploy_run_name": deploy.get("run_name"),
            },
        ),
        _make_check(
            "metadata_matches_api_core_fields",
            all(
                metadata.get(key) == api_result.get(key)
                for key in [
                    "resolved_model_checkpoint",
                    "resolved_temperature_json",
                    "resolved_threshold",
                    "resolved_manifest",
                    "resolved_dataset",
                    "resolved_run_name",
                    "resolved_split_policy",
                    "requested_device",
                    "executed_device",
                ]
            ),
            {
                "metadata": {key: metadata.get(key) for key in [
                    "resolved_model_checkpoint",
                    "resolved_temperature_json",
                    "resolved_threshold",
                    "resolved_manifest",
                    "resolved_dataset",
                    "resolved_run_name",
                    "resolved_split_policy",
                    "requested_device",
                    "executed_device",
                ]},
                "api_result": {key: api_result.get(key) for key in [
                    "resolved_model_checkpoint",
                    "resolved_temperature_json",
                    "resolved_threshold",
                    "resolved_manifest",
                    "resolved_dataset",
                    "resolved_run_name",
                    "resolved_split_policy",
                    "requested_device",
                    "executed_device",
                ]},
            },
        ),
        _make_check(
            "metadata_provenance_matches_api_provenance",
            metadata.get("model_provenance") == api_result.get("model_provenance"),
            {
                "metadata_model_provenance": metadata.get("model_provenance"),
                "api_model_provenance": api_result.get("model_provenance"),
            },
        ),
        _make_check(
            "smoke_test_requested_cuda",
            smoke.get("requested_device") == "cuda",
            {"requested_device": smoke.get("requested_device")},
        ),
        _make_check(
            "smoke_test_executed_cuda",
            smoke.get("executed_device") == "cuda",
            {
                "executed_device": smoke.get("executed_device"),
                "fallback_reason": smoke.get("fallback_reason"),
            },
        ),
        _make_check(
            "frontend_contains_model_info_labels",
            all(token in frontend_text for token in ["Run name", "Checkpoint", "Threshold", "Calibration", "Deploy config path"]),
            {"frontend_app_tsx": args.frontend_app_tsx},
        ),
        _make_check(
            "required_output_files_exist",
            all(_path_exists(path) for path in smoke.get("required_outputs", {}).values()),
            smoke.get("required_outputs", {}),
        ),
    ]

    overall_ok = all(check["ok"] for check in checks)
    audit = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "deploy_config_path": deploy_bundle["deploy_config_path"],
        "smoke_result_json": str(Path(args.smoke_result_json).resolve()),
        "api_result_json": smoke.get("api_result_json"),
        "metadata_json": smoke.get("metadata_json"),
        "checks": checks,
        "overall_ok": overall_ok,
    }

    acceptance = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "overall_ok": overall_ok,
        "criteria": [
            _make_check("deploy_config_default_load", checks[0]["ok"] and checks[1]["ok"] and checks[2]["ok"] and checks[3]["ok"], {
                "deploy_config_path": deploy_bundle["deploy_config_path"],
            }),
            _make_check("frontend_shows_run_name_checkpoint_threshold_calibration", checks[17]["ok"], {
                "frontend_app_tsx": args.frontend_app_tsx,
            }),
            _make_check("roi_inference_success", bool(smoke.get("ok")), {
                "smoke_result_json": str(Path(args.smoke_result_json).resolve()),
                "output_dir": smoke.get("output_dir"),
            }),
            _make_check("probability_overlay_saved", _path_exists(smoke.get("required_outputs", {}).get("probability_overlay_png")), {
                "probability_overlay_png": smoke.get("required_outputs", {}).get("probability_overlay_png"),
            }),
            _make_check("metadata_json_saved", _path_exists(smoke.get("metadata_json")), {
                "metadata_json": smoke.get("metadata_json"),
            }),
            _make_check(
                "metadata_contains_resolved_checkpoint_threshold_calibration_run_name",
                all(metadata.get(key) is not None for key in [
                    "resolved_model_checkpoint",
                    "resolved_threshold",
                    "resolved_temperature_json",
                    "resolved_run_name",
                ]),
                {
                    "metadata_json": smoke.get("metadata_json"),
                    "resolved_model_checkpoint": metadata.get("resolved_model_checkpoint"),
                    "resolved_threshold": metadata.get("resolved_threshold"),
                    "resolved_temperature_json": metadata.get("resolved_temperature_json"),
                    "resolved_run_name": metadata.get("resolved_run_name"),
                },
            ),
            _make_check(
                "executed_device_recorded",
                metadata.get("executed_device") is not None and api_result.get("executed_device") is not None,
                {
                    "metadata_executed_device": metadata.get("executed_device"),
                    "api_executed_device": api_result.get("executed_device"),
                },
            ),
        ],
        "output_dir": smoke.get("output_dir"),
        "api_result_json": smoke.get("api_result_json"),
        "metadata_json": smoke.get("metadata_json"),
        "executed_device": smoke.get("executed_device"),
    }
    acceptance["overall_ok"] = acceptance["overall_ok"] and all(item["ok"] for item in acceptance["criteria"])

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")

    acceptance_output = Path(args.acceptance_output_json)
    acceptance_output.parent.mkdir(parents=True, exist_ok=True)
    acceptance_output.write_text(json.dumps(acceptance, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(audit, indent=2, ensure_ascii=False))
    print(json.dumps(acceptance, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
