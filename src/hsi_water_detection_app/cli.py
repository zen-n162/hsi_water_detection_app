import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from hsi_water_detection_app.config import (
    DEFAULT_PATCH_SIZE,
    DEFAULT_STRIDE,
    HYPERION_BAD_BANDS_0BASED,
)
from hsi_water_detection_app.data.loader import (
    load_hsi_cube,
    load_hsi_window,
    load_hsi_bounds,
    parse_header_wavelengths,
)
from hsi_water_detection_app.data.preprocessing import normalize_cube, remove_bad_bands
from hsi_water_detection_app.inference.patch_infer import generate_patches, infer_patches
from hsi_water_detection_app.inference.reconstruct import (
    reconstruct_from_patches,
    reconstruct_spatial_attention_from_patches,
)
from hsi_water_detection_app.models.hyper_sigma import load_model
from hsi_water_detection_app.visualization.spatial import (
    save_probability_map,
    save_spatial_attention_map,
)
from hsi_water_detection_app.visualization.spectral import save_spectral_outputs

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DEPLOY_CONFIG_PATH = PROJECT_ROOT / "configs" / "deploy" / "hypersigma_v3_calibrated.json"


def _safe_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        return float(value)
    except Exception:
        return default


def _resolve_temperature(
    temperature: float | None,
    temperature_json: str | None,
) -> tuple[float, str | None]:
    if temperature_json:
        obj = json.loads(Path(temperature_json).read_text(encoding="utf-8"))
        value = _safe_float(obj.get("best_temperature", obj.get("temperature")), default=1.0)
        return float(value if value is not None else 1.0), str(Path(temperature_json).resolve())
    if temperature is not None:
        return float(temperature), None
    return 1.0, None


def _resolve_repo_path(path_value: str | None) -> str | None:
    if not path_value:
        return None
    path = Path(path_value)
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    return str(path)


def _flag_present(flag: str) -> bool:
    return flag in sys.argv[1:]


def _apply_deploy_config_defaults(args) -> tuple[dict[str, Any] | None, str | None]:
    if not args.deploy_config:
        return None, None

    config_path = Path(args.deploy_config)
    if not config_path.is_absolute():
        config_path = (PROJECT_ROOT / config_path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Deploy config not found: {config_path}")

    deploy_config = json.loads(config_path.read_text(encoding="utf-8"))
    args.deploy_config = str(config_path)

    if not args.model_checkpoint:
        args.model_checkpoint = _resolve_repo_path(deploy_config.get("model_checkpoint"))
    if not args.temperature_json:
        args.temperature_json = _resolve_repo_path(deploy_config.get("temperature_json"))
    if args.temperature is None and deploy_config.get("temperature") is not None:
        args.temperature = float(deploy_config["temperature"])
    if args.decision_threshold is None and deploy_config.get("threshold") is not None:
        args.decision_threshold = float(deploy_config["threshold"])
    if not args.manifest_path:
        args.manifest_path = _resolve_repo_path(deploy_config.get("manifest_path"))
    if not args.patch_dataset_path:
        args.patch_dataset_path = _resolve_repo_path(deploy_config.get("dataset_path"))
    if not args.split_policy and deploy_config.get("split_policy"):
        args.split_policy = str(deploy_config["split_policy"])
    if not args.spat_checkpoint and deploy_config.get("spat_checkpoint"):
        args.spat_checkpoint = _resolve_repo_path(deploy_config.get("spat_checkpoint"))
    if not args.spec_checkpoint and deploy_config.get("spec_checkpoint"):
        args.spec_checkpoint = _resolve_repo_path(deploy_config.get("spec_checkpoint"))

    if not _flag_present("--model_type") and deploy_config.get("model_type"):
        args.model_type = str(deploy_config["model_type"])
    if not _flag_present("--patch_size") and deploy_config.get("patch_size") is not None:
        args.patch_size = int(deploy_config["patch_size"])
    if not _flag_present("--stride") and deploy_config.get("stride") is not None:
        args.stride = int(deploy_config["stride"])
    if args.sensor == "auto" and deploy_config.get("sensor"):
        args.sensor = str(deploy_config["sensor"])

    return deploy_config, str(config_path)


def build_parser():
    parser = argparse.ArgumentParser(
        prog="hsi-water-detect",
        description="HSI Water Detection Application CLI",
    )
    parser.add_argument("--input", type=str, required=True, help="Path to input HSI file")
    parser.add_argument("--header", type=str, default=None, help="Optional ENVI header file or wavelength sidecar")
    parser.add_argument("--output_dir", type=str, default=None, help="Optional explicit output directory")
    parser.add_argument(
        "--deploy_config",
        type=str,
        default=None,
        help=f"Optional deploy config JSON. Missing inference options are filled from it. Recommended default: {DEFAULT_DEPLOY_CONFIG_PATH}",
    )

    # backward compatible
    parser.add_argument("--model_checkpoint", type=str, default=None, help="Legacy single checkpoint path")
    parser.add_argument("--temperature", type=float, default=None, help="Optional temperature scaling value")
    parser.add_argument("--temperature_json", type=str, default=None, help="Optional temperature scaling json path")
    parser.add_argument("--decision_threshold", type=float, default=None, help="Optional decision threshold kept in provenance")
    parser.add_argument("--manifest_path", type=str, default=None, help="Optional training/eval manifest provenance path")
    parser.add_argument("--patch_dataset_path", type=str, default=None, help="Optional patch dataset provenance path")
    parser.add_argument("--split_policy", type=str, default=None, help="Optional split policy provenance label")

    # new dual-checkpoint mode
    parser.add_argument("--spat_checkpoint", type=str, default=None, help="Path to spatial encoder checkpoint")
    parser.add_argument("--spec_checkpoint", type=str, default=None, help="Path to spectral encoder checkpoint")

    parser.add_argument("--patch_size", type=int, default=DEFAULT_PATCH_SIZE, help="Patch size")
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE, help="Patch stride")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Inference device")
    parser.add_argument(
        "--sensor",
        type=str,
        default="auto",
        choices=["auto", "hyperion", "hisui", "generic"],
        help="Sensor type",
    )

    parser.add_argument(
        "--model_type",
        type=str,
        default="ss",
        choices=["ss", "sa"],
        help="HyperSIGMA wrapper mode: ss=spatial+spectral, sa=spatial-only",
    )

    # Pixel ROI
    parser.add_argument("--row_start", type=int, default=None, help="ROI row start")
    parser.add_argument("--row_stop", type=int, default=None, help="ROI row stop")
    parser.add_argument("--col_start", type=int, default=None, help="ROI col start")
    parser.add_argument("--col_stop", type=int, default=None, help="ROI col stop")

    # Bounds ROI
    parser.add_argument("--xmin", type=float, default=None, help="ROI xmin in dataset CRS")
    parser.add_argument("--ymin", type=float, default=None, help="ROI ymin in dataset CRS")
    parser.add_argument("--xmax", type=float, default=None, help="ROI xmax in dataset CRS")
    parser.add_argument("--ymax", type=float, default=None, help="ROI ymax in dataset CRS")

    return parser


def infer_sensor(args_sensor: str, input_path: str) -> str:
    if args_sensor != "auto":
        return args_sensor
    input_lower = str(input_path).lower()
    if "eo1" in input_lower or "hyperion" in input_lower:
        return "hyperion"
    if "hisui" in input_lower:
        return "hisui"
    return "generic"


def build_run_name(args) -> str:
    now = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    if all(v is not None for v in [args.xmin, args.ymin, args.xmax, args.ymax]):
        roi_part = (
            f"roi_bounds_"
            f"x{int(args.xmin)}-{int(args.xmax)}_"
            f"y{int(args.ymin)}-{int(args.ymax)}"
        )
    elif all(v is not None for v in [args.row_start, args.row_stop, args.col_start, args.col_stop]):
        roi_part = (
            f"roi_pixel_"
            f"r{args.row_start}-{args.row_stop}_"
            f"c{args.col_start}-{args.col_stop}"
        )
    else:
        roi_part = "fullscene"

    return f"{now}_{roi_part}"


def build_output_dir(args, sensor: str) -> Path:
    if args.output_dir:
        return Path(args.output_dir)

    input_path = Path(args.input)
    scene_name = input_path.stem
    run_name = build_run_name(args)

    return Path("outputs") / sensor / scene_name / run_name


def main():
    parser = build_parser()
    args = parser.parse_args()
    deploy_config, resolved_deploy_config = _apply_deploy_config_defaults(args)
    resolved_temperature, resolved_temperature_json = _resolve_temperature(
        args.temperature,
        args.temperature_json,
    )

    sensor = infer_sensor(args.sensor, args.input)
    output_dir = build_output_dir(args, sensor)
    output_dir.mkdir(parents=True, exist_ok=True)

    use_pixel_roi = all(
        v is not None
        for v in [args.row_start, args.row_stop, args.col_start, args.col_stop]
    )
    use_bounds_roi = all(
        v is not None
        for v in [args.xmin, args.ymin, args.xmax, args.ymax]
    )

    if use_bounds_roi and use_pixel_roi:
        raise ValueError("Specify either pixel ROI or bounds ROI, not both.")

    if use_bounds_roi:
        cube, meta = load_hsi_bounds(
            args.input,
            xmin=args.xmin,
            ymin=args.ymin,
            xmax=args.xmax,
            ymax=args.ymax,
        )
        print(
            f"[INFO] bounds ROI mode enabled: "
            f"x=({args.xmin}, {args.xmax}), "
            f"y=({args.ymin}, {args.ymax})"
        )
    elif use_pixel_roi:
        cube, meta = load_hsi_window(
            args.input,
            row_start=args.row_start,
            row_stop=args.row_stop,
            col_start=args.col_start,
            col_stop=args.col_stop,
        )
        print(
            f"[INFO] pixel ROI mode enabled: "
            f"rows=({args.row_start}, {args.row_stop}), "
            f"cols=({args.col_start}, {args.col_stop})"
        )
    else:
        cube, meta = load_hsi_cube(
            args.input,
            allow_dummy=True,
            sensor=sensor if sensor != "generic" else None,
        )

    wavelengths = None
    if args.header:
        wavelengths = parse_header_wavelengths(args.header)
    if wavelengths is None:
        wavelengths = meta.get("wavelengths")

    if wavelengths is not None:
        print(f"[INFO] wavelength axis enabled with {len(wavelengths)} values")
    else:
        print("[INFO] wavelength axis unavailable; spectral attention will use band index")

    if sensor == "hyperion":
        before_bands = None if cube is None else cube.shape[0]
        if before_bands == 242:
            cube = remove_bad_bands(cube, bad_band_indices=HYPERION_BAD_BANDS_0BASED)

            if wavelengths is not None and before_bands is not None:
                if len(wavelengths) == before_bands:
                    bad_set = set(HYPERION_BAD_BANDS_0BASED)
                    keep_indices = [i for i in range(before_bands) if i not in bad_set]
                    wavelengths = wavelengths[keep_indices]
                    print(f"[INFO] wavelength vector adjusted with bad-band removal: {before_bands} -> {len(wavelengths)}")
                else:
                    print(
                        f"[WARN] wavelength length ({len(wavelengths)}) does not match "
                        f"pre-removal band count ({before_bands}); keeping wavelength vector unchanged"
                    )
        else:
            print(f"[INFO] skip Hyperion bad-band removal because bands={before_bands}")

    cube = normalize_cube(cube)

    model = load_model(
        model_checkpoint=args.model_checkpoint,
        device=args.device,
        in_channels=None if cube is None else cube.shape[0],
        patch_size=args.patch_size,
        model_type=args.model_type,
        spat_checkpoint=args.spat_checkpoint,
        spec_checkpoint=args.spec_checkpoint,
    )

    if cube is None:
        patch_outputs = []
        image_shape = None
        first_spectral_attention = None
    else:
        patches = generate_patches(cube, patch_size=args.patch_size, stride=args.stride)
        patch_outputs = infer_patches(
            model,
            patches,
            device=args.device,
            temperature=resolved_temperature,
        )
        image_shape = (cube.shape[1], cube.shape[2])
        first_spectral_attention = (
            patch_outputs[0]["spectral_attn"] if patch_outputs else None
        )

    prob_map = reconstruct_from_patches(
        patch_outputs,
        image_shape=image_shape,
        patch_size=args.patch_size,
        stride=args.stride,
    )

    spatial_attn_map = reconstruct_spatial_attention_from_patches(
        patch_outputs,
        image_shape=image_shape,
    )

    save_probability_map(
        prob_map,
        output_path=str(output_dir / "probability_map.npy"),
        metadata={
            "input": args.input,
            "header": args.header,
            "model_checkpoint": args.model_checkpoint,
            "spat_checkpoint": args.spat_checkpoint,
            "spec_checkpoint": args.spec_checkpoint,
            "device": args.device,
            "sensor": sensor,
            "model_type": args.model_type,
            "meta": meta,
            "crs": meta.get("crs"),
            "transform": meta.get("transform"),
        },
    )

    save_spatial_attention_map(
        spatial_attn=spatial_attn_map,
        output_dir=str(output_dir),
        metadata={
            "crs": meta.get("crs"),
            "transform": meta.get("transform"),
        },
        cube=cube,
        rgb_bands=(3, 9, 17),
        save_geotiff=True,
    )

    save_spectral_outputs(
        attention_data=first_spectral_attention,
        output_dir=str(output_dir),
        wavelengths=wavelengths,
    )

    run_config = {
        "input": args.input,
        "header": args.header,
        "deploy_config": resolved_deploy_config,
        "deploy_name": None if deploy_config is None else deploy_config.get("deploy_name"),
        "resolved_run_name": None if deploy_config is None else deploy_config.get("run_name"),
        "model_checkpoint": args.model_checkpoint,
        "spat_checkpoint": args.spat_checkpoint,
        "spec_checkpoint": args.spec_checkpoint,
        "temperature": args.temperature,
        "temperature_json": args.temperature_json,
        "resolved_temperature": resolved_temperature,
        "resolved_temperature_json": resolved_temperature_json,
        "decision_threshold": args.decision_threshold,
        "manifest_path": args.manifest_path,
        "patch_dataset_path": args.patch_dataset_path,
        "split_policy": args.split_policy,
        "patch_size": args.patch_size,
        "stride": args.stride,
        "device": args.device,
        "requested_device": args.device,
        "executed_device": getattr(model, "device", args.device),
        "sensor": sensor,
        "model_type": args.model_type,
        "band_count": None if deploy_config is None else deploy_config.get("band_count"),
        "model_load_info": getattr(model, "load_info", {}),
        "row_start": args.row_start,
        "row_stop": args.row_stop,
        "col_start": args.col_start,
        "col_stop": args.col_stop,
        "xmin": args.xmin,
        "ymin": args.ymin,
        "xmax": args.xmax,
        "ymax": args.ymax,
        "output_dir": str(output_dir),
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("Pipeline skeleton is connected.")
    print("args:", args)
    print("sensor:", sensor)
    print("resolved_temperature:", resolved_temperature)
    print("output_dir:", output_dir)
    print("cube:", "None" if cube is None else cube.shape)
    print("prob_map:", "None" if prob_map is None else prob_map.shape)
    print("spatial_attn_map:", "None" if spatial_attn_map is None else spatial_attn_map.shape)


if __name__ == "__main__":
    main()
