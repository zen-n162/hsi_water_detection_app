import argparse
import json
from datetime import datetime
from pathlib import Path

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


def build_parser():
    parser = argparse.ArgumentParser(
        prog="hsi-water-detect",
        description="HSI Water Detection Application CLI",
    )
    parser.add_argument("--input", type=str, required=True, help="Path to input HSI file")
    parser.add_argument("--header", type=str, default=None, help="Optional ENVI header file or wavelength sidecar")
    parser.add_argument("--output_dir", type=str, default=None, help="Optional explicit output directory")
    parser.add_argument("--model_checkpoint", type=str, required=True, help="Path to model checkpoint")
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

    cube = normalize_cube(cube)

    model = load_model(
        args.model_checkpoint,
        device=args.device,
        in_channels=None if cube is None else cube.shape[0],
        patch_size=args.patch_size,
    )

    if cube is None:
        patch_outputs = []
        image_shape = None
        first_spectral_attention = None
    else:
        patches = generate_patches(cube, patch_size=args.patch_size, stride=args.stride)
        patch_outputs = infer_patches(model, patches, device=args.device)
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
            "device": args.device,
            "sensor": sensor,
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
        "model_checkpoint": args.model_checkpoint,
        "patch_size": args.patch_size,
        "stride": args.stride,
        "device": args.device,
        "sensor": sensor,
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
    print("output_dir:", output_dir)
    print("cube:", "None" if cube is None else cube.shape)
    print("prob_map:", "None" if prob_map is None else prob_map.shape)
    print("spatial_attn_map:", "None" if spatial_attn_map is None else spatial_attn_map.shape)


if __name__ == "__main__":
    main()
