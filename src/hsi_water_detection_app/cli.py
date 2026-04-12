import argparse
from pathlib import Path

from hsi_water_detection_app.config import (
    DEFAULT_PATCH_SIZE,
    DEFAULT_STRIDE,
    HYPERION_BAD_BANDS_0BASED,
)
from hsi_water_detection_app.data.loader import load_hsi_cube, parse_header_wavelengths
from hsi_water_detection_app.data.preprocessing import normalize_cube, remove_bad_bands
from hsi_water_detection_app.inference.patch_infer import generate_patches, infer_patches
from hsi_water_detection_app.inference.reconstruct import reconstruct_from_patches
from hsi_water_detection_app.models.hyper_sigma import load_model
from hsi_water_detection_app.visualization.spatial import save_probability_map
from hsi_water_detection_app.visualization.spectral import save_spectral_outputs


def build_parser():
    parser = argparse.ArgumentParser(
        prog="hsi-water-detect",
        description="HSI Water Detection Application CLI",
    )
    parser.add_argument("--input", type=str, required=True, help="Path to input HSI file")
    parser.add_argument("--header", type=str, default=None, help="Optional ENVI header file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save outputs")
    parser.add_argument("--model_checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--patch_size", type=int, default=DEFAULT_PATCH_SIZE, help="Patch size")
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE, help="Patch stride")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Inference device")
    parser.add_argument(
        "--sensor",
        type=str,
        default="auto",
        choices=["auto", "hyperion", "hisui", "generic"],
        help="Sensor type"
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cube, meta = load_hsi_cube(args.input, allow_dummy=True)

    if args.header:
        wavelengths = parse_header_wavelengths(args.header)
    else:
        wavelengths = meta.get("wavelengths")

    sensor = args.sensor
    if sensor == "auto":
        input_lower = str(args.input).lower()
        if "eo1" in input_lower or "hyperion" in input_lower:
            sensor = "hyperion"
        elif "hisui" in input_lower:
            sensor = "hisui"
        else:
            sensor = "generic"

    if sensor == "hyperion":
        before_bands = None if cube is None else cube.shape[0]
        cube = remove_bad_bands(cube, bad_band_indices=HYPERION_BAD_BANDS_0BASED)

        if wavelengths is not None and before_bands is not None:
            bad_set = set(HYPERION_BAD_BANDS_0BASED)
            keep_indices = [i for i in range(before_bands) if i not in bad_set]
            if len(wavelengths) == before_bands:
                wavelengths = wavelengths[keep_indices]

    cube = normalize_cube(cube)

    model = load_model(args.model_checkpoint, device=args.device)

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

    save_probability_map(
        prob_map,
        output_path=str(output_dir / "probability_map.npy"),
        metadata={
            "input": args.input,
            "header": args.header,
            "model_checkpoint": args.model_checkpoint,
            "device": args.device,
            "meta": meta,
            "crs": meta.get("crs"),
            "transform": meta.get("transform"),
        },
    )

    save_spectral_outputs(
        attention_data=first_spectral_attention,
        output_dir=str(output_dir),
        wavelengths=wavelengths,
    )

    print("Pipeline skeleton is connected.")
    print("args:", args)
    print("meta:", meta)
    print("wavelengths:", None if wavelengths is None else getattr(wavelengths, "shape", None))
    print("model:", model)
    print("cube:", "None" if cube is None else cube.shape)
    print("prob_map:", "None" if prob_map is None else prob_map.shape)


if __name__ == "__main__":
    main()
