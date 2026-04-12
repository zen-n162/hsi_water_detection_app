import argparse

from hsi_water_detection_app.data.loader import load_hsi_cube, parse_header_wavelengths
from hsi_water_detection_app.models.hyper_sigma import load_model


def build_parser():
    parser = argparse.ArgumentParser(description="HSI Water Detection Application CLI")
    parser.add_argument("--input", type=str, required=True, help="Path to input HSI file")
    parser.add_argument("--header", type=str, default=None, help="Optional ENVI header file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save outputs")
    parser.add_argument("--model_checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--patch_size", type=int, default=64, help="Patch size")
    parser.add_argument("--stride", type=int, default=32, help="Patch stride")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Inference device")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    cube, meta = load_hsi_cube(args.input)

    if args.header:
        wavelengths = parse_header_wavelengths(args.header)
    else:
        wavelengths = None

    model = load_model(args.model_checkpoint, device=args.device)

    print("Pipeline skeleton is connected.")
    print(args)
    print("metadata:", meta)
    print("wavelengths:", wavelengths)


if __name__ == "__main__":
    main()
