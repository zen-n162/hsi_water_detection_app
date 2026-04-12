from hsi_water_detection_app.cli import build_parser


def test_build_parser():
    parser = build_parser()
    args = parser.parse_args([
        "--input", "dummy.tif",
        "--output_dir", "results",
        "--model_checkpoint", "dummy.pth",
    ])

    assert args.input == "dummy.tif"
    assert args.output_dir == "results"
    assert args.model_checkpoint == "dummy.pth"
    assert args.patch_size == 64
    assert args.stride == 32
