from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate comparison report from output directories")
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path("outputs"),
    )
    args = parser.parse_args()

    print("[INFO] generate_comparison_report.py placeholder")
    print(f"[INFO] later: collect pseudocolor / probability overlay / spatial attention / spectral attention from {args.input_dir}")


if __name__ == "__main__":
    main()
