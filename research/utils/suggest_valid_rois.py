from __future__ import annotations

import argparse
from pathlib import Path

from hsi_water_detection_app.data.loader import load_hsi_cube


def main() -> None:
    parser = argparse.ArgumentParser(description="Suggest valid 64x64 ROI candidates inside an HSI cube")
    parser.add_argument("--input", required=True, help="Input HSI path")
    parser.add_argument("--patch_h", type=int, default=64)
    parser.add_argument("--patch_w", type=int, default=64)
    parser.add_argument("--row_step", type=int, default=32)
    parser.add_argument("--col_step", type=int, default=256)
    parser.add_argument("--max_rows", type=int, default=20, help="max number of suggested ROI rows")
    args = parser.parse_args()

    cube, meta = load_hsi_cube(args.input, allow_dummy=False)
    if cube is None:
        raise RuntimeError(f"failed to load cube: {args.input}")

    _, height, width = cube.shape
    max_row_start = height - args.patch_h
    max_col_start = width - args.patch_w

    if max_row_start < 0 or max_col_start < 0:
        raise RuntimeError(
            f"patch {args.patch_h}x{args.patch_w} does not fit image size {height}x{width}"
        )

    print(f"[INFO] image size: H={height}, W={width}")
    print(f"[INFO] valid row_start range: 0..{max_row_start}")
    print(f"[INFO] valid col_start range: 0..{max_col_start}")
    print("")
    print("sample_id,row_start,row_stop,col_start,col_stop")

    count = 0
    sid = 100
    for r in range(0, max_row_start + 1, args.row_step):
        for c in range(0, max_col_start + 1, args.col_step):
            print(f"candidate_{sid:04d},{r},{r+args.patch_h},{c},{c+args.patch_w}")
            sid += 1
            count += 1
            if count >= args.max_rows:
                return


if __name__ == "__main__":
    main()
