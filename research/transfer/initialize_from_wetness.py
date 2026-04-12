from __future__ import annotations

import argparse
from pathlib import Path
import shutil


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize lunar transfer checkpoint from wetness checkpoint")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    args = parser.parse_args()

    args.target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.source, args.target)
    print(f"[INFO] copied {args.source} -> {args.target}")


if __name__ == "__main__":
    main()
