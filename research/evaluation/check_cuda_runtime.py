from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_JSON = PROJECT_ROOT / "experiments" / "eval_v3" / "cuda_runtime_check.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Record CUDA runtime visibility for the HyperSIGMA environment.")
    parser.add_argument("--output_json", type=str, default=str(DEFAULT_OUTPUT_JSON))
    return parser


def _run_nvidia_smi() -> dict[str, object]:
    try:
        proc = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            check=False,
        )
        return {
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        }
    except Exception as exc:
        return {
            "returncode": None,
            "stdout": "",
            "stderr": str(exc),
        }


def main() -> None:
    args = build_parser().parse_args()
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    cuda_available = torch.cuda.is_available()
    device_count = torch.cuda.device_count()
    data = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_is_available": cuda_available,
        "device_count": device_count,
        "device_name_0": torch.cuda.get_device_name(0) if cuda_available and device_count > 0 else None,
        "nvidia_smi_L": _run_nvidia_smi(),
    }

    output_json.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(data, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
