from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt


def save_spectral_outputs(
    attention_data: Optional[np.ndarray],
    output_dir: str,
    wavelengths: Optional[np.ndarray] = None,
) -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if attention_data is None:
        print("[INFO] save_spectral_outputs skipped because attention_data is None")
        return

    csv_path = out_dir / "spectral_attention.csv"

    if wavelengths is None:
        x_values = np.arange(len(attention_data))
        x_label = "band_index"
    else:
        x_values = wavelengths
        x_label = "wavelength"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([x_label, "attention"])
        for x, y in zip(x_values, attention_data):
            writer.writerow([float(x), float(y)])

    print(f"[INFO] spectral attention saved to: {csv_path}")

    png_path = out_dir / "spectral_attention.png"
    plt.figure(figsize=(10, 4))
    plt.plot(x_values, attention_data)
    plt.xlabel(x_label)
    plt.ylabel("attention")
    plt.title("Spectral Attention")
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()

    print(f"[INFO] spectral attention plot saved to: {png_path}")
