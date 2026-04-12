"""
HSI Water Detection Application (skeleton)

This module provides a high-level structure for loading hyperspectral image (HSI)
data, preprocessing it, applying a HyperSIGMA model for water detection,
extracting attention maps, and saving the results.

The code is intended as a starting point and does not include
complete implementations for all functions. You will need to fill in the
details based on your specific data formats, model implementation, and
processing requirements.
"""

import os
from typing import Tuple, Iterator, List, Optional

import numpy as np
import torch
# import rasterio  # Uncomment if rasterio is available


def load_hsi_cube(image_path: str) -> np.ndarray:
    """
    Load a hyperspectral image cube from a single file or directory.

    Parameters
    ----------
    image_path : str
        Path to a GeoTIFF/BigTIFF file, ENVI file, or directory containing individual band files.

    Returns
    -------
    np.ndarray
        A 3‑D array with shape (bands, rows, cols).
    """
    # This function should be adapted to your specific data format.
    # For example, to read a multi-band GeoTIFF using rasterio:
    #
    # with rasterio.open(image_path) as ds:
    #     cube = ds.read()  # rasterio returns (bands, rows, cols)
    #
    # If your data is stored in separate files per band, you will need to
    # loop over files and stack the arrays along the band dimension.
    raise NotImplementedError("load_hsi_cube needs to be implemented based on your data format.")


def parse_header_wavelengths(header_path: str) -> Optional[np.ndarray]:
    """
    Parse wavelengths from an ENVI header (.hdr) file.

    Parameters
    ----------
    header_path : str
        Path to the ENVI header file.

    Returns
    -------
    np.ndarray or None
        A 1‑D array of wavelength values, or None if unavailable.
    """
    wavelengths: List[float] = []
    if not os.path.isfile(header_path):
        return None
    with open(header_path, "r", encoding="utf-8", errors="ignore") as hdr:
        for line in hdr:
            if line.strip().lower().startswith("wavelength"):
                # The wavelength line has the form:
                # wavelength = { 425.32, 435.50, ... }
                # Extract the numbers between braces.
                start = line.find("{")
                end = line.find("}")
                if start != -1 and end != -1:
                    values_str = line[start + 1 : end]
                    for val in values_str.split(","):
                        try:
                            wavelengths.append(float(val.strip()))
                        except ValueError:
                            continue
    return np.array(wavelengths) if wavelengths else None


def remove_bad_bands(cube: np.ndarray, bad_indices: List[int]) -> np.ndarray:
    """
    Remove bands identified as bad or uncalibrated.

    Parameters
    ----------
    cube : np.ndarray
        Hyperspectral cube of shape (bands, rows, cols).
    bad_indices : list of int
        Indices (0-based) of bands to drop.

    Returns
    -------
    np.ndarray
        Hyperspectral cube with bad bands removed.
    """
    mask = np.ones(cube.shape[0], dtype=bool)
    mask[bad_indices] = False
    return cube[mask]


def normalize_cube(cube: np.ndarray) -> np.ndarray:
    """
    Perform simple normalization on the spectral cube, e.g. min-max scaling.

    Parameters
    ----------
    cube : np.ndarray
        Hyperspectral cube of shape (bands, rows, cols).

    Returns
    -------
    np.ndarray
        Normalized hyperspectral cube.
    """
    # Flatten spatial dimensions for per-band normalization
    bands, rows, cols = cube.shape
    cube_2d = cube.reshape(bands, -1)
    min_val = cube_2d.min(axis=1, keepdims=True)
    max_val = cube_2d.max(axis=1, keepdims=True)
    # Avoid division by zero
    denom = (max_val - min_val)
    denom[denom == 0] = 1.0
    cube_norm = (cube_2d - min_val) / denom
    return cube_norm.reshape(bands, rows, cols)


def generate_patches(
    cube: np.ndarray, patch_size: Tuple[int, int], stride: Tuple[int, int]
) -> Iterator[Tuple[np.ndarray, Tuple[int, int]]]:
    """
    Generate patches from the spectral cube.

    Parameters
    ----------
    cube : np.ndarray
        Hyperspectral cube of shape (bands, rows, cols).
    patch_size : tuple of int
        Size of each patch in pixels (height, width).
    stride : tuple of int
        Stride between patches in pixels (vertical, horizontal).

    Yields
    ------
    np.ndarray
        A patch of shape (bands, patch_height, patch_width).
    (int, int)
        Top-left coordinate (row, col) of the patch in the original image.
    """
    bands, height, width = cube.shape
    ph, pw = patch_size
    sh, sw = stride
    for row in range(0, height - ph + 1, sh):
        for col in range(0, width - pw + 1, sw):
            yield cube[:, row : row + ph, col : col + pw], (row, col)


def load_model(weights_path: str, device: str = "cuda") -> torch.nn.Module:
    """
    Load a pre‑trained HyperSIGMA model.

    Parameters
    ----------
    weights_path : str
        Path to the saved model weights (.pt or .pth).
    device : str
        Device to map the model to ("cuda" or "cpu").

    Returns
    -------
    torch.nn.Module
        The loaded model ready for inference.
    """
    # Placeholder example. Replace with actual model class import and initialization.
    # from hyper_sigma import HyperSigmaModel
    #
    # model = HyperSigmaModel()
    # state = torch.load(weights_path, map_location=device)
    # model.load_state_dict(state)
    # model.eval()
    # return model.to(device)
    raise NotImplementedError("load_model needs HyperSIGMA implementation.")


def infer_patches(
    model: torch.nn.Module, patches: Iterator[Tuple[np.ndarray, Tuple[int, int]]], device: str = "cuda"
) -> List[Tuple[np.ndarray, Tuple[int, int], dict]]:
    """
    Run inference on patches and return predictions with attention maps.

    Parameters
    ----------
    model : torch.nn.Module
        Loaded HyperSIGMA model.
    patches : iterator
        Iterator of (patch, (row, col)) generated by generate_patches.
    device : str
        Device for inference ("cuda" or "cpu").

    Returns
    -------
    list of (np.ndarray, (int, int), dict)
        For each patch: the predicted water probabilities (rows x cols),
        the top-left coordinate, and a dict with attention information.
    """
    results = []
    for patch, coord in patches:
        # Convert to torch tensor with shape (1, bands, H, W)
        patch_tensor = torch.from_numpy(patch).unsqueeze(0).float().to(device)
        with torch.no_grad():
            # Example of model output:
            # pred, attn_spat, attn_spec = model(patch_tensor, return_attn=True)
            # pred: tensor of shape (1, 1, H, W)
            # attn_spat: spatial attention, attn_spec: spectral attention
            raise NotImplementedError("Model inference needs to be implemented.")
        # Append placeholder for demonstration
        results.append((np.zeros((patch.shape[1], patch.shape[2])), coord, {}))
    return results


def reconstruct_from_patches(
    results: List[Tuple[np.ndarray, Tuple[int, int], dict]], image_shape: Tuple[int, int], stride: Tuple[int, int]
) -> np.ndarray:
    """
    Reconstruct a full-size probability map from patch predictions.

    Parameters
    ----------
    results : list
        Output list from infer_patches.
    image_shape : tuple of int
        (height, width) of the original image.
    stride : tuple of int
        Stride used in generate_patches.

    Returns
    -------
    np.ndarray
        Probability map of shape (height, width).
    """
    height, width = image_shape
    prob_map = np.zeros((height, width), dtype=float)
    count_map = np.zeros((height, width), dtype=int)
    for pred, (row, col), _ in results:
        ph, pw = pred.shape
        prob_map[row : row + ph, col : col + pw] += pred
        count_map[row : row + ph, col : col + pw] += 1
    # Avoid division by zero
    count_map[count_map == 0] = 1
    return prob_map / count_map


def save_outputs(
    output_dir: str,
    prob_map: np.ndarray,
    spatial_attn: Optional[np.ndarray] = None,
    spectral_attn: Optional[dict] = None,
    wavelengths: Optional[np.ndarray] = None,
):
    """
    Save the probability map and optional attention maps to disk.

    Parameters
    ----------
    output_dir : str
        Directory to store results.
    prob_map : np.ndarray
        Probability map of shape (height, width).
    spatial_attn : np.ndarray, optional
        Spatial attention map of shape (height, width), if available.
    spectral_attn : dict, optional
        Dictionary of spectral attention arrays keyed by pixel coordinates.
    wavelengths : np.ndarray, optional
        Wavelengths corresponding to spectral bands.
    """
    os.makedirs(output_dir, exist_ok=True)
    # Save probability map as NumPy file
    np.save(os.path.join(output_dir, "water_probabilities.npy"), prob_map)
    # Placeholder for saving attention maps and spectral plots
    # You can use matplotlib to create and save figures here.
    if spatial_attn is not None:
        np.save(os.path.join(output_dir, "spatial_attention.npy"), spatial_attn)
    if spectral_attn is not None:
        # Example: save as .npz or create plots per pixel
        np.savez(os.path.join(output_dir, "spectral_attention.npz"), **spectral_attn)
