import numpy as np

from hsi_water_detection_app.inference.reconstruct import reconstruct_from_patches


def test_reconstruct_from_patches_returns_expected_shape():
    patch_outputs = [
        {
            "top": 0,
            "left": 0,
            "prob_map": np.zeros((4, 4), dtype=np.float32),
        },
        {
            "top": 4,
            "left": 4,
            "prob_map": np.zeros((4, 4), dtype=np.float32),
        },
    ]
    out = reconstruct_from_patches(
        patch_outputs,
        image_shape=(8, 8),
        patch_size=4,
        stride=4,
    )
    assert out.shape == (8, 8)
