import numpy as np

from hsi_water_detection_app.data.preprocessing import normalize_cube, remove_bad_bands


def test_normalize_cube_keeps_shape():
    cube = np.random.rand(8, 10, 12).astype("float32")
    out = normalize_cube(cube)
    assert out.shape == cube.shape


def test_remove_bad_bands_reduces_band_count():
    cube = np.random.rand(8, 10, 12).astype("float32")
    out = remove_bad_bands(cube, bad_band_indices=[1, 3])
    assert out.shape[0] == 6
