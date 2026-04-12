from hsi_water_detection_app.data.loader import load_hsi_cube


def test_load_hsi_cube_returns_dummy_when_missing():
    cube, meta = load_hsi_cube("this_file_should_not_exist_12345.tif")
    assert cube is not None
    assert cube.shape == (16, 64, 64)
    assert meta["status"] == "dummy_generated"
