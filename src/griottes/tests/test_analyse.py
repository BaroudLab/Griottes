import numpy as np
import pandas
import pytest

from griottes.analyse import cell_property_extraction


@pytest.fixture
def test_image_2D():
    test_image = np.zeros((10, 10))
    test_image[:5, :] = 1
    test_image[5:, :] = 2

    return test_image.astype(int)


@pytest.fixture
def test_image_3D():
    test_image = np.zeros((4, 10, 10))
    test_image[:, :5, :] = 1
    test_image[:, 5:, :] = 2

    return test_image.astype(int)


def test_test_image(test_image_2D):
    assert isinstance(test_image_2D, np.ndarray)
    assert test_image_2D.ndim == 2


def test_test_image(test_image_3D):
    assert isinstance(test_image_3D, np.ndarray)
    assert test_image_3D.ndim == 3


def test_get_nuclei_properties(test_image_2D):
    properties = cell_property_extraction.get_nuclei_properties(
        image=test_image_2D, mask_channel=None
    )
    assert isinstance(properties, pandas.DataFrame)
    assert len(properties.label.unique()) == 2


def test_get_cell_properties_2D(test_image_2D):
    properties = cell_property_extraction.get_cell_properties(
        image=test_image_2D,
        mask_channel=None,
        analyze_fluo_channels=False,
        fluo_channel_analysis_method="basic",
        cell_geometry_properties=False,
        labeled_voronoi_tesselation=False,
        radius=5,
        min_area=1,
        percentile=95,
        ndim=2,
    )

    assert isinstance(properties, pandas.DataFrame)
    assert len(properties.label.unique()) == 2


def test_get_cell_properties_3D(test_image_3D):
    properties = cell_property_extraction.get_cell_properties(
        image=test_image_3D,
        mask_channel=None,
        analyze_fluo_channels=False,
        fluo_channel_analysis_method="basic",
        cell_geometry_properties=False,
        labeled_voronoi_tesselation=False,
        radius=5,
        min_area=1,
        percentile=95,
        ndim=3,
    )

    assert isinstance(properties, pandas.DataFrame)
    assert len(properties.label.unique()) == 2
    assert "z" in properties.columns
