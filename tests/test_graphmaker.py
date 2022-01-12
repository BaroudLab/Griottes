import numpy as np
import networkx as nx
import pytest
import pandas

from griottes.graphmaker import graph_generation_func


@pytest.fixture
def test_image_2D():

    test_image = np.zeros((10, 10))
    test_image[:5, :] = 1
    test_image[5:, 5:] = 2
    test_image[5:, :5] = 3
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


def test_prepare_user_entry(test_image_2D, test_image_3D):

    user_entry_2D = graph_generation_func.prepare_user_entry(
        user_entry=test_image_2D,
        flat_image=True,
        min_area=1,
        analyze_fluo_channels=False,
        fluo_channel_analysis_method=None,
        radius=None,
        mask_channel=None,
    )
    assert isinstance(user_entry_2D, pandas.DataFrame)
    assert len(user_entry_2D.label.unique()) == 3

    user_entry_3D = graph_generation_func.prepare_user_entry(
        user_entry=test_image_3D,
        flat_image=True,
        min_area=1,
        analyze_fluo_channels=False,
        fluo_channel_analysis_method=None,
        radius=None,
        mask_channel=None,
    )
    assert isinstance(user_entry_3D, pandas.DataFrame)
    assert len(user_entry_3D.label.unique()) == 2


def test_generate_delaunay_graph(test_image_2D):

    G_voronoi = graph_generation_func.generate_delaunay_graph(
        test_image_2D,
        descriptors=[],
        dCells=60,
        flat_image=True,
        min_area=1,
        analyze_fluo_channels=False,
        radius=1,
        mask_channel=None,
    )
    assert isinstance(G_voronoi, nx.Graph)
    assert len(G_voronoi.nodes()) == 3
    assert len(G_voronoi.edges()) == 3


def test_generate_contact_graph(test_image_2D):

    G_contact = graph_generation_func.generate_contact_graph(
        test_image_2D,
        descriptors=[],
        min_area=1,
        analyze_fluo_channels=False,
        radius=1,
        mask_channel=None,
    )
    assert isinstance(G_contact, nx.Graph)
    assert len(G_contact.nodes()) == 3
    assert len(G_contact.edges()) == 3


def test_generate_geometric_graph(test_image_2D):

    G_geometric = graph_generation_func.generate_geometric_graph(
        test_image_2D,
        descriptors=[],
        min_area=1,
        analyze_fluo_channels=False,
        radius=1,
        mask_channel=None,
    )
    assert isinstance(G_geometric, nx.Graph)
    assert len(G_geometric.nodes()) == 3
    assert len(G_geometric.edges()) == 3
