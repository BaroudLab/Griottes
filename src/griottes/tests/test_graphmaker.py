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


@pytest.fixture
def test_binary_image_2D():
    test_image = np.zeros((10, 10))
    test_image[:5, :5] = 1
    test_image[7:, 7:] = 1

    return test_image.astype(bool)


@pytest.fixture
def test_dataframe_3D():
    data = [[0, 0, 0, 0], [1, 1, 0, 0], [2, 0, 1, 0], [3, 1, 0, 2]]
    test_dataframe = pandas.DataFrame(data=data, columns=["label", "x", "y", "z"])

    return test_dataframe


def test_test_image(test_image_2D):
    assert isinstance(test_image_2D, np.ndarray)
    assert test_image_2D.ndim == 2


def test_test_image(test_image_3D):
    assert isinstance(test_image_3D, np.ndarray)
    assert test_image_3D.ndim == 3


def test_test_image(test_binary_image_2D):
    assert isinstance(test_binary_image_2D, np.ndarray)
    assert test_binary_image_2D.ndim == 2
    assert test_binary_image_2D.dtype == bool


def test_prepare_user_entry(test_image_2D, test_image_3D, test_binary_image_2D):
    user_entry_2D = graph_generation_func.prepare_user_entry(
        user_entry=test_image_2D,
        image_is_2D=True,
        min_area=1,
        analyze_fluo_channels=False,
        fluo_channel_analysis_method="basic",
        radius=None,
        mask_channel=None,
    )
    assert isinstance(user_entry_2D, pandas.DataFrame)
    assert len(user_entry_2D.label.unique()) == 3

    user_entry_3D = graph_generation_func.prepare_user_entry(
        user_entry=test_image_3D,
        image_is_2D=False,
        min_area=1,
        analyze_fluo_channels=False,
        fluo_channel_analysis_method="basic",
        radius=None,
        mask_channel=None,
    )
    assert isinstance(user_entry_3D, pandas.DataFrame)
    assert len(user_entry_3D.label.unique()) == 2

    user_entry_binary_2D = graph_generation_func.prepare_user_entry(
        user_entry=test_binary_image_2D,
        image_is_2D=True,
        min_area=1,
        analyze_fluo_channels=False,
        fluo_channel_analysis_method="basic",
        radius=None,
        mask_channel=None,
    )
    assert isinstance(user_entry_binary_2D, pandas.DataFrame)
    assert len(user_entry_binary_2D.label.unique()) == 2


def test_generate_delaunay_graph(test_image_2D):
    G_voronoi = graph_generation_func.generate_delaunay_graph(
        test_image_2D,
        descriptors=[],
        distance=60,
        image_is_2D=True,
        min_area=1,
        analyze_fluo_channels=False,
        radius=1,
        mask_channel=None,
    )
    assert isinstance(G_voronoi, nx.Graph)
    assert len(G_voronoi.nodes()) == 3
    assert len(G_voronoi.edges()) == 3


def test_generate_delaunay_graph_from_dataframe(test_dataframe_3D):
    G_voronoi = graph_generation_func.generate_delaunay_graph(
        test_dataframe_3D[["z", "y", "x", "label"]],
        descriptors=[],
        distance=5,
        image_is_2D=False,
        min_area=0,
        analyze_fluo_channels=False,
        radius=1,
        mask_channel=None,
    )

    assert isinstance(G_voronoi, nx.Graph)
    assert len(G_voronoi.nodes()) == 4
    assert len(G_voronoi.edges()) == 6

    pos = nx.get_node_attributes(G_voronoi, "pos")
    np.testing.assert_array_equal(
        test_dataframe_3D[["z", "y", "x"]], np.array(list(pos.values()))
    )


def test_generate_geometric_graph_from_dataframe(test_dataframe_3D):
    G_voronoi = graph_generation_func.generate_geometric_graph(
        test_dataframe_3D,
        descriptors=[],
        distance=5,
        image_is_2D=False,
        min_area=0,
        analyze_fluo_channels=False,
        radius=1,
        mask_channel=None,
    )

    assert isinstance(G_voronoi, nx.Graph)
    assert len(G_voronoi.nodes()) == 4
    assert len(G_voronoi.edges()) == 6

    pos_voronoi = nx.get_node_attributes(G_voronoi, "pos")

    assert isinstance(pos_voronoi, dict)
    assert len(pos_voronoi) == 4
    assert isinstance(pos_voronoi[0], tuple)
    assert len(pos_voronoi[0]) == 3


def test_generate_geometric_graph_from_dataframe(test_dataframe_3D):
    G_voronoi = graph_generation_func.generate_geometric_graph(
        test_dataframe_3D,
        descriptors=[],
        distance=5,
        image_is_2D=False,
        min_area=0,
        analyze_fluo_channels=False,
        radius=1,
        mask_channel=None,
    )

    assert isinstance(G_voronoi, nx.Graph)
    assert len(G_voronoi.nodes()) == 4
    assert len(G_voronoi.edges()) == 6

    pos_voronoi = nx.get_node_attributes(G_voronoi, "pos")

    assert isinstance(pos_voronoi, dict)
    assert len(pos_voronoi) == 4
    assert isinstance(pos_voronoi[0], tuple)
    assert len(pos_voronoi[0]) == 3


def test_generate_contact_graph_2D(test_image_2D):
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


def test_generate_contact_graph_3D(test_image_3D):
    G_contact = graph_generation_func.generate_contact_graph(
        test_image_3D,
        descriptors=[],
        min_area=1,
        analyze_fluo_channels=False,
        radius=1,
        mask_channel=None,
        image_is_2D=False,
    )
    assert isinstance(G_contact, nx.Graph)
    assert len(G_contact.nodes()) == 2
    assert len(G_contact.edges()) == 1


def test_generate_geometric_graph(test_image_2D):
    G_geometric = graph_generation_func.generate_geometric_graph(
        test_image_2D,
        descriptors=[],
        min_area=1,
        analyze_fluo_channels=False,
        radius=1,
        mask_channel=None,
        image_is_2D=True,
    )
    assert isinstance(G_geometric, nx.Graph)
    assert len(G_geometric.nodes()) == 3
    assert len(G_geometric.edges()) == 3


def test_generate_geometric_graph_from_binary(test_binary_image_2D):
    G_geometric = graph_generation_func.generate_geometric_graph(
        test_binary_image_2D,
        descriptors=[],
        min_area=1,
        analyze_fluo_channels=False,
        radius=10,
        mask_channel=None,
        image_is_2D=True,
    )
    assert isinstance(G_geometric, nx.Graph)
    assert len(G_geometric.nodes()) == 2
    assert len(G_geometric.edges()) == 1
