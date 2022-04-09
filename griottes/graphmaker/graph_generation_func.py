import numpy as np
import pandas
import networkx as nx
from scipy.spatial import Delaunay
from scipy.spatial import ConvexHull
from collections import defaultdict

from griottes.graphmaker import make_spheroids
from griottes.analyse import cell_property_extraction

# IMPORTANT CONVENTIONS: Following standard practice,
# all images have shapes Z, Y, X, C where C in the
# fluo channel.


def generate_geometric_graph(
    user_entry,
    descriptors: list = [],
    distance: float = 60,
    image_is_2D=False,
    min_area=0,
    analyze_fluo_channels=False,
    radius=30,
    mask_channel=None,
):

    """
    Creates a geometric graph.

    This function creates a geometric graph from an
    image or a dataframe object.

    Parameters
    ----------
    user_entry : pandas.DataFrame or numpy.ndarray
        contains the information on the cells.
    descriptors : list, optional
        contains the cell information included in the
        network nodes.
    distance : float, optional
        the maximum distance between two nodes.
    image_is_2D : bool, optional
        if True, the image is analyzed as a 2D image.
        The default is False.
    min_area : int, optional
        the minimum area of a cell. The default is 0.
    analyze_fluo_channels : bool, optional
        if True, the fluorescence channels are analyzed.
        The default is False.
    radius : int, optional
        Radius of the sphere within the which the fluorescence
        is analyzed. Irrelevant for the 'basic' method.
        The default is 30.
    mask_channel : int, optional
        The channel containing the cell masks
        The default is None.

    Returns
    -------
    nx.Graph
        The graph representation of the input.
    """

    prop = prepare_user_entry(
        user_entry,
        image_is_2D,
        fluo_channel_analysis_method="basic",
        min_area=min_area,
        radius=radius,
        analyze_fluo_channels=analyze_fluo_channels,
        mask_channel=mask_channel,
    )

    assert isinstance(prop, pandas.DataFrame)
    assert set(["x", "y"]).issubset(prop.columns)
    assert set(descriptors).issubset(prop.columns)
    image_is_2D = (False if "z" in prop.columns else True)

    prop.index = np.arange(len(prop))

    spheroid = make_spheroids.single_spheroid_process(prop, descriptors=descriptors)

    cells = spheroid["cells"]

    if image_is_2D:
        # Generate a dict of positions
        pos = {int(i): (cells[i]["y"], cells[i]["x"]) for i in cells.keys()}

        # Create 2D network
        G = nx.random_geometric_graph(len(cells), distance, pos=pos)

    else:
        # Generate a dict of positions
        pos = {int(i): (cells[i]["z"], cells[i]["y"], cells[i]["x"]) for i in cells.keys()}

        # Create 3D network
        G = nx.random_geometric_graph(len(cells), distance, pos=pos)

    label = {int(i): cells[i]["label"] for i in cells.keys()}
    nx.set_node_attributes(G, pos, "pos")
    nx.set_node_attributes(G, label, "label")

    for ind in list(G.nodes):

        for descriptor in descriptors:

            G.add_node(ind, descriptor=cells[ind][descriptor])

    return G


def generate_contact_graph(
    user_entry,
    mask_channel=None,
    min_area=0,
    analyze_fluo_channels=True,
    image_is_2D = True,
    fluo_channel_analysis_method="basic",
    radius=30,
    descriptors=[],
):

    """
    Creates a contact graph.

    This function creates a contact graph from an
    image. The contact graph is a graph where each node
    represents a region and each edge represents a contact
    between two adjascent regions.

    Parameters
    ----------
    user_entry : numpy.ndarray
        contains the information on the cells.
    descriptors : list, optional
        contains the cell information included in the
        network nodes.
    distance : float, optional
        the maximum distance between two nodes.
    image_is_2D : bool, optional
        if True, the image is analyzed as a 2D image.
        The default is False.
    min_area : int, optional
        the minimum area of a cell. The default is 0.
    analyze_fluo_channels : bool, optional
        if True, the fluorescence channels are analyzed.
        The default is True.
    radius : int, optional
        Radius of the sphere within the which the fluorescence
        is analyzed. Irrelevant for the 'basic' method.
        The default is 30.
    mask_channel : int, optional
        The channel containing the cell masks
        The default is None.

    Returns
    -------
    nx.Graph
        The graph representation of the input.
    """

    # create a data frame containing the relevant info
    if isinstance(user_entry, np.ndarray):

        user_entry = create_region_contact_frame(
            user_entry,
            image_is_2D=image_is_2D,
            mask_channel=mask_channel,
            min_area=min_area,
            analyze_fluo_channels=analyze_fluo_channels,
            fluo_channel_analysis_method=fluo_channel_analysis_method,
            radius=radius,
        )

    assert isinstance(user_entry, pandas.DataFrame)
    assert set(["x", "y"]).issubset(user_entry.columns)

    if not image_is_2D:
        assert set(["z"]).issubset(user_entry.columns)

    # create the connectivity graph
    G = nx.Graph()

    for ind, label_start in zip(user_entry.label.index, user_entry.label.values):

        neighbors = user_entry.neighbors[ind]

        if isinstance(neighbors, dict):

            for label_stop in neighbors.keys():

                G.add_edge(label_start, label_stop, weight=neighbors[label_stop])

    for descriptor in descriptors:
        desc = {
            int(i): (user_entry.loc[(user_entry.label == i)][descriptor].values[0])
            for i in user_entry.label
        }
        nx.set_node_attributes(G, desc, descriptor)

    # for the plotting function, pos = (z,y,x).
    if image_is_2D:
        pos = {
            int(i): (
                user_entry.loc[(user_entry.label == i)]["x"].values[0],
                user_entry.loc[(user_entry.label == i)]["y"].values[0],
            )
            for i in user_entry.label
        }
    else:
        pos = {
            int(i): (
                user_entry.loc[(user_entry.label == i)]["z"].values[0],
                user_entry.loc[(user_entry.label == i)]["y"].values[0],
                user_entry.loc[(user_entry.label == i)]["x"].values[0],
            )
            for i in user_entry.label
        }

    nx.set_node_attributes(G, pos, "pos")

    return G



def generate_delaunay_graph(
    user_entry,
    descriptors: list = [],
    distance: float = 60,
    image_is_2D=False,
    min_area=0,
    analyze_fluo_channels=False,
    radius=30,
    mask_channel=None,
):

    """
    Creates a Delaunay graph.

    This function creates a Delaunay graph from an
    image or a dataframe object.

    Parameters
    ----------
    user_entry : pandas.DataFrame or numpy.ndarray
        contains the information on the cells.
    descriptors : list, optional
        contains the cell information included in the
        network nodes.
    distance : float, optional
        the maximum distance between two nodes.
    image_is_2D : bool, optional
        if True, the image is analyzed as a 2D image.
        The default is False.
    min_area : int, optional
        the minimum area of a cell. The default is 0.
    analyze_fluo_channels : bool, optional
        if True, the fluorescence channels are analyzed.
        The default is False.
    radius : int, optional
        Radius of the sphere within the which the fluorescence
        is analyzed. Irrelevant for the 'basic' method.
        The default is 30.
    mask_channel : int, optional
        The channel containing the cell masks
        The default is None.

    Returns
    -------
    nx.Graph
        The graph representation of the input.
    """

    prop = prepare_user_entry(
        user_entry,
        image_is_2D,
        fluo_channel_analysis_method="basic",
        min_area=min_area,
        radius=radius,
        analyze_fluo_channels=analyze_fluo_channels,
        mask_channel=mask_channel,
    )
    assert isinstance(prop, pandas.DataFrame)

    # The delaunay segmentation will number the cells in the order
    # of the array. As such, we need to establish a correspondance
    # table between the index and the label.

    prop.index = np.arange(len(prop))

    spheroid = make_spheroids.single_spheroid_process(prop, descriptors=descriptors)

    cells = spheroid["cells"]

    # For the specific case of a 2D mask.
    if (len(user_entry.shape) == 2) & isinstance(user_entry, np.ndarray):
        image_is_2D = True

    if image_is_2D:
        cells_pos = prep_points_2D(cells)
        tri = Delaunay(cells_pos)
    else:
        cells_pos = prep_points(cells)
        tri = Delaunay(cells_pos)

    neighbors = find_neighbors(tri)

    G = nx.Graph()
    neighbors = dict(neighbors)

    # all nodes not necessarily in Delauney. Haven't figured out the exclusion
    # criterion yet. --> need to add missing nodes.

    for cell in cells.keys():

        if cell in neighbors:
            for node in neighbors[cell]:

                G.add_edge(cell, node)

        else:
            G.add_node(cell)

    if image_is_2D:
        pos = {
            int(i): (
                cells[i]["x"],
                cells[i]["y"],
            )
            for i in cells
        }
    else:
        pos = {int(i): (cells[i]["x"], cells[i]["y"], cells[i]["z"]) for i in cells.keys()}
    label = {int(i): cells[i]["label"] for i in cells.keys()}

    nx.set_node_attributes(G, pos, "pos")
    nx.set_node_attributes(G, label, "label")

    for descriptor in descriptors:

        desc = {int(i): (cells[i][descriptor]) for i in cells.keys()}
        nx.set_node_attributes(G, desc, descriptor)

    return trim_graph_voronoi(G, distance, image_is_2D)


def prep_points(cells: dict):

    return [
        [cells[cell_label]["z"], cells[cell_label]["y"], cells[cell_label]["x"]]
        for cell_label in cells.keys()
    ]


def prep_points_2D(cells: dict):

    return [
        [cells[cell_label]["y"], cells[cell_label]["x"]] for cell_label in cells.keys()
    ]


def find_neighbors(tess):

    neighbors = defaultdict(set)

    for simplex in tess.simplices:
        for idx in simplex:

            other = set(simplex)
            other.remove(idx)
            neighbors[idx] = neighbors[idx].union(other)

    return neighbors


def prepare_user_entry(
    user_entry,
    image_is_2D,
    min_area,
    analyze_fluo_channels,
    fluo_channel_analysis_method,
    radius,
    mask_channel,
):

    if isinstance(user_entry, np.ndarray):

        image_dim = len(user_entry.shape)
        n_dim = image_dim

        assert image_dim >= 2

        if image_dim == 2:
            n_dim = image_dim
        elif image_dim == 3:
            if image_is_2D:
                n_dim = image_dim - 1
            else:
                print(
                    f"Without any further instructions, the image is being analyzed as a {n_dim}D mono-channel mask. For a more detailed analysis, please use the `get_cell_properties` function."
                )
                n_dim = image_dim
        elif image_dim == 4:
            print(
                f"Without any further instructions, the image is being analyzed as a {n_dim-1}D multi-channel image. For a more detailed analysis, please use the `get_cell_properties` function."
            )
            n_dim = image_dim - 1

        user_entry = cell_property_extraction.get_cell_properties(
            user_entry,
            analyze_fluo_channels=analyze_fluo_channels,
            fluo_channel_analysis_method=fluo_channel_analysis_method,
            radius=radius,
            min_area=min_area,
            ndim=image_dim,
            mask_channel=mask_channel,
        )

        return user_entry

    elif isinstance(user_entry, pandas.DataFrame):

        return user_entry

    else:

        print(
            "The entered object is neither an image (numpy array) nor a pandas DataFrame"
        )

        return

def trim_graph_voronoi(G, distance, image_is_2D):

    """
    Remove slinks above the distance length. Serves to
    remove unrealistic edges from the graph.

    Parameters
    ----------
    G : nx.Graph
        The graph representation of the input image/table.
    distance : float
        The maximum distance between two nodes.
    image_is_2D : bool
        If True, the image is 2D.

    Output
    ------
    nx.Graph
    """

    pos = nx.get_node_attributes(G, "pos")
    edges = G.edges()
    to_remove = []

    if image_is_2D:
        for e in edges:

            i, j = e
            dx2 = (pos[i][0] - pos[j][0]) ** 2
            dy2 = (pos[i][1] - pos[j][1]) ** 2

            if np.sqrt(dx2 + dy2) > distance:

                to_remove.append((i, j))

    else:

        for e in edges:

            i, j = e
            dx2 = (pos[i][0] - pos[j][0]) ** 2
            dy2 = (pos[i][1] - pos[j][1]) ** 2
            dz2 = (pos[i][2] - pos[j][2]) ** 2

            if np.sqrt(dx2 + dy2 + dz2) > distance:

                to_remove.append((i, j))

    for e in to_remove:

        i, j = e
        G.remove_edge(i, j)

    return G


def attribute_layer(G, image_is_2D=False):

    npoints = G.number_of_nodes()
    pos = nx.get_node_attributes(G, "pos")

    if image_is_2D:

        points = np.zeros((npoints, 2))
        layer = 0

        # ignore z component in pos
        for ind in range(npoints):
            points[ind][0] = pos[ind][1]
            points[ind][1] = pos[ind][2]

        pts = points.tolist()
        hull = ConvexHull(points)

    else:

        points = np.zeros((npoints, 3))
        layer = 0

        for ind in range(npoints):
            points[ind][0] = pos[ind][0]
            points[ind][1] = pos[ind][1]
            points[ind][2] = pos[ind][2]

        pts = points.tolist()
        hull = ConvexHull(points)

    L = []

    while len(points) > 3:
        hull = ConvexHull(points)
        points = points.tolist()
        k = len(points)

        for l in range(1, k + 1):
            if k - l in hull.vertices:
                n = pts.index(points.pop(k - l))
                G.add_node(n, layer=layer)
                L.append(n)

        layer = layer + 1
        points = np.asarray(points)

    if len(G) - len(L) > 0:
        for l in range(len(G)):
            if l not in L:
                G.add_node(l, layer=layer)
                L.append(l)
        layer = layer + 1

    G.graph["nb_of_layer"] = layer + 1

    return G


###########


def get_region_contacts_2D(mask_image):

    """
    From the masked image create a dataframe containing the information
    on all the links between region.

    """

    assert isinstance(mask_image, np.ndarray)
    assert mask_image.ndim == 2

    # final output
    edge_frame = pandas.DataFrame(columns=["label", "neighbors"])
    region_list = np.unique(mask_image)
    region_list = region_list[region_list > 0]  # exclude background

    for region in np.unique(mask_image):

        y = mask_image == region  # convert to Boolean

        rolled = np.roll(y, 1, axis=0)  # shift down
        rolled[0, :] = False
        z = np.logical_or(y, rolled)

        rolled = np.roll(y, -1, axis=0)  # shift up
        rolled[-1, :] = False
        z = np.logical_or(z, rolled)

        rolled = np.roll(y, 1, axis=1)  # shift right
        rolled[:, 0] = False
        z = np.logical_or(z, rolled)

        rolled = np.roll(y, -1, axis=1)  # shift left
        rolled[:, -1] = False
        z = np.logical_or(z, rolled)

        neigh, length = np.unique(np.extract(z, mask_image), return_counts=True)

        # remove the current region from the neighbor region list
        ind = np.where(neigh == region)
        neigh = np.delete(neigh, ind)
        length = np.delete(length, ind)

        new_row = {
            "label": region,
            "neighbors": {neigh[i]: length[i] for i in range(len(neigh))},
        }
        edge_frame = edge_frame.append(new_row, ignore_index=True)

    return edge_frame


def get_region_contacts_3D(mask_image):

    """
    From a labeled 3D image create a dataframe containing the information
    on all the links between regions.

    """

    assert isinstance(mask_image, np.ndarray)
    assert mask_image.ndim == 3

    # final output
    edge_frame = pandas.DataFrame(columns=["label", "neighbors"])
    region_list = np.unique(mask_image)
    region_list = region_list[region_list > 0]  # exclude background

    for region in np.unique(mask_image):

        y = mask_image == region  # convert to Boolean

        rolled = np.roll(y, 1, axis=0)  # shift down
        rolled[0, :, :] = False
        z = np.logical_or(y, rolled)

        rolled = np.roll(y, -1, axis=0)  # shift up
        rolled[-1, :, :] = False
        z = np.logical_or(z, rolled)

        rolled = np.roll(y, 1, axis=1)  # shift right
        rolled[:, 0, :] = False
        z = np.logical_or(z, rolled)

        rolled = np.roll(y, -1, axis=1)  # shift left
        rolled[:, -1, :] = False
        z = np.logical_or(z, rolled)

        rolled = np.roll(y, 1, axis=2)  # shift right
        rolled[:, :, 0] = False
        z = np.logical_or(z, rolled)

        rolled = np.roll(y, -1, axis=2)  # shift left
        rolled[:, :, -1] = False
        z = np.logical_or(z, rolled)

        neigh, length = np.unique(np.extract(z, mask_image), return_counts=True)

        # remove the current region from the neighbor region list
        ind = np.where(neigh == region)
        neigh = np.delete(neigh, ind)
        length = np.delete(length, ind)

        new_row = {
            "label": region,
            "neighbors": {neigh[i]: length[i] for i in range(len(neigh))},
        }
        edge_frame = edge_frame.append(new_row, ignore_index=True)

    return edge_frame


def create_region_contact_frame(
    label_img,
    mask_channel,
    image_is_2D=True,
    min_area=0,
    analyze_fluo_channels=False,
    fluo_channel_analysis_method="basic",
    radius=0,
):

    # Need to work on label mask to get neighbors
    # and link weight.
    if mask_channel is not None:
        if label_img[..., mask_channel].ndim == 2:
            assert image_is_2D
            edge_frame = get_region_contacts_2D(label_img[..., mask_channel])
        elif label_img[..., mask_channel].ndim == 3:
            assert  not image_is_2D
            edge_frame = get_region_contacts_3D(label_img[..., mask_channel])
    else:
        if label_img.ndim == 2:
            assert image_is_2D
            edge_frame = get_region_contacts_2D(label_img)
        elif label_img.ndim == 3:
            assert not image_is_2D
            edge_frame = get_region_contacts_3D(label_img)

    # get the region properties
    user_entry = prepare_user_entry(
        label_img,
        image_is_2D=image_is_2D,
        min_area=min_area,
        analyze_fluo_channels=analyze_fluo_channels,
        fluo_channel_analysis_method=fluo_channel_analysis_method,
        radius=radius,
        mask_channel=mask_channel,
    )

    user_entry = user_entry.merge(
        edge_frame, left_on="label", right_on="label", how="outer"
    )

    return user_entry

