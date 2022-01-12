import numpy as np
import pandas


def single_spheroid_process(spheroid_frame: pandas.DataFrame, descriptors: list = []):

    """

    spheroid_frame has the label of each cell as index.

    """

    # If the original dataframe is only 2D, then transform the 2D
    # data to 3D.
    if "z" not in spheroid_frame.columns:

        spheroid_frame["z"] = 0

    assert set(["z", "x", "y"]).issubset(spheroid_frame.columns)
    assert set(descriptors).issubset(spheroid_frame.columns)

    spheroid = {}

    cells = {}

    for ind in spheroid_frame.index:

        unique_cell = {}

        unique_cell["x"] = spheroid_frame.loc[ind, "x"]
        unique_cell["y"] = spheroid_frame.loc[ind, "y"]
        unique_cell["z"] = spheroid_frame.loc[ind, "z"]
        unique_cell["label"] = spheroid_frame.loc[ind, "label"]

        for descriptor in descriptors:

            unique_cell[descriptor] = spheroid_frame.loc[ind, descriptor]

        cells[ind] = unique_cell

    spheroid["cells"] = cells

    return spheroid


def generate_artificial_spheroid(n: int, ndims: int = 3):

    data = np.random.rand(n, ndims)
    columns = ["x", "y", "z"]

    Sf = pandas.DataFrame(data=data, columns=columns)

    return single_spheroid_process(Sf)
