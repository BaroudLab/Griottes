import numpy as np
import pandas


def single_spheroid_process(spheroid_frame: pandas.DataFrame, descriptors: list = []):

    """

    spheroid_frame has the label of each cell as index.

    """

    assert set(descriptors).issubset(spheroid_frame.columns)

    spheroid = {}
    cells = {}

    # Only include "z" if it exists
    if "z" in spheroid_frame.columns:
        cols_to_add = ["label", "x", "y", "z"]
        assert set(["label", "z", "x", "y"]).issubset(spheroid_frame.columns)
    else:
        cols_to_add = ["label", "x", "y"]
        assert set(["label", "x", "y"]).issubset(spheroid_frame.columns)

    for ind in spheroid_frame.index:

        unique_cell = {}

        for col in cols_to_add:

            unique_cell[col] = spheroid_frame.loc[ind, col]

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
