from scipy.spatial import Delaunay


def in_hull(p, hull):
    """
    Test if points in `p` are in `hull`

    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the
    coordinates of `M` points in `K`dimensions for which Delaunay triangulation
    will be computed
    """

    if not isinstance(hull, Delaunay):
        hull = Delaunay(hull)

    return hull.find_simplex(p) >= 0


def get_cell_properties(cell, hull, image, mask_channel):
    # crop around cell (to avoid processing too
    # many pixels)

    xmin = min(hull[:, 0])
    ymin = min(hull[:, 1])
    zmin = min(hull[:, 2])

    xmax = max(hull[:, 0])
    ymax = max(hull[:, 1])
    zmax = max(hull[:, 2])

    new_image = image[xmin:xmax, ymin:ymax, zmin:zmax]

    # get which points are in the hull and not

    in_hull_label = in_hull(new_image, hull)

    # intersect this mask with the sphere

    # we do want to keep this mask for further studies
    # store the mask locally with option `store_voronoi_mask`.

    return
