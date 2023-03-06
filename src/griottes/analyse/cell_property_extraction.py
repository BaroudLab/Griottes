import numpy as np
import skimage.measure
import skimage
import pandas
from scipy.spatial import Delaunay
from scipy.spatial import Voronoi
from sklearn.decomposition import PCA
from tqdm import tqdm

# IMPORTANT CONVENTIONS: Following standard practice,
# all images hvae shapes C, Z, Y, X where C is the
# fluo channel.


def get_nuclei_properties(image, mask_channel):
    """
    Get properties of nuclei in image.

    Parameters
    ----------
    image : numpy.ndarray
        Image with nuclei masks.
    mask_channel : int
        Channel of the mask.

    Returns
    -------
    pandas.DataFrame
    """

    if mask_channel is None:
        properties = pandas.DataFrame(
            skimage.measure.regionprops_table(
                image, properties=["centroid", "area", "label"]
            )
        )

    else:
        properties = pandas.DataFrame(
            skimage.measure.regionprops_table(
                image[mask_channel], properties=["centroid", "area", "label"]
            )
        )

    return properties


def get_shape_properties(properties, image, mask_channel, min_area, ndim):
    for ind in tqdm(properties.index, leave=False):
        if (properties.loc[ind, "area"] > min_area) & (ndim == 3):
            label = properties.loc[ind, "label"]
            loc_mask = (image[mask_channel] == label) * 1
            nonzero = np.nonzero(loc_mask)

            pca = PCA(n_components=3)
            Y = np.c_[nonzero[0], nonzero[1], nonzero[2]]

            pca.fit(Y)
            vec = pca.components_[0]
            var = pca.explained_variance_

            properties.loc[ind, "vec_0"] = vec[0]
            properties.loc[ind, "vec_1"] = vec[1]
            properties.loc[ind, "vec_2"] = vec[2]
            properties.loc[ind, "theta"] = np.arctan2(vec[1], vec[2])
            properties.loc[ind, "psi"] = np.arctan2(
                vec[0], np.sqrt(vec[1] ** 2 + vec[2] ** 2)
            )
            properties.loc[ind, "eccentricity"] = np.abs(var[0]) / np.sqrt(
                var[1] * var[2]
            )

        if (properties.loc[ind, "area"] > min_area) & (ndim == 2):
            loc_mask = (image[mask_channel] == ind) * 1
            nonzero = np.nonzero(loc_mask)

            pca = PCA(n_components=2)
            Y = np.c_[nonzero[0], nonzero[1]]
            pca.fit(Y)
            vec = pca.components_[0]
            var = pca.explained_variance_

            properties.loc[ind, "vec_0"] = vec[0]
            properties.loc[ind, "vec_1"] = vec[1]
            properties.loc[ind, "theta"] = np.arctan2(vec[0], vec[1])
            properties.loc[ind, "eccentricity"] = np.abs(var[0]) / np.sqrt(var[1])

    return properties


def get_fluo_properties(image, fluo_channel, mask_channel=0):
    properties_fluo = pandas.DataFrame(
        skimage.measure.regionprops_table(
            image[mask_channel],
            intensity_image=image[fluo_channel],
            properties=["mean_intensity", "label"],
        )
    )

    return properties_fluo


def basic_fluo_prop_analysis(properties, image, mask_channel):
    for i in range(0, image.shape[0], 1):
        if i != mask_channel:
            properties_fluo = get_fluo_properties(
                image=image, fluo_channel=i, mask_channel=mask_channel
            )

            properties_fluo = properties_fluo.rename(
                columns={"mean_intensity": "mean_intensity_" + str(i)}
            )

            properties = properties.merge(properties_fluo, how="outer", on="label")

    return properties


## SPHERE ##


def sphere_mean_intensity(intensity_image, position, radius, percentile):
    n_Z, n_Y, n_X = np.shape(intensity_image)
    Z, Y, X = np.ogrid[:n_Z, :n_Y, :n_X]

    z_nuc, y_nuc, x_nuc = position

    mask = (
        np.sqrt((Z - z_nuc) ** 2 + (X - x_nuc) ** 2 + (Y - y_nuc) ** 2) < radius
    ).astype(int)

    points_in_sphere = np.argwhere(mask)

    return np.mean(
        intensity_image[tuple(points_in_sphere[points_in_sphere].T)]
    ), np.percentile(
        intensity_image[tuple(points_in_sphere[points_in_sphere].T)], percentile
    )


def get_fluo_properties_sphere(
    properties, image, fluo_channel, radius, mask_channel, percentile
):
    for ind in properties.index:
        position = (
            int(properties.loc[ind, "z"]),
            int(properties.loc[ind, "y"]),
            int(properties.loc[ind, "x"]),
        )

        mean, percentile = sphere_mean_intensity(
            intensity_image=image[fluo_channel],
            position=position,
            radius=radius,
            percentile=percentile,
        )

        properties.loc[ind, "mean_intensity"] = mean
        properties.loc[ind, "percentile_intensity"] = percentile

    return properties[["mean_intensity", "percentile_intensity"]]


def sphere_fluo_property_analysis(properties, image, mask_channel, radius, percentile):
    for i in range(0, image.shape[-1], 1):
        if i != mask_channel:
            properties_fluo = get_fluo_properties_sphere(
                properties=properties,
                image=image,
                fluo_channel=i,
                mask_channel=mask_channel,
                radius=radius,
                percentile=percentile,
            )

            properties_fluo = properties_fluo.rename(
                columns={"mean_intensity": "mean_intensity_" + str(i)}
            )

            properties_fluo = properties_fluo.rename(
                columns={"percentile_intensity": "percentile_intensity_" + str(i)}
            )

            properties = properties.merge(properties_fluo, how="outer", on="label")

    del properties["mean_intensity"]
    del properties["percentile_intensity"]

    return properties


### VORONOI ###


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


def make_spherical_mask(image, point_coordinates, radius):
    n_Z, n_Y, n_X = np.shape(image)
    Z, Y, X = np.ogrid[:n_Z, :n_Y, :n_X]

    z_nuc, x_nuc, y_nuc = point_coordinates

    mask = np.sqrt((Z - z_nuc) ** 2 + (X - x_nuc) ** 2 + (Y - y_nuc) ** 2) < radius

    return mask.astype(int)


def make_voronoi_mask(properties, image, mask_channel, radius):
    intensity_image = image[mask_channel]

    label_matrix = np.zeros_like(intensity_image)
    vor = Voronoi(properties[["z", "y", "x"]])

    print("Calculating voronoi")

    for cell_label, point_number in tqdm(
        zip(properties.index, np.arange(len(properties)))
    ):
        region_label = properties.index[point_number]
        region_number = vor.point_region[point_number]
        voronoi_vertices = vor.regions[region_number]

        point_coordinates = vor.points[point_number]
        vertice_array = [
            vor.vertices[vertice_number] for vertice_number in voronoi_vertices
        ]

        spherical_mask = make_spherical_mask(
            image=intensity_image, point_coordinates=point_coordinates, radius=radius
        )

        points_in_sphere = np.argwhere(spherical_mask)

        sphere_points_in_vor = in_hull(points_in_sphere, vertice_array)

        # creating the label matrix
        label_matrix[tuple(points_in_sphere[sphere_points_in_vor].T)] = cell_label

    return label_matrix


def get_fluo_properties_voronoi(
    properties, image, fluo_channel, label_matrix, percentile
):
    intensity_image = image[fluo_channel]

    for ind, cell_label in tqdm(zip(properties.index, properties.label)):
        # mask for quantification
        mask = np.zeros_like(label_matrix)
        mask[label_matrix == cell_label] = 1

        points_in_intersection = np.argwhere(mask)

        try:
            properties.loc[ind, "mean_intensity"] = np.mean(
                intensity_image[tuple(points_in_intersection[points_in_intersection].T)]
            )
            properties.loc[ind, "percentile_intensity"] = np.percentile(
                intensity_image[
                    tuple(points_in_intersection[points_in_intersection].T)
                ],
                percentile,
            )

        except:
            properties.loc[ind, "mean_intensity"] = np.nan
            properties.loc[ind, "percentile_intensity"] = np.nan

    return properties[["mean_intensity", "percentile_intensity", "label"]]


def voronoi_fluo_property_analysis(
    properties, image, mask_channel, radius, labeled_voronoi_tesselation, percentile
):
    """

    Calculate the voronoi mask, then use the mask to
    estimate the intensities inside the mask.

    """

    label_matrix = make_voronoi_mask(properties, image, mask_channel, radius)

    for i in range(image.shape[0]):
        if i != mask_channel:
            properties_fluo = get_fluo_properties_voronoi(
                properties=properties,
                image=image,
                fluo_channel=i,
                label_matrix=label_matrix,
                percentile=percentile,
            )

            properties_fluo = properties_fluo.rename(
                columns={"mean_intensity": "mean_intensity_channel_" + str(i)}
            )

            properties_fluo = properties_fluo.rename(
                columns={
                    "percentile_intensity": "percentile_intensity_channel_" + str(i)
                }
            )

            properties = properties.merge(properties_fluo, how="outer", on="label")

    del properties["mean_intensity"]
    del properties["percentile_intensity"]

    if labeled_voronoi_tesselation:
        return properties, label_matrix

    return properties


### ALL ###


def get_cell_properties(
    image,
    mask_channel=0,
    analyze_fluo_channels=False,
    fluo_channel_analysis_method="basic",
    cell_geometry_properties=False,
    labeled_voronoi_tesselation=False,
    radius=5,
    min_area=50,
    percentile=95,
    ndim=3,
):
    """
    Calculate the cell properties for a given image.

    Parameters
    ----------
    image : numpy array
        The image to be analyzed.
    mask_channel : int
        The channel to be used as a mask.
    analyze_fluo_channels : bool
        If True, the fluorescence channels will be analyzed.
    fluo_channel_analysis_method : str
        The method to be used to analyze the fluorescence channels. Either `basic`,
        `local_voronoi`, or `local_sphere`.
    cell_geometry_properties : bool
        If True, the cell geometry properties will be calculated.
    labeled_voronoi_tesselation : bool
        If True, the voronoi tesselation will be generated and returned as an array.
    radius : int
        Maximum radius within which the cell properties are measured.
    min_area : int
        Minimum area of a cell to be considered.
    percentile : int
        Percentile of the intensity distribution to be used for the percentile intensity calculation.

    Returns
    -------
    pandas.DataFrame
    """

    if image.ndim - ndim < 0:
        print(
            f"the input image has less dimensions than it should. Please check that the input is correct."
        )
        return False
    elif image.ndim - ndim > 1:
        print(
            f"the input image has {image.ndim} dimensions instead of {ndim+1}. Please check that the input is correct."
        )
        return False
    elif image.ndim - ndim == 0:
        print(
            f"The input image has {image.ndim} dimensions, it will be analyzed as a labeled image."
        )
        analyze_fluo_channels = False
        mask_channel = None

    if ndim == 2:
        properties = get_nuclei_properties(image=image, mask_channel=mask_channel)

        properties = properties.rename(columns={"centroid-0": "y", "centroid-1": "x"})
    else:
        properties = get_nuclei_properties(image=image, mask_channel=mask_channel)

        properties = properties.rename(
            columns={"centroid-0": "z", "centroid-1": "y", "centroid-2": "x"}
        )

    if cell_geometry_properties:
        print("Calculating geometrical properties")

        properties = get_shape_properties(
            properties=properties,
            image=image,
            mask_channel=mask_channel,
            min_area=min_area,
            ndim=ndim,
        )

        print("Done geometrical properties")

    if analyze_fluo_channels:
        if fluo_channel_analysis_method == "basic":
            properties = basic_fluo_prop_analysis(properties, image, mask_channel)

            properties = properties.dropna()
            properties.index = np.arange(len(properties))

            return properties

        if fluo_channel_analysis_method == "local_sphere":
            properties = sphere_fluo_property_analysis(
                properties, image, mask_channel, radius, percentile
            )

            properties = properties.dropna()
            properties.index = np.arange(len(properties))

            return properties

        if fluo_channel_analysis_method == "local_voronoi":
            # Need to create voronoi tesselation, then store the
            # vertixes and use them to mark the convex hull
            # corresponding to each cell nuclei. Once the region
            # obtained use this area as a label for regionprops.

            # ATTENTION: verify that the area used to calculate
            # the properties corresponds to the intersection of
            # the voronoi and a sphere of radius R.

            if labeled_voronoi_tesselation:
                properties, label_matrix = voronoi_fluo_property_analysis(
                    properties,
                    image,
                    mask_channel,
                    radius,
                    labeled_voronoi_tesselation,
                    percentile,
                )

                properties = properties.dropna()
                # properties.index = np.arange(len(properties))

                return properties, label_matrix

            else:
                properties = voronoi_fluo_property_analysis(
                    properties,
                    image,
                    mask_channel,
                    radius,
                    labeled_voronoi_tesselation,
                    percentile,
                )

                properties = properties.dropna()
                properties.index = np.arange(len(properties))
                properties = properties[properties.area > min_area]

                return properties

    else:
        properties = properties.dropna()
        properties.index = np.arange(len(properties))
        properties = properties[properties.area > min_area]

        return properties
