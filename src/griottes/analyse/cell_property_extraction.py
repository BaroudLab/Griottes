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
	"""
	Calculate shape properties of the nuclei.

	Parameters
	----------
	properties : pandas.DataFrame
		Dataframe containing centroid, area, and label of the nuclei.
	image : numpy.ndarray
		Image with nuclei masks.
	mask_channel : int
		Channel of the mask.
	min_area : int
		Minimum area of the nuclei to include in analysis.
	ndim : int
		Number of dimensions in the image (2 or 3).

	Returns
	-------
	pandas.DataFrame
		Dataframe with additional columns for shape properties.

	"""
	# Loop through each nucleus
	for ind in tqdm(properties.index, leave=False):
		# Check if nucleus meets minimum area requirement and has 3 dimensions
		if (properties.loc[ind, "area"] > min_area) & (ndim == 3):
			# Get mask for current nucleus
			label = properties.loc[ind, "label"]
			loc_mask = (image[mask_channel] == label) * 1
			nonzero = np.nonzero(loc_mask)

			# Fit PCA to get orientation and eccentricity
			pca = PCA(n_components=3)
			Y = np.c_[nonzero[0], nonzero[1], nonzero[2]]
			pca.fit(Y)
			vec = pca.components_[0]
			var = pca.explained_variance_

			# Store shape properties in dataframe
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

		# Check if nucleus meets minimum area requirement and has 2 dimensions
		if (properties.loc[ind, "area"] > min_area) & (ndim == 2):
			# Get mask for current nucleus
			loc_mask = (image[mask_channel] == ind) * 1
			nonzero = np.nonzero(loc_mask)

			# Fit PCA to get orientation and eccentricity
			pca = PCA(n_components=2)
			Y = np.c_[nonzero[0], nonzero[1]]
			pca.fit(Y)
			vec = pca.components_[0]
			var = pca.explained_variance_

			# Store shape properties in dataframe
			properties.loc[ind, "vec_0"] = vec[0]
			properties.loc[ind, "vec_1"] = vec[1]
			properties.loc[ind, "theta"] = np.arctan2(vec[0], vec[1])
			properties.loc[ind, "eccentricity"] = np.abs(var[0]) / np.sqrt(var[1])

	return properties

def get_fluo_properties(image, fluo_channel, mask_channel=0):
	"""
	Get mean fluorescence intensity of each labeled region in a mask.

	Parameters
	----------
	image : numpy.ndarray
		Multichannel image containing the fluorescence channel.
	fluo_channel : int
		Channel containing the fluorescence signal.
	mask_channel : int, optional
		Channel containing the mask. Default is 0.

	Returns
	-------
	pandas.DataFrame
		Dataframe containing the mean fluorescence intensity and label of each region.

	"""

	# Compute region properties for mean intensity and label
	properties_fluo = pandas.DataFrame(
		skimage.measure.regionprops_table(
			image[mask_channel],
			intensity_image=image[fluo_channel],
			properties=["mean_intensity", "label"],
		)
	)

	return properties_fluo

def basic_fluo_prop_analysis(properties, image, mask_channel):
	"""
	Calculate the mean intensity of each fluorescence channel for each nucleus.

	Parameters
	----------
	properties : pandas.DataFrame
		DataFrame containing properties of each nucleus.
	image : numpy.ndarray
		Image with fluorescence channels.
	mask_channel : int
		Channel of the mask.

	Returns
	-------
	pandas.DataFrame
		Updated properties DataFrame with mean fluorescence intensity for each channel.
	"""

	# Loop through all fluorescence channels except for the mask channel
	for i in range(image.shape[0]):
		if i != mask_channel:
			# Get fluorescence properties for current channel
			properties_fluo = get_fluo_properties(image, i, mask_channel)

			# Rename mean_intensity column with channel number
			properties_fluo = properties_fluo.rename(
				columns={"mean_intensity": "mean_intensity_" + str(i)}
			)

			# Merge fluorescence properties with existing properties DataFrame
			properties = properties.merge(properties_fluo, how="outer", on="label")

	return properties

## SPHERE ##

def sphere_mean_intensity(intensity_image, position, radius, percentile=50):
	"""
	Calculate the mean intensity and percentile of a sphere within an intensity image.

	Parameters
	----------
	intensity_image : numpy.ndarray
		The 3D intensity image.
	position : tuple
		The center of the sphere in (z, y, x) format.
	radius : float
		The radius of the sphere.
	percentile : float, optional
		The percentile to calculate, default is 50.

	Returns
	-------
	tuple of float
		The mean intensity and percentile.

	"""
	# Check if the position is inside the image
	if any(pos < radius or pos >= size - radius for pos, size in zip(position, intensity_image.shape)):
		raise ValueError("Sphere is partially or completely outside the image")

	# Create a mask for the sphere
	z_nuc, y_nuc, x_nuc = position
	z, y, x = np.ogrid[:intensity_image.shape[0], :intensity_image.shape[1], :intensity_image.shape[2]]
	mask = np.sqrt((z - z_nuc) ** 2 + (y - y_nuc) ** 2 + (x - x_nuc) ** 2) < radius

	# Get the mean intensity and percentile
	points_in_sphere = np.argwhere(mask)
	intensities = intensity_image[tuple(points_in_sphere.T)]
	return np.mean(intensities), np.percentile(intensities, percentile)

def calculate_sphere_fluo_properties(
	properties: pandas.DataFrame,
	image: np.ndarray,
	fluo_channel: int,
	radius: int,
	mask_channel: int,
	percentile: int
) -> pandas.DataFrame:
	"""
	Calculate mean and percentile fluorescence intensity within a sphere around each nucleus.

	Parameters
	----------
	properties : pandas.DataFrame
		Properties of the nuclei.
	image : numpy.ndarray
		Image with fluorescence channels.
	fluo_channel : int
		Channel containing the fluorescence signal.
	radius : int
		Radius of the sphere around each nucleus.
	mask_channel : int
		Channel of the nuclei mask.
	percentile : int
		Percentile of the fluorescence intensity to calculate.

	Returns
	-------
	pandas.DataFrame
		Mean and percentile fluorescence intensity within the sphere around each nucleus.
	"""
	for ind in properties.index:
		# Get the centroid of the nucleus
		position = (
			int(properties.loc[ind, "z"]),
			int(properties.loc[ind, "y"]),
			int(properties.loc[ind, "x"]),
		)

		# Calculate the mean and percentile fluorescence intensity within the sphere
		mean_intensity, percentile_intensity = sphere_mean_intensity(
			intensity_image=image[fluo_channel],
			position=position,
			radius=radius,
			percentile=percentile,
		)

		# Add the mean and percentile intensity to the properties DataFrame
		properties.loc[ind, "mean_intensity"] = mean_intensity
		properties.loc[ind, "percentile_intensity"] = percentile_intensity

	# Return only the mean and percentile intensity columns
	return properties[["mean_intensity", "percentile_intensity"]]

def sphere_fluo_property_analysis(properties, image, mask_channel, radius, percentile):
	"""
	Calculate sphere-based fluorescence properties for each label in the mask.

	Parameters
	----------
	properties : pandas.DataFrame
		DataFrame with properties of each label in the mask.
	image : numpy.ndarray
		3D image stack.
	mask_channel : int
		Index of the channel containing the mask.
	radius : int
		Radius of the sphere used to calculate properties.
	percentile : int
		Percentile used to calculate the fluorescence intensity.

	Returns
	-------
	pandas.DataFrame
		Updated DataFrame with fluorescence properties for each label.

	"""
	for i in range(image.shape[-1]):
		if i != mask_channel:
			# Get fluorescence properties for each label
			properties_fluo = get_fluo_properties_sphere(
				properties=properties,
				image=image,
				fluo_channel=i,
				mask_channel=mask_channel,
				radius=radius,
				percentile=percentile,
			)

			# Rename columns with channel information
			properties_fluo = properties_fluo.rename(
				columns={"mean_intensity": "mean_intensity_" + str(i)}
			)
			properties_fluo = properties_fluo.rename(
				columns={"percentile_intensity": "percentile_intensity_" + str(i)}
			)

			# Merge fluorescence properties into main DataFrame
			properties = properties.merge(properties_fluo, how="outer", on="label")

	# Remove original mean_intensity and percentile_intensity columns
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
	"""
	Create a spherical mask centered at `point_coordinates` with `radius`.

	Parameters
	----------
	image : numpy.ndarray
		Input image.
	point_coordinates : tuple of int
		Coordinates of the center of the sphere in the format (z, x, y).
	radius : int
		Radius of the sphere.

	Returns
	-------
	numpy.ndarray
		A binary mask of the same shape as `image`, where voxels inside the
		sphere are set to 1 and voxels outside the sphere are set to 0.
	"""
	# Get the dimensions of the input image
	n_Z, n_Y, n_X = np.shape(image)
	Z, Y, X = np.ogrid[:n_Z, :n_Y, :n_X]

	# Unpack the coordinates of the center of the sphere
	z_nuc, x_nuc, y_nuc = point_coordinates

	# Create a binary mask where voxels inside the sphere are set to 1
	mask = np.sqrt((Z - z_nuc) ** 2 + (X - x_nuc) ** 2 + (Y - y_nuc) ** 2) < radius

	return mask.astype(int)

def make_voronoi_mask(properties, image, mask_channel, radius):
	"""
	Create a Voronoi diagram of the nuclei centroids and use it to generate a binary mask.

	Parameters
	----------
	properties : pandas.DataFrame
		DataFrame with properties of the nuclei.
	image : numpy.ndarray
		Image with nuclei masks.
	mask_channel : int
		Channel of the mask.
	radius : float
		Radius of the spherical mask used to generate the Voronoi diagram.

	Returns
	-------
	numpy.ndarray
		Binary mask with the same shape as the input image, where each pixel is assigned to the nearest
		nucleus based on the Voronoi diagram.
	"""
	intensity_image = image[mask_channel]

	label_matrix = np.zeros_like(intensity_image)
	vor = Voronoi(properties[["z", "y", "x"]])

	# Loop over each nucleus to compute the Voronoi diagram
	for cell_label, point_number in zip(properties.index, np.arange(len(properties))):
		region_label = properties.index[point_number]
		region_number = vor.point_region[point_number]
		voronoi_vertices = vor.regions[region_number]

		point_coordinates = vor.points[point_number]
		vertice_array = [vor.vertices[vertice_number] for vertice_number in voronoi_vertices]

		spherical_mask = make_spherical_mask(
			image=intensity_image, point_coordinates=point_coordinates, radius=radius
		)

		points_in_sphere = np.argwhere(spherical_mask)

		# Determine which points in the sphere belong to the Voronoi region
		sphere_points_in_vor = in_hull(points_in_sphere, vertice_array)

		# Assign the current nucleus label to the pixels in the binary mask
		label_matrix[tuple(points_in_sphere[sphere_points_in_vor].T)] = cell_label

	return label_matrix

def get_fluo_properties_voronoi(properties, image, fluo_channel, label_matrix, percentile):
	"""
	Get fluorescence properties for each cell in a Voronoi diagram.

	Parameters
	----------
	properties : pandas.DataFrame
		DataFrame containing cell properties.
	image : numpy.ndarray
		Input image.
	fluo_channel : int
		Index of the fluorescence channel.
	label_matrix : numpy.ndarray
		Label matrix of the Voronoi diagram.
	percentile : int
		Percentile to calculate for fluorescence intensity.

	Returns
	-------
	pandas.DataFrame
		DataFrame containing fluorescence properties for each cell.
	"""
	intensity_image = image[fluo_channel]

	# Initialize NaN values for mean_intensity and percentile_intensity
	properties["mean_intensity"] = np.nan
	properties["percentile_intensity"] = np.nan

	# Iterate over rows of properties DataFrame
	for index, row in properties.iterrows():
		cell_label = row["label"]

		# Mask for quantification
		mask = np.zeros_like(label_matrix)
		mask[label_matrix == cell_label] = 1

		points_in_intersection = np.argwhere(mask)

		try:
			properties.loc[index, "mean_intensity"] = np.mean(
				intensity_image[tuple(points_in_intersection[points_in_intersection].T)]
			)
			properties.loc[index, "percentile_intensity"] = np.percentile(
				intensity_image[tuple(points_in_intersection[points_in_intersection].T)],
				percentile,
			)

		except:
			pass

	return properties[["mean_intensity", "percentile_intensity", "label"]]

def voronoi_fluo_property_analysis(
	properties, image, mask_channel, radius, return_voronoi_mask=False, percentile=50
):
	"""
	Estimate the fluorescence properties of Voronoi cells.

	Calculate the Voronoi mask for each cell, then use the mask to estimate
	the fluorescence intensities inside the mask.

	Parameters
	----------
	properties : pandas.DataFrame
		DataFrame with nuclear properties, including centroid coordinates.
	image : numpy.ndarray
		3D image stack with fluorescence channels.
	mask_channel : int
		Index of the nuclear mask channel.
	radius : int
		Radius of the spherical mask used for Voronoi quantification.
	return_voronoi_mask : bool, optional
		Whether to return the labeled Voronoi tesselation as a binary mask.
		Default is False.
	percentile : int, optional
		Percentile used for fluorescence intensity quantification. Default is 50.

	Returns
	-------
	pandas.DataFrame or tuple of (pandas.DataFrame, numpy.ndarray)
		DataFrame with fluorescence properties for each Voronoi cell, and
		optionally the labeled Voronoi tesselation as a binary mask.

	Raises
	------
	ValueError
		If the Voronoi diagram cannot be computed.

	"""

	# Compute Voronoi mask for each cell
	try:
		label_matrix = make_voronoi_mask(properties, image, mask_channel, radius)
	except Exception as e:
		raise ValueError(f"Error computing Voronoi diagram: {e}")

	# Compute fluorescence properties for each channel
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
				columns={"mean_intensity": f"mean_intensity_channel_{i}"}
			)

			properties_fluo = properties_fluo.rename(
				columns={
					"percentile_intensity": f"percentile_intensity_channel_{i}"
				}
			)

			properties = properties.merge(properties_fluo, how="outer", on="label")

	# Remove original fluorescence properties from DataFrame
	properties = properties.drop(columns=["mean_intensity", "percentile_intensity"])

	if return_voronoi_mask:
		return properties, label_matrix
	else:
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
