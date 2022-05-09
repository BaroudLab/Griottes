
![example workflow](https://github.com/BaroudLab/Griottes/actions/workflows/main.yml/badge.svg)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/BaroudLab/Griottes.git/main)
[![Documentation Status](https://readthedocs.org/projects/griottes/badge/?version=latest)](https://griottes.readthedocs.io/en/latest/?badge=latest)

[🍒  `Griottes` 🍒](#--griottes-)
- [I. Project description](#i-project-description)
- [II. Installation](#ii-installation)
  - [From source](#from-source)
  - [From docker image](#from-docker-image)
  - [Try in on Binder](#try-in-on-binder)
- [III. Documentation](#iii-documentation)
  - [Generating networks from labeled images or dataframes](#generating-networks-from-labeled-images-or-dataframes)
- [IV. Example](#iv-example)
  - [From a segmented image](#from-a-segmented-image)
  - [From a dataframe](#from-a-dataframe)
- [V. References](#v-references)
# 🍒  `Griottes` 🍒 

This is **🍒  Griottes🍒** a tool to maximize the amount of information you can extract from your microscopy images.

# I. Project description

**Griottes** is an easy-to-use, one-stop, Python library to extract single-cell information from your images and return the data in a networkx graph recapitulating the tissue structure.

 - It works on segmented **2D** and **3D** images, no extra fuss required! We like to use [CellPose](https://cellpose.readthedocs.io/en/latest/index.html) for our image segmentation - but that's just a question of taste. You can also use dataframes as inputs.

 - On both **2D** and **3D** images you can easily insert extra information from supplementary fluorescence channels and embed the information on the graph.

**Griottes** allows you to easily generate networks from your image data as shown in the image below.

![](images/griottes_example.png)

# II. Installation

## From source

For the moment, only installation via the repository is possible, so you'll have to download it from the command line. In the command prompt enter:

```
git clone https://github.com/BaroudLab/Griottes.git
```

This should download the project. Then install the required packages. In the terminal, navigate to the package directory and type:

```
pip install .
```

The library is now installed on your computer. An example of the library can be accessed [here](./example_notebooks/).

## From docker image

``` docker run -it -p 8888:8888 ghcr.io/baroudlab/griottes:latest ```

This will open jupyter lab in the folder with the sample notebooks (/home/jovyan/Griottes/example_notebooks) also containing paper figures.

If you want to customize starting folder, just run 

``` docker run -it -p 8888:8888 ghcr.io/baroudlab/griottes:latest jupyter lab --notebook-dir /home/jovyan/```

In order to provide your own data to the notebooks, bind your local folder as follows:

``` docker run -it -p 8892:8888 -v "${PWD}":/home/jovyan/work ghcr.io/baroudlab/griottes:latest jupyter lab --notebook-dir /home/jovyan/work```

## Try in on Binder
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/BaroudLab/Griottes.git/main)

# III. Documentation

The full documentation is available at [Griottes documentation](https://griottes.readthedocs.io/en/latest/).

## Generating networks from labeled images or dataframes

The standard API for generating graphs is very similar for Delaunay, geometric and contact-based graphs. To generate a Delaunay graph the following function is available:
```
griottes.graphmaker.graph_generation_func.generate_delaunay_graph(user_entry,
    descriptors: list = [],
    image_is_2D=False,
    min_area=0,
    analyze_fluo_channels=False,
    fluo_channel_analysis_method="basic",
    radius=30,
    distance=30,
    mask_channel=None,
)

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
    fluo_channel_analysis_method : str, optional
        the method used to analyze the fluorescence channels.
        'basic' measures the fluorescence properties within
        the cell mask, 'local_sphere' within a sphere of
        radius 'radius' and 'local_voronoi' within the 
        Voronoi tesselation of the cell.
    radius: float, optional
        radius of the sphere within the which the fluorescence
        is analyzed. Irrelevant for the 'basic' fluorescence 
        analysis method.
    image_is_2D : bool, optional
        if True, the image is analyzed as a 2D image.
        The default is False.
    min_area : int, optional
        the minimum area of a cell. The default is 0.
    analyze_fluo_channels : bool, optional
        if True, the fluorescence channels are analyzed.
        The default is False.
    mask_channel : int, optional
        The channel containing the cell masks
        The default is None.

    Returns
    -------
    nx.Graph
        The graph representation of the input.
    """

```

Similarly, for geometric graphs the user can call `griottes.graphmaker.graph_generation_func.generate_geometric_graph` to make a geometric graph and `griottes.graphmaker.graph_generation_func.generate_contact_graph` to make a contact graph.

# IV. Example
## From a segmented image

`Griottes` makes it easy to generate a network from segmented images. The resulting graph object is a networkx graph. Detailed examples can be found at this [link](https://github.com/BaroudLab/Griottes_paper).

```python
from griottes.graphmaker import graph_generation_func
test_image # segmented image

G = graph_generation_func.generate_contact_graph(test_image)
```

## From a dataframe

It is also possible to rapidly generate Delaunay or geometric graphs from a pandas dataframe containing single-cell information. It is necessary that the columns indicating the cell positions be named `x`, `y` (and `z` if the cells are distributed in 3D). Also, all the elemets in the `descriptors` list must be contained in the dataframe columns.

```python
from griottes.graphmaker import graph_generation_func
dataframe # dataframe containing single-cell data

# List of properties we wish to include in the graph
descriptors = ['x', 'y', 'z']

G_delaunay = graph_generation_func.generate_delaunay_graph(dataframe, 
                                                  descriptors = descriptors)

G_geometric = graph_generation_func.generate_geometric_graph(dataframe, 
                                                  descriptors = descriptors)
```

# V. References

[Griottes: a generalist tool for network generation from segmented tissue images](https://www.biorxiv.org/content/10.1101/2022.01.14.476345v1), Gustave Ronteix, Andrey Aristov, Valentin Bonnet, Sebastien Sart, Jeremie Sobel, Elric Esposito &  Charles N. Baroud.
