# 🍒  `Griottes` 🍒 

This is **🍒  Griottes🍒** a tool to maximize the amount of information you can extract from your microscopy images.

# Project description

**Griottes** is an easy-to-use, one-stop, Python library to extract single-cell information from your images and return the data in a networkx graph recapitulating the tissue structure.

 - It works on segmented **2D** and **3D** images, no extra fuss required! We like to use [CellPose](https://cellpose.readthedocs.io/en/latest/index.html) for our image segmentation - but that's just a question of taste. You can also use dataframes as inputs.

 - On both **2D** and **3D** images you can easily insert extra information from supplementary fluorescence channels and embed the information on the graph.

**Griottes** allows you to easily generate networks from your image data as shown in the image below.

![](images/griottes_example.png)

# Installation

For the moment, only installation via the repository is possible, so you'll have to download it from the command line. In the command prompt enter:

```
git clone https://github.com/BaroudLab/Griottes.git
```

This should download the project. Then install the required packages. In the terminal, navigate to the package directory and type:

```
pip install .
```

The library is now installed on your computer. An example of the library can be accessed [here](Examples/griottes_example.ipynb).

# Example


## From a segmented image

`Griottes` makes it easy to generate a network from segmented images. The resulting graph object is a networkx graph. Detailed examples can be found at this [link](https://github.com/BaroudLab/Griottes_paper).

```
from griottes.graphmaker import graph_generation_func
test_image # segmented image

G = graph_generation_func.generate_contact_graph(test_image)
```

## From a dataframe

It is also possible to rapidly generate Delaunay or geometric graphs from a pandas dataframe containing single-cell information. It is necessary that the columns indicating the cell positions be named `x`, `y` (and `z` if the cells are distributed in 3D). Also, all the elemets in the `descriptors` list must be contained in the dataframe columns.

```
from griottes.graphmaker import graph_generation_func
dataframe # dataframe containing single-cell data

# List of properties we wish to include in the graph
descriptors = ['x', 'y', 'z']

G_delaunay = graph_generation_func.generate_delaunay_graph(dataframe, 
                                                  descriptors = descriptors)

G_geometric = graph_generation_func.generate_geometric_graph(dataframe, 
                                                  descriptors = descriptors)
```
