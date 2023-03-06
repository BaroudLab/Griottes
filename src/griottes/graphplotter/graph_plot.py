import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import networkx as nx
import scipy as sp
import scipy.spatial as sptl
from scipy.spatial import KDTree
from scipy.spatial import Voronoi
from scipy.spatial import ConvexHull
from scipy.spatial import distance
import pandas


def network_plot_2D(
    G,
    background_image=None,
    figsize: tuple = (8, 8),
    alpha_line=0.6,
    scatterpoint_size=20,
    legend=False,
    edge_color="k",
    line_factor=1,
    legend_fontsize=18,
    include_weights=False,
):
    # Get node positions
    pos = nx.get_node_attributes(G, "pos")

    # We fill each node with its attributed color. If none
    # then color the node in red.
    colors = {}
    for node in G.nodes():
        if "color" in G.nodes[node]:
            colors[node] = G.nodes[node]["color"]
        else:
            colors[node] = "tab:blue"

    if legend:
        legend = nx.get_node_attributes(G, "legend")

    fig, ax = plt.subplots(figsize=figsize)

    if background_image is not None:
        plt.imshow(background_image, cmap="gray")

    # Loop on the pos dictionary to extract the x,y,z coordinates of each node
    xy = {k: (v[1], v[0]) for k, v in pos.items()}

    lines = [np.array([xy[i] for i in ids]) for ids in list(G.edges)]

    if include_weights:
        try:
            weights = [1 * e[2]["weight"] for e in G.edges(data=True)]
        except KeyError:
            print(
                "no weights",
            )
            weights = [1] * len(lines)
        _ = [
            [
                ax.plot(*l.T, c=edge_color, lw=w * line_factor, alpha=alpha_line)
                for l, w in zip(lines, weights)
            ]
        ]

    else:
        weights = [1] * len(lines)
        _ = [
            [
                ax.plot(*l.T, c=edge_color, lw=w * line_factor, alpha=alpha_line)
                for l, w in zip(lines, weights)
            ]
        ]

    df = pandas.DataFrame(
        [
            {
                "x": v[0],
                "y": v[1],
                "s": scatterpoint_size,
                "nodeColor": colors[k],
                "legend": (legend[k] if legend else None),
            }
            for k, v in xy.items()
        ]
    )

    groups = df.groupby("nodeColor")

    for nodeColor, group in groups:
        if legend:
            name = group.legend.unique()[0]

            ax.plot(
                group.x,
                group.y,
                marker="o",
                c=nodeColor,
                markeredgewidth=1.5,
                markeredgecolor="k",
                linestyle="",
                ms=scatterpoint_size,
                label=name,
            )

            ax.legend(fontsize=legend_fontsize)

        else:
            ax.plot(
                group.x,
                group.y,
                marker="o",
                c=nodeColor,
                markeredgewidth=1.5,
                markeredgecolor="k",
                linestyle="",
                ms=scatterpoint_size,
            )

    # Remove tick labels
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # No ticks
    ax.set_xticks([])
    ax.set_yticks([])


def network_plot_3D(
    G,
    figsize: tuple = (8, 8),
    alpha_line=0.6,
    scatterpoint_size=20,
    legend=False,
    legend_fontsize=12,
    theta=0,
    psi=0,
    xlim=None,
    ylim=None,
    zlim=None,
):
    # Get node positions
    pos = nx.get_node_attributes(G, "pos")

    # We fill each node with its attributed color. If none
    # then color the node in red.
    colors = {}
    for node in G.nodes():
        if "color" in G.nodes[node]:
            colors[node] = G.nodes[node]["color"]
        else:
            colors[node] = "r"

    if legend:
        legend = nx.get_node_attributes(G, "legend")

    # 3D network plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    # Loop on the pos dictionary to extract the x,y,z coordinates of each node

    xyz = {k: (v[2], v[1], v[0]) for k, v in pos.items()}

    lines = [np.array([xyz[i] for i in ids]) for ids in list(G.edges)]

    _ = [plt.plot(*l.T, c="k", alpha=alpha_line) for l in lines]

    df = pandas.DataFrame(
        [
            {
                "x": v[0],
                "y": v[1],
                "z": v[2],
                "s": scatterpoint_size,
                "nodeColor": colors[k],
                "legend": (legend[k] if legend else None),
            }
            for k, v in xyz.items()
        ]
    )

    groups = df.groupby("nodeColor")

    for nodeColor, group in groups:
        if legend:
            name = group.legend.unique()[0]

            plt.plot(
                group.x,
                group.y,
                group.z,
                marker="o",
                c=nodeColor,
                markeredgewidth=1.5,
                markeredgecolor="k",
                linestyle="",
                ms=scatterpoint_size,
                label=name,
            )

            plt.legend(fontsize=legend_fontsize)

        else:
            plt.plot(
                group.x,
                group.y,
                group.z,
                marker="o",
                c=nodeColor,
                markeredgewidth=1.5,
                markeredgecolor="k",
                linestyle="",
                ms=scatterpoint_size,
            )

            plt.legend(fontsize=legend_fontsize)

    fig.patch.set_facecolor((1.0, 1, 1))
    ax.set_facecolor((1.0, 1, 1))
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    plt.rc("grid", linestyle="-", color="black")

    # Hide grid lines
    ax.grid(False)

    # Make panes transparent
    ax.xaxis.pane.fill = False  # Left pane
    ax.yaxis.pane.fill = False  # Right pane

    ax.grid(False)

    # Remove tick labels
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    # Transparent spines
    ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

    # Transparent panes
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    # No ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    # Remove grid lines
    ax.grid(False)

    # Remove tick labels
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    # Transparent spines
    ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

    # Transparent panes
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    # No ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    # No ticks
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if zlim is not None:
        ax.set_zlim(zlim)

    ax.view_init(theta, psi)
    plt.tight_layout()


def Voronoi_3D(G, angle):
    npoints = G.number_of_nodes()
    colors = nx.get_node_attributes(G, "color")

    pos = nx.get_node_attributes(G, "pos")

    points = np.zeros((npoints, 3))

    for ind in list(G.nodes):
        points[ind][0] = pos[ind][0]
        points[ind][1] = pos[ind][1]
        points[ind][2] = pos[ind][2]
    hull = ConvexHull(points)
    vor = Voronoi(points)
    nodes = vor.vertices
    center = points.mean(axis=0)
    vol = hull.volume
    dsph = (vol / 4.18) ** (1 / 3)
    tree = KDTree(hull.points)
    with plt.style.context(("ggplot")):
        fig = plt.figure(figsize=(18, 15))
        ax = Axes3D(fig)
        ax.set_xlim3d(600, 1500)
        ax.set_ylim3d(400, 1100)
        ax.set_zlim3d(0, 700)

        # Loop on the pos dictionary to extract the x,y,z coordinates of each node
        for simplex in hull.simplices:
            x1 = np.array((points[simplex[0]][0], points[simplex[1]][0]))
            x2 = np.array((points[simplex[1]][0], points[simplex[2]][0]))
            x3 = np.array((points[simplex[2]][0], points[simplex[0]][0]))
            y1 = np.array((points[simplex[0]][1], points[simplex[1]][1]))
            y2 = np.array((points[simplex[1]][1], points[simplex[2]][1]))
            y3 = np.array((points[simplex[2]][1], points[simplex[0]][1]))
            z1 = np.array((points[simplex[0]][2], points[simplex[1]][2]))
            z2 = np.array((points[simplex[1]][2], points[simplex[2]][2]))
            z3 = np.array((points[simplex[2]][2], points[simplex[0]][2]))
            ax.plot(x1, y1, z1, c="blue", alpha=0.2)
            ax.plot(x2, y2, z2, c="blue", alpha=0.2)
            ax.plot(x3, y3, z3, c="blue", alpha=0.2)

        for facet in vor.ridge_vertices:
            facet = np.asarray(facet)
            n = len(facet)
            for ind in range(0, n - 1):
                dst1 = distance.euclidean(nodes[facet[ind]], center)
                dst2 = np.array(distance.euclidean(nodes[facet[ind + 1]], center))

                if np.all(facet >= 0) and dst1 < dsph and dst2 < dsph:
                    x = np.array((nodes[facet[ind]][0], nodes[facet[ind + 1]][0]))
                    y = np.array((nodes[facet[ind]][1], nodes[facet[ind + 1]][1]))
                    z = np.array((nodes[facet[ind]][2], nodes[facet[ind + 1]][2]))
                    ax.plot(x, y, z, c="gray", linewidth=2.0, alpha=0.4)

        x = []
        y = []
        z = []
        nodeColor = []
        s = []

        for key, value in pos.items():
            x.append(value[0])
            y.append(value[1])
            z.append(value[2])
            nodeColor.append(colors[key])
            s.append(20 + 20 * G.degree(key))
        # Scatter plot
        sc = ax.scatter(x, y, z, c=nodeColor, s=s, edgecolors="k", alpha=0.7)

        # Loop on the list of edges to get the x,y,z, coordinates of the connected nodes
        # Those two points are the extrema of the line to be plotted

    # Set the initial view
    ax.view_init(30, angle)
    fig.patch.set_facecolor((1.0, 1, 1))
    ax.set_facecolor((1.0, 1, 1))
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    plt.rc("grid", linestyle="-", color="black")
    # plt.axis('off')

    # ax.set_xlabel('X axis ($\mu m$)')
    # ax.set_ylabel('Y axis ($\mu m$)')
    # ax.set_zlabel('Z axis ($\mu m$)')
    # ax.set_xlim(100, 250)
    # ax.set_ylim(100, 250)
    # ax.set_zlim(50, 90)
    return
