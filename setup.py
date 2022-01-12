#!/usr/bin/env python

from setuptools import setup

setup(
    name="griottes",
    version="0.0.1",
    description="Extracting single-cell data from segmented images",
    author="Gustave Ronteix",
    author_email="gustave.ronteix@pasteur.fr",
    url="https://github.com/microfluidix/graphs-on-chip",
    install_requires=[
        "numpy",
        "tifffile",
        "scipy",
        "networkx",
        "pytest",
        "tqdm",
        "pandas",
        "click",
        "pip",
        "mkl",
        "scikit-image",
        "matplotlib",
        "sklearn",
        "tox",
    ],
    packages=[
        "griottes",
        "griottes.analyse",
        "griottes.graphmaker",
        "griottes.graphplotter",
    ],
)
