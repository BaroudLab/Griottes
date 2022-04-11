#!/usr/bin/env pytho[n

from setuptools import setup

# with open("requirements.txt", "r") as f:
#     REQUIREMENTS = f.read().splitlines()

setup(
    name="griottes",
    version="0.0.1",
    description="Python program to generate NetworkX graphs from segmented images.",
    author="Gustave Ronteix",
    author_email="gustave.ronteix@pasteur.fr",
    url="https://github.com/BaroudLab/Griottes",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "numpy",
        "scipy",
        "networkx",
        "tqdm",
        "pandas",
        "scikit-image",
        "sklearn",
    ],
    packages=[
        "griottes",
        "griottes.analyse",
        "griottes.graphmaker",
        "griottes.graphplotter",
    ],
)
