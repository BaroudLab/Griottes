#!/usr/bin/env pytho[n

from setuptools import setup

# with open("requirements.txt", "r") as f:
#     REQUIREMENTS = f.read().splitlines()

setup(
    name="griottes",
    version="0.0.5",
    description="Python program to generate NetworkX graphs from segmented images.",
    author="Gustave Ronteix, Andrey Aristov",
    author_email="gustave.ronteix@pasteur.fr",
    url="https://github.com/BaroudLab/Griottes",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
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
        "matplotlib",
        "seaborn",
    ],
    packages=[
        "griottes",
        "griottes.analyse",
        "griottes.graphmaker",
        "griottes.graphplotter",
    ],
)
