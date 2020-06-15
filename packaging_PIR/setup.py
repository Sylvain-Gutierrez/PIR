import setuptools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from random import randint
from mpl_toolkits.mplot3d import Axes3D


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="neural_data_treatment_pkg", # Replace with your own username
    version="0.0.1",
    author="Gutierrez and Placidet",
    author_email="author@example.com",
    description="A package containing averything you need to find, print and clusterinzed spikes from multi-electrods array",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Sylvain-Gutierrez/PIR",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)