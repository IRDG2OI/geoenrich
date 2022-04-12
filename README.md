# **geoenrich**

[![Read the Docs](https://img.shields.io/readthedocs/geoenrich)](https://geoenrich.readthedocs.io/en/latest/)
[![License](https://img.shields.io/github/license/morand-g/geoenrich?color=green)](https://github.com/morand-g/geoenrich/blob/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/geoenrich?color=green)](https://pypi.org/project/geoenrich/)
[![Python versions](https://img.shields.io/pypi/pyversions/geoenrich)](https://www.python.org/downloads/)
[![Last commit](https://img.shields.io/github/last-commit/morand-g/geoenrich)](https://github.com/morand-g/geoenrich/)


# Package description
This package provides functionalities to enrich georeferenced events (such as species occurrences) with environmental data from satellites or models. It is intended for large numbers of occurrences: local storage is implemented to avoid redundant requests to remote servers. All downloaded environmental data are stored locally in netCDF files and can be retrieved as multidimensional numpy arrays to be used in any way.

The package provides functions to retrieve occurrence data directly from GBIF, or open a custom DarwinCore archive from any source. These occurrences are then stored into csv databases that record the netCDF coordinates for the relevant subsets of each environmental variables.

All environmental variables used for enrichment must have latitude and longitude dimensions. The package also handles time and depth dimensions.

Documentation on [Read the Docs](https://geoenrich.readthedocs.io).

![Illustration of an occurrence dataset enriched with bathymetry data](data/readme_illus_1.png?raw=true "Illustration of an occurrence dataset enriched with bathymetry data")

## Installation

Installation instructions [in the documentation](https://geoenrich.readthedocs.io/en/latest/install.html).

## Using the plugin

A Jupyter Notebook tutorial is available at the [root of the repository](https://github.com/morand-g/geoenrich/blob/main/Geoenrich%20tutorial.ipynb).

## Issues and further developments

### User suggestions

Please feel free to raise issues or suggest improvements in the [Issues tab](https://github.com/morand-g/geoenrich/issues).

### Planned improvements

#### Depth requests

There are currently only two options regarding the depth dimension : only surface data, or data for all depths.
We plan to add an option to download data for any range of depth.

#### Temporary netCDF files copies

To improve reliability we will make temporary backups of netCDF files being written in, to be able to recover data in case writing is interrupted or fails.

#### Variable statistics.

For each occurrence, we will return average, standard deviation, minimum and maximum of the variable in the bounding box.
