# **geoenrich 0.2**

[![Read the Docs](https://img.shields.io/readthedocs/geoenrich)](https://geoenrich.readthedocs.io/en/latest/)
[![License](https://img.shields.io/github/license/morand-g/geoenrich?color=green)](https://github.com/morand-g/geoenrich/blob/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/geoenrich?color=green)](https://pypi.org/project/geoenrich/)
[![Python versions](https://img.shields.io/pypi/pyversions/geoenrich)](https://www.python.org/downloads/)
[![Last commit](https://img.shields.io/github/last-commit/morand-g/geoenrich)](https://github.com/morand-g/geoenrich/)
[![DOI](https://zenodo.org/badge/474973185.svg)](https://zenodo.org/badge/latestdoi/474973185)

# Package description
This package provides functionalities to enrich georeferenced events (such as species occurrences) with environmental data from satellites or models. It is intended for large numbers of occurrences: local storage is implemented to avoid redundant requests to remote servers. All downloaded environmental data are stored locally in netCDF files and can be retrieved as multidimensional numpy arrays to be used in any way.

The package provides functions to retrieve occurrence data directly from GBIF, or open a custom dataset from any source. These occurrences are then stored into csv databases that record the netCDF coordinates for the relevant subsets of each environmental variables.

All environmental variables used for enrichment must have latitude and longitude dimensions. The package also handles time and depth dimensions.

Documentation on [Read the Docs](https://geoenrich.readthedocs.io).

This project is being developed as part of the G2OI project, cofinanced by the European union, the Reunion region, and the French Republic.

[![Europe](http://141.95.158.113/uploaded/img/2022/01/union_europeenne_FBtZcHO.png | height=25px)](https://european-union.europa.eu/index_fr)
[![Reunion](http://141.95.158.113/uploaded/img/2022/01/region_reunion.png | height=25px)](https://regionreunion.com/)
[![France](http://141.95.158.113/uploaded/img/2022/01/republique_francaise.png | height=25px)](https://www.gouvernement.fr/)

![Illustration of an occurrence dataset enriched with bathymetry data](https://github.com/morand-g/geoenrich/blob/main/geoenrich/data/readme_illus_1.png?raw=true "Illustration of an occurrence dataset enriched with bathymetry data")

## Installation

Installation instructions [in the documentation](https://geoenrich.readthedocs.io/en/latest/install.html).

## Using the plugin

A Jupyter Notebook tutorial is available [here](https://github.com/morand-g/geoenrich/blob/main/geoenrich/tutorial.ipynb).

## Issues and further developments

### User suggestions

Please feel free to raise issues or suggest improvements in the [Issues tab](https://github.com/morand-g/geoenrich/issues).

### Planned improvements

#### Depth requests

There are currently only two options regarding the depth dimension: only surface data, or data for all depths.
We plan to add an option to download data for any range of depth.

#### Data export formats

There is a plan to add export options for enrichment data: netCDF files (per occurrence or per dataset), as well as images for visualization or CNN training.