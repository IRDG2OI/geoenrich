# **geoenrich 0.5.5**

[![Read the Docs](https://img.shields.io/readthedocs/geoenrich)](https://geoenrich.readthedocs.io/en/latest/)
[![License](https://img.shields.io/github/license/morand-g/geoenrich?color=green)](https://github.com/morand-g/geoenrich/blob/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/geoenrich?color=green)](https://pypi.org/project/geoenrich/)
[![Python versions](https://img.shields.io/pypi/pyversions/geoenrich)](https://www.python.org/downloads/)
[![Last commit](https://img.shields.io/github/last-commit/morand-g/geoenrich)](https://github.com/morand-g/geoenrich/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6458090.svg)](https://doi.org/10.5281/zenodo.6458090)

# Package description

GeoEnrich provides functionalities to enrich georeferenced events (such as species occurrences) with environmental data from satellites or models. Users can specify a geographic or temporal buffer to include data in the neighbourhood of occurrences into their analyses. Two main outputs are available: a simple summary of the variable in the requested area, or the full data (as a geotiff raster, a png image, or a numpy array).

Sea surface temperature, chlorophyll, and 40 other environmental variables are available natively, and others can easily be added by the user. This package is intended for large numbers of occurrences: local storage is implemented to avoid redundant requests to remote servers.

The package provides functions to retrieve occurrence data directly from GBIF, or open a custom dataset from any source. Arbitrary areas defined by the user can also be enriched.

Documentation on [Read the Docs](https://geoenrich.readthedocs.io).

![Illustration of an occurrence dataset enriched with bathymetry data](https://github.com/morand-g/geoenrich/blob/main/geoenrich/data/readme_illus_1.png?raw=true "Illustration of an occurrence dataset enriched with bathymetry data")

This project is being developed as part of the G2OI project, cofinanced by the European union, the Reunion region, and the French Republic.

<a href="https://european-union.europa.eu/index_fr"><img alt='Europe' src="http://141.95.158.113/uploaded/img/2022/01/union_europeenne_FBtZcHO.png" height="40" ></a> &nbsp; &nbsp; <a href="https://regionreunion.com/"><img alt='Reunion' src="http://141.95.158.113/uploaded/img/2022/01/region_reunion.png" height="40" ></a> &nbsp; &nbsp; <a href="https://www.gouvernement.fr/"><img alt='France' src="http://141.95.158.113/uploaded/img/2022/01/republique_francaise.png" height="40" ></a>

## Installation

Installation instructions are in the documentation, for [python](https://geoenrich.readthedocs.io/en/latest/install.html) and [R](https://geoenrich.readthedocs.io/en/latest/r-install.html).

## Using the plugin

Jupyter Notebook tutorials are available for [python](https://geoenrich.readthedocs.io/en/latest/tutorial.html) and [R](https://geoenrich.readthedocs.io/en/latest/r-tutorial.html).

## Issues and further developments

### User suggestions

Please feel free to raise issues or suggest improvements in the [Issues tab](https://github.com/morand-g/geoenrich/issues).

### Planned improvements

- Enrich an area defined by a shapefile.
- Add bathymetry from GEBCO.
