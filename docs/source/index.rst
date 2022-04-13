geoenrich 0.2 documentation
===========================

[![Read the Docs](https://img.shields.io/readthedocs/geoenrich)](https://geoenrich.readthedocs.io/en/latest/)
[![License](https://img.shields.io/github/license/morand-g/geoenrich?color=green)](https://github.com/morand-g/geoenrich/blob/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/geoenrich?color=green)](https://pypi.org/project/geoenrich/)
[![Python versions](https://img.shields.io/pypi/pyversions/geoenrich)](https://www.python.org/downloads/)
[![Last commit](https://img.shields.io/github/last-commit/morand-g/geoenrich)](https://github.com/morand-g/geoenrich/)

This package provides functionalities to enrich georeferenced events (such as species occurrences) with environmental data from satellites or models. It is intended for large numbers of occurrences: local storage is implemented to avoid redundant requests to remote servers. All downloaded environmental data are stored locally in netCDF files and can be retrieved as multidimensional numpy arrays to be used in any way.

The package provides functions to retrieve occurrence data directly from GBIF, or open a custom dataset from any source. These occurrences are then stored into csv databases that record the netCDF coordinates for the relevant subsets of each environmental variables.

All environmental variables used for enrichment must have latitude and longitude dimensions. The package also handles time and depth dimensions.

Source on GitHub at `morand-g/geoenrich <https://github.com/morand-g/geoenrich>`_

Jupyter notebook tutorial in the `Github repository <https://github.com/morand-g/geoenrich/blob/main/geoenrich/tutorial.ipynb>`_

Installation
============


.. toctree::
   :caption: Installation
   :hidden:

   install

:doc:`install`
    How to install geoenrich.


Modules
=======


.. toctree::
   :caption: Modules
   :hidden:

   enrichment
   biodiv
   satellite

:doc:`biodiv`
   The biodiv module: import DarwinCore archives, download data from GBIF.

:doc:`enrichment`
   The enrichment module: download enrichment data, retrieve downloaded data.

:doc:`satellite`
   The satellite module (internal use): netCDF file handling, data download, data retrieval.



Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
