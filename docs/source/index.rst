geoenrich 0.4.2 documentation
==============================

|Read the Docs| |License| |PyPI| |Python versions| |Last commit| |DOI|


GeoEnrich provides functionalities to enrich georeferenced events (such as species occurrences) with environmental data from satellites or models. Users can specify a geographic or temporal buffer to include data in the neighbourhood of occurrences into their analyses. Two main outputs are available: the full multidimensional data array, and a simple summary of the variable in the requested area.

Sea surface temperature, chlorophyll, and 40 other environmental variables are available natively, and others can easily be added by the user. This package is intended for large numbers of occurrences: local storage is implemented to avoid redundant requests to remote servers.

The package provides functions to retrieve occurrence data directly from GBIF, or open a custom dataset from any source. Arbitrary areas defined by the user can also be enriched.

Source on GitHub at `morand-g/geoenrich <https://github.com/morand-g/geoenrich>`_


.. image:: https://github.com/morand-g/geoenrich/blob/main/geoenrich/data/readme_illus_1.png?raw=true
   :alt: Illustration of an occurrence dataset enriched with bathymetry data


.. toctree::
   :caption: Installation

   install
   r-install


.. toctree::
   :caption: Usage

   tutorial
   r-tutorial
   examples
   variables


.. toctree::
   :caption: Online version

   online

Modules
=======


.. toctree::
   :caption: Modules
   :hidden:

   dataloader
   enrichment
   exports
   satellite

:doc:`dataloader`
   The dataloader module: import DarwinCore archives, download data from GBIF.

:doc:`enrichment`
   The enrichment module: download enrichment data, handle enrichment files.

:doc:`exports`
   The exports module: retrieve downloaded data, calculate statistics, export data to pictures. 

:doc:`satellite`
   The satellite module (internal use): netCDF file handling, data download, data retrieval.



Indices and tables
==================

* :ref:`genindex`
* :ref:`search`

.. |Read the Docs| image:: https://img.shields.io/readthedocs/geoenrich
   :target: https://geoenrich.readthedocs.io/en/latest/

.. |License| image:: https://img.shields.io/github/license/morand-g/geoenrich?color=green
   :target: https://github.com/morand-g/geoenrich/blob/main/LICENSE

.. |PyPI| image:: https://img.shields.io/pypi/v/geoenrich?color=green
   :target: https://pypi.org/project/geoenrich/

.. |Python versions| image:: https://img.shields.io/pypi/pyversions/geoenrich
   :target: https://www.python.org/downloads/

.. |Last commit| image:: https://img.shields.io/github/last-commit/morand-g/geoenrich
   :target: https://github.com/morand-g/geoenrich/

.. |DOI| image:: https://zenodo.org/badge/474973185.svg
   :target: https://zenodo.org/badge/latestdoi/474973185
