Dataloader module
=================

This module provides functions to load input data before enrichment.
The package supports two types of input: occurrences or areas.
Occurrences can be loaded straight from GBIF, from a local DarwinCore archive, or from a custom csv file.
Areas have to be loaded from a csv file. See :func:`geoenrich.dataloader.load_areas_file`.


.. autofunction:: geoenrich.dataloader.download_requested

.. autofunction:: geoenrich.dataloader.get_taxon_key

.. autofunction:: geoenrich.dataloader.import_occurrences_csv

.. autofunction:: geoenrich.dataloader.load_areas_file

.. autofunction:: geoenrich.dataloader.open_dwca

.. autofunction:: geoenrich.dataloader.request_from_gbif