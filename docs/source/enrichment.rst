Enrichment module
=================

This is the main module of the package.
It handles the local enrichment files, as well as the download of enrichment data from remote servers.

Main functions
--------------

.. autofunction:: geoenrich.enrichment.create_enrichment_file

.. autofunction:: geoenrich.enrichment.enrich

.. autofunction:: geoenrich.enrichment.enrichment_status

.. autofunction:: geoenrich.enrichment.read_ids

.. autofunction:: geoenrich.enrichment.reset_enrichment_file


Other functions (for internal use)
----------------------------------

.. autofunction:: geoenrich.enrichment.add_bounds

.. autofunction:: geoenrich.enrichment.calculate_indices

.. autofunction:: geoenrich.enrichment.checksize

.. autofunction:: geoenrich.enrichment.compute_variable

.. autofunction:: geoenrich.enrichment.download_data

.. autofunction:: geoenrich.enrichment.enrich_compute

.. autofunction:: geoenrich.enrichment.enrich_copernicus

.. autofunction:: geoenrich.enrichment.enrich_download

.. autofunction:: geoenrich.enrichment.get_enrichment_id

.. autofunction:: geoenrich.enrichment.load_enrichment_file

.. autofunction:: geoenrich.enrichment.parse_columns

.. autofunction:: geoenrich.enrichment.row_compute

.. autofunction:: geoenrich.enrichment.row_enrich

.. autofunction:: geoenrich.enrichment.save_enrichment_config