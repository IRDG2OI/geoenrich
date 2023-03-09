Exports module
==============

After enriching occurrences, you can use the exports module to use the downloaded data. Several options are available:

- Produce a stats file that will give you the average, standard deviation, minimum and maximum values of your variable in the buffer around each occurrence. See :func:`geoenrich.exports.produce_stats`.
- Calculate the derivative of the environmental data between two different dates, with :func:`geoenrich.exports.get_derivative`.
- Export data as geotiff rasters with :func:`geoenrich.exports.export_raster`.
- Export png pictures, for visualization or training deep learning models for instance. See :func:`geoenrich.exports.export_png`.
- Retrieve the raw data as a numpy array with :func:`geoenrich.exports.retrieve data`.


Main functions
--------------

.. autofunction:: geoenrich.exports.collate_npy

.. autofunction:: geoenrich.exports.export_png

.. autofunction:: geoenrich.exports.export_raster

.. autofunction:: geoenrich.exports.export_to_array

.. autofunction:: geoenrich.exports.get_derivative

.. autofunction:: geoenrich.exports.produce_stats

.. autofunction:: geoenrich.exports.retrieve_data


Other functions (for internal use)
----------------------------------

.. autofunction:: geoenrich.exports.compute_stats

.. autofunction:: geoenrich.exports.fetch_data