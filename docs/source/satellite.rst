Satellite module
================

This module handles netCDF file operations. Functions are not to be used directly, except :func:`geoenrich.exports.dump_metadata`.

Main functions
--------------

.. autofunction:: geoenrich.satellite.dump_metadata


Other functions (for internal use)
----------------------------------

.. autofunction:: geoenrich.satellite.create_nc

.. autofunction:: geoenrich.satellite.create_nc_calculated

.. autofunction:: geoenrich.satellite.create_nc_copernicus

.. autofunction:: geoenrich.satellite.ellipsoid_mask

.. autofunction:: geoenrich.satellite.get_metadata

.. autofunction:: geoenrich.satellite.get_metadata_copernicus

.. autofunction:: geoenrich.satellite.get_var_catalog

.. autofunction:: geoenrich.satellite.insert_multidimensional_slice

.. autofunction:: geoenrich.satellite.multidimensional_slice

.. autoclass:: geoenrich.satellite.NpEncoder