Installation instructions
=========================


1. Work environment
-------------------

This package was tested on Ubuntu 20.04 with Python 3.8.
It should work on other operating systemes and with other versions of Python 3, but his wasn't tested yet.

2. Prerequisites
----------------

Some libraries are needed:

The python interfaces for these libraries will be downloaded automatically by pip when installing the package.

3. Installation
---------------

Requirements: geopandas, pygbif, netCDF4, python-dwca-reader, shapely, cftime, tqdm

Înstallation is done in the classi way::

	pip install geoenrich


4. Configuration
----------------

4.1. First configuration
^^^^^^^^^^^^^^^^^^^^^^^^

When pip installs geoenrich, it displays its installation path. Take a note of it as you will need to edit the *credentials-example.py* configuration file that is located there. After editing, remove *_example* from the file name so its name is just *credentials.py*

You need to specify the *root_path* where all persistent files will be stored. You should pick a stable location with plenty of free space available (depending on your data download needs).

If you want to use services that require authentification, you need to specify your credentials there.
You will see 3 variables for GBIF credentials that need to be filled if you want to download occurrence data from GBIF.

There is also a dictionary named *dap_creds* that is intended to store credentials to thredds servers. Use the domain as a key, like the example provided for Copernicus. You can add as many credentials as you want into that dictionary.

4.2. Adding other data sources
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

At the same location, there is a *catalog.csv* file that you can update to add other opendap servers.

5. Usage
--------

All done, you can now follow the jupyter notebook tutorial located at the `root of the Github repository <https://github.com/morand-g/geoenrich/blob/main/Geoenrich%20tutorial.ipynb>`_