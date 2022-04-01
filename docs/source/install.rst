Installation instructions
=========================


1. Work environment
-------------------

This package was tested on Ubuntu 20.04 with Python 3.8.
It should work on other operating systems and with other versions of Python 3, but this wasn't tested yet.

2. Prerequisites
----------------

Assuming you have Python3 and pip installed.


3. Installation
---------------

First some packages need to be installed::

	pip install matplotlib appdirs geojson_rewind geomet requests_cache


(We are investigating why they are not being installed automatically)

Installation of geoenrich is then done in the classic way::

	pip install geoenrich


4. Configuration
----------------

4.1. First configuration
^^^^^^^^^^^^^^^^^^^^^^^^

The first time you import one of the modules, it will display the location of the *credentials-example.py* configuration file. You will need to edit it and then remove *_example* from the file name so its name is just *credentials.py*.

You need to specify the *root_path* where all persistent files will be stored. You should pick a stable location with plenty of free space available (depending on your data download needs).

If you want to use services that require authentification, you need to specify your credentials there.
You will see 3 variables that need to be filled with GBIF credentials if you want to download occurrence data from GBIF. If you don't already have an account you can register on the `GBIF website <https://www.gbif.org/user/profile/>`_.

There is also a dictionary named *dap_creds* that is intended to store credentials to OpenDAP servers. The server domains are the keys, like the example provided for Copernicus. You can add as many credentials as you need into that dictionary.

4.2. Adding other data sources
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

At the same location, there is a *catalog.csv* file that you can update to add other opendap servers. Three columns are compulsory:

- *variable*: A unique name for that variable (user defined).
- *url*: OpenDAP URL.
- *varname*: Name of the variable in the remote dataset.

5. Usage
--------

All done, you can now follow the jupyter notebook tutorial located at the `root of the Github repository <https://github.com/morand-g/geoenrich/blob/main/Geoenrich%20tutorial.ipynb>`_