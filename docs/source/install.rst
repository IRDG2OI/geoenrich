Installation instructions for Python
====================================


1. Work environment
-------------------

This package was tested on Ubuntu 20.04 with Python 3.8.
It should work on other operating systems and with other versions of Python 3, but this wasn't tested yet.

2. Prerequisites
----------------

Assuming you have Python3 and pip installed. This is automatic in all recent Linux distributions. Otherwise instructions are available here: `Python <https://wiki.python.org/moin/BeginnersGuide/Download>`_ and `pip <https://pip.pypa.io/en/stable/installation/>`_.


3. Installation
---------------

Installation of geoenrich is done in the classic way::

	python3 -m pip install geoenrich


4. Configuration
----------------

4.1. First configuration
^^^^^^^^^^^^^^^^^^^^^^^^

The first time you import the dataloader or enrichment module, it will display the location of the *credentials_example.py* configuration file. You will need to edit it and then remove *_example* from the file name so its name is just *credentials.py*.

In this file, you need to specify the *root_path* where all persistent data will be stored. You should pick a stable location with plenty of free space available (depending on your data download needs).

If you want to use services that require authentification, you need to specify your credentials there.
You will see 3 variables that need to be filled with GBIF credentials if you want to download occurrence data from GBIF. If you don't already have an account you can register on the `GBIF website <https://www.gbif.org/user/profile/>`_.

There is also a dictionary named *dap_creds* that is intended to store credentials to OpenDAP servers. The server domains are the keys, like the example provided for Copernicus. You can add as many credentials as you need into that dictionary.

4.2. Adding other data sources
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

At the same location, there is a *catalog.csv* file that already contains a list of available variables. If you want to use a dataset from Copernicus, you first need to register on `their website <https://resources.marine.copernicus.eu/registration-form>`_ and write your credentials in the *credentials.py* file.

If you need additional variables, you can update add a *personal_catalog.csv* file to the same folder (template on `GitHub <https://github.com/morand-g/geoenrich/blob/main/geoenrich/data/personal_catalog.csv>`_). Three columns are compulsory:

- *variable*: A unique name for that variable (user defined). It needs to be different from the variable names already in the built-in catalog.
- *url*: OpenDAP URL.
- *varname*: Name of the variable in the remote dataset.


6. Using the package
--------------------

Congrats, you can now use the `tutorial <https://geoenrich.readthedocs.io/en/latest/tutorial.html>`_ and start doing science!