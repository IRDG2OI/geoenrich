Installation instructions for R
===============================


1. Prerequisites
----------------

Assuming you have a version of R installed on your computer, as well as Python3 and pip. This is automatic in all recent Linux distributions. Otherwise instructions are available here: `Python <https://wiki.python.org/moin/BeginnersGuide/Download>`_ and `pip <https://pip.pypa.io/en/stable/installation/>`_.



2. Installation
---------------

The reticulate library is used to load the python package into R::

	install.packages("reticulate")

Then the package needs to be installed. This can be done directly in R. If you do not have conda on your computer, you will be asked to install Miniconda. You should say yes and R will take care of everything. This will isolate this python environment from your system environment::

	library(reticulate)

	py_install("geoenrich", pip = TRUE)


If you already have conda installed, R should find it automatically. If it does not, you can try installing the packages this way::

	conda_create("r-reticulate-geoenrich")
	conda_install("r-reticulate-geoenrich", "geoenrich", pip = TRUE)


Finally, you can check that geoenrich submodules can be imported properly::

	dataloader <- import("geoenrich.dataloader")


3. Configuration
----------------

3.1. First configuration
^^^^^^^^^^^^^^^^^^^^^^^^

The first time you import the dataloader or enrichment module, it will display the location of the *credentials_example.py* configuration file. You will need to edit it and then remove *_example* from the file name so its name is just *credentials.py*.

In this file, you need to specify the *root_path* where all persistent data will be stored. You should pick a stable location with plenty of free space available (depending on your data download needs).

If you want to use services that require authentification, you need to specify your credentials there.
You will see 3 variables that need to be filled with GBIF credentials if you want to download occurrence data from GBIF. If you don't already have an account you can register on the `GBIF website <https://www.gbif.org/user/profile/>`_.

There is also a dictionary named *dap_creds* that is intended to store credentials to OpenDAP servers. The server domains are the keys, like the example provided for Copernicus. You can add as many credentials as you need into that dictionary.

3.2. Adding other data sources
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

At the same location, there is a *catalog.csv* file that already contains a list of available variables. If you want to use a dataset from Copernicus, you first need to register on `their website <https://resources.marine.copernicus.eu/registration-form>`_ and write your credentials in the *credentials.py* file.

If you need additional variables, you can update add a *personal_catalog.csv* file to the same folder (template on `GitHub <https://github.com/morand-g/geoenrich/blob/main/geoenrich/data/personal_catalog.csv>`_). Three columns are compulsory:

- *variable*: A unique name for that variable (user defined). It needs to be different from the variable names already in the built-in catalog.
- *url*: OpenDAP URL.
- *varname*: Name of the variable in the remote dataset.


4. Using the package
--------------------

Congrats, you can now use the `tutorial <https://geoenrich.readthedocs.io/en/latest/r-tutorial.html>`_ and start doing science!