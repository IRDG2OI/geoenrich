Geoenrich online (beta)
=======================

A beta version of the GeoEnrich web app is `available <https://geoenrich.marbec-tools.ird.fr/>`_ for basic uses. For more advanced uses, a self-hosted version is available as a `docker image <https://github.com/morand-g/geoenrich/tree/main/docker>`_.


1. File format
------------------------

Only DarwinCore archives and csv files are supported. The csv files *must* follow the following format requirements.

.. warning::
  CSV separators must be commas. A common mistake (caused by a well-known proprietary spreadsheet software) is that separators in the csv files are semicolons instead of commas. If you're having a mysterious error message when using geoenrich, please double-check this.


1.1. Species occurrences
^^^^^^^^^^^^^^^^^^^^^^^^

The following columns are mandatory: a unique ID, an occurrence date, its latitude and longitude. They must be named respectively  *id*, *date*, *latitude* and *longitude*. Here is an exemple of such a file:

.. csv-table:: Turtle occurrences
   :file: ../../geoenrich/data/webapp_turtles.csv
   :widths: 20 20 20 20 20
   :header-rows: 1

Download sample csv file `webapp_turtles.csv <https://raw.githubusercontent.com/morand-g/geoenrich/main/geoenrich/data/webapp_turtles.csv>`_ (right click -> Save As).


1.2. Areas
^^^^^^^^^^

A unique ID is mandatory, as well as the bounds of the areas that are requested. Columns must be named as in the following example:


.. csv-table:: Areas of interest
   :file: ../../geoenrich/data/webapp_areas.csv
   :widths: 10 15 15 15 15 15 15
   :header-rows: 1

Download sample csv file `webapp_areas.csv <https://raw.githubusercontent.com/morand-g/geoenrich/main/geoenrich/data/webapp_areas.csv>`_ (right click -> Save As).


2. Restrictions
------------------

For more features, you may install geoenrich (python or R) onto your computer and this will enable the following features:

- Support for csv files with custom column names.
- Support for calculated variables (Eddy Kinetic Energy and derivatives).
- No maximum download size.


3. Setting up the self-hosted version
--------------------------------------

You may use GeoEnrich and its associated webapp locally by loading a Docker container.

To do this you can download *docker-compose.yml* and *Dockerfile*, and load the container the following way::

  docker-compose up -d --build


You can then use geoenrich from the command line::

  docker exec -it geoenrich python

Or launch the web app in a browser::

  localhost:8080
