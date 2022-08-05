Geoenrich online (beta)
=======================

A web app will be available online soon. In the meantime a self-hosted version is available as a `docker image <https://github.com/morand-g/geoenrich/tree/main/docker>`_.


1. File format
------------------------

Only csv files are supported.

1.1. Species occurrences
^^^^^^^^^^^^^^^^^^^^^^^^

A column with a unique ID is mandatory, to be able to link downloaded data to the corresponding occurrence. Date, latitude, and longitude columns are mandatory. Here is an exemple of such a file:

.. csv-table:: Turtle occurrences
   :file: ../../geoenrich/data/webapp_turtles.csv
   :widths: 20 20 20 20 20
   :header-rows: 1

Download sample csv file `here <https://raw.githubusercontent.com/morand-g/geoenrich/main/geoenrich/data/webapp_turtles.csv>`_ (right click -> Save As).


1.2. Areas
^^^^^^^^^^

A column with a unique ID is mandatory, to be able to link downloaded data to the corresponding area. All columns in the example file below are also mandatory:


.. csv-table:: Areas of interest
   :file: ../../geoenrich/data/webapp_areas.csv
   :widths: 10 15 15 15 15 15 15
   :header-rows: 1

Download sample csv file `here <https://raw.githubusercontent.com/morand-g/geoenrich/main/geoenrich/data/webapp_areas.csv>`_ (right click -> Save As).


2. Restrictions
------------------

For more features, you may install geoenrich (python or R) onto your computer and this will enable the following features:

- Support for DarwinCore archives.
- Support for csv files with custom column names.
- Support for calculated variables (eke and derivatives).
- No maximum download size.

3. Setting up the self-hosted version
--------------------------------------

You may use GeoEnrich and its associated webapp locally by loading a Docker container.

To do this you can download *docker-compose.yml* and *Dockerfile*, and load the container the following way::

  docker-compose up -d --build


You can then use geoenrich from the command line::

  docker exec -it python-flask-2 python

Or launch the web app in a browser::

  localhost:8080
