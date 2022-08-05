Geoenrich online (beta)
=======================

A web app is available as a `docker image <https://github.com/morand-g/geoenrich/tree/main/docker>`_ for self hosting.


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

Download sample csv file `here <https://raw.githubusercontent.com/morand-g/geoenrich/main/geoenrich/data/webapp_turtles.csv>`_.


1.2. Areas
^^^^^^^^^^

A column with a unique ID is mandatory, to be able to link downloaded data to the corresponding area. All columns in the example file below are also mandatory:


.. csv-table:: Areas of interest
   :file: ../../geoenrich/data/webapp_areas.csv
   :widths: 10 15 15 15 15 15 15
   :header-rows: 1

Download sample csv file `here <https://raw.githubusercontent.com/morand-g/geoenrich/main/geoenrich/data/webapp_areas.csv>`_.


2. Restrictions
------------------

For more features, you may install geoenrich (python or R) onto your computer and this will enable the following features:

- Support for DarwinCore archives.
- Support for csv files with custom column names.
- Support for calculated variables (eke and derivatives).
- No maximum download size.
