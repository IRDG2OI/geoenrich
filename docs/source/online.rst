Geoenrich online
================


1. File format
------------------------

Only csv files are supported.

1.1. Species occurrences
^^^^^^^^^^^^^^^^^^^^^^^^

A column with a unique ID is mandatory, to be able to link downloaded data to the corresponding occurrence. Date, latitude, and longitude columns are mandatory. Here is an exemple of such a file:

.. list-table:: turtles.csv
   :widths: 20 20 20 20 20
   :header-rows: 1

   * - id
     - latitude
     - longitude
     - date
     - Comments
   * - turtle1
     - -28.752241
     - 154.8926541
     - 2018-07-29
     - bottom feeding
   * - turtle2
     - 2.5754611
     - 72.964164
     - 2019-02-13
     - cruising
   * - turtle3
     - -21.2871554
     - 55.316446
     - 2021-01-05
     - resting


1.2. Areas
^^^^^^^^^^

A column with a unique ID is mandatory, to be able to link downloaded data to the corresponding area. All columns in the example file below are also mandatory:

.. list-table:: areas.csv
   :widths: 10 15 15 15 15 15 15
   :header-rows: 1

   * - id
     - latitude_min
     - latitude_max
     - longitude_min
     - longitude_max
     - date_min
     - date_max
   * - corsica
     - 41.2
     - 43.2
     - 8.3
     - 9.7
     - 2015-06-01
     - 2015-06-30
   * - galapagos
     - -1.5
     - 0.79
     - -91.9
     - -89
     - 2022-03-18
     - 2022-03-18
   * - samoa
     - -14.1
     - -13.3
     - -172.9
     - -171.3
     - 2018-11-21
     - 2018-11-28


2. Restrictions
------------------

For more features, you may install geoenrich (python or R) onto your computer and this will enable the following features:

- Support for DarwinCore archives.
- Support for csv files with custom column names.
- Support for calculated variables (eke and derivatives).
- No maximum download size.
