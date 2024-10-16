Examples of input data
======================

1. Enriching occurrences
------------------------

If you wish to enrich the area in a buffer around a specific point (such as a species occurrence), you have two choices:

1.1. DarwinCore archive
^^^^^^^^^^^^^^^^^^^^^^^

A DarwinCore archive is a standard format for biodiversity data. In this case, column names and content follow CF conventions, which means that you don't have to worry about formatting or column names. You can use the :func:`geoenrich.dataloader.open_dwca` function straight away.

1.2. CSV file
^^^^^^^^^^^^^

You may also use a custom csv file that does not follow any standard. In this case, when you use the :func:`geoenrich.dataloader.import_occurrences_csv` function, you have to specify the column names of your file.

A column with a unique ID is mandatory, to be able to link downloaded data to the corresponding occurrence. Date, latitude, and longitude columns are mandatory. Here is an exemple of such a file:

.. list-table:: turtles.csv
   :widths: 10 20 20 10 20 20
   :header-rows: 1

   * - ID
     - Lat
     - Lon
     - Depth
     - Day
     - Comments
   * - turtle1
     - -28.752241
     - 154.8926541
     - 12
     - 2018-07-29
     - bottom feeding
   * - turtle2
     - 2.5754611
     - 72.964164
     - 4
     - 2019-02-13
     - cruising
   * - turtle3
     - -21.2871554
     - 55.316446
     - 3
     - 2021-01-05
     - resting

This file can be imported the following way::

	geodf = import_occurrences_csv(	path = 'path-to-folder/turtles.csv',
					id_col = 'ID',
					date_col = 'Day',
					lat_col = 'Lat',
					lon_col = 'Lon',
          depth_col = 'Depth')

The date parser should work with any common date format. If you encounter problems with a custom date format, you can try to provide an explicit format string using the *date_format* parameter. See *strptime* documentation `here <https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior>`_.

2. Enriching areas
------------------

If you wish to download environmental data for arbitrary areas and dates, you have to provide a csv with predefined column names. Here is an example of such a file:

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

You can then use the :func:`geoenrich.dataloader.load_areas_file` function straight away.