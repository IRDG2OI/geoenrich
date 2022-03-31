# **geoenrich**

# 1. Package description
This package provides functionalities to enrich georeferenced events (such as species occurrences) with environmental data (from satellites or models). It is intended for large numbers of occurrences: local storage is implemented to avoid redundant requests to remote servers. All downloaded environmental data are stored locally in netCDF files and can be retrieved as numpy arrays to be used in any way.

The package provide functions to retrieve occurrence data directly from GBIF, or open a custom DarwinCore archive from any source. These occurrences are then stored into csv databases that record the netCDF coordinates for the relevant subsets of each environmental variables.

All environmental variables used for enrichment must have latitude and longitude dimensions. The package also handles time and depth dimensions.

Documentation on [read the docs](https://geoenrich.readthedocs.io).

## 2. Installation

Installation instructions [in the documentation](https://geoenrich.readthedocs.io/en/latest/install.html).

## 3. Using the plugin



## 4. Issues and further developments

### 4.1. User suggestions

Please feel free to raise issues or suggest improvements in the [Issues tab](https://github.com/morand-g/geoenrich/issues).

### 4.2. Planned improvements

#### 4.2.1. Depth requests

There are currently only two options regarding the depth dimension : only surface data, or data for all depths.
We plan to add an option to download data for any range of depth.

#### 4.2.2. Temporary netCDF files copies

To improve reliability we will make temporary backups of netCDF files being written in, to be able to recover data in case writing is interrupted or fails.

#### 4.2.3. Variable statistics.

For each occurrence, we will return average, standard deviation, minimum and maximum of the variable in the bounding box.
