# **geo-enrich**

This package provides functionalities to enrich georeferenced events (such as species occurrences) with environmental data (from satellites or models). It is intended for large numbers of occurrences: local storage is implemented to avoid redundant requests to remote servers. All downloaded environmental data are stored locally in netCDF files and can be retrieved as numpy arrays to be used in any way.

The package provide functions to retrieve occurrence data directly from GBIF, or open a custom DarwinCore archive from any source. These occurrences are then stored into csv databases that record the netCDF coordinates for the relevant subsets of each environmental variables.

All environmental variables used for enrichment must have latitude and longitude dimensions. The package also handles time and depth dimensions.

## 1. Work environment

### 1.1 Prerequisites

Developed and tested on Ubuntu 20.04 with Python 3.8

Requirements: geopandas, pygbif, netCDF4, python-dwca-reader, shapely, cftime, tqdm

### 1.2 Paths and Credentials

Fill in the credentials file with the root path where you want to store data

Fill in credentials for servers that need authentication (gbif, copernicus)

## 2. Using the plugin

### 2.1. Prepare databases

#### 2.1.1. Loading datasets from gbif

##### getting taxon key(query)

Look for a taxonomic category in GBIF database, print the best result and return its unique ID.

* query (str) : search query

##### request_from_bif(taxonKey, override = False)

Request all georeferenced occurrences for the given taxonKey. Return the request id.

If the same request was already done for this gbif account, return the id of the first request. A new request can be made with *override = True*.

* taxonKey (int) : GBIF id of the taxonomic category to request.
* override (bool) : force new request to be made if one already exists.

##### download_requested(request_key)

Download previously requested data if available, otherwise print request status.

* request_key (int) : request key as returned by the request_from_gbif function.

#### 2.1.2. Loading datasets from darwincore archives

##### open_dwca(path = None, taxonKey = None, max*_*number = 10000)

Load data from DarwinCoreArchive located at given path. If no path is given, try to open a previously downloaded gbif archive for the given taxonomic key.

Return a GeoDataFrame with archive data.

* path (str) : path to the DarwinCoreArchive (.zip) to open.
* taxonKey (int) : taconomic key of a previously downloaded archive from GBIF.
* max_number (int) : maximum number of rows to import. A random sample is selected.

### 2.2. Enrichment

##### create_enrichment_file(gdf, dataset_ref)

Create database file that will be used to save enrichment data.

* gdf (GeoDataFrame) : data to enrich (output of open_dwca)
* dataset_ref (str) : a unique id referring to the source dataset (e.g. gbif_taxonKey)

##### reset_enrichment*_*file(dataset_ref)

Remove all enrichment data from the enrichment file. Does not remove downloaded data from netCDF files

* dataset_ref (str) : a unique id referring to the source dataset (e.g. gbif*_*taxonKey)

##### enrich(dataset_ref, var_id, geo*_*buff = 115000, time_buff = 0, depth*_*request = 'surface', slice = None, time_offset = 0)

Enrich the given dataset with data of the requested variable. All Data within the given buffers are downloaded and stored locally in netCDF files.  The Csv database files are updated with the coordinates of the relevant netCDF subsets.

* dataset_ref (str) :
* var_id (str) :
* geo_buf (int) :
* time_buff (int) :
* depth_request (str) :
* slice (int tuple) :
* time_offset (int) :

### 2.3 Using enriched data

##### retrieve_data(dataset*_*ref, occ_id)

Retrieve all the data available for a specific occurrence.

* dataset_ref (str) : 
* occ_id (str) : 

##### enrichment_state(dataset*_*ref)

Return the number of occurrences of the given dataset that are already enriched, for each variable.

* datset_ref (str) :

## 3. Issues and further developments

### 3.1. Depth

There are only two options regarding the depth dimension : only surface data, or data for all depths