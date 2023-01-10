## v0.5.5

#### New functions:
- Included all packages into setup.cfg so the installation process becomes a one-liner. Updated documentation accordingly.

## v0.5.4

#### Bug fix:
- Fixed critical bug that occurred when trying to enrich a new dataset (introduced in v0.5.3).

## v0.5.3

#### New functions:
- Ability to serialize exports.retrieve_data for batch processing. Running time divided by 20.

#### Bug fixes:
- Updated data source for ph
- Fixed critical bug in geoenrich.enrichment.reset\_enrichment\_file
- Fixed bug that occurred when making new enrichments after deleting previous ones (ID assignment issue).

## v0.5.2

#### Bug fixes:
- Fix missing import in satellite.py
- Updated links for Copernicus Ocean Color / Plankton dataset.
- Error handling when enriching empty dataframe.

## v0.5.1

#### Bug fixes:
- Fix critical error when a csv with custom ID column name is imported.

## v0.5

#### New functions:
- Export data as geotiff raster. Added examples in the tutorial notebooks.
- Added a new source for wind.
- Added option to keep depth or time dimensions in exports.export\_to\_array.
- Added dump_metadata function to retrieve metadata associated with enriched variables.

#### Bug fixes:
- Data retrieval fixed when *downsample* or *depth\_request* are not default values.
- Fixed remaining errors due to the migration to pathlib paths.
- Fixed axes order errors in exports.export\_to\_array

## v0.4.6

#### New functions:
- Web app now works with most common column names, including standard DarwinCore names.

#### Bug fixes:
- Fixed a bug that occurred when data is partially in cache.


## v0.4.5

#### New functions:
- Docker image to run GeoEnrich locally as well as the associated web app (currently in beta).

#### Bug fixes:
- NetCDF4 version must be 1.5.8. Updated setup.cfg file accordingly.

## v0.4.4

#### Bug fixes:
- All paths were converted to pathlib format to improve compatibility. This means that CREDENTIALS file format changed. Please update the new credentials_example file with your credentials and path.
- Sync time dimension in calculated datasets with time dimensions in the source datasets to account for new datapoints.
- Updated Chlorophyll dataset url following a recent Copernicus reorganization.
- Fixed dataset source for CCMP surface winds

## v0.4.3

#### Bug fixes:
- Bug fixed in enrichment.calculate_indices when using all depth levels.
- Eke formula was fixed (0.5 factor was missing).
- Fixed error in exports.export_to_array that produced abnormal value in the cells neighboring NaNs.


## v0.4.2

#### New functions:
- New function to estimate download size before starting enrichment.

## v0.4.1

#### Bug fixes:
- Fixed an error in load\_areas\_file.

## v0.4

#### New functions:
- Possibility to specify colormap when exporting png files.
- Possibility to export data as resized and normalized numpy arrays.
- Possibility to provide enrichment file to retrieve_data to reduce processing time.

#### Bug fixes:
- Fixed an error that occurred when data is not available for all occurrences of the dataset.
- Fixed an error when trying to export a png file with no data.

## v0.3.2

#### Bug fixes:
- Fixed an error that was occurring when time or depth have a non-standard name in the source dataset.
- Fixed an error due to time dimension not being flagged as 'Unlimited'. New time points could not be downloaded if data points were more recent than the creation of the local netcdf file.

## v0.3.1

#### Bug fixes:
- Fixed errors in *dataloader.import_occurrences_csv*.
- Removed an empty line in catalog.csv that was causing errors.


## v0.3

#### New functions:
- Arbitrary areas can be enriched in addition to buffers around occurrences.
- Possibility to sample down remote data to achieve faster enrichment on large areas.
- Added possibility to calculate variables: EKE + derivatives
- Added documentation and tutorial about using the package with R
- Added possibility to export data as .png files.
- Modified metadata structure to handle several enrichments with different buffers on the same dataset and variable.



## v0.2

#### New functions:
- Generate a stats file with variable summaries.
- Added a DarwinCore archive (data/AcDigitifera.zip) to enable testing the package without logging into GBIF.
- Added import_csv function to load occurrence datasets in custom formats.
- Added a mask calculation function to return only datapoints within the buffer distance from the occurrence location.
- The package is now writing into temporary copies of netCDF files in case errors occur. The definitive file is only overwritten after writing is complete.
- The time buffer can now be any pair of bounds, e.g. (-14, -7) will download data between 14 days before the occurrence and 7 days before the occurrence.

#### Bug fixes:
- Handle singularity at +/- 180° longitude.
- Fixed buffer calculation so that buffer size is respected. From now on, occurrences within the buffer distance of a pole are not enriched because that represents too much data to download (because of the lat/long grid).


