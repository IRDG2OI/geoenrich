## v0.3.1

#### Bug fixes:
- Removed depth column in *import_occurrences_csv*



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


