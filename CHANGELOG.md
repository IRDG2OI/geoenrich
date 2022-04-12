v0.2

New functions:
- Generate a stats file with variables summaries.
- Added a DarwinCore archive (data/AcDigitifera.zip) to enable testing the package with logging into GBIF.
- Added import_csv function to load occurrence datasets in custom formats.
- Added a mask calculation function to return only datapoints within the buffer distance from the occurrence location.

Bug fixes:
- Handle singularity at +/- 180° longitude.
- Fixed buffer calculation so that buffer size is respected. From now occurrences which buffers contains a pole are not enriched because that represents too much data (because of the lat/long grid)


