v0.2

New functions:
- Generate a stats file with variables summaries.
- Added a DarwinCore archive (data/AcDigitifera.zip) to enable testing the package with logging into GBIF.

Bug fixes:
- Handle singularity at +/- 180° longitude.
- Fixed buffer calculation so that buffer size is respected. From now occurrences which buffers contains a pole are not enriched because that represents too much data (because of the lat/long grid)

