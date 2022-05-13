"""
The module to load input data before enrichment.
The package supports two types of input: occurrences or areas.
Occurrences can be loaded straight from GBIF, from a local DarwinCore archive, or from a custom csv file.
Areas have to be loaded from a csv file. See :func:`geoenrich.dataloader.load_areas_file`
"""

import os
import pandas as pd
import geopandas as gpd

from dwca.read import DwCAReader
from pygbif import species, caching
from pygbif import occurrences as occ

import geoenrich

from geoenrich.enrichment import parse_columns

try:
    from geoenrich.credentials import *
except:
    from geoenrich.credentials_example import *
    print('Please rename credentials_example.py to credentials.py and fill in the blanks')
    print('File location: ' + os.path.split(geoenrich.__file__)[0])

pd.options.mode.chained_assignment = None
#caching(True) # gbif caching


############################ GBIF requests and downloads ###########################


def get_taxon_key(query):
    
    """
    Look for a taxonomic category in GBIF database, print the best result and return its unique ID.

    Args:
        query (str): Scientific name of the genus or species to search for.
    Returns:
        int: GBIF taxon ID
    """

    search_results = species.name_suggest(query)
    taxon = ''

    for d in search_results:
        if d['status'] == 'ACCEPTED':
            taxon = d
            break

    if isinstance(taxon, dict):
        print('Selected taxon: {}: {}'.format(taxon['rank'], taxon['scientificName']))
        return(d['key'])
    else:
        print('No taxon found')
        return(-1)



def request_from_gbif(taxonKey, override = False):


    """
    Request all georeferenced occurrences for the given taxonKey. Return the request key.
    If the same request was already done for this gbif account, return the key of the first request.
    In this case a new request can be made with *override = True*.
    
    Args:
        taxonKey (int): GBIF ID of the taxonomic category to request.
        override (bool): Force new request to be made if one already exists.
    Returns:
        int: Request key

    """

    l = occ.download_list(user = gbif_username, pwd = gbif_pw, limit = 100)

    existing = False
    request_id = None

    for e in l['results']:
        preds = e['request']['predicate']['predicates']
        for predicate in preds:
            if predicate['key'] == 'TAXON_KEY' and predicate['value'] == str(taxonKey):
                existing = True
                if not(override):
                    print('Request already made on ' + e['created'])
                    print('Run again with override = True to force new request.')
                    request_id = e['key']

    if not(existing) or override:
        req = ['taxonKey = {}'.format(taxonKey), 'hasCoordinate = True']
        res = occ.download(req, user=gbif_username, pwd=gbif_pw, email = email, pred_type='and')

        return(res[0])

    else:
        return(request_id)



def download_requested(request_key):

    """
    Download GBIF data for the given request key.
    Download previously requested data if available, otherwise print request status.
    
    Args:
        request_key (int): Request key as returned by the :func:`geoenrich.dataloader.request_from_gbif` function.
    Returns:
        None
    """

    metadata = occ.download_meta(request_key)

    if metadata['status'] == 'SUCCEEDED':

        taxKey = ''
        for p in metadata['request']['predicate']['predicates']:
            if p['key'] == 'TAXON_KEY':
                taxKey = p['value']

        path = biodiv_path + 'gbif'
        if not(os.path.exists(path)):
            os.mkdir(path)

        occ.download_get(request_key, path)
        os.rename(path + '/' + request_key + '.zip', path + '/' + str(taxKey) + '.zip')

    else:
        print('Requested data not available. Request status: ' + metadata['status'])


############################ Loading input files ###########################


def open_dwca(path = None, taxonKey = None, max_number = 10000):


    """
    Load data from DarwinCoreArchive located at given path.
    If no path is given, try to open a previously downloaded gbif archive for the given taxonomic key.
    Remove rows with a missing event date. Return a geodataframe with all occurences if fewer than max_number.
    Otherwise, return a random sample of max_number occurrences.
    
    Args:
        path (str): Path to the DarwinCoreArchive (.zip) to open.
        taxonKey (int): Taxonomic key of a previously downloaded archive from GBIF.
        max_number (int): Maximum number of rows to import. A random sample is selected.
    Returns:
        GeoDataFrame: occurrences data (only relevant columns are included)
    """

    # Load file

    if path is None:
        path = biodiv_path + 'gbif' + '/' + str(taxonKey) + '.zip'

    dsA = DwCAReader(path)

    columns = ['id', 'eventDate', 'decimalLatitude', 'decimalLongitude', 'depth', 'basisOfRecord']
    rawdf = dsA.pd_read(dsA.descriptor.core.file_location, parse_dates=True, usecols = columns)

    # Pre-sample 2*max_number to reduce processing time.
    if len(rawdf) > 2*max_number:
        rawdf = rawdf.sample(2*max_number)

    
    # Remove rows that do not correspond with in-situ observations
    idf = rawdf[rawdf['basisOfRecord'].isin(['HUMAN_OBSERVATION', 'MACHINE_OBSERVATION', 'OCCURRENCE'])]

    # Convert Lat/Long to GEOS POINT
    idf['geometry'] = gpd.points_from_xy(idf['decimalLongitude'], idf['decimalLatitude'], idf['depth'],crs="EPSG:4326")

    # Remove rows with no event date
    idf['eventDate'] = pd.to_datetime(idf['eventDate'], errors = 'coerce')
    df = idf[idf['eventDate'].notna()]

    if len(df) > max_number:
        df = df.sample(max_number)
        print('Selected {} random occurrences from the dataset'.format(max_number))

    # Convert to GeoDataFrame & standardize Date
    geodf = gpd.GeoDataFrame(df[['id', 'geometry', 'eventDate']])
    geodf.set_index(pd.Index(geodf['id'].astype(str), name='id'), inplace = True)
    geodf.drop(['id'], axis='columns', inplace = True)

    print('{} occurrences were loaded.'.format(len(geodf)))
    
    return(geodf)




def import_occurrences_csv(path, id_col, date_col, lat_col, lon_col, depth_col = None, date_format = None,
                     crs="EPSG:4326", *args, **kwargs):


    """
    Load data from a custom csv file. Additional arguments are passed down to *pandas.read_csv*.
    Remove rows with a missing event date or missing coordinates.
    Return a geodataframe with all occurences if fewer than max_number.
    Otherwise, return a random sample of max_number occurrences.
    
    Args:
        path (str): Path to the csv file to open.
        id_col (int or str): Name or index of the column containing individual occurence ids.
        date_col (int or str): Name or index of the column containing occurrence dates.
        lat_col (int or str): Name or index of the column containing occurrence latitudes (decimal degrees).
        lon_col (int or str): Name or index of the column containing occurrence longitudes (decimal degrees).
        depth_col (int or str): Name or index of the column containing occurrence depths.
        date_format (str): To avoid date parsing mistakes, specify your date format (according to strftime syntax).
        crs (str): Crs of the provided coordinates.
    Returns:
        GeoDataFrame: occurrences data (only relevant columns are included)
    """

    # Load file

    columns = [id_col, date_col, lat_col, lon_col, depth_col]
    rawdf = pd.read_csv(path, usecols = columns, index_col = id_col, *args, **kwargs)
    idf = rawdf.dropna(subset = [lat_col, lon_col])

    # Remove rows with missing coordinate
    if len(rawdf) != len(idf):
        print('Dropped {} rows with missing coordinates'.format(len(rawdf) - len(idf)))
    
    # Convert Lat/Long to GEOS POINT
    if depth_col is None:
        idf['geometry'] = gpd.points_from_xy(idf[lon_col], idf[lat_col], crs=crs)
    else:
        idf['geometry'] = gpd.points_from_xy(idf[lon_col], idf[lat_col], idf[depth_col], crs=crs)

    # Remove rows with no event date
    idf['eventDate'] = pd.to_datetime(idf[date_col], errors = 'coerce', format = date_format,
                                    dayfirst = True, infer_datetime_format = True)
    df = idf.dropna(subset = ['eventDate'])

    if len(idf) != len(df):
        print('Dropped {} rows with missing or badly formated dates'.format(len(idf) - len(df)))

    # Convert to GeoDataFrame & standardize Date
    df['id'] = df[id_col]
    geodf = gpd.GeoDataFrame(df[['id', 'geometry', 'eventDate']])
    geodf.set_index(pd.Index(geodf['id'].astype(str), name='id'), inplace = True)
    geodf.drop(['id'], axis='columns', inplace = True)

    print('{} occurrences were loaded.'.format(len(geodf)))
    
    return(geodf)



def load_areas_file(path, id_col = None, date_format = None, crs = "EPSG:4326", *args, **kwargs):


    """
    Load data to download a variable for specific areas.
    Bounds must be provided for all available dimensions.
    Bound columns must be named *{dim}_min* and *{dim}_max*, with {dim} in latitude, longitude, depth, date
    Additional arguments are passed down to *pandas.read_csv*.

    Args:
        path (str): Path to the csv file to open.
        id_col (int or str): Name or index of the column containing individual occurence ids.
        date_format (str): To avoid date parsing mistakes, specify your date format (according to strftime syntax).
        crs (str): Crs of the provided coordinates.

    Returns:
        GeoDataFrame: occurrences data (only relevant columns are included)
    """

    rawdf = pd.read_csv(path, index_col = id_col, parse_dates = ['date_min', 'date_max'],
                infer_datetime_format = True, *args, **kwargs)
    rawdf.index.rename('id', inplace=True)
    idf = pd.DataFrame()

    if 'date_min' in rawdf.columns:
        idf['mint'] = pd.to_datetime(rawdf['date_min'], errors = 'coerce', format = date_format,
                                        dayfirst = True, infer_datetime_format = True)
        idf['maxt'] = pd.to_datetime(rawdf['date_max'], errors = 'coerce', format = date_format,
                                        dayfirst = True, infer_datetime_format = True)

    idf['minx'], idf['maxx'] = rawdf['longitude_min'], rawdf['longitude_max']
    idf['miny'], idf['maxy'] = rawdf['latitude_min'], rawdf['latitude_max']

    df = idf.dropna()
    if len(idf) != len(df):
        print('Dropped {} rows with missing or badly formated coordinates'.format(len(idf) - len(df)))
    
    print('{} areas were loaded.'.format(len(df)))

    return(df)
