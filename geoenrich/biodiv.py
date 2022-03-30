"""
The module to load biodiversity data from GBIF or a local DarwinCore archive 
"""

import os
import pandas as pd
import geopandas as gpd

from dwca.read import DwCAReader
from pygbif import species, caching
from pygbif import occurrences as occ

try:
    from geoenrich.credentials import *
except:
    from geoenrich.credentials_example import *
    print('Please rename credentials_example.py to credentials.py fill in the blanks')

#caching(True) # gbif caching


############################ GBIF / Darwin Core ###########################


def get_taxon_key(query):
    
    """
    Look for a taxonomic category in GBIF database, print the best result and return its unique ID.

    Args:
        query (str): Scientific name of the genus or species to search for.
    Returns:
        int: Gbif taxon ID
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
    Download gbif data for the given request key.
    Download previously requested data if available, otherwise print request status.
    
    Args:
        request_key (int): Request key as returned by the request_from_gbif function.
    Returns:
        No return
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




def open_dwca(path = None, taxonKey = None, max_number = 10000):

    # Load Darwin Core archive located at given path.
    # Remove rows with missing coordinates or missing event date
    # Return a geodataframe with all occurences if fewer than max_number.
    # Return a random sample of max_number occurrences otherwise

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
        path = biodiv_path + 'gbif' + '/' + str(taxKey) + '.zip'

    dsA = DwCAReader(path)

    columns = ['id', 'taxonKey', 'eventDate', 'basisOfRecord',
               'decimalLatitude', 'decimalLongitude', 'depth',
               'coordinatePrecision', 'coordinateUncertaintyInMeters', 'depthAccuracy']
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
    geodf = gpd.GeoDataFrame(df)

    print('{} occurrences were loaded.'.format(len(geodf)))
    
    return(geodf)

