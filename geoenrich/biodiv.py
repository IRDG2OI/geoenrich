"""
biodiv module
====================================
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
    Return taxon ID for the best match to the query.

    Parameters
    ----------
    query
        scientific name of the genus or species to search for.
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

    # Make a download request and return the download request key
    # Request all occurences of the given taxson that have coordinates

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

    # Download gbif data for the given request key.
    # Need to wait a few minutes between request and download

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

