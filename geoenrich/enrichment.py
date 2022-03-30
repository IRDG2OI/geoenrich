"""
The core module of geoenrich
"""

import os

import numpy as np
import pandas as pd
import geopandas as gpd
import netCDF4 as nc

from shapely import wkt
from datetime import datetime

from copy import deepcopy

from tqdm import tqdm

from geoenrich.satellite import *

try:
    from geoenrich.credentials import *
except:
    from geoenrich.credentials_example import *
    print('Please rename credentials_example.py to credentials.py fill in the blanks')


tqdm.pandas()

pd.options.mode.chained_assignment = None 



##########################################################################
######                         Enrichment                           ######
##########################################################################


############################# Batch operations #################################



def enrich(dataset_ref, var_id, geo_buff = 115000, time_buff = 0, depth_request = 'surface', slice = None, time_offset = 0):

    # Open local enrichment file, enrich with requested variable
    # Update local netcdf file with downloaded data
    # If the enrichment file is large, use slice argument to only enrich some rows.

    """
    Enrich the given dataset with data of the requested variable.
    All Data within the given buffers are downloaded (if needed) and stored locally in netCDF files.
    The enrichment file is updated with the coordinates of the relevant netCDF subsets.
    If the enrichment file is large, use slice argument to only enrich some rows.
    
    Args:
        dataset_ref (str): The enrichment file name (e.g. gbif_taxonKey).
        var_id (str) : ID of the variable to download
        geo_buf (int) : Geographic buffer for which to download data around occurrence point (kilometers)
        time_buff (int) : Time buffer for which to download data around occurrence time (hours)
        depth_request (str) : For 4D data: 'surface' only download surface data. Anything else downloads everything.
        slice (int tuple) : Slice of the enrichment file to use for enrichment.
        time_offset (int) : Download environmental data *time_offset* days before occurrence time.
    Returns:
        None
    """

    # Load biodiv file
    original = load_enrichment_file(dataset_ref)


    # Load variable information
    var = get_var_catalog()[var_id]
    
    
    # Test enrichment on random rows
    to_enrich = original[['geometry', 'eventDate']]

    if slice is not None:
        to_enrich = to_enrich.iloc[slice[0]:slice[1]]

    indices = enrich_download(to_enrich, var['varname'], var['var_id'], var['url'], geo_buff, time_buff, depth_request, time_offset)

    # If variable is already present, update it
    if any(var['var_id'] in col for col in original.columns):
        original.update(indices)
        updated = original

    # Else add new columns
    else:
        updated = original.merge(indices, how = 'left', left_index = True, right_index = True)


    # Save file
    updated.to_csv(biodiv_path + dataset_ref + '.csv')



def enrich_download(geodf, varname, var_id, url, geo_buff, time_buff, depth_request, time_offset):
    
    # Download data for the requested occurrences and buffer into loca netcdf file
    # Add columns to geodf with indices of the data of interest in the ncdf file
    # Returns updated GeoDataFrame

    
    # Calculate cube bounds

    geodf = add_bounds(geodf, geo_buff, time_buff, time_offset)


    # Get netcdf metadata

    remote_ds = nc.Dataset(url)
    dimdict, var = get_metadata(remote_ds, varname)
    var['var_id'] = var_id

    # Check if local netcdf files already exist

    if  not(os.path.exists(sat_path + var['var_id'] + '.nc')) or \
        not(os.path.exists(sat_path + var['var_id'] + '_downloaded.nc')):

        create_nc(get_var_catalog()[var_id])

    local_ds = nc.Dataset(sat_path + var['var_id'] + '.nc', mode ='r+')
    bool_ds = nc.Dataset(sat_path + var['var_id'] + '_downloaded.nc', mode ='r+')

    # Remove out of timeframe datapoints

    if 'time' in dimdict:
        t1, t2 = min(dimdict['time']['vals']), max(dimdict['time']['vals'])
        geodf2 = geodf[(geodf['mint'] >= t1) & (geodf['maxt'] <= t2)]
        geodf_na = geodf[(geodf['mint'] < t1) | (geodf['maxt'] > t2)].index
        print('Ignoring {} rows because data is not available at these dates'.format(len(geodf) - len(geodf2)))
    else:
        geodf2 = geodf

    # Apply query to each row sequentially

    res = geodf2.progress_apply(row_enrich, axis=1, args = (remote_ds, local_ds, bool_ds, dimdict, var, depth_request), 
                            result_type = 'expand')

    local_ds.close()
    bool_ds.close()
    remote_ds.close()
    
    if 'time' in dimdict:
        missing = pd.DataFrame(-1, columns = res.columns, index = geodf_na)
        print('Enrichment over')
        return(pd.concat([res, missing]))

    else:
        print('Enrichment over')
        return(res)


def add_bounds(geodf, geo_buff, time_buff, time_offset):

    # Calculate geo buffer and time buffer
    # Add columns for cube limits: 'minx', 'maxx', 'miny', 'maxy', 'mint', 'maxt'


    # Prepare geo bounds
    buffers = geodf['geometry'].to_crs('+proj=cea').buffer(geo_buff, cap_style = 3).to_crs(geodf.crs)
    geodf = geodf.join(buffers.bounds)
    
    # Prepare time bounds

    buff = np.timedelta64(time_buff, 'h')
    offset = np.timedelta64(time_offset, 'D')
    geodf['mint'] = pd.to_datetime(geodf['eventDate'] - offset - buff)
    geodf['maxt'] = pd.to_datetime(geodf['eventDate'] - offset + buff)
    geodf.rename(columns={'geometry':'point'}, inplace=True)

    return(gpd.GeoDataFrame(geodf, geometry = buffers))


############################# Element-wise enrichment #################################


def row_enrich(row, remote_ds, local_ds, bool_ds, dimdict, var, depth_request):

    # Query geospatial data for the given GeoDataFrame row
    # Save netcdf data to disk and return their coordinates

    # Find indices for region of interest

    ind = calculate_indices(dimdict, row, var, depth_request)
    params = [dimdict[n]['standard_name'] for n in var['params']]
    ordered_indices = [ind[p] for p in params]

    
    download_data(remote_ds, local_ds, bool_ds, var, dimdict, ind)

    # Return coordinates of the saved subset for data retrieval

    colnames = []
    coords = []

    for p in params:
        colnames.extend([var['var_id'] + '_' + dimdict[p]['standard_name'] + '_min',
                         var['var_id'] + '_' + dimdict[p]['standard_name'] + '_max'])
        coords.extend([ind[p]['min'], ind[p]['max']])

    return(pd.Series(coords, index = colnames))




def calculate_indices(dimdict, row, var, depth_request):

    # Calculate indices of interest for the given bounds

    ind = {}

    # latitude lower and upper index
    # make sure the slice contains at least one element

    lat1 = np.argmin( np.abs( dimdict['latitude']['vals'] - row['miny'] ) )
    lat2 = np.argmin( np.abs( dimdict['latitude']['vals'] - row['maxy'] ) )
    ind['latitude'] = {'min': min(lat1, lat2), 'max': max(lat1, lat2)}
    if lat1 == lat2:
        ind['latitude']['max'] = ind['latitude']['max'] + 1

    # longitude lower and upper index
    # make sure the slice contains at least one element

    lon1 = np.argmin( np.abs( dimdict['longitude']['vals'] - row['minx'] ) )
    lon2 = np.argmin( np.abs( dimdict['longitude']['vals']  - row['maxx'] ) )  
    ind['longitude'] = {'min': min(lon1, lon2), 'max': max(lon1, lon2)}
    if lon1 == lon2:
        ind['longitude']['max'] = ind['longitude']['max'] + 1


    params = [dimdict[n]['standard_name'] for n in var['params']]

    # if time in dimensions, get lower, upper, and best fit indices
    # make sure the slice contains at least one element

    if 'time' in params:
        t1 = np.argmin( np.abs( dimdict['time']['vals'] - row['mint'] ) )
        t2 = np.argmin( np.abs( dimdict['time']['vals'] - row['maxt'] ) ) 
        ind['time'] = {'min': min(t1, t2), 'max': max(t1, t2)}
        if t1 == t2:
            ind['time']['max'] = ind['time']['max'] + 1

    # if depth is a dimension, either select surface layer or return everything

    if ('depth' in dimdict) and (dimdict['depth']['name'] in params):
        if depth_request == 'surface':
            d1 = np.argmin( np.abs( dimdict['depth']['vals'] ) )
            ind['depth'] = {'min': d1, 'max': d1+1}
        else:
            ind['depth'] = {'min': 0, 'max': len(dimdict['depth']['vals'])}

    
    return(ind)


def download_data(remote_ds, local_ds, bool_ds, var, dimdict, ind):

    # Download missing data to the disk

    params = [dimdict[n]['standard_name'] for n in var['params']]
    ordered_indices = [ind[p] for p in params]
    check = multidimensional_slice(bool_ds, var['name'], ordered_indices).data
    totalsize = np.prod([i['max'] - i['min'] for i in ind.values()])

    if 'time' in ind:

        lentime = ind['time']['max'] - ind['time']['min']
        flatcheck = check.reshape((lentime, -1)).sum(axis = 1)
        checklist = (flatcheck == (totalsize / lentime))

    else:
        checklist = np.array(check.sum() == totalsize)

    # If all data is present, do nothing

    if checklist.all():

        pass

    # If some data is present, separate present and missing data along time axis
    # Only download data for time points where data is missing

    elif checklist.any():

        start = 0
        started = False

        for i in range(lentime):
            is_present = checklist[i]

            if not(started) and not(is_present):
                start = i
                started = True
            elif started and is_present:
                started = False
                new_ind = deepcopy(ind)
                new_ind['time']['min'] = ind['time']['min'] + start
                new_ind['time']['max'] = ind['time']['min'] + i
                download_data(remote_ds, local_ds, bool_ds, var, dimdict, new_ind)
                

        if(started):
            new_ind = deepcopy(ind)
            new_ind['time']['min'] = ind['time']['min'] + start
            new_ind['time']['max'] = ind['time']['min'] + lentime
            download_data(remote_ds, local_ds, bool_ds, var, dimdict, new_ind)
            

    # Otherwise download everything

    else:
        data = multidimensional_slice(remote_ds, var['name'], ordered_indices)
        # print('DL: ' + str(ordered_indices[0]['max'] - ordered_indices[0]['min']))
        insert_multidimensional_slice(local_ds, var['name'], data, ordered_indices)
        insert_multidimensional_slice(bool_ds, var['name'], np.ones(data.shape), ordered_indices)

        # Update time variable in case new points were added in the remote dataset
        if ('time' in ind) and (ind['time']['max'] > len(dimdict['time']['vals'])):
            timename = dimdict['time']['name']
            local_ds.variables[timename][ind['time']['min']:ind['time']['max']] = \
                    remote_ds.variables[timename][ind['time']['min']:ind['time']['max']]




##########################################################################
###########                 Enrichment files                ##############
##########################################################################



def load_enrichment_file(dataset_ref):

    # Load biodiversity file and all geospatial data columns

    filepath = biodiv_path + dataset_ref + '.csv'

    df = pd.read_csv(filepath, parse_dates = ['eventDate'], infer_datetime_format = True, index_col = 0)
    df['geometry'] = df['geometry'].apply(wkt.loads)

    print('{} occurrences were loaded from enrichment file'.format(len(df)))
    return(gpd.GeoDataFrame(df, crs = 'epsg:4326'))



def create_enrichment_file(gdf, dataset_ref):

    """
    Create database file that will be used to save enrichment data.
    
    Args:  
        gdf (geopandas.GeoDataFrame): data to enrich (output of :ref:`geoenrich.Biodiv.open_dwca`)
        dataset_ref (str): The enrichment file name (e.g. gbif_taxonKey).
    Returns:
        None
    """

    filepath = biodiv_path + dataset_ref + '.csv'

    if(os.path.exists(filepath)):
        print('Abort. File already exists at ' + filepath)
    else:
        to_save = gdf[['id', 'taxonKey', 'eventDate', 'geometry']]
        to_save.set_index(pd.Index('gbif_' + to_save['id'].astype(str), name='id'), inplace = True)
        to_save.drop(['id'], axis='columns', inplace = True)
        to_save = to_save.reindex(columns=['taxonKey', 'geometry', 'eventDate'])
        to_save.to_csv(filepath)
        print('File saved at ' + filepath)



def reset_enrichment_file(dataset_ref):

    """
    Remove all enrichment data from the enrichment file. Does not remove downloaded data from netCDF files

    Args:
        dataset_ref (str): The enrichment file name (e.g. gbif_taxonKey).
    Returns:
        None
    """

    filepath = biodiv_path + dataset_ref + '.csv'
    df = pd.read_csv(filepath, parse_dates = ['eventDate'], infer_datetime_format = True, index_col = 0)
    to_save = df[['taxonKey', 'eventDate', 'geometry']]
    to_save.to_csv(filepath)
    print('Enrichment file for dataset ' + dataset_ref + ' was reset.')




def enrichment_status(dataset_ref):

    """
    Return the number of occurrences of the given dataset that are already enriched, for each variable.

    Args:
        datset_ref (str): The enrichment file name (e.g. gbif_taxonKey).
    Returns:
        pandas.DataFrame: A table of variables and statuses of enrichment.
    """

    filepath = biodiv_path + dataset_ref + '.csv'
    df = pd.read_csv(filepath, parse_dates = ['eventDate'], infer_datetime_format = True, index_col = 0)
    col_indices = parse_columns(df)
    res = pd.DataFrame(index = ['Enriched', 'Not enriched', 'Data not available'])

    for v in col_indices:
        any_col = list(col_indices[v].values())[0]['min']
        colname = df.columns[any_col]
        is_downloaded = pd.Series(dtype = str, index = df.index)
        is_downloaded.loc[df[colname] > 0] = 'Enriched'
        is_downloaded.loc[df[colname] == -1] = 'Data not available'
        is_downloaded.loc[df[colname].isna()] = 'Not enriched'
        counts = is_downloaded.value_counts()
        res = res.join(pd.Series(counts, name = v))
    
    return(res.fillna(0).astype(int))




def parse_columns(df):

    # Return column indices sorted by variable and dimension.

    cols = [c.split('_') for c in df.columns]
    cat = get_var_catalog()
    ind = {}

    for i in range(len(cols)):
        c = cols[i]
        if c[0] in cat:
            if c[0] in ind:
                if c[1] in ind[c[0]]:
                    ind[c[0]][c[1]][c[2]] = i
                else:
                    ind[c[0]][c[1]] = {c[2]: i}
            else:
                ind[c[0]] = {c[1]: {c[2]: i}}

    return(ind)




def retrieve_data(dataset_ref, occ_id):

    """
    Retrieve all available data for a specific occurrence.
    
    Args:
        dataset_ref (str): The enrichment file name (e.g. gbif_taxonKey).
        occ_id (str): ID of the occurrence to get data for. Can be obtained with :function:`geoenrich.Enrichment.read_ids`
    Returns:
        dict: A dictionary of all available variables with corresponding data, unit, and coordinates.
    """

    filepath = biodiv_path + dataset_ref + '.csv'
    df = pd.read_csv(filepath, parse_dates = ['eventDate'], infer_datetime_format = True, index_col = 0)
    row = df.loc[occ_id]
    cat = get_var_catalog()
    
    ind = parse_columns(df)

    # Read indices into a dictionary

    results = {}

    for v in ind:
        var_ind = ind[v]

        if -1 in [row.iloc[d['min']] for d in var_ind.values()]:
            results[v] = {'coords': None, 'values': None}

        else:
            ds = nc.Dataset(sat_path + v + '.nc')
            unit = getattr(ds.variables[cat[v]['varname']], 'units', 'Unspecified')

            dimdict, var = get_metadata(ds, cat[v]['varname'])

            params = [dimdict[n]['standard_name'] for n in var['params']]
            ordered_indices_cols = [var_ind[p] for p in params]
            ordered_indices = [{'min': int(row.iloc[d['min']]),
                                'max': int(row.iloc[d['max']])}
                                for d in ordered_indices_cols]

            data = multidimensional_slice(ds, var['name'], ordered_indices)
            coordinates = []
            for p in params:
                i1, i2 = int(row.iloc[var_ind[p]['min']]), int(row.iloc[var_ind[p]['min']])
                coordinates.append([p, ds.variables[dimdict[p]['name']][i1:i2]])

            ds.close()
            results[v] = {'coords': coordinates, 'values': data, 'unit': unit}

    return(results)


def read_ids(dataset_ref):

    # Return a list of all ids of the given dataset

    filepath = biodiv_path + dataset_ref + '.csv'
    df = pd.read_csv(filepath, parse_dates = ['eventDate'], infer_datetime_format = True, index_col = 0)

    return(list(df.index))


##########################################################################
######                         Exploitation                         ######
##########################################################################


def generate_images(data):

    return('')