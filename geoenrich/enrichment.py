"""
The core module of geoenrich
"""

import os
import shutil

import numpy as np
import pandas as pd
import geopandas as gpd
import netCDF4 as nc

from shapely import wkt
from datetime import datetime

from copy import deepcopy

from tqdm import tqdm

import geoenrich
from geoenrich.satellite import *

try:
    from geoenrich.credentials import *
except:
    from geoenrich.credentials_example import *


tqdm.pandas()

pd.options.mode.chained_assignment = None 



##########################################################################
######                         Enrichment                           ######
##########################################################################


############################# Batch operations #################################



def enrich(dataset_ref, var_id, geo_buff = 115, time_buff = (0,0), depth_request = 'surface', slice = None, downsample = {}):

    """
    Enrich the given dataset with data of the requested variable.
    All Data within the given buffers are downloaded (if needed) and stored locally in netCDF files.
    The enrichment file is updated with the coordinates of the relevant netCDF subsets.
    If the enrichment file is large, use slice argument to only enrich some rows.
    
    Args:
        dataset_ref (str): The enrichment file name (e.g. gbif_taxonKey).
        var_id (str): ID of the variable to download.
        geo_buf (int): Geographic buffer for which to download data around occurrence point (kilometers).
        time_buff (float tuple): Time bounds for which to download data around occurrence day (days). For instance, time_buff = (-7, 0) will download data from 7 days before the occurrence to the occurrence date.
        depth_request (str): Used when depth is a dimension. 'surface' only downloads surface data. Anything else downloads everything.
        slice (int tuple): Slice of the enrichment file to use for enrichment.
        downsample (dict): Number of points to skip between each downloaded point, for each dimension, using its standard name as a key.
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

    # Calculate cube bounds

    to_enrich = add_bounds(to_enrich, geo_buff, time_buff)

    if var['url'] == 'calculated':
        indices = enrich_compute(to_enrich, var['var_id'], downsample)
    else:
        indices = enrich_download(to_enrich, var['varname'], var['var_id'], var['url'], depth_request, downsample)

    # If variable is already present, update it
    if any(var_id + '_' in col for col in original.columns) and len(indices):
        original.update(indices)
        updated = original

    # If indices is empty
    elif not(len(indices)):
        updated = original

    # Else add new columns
    else:
        updated = original.merge(indices, how = 'left', left_index = True, right_index = True)

    # Fill unenriched rows with -1
    new_columns = [var_id + '_'  in name for name in updated.columns]
    missing_index = updated.loc[:,new_columns].isnull().all(axis=1)
    updated.loc[missing_index,new_columns] = -1

    # Save file
    updated.to_csv(biodiv_path + dataset_ref + '.csv')



def enrich_compute(geodf, var_id, downsample):

    """
    Compute a calculated variable for the provided bounds and save into local netcdf file.
    Calculate and return indices of the data of interest in the ncdf file.

    Args:
        geodf (geopandas.GeoDataFrame): Data to be enriched.
        var_id (str): ID of the variable to download.
        downsample (dict): Number of points to skip between each downloaded point, for each dimension, using its standard name as a key.
    Returns:
        pandas.DataFrame: DataFrame with indices of relevant data in the netCDF file.

    """

    # Check if local netcdf files already exist

    if  not(os.path.exists(sat_path + var_id + '.nc')) or \
        not(os.path.exists(sat_path + var_id + '_downloaded.nc')):

        create_nc_calculated(var_id)

    # Backup local netCDF files

    timestamp = datetime.now().strftime('%d-%H-%M')
    shutil.copy2(sat_path + var_id + '.nc', sat_path + var_id + '.nc.' + timestamp)
    shutil.copy2(sat_path + var_id + '_downloaded.nc', sat_path + var_id + '_downloaded.nc.' + timestamp)

    # Load files

    local_ds = nc.Dataset(sat_path + var_id + '.nc.' + timestamp, mode ='r+')
    bool_ds = nc.Dataset(sat_path + var_id + '_downloaded.nc.' + timestamp, mode ='r+')

    dimdict, var = get_metadata(local_ds, var_id)

    # Remove out of timeframe datapoints

    if 'time' in dimdict:
        t1, t2 = min(dimdict['time']['vals']), max(dimdict['time']['vals'])
        geodf2 = geodf[(geodf['mint'] >= t1) & (geodf['maxt'] <= t2)]
        print('Ignoring {} rows because data is not available at these dates'.format(len(geodf) - len(geodf2)))
    else:
        geodf2 = geodf

    # Open needed datasets (read-only)

    base_datasets = {}
    cat = get_var_catalog()

    for sec_var_id in var['derived_from']:
        base_datasets[sec_var_id] = {}
        base_datasets[sec_var_id]['ds'] = nc.Dataset(sat_path + sec_var_id + '.nc')
        base_datasets[sec_var_id]['bool_ds'] = nc.Dataset(sat_path + sec_var_id + '_downloaded.nc')
        base_datasets[sec_var_id]['varname'] = cat[sec_var_id]['varname']

    # Apply query to each row sequentially

    res = geodf2.progress_apply(row_compute, axis=1, args = (local_ds, bool_ds, base_datasets,
                                                             dimdict, var, downsample), 
                                result_type = 'expand')

    local_ds.close()
    bool_ds.close()

    for key in base_datasets:
        base_datasets[key]['ds'].close()
        base_datasets[key]['bool_ds'].close()

    # Remove backup

    os.remove(sat_path + var_id + '.nc')
    os.remove(sat_path + var_id + '_downloaded.nc')

    os.rename(sat_path + var_id + '.nc.' + timestamp, sat_path + var_id + '.nc')
    os.rename(sat_path + var_id + '_downloaded.nc.' + timestamp, sat_path + var_id + '_downloaded.nc')

    return(res)



def enrich_download(geodf, varname, var_id, url, depth_request, downsample):
    
    """
    Download data for the requested occurrences and buffer into local netcdf file.
    Calculate and return indices of the data of interest in the ncdf file.

    Args:
        geodf (geopandas.GeoDataFrame): Data to be enriched.
        varname(str): Variable name in the dataset.
        var_id (str): ID of the variable to download.
        url (str): Dataset url (including credentials if needed).
        depth_request (str): For 4D data: 'surface' only download surface data. Anything else downloads everything.
        downsample (dict): Number of points to skip between each downloaded point, for each dimension, using its standard name as a key.
    Returns:
        pandas.DataFrame: DataFrame with indices of relevant data in the netCDF file.

    """

    # Get netcdf metadata

    remote_ds = nc.Dataset(url)

    dimdict, var = get_metadata(remote_ds, varname)
    var['var_id'] = var_id

    # Check if local netcdf files already exist

    if  not(os.path.exists(sat_path + var['var_id'] + '.nc')) or \
        not(os.path.exists(sat_path + var['var_id'] + '_downloaded.nc')):

        create_nc(get_var_catalog()[var_id])

    # Backup local netCDF files

    timestamp = datetime.now().strftime('%d-%H-%M')
    shutil.copy2(sat_path + var['var_id'] + '.nc', sat_path + var['var_id'] + '.nc.' + timestamp)
    shutil.copy2(sat_path + var['var_id'] + '_downloaded.nc', sat_path + var['var_id'] + '_downloaded.nc.' + timestamp)

    # Load files

    local_ds = nc.Dataset(sat_path + var['var_id'] + '.nc.' + timestamp, mode ='r+')
    bool_ds = nc.Dataset(sat_path + var['var_id'] + '_downloaded.nc.' + timestamp, mode ='r+')

    # Remove out of timeframe datapoints

    if 'time' in dimdict:
        t1, t2 = min(dimdict['time']['vals']), max(dimdict['time']['vals'])
        geodf2 = geodf[(geodf['mint'] >= t1) & (geodf['maxt'] <= t2)]
        print('Ignoring {} rows because data is not available at these dates'.format(len(geodf) - len(geodf2)))
    else:
        geodf2 = geodf

    # Apply query to each row sequentially

    res = geodf2.progress_apply(row_enrich, axis=1, args = (remote_ds, local_ds, bool_ds, dimdict, var, depth_request, downsample), 
                            result_type = 'expand')

    local_ds.close()
    bool_ds.close()
    remote_ds.close()


    # Remove backup

    os.remove(sat_path + var['var_id'] + '.nc')
    os.remove(sat_path + var['var_id'] + '_downloaded.nc')

    os.rename(sat_path + var['var_id'] + '.nc.' + timestamp, sat_path + var['var_id'] + '.nc')
    os.rename(sat_path + var['var_id'] + '_downloaded.nc.' + timestamp, sat_path + var['var_id'] + '_downloaded.nc')

    print('Enrichment over')
    return(res)



def add_bounds(geodf1, geo_buff, time_buff):

    """
    Calculate geo buffer and time buffer.
    Add columns for cube limits: 'minx', 'maxx', 'miny', 'maxy', 'mint', 'maxt'.

    Args:
        geodf1 (geopandas.GeoDataFrame): Data to calculate buffers for.
        geo_buf (int): Geographic buffer for which to download data around occurrence point (kilometers).
        time_buff (float tuple): Time bounds for which to download data around occurrence day (days). For instance, time_buff = (-7, 0) will download data from 7 days before the occurrence to the occurrence date.
    Returns:
        geopandas.GeoDataFrame: Updated GeoDataFrame with geographical and time boundaries.
    """

    # Prepare geo bounds
    
    earth_radius = 6371
    lat_buf = 180 * geo_buff / (np.pi * earth_radius)
    geodf = geodf1.loc[abs(geodf1['geometry'].y) + lat_buf < 90]

    if len(geodf1) != len(geodf):
        print('Warning: {} occurrences were dropped because a pole is inside the buffer (too much data to download)'.format(len(geodf1) - len(geodf)))

    # Calculate pseudo square in degrees that contains the square buffers in kilometers
    latitudes = geodf['geometry'].y
    min_radius = np.sin(np.pi * (90 - abs(latitudes) - lat_buf) / 180)
    lon_buf = 180 * geo_buff / (np.pi * earth_radius * min_radius)

    geodf['minx'] = (geodf['geometry'].x - lon_buf + 180) % 360 - 180
    geodf['maxx'] = (geodf['geometry'].x + lon_buf + 180) % 360 - 180
    geodf['miny'] = (geodf['geometry'].y - lat_buf)
    geodf['maxy'] = (geodf['geometry'].y + lat_buf)


    # Prepare time bounds

    buff1 = np.timedelta64(time_buff[0], 'D')
    buff2 = np.timedelta64(time_buff[1], 'D')

    geodf['mint'] = pd.to_datetime(geodf['eventDate'] + buff1)
    geodf['bestt'] = pd.to_datetime(geodf['eventDate'])
    geodf['maxt'] = pd.to_datetime(geodf['eventDate'] + buff2)

    return(geodf)



def enrich_areas(df, var_id, outfile, depth_request = 'surface', slice = None, downsample = {}):

    """
    Download data for arbitrary areas specified in df.
    The downsample argument can be used to skip part of the data in the case of large areas.

    Args:
        df (pandas.DataFrame)
        var_id (str): ID of the variable to download.
        outfile (str): path where the output file will be written.
        depth_request (str): Used when depth is a dimension. 'surface' only downloads surface data. Anything else downloads everything.
        slice (int tuple): Slice of the enrichment file to use for enrichment.
        downsample (dict): Number of points to skip between each downloaded point, for each dimension, using its standard name as a key.
    
    Returns:
        None
    """

    var = get_var_catalog()[var_id]

    if slice is None:
        to_enrich = df
    else:
        to_enrich = df.iloc[slice[0]:slice[1]]

    if var['url'] == 'calculated':
        indices = enrich_compute(to_enrich, var['var_id'], downsample)
    else:
        indices = enrich_download(to_enrich, var['varname'], var['var_id'], var['url'], depth_request, downsample)

    var_ind = parse_columns(indices)

    ds = nc.Dataset(sat_path + var_id + '.nc')
    dimdict, var = get_metadata(ds, var['varname'])

    res = indices.progress_apply(compute_stats, axis=1, args = (var_id, var_ind[var_id], ds, dimdict, var, downsample), result_type = 'expand')
    ds.close()

    output = df.merge(res, how = 'left', left_index = True, right_index = True)


    output.to_csv(outfile)
    print('File saved at ' + outfile)


############################# Element-wise enrichment #################################


def row_enrich(row, remote_ds, local_ds, bool_ds, dimdict, var, depth_request, downsample):

    """
    Query geospatial data for the given GeoDataFrame row.
    Save netCDF data to disk and return their coordinates.

    Args:
        row (pandas.Series): GeoDataFrame row to enrich.
        remote_ds (netCDF4.Dataset): Remote dataset.
        local_ds (netCDF4.Dataset): Local dataset.
        bool_ds (netCDF4.Dataset): Local dataset recording whether data has already been downloaded.
        dimdict (dict): Dictionary of dimensions as returned by geoenrich.satellite.get_metadata.
        var (dict): Variable dictionary as returned by geoenrich.satellite.get_metadata.
        depth_request (str): For 4D data: 'surface' only download surface data. Anything else downloads everything.
        downsample (dict): Number of points to skip between each downloaded point, for each dimension, using its standard name as a key.
    Returns:
        pandas.Series: Coordinates of the data of interest in the netCDF file.

    """

    # Find indices for region of interest

    ind = calculate_indices(dimdict, row, var, depth_request, downsample)
    params = [dimdict[n]['standard_name'] for n in var['params']]
    ordered_indices = [ind[p] for p in params]
    
    download_data(remote_ds, local_ds, bool_ds, var, dimdict, ind)

    # Return coordinates of the saved subset for data retrieval

    colnames = []
    coords = []

    for p in params:

        if 'best' in ind[p]:
            colnames.extend([var['var_id'] + '_' + dimdict[p]['standard_name'] + '_min',
                             var['var_id'] + '_' + dimdict[p]['standard_name'] + '_best',
                             var['var_id'] + '_' + dimdict[p]['standard_name'] + '_max'])
            coords.extend([ind[p]['min'], ind[p]['best'], ind[p]['max']])
        else:
            colnames.extend([var['var_id'] + '_' + dimdict[p]['standard_name'] + '_min',
                             var['var_id'] + '_' + dimdict[p]['standard_name'] + '_max'])
            coords.extend([ind[p]['min'], ind[p]['max']])

    return(pd.Series(coords, index = colnames))




def row_compute(row, local_ds, bool_ds, base_datasets, dimdict, var, downsample):

    """
    Calculate variable for the given row.
    Save netCDF data to disk and return their coordinates.

    Args:
        row (pandas.Series): GeoDataFrame row to enrich.
        local_ds (netCDF4.Dataset): Local dataset.
        bool_ds (netCDF4.Dataset): Local dataset recording whether data has already been downloaded.
        base_datasets (netCDF4.Dataset dict): Required datasets for the computation.
        dimdict (dict): Dictionary of dimensions as returned by geoenrich.satellite.get_metadata.
        var (dict): Variable dictionary as returned by geoenrich.satellite.get_metadata.
        downsample (dict): Number of points to skip between each downloaded point, for each dimension, using its standard name as a key.
    Returns:
        pandas.Series: Coordinates of the data of interest in the netCDF file.

    """

    ind = calculate_indices(dimdict, row, var, 'surface', downsample)
    params = [dimdict[n]['standard_name'] for n in var['params']]
    ordered_indices = [ind[p] for p in params]
    lons = dimdict['longitude']['vals']
    lon_pos = var['params'].index(dimdict['longitude']['name'])
    totalsize = np.prod([1 + (ind[p]['max'] - ind[p]['min']) // ind[p]['step'] for p in params])

    if ind['longitude']['min'] > ind['longitude']['max']:
        # Handle longitude singularity
        totalsize = totalsize / (1 + (ind['longitude']['max'] - ind['longitude']['min']) // ind['longitude']['step'])
        act_len = (len(lons) - ind['longitude']['min']) // ind['longitude']['step'] + \
                  (ind['longitude']['max'] - len(lons) % ind['longitude']['step'] + 1) // ind['longitude']['step']
        totalsize = totalsize * act_len

    base_data = {}

    # Check that required variables were already downloaded.

    for key in base_datasets:
        name = base_datasets[key]['varname']
        check = multidimensional_slice(base_datasets[key]['bool_ds'], name, ordered_indices, lons, lon_pos).data
        if check.sum() != totalsize:
            raise LookupError('Data was not fully downloaded for required variable ' + key)

        base_data[key] = multidimensional_slice(base_datasets[key]['ds'], name, ordered_indices, lons, lon_pos)

    # Calculate and save variable

    result = compute_variable(var['name'], base_data)

    insert_multidimensional_slice(local_ds, var['name'], result, ordered_indices, lons, lon_pos)
    insert_multidimensional_slice(bool_ds, var['name'], np.ones(result.shape), ordered_indices, lons, lon_pos)

    # Return coordinates of the saved subset for data retrieval

    colnames = []
    coords = []

    for p in params:

        if 'best' in ind[p]:
            colnames.extend([var['name'] + '_' + dimdict[p]['standard_name'] + '_min',
                             var['name'] + '_' + dimdict[p]['standard_name'] + '_best',
                             var['name'] + '_' + dimdict[p]['standard_name'] + '_max'])
            coords.extend([ind[p]['min'], ind[p]['best'], ind[p]['max']])
        else:
            colnames.extend([var['name'] + '_' + dimdict[p]['standard_name'] + '_min',
                             var['name'] + '_' + dimdict[p]['standard_name'] + '_max'])
            coords.extend([ind[p]['min'], ind[p]['max']])

    return(pd.Series(coords, index = colnames))



def calculate_indices(dimdict, row, var, depth_request, downsample):

    """
    Calculate indices of interest for the given bounds, according to variable dimensions.
    
    Args:
        dimdict (dict): Dictionary of dimensions as returned by geoenrich.satellite.get_metadata.
        row (pandas.Series): GeoDataFrame row to enrich.
        var (dict): Variable dictionary as returned by geoenrich.satellite.get_metadata.
        depth_request (str): For 4D data: 'surface' only download surface data. Anything else downloads everything.
        downsample (dict): Number of points to skip between each downloaded point, for each dimension, using its standard name as a key.
    Returns:
        dict: Dictionary of indices for each dimension (keys are standard dimension names).
    """

    ind = {}

    # latitude lower, best and upper index
    # make sure the slice contains at least one element

    lat0 = np.argmin( np.abs( dimdict['latitude']['vals'] - row['miny'] ) )
    lat2 = np.argmin( np.abs( dimdict['latitude']['vals'] - row['maxy'] ) )
    ind['latitude'] = {'min': min(lat0, lat2), 'max': max(lat0, lat2), 'step': 1}

    # longitude lower, best and upper index
    # make sure the slice contains at least one element

    lon0 = np.argmin( np.abs( dimdict['longitude']['vals'] - row['minx'] ) )
    lon2 = np.argmin( np.abs( dimdict['longitude']['vals']  - row['maxx'] ) )  
    ind['longitude'] = {'min': lon0, 'max': lon2, 'step': 1}

    # Add best match indices if centered on an occurrence
    if 'geometry' in row:
        lat1 = np.argmin( np.abs( dimdict['latitude']['vals'] - row['geometry'].y ) )
        lon1 = np.argmin( np.abs( dimdict['longitude']['vals'] - row['geometry'].x ) )
        ind['latitude']['best'] = lat1
        ind['longitude']['best'] = lon1

    params = [dimdict[n]['standard_name'] for n in var['params']]

    # if time in dimensions, get lower, upper, and best fit indices
    # make sure the slice contains at least one element

    if ('time' in dimdict) and (dimdict['time']['name'] in params):


        t0 = np.argmin( np.abs( dimdict['time']['vals'] - row['mint'] ) )
        t2 = np.argmin( np.abs( dimdict['time']['vals'] - row['maxt'] ) ) 
        ind['time'] = {'min': min(t0, t2), 'max': max(t0, t2), 'step': 1}

        if 'bestt' in row:
            t1 = np.argmin( np.abs( dimdict['time']['vals'] - row['bestt'] ) )
            ind['time']['best'] = t1

    # if depth is a dimension, either select surface layer or return everything

    if ('depth' in dimdict) and (dimdict['depth']['name'] in params):
        if depth_request == 'surface':
            d1 = np.argmin( np.abs( dimdict['depth']['vals'] ) )
            ind['depth'] = {'min': d1, 'max': d1, 'best': d1, 'step': 1}
        else:
            ind['depth'] = {'min': 0, 'max': len(dimdict['depth']['vals'] - 1), 'best': None, 'step': 1}

    for dim in downsample:
        ind[dim]['step'] = downsample[dim] + 1

    return(ind)



def download_data(remote_ds, local_ds, bool_ds, var, dimdict, ind):

    """
    Download missing data from the remote dataset to the local dataset.

    Args:
        remote_ds (netCDF4.Dataset): Remote dataset.
        local_ds (netCDF4.Dataset): Local dataset.
        bool_ds (netCDF4.Dataset): Local dataset recording whether data has already been downloaded.
        var (dict): Variable dictionary as returned by geoenrich.satellite.get_metadata.
        dimdict (dict): Dictionary of dimensions as returned by geoenrich.satellite.get_metadata.
        ind (dict): Dictionary with ordered slicing indices for all dimensions.
    Returns:
        None
    """

    params = [dimdict[n]['standard_name'] for n in var['params']]
    ordered_indices = [ind[p] for p in params]
    lons = dimdict['longitude']['vals']
    lon_pos = var['params'].index(dimdict['longitude']['name'])
    check = multidimensional_slice(bool_ds, var['name'], ordered_indices, lons, lon_pos).data
    totalsize = np.prod([1 + (ind[p]['max'] - ind[p]['min']) // ind[p]['step'] for p in params])

    if ind['longitude']['min'] > ind['longitude']['max']:
        # Handle longitude singularity
        totalsize = totalsize / (1 + (ind['longitude']['max'] - ind['longitude']['min']) // ind['longitude']['step'])
        act_len = (len(lons) - ind['longitude']['min']) // ind['longitude']['step'] + \
                  (ind['longitude']['max'] - len(lons) % ind['longitude']['step'] + 1) // ind['longitude']['step']
        totalsize = totalsize * act_len
        

    if 'time' in ind:

        lentime = 1 + (ind['time']['max'] - ind['time']['min']) // ind['time']['step']
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
                new_ind['time']['min'] = ind['time']['min'] + start * ind['time']['step']
                new_ind['time']['max'] = ind['time']['min'] + (i - 1) * ind['time']['step']
                download_data(remote_ds, local_ds, bool_ds, var, dimdict, new_ind)
                

        if(started):
            new_ind = deepcopy(ind)
            new_ind['time']['min'] = ind['time']['min'] + start * ind['time']['step']
            new_ind['time']['max'] = ind['time']['min'] + (lentime - 1) * ind['time']['step']
            download_data(remote_ds, local_ds, bool_ds, var, dimdict, new_ind)
            

    # Otherwise download everything

    else:
        lons = dimdict['longitude']['vals']
        lon_pos = var['params'].index(dimdict['longitude']['name'])
        data = multidimensional_slice(remote_ds, var['name'], ordered_indices, lons, lon_pos)
        insert_multidimensional_slice(local_ds, var['name'], data, ordered_indices, lons, lon_pos)
        insert_multidimensional_slice(bool_ds, var['name'], np.ones(data.shape), ordered_indices, lons, lon_pos)



def compute_variable(var_id, base_data):

    """
    Calculate a composite variable.

    Args:
        var_id (str): ID of the variable to compute.
        base_data (numpy.ma.MaskedArray dict): Required data for the computation.
    Returns:
        numpy.ma.MaskedArray: Output data.
    """

    if var_id == 'eke':

        result = base_data['geos-current-u']**2 + base_data['geos-current-v']**2

    else:

        raise NotImplementedError('Calculation of this variable is not implemented')
        
    return(result)



##########################################################################
###########                 Enrichment files                ##############
##########################################################################



def load_enrichment_file(dataset_ref):

    """
    Load enrichment file.

    Args:
        dataset_ref (str): The enrichment file name (e.g. gbif_taxonKey).
    Returns:
        geopandas.GeoDataFrame: Data to enrich (including previously added columns).
    """

    filepath = biodiv_path + dataset_ref + '.csv'

    df = pd.read_csv(filepath, parse_dates = ['eventDate'], infer_datetime_format = True, index_col = 0)
    df['geometry'] = df['geometry'].apply(wkt.loads)

    print('{} occurrences were loaded from enrichment file'.format(len(df)))
    return(gpd.GeoDataFrame(df, crs = 'epsg:4326'))



def create_enrichment_file(gdf, dataset_ref, id_prefix = ''):

    """
    Create database file that will be used to save enrichment data.
    
    Args:  
        gdf (geopandas.GeoDataFrame): Data to enrich (output of :func:`geoenrich.Biodiv.open_dwca` or :func:`geoenrich.Biodiv.import_csv`).
        dataset_ref (str): The enrichment file name (e.g. gbif_taxonKey).
        id_prefix (str): Optional, if you want to add a prefix to occurrence ids.
    Returns:
        None
    """

    filepath = biodiv_path + dataset_ref + '.csv'

    if(os.path.exists(filepath)):
        print('Abort. File already exists at ' + filepath)
    else:
        to_save = gdf[['id', 'eventDate', 'geometry']]
        to_save.set_index(pd.Index(id_prefix + to_save['id'].astype(str), name='id'), inplace = True)
        to_save.drop(['id'], axis='columns', inplace = True)
        to_save = to_save.reindex(columns=['geometry', 'eventDate'])
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
    to_save = df[['eventDate', 'geometry']]
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

    """
    Return column indices sorted by variable and dimension.

    Args:
        df (pandas.DataFrame): Enrichment file as a DataFrame, as returned by geoenrich.enrichment.load_enrichment_file.
    Returns:
        dict: Dictionary of column indices, with variable as a primary key, dimension as a secondary key, and min/max as tertiary key.
    """

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




def retrieve_data(occ_id, dataset_ref = None, path = None, id_col = 0, shape = 'rectangle', geo_buff = None, downsample = {}):

    """
    Retrieve all available data for a specific occurrence or area.
    Use dataset_ref if enriching occurrences, and path if enriching arbitrary areas.
    geo_buff and downsample must be identical to the values you used for enrichment.
    
    Args:
        occ_id (str): ID of the occurrence to get data for. Can be obtained with :func:`geoenrich.enrichment.read_ids`.
        dataset_ref (str): The enrichment file name (e.g. gbif_taxonKey).
        path (str): Path to the areas file that was enriched.
        id_col (str or int): Index or name of the ID column.
        shape (str): If 'rectangle', return data inside the rectangle containing the buffer. If 'buffer', only return data within the buffer distance from the occurrence location.
        geo_buffer (int): Ther buffer you used to enrich your dataset (or a smaller one).
        downsample (dict): Number of points to skip between each downloaded point, for each dimension, using its standard name as a key.

    Returns:
        dict: A dictionary of all available variables with corresponding data (numpy.ma.MaskedArray), unit (str), and coordinates (ordered list of dimension names and values).
    """

    if dataset_ref is not None:
        path = biodiv_path + dataset_ref + '.csv'
    else:
        shape = 'rectangle'

    df = pd.read_csv(path, index_col = id_col)

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

            data, coords = fetch_data(row, v, var_ind, ds, dimdict, var, downsample)
            ds.close()

        if shape == 'buffer' and geo_buff is not None:
            mask = ellipsoid_mask(data, coords, row['geometry'], geo_buff)
            results[v] = {'coords': coords, 'values': np.ma.masked_where(mask, data), 'unit': unit}
        else:
            results[v] = {'coords': coords, 'values': data, 'unit': unit}


    return(results)



def fetch_data(row, var_id, var_indices, ds, dimdict, var, downsample):

    """
    Fetch data locally for a specific occurrence and variable.
    
    Args:
        row (pandas.Series): One row of an enrichment file.
        var_id (str): ID of the variable to download.
        var_indices (dict):  Dictionary of column indices for the selected variable, output of :func:`geoenrich.enrichment.parse_columns`.
        ds (netCDF4.Dataset): Local dataset.
        dimdict (dict): Dictionary of dimensions as returned by geoenrich.satellite.get_metadata.
        var (dict): Variable dictionary as returned by geoenrich.satellite.get_metadata.
        downsample (dict): Number of points to skip between each downloaded point, for each dimension, using its standard name as a key.

    Returns:
        numpy.ma.MaskedArray: Raw data.
    """


    params = [dimdict[n]['standard_name'] for n in var['params']]
    ordered_indices_cols = [var_indices[p] for p in params]
    ordered_indices = [{'min': int(row.iloc[d['min']]),
                        'max': int(row.iloc[d['max']]),
                        'step': 1}
                        for d in ordered_indices_cols]

    for i in range(len(params)):
        p = params[i]
        if p in downsample:
            ordered_indices[i]['step'] = downsample[p] + 1

    lons = dimdict['longitude']['vals']
    lon_pos = var['params'].index(dimdict['longitude']['name'])
    data = multidimensional_slice(ds, var['name'], ordered_indices, lons, lon_pos)

    coordinates = []
    for p in params:
        i1, i2 = int(row.iloc[var_indices[p]['min']]), int(row.iloc[var_indices[p]['max']])
        if p in downsample:
            step = downsample[p] + 1
        else:
            step = 1
        if (p == 'longitude' and i1 > i2):
            part1 = ds.variables[dimdict[p]['name']][i1::step]
            part2 = ds.variables[dimdict[p]['name']][len(lons)%step:i2 + 1:step]
            coordinates.append([p, np.ma.concatenate((part1, part2))])
        elif p == 'time':
            time_var = ds.variables[dimdict[p]['name']]
            if 'months since' in time_var.__dict__['units']:
                times = num2date(time_var[i1:i2+1:step], time_var.__dict__['units'], '360_day')
            else:
                times = num2pydate(time_var[i1:i2+1:step], time_var.__dict__['units'])
            coordinates.append([p, times])
        else:
            coordinates.append([p, ds.variables[dimdict[p]['name']][i1:i2+1:step]])

    return(data, coordinates)



def read_ids(dataset_ref = None, filepath = None, id_col = 0):

    """
    Return a list of all ids of the given dataset.
    Use dataset_ref if enriching occurrences, and path if enriching arbitrary areas.
    
    Args:
        dataset_ref (str): The enrichment file name (e.g. gbif_taxonKey).
        filepath (str): Path to the areas file that was enriched.
        id_col (str or int): Index or name of the ID column.
    Returns:
        list: List of all present ids.
    """

    if dataset_ref is not None:
        filepath = biodiv_path + dataset_ref + '.csv'

    df = pd.read_csv(filepath, index_col = id_col)

    return(list(df.index))




def produce_stats(dataset_ref, geo_buff, var_list = None, downsample = {}):

    """
    Produce a document named *dataset\_ref*\_stats.csv with summary stats of all enriched data.

    Args:
        dataset_ref (str): The enrichment file name (e.g. gbif_taxonKey).
        geo_buffer (int): The buffer you used to enrich your dataset (or a smaller one).
        var_list (var): A sublist of enriched variable to compute statistics for. If None, use all available variables.
        downsample (dict): Number of points to skip between each downloaded point, for each dimension, using its standard name as a key.


    Returns:
        None
    """

    filepath = biodiv_path + dataset_ref + '.csv'
    df = pd.read_csv(filepath, parse_dates = ['eventDate'], infer_datetime_format = True, index_col = 0)
    df['geometry'] = df['geometry'].apply(wkt.loads)
    output = df[['taxonKey', 'geometry', 'eventDate']]
    cat = get_var_catalog()
    ind = parse_columns(df)

    for v in ind:

        if var_list is None or v in var_list:
            var_ind = ind[v]
            ds = nc.Dataset(sat_path + v + '.nc')
            dimdict, var = get_metadata(ds, cat[v]['varname'])

            print('Computing stats for ' + v + '...')
            res = df.progress_apply(compute_stats, axis=1, args = (v, var_ind, ds, dimdict, var, downsample, geo_buff), result_type = 'expand')
            ds.close()

            if res is not None:
                output = output.merge(res, how = 'left', left_index = True, right_index = True)


    output.to_csv(biodiv_path + dataset_ref + '_stats.csv')
    print('File saved at ' + biodiv_path + dataset_ref + '_stats.csv')



def compute_stats(row, var_id, var_indices, ds, dimdict, var, downsample, geo_buff = None):

    """
    Compute and return stats for the given row.

    
    Args:
        row (pandas.Series): One row of an enrichment file.
        var_id (str): ID of the variable to download.
        var_indices (dict):  Dictionary of column indices for the selected variable, output of :func:`geoenrich.enrichment.parse_columns`.
        ds (netCDF4.Dataset): Local dataset.
        dimdict (dict): Dictionary of dimensions as returned by geoenrich.satellite.get_metadata.
        var (dict): Variable dictionary as returned by geoenrich.satellite.get_metadata.
        downsample (dict): Number of points to skip between each downloaded point, for each dimension, using its standard name as a key.
    
    Returns:
        pandas.Series: Statistics for the given row.
    """

    min_columns = [row.iloc[d['min']] for d in var_indices.values()]

    if -1 in min_columns or any([np.isnan(c) for c in min_columns]):
        # Missing data
        names = [var_id + '_av', var_id + '_std', var_id + '_min', var_id + '_max', var_id + '_count']
        return(pd.Series(index=names, dtype = float))

    data, coords = fetch_data(row, var_id, var_indices, ds, dimdict, var, downsample)

    params = [dimdict[n]['standard_name'] for n in var['params']]
    ordered_indices_cols = [var_indices[p] for p in params]

    if geo_buff is not None:

        # If data was calulated around an occurrence

        mask = ellipsoid_mask(data, coords, row['geometry'], geo_buff)
        data = np.ma.masked_where(mask, data)

        av, std = np.ma.average(data), np.ma.std(data)
        minv, maxv, count = np.ma.min(data), np.ma.max(data), np.ma.count(data)

        names = [var_id + '_av', var_id + '_std', var_id + '_min', var_id + '_max', var_id + '_count']

        ret = pd.Series([av, std, minv, maxv, count], index = names)

    else:

        # If there is no occurrence

        av, std = np.ma.average(data), np.ma.std(data)
        minv, maxv, count = np.ma.min(data), np.ma.max(data), np.ma.count(data)
        names = [var_id + '_av', var_id + '_std', var_id + '_min', var_id + '_max', var_id + '_count']
        ret = pd.Series([av, std, minv, maxv, count], index = names)

    return(ret)



def get_derivative(occ_id, var_id, days = (0,0), dataset_ref = None, path = None, id_col = 0, shape = 'rectangle', geo_buff = None, downsample = {}):

    """
    Retrieve data for both specified days and return the derivative.
    Use dataset_ref if enriching occurrences, and path if enriching arbitrary areas.
    geo_buff and downsample must be identical to the values you used for enrichment.
    
    Args:
        occ_id (str): ID of the occurrence to get data for. Can be obtained with :func:`geoenrich.enrichment.read_ids`.
        var_id (str): ID of the variable to derivate.
        days (int tuple): Start and end days for derivative calculation, relatively to occurrence, eg. (-7, 0)
        dataset_ref (str): The enrichment file name (e.g. gbif_taxonKey).
        path (str): Path to the areas file that was enriched.
        id_col (str or int): Index or name of the ID column.
        shape (str): If 'rectangle', return data inside the rectangle containing the buffer. If 'buffer', only return data within the buffer distance from the occurrence location.
        geo_buffer (int): Ther buffer you used to enrich your dataset (or a smaller one).
        downsample (dict): Number of points to skip between each downloaded point, for each dimension, using its standard name as a key.

    Returns:
        dict: A dictionary of all available variables with corresponding data (numpy.ma.MaskedArray), unit (str), and coordinates (ordered list of dimension names and values).
    """

    if dataset_ref is not None:
        path = biodiv_path + dataset_ref + '.csv'
    else:
        shape = 'rectangle'

    df = pd.read_csv(path, index_col = id_col)

    row = df.loc[occ_id]
    cat = get_var_catalog()
    
    var_ind = parse_columns(df)[var_id]

    # Read indices into a dictionary

    results = {}

        if -1 in [row.iloc[d['min']] for d in var_ind.values()]:
            results[v] = {'coords': None, 'values': None}

        else:
            ds = nc.Dataset(sat_path + v + '.nc')
            unit = getattr(ds.variables[cat[v]['varname']], 'units', 'Unspecified')

            dimdict, var = get_metadata(ds, cat[v]['varname'])

            data, coords = fetch_data(row, v, var_ind, ds, dimdict, var, downsample)
            ds.close()

        if shape == 'buffer' and geo_buff is not None:
            mask = ellipsoid_mask(data, coords, row['geometry'], geo_buff)
            results[v] = {'coords': coords, 'values': np.ma.masked_where(mask, data), 'unit': unit}
        else:
            results[v] = {'coords': coords, 'values': data, 'unit': unit}


    return(results)