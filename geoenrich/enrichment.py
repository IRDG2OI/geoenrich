"""
This is the main module of the package.
It handles the local enrichment files, as well as the download of enrichment data from remote servers.
"""

import os
import shutil

import json
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



def enrich(dataset_ref, var_id, geo_buff = None, time_buff = None, depth_request = 'surface', 
    downsample = {}, slice = None):

    """
    Enrich the given dataset with data of the requested variable.
    All Data within the given buffers are downloaded (if needed) and stored locally in netCDF files.
    The enrichment file is updated with the coordinates of the relevant netCDF subsets.
    If the enrichment file is large, use slice argument to only enrich some rows.
    
    Args:
        dataset_ref (str): The enrichment file name (e.g. gbif_taxonKey). Must be unique.
        var_id (str): ID of the variable to download.
        geo_buff (int): Geographic buffer for which to download data around occurrence point (kilometers).
        time_buff (float list): Time bounds for which to download data around occurrence day (days). For instance, time_buff = [-7, 0] will download data from 7 days before the occurrence to the occurrence date.
        depth_request (str): Used when depth is a dimension. 'surface' only downloads surface data. Anything else downloads everything.
        downsample (dict): Number of points to skip between each downloaded point, for each dimension, using its standard name as a key.
        slice (int tuple): Slice of the enrichment file to use for enrichment.

    Returns:
        None
    """

    try:
        f = open(biodiv_path + dataset_ref + '-config.json')
        enrichment_metadata = json.load(f)
        f.close()
    except FileNotFoundError:
        print('Please create an enrichment file first (use geoenrich.enrichmet.create_enrichment_file)')

    input_type = enrichment_metadata['input_type']
    enrichments = enrichment_metadata['enrichments']

    enrichment_id = get_enrichment_id(enrichments, var_id, geo_buff, time_buff, depth_request, downsample)
    new_enrichment = False
    if enrichment_id == -1:
        new_enrichment = True
        enrichment_id = len(enrichments)

    original = load_enrichment_file(dataset_ref)

    if slice is None:
        to_enrich = original
    else:
        to_enrich = original.iloc[slice[0]:slice[1]]

    # Load variable information
    var_source = get_var_catalog()[var_id]
    

    if var_source['url'] == 'calculated':
        indices = enrich_compute(to_enrich, var_id, geo_buff, time_buff, downsample)
    else:
        indices = enrich_download(to_enrich, var_source['varname'], var_id, var_source['url'],
                                    geo_buff, time_buff, depth_request, downsample)

    prefix = str(enrichment_id) + '_'
    indices = indices.add_prefix(prefix)

    # If variable is already present, update it

    if not(new_enrichment) and len(indices):
        original.update(indices)
        updated = original

    # If indices is empty
    elif not(len(indices)):
        updated = original

    # Else add new columns
    else:
        updated = original.merge(indices, how = 'left', left_index = True, right_index = True)

    # Fill unenriched rows with -1
    missing_index = to_enrich.index.difference(indices.index)
    updated.loc[missing_index,indices.columns] = -1

    # Save file
    if new_enrichment:
        save_enrichment_config(dataset_ref, enrichment_id, var_id, geo_buff, time_buff, depth_request, downsample)
    updated.to_csv(biodiv_path + dataset_ref + '.csv')



def enrich_compute(geodf, var_id, geo_buff, time_buff, downsample):

    """
    Compute a calculated variable for the provided bounds and save into local netcdf file.
    Calculate and return indices of the data of interest in the ncdf file.

    Args:
        geodf (geopandas.GeoDataFrame): Data to be enriched.
        var_id (str): ID of the variable to download.
        geo_buff (int): Geographic buffer for which to download data around occurrence point (kilometers).
        time_buff (float list): Time bounds for which to download data around occurrence day (days). For instance, time_buff = [-7, 0] will download data from 7 days before the occurrence to the occurrence date.
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

    # Add bounds if occurrences

    if 'minx' not in geodf.columns:
        if geo_buff is None or (time_buff is None and 'time' in dimdict):
            raise BufferError('Please specify time_buff and geo_buff.')
        geodf = add_bounds(geodf, geo_buff, time_buff)

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



def enrich_download(geodf, varname, var_id, url, geo_buff, time_buff, depth_request, downsample):
    
    """
    Download data for the requested occurrences and buffer into local netcdf file.
    Calculate and return indices of the data of interest in the ncdf file.

    Args:
        geodf (geopandas.GeoDataFrame): Data to be enriched.
        varname(str): Variable name in the dataset.
        var_id (str): ID of the variable to download.
        url (str): Dataset url (including credentials if needed).
        geo_buff (int): Geographic buffer for which to download data around occurrence point (kilometers).
        time_buff (float list): Time bounds for which to download data around occurrence day (days). For instance, time_buff = [-7, 0] will download data from 7 days before the occurrence to the occurrence date.
        depth_request (str): For 4D data: 'surface' only download surface data. Anything else downloads everything.
        downsample (dict): Number of points to skip between each downloaded point, for each dimension, using its standard name as a key.

    Returns:
        pandas.DataFrame: DataFrame with indices of relevant data in the netCDF file.

    """

    # Get netcdf metadata

    remote_ds = nc.Dataset(url)

    dimdict, var = get_metadata(remote_ds, varname)
    var['var_id'] = var_id

    # Add bounds if occurrences

    if 'minx' not in geodf.columns:
        if geo_buff is None or (time_buff is None and 'time' in dimdict):
            raise BufferError('Please specify time_buff and geo_buff.')
        geodf = add_bounds(geodf, geo_buff, time_buff)


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
        geo_buff (int): Geographic buffer for which to download data around occurrence point (kilometers).
        time_buff (float list): Time bounds for which to download data around occurrence day (days). For instance, time_buff = [-7, 0] will download data from 7 days before the occurrence to the occurrence date.
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

    if time_buff is not None and 'mint' not in geodf.columns:

        buff1 = np.timedelta64(time_buff[0], 'D')
        buff2 = np.timedelta64(time_buff[1], 'D')

        geodf['mint'] = pd.to_datetime(geodf['eventDate'] + buff1)
        geodf['bestt'] = pd.to_datetime(geodf['eventDate'])
        geodf['maxt'] = pd.to_datetime(geodf['eventDate'] + buff2)

    return(geodf)



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
            colnames.extend([dimdict[p]['standard_name'] + '_min',
                             dimdict[p]['standard_name'] + '_best',
                             dimdict[p]['standard_name'] + '_max'])
            coords.extend([ind[p]['min'], ind[p]['best'], ind[p]['max']])
        else:
            colnames.extend([dimdict[p]['standard_name'] + '_min',
                             dimdict[p]['standard_name'] + '_max'])
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
            colnames.extend([dimdict[p]['standard_name'] + '_min',
                             dimdict[p]['standard_name'] + '_best',
                             dimdict[p]['standard_name'] + '_max'])
            coords.extend([ind[p]['min'], ind[p]['best'], ind[p]['max']])
        else:
            colnames.extend([dimdict[p]['standard_name'] + '_min',
                             dimdict[p]['standard_name'] + '_max'])
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



def load_enrichment_file(dataset_ref, mute = False):

    """
    Load enrichment file.

    Args:
        dataset_ref (str): The enrichment file name (e.g. gbif_taxonKey).
        mute (bool): Not printing load message if mute is True.
    Returns:
        geopandas.GeoDataFrame or pandas.DataFrame: Data to enrich (including previously added columns).
    """


    with open(biodiv_path + dataset_ref + '-config.json') as f:
        enrichment_metadata = json.load(f)
    input_type = enrichment_metadata['input_type']

    filepath = biodiv_path + dataset_ref + '.csv'


    if input_type == 'occurrence':
        df = pd.read_csv(filepath, parse_dates = ['eventDate'], infer_datetime_format = True, index_col = 'id')
        df['geometry'] = df['geometry'].apply(wkt.loads)
        df = gpd.GeoDataFrame(df, crs = 'epsg:4326')

    else:
        df = pd.read_csv(filepath, parse_dates = ['mint', 'maxt'], infer_datetime_format = True, index_col = 'id')

    if not(mute):
        print('{} {}s were loaded from enrichment file'.format(len(df), input_type))
    
    return(df)



def create_enrichment_file(gdf, dataset_ref):

    """
    Create database file that will be used to save enrichment metadata.
    Dataframe index will be used as unique occurrences ids.
    
    Args:  
        gdf (geopandas.GeoDataFrame or pandas.DataFrame): Data to enrich (output of :func:`geoenrich.dataloader.open_dwca`
            or :func:`geoenrich.dataloader.import_csv` or :func:`geoenrich.dataloader.load_areas_file`).
        dataset_ref (str): The enrichment file name (e.g. gbif_taxonKey). Must be unique.
    Returns:
        None
    """

    filepath = biodiv_path + dataset_ref + '.csv'
    filepath_json = biodiv_path + dataset_ref + '-config.json'

    if(os.path.exists(filepath)):
        print('Abort. File already exists at ' + filepath)
    else:

        # Write input data to csv
        gdf.to_csv(filepath)

        # Create config file
        config = {}
        
        if 'geometry' in gdf.columns:
            # Occurrences input type
            config['input_type'] = 'occurrence'

        else:
            # Areas input type
            config['input_type'] = 'area'

        config['enrichments'] = []

        # Writing json config file 

        with open(filepath_json, 'x') as f:
            json.dump(config, f, ensure_ascii=False, indent=4)

        print('File saved at ' + filepath)



def reset_enrichment_file(dataset_ref, var_ids_to_remove):

    """
    Remove all enrichment data from the enrichment file. Does not remove downloaded data from netCDF files

    Args:
        dataset_ref (str): The enrichment file name (e.g. gbif_taxonKey).
        var_ids_to_remove (str list): List of variables to delete from the enrichment file. *\_all\_* removes everything.
    Returns:
        None
    """
    try:
        f = open(biodiv_path + dataset_ref + '-config.json')
        enrichment_metadata = json.load(f)
        f.close()
    except FileNotFoundError:
        print('No enrichment file was found for this dataset')

    enrichments = enrichment_metadata['enrichments']
    input_type = enrichment_metadata['input_type']
    remaining_enrichments = []

    df = load_enrichment_file(dataset_ref)

    to_drop = []

    for enrichment in enrichments:
        if enrichment['parameters']['var_id'] in var_ids_to_remove or \
            var_ids_to_remove == '_all_':

            prefix = str(enrichment['id']) + '_'
            for col in df.columns:
                if col[:len(prefix)] == prefix:
                    to_drop.append(col)
        else:
            remaining_enrichments.append(enrichment)


    to_save = df.drop(columns = to_drop)
    to_save.to_csv(filepath)

    enrichment_metadata['enrichments'] = remaining_enrichments

    with open(biodiv_path + dataset_ref + '-config.json', 'w') as f:
        json.dump(enrichment_metadata, f, ensure_ascii=False, indent=4)

    print('Enrichment file for dataset ' + dataset_ref + ' was reset.')




def enrichment_status(dataset_ref):

    """
    Return the number of occurrences of the given dataset that are already enriched, for each variable.

    Args:
        datset_ref (str): The enrichment file name (e.g. gbif_taxonKey).
    Returns:
        pandas.DataFrame: A table of variables and statuses of enrichment.
    """

    with open(biodiv_path + dataset_ref + '-config.json') as f:
        enrichment_metadata = json.load(f)

    enrichments = enrichment_metadata['enrichments']
    input_type = enrichment_metadata['input_type']

    df = load_enrichment_file(dataset_ref)

    col_indices = parse_columns(df)

    params_dict = {en['id']:en['parameters'] for en in enrichments}
    params_df = pd.DataFrame.from_dict(params_dict)

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
    

    return(pd.concat([params_df, res.fillna(0).astype(int)]))




def parse_columns(df):

    """
    Return column indices sorted by variable and dimension.

    Args:
        df (pandas.DataFrame): Enrichment file as a DataFrame, as returned by geoenrich.enrichment.load_enrichment_file.
    Returns:
        dict: Dictionary of column indices, with enrichment ID as a primary key, dimension as a secondary key, and min/max as tertiary key.
    """

    cols = [c.split('_') for c in df.columns]
    ind = {}

    for i in range(len(cols)):
        c = cols[i]
        if c[0].isnumeric():
            enrich_id = int(c[0])
            if enrich_id in ind:
                if c[1] in ind[enrich_id]:
                    ind[enrich_id][c[1]][c[2]] = i
                else:
                    ind[enrich_id][c[1]] = {c[2]: i}
            else:
                ind[enrich_id] = {c[1]: {c[2]: i}}

    return(ind)



def get_enrichment_id(enrichments, var_id, geo_buff, time_buff, depth_request, downsample):

    """
    Return ID of the requested enrichment if it exists, -1 otherwise.

    Args:
        enrichments (dict): Enrichments metadata as stored in the json config file.
        var_id (str): ID of the variable to download.
        geo_buff (int): Geographic buffer for which to download data around occurrence point (kilometers).
        time_buff (float list): Time bounds for which to download data around occurrence day (days). For instance, time_buff = [-7, 0] will download data from 7 days before the occurrence to the occurrence date.
        depth_request (str): Used when depth is a dimension. 'surface' only downloads surface data. Anything else downloads everything.
        downsample (dict): Number of points to skip between each downloaded point, for each dimension, using its standard name as a key.
    
    Returns:
        int: Enrichment ID.
    """

    result_id = -1

    current_parameters = {  'var_id':           var_id,
                            'geo_buff':         geo_buff,
                            'time_buff':        time_buff,
                            'depth_request':    depth_request,
                            'downsample':       downsample}

    if time_buff is not None:
        current_parameters['time_buff'] = list(time_buff)

    for enrichment in enrichments:
        if enrichment['parameters'] == current_parameters:
            result_id = enrichment['id']


    return(result_id)


def save_enrichment_config(dataset_ref, enrichment_id, var_id, geo_buff, time_buff, depth_request, downsample):

    """
    Save enrichment metadata in the json config file.

    Args:
        dataset_ref (str): The enrichment file name (e.g. gbif_taxonKey). Must be unique.
        enrichment_id (int): Enrichment ID.
        var_id (str): ID of the variable to download.
        geo_buff (int): Geographic buffer for which to download data around occurrence point (kilometers).
        time_buff (float list): Time bounds for which to download data around occurrence day (days). For instance, time_buff = [-7, 0] will download data from 7 days before the occurrence to the occurrence date.
        depth_request (str): Used when depth is a dimension. 'surface' only downloads surface data. Anything else downloads everything.
        downsample (dict): Number of points to skip between each downloaded point, for each dimension, using its standard name as a key.
    Returns:
        None
    """

    with open(biodiv_path + dataset_ref + '-config.json') as f:
        enrichment_metadata = json.load(f)

    new_enrichment = {'id': enrichment_id,
                      'parameters':
                           {'var_id':           var_id,
                            'geo_buff':         geo_buff,
                            'time_buff':        time_buff,
                            'depth_request':    depth_request,
                            'downsample':       downsample
                            }
                     }

    enrichment_metadata['enrichments'].append(new_enrichment)

    with open(biodiv_path + dataset_ref + '-config.json', 'w') as f:
        json.dump(enrichment_metadata, f, ensure_ascii=False, indent=4)



def read_ids(dataset_ref):

    """
    Return a list of all ids of the given enrichment file.
    
    Args:
        dataset_ref (str): The enrichment file name (e.g. gbif_taxonKey).
    Returns:
        list: List of all present ids.
    """

    filepath = biodiv_path + dataset_ref + '.csv'
    df = pd.read_csv(filepath, index_col = 'id')

    return(list(df.index))
