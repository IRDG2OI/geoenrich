"""
The downstream module handles all operations on data after it is downloaded.

"""

import pandas as pd
import cv2
from matplotlib import cm

import geoenrich

try:
    from geoenrich.credentials import *
except:
    from geoenrich.credentials_example import *

from geoenrich.enrichment import *
from geoenrich.satellite import *



def retrieve_data(dataset_ref, occ_id, var_id, geo_buff = None, time_buff = None, depth_request = 'surface',
                    downsample = {}, shape = 'rectangle'):

    """
    Retrieve downloaded data for the given occurrence id and variable.
    If enrichment was done several times with different buffers, specify
    
    Args:        
        dataset_ref (str): The enrichment file name (e.g. gbif_taxonKey).
        occ_id (str): ID of the occurrence to get data for. Can be obtained with :func:`geoenrich.enrichment.read_ids`.
        var_id (str): ID of the variable to retrieve.
        geo_buff (int): (Optional) Geo_buff that was used for enrichment.
        time_buff (float list): (Optional) Time_buff that was used for enrichment.
        depth_request (str): (Optional) Depth request that was used for enrichment.
        downsample (dict): (Optional) Downsample that was used for enrichment.
        shape (str): If 'rectangle', return data inside the rectangle containing the buffer. If 'buffer', only return data within the buffer distance from the occurrence location.
    Returns:
        dict: A dictionary of all available variables with corresponding data (numpy.ma.MaskedArray), unit (str), and coordinates (ordered list of dimension names and values).
    """


    with open(biodiv_path + dataset_ref + '-config.json') as f:
        enrichment_metadata = json.load(f)

    enrichments = enrichment_metadata['enrichments']
    input_type = enrichment_metadata['input_type']

    df = load_enrichment_file(dataset_ref, input_type)
    row = df.loc[occ_id]

    # Identify relevant enrichment ids

    relevant = []
    for en in enrichments:
        if en['parameters']['var_id'] == var_id:
            if  (geo_buff is None or en['parameters']['geo_buff'] == geo_buff)                  and \
                (time_buff is None or en['parameters']['time_buff'] == time_buff)               and \
                (depth_request is None or en['parameters']['depth_request'] == depth_request)   and \
                (downsample is None or en['parameters']['downsample'] == downsample):

                relevant.append(en)

    if len(relevant) == 0:
        print("No enrichment was found with the provided parameters. Try again with fewer current_parameters \
                or make sure you ar eusing the same as during enrichment")

    elif len(relevant) > 1:
        print('Several enrichment sessions were found with the given parameters.\
                Please specify more parameters to narrow down.')
    
    else:
        # Read indices into a dictionary

        en = relevant[0]
        var_ind = parse_columns(df)[en['id']]
        var_source = get_var_catalog()[var_id]

        if -1 in [row.iloc[d['min']] for d in var_ind.values()]:
            results = {'coords': None, 'values': None}

        else:
            ds = nc.Dataset(sat_path + var_id + '.nc')
            unit = getattr(ds.variables[var_source['varname']], 'units', 'Unspecified')

            dimdict, var = get_metadata(ds, var_source['varname'])

            data, coords = fetch_data(row, var_id, var_ind, ds, dimdict, var, downsample)
            ds.close()

        if shape == 'buffer' and input_type == 'occurrence':
            geo_buff = en['parameters']['geo_buff']
            mask = ellipsoid_mask(data, coords, row['geometry'], geo_buff)
            return({'coords': coords, 'values': np.ma.masked_where(mask, data), 'unit': unit})
        else:
            return({'coords': coords, 'values': data, 'unit': unit})


        return(results)



def fetch_data(row, var_id, var_indices, ds, dimdict, var, downsample, indices = None):

    """
    Fetch data locally for a specific occurrence and variable.
    
    Args:
        row (pandas.Series): One row of an enrichment file.
        var_id (str): ID of the variable to fetch.
        var_indices (dict):  Dictionary of column indices for the selected variable, output of :func:`geoenrich.enrichment.parse_columns`.
        ds (netCDF4.Dataset): Local dataset.
        dimdict (dict): Dictionary of dimensions as returned by geoenrich.satellite.get_metadata.
        var (dict): Variable dictionary as returned by geoenrich.satellite.get_metadata.
        downsample (dict): Number of points to skip between each downloaded point, for each dimension, using its standard name as a key.
        indices (dict): Coordinates of the netCDF subset. If None, they are read from row and var_indices arguments. 
        numpy.ma.MaskedArray: Raw data.
    """


    params = [dimdict[n]['standard_name'] for n in var['params']]

    if indices is None:
        ordered_indices_cols = [var_indices[p] for p in params]
        ordered_indices = [{'min': int(row.iloc[d['min']]),
                            'max': int(row.iloc[d['max']]),
                            'step': 1}
                            for d in ordered_indices_cols]
        indices = {params[i]:ordered_indices[i] for i in range(len(params))}
    else:
        ordered_indices = [indices[p] for p in params]

    for i in range(len(params)):
        p = params[i]
        if p in downsample:
            ordered_indices[i]['step'] = downsample[p] + 1

    lons = dimdict['longitude']['vals']
    lon_pos = var['params'].index(dimdict['longitude']['name'])
    data = multidimensional_slice(ds, var['name'], ordered_indices, lons, lon_pos)

    coordinates = []
    for p in params:
        i1, i2 = indices[p]['min'], indices[p]['max']
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



def read_ids(dataset_ref):

    """
    Return a list of all ids of the given enrichment file.
    
    Args:
        dataset_ref (str): The enrichment file name (e.g. gbif_taxonKey).
        id_col (str or int): Index or name of the ID column.
    Returns:
        list: List of all present ids.
    """

    filepath = biodiv_path + dataset_ref + '.csv'
    df = pd.read_csv(filepath, index_col = 'id')

    return(list(df.index))




def produce_stats(dataset_ref, var_id, geo_buff = None, time_buff = None, depth_request = 'surface',
                    downsample = {}, out_path = biodiv_path):

    """
    Produce a document named *dataset\_ref*\_stats.csv with summary stats of all enriched data.
    If input data were occurrences, only data within the buffer distance are used for calculations.

    Args:
        dataset_ref (str): The enrichment file name (e.g. gbif_taxonKey).
        var_id (str): ID of the variable to retrieve.
        geo_buff (int): (Optional) Geo_buff that was used for enrichment.
        time_buff (float list): (Optional) Time_buff that was used for enrichment.
        depth_request (str): (Optional) Depth request that was used for enrichment.
        downsample (dict): (Optional) Downsample that was used for enrichment.
        out_path (str): Path where you want to save the output stats file.

    Returns:
        None
    """

    with open(biodiv_path + dataset_ref + '-config.json') as f:
        enrichment_metadata = json.load(f)

    enrichments = enrichment_metadata['enrichments']
    input_type = enrichment_metadata['input_type']

    df = load_enrichment_file(dataset_ref, input_type)

    # Identify relevant enrichment ids

    relevant = []
    for en in enrichments:
        if en['parameters']['var_id'] == var_id:
            if  (geo_buff is None or en['parameters']['geo_buff'] == geo_buff)                  and \
                (time_buff is None or en['parameters']['time_buff'] == time_buff)               and \
                (depth_request == 'surface' or en['parameters']['depth_request'] == depth_request)   and \
                (en['parameters']['downsample'] == downsample):

                relevant.append(en)

    if len(relevant) == 0:
        print("No enrichment was found with the provided parameters. Try again with fewer current_parameters \
                or make sure you are using the same as during enrichment")

    elif len(relevant) > 1:
        print('Several enrichment sessions were found with the given parameters.\
                Please specify more parameters to narrow down.')
    
    else:
        # Read indices into a dictionary

        en = relevant[0]
        var_ind = parse_columns(df)[en['id']]
        results = {}

        var_source = get_var_catalog()[var_id]
        ds = nc.Dataset(sat_path + var_id + '.nc')
        dimdict, var = get_metadata(ds, var_source['varname'])

        print('Computing stats for ' + var_id + '...')
        res = df.progress_apply(compute_stats, axis=1, args = (en['parameters'], input_type, var_ind, ds, dimdict, var),
                                     result_type = 'expand')
        ds.close()


        filepath = out_path + dataset_ref + '_' + str(en['id']) + '_stats.csv'
        res.to_csv(filepath)
        print('File saved at ' + filepath)



def compute_stats(row, en_params, input_type, var_indices, ds, dimdict, var):

    """
    Compute and return stats for the given row.

    
    Args:
        row (pandas.Series): One row of an enrichment file.
        enrichments (dict): Enrichment parameters as stored in the json config file.
        input_type (str): 'occurrence' or 'area'.
        var_indices (dict):  Dictionary of column indices for the selected variable, output of :func:`geoenrich.enrichment.parse_columns`.
        ds (netCDF4.Dataset): Local dataset.
        dimdict (dict): Dictionary of dimensions as returned by geoenrich.satellite.get_metadata.
        var (dict): Variable dictionary as returned by geoenrich.satellite.get_metadata.
    
    Returns:
        pandas.Series: Statistics for the given row.
    """

    var_id = en_params['var_id']
    geo_buff = en_params['geo_buff']
    downsample = en_params['downsample']

    min_columns = [row.iloc[d['min']] for d in var_indices.values()]

    if -1 in min_columns or any([np.isnan(c) for c in min_columns]):
        # Missing data
        names = [var_id + '_av', var_id + '_std', var_id + '_min', var_id + '_max', var_id + '_count']
        return(pd.Series(index=names, dtype = float))

    data, coords = fetch_data(row, var_id, var_indices, ds, dimdict, var, downsample)

    params = [dimdict[n]['standard_name'] for n in var['params']]
    ordered_indices_cols = [var_indices[p] for p in params]

    if input_type == 'occurrences':

        # If data was calulated around an occurrence

        mask = ellipsoid_mask(data, coords, row['geometry'], geo_buff)
        data = np.ma.masked_where(mask, data)

    av, std = np.ma.average(data), np.ma.std(data)
    minv, maxv, count = np.ma.min(data), np.ma.max(data), np.ma.count(data)
    names = [var_id + '_av', var_id + '_std', var_id + '_min', var_id + '_max', var_id + '_count']
    ret = pd.Series([av, std, minv, maxv, count], index = names)

    return(ret)



def get_derivative(dataset_ref, occ_id, var_id, days = (0,0), geo_buff = None, depth_request = 'surface',
                        downsample = {}, shape = 'rectangle'):

    """

    Retrieve data for both specified days and return the derivative.
    geo_buff and downsample must be identical to the values you used for enrichment.
    
    Args:
        dataset_ref (str): The enrichment file name (e.g. gbif_taxonKey).
        occ_id (str): ID of the occurrence to get data for. Can be obtained with :func:`geoenrich.enrichment.read_ids`.
        var_id (str): ID of the variable to derivate.
        days (int tuple): Start and end days for derivative calculation.
                If enriching occurrences, provide bounds relatively to occurrence, eg. (-7, 0).
                If enriching areas, provide bounds relatively to date_max, eg. (-7, 0).
        geo_buff (int): (Optional) Geo_buff that was used for enrichment.
        depth_request (str): (Optional) Depth request that was used for enrichment.
        downsample (dict): (Optional) Downsample that was used for enrichment.
        shape (str): If 'rectangle', return data inside the rectangle containing the buffer. If 'buffer', only return data within the buffer distance from the occurrence location.

    Returns:
        dict: A dictionary of all available variables with corresponding data (numpy.ma.MaskedArray), unit (str), and coordinates (ordered list of dimension names and values).
    """

    with open(biodiv_path + dataset_ref + '-config.json') as f:
        enrichment_metadata = json.load(f)
    input_type = enrichment_metadata['input_type']

    row = load_enrichment_file(dataset_ref, input_type).loc[[occ_id]]
    row1, row2 = deepcopy(row), deepcopy(row)
    
    
    if input_type == 'occurrence':
        row1['eventDate'] = pd.to_datetime(row1['eventDate'] + np.timedelta64(days[0], 'D'))
        row2['eventDate'] = pd.to_datetime(row2['eventDate'] + np.timedelta64(days[1], 'D'))
        row1 = add_bounds(row1, geo_buff, (0,0))
        row2 = add_bounds(row2, geo_buff, (0,0))
    else:
        row1['mint'] = pd.to_datetime(row1['maxt'] + np.timedelta64(days[0], 'D'))
        row1['maxt'] = pd.to_datetime(row1['maxt'] + np.timedelta64(days[0], 'D'))
        row2['mint'] = pd.to_datetime(row2['maxt'] + np.timedelta64(days[1], 'D'))
        row2['maxt'] = pd.to_datetime(row2['maxt'] + np.timedelta64(days[1], 'D'))

    # Read indices into a dictionary

    var_source = get_var_catalog()[var_id]


    ds = nc.Dataset(sat_path + var_id + '.nc')
    unit = getattr(ds.variables[var_source['varname']], 'units', 'Unspecified')
    dimdict, var = get_metadata(ds, var_source['varname'])

    ind1 = calculate_indices(dimdict, row1.iloc[0], var, depth_request, downsample)
    ind2 = calculate_indices(dimdict, row2.iloc[0], var, depth_request, downsample)

    data1, coords1 = fetch_data(None, var_id, None, ds, dimdict, var, downsample, ind1)
    data2, coords2 = fetch_data(None, var_id, None, ds, dimdict, var, downsample, ind2)
    ds.close()

    data = (data2 - data1) / (days[1] - days[0])

    coords = []
    for c in coords1:
        if c[0] != 'time':
            coords.append(c)

    if shape == 'buffer' and geo_buff is not None:
        mask = ellipsoid_mask(data, coords, row['geometry'], geo_buff)
        return({'coords': coords, 'values': np.ma.masked_where(mask, data), 'unit': unit + ' per day'})
    else:
        return({'coords': coords, 'values': data, 'unit': unit + ' per day'})


def export_png(dataset_ref, occ_id, var_id, target_size = None, value_range = None, path = biodiv_path, geo_buff = None,
                    time_buff = None, depth_request = 'surface', downsample = {}):

    """
    Export a png image of the requested data.
    If depth is a dimension, the shallowest layer is selected.
    If time is a dimension, the most recent layer is selected.

    Args:
        dataset_ref (str): The enrichment file name (e.g. gbif_taxonKey).
        occ_id (str): ID of the occurrence to get data for. Can be obtained with :func:`geoenrich.enrichment.read_ids`.
        var_id (str): ID of the variable to retrieve.
        target_size (int tuple): Size of the target picture (width, height). If None, using the native data resolution.
        value_range (float list): Range of the variable. Necessary for consistency between all images.
        path (str): Path where image files will be saved.
        geo_buff (int): (Optional) Geo_buff that was used for enrichment.
        time_buff (float list): (Optional) Time_buff that was used for enrichment.
        depth_request (str): (Optional) Depth request that was used for enrichment.
        downsample (dict): (Optional) Downsample that was used for enrichment.

    Returns:
        None
    """

    folderpath = path + dataset_ref + '/'
    if not(os.path.exists(folderpath)):
        os.mkdir(folderpath)

    # Retrieve data
    res = retrieve_data(dataset_ref, occ_id, var_id, geo_buff, time_buff, downsample = downsample)
    im = res['values']

    params = [c[0] for c in res['coords']]
    
    # Transform to 2D data by removing additionnal dimensions.
    lat_ax = params.index('latitude')
    lon_ax = params.index('longitude')

    if 'time' in params:
        time_ax = params.index('time')
        im = im.take(-1, axis = time_ax)

    if 'depth' in params:
        depth_ax = params.index('depth')
        im = im.take(np.argmin(res['coords'][depth_ax][1]), axis = depth_ax)

    # Scale from value range to [0,1]
    if value_range is None:
        value_range = [im.min(), im.max()]

    im1 = np.interp(im, value_range,[0,1])

    # Transpose if needed
    if lon_ax > lat_ax:
        im1 = np.transpose(im1)

    # Flip latitude (because image vertical axis is downwards)
    im1 = np.flipud(im1)

    # Map values to color scale
    im2 = cm.coolwarm(im1)
    im2[:,:,3] =  1 - im.mask.astype(int)

    # Resize
    if target_size is not None:
        if im2.shape[0] < target_size[0] or im2.shape[1] < target_size[1]:
            im2 = cv2.resize(im2, target_size, interpolation = cv2.INTER_AREA)
        else:
            im2 = cv2.resize(im2, target_size, interpolation = cv2.INTER_CUBIC)

    cv2.imwrite(folderpath + occ_id + '_' + var_id + '.png', 255*im2)