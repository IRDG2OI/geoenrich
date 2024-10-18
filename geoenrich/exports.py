"""
After enriching occurrences, you can use the exports module to use the downloaded data. Several options are available:
- Produce a stats file that will give you the average, standard deviation, minimum and maximum values of your variable in the buffer around each occurrence. See :func:`geoenrich.exports.produce_stats`.
- Calculate the derivative of the environmental data between two different dates, with :func:`geoenrich.exports.get_derivative`.
- Export png pictures, for visualization or training deep learning models for instance. See :func:`geoenrich.exports.export_png`.
- Retrieve the raw data as a numpy array with :func:`geoenrich.exports.retrieve data`.

"""
from pathlib import Path
import json
import pandas as pd
import cv2
import random
from matplotlib import cm

import rasterio
from rasterio.transform import from_origin

import geoenrich

try:
    from geoenrich.credentials import *
except:
    from geoenrich.credentials_example import *

from geoenrich.enrichment import *
from geoenrich.satellite import *


def retrieve_data(dataset_ref, occ_id, var_id, geo_buff = None, time_buff = None, depth_request = None,
                    downsample = None, shape = 'rectangle', serialized = {}):

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
        serialized (dict): (Optional) provide a dictionary of variables to reduce processing time (supports df, dimdict, var, var_source, ds).
    Returns:
        dict: A dictionary of all available variables with corresponding data (numpy.ma.MaskedArray), unit (str), and coordinates (ordered list of dimension names and values).
    """


    with Path(biodiv_path, dataset_ref + '-config.json').open() as f:
        enrichment_metadata = json.load(f)

    enrichments = enrichment_metadata['enrichments']
    input_type = enrichment_metadata['input_type']

    if 'df' in serialized:
        df = serialized['df']
    else:
        df, _ = load_enrichment_file(dataset_ref, mute = True)
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
                or make sure you are using the same as during enrichment")

    elif len(relevant) > 1:
        print('Several enrichment sessions were found with the given parameters.\
                Please specify more parameters to narrow down.')
    
    else:
        # Read indices into a dictionary

        en = relevant[0]
        var_ind = parse_columns(df)[en['id']]
        if 'var_source' in serialized:
            var_source = serialized['var_source']
        else:
            var_source = get_var_catalog()[var_id]

        if -1 in [row.iloc[d['min']] for d in var_ind.values()]:
            results = {'coords': None, 'values': None}

        else:

            # Recover serialized variables

            if 'ds' in serialized:
                ds = serialized['ds']
            else:
                ds = nc.Dataset(str(Path(sat_path, var_id + '.nc')))

            if ('dimdict' in serialized) and ('var' in serialized):
                dimdict, var = serialized['dimdict'], serialized['var']
            else:
                dimdict, var = get_metadata(ds, var_source['varname'])



            unit = getattr(ds.variables[var_source['varname']], 'units', 'Unspecified')

            data, coords = fetch_data(row, var_id, var_ind, ds, dimdict, var, downsample)

            if 'ds' not in serialized:
                ds.close()

            if shape == 'buffer' and input_type == 'occurrence':
                geo_buff = en['parameters']['geo_buff']
                mask = ellipsoid_mask(data, coords, row['geometry'], geo_buff)
                return({'coords': coords, 'values': np.ma.masked_where(mask, data), 'unit': unit})
            else:
                return({'coords': coords, 'values': data, 'unit': unit})


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
    Returns:
        numpy.ma.MaskedArray, list: Raw data and coordinates along all dimensions.
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

    if downsample is None:
        downsample = {}

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
                if var['name'] in ['uwnd', 'vwnd']:
                    times = num2pydate(time_var[i1:i2+1:step] - 725563, 'days since 1987-01-01 00:00:00')
                else:
                    times = num2pydate(time_var[i1:i2+1:step], time_var.__dict__['units'])
            coordinates.append([p, times])
        else:
            coordinates.append([p, ds.variables[dimdict[p]['name']][i1:i2+1:step]])

    return(data, coordinates)




def produce_stats(dataset_ref, var_id, geo_buff = None, time_buff = None, depth_request = None,
                    downsample = None, out_path = Path('./')):

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
        out_path (str or pathlib.Path): Path where you want to save the output stats file.

    Returns:
        None
    """

    df, enrichment_metadata = load_enrichment_file(dataset_ref)
    enrichments = enrichment_metadata['enrichments']
    input_type = enrichment_metadata['input_type']


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
        print("No enrichment was found with the provided parameters. Try again with fewer parameters \
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
        ds = nc.Dataset(str(Path(sat_path, var_id + '.nc')))
        dimdict, var = get_metadata(ds, var_source['varname'])

        print('Computing stats for ' + var_id + '...')
        res = df.progress_apply(compute_stats, axis=1, args = (en['parameters'], input_type, var_ind, ds, dimdict, var),
                                     result_type = 'expand')
        ds.close()


        filepath = Path(out_path, dataset_ref + '_' + str(en['id']) + '_stats.csv')
        res.to_csv(str(filepath))
        print(f'File saved at {filepath}')



def compute_stats(row, en_params, input_type, var_indices, ds, dimdict, var):

    """
    Compute and return stats for the given row.

    
    Args:
        row (pandas.Series): One row of an enrichment file.
        en_params (dict): Enrichment parameters as stored in the json config file.
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

    df, enrichment_metadata = load_enrichment_file(dataset_ref)

    input_type = enrichment_metadata['input_type']

    row = df.loc[[occ_id]]
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


    ds = nc.Dataset(str(Path(sat_path, var_id + '.nc')))
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


def export_png(dataset_ref, occ_id, var_id, target_size = None, value_range=None, path = Path('./'),
               geo_buff = None, time_buff = None, depth_request = None, downsample = None,
               cmap = 'coolwarm', shape = 'rectangle'):

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
        path (str or pathlib.Path): Path where image files will be saved.
        geo_buff (int): (Optional) Geo_buff that was used for enrichment.
        time_buff (float list): (Optional) Time_buff that was used for enrichment.
        depth_request (str): (Optional) Depth request that was used for enrichment.
        downsample (dict): (Optional) Downsample that was used for enrichment.
        cmap (str): (Optional) Specify a colormap (see matplotlib.cm for reference).
        shape (str): If 'rectangle', return data inside the rectangle containing the buffer. If 'buffer', only return data within the buffer distance from the occurrence location.

    Returns:
        None
    """

    folderpath = Path(path) / dataset_ref
    if not folderpath.exists():
        folderpath.mkdir()

    # Retrieve data
    res = retrieve_data(dataset_ref, occ_id, var_id, geo_buff, time_buff, depth_request=depth_request,
                        downsample = downsample, shape=shape)

    if res is not None:

        if value_range is None:
            value_range = [np.nanmin(res['values']), np.nanmax(res['values'])]

        im = export_to_array(res, target_size, value_range)

        # Flip latitude (because image vertical axis is downwards)
        lat_ax = [c[0] for c in res['coords']].index('latitude')
        lats = res['coords'][lat_ax][1]
        if len(lats)>1 and lats[0] < lats[1]:
            im = np.flipud(im)

        # Map values to color scale
        im2 = getattr(cm, cmap)(im)
        im3 = cv2.cvtColor(np.float32(im2), cv2.COLOR_BGR2RGBA)
        im3[:,:,3] =  1 - np.isnan(im)

        im_path = Path(folderpath, str(occ_id) + '_' + var_id + '.png')
        cv2.imwrite(str(im_path), 255*im3)

        print(f'Image saved at {im_path}')



def export_to_array(res, target_size=None, value_range=None, stack=None, squeeze=True, target_len=None):

    """
    Export data as a 3D numpy array where the first 2 dimensions represent geographical coordinates.
    Option to standardize data by specifiying target size and target value range.
    The third dimensions stores multiples bands if stack is set to *depth*, *time* or *all*.


    Args:
        res (dict): output of :func:`geoenrich.exports.retrieve_data`.
        target_size (int tuple): Size of the target array (width, height). If None, using the native data resolution.
        value_range (float list): Range of the variable. Necessary for consistency between all images.
        stack (str): If True, keep values for all depths or all times (returns 3D array).
        squeeze (bool): If true, remove unused dimensions in the output.
        target_len (int): Length of the third dimension if data is None (to return uniform results).
    Returns:
        numpy.array: output data, scaled and resized.

    """

    if res is not None:

        im = res['values']
        params = [c[0] for c in res['coords']]
        lat_ax = params.index('latitude')
        lon_ax = params.index('longitude')

        im1 = deepcopy(im)


        # Reorder axes

        if lat_ax != 0:
            im1 = np.swapaxes(im1, lat_ax, 0)
            params[0], params[lat_ax] = params[lat_ax], params[0]
            lat_ax = 0

        if lon_ax != 1:
            im1 = np.swapaxes(im1, lon_ax, 1)
            params[1], params[lon_ax] = params[lon_ax], params[1]
            lon_ax = 1

        # Remove unwanted dimensions

        if stack is None:

            # Keep only shallowest and most recent

            if 'time' in params:
                time_ax = params.index('time')
                im1 = im1.take(-1, axis=time_ax)

            if 'depth' in params:
                depth_ax = params.index('depth')
                if 'time' in params and params.index('time') < params.index('depth'):
                    depth_ax -= 1
                
                im1 = im1.take(np.argmin(res['coords'][depth_ax][1]), axis=depth_ax)

        elif stack == 'time':

            # Keep all times but only most shallow

            if 'depth' in params:
                depth_ax = params.index('depth')                
                im1 = im1.take(np.argmin(res['coords'][depth_ax][1]), axis=depth_ax)


        elif stack == 'depth':

            # Keep all depths but only most recent

            if 'time' in params:
                time_ax = params.index('time')
                im1 = im1.take(-1, axis=time_ax)


        else:

            # Keep everything

            pass


        im1 = im1.reshape([im1.shape[0], im1.shape[1], -1])
        mask = im1.mask

        im2 = deepcopy(im1)

        # Resize
        if target_size is not None:

            if im1.shape[0] < target_size[0] or im1.shape[1] < target_size[1]:
                im2 = cv2.resize(im1, target_size, interpolation = cv2.INTER_AREA)
                if len(im1.mask.shape):
                    mask = cv2.resize(im1.mask.astype('float32'), target_size, interpolation = cv2.INTER_AREA)
            else:
                im2 = cv2.resize(im1, target_size, interpolation = cv2.INTER_LINEAR)
                if len(im1.mask.shape):
                    mask = cv2.resize(im1.mask.astype('float32'), target_size, interpolation = cv2.INTER_LINEAR)
            
            # If there is only one band, cv2 returns squeezed version
            im2 = im2.reshape([im2.shape[0], im2.shape[1], -1])
            
        # Scale from value range to [0,1]
        if value_range is not None:
            im2 = np.interp(im2, value_range, [0, 1])

        im3 = np.ma.masked_array(im2.data, mask=mask)
        im3 = np.ma.filled(im3, np.nan)

        if squeeze:
            return (im3.squeeze())
        else:
            return (im3)

    elif (target_size is not None) and (target_len is not None):

        empty = np.full([*target_size, target_len], np.nan)
        return (empty)

    else:
        return (None)

    

def export_raster(dataset_ref, occ_id, var_id, path = Path('./'), geo_buff = None, time_buff = None,
                    depth_request = None, downsample = None, shape = 'rectangle', multiband = None):

    """
    Export a GeoTiff raster of the requested data.
    Depth or time dimension (not both) can be stored as band (see multiband argument)
    Otherwise, the shallowest depth and most recent time are selected.

    Args:
        dataset_ref (str): The enrichment file name (e.g. gbif_taxonKey).
        occ_id (str): ID of the occurrence to get data for. Can be obtained with :func:`geoenrich.enrichment.read_ids`.
        var_id (str): ID of the variable to retrieve.
        path (str or pathlib.Path): Path where image files will be saved.
        geo_buff (int): (Optional) Geo_buff that was used for enrichment.
        time_buff (float list): (Optional) Time_buff that was used for enrichment.
        depth_request (str): (Optional) Depth request that was used for enrichment.
        downsample (dict): (Optional) Downsample that was used for enrichment.
        shape (str): If 'rectangle', return data inside the rectangle containing the buffer. If 'buffer', only return data within the buffer distance from the occurrence location.
        multiband (str): If multiband='depth' or 'time', the corresponding dimension is saved into multiple bands.
    Returns:
        None
    """

    folderpath = Path(path) / (dataset_ref + '_rasters')
    if not folderpath.exists():
        folderpath.mkdir()

    # Retrieve data
    res = retrieve_data(dataset_ref, occ_id, var_id, geo_buff, time_buff, depth_request=depth_request,
                        downsample = downsample, shape=shape)

    if res is not None:

        assert multiband in [None, 'time', 'depth']

        im = export_to_array(res, target_size=None, value_range=None, stack=multiband)

        # Flip latitude (because image vertical axis is downwards)
        lat_ax = [c[0] for c in res['coords']].index('latitude')
        lats = res['coords'][lat_ax][1]

        lon_ax = [c[0] for c in res['coords']].index('longitude')
        lons = res['coords'][lon_ax][1]


        if len(lats)>1 and lats[0] > lats[1]:
            im = np.flipud(im)

        if im.shape[0] > 1 and im.shape[1] > 1:

            x_pxl = lons[1] - lons[0]
            y_pxl = lats[0] - lats[1]

            transform = from_origin(lons[0] - .5*x_pxl, lats[0] + .5*y_pxl, x_pxl, y_pxl)

            im_path = Path(folderpath, str(occ_id) + '_' + var_id + '.tiff')

            if len(im.shape) == 2:
                new_raster = rasterio.open(im_path, 'w', driver='GTiff',
                                height = im.shape[0], width = im.shape[1],
                                count=1, dtype=str(im.dtype),
                                crs='EPSG:4326',
                                transform=transform)

                new_raster.write(im, 1)
                new_raster.close()
            else:
                band_ax = [c[0] for c in res['coords']].index(multiband)
                bandnames = res['coords'][band_ax][1]

                new_raster = rasterio.open(im_path, 'w', driver='GTiff',
                                height = im.shape[0], width = im.shape[1],
                                count=len(bandnames), dtype=str(im.dtype),
                                crs='EPSG:4326',
                                transform=transform)

                for i in range(im.shape[2]):
                    band = im.take(i, axis=2)
                    new_raster.write_band(i+1, band)
                    new_raster.set_band_description(i+1, f'{multiband}={bandnames[i]}')

                new_raster.close()

            print(f'Raster saved at {im_path}')

        else:

            print('Abort. Array is smaller than 2x2 pixels.')


def collate_npy(ds_ref, data_path, output_res = 32, slice = None, dimension3 = {'example-var': 2}, duplicates = {'var_to_remove':'var_to_keep'}):

    """
    Export a 3D numpy array with all layers for each occurrence of a dataset.
    WARNING: the dimension3 dictionary must be provided if some variables have a time or depth dimension.

    Args:
        ds_ref (str): The enrichment file name (e.g. gbif_taxonKey).
        data_path (str): path where numpy files will be saved.
        output_res (int) : output data resolution along lat and lon axes.
        slice (list[int]): if not None, only process the given slice of the dataset.
        dimension3 (dict): provides the expected 3rd dimension length (time dimension * depth dimension) for each variable where it is larger than 1.
        duplicates (dict): dictionnary of variables which should be merged. If var_to_keep is empty, data from var_to_remove are used instead.


    Returns:
        None
    """

    folderpath = data_path / (ds_ref + '-npy')
    if not(folderpath.exists()):
        folderpath.mkdir()

    df, enrichment_metadata = load_enrichment_file(ds_ref, mute=True)
    enrichments = enrichment_metadata['enrichments']
    input_type = enrichment_metadata['input_type']
    serial_dict = {}


    # Prepare variable-specific data to be passed to retrieve_data function.

    for en in enrichments:
            
        params = en['parameters']
        var_id = params['var_id']

        ds = nc.Dataset(str(Path(sat_path, var_id + '.nc')))
        var_source = get_var_catalog()[var_id]
        serial_dict[en['id']] = {'df': df} 
        serial_dict[en['id']]['var_source'] = var_source
        serial_dict[en['id']]['dimdict'], serial_dict[en['id']]['var'] = get_metadata(ds, var_source['varname'])
        serial_dict[en['id']]['ds'] = ds


    # Define ids to be exported

    if slice is None:
        ids = df.index
    else:
        ids = list(df.index)[slice[0]:slice[1]]


    # Export np arrays for each occurrence

    var_list = [en['parameters']['var_id'] for en in enrichments]
    for v in duplicates.keys():
        var_list.remove(v)

    for occ_id in tqdm(ids):
        all_bands = {}
        for en in enrichments:
            
            params = en['parameters']
            var_id = params['var_id']

            res = retrieve_data(ds_ref, occ_id, var_id, geo_buff=params['geo_buff'],
                                time_buff=params['time_buff'],
                                depth_request=params['depth_request'],
                                downsample=params['downsample'],
                                serialized=serial_dict[en['id']])

            # Specify target length in case res is empty.

            if var_id in dimension3:
                target_len = dimension3[var_id]
            else:
                target_len = 1

            band = export_to_array(res, target_size = [output_res, output_res],
                                              value_range = None,
                                              stack = True,
                                              squeeze = False,
                                              target_len = target_len)
            all_bands[var_id] = band
        
        # replace missing values with value from duplicate variable; and remove said duplicates
        for to_rem in duplicates:
            if np.isnan(all_bands[duplicates[to_rem]]).all():
                all_bands[duplicates[to_rem]] = all_bands[to_rem]
            all_bands.pop(to_rem)

        var_data = [all_bands[k] for k in var_list]

        to_save = np.concatenate(var_data, -1)
        np.save(folderpath / (str(occ_id) + '.npy'), to_save)


    with open(folderpath / '0000_npy_metadata.txt', 'w') as f:
        for line in var_list:
            f.write(f"{line}\n")



    # close NC datasets

    for en in enrichments:
        serial_dict[en['id']]['ds'].close()