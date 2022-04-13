import os
from copy import deepcopy

import numpy as np
import pandas as pd
import netCDF4 as nc

from datetime import datetime
from cftime import num2pydate

import geoenrich

try:
    from geoenrich.credentials import *
except:
    from geoenrich.credentials_example import *
    print('Please rename credentials_example.py to credentials.py and fill in the blanks')
    print('File location: ' + os.path.split(geoenrich.__file__)[0])


def get_metadata(ds, varname):

    """
    Download and format useful metadata on dimensions and variables.
    Generate a dictionary where dimensions can be accessed both with their original name and their standard name (if available).
    
    Args:
        ds (netCDF4.Dataset): Dataset of interest.
        varname (str): Name of the variable of interest in the dataset.
    Returns:
        dict, dict: dictionary with standardized information on dimensions, dictionary with information on the variable.
    """

    dimdict = {}
    var = None

    for name in ds.variables:

        # Format time dimension
        # Need to convert netcdf datetime to python datetime

        if 'standard_name' in ds.variables[name].__dict__ \
            and ds.variables[name].__dict__['standard_name'] == 'time':


            times = num2pydate(ds.variables[name][:], ds.variables[name].__dict__['units'])
            times = pd.Series([datetime(*d.timetuple()[:-3]) for d in times])
            item = {'name': name, 'standard_name': 'time', 'vals': times, 'unit': None}
            dimdict[name] = item
            dimdict['time'] = item

        # Format lon & lat dimensions

        elif 'standard_name' in ds.variables[name].__dict__ \
            and ds.variables[name].__dict__['standard_name'] in ('longitude', 'latitude', 'depth'):

            item = {'name': name,
                    'standard_name': ds.variables[name].__dict__['standard_name'],
                    'vals': ds.variables[name][:],
                    'unit': ds.variables[name].__dict__['units']}
            dimdict[name] = item
            dimdict[ds.variables[name].standard_name] = item

        # Format requested variable

        elif name == varname:
            
            var = {'name':name,
                    'unit': ds.variables[name].__dict__['units'],
                    'params': [v.name for v in ds.variables[name].get_dims()],
                    'add_offset': getattr(ds.variables[name], 'add_offset', 0),
                    'scale_factor': getattr(ds.variables[name], 'scale_factor', 1)}

            if 'standard_name' in ds.variables[name].__dict__:
                var['standard_name'] = ds.variables[name].__dict__['standard_name']
        

        # Search for latitude and longitude in case standard names were not provided

        elif name in ['lat', 'latitude']:

            item = {'name': name,
                    'standard_name': 'latitude',
                    'vals': ds.variables[name][:],
                    'unit': ds.variables[name].__dict__['units']}
            dimdict[name] = item
            dimdict['latitude'] = item

        elif name in ['lon', 'longitude']:

            item = {'name': name,
                    'standard_name': 'longitude',
                    'vals': ds.variables[name][:],
                    'unit': ds.variables[name].__dict__['units']}
            dimdict[name] = item
            dimdict['longitude'] = item


    return dimdict, var



def get_var_catalog():

    """
    Return available variables, with dataset attributes.
    catalog.csv can be edited to add additional variables.

    Args:
        None
    Returns:
        dict: Dictionary with variable id, variable name in dataset, and dataset url
    """

    path, _ = os.path.split(geoenrich.__file__)
    var_catalog = pd.read_csv(path + '/data/catalog.csv', index_col = 0).to_dict('index')

    for v in var_catalog:
        var_catalog[v]['var_id'] = v
        for domain in dap_creds:
            if domain in var_catalog[v]['url']:
                protocol, tail = var_catalog[v]['url'].split('://')
                creds = dap_creds[domain]['user'] + ':' + dap_creds[domain]['pw']
                var_catalog[v]['url'] = protocol + '://' + creds + '@' + tail

    
    return(var_catalog)



def create_nc(var):

    """
    Create empty netcdf file for requested variable for subsequent local storage.
    Same dimensions as the online dataset.

    Args:
        var (dict): Variable dictionary, as returned by :func:`geoenrich.satellite.get_var_catalog`.
    Returns:
        None
    """

    path = sat_path + var['var_id'] + '.nc'
    pathd = sat_path + var['var_id'] + '_downloaded.nc'

    remote_ds = nc.Dataset(var['url'])
    varname = var['varname']
    dimdict, var = get_metadata(remote_ds, varname)

    local_ds = nc.Dataset(path, mode = 'w')
    local_ds.set_fill_off()
    bool_ds = nc.Dataset(pathd, mode = 'w')

    for name, dimension in remote_ds.dimensions.items():
        local_ds.createDimension(name, len(dimension) if not dimension.isunlimited() else None)
        bool_ds.createDimension(name, len(dimension) if not dimension.isunlimited() else None)


    for name, variable in remote_ds.variables.items():
        if (name in dimdict) and (dimdict[name]['standard_name'] in ['time', 'latitude', 'longitude', 'depth']):
            local_ds.createVariable(name, variable.datatype, variable.dimensions, zlib= True)
            local_ds.variables[name].setncatts({k: variable.getncattr(k) for k in variable.ncattrs()})
            local_ds.variables[name][:] = variable[:]


    variable = remote_ds.variables[varname]
    local_ds.createVariable(varname, variable.dtype, variable.dimensions, zlib = True)
    local_ds.variables[varname].setncatts({k: variable.getncattr(k) for k in variable.ncattrs()})

    bool_ds.createVariable(varname, 'B', remote_ds.variables[varname].dimensions, zlib = True, fill_value = 0)

    local_ds.close()
    bool_ds.close()
    remote_ds.close()



def multidimensional_slice(nc_dataset, varname, ind, lons, lon_pos):

    """
    Return a slice from a dataset (can be local or remote).
    
    Args:
        nc_dataset (netCDF4.Dataset): Dataset to query.
        varname (str): Variable name in the dataset.
        ind (dict): Dictionary with ordered slicing indices for all dimensions.
        lons (list): Longitude values.
        lon_pos (int): Position of longitude in the dataset's dimensions.
    Returns:
        numpy.ma.MaskedArray: Requested data.
    """

    try:

        if ind[lon_pos]['min'] > ind[lon_pos]['max']:

            # Longitude singularity

            ind_part1, ind_part2 = deepcopy(ind), deepcopy(ind)
            ind_part1[lon_pos]['max'] = len(lons) - 1
            ind_part2[lon_pos]['min'] = 0

            part1 = multidimensional_slice(nc_dataset, varname, ind_part1, lons, lon_pos)
            part2 = multidimensional_slice(nc_dataset, varname, ind_part2, lons, lon_pos)

            data = np.ma.concatenate((part1, part2), axis = lon_pos)
            return(data)

        else:

            if len(ind) == 2:
                data = nc_dataset.variables[varname][ind[0]['min']:ind[0]['max']+1, ind[1]['min']:ind[1]['max']+1]
            elif len(ind) == 3:
                data = nc_dataset.variables[varname][ind[0]['min']:ind[0]['max']+1, ind[1]['min']:ind[1]['max']+1,
                                                     ind[2]['min']:ind[2]['max']+1]
            elif len(ind) == 4:
                data = nc_dataset.variables[varname][ind[0]['min']:ind[0]['max']+1, ind[1]['min']:ind[1]['max']+1,
                                                     ind[2]['min']:ind[2]['max']+1, ind[3]['min']:ind[3]['max']+1]
            else:
                print('Unsupported number of dimensions (only lat, lon, time and depth are supported')

            return(data)

    except:
        print('Corrupt netCDF file', ind)



def insert_multidimensional_slice(nc_dataset, varname, data, ind, lons, lon_pos):

    """
    Insert a slice into a local dataset.

    Args:
        nc_dataset (netCDF4.Dataset): Dataset to query.
        varname (str): Variable name in the dataset.
        data (numpy.array): Data to insert.
        ind (dict): Dictionary with ordered slicing indices for all dimensions.
        lons (list): Longitude values.
        lon_pos (int): Position of longitude in the dataset's dimensions.
    Returns:
        None
    """
    if ind[lon_pos]['min'] > ind[lon_pos]['max']:

        # Longitude singularity
        width1 = len(lons) - ind[lon_pos]['min']

        ind_part1, ind_part2 = deepcopy(ind), deepcopy(ind)
        ind_part1[lon_pos]['max'] = len(lons) - 1
        ind_part2[lon_pos]['min'] = 0

        part1, part2 = np.split(data, [width1], axis = lon_pos)

        insert_multidimensional_slice(nc_dataset, varname, part1, ind_part1, lons, lon_pos)
        insert_multidimensional_slice(nc_dataset, varname, part2, ind_part2, lons, lon_pos)

    else:
        if len(ind) == 2:
            nc_dataset.variables[varname][ind[0]['min']:ind[0]['max']+1, ind[1]['min']:ind[1]['max']+1] \
            = data
        elif len(ind) == 3:
            nc_dataset.variables[varname][ind[0]['min']:ind[0]['max']+1, ind[1]['min']:ind[1]['max']+1,
                                          ind[2]['min']:ind[2]['max']+1]                              \
            = data
        elif len(ind) == 4:
            nc_dataset.variables[varname][ind[0]['min']:ind[0]['max']+1, ind[1]['min']:ind[1]['max']+1,
                                          ind[2]['min']:ind[2]['max']+1, ind[3]['min']:ind[3]['max']+1] \
            = data
        else:
            print('Unsupported number of dimensions (only lat, lon, time and depth are supported')



def ellipsoid_mask(data, coords, center, geo_buff):

    """
    Calculate ellipsoid mask for the given point and data array.

    Args:
        data (numpy.array): data array as output by :func:`geoenrich.enrichment.fetch_data`.
        coords (list): Coordinates of the given data, as output by :func:`geoenrich.enrichment.fetch_data`.
        center (point): Occurrence point.
        geo_buff (int): Radius of the area of interest.
    Returns:
        numpy.array: Mask.
    """

    lat_dim = [c[0] for c in coords].index('latitude')
    lon_dim = [c[0] for c in coords].index('longitude')
    lats = coords[lat_dim][1]
    longs = coords[lon_dim][1]

    earth_radius = 6371
    y_radius = 180 * geo_buff / (np.pi * earth_radius)
    proj_earth_radius = np.sin(np.pi * (90 - abs(center.y)) / 180)
    
    x_radius = 180 * geo_buff / (np.pi * earth_radius * proj_earth_radius)


    Y, X = np.ogrid[:len(lats), :len(longs)]

    long_diff = np.minimum(360 - abs(longs[X]-center.x), abs(longs[X]-center.x))
    distance = np.sqrt(long_diff**2/x_radius**2 + (lats[Y]-center.y)**2/y_radius**2)

    mask2d = distance > 1
    mask = np.broadcast_to(mask2d, data.shape)

    return(mask)