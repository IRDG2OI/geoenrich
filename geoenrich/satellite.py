import os
from copy import deepcopy

import json
import numpy as np
import pandas as pd
import netCDF4 as nc
from pathlib import Path

from datetime import datetime
from cftime import num2date, num2pydate

import geoenrich

try:
    from geoenrich.credentials import *
except:
    from geoenrich.credentials_example import *
    print('Please rename credentials_example.py to credentials.py and fill in the root path and credentials, if needed')
    print('File location: ' + str(Path(geoenrich.__file__).with_name('credentials_example.py')))


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

        if getattr(ds.variables[name], 'standard_name', 'Unknown') == 'time' or name in ['time', 'time_agg']:

            cal = None
            if 'months since' in ds.variables[name].__dict__['units']:
                times = num2date(ds.variables[name][:], ds.variables[name].__dict__['units'], '360_day')
            else:
                if varname in ['uwnd', 'vwnd']:
                    times = num2pydate(ds.variables[name][:] - 725563, 'days since 1987-01-01 00:00:00')
                else:
                    times = num2pydate(ds.variables[name][:], ds.variables[name].__dict__['units'])
            times = pd.Series([datetime(*d.timetuple()[:-3]) for d in times])
            item = {'name': name, 'standard_name': 'time', 'vals': times, 'unit': None}
            dimdict[name] = item
            dimdict['time'] = item

        # Format lon & lat dimensions

        elif getattr(ds.variables[name], 'standard_name', 'Unknown') in ('longitude', 'latitude', 'depth'):

            item = {'name': name,
                    'standard_name': ds.variables[name].__dict__['standard_name'],
                    'vals': ds.variables[name][:],
                    'unit': getattr(ds.variables[name], 'units', 'Unknown')}
            dimdict[name] = item
            dimdict[ds.variables[name].standard_name] = item

        # Format requested variable

        elif name == varname:
            
            var = {'name':name,
                    'unit': getattr(ds.variables[name], 'units', 'Unknown'),
                    'params': [v.name for v in ds.variables[name].get_dims()],
                    'add_offset': getattr(ds.variables[name], 'add_offset', 0),
                    'scale_factor': getattr(ds.variables[name], 'scale_factor', 1),
                    'derived_from': getattr(ds.variables[name], 'derived_from', [])}

            if 'standard_name' in ds.variables[name].__dict__:
                var['standard_name'] = ds.variables[name].__dict__['standard_name']
        

        # Search for latitude and longitude in case standard names were not provided

        elif name in ['lat', 'latitude']:

            item = {'name': name,
                    'standard_name': 'latitude',
                    'vals': ds.variables[name][:],
                    'unit': getattr(ds.variables[name], 'units', 'Unknown')}
            dimdict[name] = item
            dimdict['latitude'] = item

        elif name in ['lon', 'longitude']:

            item = {'name': name,
                    'standard_name': 'longitude',
                    'vals': ds.variables[name][:],
                    'unit': getattr(ds.variables[name], 'units', 'Unknown')}
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

    path = Path(geoenrich.__file__).parent
    var_catalog = pd.read_csv(path / 'data' / 'catalog.csv', index_col = 0).to_dict('index')

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

    path = Path(sat_path, var['var_id'] + '.nc')
    pathd = Path(sat_path, var['var_id'] + '_downloaded.nc')

    remote_ds = nc.Dataset(var['url'])
    varname = var['varname']
    dimdict, var = get_metadata(remote_ds, varname)

    local_ds = nc.Dataset(str(path), mode = 'w')
    local_ds.set_fill_off()
    bool_ds = nc.Dataset(str(pathd), mode = 'w')

    for name, dimension in remote_ds.dimensions.items():
        if getattr(remote_ds.variables[name], 'standard_name', 'Unknown') == 'time' or name in ['time', 'time_agg']:
            local_ds.createDimension(name, None)
            bool_ds.createDimension(name, None)
        else:
            local_ds.createDimension(name, len(dimension))
            bool_ds.createDimension(name, len(dimension))


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



def create_nc_calculated(var_id):

    """
    Create empty netcdf file for requested variable for subsequent local storage.
    Same dimensions as the online dataset.

    Args:
        var_id (str): ID of the variable to calculate.
    Returns:
        None
    """

    calculated =    { 'eke':{   'long_name':    'Eddy kinetic energy',
                                'unit':         'm2/s2',
                                'derived_from': ['geos-current-u', 'geos-current-v']}
                    }

    var_meta = calculated[var_id]

    path = Path(sat_path, var_id + '.nc')
    pathd = Path(sat_path, var_id + '_downloaded.nc')

    var_catalog = get_var_catalog()
    like_ds = nc.Dataset(str(Path(sat_path, var_meta['derived_from'][0] + '.nc')))
    like_varname = var_catalog[var_meta['derived_from'][0]]['varname']
    dimdict, var = get_metadata(like_ds, like_varname)

    local_ds = nc.Dataset(str(path), mode = 'w')
    local_ds.set_fill_off()
    bool_ds = nc.Dataset(str(pathd), mode = 'w')

    for name, dimension in like_ds.dimensions.items():
        local_ds.createDimension(name, len(dimension) if not dimension.isunlimited() else None)
        bool_ds.createDimension(name, len(dimension) if not dimension.isunlimited() else None)


    for name, variable in like_ds.variables.items():
        if (name in dimdict) and (dimdict[name]['standard_name'] in ['time', 'latitude', 'longitude', 'depth']):
            local_ds.createVariable(name, variable.datatype, variable.dimensions, zlib= True)
            local_ds.variables[name].setncatts({k: variable.getncattr(k) for k in variable.ncattrs()})
            local_ds.variables[name][:] = variable[:]


    like_variable = like_ds.variables[like_varname]
    local_ds.createVariable(var_id, like_variable.dtype, like_variable.dimensions, zlib = True)
    local_ds.variables[var_id].setncatts({'units': var_meta['unit'], 'long_name': var_meta['long_name']})
    local_ds.variables[var_id].setncatts({'derived_from': var_meta['derived_from']})

    bool_ds.createVariable(var_id, 'B', like_variable.dimensions, zlib = True, fill_value = 0)

    local_ds.close()
    bool_ds.close()
    like_ds.close()




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
            ind_part2[lon_pos]['min'] = len(lons) % ind[lon_pos]['step']

            part1 = multidimensional_slice(nc_dataset, varname, ind_part1, lons, lon_pos)
            part2 = multidimensional_slice(nc_dataset, varname, ind_part2, lons, lon_pos)

            data = np.ma.concatenate((part1, part2), axis = lon_pos)
            return(data)

        else:

            if len(ind) == 2:
                data = nc_dataset.variables[varname][ind[0]['min']:ind[0]['max']+1:ind[0]['step'],
                                                     ind[1]['min']:ind[1]['max']+1:ind[1]['step']]
            elif len(ind) == 3:
                data = nc_dataset.variables[varname][ind[0]['min']:ind[0]['max']+1:ind[0]['step'],
                                                     ind[1]['min']:ind[1]['max']+1:ind[1]['step'],
                                                     ind[2]['min']:ind[2]['max']+1:ind[2]['step']]
            elif len(ind) == 4:
                data = nc_dataset.variables[varname][ind[0]['min']:ind[0]['max']+1:ind[0]['step'],
                                                     ind[1]['min']:ind[1]['max']+1:ind[1]['step'],
                                                     ind[2]['min']:ind[2]['max']+1:ind[2]['step'],
                                                     ind[3]['min']:ind[3]['max']+1:ind[3]['step']]
            else:
                print('Unsupported number of dimensions (only lat, lon, time and depth are supported')

            return(data)

    except:
        print('Read error in netCDF file. If local file, try restoring a backup', ind)



def insert_multidimensional_slice(nc_dataset, varname, data, ind, lons, lon_pos):

    """
    Insert a slice into a local dataset.

    Args:
        nc_dataset (netCDF4.Dataset): Dataset to query.
        varname (str): Variable name in the dataset.
        data (numpy.array): Data to insert.
        ind (dict list): List with ordered slicing indices for all dimensions.
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
        ind_part2[lon_pos]['min'] = len(lons) % ind[lon_pos]['step']

        part1, part2 = np.split(data, [width1], axis = lon_pos)

        insert_multidimensional_slice(nc_dataset, varname, part1, ind_part1, lons, lon_pos)
        insert_multidimensional_slice(nc_dataset, varname, part2, ind_part2, lons, lon_pos)

    else:
        if len(ind) == 2:
            nc_dataset.variables[varname][ind[0]['min']:ind[0]['max']+1:ind[0]['step'],
                                          ind[1]['min']:ind[1]['max']+1:ind[1]['step']]   = data
        elif len(ind) == 3:
            nc_dataset.variables[varname][ind[0]['min']:ind[0]['max']+1:ind[0]['step'],
                                          ind[1]['min']:ind[1]['max']+1:ind[1]['step'],
                                          ind[2]['min']:ind[2]['max']+1:ind[2]['step']]   = data
        elif len(ind) == 4:
            nc_dataset.variables[varname][ind[0]['min']:ind[0]['max']+1:ind[0]['step'],
                                          ind[1]['min']:ind[1]['max']+1:ind[1]['step'],
                                          ind[2]['min']:ind[2]['max']+1:ind[2]['step'],
                                          ind[3]['min']:ind[3]['max']+1:ind[3]['step']]   = data
        else:
            print('Unsupported number of dimensions (only lat, lon, time and depth are supported')



def ellipsoid_mask(data, coords, center, geo_buff):

    """
    Calculate ellipsoid mask for the given point and data array.

    Args:
        data (numpy.array): Data array as output by :func:`geoenrich.exports.fetch_data`.
        coords (list): Coordinates of the given data, as output by :func:`geoenrich.exports.fetch_data`.
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



class NpEncoder(json.JSONEncoder):

    """
    Custom encoder to handle numpy data formats in json dump.
    """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)



def dump_metadata(var_id, out_path=Path('./')):

    """
    Dump metadata related to the given variable.

    Args:
        var_id (str): ID of the variable to retrieve metadata for.
        out_path (str or pathlib.Path): Path where metadata file will be saved.

    Returns:
        None
    """

    var = get_var_catalog()[var_id]
    ds_path = Path(sat_path, var['var_id'] + '.nc')

    ds = nc.Dataset(ds_path)
    variable = ds[var['varname']]
    args = {k: variable.getncattr(k) for k in variable.ncattrs()}

    write_path = Path(out_path, f'{var_id}_metadata.json')
    with (write_path).open('w') as f:
        json.dump(args, f, ensure_ascii=False, indent=4, cls=NpEncoder)

    print(f"File saved at {write_path}")