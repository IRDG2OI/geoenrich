"""
The module that handles netCDF files
"""
import os

import pandas as pd
import netCDF4 as nc

from datetime import datetime
from cftime import num2pydate

import geoenrich

try:
    from geoenrich.credentials import *
except:
    from geoenrich.credentials_example import *
    print('Please rename credentials_example.py to credentials.py fill in the blanks')


def get_metadata(ds, varname):

    """
    Download and format useful metadata on dimensions and variables.
    Generate a dictionary where dimensions can be accessed both with their original name and their standard name (if available).
    For internal use.
    
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
    For internal use.

    Args:
        None
    Returns:
        dict: Dictionary with variable id, variable name in dataset, and dataset url
    """

    path, _ = os.path.split(geoenrich.__file__)
    var_catalog = pd.read_csv(path + 'catalog.csv', index_col = 0).to_dict('index')

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
    For internal use.

    Args:
        var (dict): Variable dictionary, as returned by get_var_catalog.
    Returns:
        None
    """

    path = sat_path + var['var_id'] + '.nc'
    pathd = sat_path + var['var_id'] + '_downloaded.nc'

    remote_ds = nc.Dataset(var['url'])
    varname = var['varname']
    dimdict, var = get_metadata(remote_ds, varname)

    local_ds = nc.Dataset(path, mode = 'w')
    bool_ds = nc.Dataset(pathd, mode = 'w')

    for name, dimension in remote_ds.dimensions.items():
        local_ds.createDimension(name, len(dimension) if not dimension.isunlimited() else None)
        bool_ds.createDimension(name, len(dimension) if not dimension.isunlimited() else None)


    for name, variable in remote_ds.variables.items():
        if (name in dimdict) and (dimdict[name]['standard_name'] in ['time', 'latitude', 'longitude', 'depth']):
            local_ds.createVariable(name, variable.datatype, variable.dimensions, zlib= True)
            local_ds.variables[name].setncatts({k: variable.getncattr(k) for k in variable.ncattrs()})
            local_ds.variables[name][:] = variable[:]


    local_ds.createVariable(varname, remote_ds.variables[varname].dtype,
                              remote_ds.variables[varname].dimensions, zlib = True)
    local_ds[varname].setncatts({k: remote_ds.variables[varname].getncattr(k) \
                            for k in remote_ds.variables[varname].ncattrs()})

    bool_ds.createVariable(varname, 'B', remote_ds.variables[varname].dimensions, zlib = True, fill_value = 0)

    local_ds.close()
    bool_ds.close()
    remote_ds.close()



def multidimensional_slice(nc_dataset, varname, ind):

    """
    Return a slice from a dataset (can be local or remote).
    For internal use.
    
    Args:
        nc_dataset (netCDF4.Dataset): Dataset to query.
        varname (str): Variable name in the dataset.
        ind (dict): Dictionary with ordered slicing indices for all dimensions.
    Returns:
        numpy.masked_array: Requested data.
    """

    data = None

    try:

        if len(ind) == 2:
            data = nc_dataset.variables[varname][ind[0]['min']:ind[0]['max'], ind[1]['min']:ind[1]['max']]
        elif len(ind) == 3:
            data = nc_dataset.variables[varname][ind[0]['min']:ind[0]['max'], ind[1]['min']:ind[1]['max'],
                                                 ind[2]['min']:ind[2]['max']]
        elif len(ind) == 4:
            data = nc_dataset.variables[varname][ind[0]['min']:ind[0]['max'], ind[1]['min']:ind[1]['max'],
                                                 ind[2]['min']:ind[2]['max'], ind[3]['min']:ind[3]['max']]
        else:
            print('Unsupported number of dimensions (only lat, lon, time and depth are supported')

        return(data)

    except:
        print(varname, ind)



def insert_multidimensional_slice(nc_dataset, varname, data, ind):

    """
    Insert a slice into a local dataset.
    For internal use.

    Args:
        nc_dataset (netCDF4.Dataset): Dataset to query.
        varname (str): Variable name in the dataset.
        data (numpy.array): Data to insert.
        ind (dict): Dictionary with ordered slicing indices for all dimensions.
    Returns:
        None
    """

    if len(ind) == 2:
        nc_dataset.variables[varname][ind[0]['min']:ind[0]['max'], ind[1]['min']:ind[1]['max']] \
        = data
    elif len(ind) == 3:
        nc_dataset.variables[varname][ind[0]['min']:ind[0]['max'], ind[1]['min']:ind[1]['max'],
                                      ind[2]['min']:ind[2]['max']]                              \
        = data
    elif len(ind) == 4:
        data = nc_dataset.variables[varname][ind[0]['min']:ind[0]['max'], ind[1]['min']:ind[1]['max'],
                                             ind[2]['min']:ind[2]['max'], ind[3]['min']:ind[3]['max']] \
        = data
    else:
        print('Unsupported number of dimensions (only lat, lon, time and depth are supported')


