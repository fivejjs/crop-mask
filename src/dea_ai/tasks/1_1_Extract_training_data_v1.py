# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# ### Load packages
#

# +
# %matplotlib inline

import sys
import os
import warnings
import datacube
import numpy as np
import xarray as xr
import subprocess as sp
import geopandas as gpd
from datacube.utils.geometry import assign_crs
from datacube.utils.rio import configure_s3_access
configure_s3_access(aws_unsigned=True, cloud_defaults=True)

#import deafrica specific functions
sys.path.append('../Scripts')
from deafrica_plotting import map_shapefile
from deafrica_classificationtools import collect_training_data 

#import the custom feature layer functions
from feature_layer_functions import gm_mads_two_seasons_training

warnings.filterwarnings("ignore")
# -

path = 'data/Eastern_training_data_20201215.geojson' 
field = 'Class'

# +
from dask.distributed import Client, LocalCluster

cluster = LocalCluster(threads_per_worker=1, n_workers=25, memory_limit=400e6)
client = Client(cluster)
client
# -

# Load input data shapefile
gdf = gpd.read_file(path)

gdf.head()

# First, we need to set up a few extra inputs for `collect_training_data` and the datacube.  See the function docs [here](https://github.com/digitalearthafrica/deafrica-sandbox-notebooks/blob/03b7b41d5f6526ff3f33618f7a0b48c0d10a155f/Scripts/deafrica_classificationtools.py#L650) for more information on these parameters.
#
#

# +
#set up our inputs to collect_training_data
zonal_stats = 'median'
return_coords = True

# Set up the inputs for the ODC query
products = ['s2_l2a']
time = ('2019-01', '2019-12')
measurements = [
    'red', 'blue', 'green', 'nir', 'swir_1', 'swir_2', 'red_edge_1',
    'red_edge_2', 'red_edge_3'
]
resolution = (-20, 20)
output_crs = 'epsg:6933'

#generate a new datacube query object
query = {
    'time': time,
    'measurements': measurements,
    'resolution': resolution,
    'output_crs': output_crs,
    'group_by' : 'solar_day',
}

query
# -

# %%time
for index, row in enumerate(gdf.itertuples()):
    pass

row.geometry

# %%time
for index, row in gdf.iterrows():
    pass
# CPU times: user 243 ms, sys: 4.01 ms, total: 247 ms
# Wall time: 247 ms

custom_func = gm_mads_two_seasons_training

# +
# results = []
# column_names = []

# # loop through polys and extract training data
# _get_training_data_for_shp(gdf, index, row, results, column_names,
#                                products, dc_query, return_coords=True,
#                                custom_func, # gm_mads_two_seasons_training
#                               field, # class label
#                                calc_indices, # None
#                                reduce_func, drop, zonal_stats)

# -

configure_s3_access(aws_unsigned=True, cloud_defaults=True)

from datacube.utils import geometry

# set up query based on polygon (convert to WGS84)
geom = geometry.Geometry(gdf.geometry.values[index],
                         geometry.CRS('epsg:4326'))

type(gdf.geometry.values[index]), type(row.geometry)

type(geom)

# +
# print(geom)
q = {"geopolygon": geom}

# merge polygon query with user supplied query params
query.update(q)

query['dask_chunk'] = {'time': -1, 'x': 10000, 'y': 1000}
# -

query

custom_func

from deafrica_datahandling import mostcommon_crs, load_ard

# +
# ds = load_ard(dc=dc, products=products, **query)
# -

measurements

# +
product_type = 's2'
fmask_band = 'SCL'

measurements.append(fmask_band)
# 
data_bands = [band for band in measurements if band not in (fmask_band)]
mask_bands = [band for band in measurements if band not in data_bands]
# -

data_bands, mask_bands

query = {'time': ('2019-01', '2019-06'),
 'measurements': ['red',
  'blue',
  'green',
  'nir',
  'swir_1',
  'swir_2',
  'red_edge_1',
  'red_edge_2',
  'red_edge_3'],
 'resolution': (-20, 20),
 'output_crs': 'epsg:6933',
 'group_by': 'solar_day'}
 #'geopolygon': Geometry(POLYGON ((31.65112326883945 -4.243618921933441, 31.6515378355519 -4.243618921933441, 31.6515378355519 -4.243304537816352, 31.65112326883945 -4.243304537816352, 31.65112326883945 -4.243618921933441)), epsg:4326),
 #'dask_chunk': {'time': -1, 'x': 10000, 'y': 1000}}

# ## find_datasets too slow, one thread ?

# +
# dataset_list = []
# datasets = dc.find_datasets(product=products, **query)
# dataset_list.extend(datasets)
# -

len(dataset_list)

dataset_list[0]

# +
import pickle

with open('intermediate/dataset_list.pkl', 'rb') as fh:
    dataset_list = pickle.load(fh)
# -

type(dataset_list), len(dataset_list)

pq_categories_s2=['vegetation','snow or ice',
                               'water','bare soils',
                               'unclassified', 'dark area pixels']

dc = datacube.Datacube(app='training_data')

# ## loading too slow, one thread ?

# +
# %%time
dask_chunks = {'time': -1, 'x': 10000, 'y': 10000}

ds = dc.load(datasets=dataset_list,
#              dask_chunks=dask_chunks, 
             **query)
# -

ds

dir()

# !mkdir intermediate

# %%time
ds.load()


ds

ds.to_zarr('intermediate/data_retrieved.zarr', 'w')

# +
# sentinel 2                     
if product_type == 's2':
    # product_type is s2
    # currently broken for mask band values >=8
    # pq_mask = odc.algo.fmask_to_bool(ds[fmask_band],
    #                             categories=pq_categories_s2)
    flags_s2 = dc.list_measurements().loc[products[0]].loc[fmask_band]['flags_definition']['qa']['values']
    pq_mask = ds[fmask_band].isin([int(k) for k, v in flags_s2.items() if v in pq_categories_s2])


###############
# Apply masks #
###############

# Generate good quality data mask
mask = None
if mask_pixel_quality:
    print('Applying pixel quality/cloud mask')
    mask = pq_mask

ds_data = ds[data_bands]
ds_masks = ds[mask_bands]

# Mask data if either of the above masks were generated
if mask is not None:
    ds_data = odc.algo.keep_good_only(ds_data, where=mask)

# Automatically set dtype to either native or float32 depending
# on whether masking was requested
if dtype == 'auto':
    dtype = 'native' if mask is None else 'float32'

# Set nodata values using odc.algo tools to reduce peak memory
# use when converting data dtype    
if dtype != 'native':
    ds_data = odc.algo.to_float(ds_data, dtype=dtype)


attrs = ds.attrs
ds = xr.merge([ds_data, ds_masks])
ds.attrs.update(attrs)

###############
# Return data #
###############

# Drop bands not originally requested by user
requested_measurements = measurements
if requested_measurements:
    ds = ds[requested_measurements]


# If user supplied dask_chunks, return data as a dask array without
# actually loading it in
if dask_chunks is not None:
    # skipped
    print(f'Returning {len(ds.time)} time steps as a dask array')
    return ds
else:
    print(f'Loading {len(ds.time)} time steps')
    # TODO: only use lazay, remove compute() below
    return ds.compute()

# +
# remaining of _get_training_data_for_shp
# create polygon mask
with HiddenPrints():
    mask = xr_rasterize(gdf.iloc[[index]], ds)

# Use custom function for training data if it exists
if custom_func is not None:
    with HiddenPrints():
        data = custom_func(ds)
        data = data.where(mask)

else:
    # mask dataset
    ds = ds.where(mask)
    # first check enough variables are set to run functions
    if (len(ds.time.values) > 1) and (reduce_func == None):
        raise Exception(
            "You're dataset has " + str(len(ds.time.values)) +
            " time-steps, please provide a time reduction function," +
            " e.g. reduce_func='mean'")

    if calc_indices is not None:
        # determine which collection is being loaded
        if 'level2' in products[0]:
            collection = 'c2'
        elif 'gm' in products[0]:
            collection = 'c2'
        elif 'sr' in products[0]:
            collection = 'c1'
        elif 's2' in products[0]:
            collection = 's2'

        if len(ds.time.values) > 1:

            if reduce_func in ['mean', 'median', 'std', 'max', 'min']:
                with HiddenPrints():
                    data = calculate_indices(ds,
                                             index=calc_indices,
                                             drop=drop,
                                             collection=collection)
                    # getattr is equivalent to calling data.reduce_func
                    method_to_call = getattr(data, reduce_func)
                    data = method_to_call(dim='time')

            elif reduce_func == 'geomedian':
                data = GeoMedian().compute(ds)
                with HiddenPrints():
                    data = calculate_indices(data,
                                             index=calc_indices,
                                             drop=drop,
                                             collection=collection)

            else:
                raise Exception(
                    reduce_func + " is not one of the supported" +
                    " reduce functions ('mean','median','std','max','min', 'geomedian')"
                )

        else:
            with HiddenPrints():
                data = calculate_indices(ds,
                                         index=calc_indices,
                                         drop=drop,
                                         collection=collection)

    # when band indices are not required, reduce the
    # dataset to a 2d array through means or (geo)medians
    if calc_indices is None:

        if len(ds.time.values) > 1:

            if reduce_func == 'geomedian':
                data = GeoMedian().compute(ds)

            elif reduce_func in ['mean', 'median', 'std', 'max', 'min']:
                method_to_call = getattr(ds, reduce_func)
                data = method_to_call('time')
        else:
            data = ds.squeeze()

if return_coords == True:
    # turn coords into a variable in the ds
    data['x_coord'] = ds.x + 0 * ds.y
    data['y_coord'] = ds.y + 0 * ds.x

if zonal_stats is None:
    # If no zonal stats were requested then extract all pixel values
    flat_train = sklearn_flatten(data)
    flat_val = np.repeat(row[field], flat_train.shape[0])
    stacked = np.hstack((np.expand_dims(flat_val, axis=1), flat_train))

elif zonal_stats in ['mean', 'median', 'std', 'max', 'min']:
    method_to_call = getattr(data, zonal_stats)
    flat_train = method_to_call()
    flat_train = flat_train.to_array()
    stacked = np.hstack((row[field], flat_train))

else:
    raise Exception(zonal_stats + " is not one of the supported" +
                    " reduce functions ('mean','median','std','max','min')")

#return unique-id so we can index if dc.load fails silently
_id = gdf.iloc[index]['id']

# Append training data and labels to list
out_arrs.append(np.append(stacked, _id))
out_vars.append([field] + list(data.data_vars) + ['id'])
# -

print(column_names)
print('')
print(np.array_str(model_input, precision=2, suppress_small=True))

# ## Seperate the coordinates
#
# By setting `return_coords=True` in the `collect_training_data` function, our training data now has two extra columns called `x_coord` and `y_coord`.  We need to seperate these from our training dataset as they will not be used to train the machine learning model. Instead, these variables will be used to help conduct Spatial K-fold Cross validation (SKVC) in the notebook `3_Train_fit_evaluate_classifier`.  For more information on why this is important, see this [article](https://www.tandfonline.com/doi/abs/10.1080/13658816.2017.1346255?journalCode=tgis20).

coordinates_filename = "results/training_data/training_data_coordinates_20201217.txt"

# +
coord_variables = ['x_coord', 'y_coord']
model_col_indices = [column_names.index(var_name) for var_name in coord_variables]

np.savetxt(coordinates_filename, model_input[:, model_col_indices])
# -

# ## Export training data
#
# Once we've collected all the training data we require, we can write the data to disk. This will allow us to import the data in the next step(s) of the workflow.
#

#set the name and location of the output file
output_file = "results/training_data/gm_mads_two_seasons_training_data_20201217.txt"

#grab all columns except the x-y coords
model_col_indices = [column_names.index(var_name) for var_name in column_names[0:-2]]
#Export files to disk
np.savetxt(output_file, model_input[:, model_col_indices], header=" ".join(column_names[0:-2]), fmt="%4f")

# ## Next steps
#
# To continue working through the notebooks in this `Eastern Africa Cropland Mask` workflow, go to the next notebook `2_Inspect_training_data.ipynb`.
#
# 1. **Extracting_training_data (this notebook)** 
# 2. [Inspect_training_data](2_Inspect_training_data.ipynb)

# ***
#
# ## Additional information
#
# **License:** The code in this notebook is licensed under the [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0). 
# Digital Earth Africa data is licensed under the [Creative Commons by Attribution 4.0](https://creativecommons.org/licenses/by/4.0/) license.
#
# **Contact:** If you need assistance, please post a question on the [Open Data Cube Slack channel](http://slack.opendatacube.org/) or on the [GIS Stack Exchange](https://gis.stackexchange.com/questions/ask?tags=open-data-cube) using the `open-data-cube` tag (you can view previously asked questions [here](https://gis.stackexchange.com/questions/tagged/open-data-cube)).
# If you would like to report an issue with this notebook, you can file one on [Github](https://github.com/digitalearthafrica/deafrica-sandbox-notebooks).
#
# **Last modified:** Dec 2020
#
