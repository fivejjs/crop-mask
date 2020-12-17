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

# # Extracting training data from the ODC <img align="right" src="../../Supplementary_data/DE_Africa_Logo_Stacked_RGB_small.jpg">
#
# * **Products used:** 
# [s2_l2a](https://explorer.digitalearth.africa/s2_l2a)
#

# ## Description
# This notebook will extract training data over Eastern Africa using geometries within a shapefile (or geojson). To do this, we rely on a custom `deafrica-sandbox-notebooks` function called `collect_training_data`, contained within the [deafrica_classificationtools](../Scripts/deafrica_classificationtools.py) script.
#
# 1. Import, and preview our training data contained in the file: `'data/Eastern_training_data_20201215.geojson'`
# 2. Extract training data from the datacube using a custom defined feature layer function that we can pass to `collect_training_data`. The training data function is stored in the python file `feature_layer_functions.py` - the functions are stored in a seperate file simply to keep this notebook tidy.
#
#     - **The features used to create the cropland mask are as follows:**
#         - For two seasons, January to June, and July to Decemeber:
#             - A geomedian composite of nine Sentinel-2 spectral bands
#             - Three measures of median absolute deviation
#             - NDVI, MNDWI, and LAI
#             - Cumulative Rainfall from CHIRPS
#             - Slope from SRTM (not seasonal, obviously)
#           
#           
# 3. Seperate the coordinate values in the returned training data from step 2, and export the coordinates as a text file.
# 4. Export the remaining training data (features other than coordinates) to disk as a text file for use in subsequent scripts
#
#
#
# ***

# ## Getting started
#
# To run this analysis, run all the cells in the notebook, starting with the "Load packages" cell. 

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

# ## Analysis parameters
#
# * `path`: The path to the input shapefile from which we will extract training data.
# * `field`: This is the name of column in your shapefile attribute table that contains the class labels. **The class labels must be integers**
#

path = 'data/Eastern_training_data_20201215.geojson' 
field = 'Class'

# ### Automatically find the number of cpus
#
# > **Note**: With supervised classification, its common to have many, many labelled geometries in the training data. `collect_training_data` can parallelize across the geometries in order to speed up the extracting of training data. Setting `ncpus>1` will automatically trigger the parallelization, however, its best to set `ncpus=1` to begin with to assist with debugging before triggering the parallelization. 

# +
try:
    ncpus = int(float(sp.getoutput('env | grep CPU')[-4:]))
except:
    ncpus = int(float(sp.getoutput('env | grep CPU')[-3:]))

print('ncpus = '+str(ncpus))
# -

# ## Load & preview polygon data
#
# We can load and preview our input data shapefile using `geopandas`. The shapefile should contain a column with class labels (e.g. 'class'). These labels will be used to train our model. 
#
# > Remember, the class labels **must** be represented by `integers`.
#

# +
# Load input data shapefile
input_data = gpd.read_file(path)

# Plot first five rows
input_data.head()

# +
# Plot training data in an interactive map
# map_shapefile(input_data, attribute=field)
# -

# Now, we can pass this shapefile to `collect_training_data`.  For each of the geometries in our shapefile we will extract features in accordance with the function `feature_layer_functions.gm_mads_two_seasons_training`. These will include:
#
# For two seasons, January to June, and July to Decemeber:
# - A geomedian composite of nine Sentinel-2 spectral bands
# - Three measures of median absolute deviation
# - NDVI, MNDWI, and LAI
# - Cumulative Rainfall from the CHIRPS
# - Slope from SRTM

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
# -

#generate a new datacube query object
query = {
    'time': time,
    'measurements': measurements,
    'resolution': resolution,
    'output_crs': output_crs,
    'group_by' : 'solar_day',
}

# ## Extract training data
#
# > Remember, if running this function for the first time, its advisable to set `ncpus=1` to assist with debugging before triggering the parallelization (which won't return errors if something is not working correctly).  You can also limit the number of polygons to run for the first time by passing in `gdf=input_data[0:5]`, for example.

# %%time
column_names, model_input = collect_training_data(
                                    gdf=input_data,
                                    products=products,
                                    dc_query=query,
                                    ncpus=25,
                                    return_coords=return_coords,
                                    field=field,
                                    zonal_stats=zonal_stats,
                                    custom_func=gm_mads_two_seasons_training,
                                    fail_threshold=0.015,
                                    max_retries=4
                                    )

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
