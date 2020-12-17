#!/usr/bin/env python3
# coding: utf-8

import sys
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


path = 'data/Eastern_training_data_20201215.geojson'
field = 'Class'

try:
    ncpus = int(float(sp.getoutput('env | grep CPU')[-4:]))
except:
    ncpus = int(float(sp.getoutput('env | grep CPU')[-3:]))

print('ncpus = '+str(ncpus))

input_data = gpd.read_file(path)

# Plot first five rows
input_data.head()


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


# In[6]:


#generate a new datacube query object
query = {
    'time': time,
    'measurements': measurements,
    'resolution': resolution,
    'output_crs': output_crs,
    'group_by' : 'solar_day',
}


column_names, model_input = collect_training_data(gdf=input_data[:2],
                                                  products=products,
                                                  dc_query=query,
                                                  ncpus=1,
                                                  return_coords=return_coords,
                                                  field=field,
                                                  zonal_stats=zonal_stats,
                                                  custom_func=gm_mads_two_seasons_training,
                                                  fail_threshold=0.015,
                                                  max_retries=4)


print(column_names)
print('')
print(np.array_str(model_input, precision=2, suppress_small=True))



coordinates_filename = "results/training_data/training_data_coordinates_20201217.txt"


# In[12]:


coord_variables = ['x_coord', 'y_coord']
model_col_indices = [column_names.index(var_name) for var_name in coord_variables]

np.savetxt(coordinates_filename, model_input[:, model_col_indices])



#set the name and location of the output file
output_file = "results/training_data/gm_mads_two_seasons_training_data_20201217.txt"

#grab all columns except the x-y coords
model_col_indices = [column_names.index(var_name) for var_name in column_names[0:-2]]
#Export files to disk
np.savetxt(output_file, model_input[:, model_col_indices], header=" ".join(column_names[0:-2]), fmt="%4f")

