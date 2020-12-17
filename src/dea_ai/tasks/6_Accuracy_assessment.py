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

# # Accuracy assessment of the Eastern Africa Cropland Mask<img align="right" src="../../Supplementary_data/DE_Africa_Logo_Stacked_RGB_small.jpg">
#
#

# ## Description
#
# Now that we have run classifications for the Eastern Africa AEZ, its time to conduct an accuracy assessment. The data used for assessing the accuracy was collected previously and set aside. Its stored in the data/ folder: `data/Validation_samples.shp` 
#
# This notebook will output a `confusion error matrix` containing Overall, Producer's, and User's accuracy, along with the F1 score for each class.

# ***
# ## Getting started
#
# To run this analysis, run all the cells in the notebook, starting with the "Load packages" cell. 

# ### Load Packages

# +
import os
import sys
import glob
import rasterio
import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import geopandas as gpd
from sklearn.metrics import f1_score

sys.path.append('../Scripts')
from deafrica_spatialtools import zonal_stats_parallel
# -

# ## Analysis Parameters
#
# * `pred_tif` : a binary classification of crop/no-crop output by the ML script.
# * `grd_truth` : a shapefile containing crop/no-crop points to serve as the "ground-truth" dataset
# * `aez_region` : a shapefile used to limit the ground truth points to the region where the model has classified crop/non-crop
#

pred_tif = 'results/classifications/predicted/20201215/mosaic.tif'
grd_truth = '../pre-post_processing/data/training_validation/GFSAD2015/cropland_prelim_validation_GFSAD.shp'
aez_region = 'data/Eastern.shp'

# ### Load the datasets
#
# `Ground truth points`

#ground truth shapefile
ground_truth = gpd.read_file(grd_truth).to_crs('EPSG:6933')

# rename the class column to 'actual'
ground_truth = ground_truth.rename(columns={'class':'Actual'})
ground_truth.head()

# Clip ground_truth data points to the simplified AEZ

#open shapefile
aez=gpd.read_file(aez_region).to_crs('EPSG:6933')
# clip points to region
ground_truth = gpd.overlay(ground_truth,aez, how='intersection')

# ### Convert points into polygons
#
# When the validation data was collected, 40x40m polygons were evaluated as either crop/non-crop rather than points, so we want to sample the raster using the same small polygons. We'll find the majority or 'mode' statistic within the polygon and use that to compare with the validation dataset.
#

# +
#set radius (in metres) around points
radius = 20

#convert to equal area to set polygon size in metres
ground_truth = ground_truth

#create circle buffer around points, then find envelope
ground_truth['geometry'] = ground_truth['geometry'].buffer(radius).envelope

#export to file for use in zonal-stats
ground_truth.to_file(grd_truth[:-4]+"_poly.shp")
# -

# ### Calculate zonal statistics
#
# We want to know what the majority pixel value is inside each validation polygon.

# +
zonal_stats_parallel(shp=grd_truth[:-4]+"_poly.shp",
                    raster=pred_tif,
                    statistics=['majority'],
                    out_shp=grd_truth[:-4]+"_poly.shp",
                    ncpus=2,
                    nodata=-999)

#read in the results
x=gpd.read_file(grd_truth[:-4]+"_poly.shp")

#add result to original ground truth array
ground_truth['Prediction'] = x['majority'].astype(np.int16)

#Remove the temporary shapefile we made
[os.remove(i) for i in glob.glob(grd_truth[:-4]+"_poly"+'*')]
# -

# ---
#
# ## Create a confusion matrix

# +
confusion_matrix = pd.crosstab(ground_truth['Actual'],
                               ground_truth['Prediction'],
                               rownames=['Actual'],
                               colnames=['Prediction'],
                               margins=True)

confusion_matrix
# -

# ### Calculate User's and Producer's Accuracy

# `Producer's Accuracy`

confusion_matrix["Producer's"] = [confusion_matrix.loc[0, 0] / confusion_matrix.loc[0, 'All'] * 100,
                              confusion_matrix.loc[1, 1] / confusion_matrix.loc[1, 'All'] * 100,
                              np.nan]

# `User's Accuracy`

# +
users_accuracy = pd.Series([confusion_matrix[0][0] / confusion_matrix[0]['All'] * 100,
                                confusion_matrix[1][1] / confusion_matrix[1]['All'] * 100]
                         ).rename("User's")

confusion_matrix = confusion_matrix.append(users_accuracy)
# -

# `Overall Accuracy`

confusion_matrix.loc["User's","Producer's"] = (confusion_matrix.loc[0, 0] + 
                                                confusion_matrix.loc[1, 1]) / confusion_matrix.loc['All', 'All'] * 100

# `F1 Score`
#
# The F1 score is the harmonic mean of the precision and recall, where an F1 score reaches its best value at 1 (perfect precision and recall), and is calculated as:
#
# $$
# \begin{aligned}
# \text{Fscore} = 2 \times \frac{\text{UA} \times \text{PA}}{\text{UA} + \text{PA}}.
# \end{aligned}
# $$
#
# Where UA = Users Accuracy, and PA = Producer's Accuracy

# +
fscore = pd.Series([(2*(confusion_matrix.loc["User's", 0]*confusion_matrix.loc[0, "Producer's"]) / (confusion_matrix.loc["User's", 0]+confusion_matrix.loc[0, "Producer's"])) / 100,
                    f1_score(ground_truth['Actual'].astype(np.int8), ground_truth['Prediction'].astype(np.int8), average='binary')]
                         ).rename("F-score")

confusion_matrix = confusion_matrix.append(fscore)
# -

# ### Tidy Confusion Matrix
#
# * Limit decimal places,
# * Add readable class names
# * Remove non-sensical values 

# round numbers
confusion_matrix = confusion_matrix.round(decimals=2)

# rename booleans to class names
confusion_matrix = confusion_matrix.rename(columns={0:'Non-crop', 1:'Crop', 'All':'Total'},
                                            index={0:'Non-crop', 1:'Crop', 'All':'Total'})

#remove the nonsensical values in the table
confusion_matrix.loc["User's", 'Total'] = '--'
confusion_matrix.loc['Total', "Producer's"] = '--'
confusion_matrix.loc["F-score", 'Total'] = '--'
confusion_matrix.loc["F-score", "Producer's"] = '--'

confusion_matrix

# ### Export csv

confusion_matrix.to_csv('results/Eastern_confusion_matrix.csv')

# ## Next steps
#
# This is the last notebook in the `Eastern Africa Cropland Mask` workflow! To revist any of the other notebooks, use the links below.
#
# 1. [Extracting_training_data](1_Extracting_training_data.ipynb) 
# 2. [Inspect_training_data](2_Inspect_training_data.ipynb)
# 3. [Train_fit_evaluate_classifier](3_Train_fit_evaluate_classifier.ipynb)
# 4. [Predict](4_Predict.ipynb)
# 5. [Object-based_filtering](5_Object-based_filtering.ipynb)
# 6. **Accuracy_assessment (this notebook)**

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
