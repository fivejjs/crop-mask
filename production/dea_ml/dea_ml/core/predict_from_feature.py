import datetime
import json
import logging
import math
import os
import os.path as osp
import uuid
from typing import Optional, Dict, List

import psutil
import xarray as xr
from datacube.utils.cog import write_cog
from datacube.utils.dask import start_local_dask
from datacube.utils.geometry import GeoBox
from datacube.utils.geometry import assign_crs
from datacube.utils.rio import configure_s3_access

from dea_ml.core.feature_layer import get_xy_from_task
from dea_ml.core.stac_to_dc import StacIntoDc

from deafrica_tools.classification import predict_xr
from distributed import Client
from odc.io.cgroups import get_cpu_quota, get_mem_quota
from odc.stats._cli_common import setup_logging


def get_max_mem() -> int:
    """
    Max available memory, takes into account pod resource allocation
    """
    total = psutil.virtual_memory().total
    mem_quota = get_mem_quota()
    if mem_quota is None:
        return total
    return min(mem_quota, total)


def get_max_cpu() -> int:
    """
    Max available CPU (rounded up if fractional), takes into account pod
    resource allocation
    """
    ncpu = get_cpu_quota()
    if ncpu is not None:
        return int(math.ceil(ncpu))
    return psutil.cpu_count()


def predict_with_model(
    training_features: List[str],
    model,
    data: xr.Dataset,
    chunk_size: Dict = None
    urls: Dict[Any, Any],
) -> xr.Dataset:
    """
    run the prediction here
    """
    # step 1: select features

    # load the column_names to ensure
    # the bands are in the right order
    with open(urls['td'], 'r') as file:
        header = file.readline()
    
    column_names = header.split()[1:][1:]
    
    input_data = data[column_names]
    
    print('!!!!input_data!!!!!!!')
    print(input_data)
    # step 2: prediction
    predicted = predict_xr(
        model,
        input_data,
        chunk_size=chunk_size,
        clean=True,
        proba=True,
        return_input=True,
    )

    predicted["Predictions"] = predicted["Predictions"].astype("uint8")
    predicted["Probabilities"] = predicted["Probabilities"].astype("uint8")

    return predicted
