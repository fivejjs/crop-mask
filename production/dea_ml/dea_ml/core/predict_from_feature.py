import math

import fsspec
import psutil
import xarray as xr
from deafrica_tools.classification import predict_xr
from odc.io.cgroups import get_cpu_quota, get_mem_quota


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


def predict_with_model(model, data, td_url) -> xr.Dataset:
    """
    run the prediction here
    """
    # step 1: select features

    # load the column names from the
    # training data file to ensure
    # the bands are in the right order
    with fsspec.open(td_url, "r") as file:
        header = file.readline()
    column_names = header.split()[1:][1:]

    # reorder input data according to column names
    input_data = data[column_names]

    # step 2: prediction
    predicted = predict_xr(
        model,
        input_data,
        clean=True,
        proba=True,
        return_input=True,
    )

    predicted["Predictions"] = predicted["Predictions"].astype("uint8")
    predicted["Probabilities"] = predicted["Probabilities"].astype("uint8")

    return predicted
