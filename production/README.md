<img align="centre" src="../figs/Github_banner.jpg" width="101%">

# Digital Earth Africa Continental Cropland Mask - machine learning methods

The code in dea-ml folder is to us machine learning methods to predict crop mask.

## How to build and install the code.

In folder ```crop-mask/deal-ml```, run the following shell commands to install the dependency ```poetry```.
```bash
pip install poety && poetry build
```

After you have dea_ml built by ```poetry```, run
```bash
pip install dist/dea-ml-0.1.6.tar.gz
```

The version '0.1.6' will be changed according to the crop-mask development.

After the install ```dea-ml``` in your python environment, you can use package through python imports.

The sample command to run command line with AWS SQS,
```bash
cm-pred run \
s3://deafrica-data-dev-af/crop_mask_eastern/0-1-0/gm_s2_semiannual_all.db \
--config=./dea_ml/config/plugin_product.yaml \
--plugin-config=./dea_ml/config/ml_config.yaml \
--from-sqs=deafrica-dev-eks-stats-geomedian-semiannual \
--resolution=10 \
--threads=4 \
--memory-limit=4Gi \
--location=s3://deafrica-data-dev-af/{product}/{versoin}
```

It is also possible to use local db file on dev server to test run the task,
```bash
cm-pred run ../../../gm_s2_semiannual_all.db \
--config=./dea_ml/config/plugin_product.yaml \
--plugin-config=./dea_ml/config/ml_config.yaml \
--resolution=10 \
--threads=62 \
--memory-limit=400Gi \
--location=s3://deafrica-data-dev-af/{product}/{version} 4005:4010
```
The `--location` also can be assigned to a local location: `--location=file:///home/<data>/<path>`.

## Additional information

**License:** The code in this notebook is licensed under the [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0).
Digital Earth Africa data is licensed under the [Creative Commons by Attribution 4.0](https://creativecommons.org/licenses/by/4.0/) license.

**Contact:** If you need assistance, please post a question on the [Open Data Cube Slack channel](http://slack.opendatacube.org/) or on the [GIS Stack Exchange](https://gis.stackexchange.com/questions/ask?tags=open-data-cube) using the `open-data-cube` tag (you can view previously asked questions [here](https://gis.stackexchange.com/questions/tagged/open-data-cube)).
If you would like to report an issue with this notebook, you can file one on [Github](https://github.com/digitalearthafrica/crop-mask/issues).
