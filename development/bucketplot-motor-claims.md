---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.4.2
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Bucket plot visualisations for motor insurance claim frequency models
Using an open source data set of French Motor Claims as an example to demonstrate how `bucketplot` visualisations can be used in the modelling process.

This notebook is available in the following locations. These versions are kept in sync *manually* - there should not be discrepancies, but it is possible.
- On Kaggle: <https://www.kaggle.com/btw78jt/bucketplot-motor-claims>
- In the GitHub project repo: <https://github.com/A-Breeze/bucketplot>. See the project `README.md` for important information and further instructions.

<!-- This table of contents is updated *manually* -->
# Contents
1. [Setup](#Setup)
1. [Modelling data](#Modelling-data): Load data, Subset, Pre-processing, Split for modelling
1. [Fit models](#Fit-models): Mean model, Simple features model
1. [Visualise results](#Visualise-results)


<div style="text-align: right"><a href="#Contents">Back to Contents</a></div>

# Setup

```python
# Set warning messages
import warnings
# Show all warnings in IPython
warnings.filterwarnings('always')
# Ignore specific numpy warnings (as per <https://github.com/numpy/numpy/issues/11788#issuecomment-422846396>)
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
# Other warnings that sometimes come up
warnings.filterwarnings("ignore", message="unclosed file <_io.TextIOWrapper")
warnings.filterwarnings("ignore", message="Anscombe residuals currently unscaled")
```

```python
# Determine whether this notebook is running on Kaggle
from pathlib import Path

on_kaggle = False
print("Current working directory: " + str(Path('.').absolute()))
if str(Path('.').absolute()) == '/kaggle/working':
    on_kaggle = True
```

```python
# Import built-in modules
import sys
import platform
import os
from pathlib import Path

# Import external modules
from IPython import __version__ as IPy_version
import numpy as np
import pandas as pd
from bokeh import __version__ as bk_version
from sklearn import __version__ as skl_version
from sklearn.model_selection import train_test_split
import matplotlib as mpl
import matplotlib.pyplot as plt
from statsmodels import __version__ as sm_version
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Import project modules
if not on_kaggle:
    # Allow modules to be imported relative to the project root directory
    from pyprojroot import here
    root_dir_path = here()
    if not sys.path[0] == root_dir_path:
        sys.path.insert(0, str(root_dir_path))
import bucketplot as bplt

# Check they have loaded and the versions are as expected
assert platform.python_version_tuple() == ('3', '6', '6')
print(f"Python version:\t\t{sys.version}")
assert IPy_version == '7.13.0'
print(f'IPython version:\t{IPy_version}')
assert np.__version__ == '1.18.2'
print(f'numpy version:\t\t{np.__version__}')
assert pd.__version__ == '0.25.3'
print(f'pandas version:\t\t{pd.__version__}')
assert bk_version == '2.0.1'
print(f'bokeh version:\t\t{bk_version}')
assert skl_version == '0.22.2.post1'
print(f'sklearn version:\t{skl_version}')
assert mpl.__version__ == '3.2.1'
print(f'matplotlib version:\t{mpl.__version__}')
assert sm_version == '0.11.0'
print(f'statsmodels version:\t{sm_version}')
print(f'bucketplot version:\t{bplt.__version__}')
```

```python
# Load Bokeh for use in a notebook
from bokeh.io import output_notebook
output_notebook()
```

```python
# Output exact environment specification, in case it is needed later
print("Capturing full package environment spec")
print("(But note that not all these packages are required)")
!pip freeze > requirements_snapshot.txt
!jupyter --version > jupyter_versions_snapshot.txt
```

```python
# Configuration variables
if on_kaggle:
    claims_data_filepath = Path('/kaggle/input/french-motor-claims-datasets-fremtpl2freq/freMTPL2freq.csv')
else:
    claims_data_filepath = Path('freMTPL2freq.csv')
```

<div style="text-align: right"><a href="#Contents">Back to Contents</a></div>

# Modelling data


## Load data

```python
expected_dtypes = {
    **{col: np.dtype('int64') for col in [
        'IDpol', 'ClaimNb', 'VehPower', 'VehAge', 'DrivAge', 'BonusMalus', 'Density']},
    **{col: np.dtype('float64') for col in ['Exposure']},
    **{col: np.dtype('O') for col in ['Area', 'VehBrand', 'VehGas', 'Region']},
}
```

```python
%%time
# The first download can take approx 1 min on Binder
if not claims_data_filepath.is_file():
    from sklearn.datasets import fetch_openml
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=UserWarning,
            message='Version 1 of dataset freMTPL2freq is inactive'
        )
        print("Fetching data...")
        df_raw = fetch_openml(data_id=41214, as_frame=True, cache=False).frame.apply(
            lambda col_sers: col_sers.astype(expected_dtypes[col_sers.name])
        ).sort_values('IDpol').reset_index(drop=True)
    # Cache data within the repo, so we don't have to download it many times
    print("Saving data...")
    df_raw.to_csv(claims_data_filepath, index=False)
    print("Save complete")

df_raw = pd.read_csv(
    claims_data_filepath, delimiter=',', dtype=expected_dtypes
    # Get index sorted with ascending IDpol, just in case it is out or order
).sort_values('IDpol').reset_index(drop=True)
```

```python
# Reasonableness checks that it has loaded as expected
nRows, nCols = (678013, 12)
assert df_raw.shape == (nRows, nCols)
print(f"Correct: Shape of DataFrame is as expected: {nRows} rows, {nCols} cols")
assert df_raw.dtypes.equals(pd.Series(expected_dtypes)[df_raw.columns])
print("Correct: Data types are as expected")
assert df_raw.isna().sum().sum() == 0
print("Correct: There are no missing values in the dataset")
```

## Subset
This notebook is about visualisation, rather than model building, so we can choose a smaller sample of the data to allow the code to run quickly. If you are working on a larger system, consider increasing the sample size to the whole data set.

```python
# Hard-coded stats for reasonableness checks
mean_approx = pd.Series({
    'ClaimNb': 0.0532,
    'Exposure': 0.5288,
    'VehPower': 6.45,
    'VehAge': 7.04,
    'DrivAge': 45.50,
    'BonusMalus': 59.8,
    'Density': 1792.,
})
```

```python
nrows_sample = int(1e4)
if not on_kaggle:
    df_raw, df_unused = train_test_split(
        df_raw, train_size=nrows_sample, random_state=3, shuffle=True
    )
```

```python
# Check it is as expected, within reason
tol_pc = 0.05
assert df_raw[mean_approx.index].mean().between(
    mean_approx * (1 - tol_pc),
    mean_approx * (1 + tol_pc)
).all()
print("Correct: Reasonableness checks have passed")
```

## Pre-processing

```python
def get_df_extra(df_raw):
    """
    Given a DataFrame of that contains the raw data columns (and possibly additional columns), 
    return the DataFrame with additional pre-processed columns
    """
    df_extra = df_raw.copy()
    
    # Calculate frequency per year on each row
    df_extra['Frequency'] = df_extra['ClaimNb'] / df_extra['Exposure']
    
    # TODO: Consider which of the below is needed
    
#     # Feature engineering (the results of the analysis below)
#     VehBrand_map_sers = pd.Series({
#         'B12': 'X', 'B14': 'X', 'B13': 'X',
#         'B3': 'X', 'B11': 'X','B4': 'X', 'B5': 'X',
#         'B1': 'Y', 'B6': 'Y',
#         'B2': 'Z', 'B10': 'Z'
#     })

#     Region_map_sers = pd.Series({
#         **{reg: 'W' for reg in ['R21', 'R94', 'R11', 'R42', 'R22', 'R74']},
#         **{reg: 'X' for reg in ['R91', 'R82']},
#         **{reg: 'Y' for reg in ['R93', 'R53']},
#         **{reg: 'Z' for reg in ['R26', 'R25', 'R52', 'R31', 'R54', 'R73', 
#                                 'R23', 'R72', 'R83', 'R41', 'R43']},
#         **{reg: 'A' for reg in ['R24']},
#     })

#     df_extra = df_extra.assign(
#         DrivAge_capped=lambda x: np.clip(x.DrivAge, None, 80),
#         DrivAge_pow2=lambda x: np.power(x.DrivAge_capped, 2),
#         BonusMalus_over_50=lambda x: np.select([x.BonusMalus > 50], ["Y"], default="N"),
#         BonusMalus_mod3=lambda x: np.floor((np.clip(x.BonusMalus, None, 90) - 48)/3)*3 + 50,
#         VehAge_new=lambda x: np.select([x.VehAge == 0], ["Y"], default="N"),
#         VehAge_capped=lambda x: np.clip(x.VehAge, None, 18),
#         VehBrand_grd=lambda x: VehBrand_map_sers.loc[x.VehBrand].values,
#         Density_log=lambda x: np.log10(np.clip(x.Density, 10, np.power(10, 4))),
#         Region_grd=lambda x: Region_map_sers.loc[x.Region].values,
#     )
    
    return(df_extra)
```

```python
# Run pre-processing to get a new DataFrame
df_extra = get_df_extra(df_raw.iloc[:10000,])
```

```python
expl_var_names = [
    col_name for col_name in df_extra.columns.to_list() 
     if col_name not in ['IDpol', 'ClaimNb', 'Exposure', 'Frequency']
]
print("Explanatory variables\n" + '\t'.join(expl_var_names))
simple_features = expl_var_names[:9]
print("\nOf which the following are simple features\n" + '\t'.join(simple_features))
```

## Split for modelling

```python
# Split training into train and test
df_train, df_test = train_test_split(
    df_extra, test_size=0.3, random_state=34, shuffle=True
)
print("Train sample size: " + str(df_train.shape))
```

## Useful functions

```python
def score_data(data_df, GLMRes_obj):
    raw_exog_names = pd.Series(GLMRes_obj.model.exog_names[1:]).str.split(
        '[', expand=True, n=1).iloc[:,0].drop_duplicates().to_list()
    scored_df = data_df.assign(
        wgt=lambda x: x.Exposure,
        act_freq=lambda x: x[GLMRes_obj.model.endog_names] / x.wgt,
        pred_freq=lambda x: GLMRes_obj.predict(x[raw_exog_names]),
        act_Nb=lambda x: x[GLMRes_obj.model.endog_names],
        pred_Nb=lambda x: x.pred_freq * x.wgt,
    )
    return(scored_df)
```

<div style="text-align: right"><a href="#Contents">Back to Contents</a></div>

# Fit models


## Mean model
Just for checking that the code is working for the simplest case.

```python
%%time
GLMres_mean = smf.glm(
    "ClaimNb ~ 1",
    data=df_train, exposure=np.asarray(df_train['Exposure']),
    family=sm.families.Poisson(sm.genmod.families.links.log()),
).fit()
print(GLMres_mean.summary())
```

```python
# Check that this is the mean model
mean_mod_pred = np.exp(GLMres_mean.params[0])
assert np.abs(
    GLMres_mean.family.link.inverse(GLMres_mean.params[0]) - 
    GLMres_mean.predict(pd.DataFrame([1]))[0]
) < 1e-10
assert np.abs(
    df_train.ClaimNb.sum() / df_train.Exposure.sum() - 
    mean_mod_pred
) < 1e-10
print("Correct: Reasonableness tests have passed")
```

## Vehicle-only model

```python
veh_features =['VehPower', 'VehAge', 'VehBrand', 'VehGas']
```

```python
%%time
# Takes a few secs
GLMres_veh = smf.glm(
    "ClaimNb ~ " +  ' + '.join(veh_features),
    data=df_train, exposure=np.asarray(df_train['Exposure']),
    family=sm.families.Poisson(sm.genmod.families.links.log()),
).fit()
print(GLMres_veh.summary())
```

## Simple features model

```python
%%time
# Takes approx 10 secs
GLMres_simple = smf.glm(
    "ClaimNb ~ " +  ' + '.join(simple_features),
    data=df_train, exposure=np.asarray(df_train['Exposure']),
    family=sm.families.Poisson(sm.genmod.families.links.log()),
).fit()
print(GLMres_simple.summary())
```

```python
%%time
# Score all the training data for analysis
# Takes under 10 secs
scored_dfs[0] = score_data(df_train, mods_df.GLMResults[0])
```

```python

```

<div style="text-align: right"><a href="#Contents">Back to Contents</a></div>

# Visualise results

```python

```

```python

```

```python

```

<div style="text-align: right"><a href="#Contents">Back to Contents</a></div>
