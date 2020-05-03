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
1. [Build models](#Build-models): Mean model, Simple features model
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
import bokeh
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

# For development, allow the project modules to be reloaded every time they are used
%load_ext autoreload
%aimport bucketplot
%autoreload 1

# Check they have loaded and the versions are as expected
assert platform.python_version_tuple() == ('3', '6', '6')
print(f"Python version:\t\t{sys.version}")
assert IPy_version == '7.13.0'
print(f'IPython version:\t{IPy_version}')
assert np.__version__ == '1.18.2'
print(f'numpy version:\t\t{np.__version__}')
assert pd.__version__ == '0.25.3'
print(f'pandas version:\t\t{pd.__version__}')
assert bokeh.__version__ == '2.0.1'
print(f'bokeh version:\t\t{bokeh.__version__}')
assert skl_version == '0.22.2.post1'
print(f'sklearn version:\t{skl_version}')
assert mpl.__version__ == '3.2.1'
print(f'matplotlib version:\t{mpl.__version__}')
assert sm_version == '0.11.0'
print(f'statsmodels version:\t{sm_version}')
print(f'bucketplot version:\t{bplt.__version__}')
```

```python
# Set the matplotlib defaults
plt.style.use('seaborn')
plt.rcParams['figure.figsize'] = 8, 6

# Load Bokeh for use in a notebook
bokeh.io.output_notebook()
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
if claims_data_filepath.is_file():
    print("Correct: CSV file is available for loading")
else:
    print("Warning: CSV file not yet available in that location")
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
else:
    print("Loading data from CSV...")
    df_raw = pd.read_csv(
        claims_data_filepath, delimiter=',', dtype=expected_dtypes,
        # Get index sorted with ascending IDpol, just in case it is out or order
    ).sort_values('IDpol').reset_index(drop=True)
    print("Load complete")
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
        df_raw, train_size=nrows_sample, random_state=35, shuffle=True
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
df = get_df_extra(df_raw)
```

<div style="text-align: right"><a href="#Contents">Back to Contents</a></div>

# Visualise data
Suppose we want to view the one-way of how the response `ClaimNb` varies according to some other variable. Let's try a scatter plot for `Exposure`.

```python
x_var, stat_var = 'Exposure', 'ClaimNb'
```

```python
df.plot.scatter(x_var, stat_var)
plt.show()
```

This isn't very helpful because:
- With a large amount of data, a scatter plot doesn't give a good indication of where it is most concentrated. Points overlap.
- For imbalanced count data, a large proportion of the data rows have a response of zero.
- The plot must cover the whole range, including any outliers on either axis.

What we want to do is partition the x-axis variable into *buckets*, and plot the *average* respose in each bucket. At the same time, we also want to plot along with the *distribution* of the x-axis variable, to give a sense of the relative credbility of the estimate from each bucket. As follows, this is a task for `pd.cut()` + `groupby` + `agg`. 

Note that we are continuing to parameterise the `x_var` and `stat_var`. It will also be useful to parametrise the weight `stat_wgt` which is attributed to each row for the purpose of finding the *weighted* average of the `stat_var` in each bucket - to start with, the values of `stat_wgt` will just be a constant. We'll keep track of these parameters, so they can be passed between functions.

```python
stat_wgt = None
```

```python
# TODO: Move to util functions

BARGS_DEFAULT = {
    'stat_wgt': 'const',
    'n_bins': 10,
}

def update_bargs(bargs_new, bargs_prev, func_name, bargs_default=BARGS_DEFAULT):
    """Merge two dictionaries, favouring values from dict_override if the keys overlap"""
    # Convert input data types
    if bargs_prev is None:
        bargs_prev = dict()
    # Allocate a non-None value for every key in bargs_new or bargs_prev
    res = dict()
    for key in {**bargs_new, **bargs_prev}.keys():
        if key in bargs_new.keys() and bargs_new[key] is not None:
            res[key] = bargs_new[key]
        elif key in bargs_prev.keys():
            res[key] = bargs_prev[key]
        elif key in bargs_default.keys():
            res[key] = bargs_default[key]
        else: 
            raise ValueError(
                f"{func_name}: '{key}' is required but has not been supplied"
            )
    return(res)
```

```python
def add_df_cols(df, stat_wgt=None, bargs=None):
    """Append additional columns to df needed for aggregation"""
    bargs = update_bargs({
        'stat_wgt': stat_wgt,
    }, bargs, 'add_df_cols')
    
    df_w_cols = df.assign(
        stat_wgt=lambda df: (
            1 if bargs['stat_wgt'] == 'const' 
            else df[bargs['stat_wgt']]
        ),
    )
    return(df_w_cols, bargs)

df_w_cols, bargs = add_df_cols(df)
print(f"'bargs' so far: {bargs}")
df_w_cols.head()
```

```python
def assign_buckets(df_w_cols, cut_var=None, n_bins=10, bargs=None):
    """Cut df into buckets and assign each row to a bucket"""
    bargs = update_bargs({
        'cut_var': cut_var,
        'n_bins': n_bins,
    }, bargs, 'assign_buckets')
    
    df_w_buckets = df_w_cols.assign(
        # Put each row into a "bucket" by "cutting" the data
        bucket=lambda df: pd.cut(
            df[bargs['cut_var']],
            bins=bargs['n_bins']
        ),
    )
    return(df_w_buckets, bargs)

df_w_buckets, bargs = assign_buckets(df_w_cols, cut_var=x_var, bargs=bargs)
print(f"'bargs' so far: {bargs}")
df_w_buckets.head()
```

```python
def group_and_agg(df_w_buckets, x_var, stat_vars, bargs=None):
    """Group by bucket and calculate aggregate values in each bucket"""
    if not isinstance(stat_vars, list):
        stat_vars = [stat_vars]
    bargs = update_bargs({
        'x_var': x_var,
        'stat_vars': stat_vars,
    }, bargs, 'group_and_agg')
    
    df_agg = df_w_buckets.groupby(
        # Group by the buckets
        'bucket', sort=False
    ).agg(
        # Aggregate calculation for rows in each bucket
        n_obs=('bucket', 'size'),  # Just for information
        stat_wgt_sum=('stat_wgt', 'sum'),
        **{stat_var + '_sum': (stat_var, 'sum') for stat_var in bargs['stat_vars']},
        x_min=(bargs['x_var'], 'min'),
        x_max=(bargs['x_var'], 'max'),
    ).pipe(
        # Convert the index to an IntervalIndex
        lambda df: df.set_index(df.index.categories)
    ).sort_index()
    return(df_agg, bargs)

df_agg, bargs = group_and_agg(df_w_buckets, x_var, stat_vars=stat_var, bargs=bargs)
print(f"'bargs' so far: {bargs}")
df_agg.head()
```

```python
def add_agg_cols(df_agg, bargs=None):
    """Append additional columns to aggregated df needed for plotting"""
    bargs = update_bargs({
    }, bargs, 'add_agg_cols')
    
    df_for_plot = df_agg.assign(
        # Get the coordinates of the line points to plot
        **{stat_var + '_wgt_av': (
            lambda df, stat_var=stat_var: df[stat_var + '_sum'] / df['stat_wgt_sum']
        ) for stat_var in bargs['stat_vars']},
        x_left=lambda df: df['x_min'],
        x_right=lambda df: df['x_max'],
        x_mid=lambda df: (df['x_right'] + df['x_left'])/2.,
    )
    return(df_for_plot, bargs)

df_for_plot, bargs = add_agg_cols(df_agg, bargs=bargs)
print(f"'bargs' so far: {bargs}")
df_for_plot.head()
```

For plotting, we switch away from `matplotlib` to `bokeh` because:
- It is cumbersome to use `matplotlib` for plotting both a histogram and line plot on the same axes
- It would be nice to have interactivity (e.g. zooming) in our resulting plots

On the downside, it does require a bit of code to produce the output.

```python
# TODO: Move to util functions

def expand_lims(df, pct_buffer_below=0.05, pct_buffer_above=0.05):
    """
    Find the range over all columns of df. Then expand these 
    below and above by a percentage of the total range.
    Returns: Series with rows 'start' and 'end' of the expanded range
    """
    # If a Series is passed, convert it to a DataFrame
    try:
        df = df.to_frame()
    except:
        pass
    res_range = df.apply(
        # Get the range (min and max) over the DataFrame
        ['min', 'max']).agg({'min': 'min', 'max': 'max'}, axis=1).agg({
        # Expanded range
        'start': lambda c: c['max'] - (1 + pct_buffer_below) * (c['max'] - c['min']),
        'end': lambda c: c['min'] + (1 + pct_buffer_above) * (c['max'] - c['min']),
    })
    return(res_range)
```

```python
# TODO: Refactor with bargs

def create_plot(plt_data_df, x_var, stat_var, stat_wgt):
    # Set up the figure
    bkp = bokeh.plotting.figure(
        title="One-way plot", x_axis_label=x_var, y_axis_label="Weight", 
        tools="reset,box_zoom,pan,wheel_zoom,save", background_fill_color="#fafafa",
        plot_width=800, plot_height=400
    )

    # Plot the histogram squares...
    bkp.quad(
        top=plt_data_df['stat_wgt_sum'], bottom=0,
        left=plt_data_df['x_left'], right=plt_data_df['x_right'],
        fill_color="khaki", line_color="white", legend_label="Weight"
    )
    # ...at the bottom of the graph
    bkp.y_range = bokeh.models.ranges.Range1d(
        **expand_lims(plt_data_df['stat_wgt_sum'], 0, 1)
    )

    # Plot the weight average statistic points joined by straight lines
    # Set up the secondary axis
    bkp.extra_y_ranges['y_range_2'] = bokeh.models.ranges.Range1d(
        **expand_lims(plt_data_df[stat_var + '_wgt_av'])
    )
    bkp.add_layout(bokeh.models.axes.LinearAxis(
        y_range_name='y_range_2',
        axis_label="Weighted average statistic"
    ), 'right')
    # The following parameters need to be passed to both circle() and line()
    stat_line_args = {
        'x': plt_data_df['x_mid'],
        'y': plt_data_df[stat_var + '_wgt_av'],
        'y_range_name': 'y_range_2',
        'color': 'purple',
        'legend_label': stat_var,
    }
    bkp.circle(**stat_line_args, size=4)
    bkp.line(**stat_line_args)

    bkp.legend.location = "top_left"
    bkp.legend.click_policy="hide"
    
    return(bkp)

bkp = create_plot_v1(plt_data_df, **plt_args)
bokeh.plotting.show(bkp)
```

That's a bit more informative! 

Let's try a different `x_var` - one of the actual explanatory variables this time, e.g. `DrivAge`. And this time we'll weight each observation by `Exposure` (as consistent with the model we hope to build). That means we should pick `Frequency` (not `ClaimNb`) to the `stat_var`, so that the *weighted average statistic* in a given bucket is
$$
\frac{\sum_i \textrm{Freq}_i \times \textrm{Exposure}_i}{\sum_i \textrm{Exposure}_i} = 
\frac{\sum_i \textrm{ClaimNb}_i}{\sum_i \textrm{Exposure}_i} =
\textrm{Overall frequency of claims in the bucket}
$$
where the sum is over all rows $i$ that are allocated to the bucket.

```python
x_var, stat_var, stat_wgt = 'DrivAge', 'Frequency', 'Exposure'
```

```python
plt_data_df, plt_args = get_agg_plot_data_v1(df_extra, x_var, stat_var, stat_wgt, n_bins=20)
plt_data_df.head()
```

```python
bkp = create_plot_v1(plt_data_df, **plt_args)
# Here is an example of overwriting attributes that were set in the function
bkp.legend.location = 'top_right'
bokeh.plotting.show(bkp)
```

Looking good! 

```python
# TODO: More buckets => 'all' discrete levels
```

How about another one: `Density`.

```python
x_var, stat_var, stat_wgt = 'Density', 'Frequency', 'Exposure'
plt_data_df, plt_args = get_agg_plot_data_v1(df_extra, x_var, stat_var, stat_wgt, n_bins=20)
bkp = create_plot_v1(plt_data_df, **plt_args)
bokeh.plotting.show(bkp)
```

Uh oh! That graph isn't very helpful - it turns out that `Density` is very skewed, so the weight is concentrated at one end. There are a couple of approaches we could take:
- Transform `Density` (e.g. taking the `log`) to aim for a more symmetric distribution.
- Instead of cutting the range into equal width buckets, try cutting into weighted quantiles.

Let's consider how *weighted quantiles* would work:
1. Order the rows according to `x_var` - we'll parametrise this as `order_by`
1. Calculate the cumulative weight `cum_wgt` of the weight `bucket_wgt` we want to use. In this case, it'll be equal to `stat_wgt`.
1. Cut `cum_wgt` into equal width intervals (rather than by `x_var`). Allocate rows to these buckets, and then proceed as before.

```python
x_var, stat_var, stat_wgt = 'Density', 'Frequency', 'Exposure'
bucket_wgt = stat_wgt
cut_by = 'cum_wgt'
```

```python
# <<<<<<<<<<<<<<<<<<< NOT COMPLETE >>>>>>>>>>>>>>>>>>>>>>>>

def get_agg_plot_data_v2(
    df_extra,
    # What to plot
    x_var, stat_var, stat_wgt=None,
    # Variables to define the buckets
    order_by=None, bucket_wgt=None, cut_by=None, n_bins=10
):
    # Set defaults
    if order_by is None:
        order_by = x_var
    if cut_by is None:
        cut_by = x_var
    if cut_by == 'cum_wgt':
        if bucket_wgt is None:
            bucket_wgt = stat_wgt
    
    # Capture some arguments, so we can pass them to the plotting function
    plt_args = {
        'x_var': x_var,
        'stat_var': stat_var,
        'stat_wgt': stat_wgt,
        'order_by': order_by,
        'bucket_wgt': bucket_wgt,
    }
    if cut_by == 'cum_wgt':
        if 
        df_extra.assign(
            bucket_wgt=lambda df: df[bucket_wgt],
            # Also create a row number column to ensure the ordering is unique
            row_num_ID=lambda df: np.arange(len(df))
        ).sort_values([order_by, 'row_num_ID']).assign(
            cum_wgt=lambda df: df.bucket_wgt.cumsum(),
            # Put each row into a "bucket" by "cutting" the data
            bucket=lambda df: pd.cut(df[cut_by], bins=n_bins),
            stat_wgt=lambda df: 1 if stat_wgt is None else df[stat_wgt],
        )
    
    plt_data_df = df_extra.assign(
        bucket_wgt=lambda df: 1 if bucket_wgt is None else df[bucket_wgt],
        # Also create a row number column to ensure the ordering is unique
        row_num_ID=lambda df: np.arange(len(df))
    ).sort_values([order_by, 'row_num_ID']).assign(
        cum_wgt=lambda df: df.bucket_wgt.cumsum(),
        # Put each row into a "bucket" by "cutting" the data
        bucket=lambda df: pd.cut(df[cut_by], bins=n_bins),
        stat_wgt=lambda df: 1 if stat_wgt is None else df[stat_wgt],
    ).groupby(
        # Group by the buckets
        'bucket', sort=False
    ).agg(
        # Aggregate calculation for rows in each bucket
        n_obs=('bucket', 'size'),  # Just for information
        stat_wgt_sum=('stat_wgt', 'sum'),
        **{stat_var + '_sum': (stat_var, 'sum')},
        x_min=(x_var, 'min'), x_max=(x_var, 'max'),
    ).pipe(
        # Convert the index to an IntervalIndex
        lambda df: df.set_index(df.index.categories)
    ).sort_index().assign(
        # Get the coordinates of the line points to plot
        **{stat_var + '_wgt_av': (
            lambda df, stat_var=stat_var: df[stat_var + '_sum'] / df.stat_wgt_sum
        )},
        x_left=lambda df: df.x_min,
        x_right=lambda df: df.x_max,
        x_mid=lambda df: (df.x_right + df.x_left)/2.,
    )
    return(plt_data_df, plt_args)

plt_data_df, plt_args = get_agg_plot_data_v1(df_extra, x_var, stat_var, stat_wgt)
plt_data_df
```

```python

```

```python

```

```python

```

```python
df_raw.head()
```

```python
bplt.get_agg_plot_data(df_raw, order_by='Exposure', n_bins=10)
```

```python
get_agg_plot_data(
    df_raw,
    stat_cols=None, stat_wgt=None,
    cut_by=None, n_bins=None, order_by=None, bucket_wgt=None,
    x_axis_var=None,
    set_config=None,
)
```

```python

```

```python

```

```python

```

<div style="text-align: right"><a href="#Contents">Back to Contents</a></div>

# Build models

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
