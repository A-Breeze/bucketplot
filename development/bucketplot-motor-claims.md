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
1. [Modelling data](#Modelling-data): Load data, Subset, Pre-processing
1. [Build models](#Build-models): Split for modelling, Mean model, Vehicle-only model, Simple features model, Score
1. [Bucket plot visualisation](#Bucket-plot-visualisation): Motivation, Steps, Data types, Examples to use
1. [Assigning buckets](#Assigning-buckets): [divide_n](#divide_n), [custom_width](#custom_width), [weighted_quantiles](#weighted_quantiles), [all_levels](#all_levels)
1. [Group and aggregate](#Group-and-aggregate): NOT COMPLETE
1. [Plot](#Plot): NOT COMPLETE
1. [Worked examples](#Worked-example): NOT COMPLETE
1. [Rough work](#Rough-work)


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
warnings.filterwarnings("ignore", message="unclosed file <_io.Buffered")
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
import functools
import inspect

# Import external modules
from IPython import __version__ as IPy_version
import numpy as np
import pandas as pd
import bokeh
import bokeh.palettes
import bokeh.io
import bokeh.plotting
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
if on_kaggle:
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
    import proj_config
    claims_data_filepath = proj_config.EXAMPLE_DATA_DIR_PATH / 'freMTPL2freq.csv'
if claims_data_filepath.is_file():
    print("Correct: CSV file is available for loading")
else:
    print(
        "Warning: CSV file not yet available in that location\n"
        "Please download it manually from here:\n"
        "https://www.openml.org/d/41214"
    )
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
if not on_kaggle or on_kaggle:  # <<<<<<<<<<<<<<<<<<< TODO: redo this
    df_raw, df_unused = train_test_split(
        df_raw, train_size=nrows_sample, random_state=35, shuffle=True
    )
```

```python
# Check it is as expected, within reason
tol_pc = 0.05
df_sample_means = df_raw[mean_approx.index].mean()
assert df_sample_means.between(
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
    
    return(df_extra)
```

```python
# Run pre-processing to get a new DataFrame
df = get_df_extra(df_raw)
```

```python
expl_var_names = [
    col_name for col_name in df.columns.to_list() 
     if col_name not in ['IDpol', 'ClaimNb', 'Exposure', 'Frequency']
]
print("Explanatory variables\n" + '\t'.join(expl_var_names))
simple_features = expl_var_names[:9]
print("\nOf which the following are simple features\n" + '\t'.join(simple_features))
```

<div style="text-align: right"><a href="#Contents">Back to Contents</a></div>

# Build models
## Split modelling data

```python
# Split training into train and test
df_train, df_test = train_test_split(
    df, test_size=0.3, random_state=34, shuffle=True
)
print("Train sample size: " + str(df_train.shape))
```

```python
# Add indicator column
df = df.assign(
    split=lambda df: np.select(
        [df.index.isin(df_train.index)],
        ['Train'],
        default='Test'
    )
)
df['split'].value_counts()
```

## Mean model

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

## Simple factors model

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

## Score data
All data (training and test rows) on both models

```python
%%time
# Score all data (training and test)
df = df.assign(
    Freq_pred_mean=lambda df: GLMres_mean.predict(df),
    Freq_pred_veh=lambda df: GLMres_veh.predict(df),
    Freq_pred_simple=lambda df: GLMres_simple.predict(df),
)
```

```python
# Check reasonableness
# The actual sum of ClaimNB should exactly match each model's predicted sum on the training data
pred_claims_df = df.assign(
    ClaimNb_pred_mean=lambda df: df['Freq_pred_mean'] * df['Exposure'],
    ClaimNb_pred_veh=lambda df: df['Freq_pred_veh'] * df['Exposure'],
    ClaimNb_pred_simple=lambda df: df['Freq_pred_simple'] * df['Exposure'],
).groupby('split').agg(
    n_obs=('split', 'size'),
    Exposure=('Exposure', 'sum'),
    ClaimNb=('ClaimNb', 'sum'),
    ClaimNb_pred_mean=('ClaimNb_pred_mean', 'sum'),
    ClaimNb_pred_veh=('ClaimNb_pred_veh', 'sum'),
    ClaimNb_pred_simple=('ClaimNb_pred_simple', 'sum'),
)
assert np.max(np.abs(
    pred_claims_df.loc['Train'].iloc[-3:] - pred_claims_df.loc['Train', 'ClaimNb']
)) < 1e-8
pred_claims_df
```

<div style="text-align: right"><a href="#Contents">Back to Contents</a></div>

# Bucket plot visualisation
## Motivation
Suppose we want to view the one-way of how the response `Frequency` varies according to some other variable. Let's try a scatter plot for `DrivAge`.

```python
x_var, stat_var = 'DrivAge', 'Frequency'
df.plot.scatter(x_var, stat_var)
plt.show()
```

This isn't very helpful because:
- With a large amount of data, a scatter plot doesn't give a good indication of where it is most concentrated. Points overlap.
- For imbalanced count (or frequency) data, a large proportion of the data rows have a response of zero.
- The plot must cover the whole range, including any outliers on either axis.

What we want to do is partition the x-axis variable into *buckets*, and plot the *average* respose in each bucket. At the same time, we also want to plot along with the *distribution* of the x-axis variable, to give a sense of the relative credbility of the estimate from each bucket. As follows, this is a task for `pd.cut()` + `groupby` + `agg`. 


## Steps
1. Assign rows to buckets. 
    - *Every* row is assigned to one bucket. There may be buckets that contain no rows (use `n_obs` aggregate variable to determine this).
    - The buckets are `Categorical` with each bucket being an `Interval`. There are no gaps between the intervals. The bucket can be missing (i.e. `NaN`), in which case the row will be excluded on grouping the data.
1. Group the data by bucket. Within each bucket consider the distribution of columns values *weighted* by `stat_wgt`. Calculate aggregate figures for each bucket:
    - *x* coordinate edges: (`x_left`, `x_right`). Must not overlap (but there can be gaps). Must be `x_left` < `x_right` *or* both `NaN`.
    - Sum of `stat_wgt` to be plotted as a histogram
    - One point coordinate per statistic to be plotted at (`x_point`, `stat_val`)
1. Plot 


## Data types
Data types of individual columns for bucket plots:
- Numeric (`int` or `float`)
    - Continuous = a high number (or high proportion) of unique values
        - Pure
        - With concentration at one value
    - Discrete / ordinal = a low number of unique values. Ordered. May have categories with no occurrences in the data.
- Non-numeric (`str`)
    - Ordinal = as above.
    - Nominal = finite number of categories, not ordered.
- Others: not appropriate to plot

Other specifications:
- Weights:
    - All non-negative with at least one positive.
    - Might contain repeated values.


## Examples to use

```python
# Look at first few rows
df.head()
```

```python
# Weights
# df['Exposure'] - none are zero
# df['ClaimNb'] - many are zero

# Numeric
# df['Density'] - close to continuous with a large skew
# df['DrivAge'] - between discrete (ordinal) and continuous
# df['Exposure'] - close to continuous, odd distribution
# df['Frequency'] - continuous with a concentration at 0
# df['Freq_pred_veh'] and df['Freq_pred_simple'] - continuous and positive

# Non-numeric
# df['Area'] - nominal with a low number of levels
# df['Region'] and df['VehBrand'] - nominal with a higher number of levels
```

<div style="text-align: right"><a href="#Contents">Back to Contents</a></div>

# Assigning buckets
Every method:
- Takes a DataFrame `df` and the name `bucket_var` of a column.
- Uses other arguments to append a column `bucket` of which bucket each row is assigned to. The column data type will be `Categorical` where the categories are:
    - `Interval`s for numeric bucket methods.
    - Single values for non-numeric bucket methods.
- Returns the enlarged `df`.

Possible methods:
- Numeric:
    - `divide_n`: Split range into `n_bins` equal width buckets. Any numeric data. For discrete data, want `n_bins` to be much smaller than the number of unique values.
    - `custom_width`: Specify the `width` and `boundary` point. Options for `first_break` and `last_break` for larger width buckets at either end. Any numeric data.
    - `weighted_quantile`: Quantiles of `bucket_var` weighted by `bucket_wgt` (which can be special value `const`). Aim for `n_bins` but note that there are only a finite number of cut points (especially when `bucket_var` is discrete or is concentrated at only a few points).
- Non-numeric:
    - `all_levels`: One bucket for each level.

Other methods would also be possible.


### Technicalities
- An interval index can only contain half-open intervals that are all closed on the same side, i.e. all $(a,b]$ or $[a,b)$. We will *always* stick to the default of closed on the *right*. To ensure all rows fall within a bucket, `pd.cut()` extends the bottom bucket by 0.1% of the entire range (when passing an `int` to `bins`). In any custom implementations, we replicate this convention.


<div style="text-align: right"><a href="#Contents">Back to Contents</a></div>

## divide_n

```python
def divide_n(df, bucket_var, n_bins=10, bucket_col='bucket'):
    """
    Assign each row of `df` to a bucket by dividing the range of the 
    `bucket_var` column into `n_bins` number of equal width intervals.
    
    df: DataFrame
    bucket_var: Name of the column of df to use for dividing
    n_bins: positive integer number of buckets
    bucket_col: Name of the resulting `bucket` column
    
    Returns: df with the additional `bucket` column 
        The `bucket` column is Categorical data type consisting of Intervals
        that partition the interval from just below min(bucket_var) to 
        max(bucket_var).
    """
    df_w_buckets = df.assign(**{bucket_col: (
        lambda df: pd.cut(df[bucket_var], bins=n_bins)
    )})
    return(df_w_buckets)
```

```python
bucket_var = 'Density'
# bucket_var = 'Exposure'
# bucket_var = 'DrivAge'
tmp1 = df.pipe(divide_n, bucket_var, 10)
tmp1.groupby('bucket').agg(
    n_obs=('bucket', 'size'),
    stat_wgt_sum=('Exposure', 'sum'),
    stat_sum=('ClaimNb', 'sum'),
    x_min=(bucket_var, 'min'),
    x_max=(bucket_var, 'max'),
)
```

```python
# Edge cases
# Resulting bucket with no obs
pd.Series([0, 1]).to_frame('val').pipe(
    divide_n, 'val', 3
).groupby('bucket').agg(n_rows=('bucket', 'size'))

# Constant bucket_var
pd.Series([0, 0]).to_frame('val').pipe(
    divide_n, 'val', 2
).groupby('bucket').agg(n_rows=('bucket', 'size'))

# n_bins = 1
pd.Series([0, 1]).to_frame('val').pipe(
    divide_n, 'val', 1
).groupby('bucket').agg(n_rows=('bucket', 'size'))
```

```python
# Missing vals
unit_w_miss = pd.Series([0, 1, np.nan]).to_frame('val').pipe(
    divide_n, 'val', 3
)
display(unit_w_miss)  # Given a bucket 'NaN'...
display(  # ...which is not included after grouping
    unit_w_miss.groupby('bucket').agg(n_rows=('bucket', 'size'))  
)
```

```python
# Use a missing indicator to cope with missing values
unit_filled = pd.Series([0, 1, np.nan]).to_frame('val').assign(
    val_miss_ind=lambda df: df.val.isna() + 0,
    val=lambda df: df.val.fillna(0),
).pipe(
    divide_n, 'val', 3).pipe(
    # Would be more natural to use all_levels() in this case
    divide_n, 'val_miss_ind', 2, 'bucket_miss_ind'  
)
display(unit_filled)
unit_filled.groupby(['bucket_miss_ind', 'bucket']).agg(
    n_rows=('bucket', 'size')
)
```

<div style="text-align: right"><a href="#Contents">Back to Contents</a></div>

## custom_width

```python
def custom_width(
    df, bucket_var,
    width, boundary=0,
    first_break=None, last_break=None,
    bucket_col='bucket'
):
    """
    Assign each row of `df` to a bucket by dividing the range of the 
    `bucket_var` column into `n_bins` number of equal width intervals.
    
    df: DataFrame
    bucket_var: Name of the column of df to use for dividing.
    width: Positive width of the buckets
    boundary: Edge of one of the buckets, if the data extended that far
    first_break: All values below this (if any) are grouped into one bucket
    last_break: All values above this (if any) are grouped into one bucket
    bucket_col: Name of the resulting `bucket` column
    
    Returns: df with the additional `bucket` column 
        The `bucket` column is Categorical data type consisting of Intervals
        that partition the interval from just below min(bucket_var) to 
        max(bucket_var).
    """
    var_min, var_max = df[bucket_var].min(), df[bucket_var].max()
    extended_min = var_min - 0.001 * np.min([(var_max - var_min), width])

    # Set bucket edges
    start = np.floor((extended_min - boundary) / width) * width + boundary
    stop = np.ceil((var_max - boundary) / width) * width + boundary
    num = int((stop - start) / width) + 1
    breaks_all = np.array([
        extended_min,
        *np.linspace(start, stop, num)[1:-1],
        var_max,
    ])
    
    # Clip lower and upper buckets
    breaks_clipped = breaks_all
    if first_break is not None or last_break is not None:
        breaks_clipped = np.unique(np.array([
            breaks_all.min(),
            *np.clip(breaks_all, first_break, last_break),
            breaks_all.max(),
        ]))
    
    df_w_buckets = df.assign(**{bucket_col: (
        lambda df: pd.cut(df[bucket_var], bins=breaks_clipped)
    )})
    return(df_w_buckets)
```

```python
# bucket_var, width, boundary, first_break, last_break = 'DrivAge', 3, 17.5, None, None
# bucket_var, width, boundary, first_break, last_break = 'DrivAge', 3, 0.5, 30, 70
# bucket_var, width, boundary, first_break, last_break = 'DrivAge', 100, 0.5, None, None
bucket_var, width, boundary, first_break, last_break = 'Density', 100, 0.5, None, 1500.5
tmp6 = custom_width(df, bucket_var, width, boundary, first_break, last_break)
tmp6_grpd = tmp6.groupby('bucket').agg(
    n_obs=('bucket', 'size'),
    stat_wgt_sum=('Exposure', 'sum'),
    stat_sum=('ClaimNb', 'sum'),
    x_min=(bucket_var, 'min'),
    x_max=(bucket_var, 'max'),
    x_nunique=(bucket_var, 'nunique'),
).assign(
    bucket_width=lambda df: df.index.categories.length
)
tmp6_grpd
```

<div style="text-align: right"><a href="#Contents">Back to Contents</a></div>

## weighted_quantiles

```python
def weighted_quantiles(df, bucket_var, n_bins=10, bucket_wgt=None, bucket_col='bucket'):
    """
    Assign each row of `df` to a bucket by splitting column `bucket_var`
    into `n_bins` weighted quantiles, weighted by `bucket_wgt`.
    
    bucket_var: Column name of the values to find the quantiles.
        Must not be constant (i.e. just one value for all rows).
    n_bins: Target number of quantiles, but could end up with fewer because
        there are only a finite number of potential cut points.
    bucket_wgt: Weights to use to calculate the weighted quantiles.
        If None (default) or 'const' then equal weights are used for all rows.
        Must be non-negative with at least one postive value.
    bucket_col: Name of the resulting `bucket` column.
    
    Returns: df with the additional `bucket` column 
        The `bucket` column is Categorical data type consisting of Intervals
        that partition the interval from 0 to sum(bucket_wgt).
    """
    if bucket_wgt is None:
        bucket_wgt = 'const'
    if bucket_wgt == 'const' and 'const' not in df.columns:
        df = df.assign(const = 1)

    res = df.sort_values(bucket_var).assign(**{
        'cum_rows_' + bucket_wgt: lambda df: (
            df[bucket_wgt].cumsum()
        ),
        # Ensure that the quantiles cannot split rows with the same value of bucket_var
        'cum_' + bucket_wgt: lambda df: (
            df.groupby(bucket_var)['cum_rows_' + bucket_wgt].transform('max')
        ),
        bucket_col: (
            lambda df: pd.cut(df['cum_' + bucket_wgt], bins=n_bins)
        )
    })
    return(res)
```

```python
# bucket_var, bucket_wgt = 'Density', 'const'
# bucket_var, bucket_wgt = 'Density', 'Exposure'
# bucket_var, bucket_wgt = 'Density', 'Frequency'
# bucket_var, bucket_wgt = 'DrivAge', 'Exposure'
# bucket_var, bucket_wgt = 'Region', 'Exposure'  # Does *not* make sense to order by nominal variable 'Region'
# bucket_var, bucket_wgt = 'Freq_pred_mean', 'Exposure'  # Does *not* make sense for bucket_var to be constant
bucket_var, bucket_wgt = 'Freq_pred_veh', 'Exposure'  # Example for lift chart
tmp2 = weighted_quantiles(df, bucket_var, 8, bucket_wgt)
tmp2_grpd = tmp2.groupby('bucket').agg(
    n_obs=('bucket', 'size'),
    stat_wgt_sum=('Exposure', 'sum'),
    stat_sum=('ClaimNb', 'sum'),
    x_min=(bucket_var, 'min'),
    x_max=(bucket_var, 'max'),
    x_nunique=(bucket_var, 'nunique'),
)
tmp2.head()
```

```python
# Cases
# It is still possible to end up with no rows in a bucket
pd.Series([2, 2, 3, 3]).to_frame('val').assign(
    bucket=lambda df: pd.cut(df['val'], bins=5)
).groupby('bucket').agg(n_rows=('bucket', 'size'))
```

```python
# Illustration of why we don't want to split rows that have the same value of bucket_var
pd.DataFrame({
    'bucket_var': [0, 0, 1],
    'bucket_wgt': [1, 1, 1],
}).sort_values('bucket_wgt').assign(
    cum_wgt_rows=lambda df: df['bucket_wgt'].cumsum(),
    bucket_rows=lambda df: pd.cut(df['cum_wgt_rows'], bins=3),
    cum_wgt=lambda df: df.groupby('bucket_var')['cum_wgt_rows'].transform('max'),
    bucket=lambda df: pd.cut(df['cum_wgt'], bins=3),
)
```

<div style="text-align: right"><a href="#Contents">Back to Contents</a></div>

## all_levels

```python
def all_levels(df, bucket_var, include_levels=None, ret_map=False, bucket_col='bucket'):
    """
    Assign each row of `df` to a bucket according to the unique 
    values of `bucket_var`.
    
    bucket_var: Column name of the values to split on.
        Missing values will not be assigned to an interval.
    include_levels: Level values to guarantee to include 
        even if they do not appear in the values of bucket_var.
        Missing values are ignored.
    ret_map: Whether to also return the bucket_map Series.
    bucket_col: Name of the resulting `bucket` column.
    
    Returns: 
        df with the additional `bucket` column
            The `bucket` column is Categorical data type consisting of 
            Intervals that partition a range, plus possible NaN.
        If ret_map is True, also return a Series mapping bucket values
            to bucket intervals.
    """
    # Format inputs
    if include_levels is not None:
        if not isinstance(include_levels, pd.Series):
            include_levels = pd.Series(include_levels)
    
    # Get the mapping from level value to an appropriate interval
    buckets_vals = pd.concat([
        df[bucket_var], include_levels
    ]).drop_duplicates().sort_values(
    ).reset_index(drop=True).dropna().to_frame('val')
    
    # Add a column of intervals (there may be some intervals with no rows)
    if np.issubdtype(df[bucket_var].dtype, np.number):
        # If the values are numeric then take the smallest width
        min_diff = np.min(np.diff(buckets_vals['val']))
        buckets_map = buckets_vals.assign(
            interval=lambda df: pd.cut(df['val'], pd.interval_range(
                start=df['val'].min() - min_diff/2,
                end=df['val'].max() + min_diff/2,
                freq=min_diff
            ))
        )
    else:
        # If the values are not numeric then take unit intervals
        buckets_map = buckets_vals.assign(
            interval=lambda df: pd.interval_range(start=0., periods=df.shape[0], freq=1.)
        )
    
    # Convert to a Series
    buckets_map = buckets_map.reset_index(drop=True)
    
    # Assign buckets and map to intervals
    res = df.assign(**{bucket_col: lambda df: (
        df[bucket_var].astype(
            # Cast the bucket variable as Categorical
            pd.CategoricalDtype(buckets_map['val'], ordered=True)
        ).cat.rename_categories(
            # Swap the bucket levels with the bucket intervals
            buckets_map.set_index('val')['interval']
        )
    )})
    
    if ret_map:
        return(res, buckets_map)
    return(res)
```

```python
# bucket_var, include_levels = 'DrivAge', None  # Discrete all levels
bucket_var, include_levels = 'Area', 'X'  # Categorical all levels
# bucket_var, include_levels = 'DrivAge', pd.Series([18.5])
# bucket_var, include_levels = 'Area', np.nan  # With missing vals
# bucket_var, include_levels = 'Area', None  # Slightly different
tmp3, tmp3_bucket_map = df.pipe(all_levels, bucket_var, include_levels, ret_map=True)
tmp3_grpd = tmp3.groupby('bucket').agg(
    n_obs=('bucket', 'size'),
    stat_wgt_sum=('Exposure', 'sum'),
    stat_sum=('ClaimNb', 'sum'),
    x_min=(bucket_var, 'min'),
    x_max=(bucket_var, 'max'),
    x_nunique=(bucket_var, 'nunique'),
)
# Use the bucket_map to assign labels to each bucket interval
tmp3_grpd.assign(
    x_label=lambda df: pd.Categorical(df.index).rename_categories(
        tmp3_bucket_map.set_index('interval')['val']
    )
)
```

```python
# Missing vals
unit_w_miss, bucket_map = pd.Series([0, 1, np.nan]).to_frame('val').pipe(
    lambda df: all_levels(df, 'val', ret_map=True)
)
display(unit_w_miss)
display(bucket_map)
```

```python
# Use a missing indicator to cope with missing values
unit_filled, b_map = pd.Series([0, 1, np.nan]).to_frame('val').assign(
    val_miss_ind=lambda df: df.val.isna(),
    val=lambda df: df.val.fillna(0),
).pipe(divide_n, 'val', 3).pipe(
    all_levels, 'val_miss_ind', bucket_col='bucket_miss_ind', ret_map=True 
)
display(unit_filled)
unit_filled.groupby(['bucket_miss_ind', 'bucket']).agg(
    n_rows=('bucket', 'size')
).assign(y_label=lambda df: (
    df.index.get_level_values('bucket_miss_ind').rename_categories(
        b_map.set_index('interval')['val'].to_dict()
    )
))
```

```python
# Interesting case: We can now group a nominal variable first by all_levels
# and then by weighted_quantiles, to group the levels in order of increasing
# stat_wgt_av. This is a possible way to group levels of a nominal variable
# that makes sense.
bucket_var, include_levels = 'Region', None
tmp4_grpd = df.pipe(
    all_levels, bucket_var, include_levels
).groupby('bucket').agg(
    n_obs=('bucket', 'size'),
    stat_wgt_sum=('Exposure', 'sum'),
    stat_sum=('ClaimNb', 'sum'),
    x_min=(bucket_var, 'min'),
    x_max=(bucket_var, 'max'),
    x_nunique=(bucket_var, 'nunique'),
).assign(
    stat_wgt_av=lambda df: df['stat_sum'] / df['stat_wgt_sum']
)
tmp4_grpd
```

```python
tmp4_grpd.rename_axis(index='index').pipe(
    weighted_quantiles, 'stat_wgt_av', 8, 'stat_wgt_sum'
).groupby('bucket').agg(
    n_obs=('bucket', 'size'),
    stat_wgt_sum=('stat_wgt_sum', 'sum'),
    stat_sum=('stat_sum', 'sum'),
    x_min=('x_min', 'min'),
    x_max=('x_min', 'max'),
    x_nunique=('x_min', 'nunique'),
).assign(
    stat_wgt_av=lambda df: df['stat_sum'] / df['stat_wgt_sum']
).style.bar(subset='stat_wgt_av')
```

<div style="text-align: right"><a href="#Contents">Back to Contents</a></div>

# Group and aggregate
Grouping and aggregating almost certainly results in a much smaller DataFrame, so do that *first* in one function and *then* add additional columns for plotting in a subsequent function.

```python
def agg_wgt_av(
    df_w_buckets, stat_wgt=None,
    x_var=None, stat_vars=None,
    bucket_col=None, split_cols=None,
):
    """
    Group by bucket and calculate aggregate values in each bucket
    
    df_w_buckets: Result of an 'assign_buckets' function.
        i.e. a DataFrame with a `bucket` column the is Categorical
        with Interval categories that partition a range.
        Rows with missing `bucket` value are excluded from the grouping.
    stat_wgt: Weights for the weighted distributions of stat_vars.
        If None (default) then it is set to 'const' and equal weights are used
        for all rows. Must be non-negative with at least one postive value.
    x_var: Column name of variable that will be plotted on the x axis.
        If None, no x axis variables are calculated.
    stat_vars: 
        If None (default) or empty list, no values are calculated.
    bucket_col: Name of bucket column to group by.
        Must be in df_w_buckets. Default 'bucket'.
    split_cols:
        None (default): Do not split buckets.
        str: Name of column to split buckets by.
    
    Returns: DataFrame with one row per group and aggregate statistics.
    """
    # Set defaults
    if x_var is None:
        x_var_lst = []
    else:
        x_var_lst = [x_var]
    if stat_wgt is None:
        stat_wgt = '__const__'
        df_w_buckets = df_w_buckets.assign(**{stat_wgt: 1})
    if stat_vars is None:
        stat_vars = []
    if bucket_col is None:
        bucket_col = 'bucket'
    split_cols_lst = split_cols
    if split_cols is None:
        split_cols_lst = []
    if isinstance(split_cols, str):
        split_cols_lst = [split_cols]
    
    # Format inputs
    if not isinstance(stat_vars, list):
        stat_vars = [stat_vars]
    
    # Variables for which we want the (weighted) distribution in each bucket
    agg_vars_all = stat_vars
    if x_var is not None and np.issubdtype(df_w_buckets[x_var].dtype, np.number):
        agg_vars_all = [x_var] + agg_vars_all
    # Ensure they are unique (and maintain order)
    agg_vars = pd.Series(agg_vars_all).drop_duplicates()
    
    df_agg = df_w_buckets.groupby(
        # Group by the buckets
        [bucket_col] + split_cols_lst, sort=False
    ).apply(lambda df: pd.Series({
        # Aggregate calculation for rows in each bucket
       'n_obs': df.iloc[:, 0].size,  # It is possible that a bucket contains zero rows
        stat_wgt: df[stat_wgt].sum(),
        **{stat_var + '_wgt_av': np.average(df[stat_var], weights=df[stat_wgt]) 
           for stat_var in agg_vars},
        **{"x_" + func[0]: func[1](df[x_var]) 
           for func in [('min', np.amin), ('max', np.amax)] for x_var in x_var_lst},
    })).sort_index()
    return(df_agg)
```

```python
# Example for lift chart
bucket_var, bucket_wgt = 'Freq_pred_simple', 'Exposure'
x_var, stat_wgt, stat_vars = 'cum_' + bucket_wgt, bucket_wgt, ['Frequency', 'Freq_pred_simple']
tmp7_w_buckets = df.pipe(weighted_quantiles, bucket_var, 4, bucket_wgt)
tmp7_agg_all = tmp7_w_buckets.pipe(agg_wgt_av, stat_wgt, x_var, stat_vars)
assert isinstance(tmp7_agg_all.index, pd.IntervalIndex)
tmp7_agg_all
```

```python
tmp7_agg_split0 = agg_wgt_av(tmp7_w_buckets, stat_wgt, x_var, stat_vars, split_cols='split')
tmp7_agg_split0
```

```python
def combined_hierarchical_index(df1, df2, index_val='__all__'):
    """
    Combine two DataFrames where one of them has a MultiIndex with more levels than the other
    """
    if len(df1.index.names) > len(df2.index.names):
        df_split = df1
        df_all = df2
    else:
        df_split = df2
        df_all = df1
    extra_level_names = list(set(df_split.index.names) - set(df_all.index.names))
    if len(extra_level_names) == 0:
        raise ValueError(
            "\n\tagg_combine: The indices of `df1` and `df2` must be the same"
            "\n\tbut with one of them having at least one fewer level."
        )
    idx_names_original = df_split.index.names
    idx_names_ordered = df_all.index.names + extra_level_names
    df_combined = pd.concat([
        df_split.reorder_levels(idx_names_ordered),
        df_all.reset_index().assign(**{
            extra_level_name: index_val for extra_level_name in extra_level_names
        }).set_index(idx_names_ordered),
    ], axis=0, sort=False).reorder_levels(idx_names_original).sort_index()
    return df_combined
```

```python
def select_hierarchical_levels(df_combined, levels=None, index_val='__all__'):
    """
    Uncombine a DataFrame into one that only contains `levels`
    """
    # Parse arguments
    if levels is None or levels == []:
        levels = [0]
    if isinstance(levels, int) or isinstance(levels, str):
        levels = [levels]
    # Get index names, along with which have been selected
    idx_levels_all = pd.Series(
        df_combined.index.names, name='level_name'
    ).to_frame().assign(
        selected=lambda df: df.index.isin(levels) | df['level_name'].isin(levels)
    )
    # Warning if some levels provided cannot be matched
    unmatched_levels = [level for level in levels if not(
        (level in idx_levels_all.index) or (level in idx_levels_all['level_name'].values)
    )]
    if len(unmatched_levels) > 0:
        warnings.warn(
            "The following levels could not be found in the index "
            f"names of `df_combined`: {unmatched_levels}"
        )
    idx_levels = idx_levels_all[1:]
    # Include or drop the relevant levels
    res = df_combined
    for _, level_row in idx_levels.iterrows():
        if level_row['selected']:
            res = res.loc[
                res.index.get_level_values(level_row['level_name']) != index_val
            ]
        else:
            res = res.xs(index_val, level=level_row['level_name'])
    if len(res.index.names) > 1:
        res.index = res.index.remove_unused_levels()
    return res.sort_index()
```

```python
import itertools
def powerset(iterable):
    """powerset([1,2,3]) --> [] [1,] [2,] [3,] [1,2] [1,3] [2,3] [1,2,3]"""
    s = list(iterable)
    return [list(comb) for comb in itertools.chain.from_iterable(
        itertools.combinations(s, r) for r in range(len(s)+1)
    )]
```

```python
def agg_combined(
    df_w_buckets, agg_function, *agg_args, split_cols=None, **agg_kwargs,
):
    # Format inputs
    split_cols_lst = split_cols
    if split_cols is None:
        split_cols_lst = []
    if isinstance(split_cols, str):
        split_cols_lst = [split_cols]
    
    for set_num, split_set in enumerate(powerset(split_cols_lst)[::-1]):
        df_agg = agg_wgt_av(df_w_buckets, split_cols=split_set, *agg_args, **agg_kwargs)
        if set_num == 0:  # First loop is for all split columns. Instantiate the resulting df
            df_combined = df_agg
        else:
            df_combined = combined_hierarchical_index(df_combined, df_agg)
    return df_combined
```

```python
def assign_hierarchical(df, **col_funcs):
    split_cols_lst = df.index.names[1:]
    for set_num, split_set in enumerate(powerset(split_cols_lst)[::-1]):
        df_selected = select_hierarchical_levels(df, split_set).assign(**col_funcs)
        if set_num == 0:  # First loop is for all split columns. Instantiate the resulting df
            df_combined = df_selected
        else:
            df_combined = combined_hierarchical_index(df_combined, df_selected)
    return df_combined
```

```python
# Examples with various split columns
bucket_var, bucket_wgt = 'Freq_pred_simple', 'Exposure'
x_var, stat_wgt, stat_vars = 'cum_' + bucket_wgt, bucket_wgt, ['Frequency', 'Freq_pred_simple']
tmp71_w_buckets = df.pipe(weighted_quantiles, bucket_var, 4, bucket_wgt)

# Default args: No split or stat_vars
tmp71_combined_none = tmp71_w_buckets.pipe(agg_combined, agg_wgt_av)
tmp71_combined_none
```

```python
# Recreate df with no split
tmp71_combined_all = tmp71_w_buckets.pipe(agg_combined, agg_wgt_av, stat_wgt, x_var, stat_vars)
display(tmp71_combined_all.iloc[:5, :5])
assert tmp71_combined_all.equals(tmp7_agg_all)
print("Correct: Matches previous example as expected")
```

```python
# Recreate df with one split
tmp71_combined_1 = tmp71_w_buckets.pipe(
    agg_combined, agg_wgt_av, stat_wgt, x_var, stat_vars,
    split_cols='split'
)
display(tmp71_combined_1.iloc[:5, :5])
assert select_hierarchical_levels(tmp71_combined_1).equals(tmp7_agg_all)
assert select_hierarchical_levels(tmp71_combined_1, 1).equals(tmp7_agg_split0)
assert select_hierarchical_levels(tmp71_combined_1, 1).index.levels[1].equals(
    tmp7_agg_split0.index.levels[1]
)
print("Correct: Matches previous examples as expected")
```

```python
# Example of assign_hierarchical
col_funcs = {
    'quad_upr': lambda df: df.groupby(df.index.names[:1] + df.index.names[1:-1])[stat_wgt].transform('cumsum'),
    'quad_lwr': lambda df: df.groupby(df.index.names[:1] + df.index.names[1:-1])['quad_upr'].shift(fill_value=0),
}
assign_hierarchical(tmp71_combined_1, **col_funcs).iloc[:5]
# tmp71_combined_1.pipe(assign_hierarchical, **col_funcs).iloc[:5]  # Also using pipe
```

```python
# Create df with multiple splits
tmp71_combined_2 = tmp71_w_buckets.pipe(
    agg_combined, agg_wgt_av, stat_wgt, x_var, stat_vars,
    split_cols=['split', 'Area']
)
display(tmp71_combined_2.iloc[:5, :5])
assert select_hierarchical_levels(tmp71_combined_2, None).equals(tmp7_agg_all)
assert select_hierarchical_levels(tmp71_combined_2, 1).equals(tmp7_agg_split0)
assert select_hierarchical_levels(tmp71_combined_2, ['split', 'Area']).equals(
    tmp71_w_buckets.pipe(
        agg_wgt_av, stat_wgt, x_var, stat_vars,
        split_cols=['split', 'Area']
    )
)
print("Correct: Matches previous examples as expected")
```

```python
warn_obj = None
with warnings.catch_warnings(record=True) as _warn_obj:
    warnings.simplefilter("always")
    _ = select_hierarchical_levels(tmp71_combined_2, [0, 'foo'])
    warn_obj = _warn_obj
# Verify some things
assert len(warn_obj) == 1
assert issubclass(warn_obj[-1].category, UserWarning)
assert 'The following levels could not be found' in str(warn_obj[-1].message)
print("Correct: Warning message shown in appropriate situation")
```

```python
# Further examples
display(select_hierarchical_levels(tmp71_combined_2, 'Area').iloc[:7, :5])
display(select_hierarchical_levels(tmp71_combined_2, ['split', 'Area']).iloc[:7, :5])
```

```python
# Use a missing indicator to cope with missing values
unit_filled, b_map = pd.Series([0, 1, np.nan]).to_frame('val').assign(
    val_miss_ind=lambda df: df.val.isna(),
    val=lambda df: df.val.fillna(-1),
).pipe(divide_n, 'val', 2).pipe(
    all_levels, 'val_miss_ind', bucket_col='bucket_miss_ind', ret_map=True 
)
display(unit_filled)
unit_agg_all = unit_filled.pipe(agg_wgt_av, x_var='val')
display(unit_agg_all)
unit_agg_split = unit_filled.pipe(agg_wgt_av, stat_vars='val', split_cols='val_miss_ind')
display(unit_agg_split)
```

```python
# TODO: Other examples
```

```python

```

<div style="text-align: right"><a href="#Contents">Back to Contents</a></div>

# Plot
For plotting, we switch away from `matplotlib` to `bokeh` because:
- It is cumbersome to use `matplotlib` for plotting both a histogram and line plot on the same axes
- It would be nice to have interactivity (e.g. zooming) in our resulting plots

On the downside, it does require a bit of code to produce the output.

```python
# TODO: Move to util functions

def expand_lims(df, pct_buffer_below=0.05, pct_buffer_above=0.05, include_vals=None):
    """
    Find the range over all columns of df. Then expand these 
    below and above by a percentage of the total range.
    
    df: Consider all values in all columns
    include_vals: Additional values to consider
    
    Returns: Series with rows 'start' and 'end' of the expanded range
    """
    # If a Series is passed, convert it to a DataFrame
    try:
        df = df.to_frame()
    except:
        pass
    # Case where df has no columns, just fill in default vals
    if df.shape[1] == 0:
        res_range = pd.Series({'start': 0, 'end': 1})
        return(res_range)
    if include_vals is None:
        include_vals = []
    if not isinstance(include_vals, list):
        include_vals = [include_vals]
    
    res_range = pd.concat([
        df.reset_index(drop=True),
        # Add a column of extra values to the DataFrame to take these into account
        pd.DataFrame({'_extra_vals': include_vals}),
    ], axis=1).apply(
        # Get the range (min and max) over the DataFrame
        ['min', 'max']).agg({'min': 'min', 'max': 'max'}, axis=1).agg({
        # Expanded range
        'start': lambda c: c['max'] - (1 + pct_buffer_below) * (c['max'] - c['min']),
        'end': lambda c: c['min'] + (1 + pct_buffer_above) * (c['max'] - c['min']),
    })
    return(res_range)
```

```python
# Example lift chart
bucket_var, bucket_wgt = 'Freq_pred_simple', 'Exposure'
x_var, stat_wgt, stat_vars = 'cum_' + bucket_wgt, bucket_wgt, ['Frequency', 'Freq_pred_simple']
tmp8_agg = df.pipe(
    weighted_quantiles, bucket_var, 10, bucket_wgt
).pipe(
    agg_wgt_av, stat_wgt, x_var, stat_vars
)
tmp8_agg.assign(
    # Get the coordinates for plot: interval edges
    x_left=lambda df: df.index.left,
    x_right=lambda df: df.index.right,
    x_point=lambda df: (df['x_right'] + df['x_left'])/2.,
)
```

```python
# Functions to set the x-axis edges `x_left` and `x_right`
def x_edges_min_max(df_agg):
    """
    Set the x-axis edges to be the min and max values of `x_var`.
    Does not make sense to use this option when min and max are not numeric.
    This might result in zero width intervals, in which case a warning is given.
    """
    if not np.issubdtype(df_agg['x_min'].dtype, np.number):
        raise ValueError(
            "\n\tx_edges_min_max: This method can only be used when"
            "\n\tx_min and x_max are numeric data types."
        )
        
    if (df_agg['x_min'] == df_agg['x_max']).any():
        warning(
            "x_edges_min_max: At least one bucket has x_min == x_max, "
            "so using this method will result in zero width intervals."
        )
    
    res = df_agg.assign(
        # Get the coordinates for plot: interval edges
        x_left=lambda df: df['x_min'],
        x_right=lambda df: df['x_max'],
    )
    return(res)


def x_edges_interval(df_agg, bucket_col='bucket'):
    """Set the x-axis edges to be the edges of the bucket interval"""
    res = df_agg.assign(
        x_left=lambda df: [intval.left for intval in df.index.get_level_values(bucket_col)],
        x_right=lambda df: [intval.right for intval in df.index.get_level_values(bucket_col)],
    )
    return(res)


def x_edges_unit(df_agg, bucket_col='bucket'):
    """
    Set the x-axis edges to be the edges of equally spaced intervals
    of width 1.
    """
    res = df_agg.assign(
        x_left=lambda df: pd.Categorical(df.index.get_level_values(bucket_col)).codes,
        x_right=lambda df: df['x_left'] + 1,
    )
    return(res)
```

```python
# Functions to set the x-axis point
def x_point_mid(df_agg):
    """Set the x_point to be mid-way between x_left and x_right"""
    res = df_agg.assign(
        x_point=lambda df: (df['x_left'] + df['x_right']) / 2.
    )
    return(res)

def x_point_wgt_av(df_agg, x_var):
    """
    Set the x_point to be the weighted average of x_var within the bucket,
    weighted by stat_wgt.
    """
    if not (x_var + '_wgt_av') in df_agg.columns:
        raise ValueError(
            "\n\tx_point_wgt_av: This method can only be used when"
            "\n\tthe weighted average has already been calculated."
        )
    
    res = df_agg.assign(
        x_point=lambda df: df[x_var + '_wgt_av']
    )
    return(res)
```

```python
def x_label_none(df_agg):
    res = df_agg.copy()
    if 'x_label' in df_agg.columns:
        res = res.drop(columns='x_label')
    return(res)

def x_label_map(df_agg, bucket_map, bucket_col='bucket'):
    res = df_agg.assign(
        x_label=lambda df: pd.Categorical(
            df.index.get_level_values(bucket_col)
        ).rename_categories(
            bucket_map.set_index('interval')['val']
        )
    )
    return(res)
```

```python
# TODO: Fill missing values in these functions
# def y_quad_cumulative(df_agg, stat_wgt, bucket_col='bucket'):
#     res = df_agg.assign(
#         quad_upr=lambda df: df.groupby([bucket_col])[stat_wgt].transform('cumsum'),
#         quad_lwr=lambda df: df.groupby([bucket_col])['quad_upr'].shift(fill_value=0).fillna(method='ffill'),
#     )
#     return(res)

def groupby_first_not_last(df):
    return df.groupby(df.index.names[:1] + df.index.names[1:-1])

def y_quad_cumulative(df_agg, stat_wgt):
    res = df_agg.pipe(assign_hierarchical, **{
        'quad_upr': lambda df: groupby_first_not_last(df)[stat_wgt].transform('cumsum'),
        'quad_lwr': lambda df: groupby_first_not_last(df)['quad_upr'].shift(fill_value=0),
    })
    return res

def y_quad_area(df_agg, stat_wgt):
    res = df_agg.pipe(assign_hierarchical, **{
        'x_width': lambda df: df['x_right'] - df['x_left'],
        'quad_upr': lambda df: groupby_first_not_last(df)[stat_wgt].transform('cumsum') / df['x_width'],
        'quad_lwr': lambda df: groupby_first_not_last(df)['quad_upr'].shift(fill_value=0),
    })
    return res

def y_quad_proportion(df_agg, stat_wgt):
    res = df_agg.pipe(assign_hierarchical, **{
        'quad_upr': lambda df: (
            groupby_first_not_last(df)[stat_wgt].transform('cumsum') / 
            groupby_first_not_last(df)[stat_wgt].transform('sum')
        ),
        'quad_lwr': lambda df: groupby_first_not_last(df)['quad_upr'].shift(fill_value=0),
    })
    return res
```

```python
# Examples
stat_wgt='Exposure'
tmp71_combined_1.pipe(
#     x_edges_interval
    x_edges_min_max
).pipe(
#     y_quad_cumulative, stat_wgt
#     y_quad_area, stat_wgt
    y_quad_proportion, stat_wgt
)
```

```python
pipe_funcs_df = pd.DataFrame(
    columns=['task', 'func', 'alias'],
    data = [
        ('x_edges', x_edges_interval, ['interval']),
        ('x_edges', x_edges_min_max, ['min_max', 'range']),
        ('x_edges', x_edges_unit, ['unit']),
        ('x_point', x_point_mid, ['mid']),
        ('x_point', x_point_wgt_av, ['wgt_av']),
        ('x_label', x_label_none, ['none']),
        ('x_label', x_label_map, ['map']),
        ('y_quad', y_quad_cumulative, ['cum']),
        ('y_quad', y_quad_area, ['area']),
        ('y_quad', y_quad_proportion, ['prop']),
    ],
).assign(
    name=lambda df: df['func'].apply(lambda f: f.__name__),
    arg_names=lambda df: df['func'].apply(
        lambda f: [
            arg_name for arg_name, val 
            in inspect.signature(f).parameters.items()
        ][1:]  # Not the "df" argument
    ),
    req_arg_names=lambda df: df['func'].apply(
        lambda f: [
            arg_name for arg_name, val 
            in inspect.signature(f).parameters.items()
            if val.default == inspect.Parameter.empty
        ][1:]  # Not the "df" argument
    ),
).set_index(['task', 'name'])

pipe_funcs_df
```

```python
def get_pipeline_func(
    task, search_term,
    kwarg_keys=None, calling_func='',
    pipe_funcs_df=pipe_funcs_df
):
    """
    TODO: Write docstring <<<<<<<<<<<<<
    """
    # Set defaults
    if kwarg_keys is None:
        kwarg_keys = []
    
    # Find function row
    task_df = pipe_funcs_df.loc[task,:]
    func_row = task_df.loc[task_df.index == search_term, :]    
    if func_row.shape[0] != 1:
        func_row = task_df.loc[[search_term in ali for ali in task_df.alias], :]
    if func_row.shape[0] != 1:
        raise ValueError(
            f"\n\t{calling_func}: Cannot find '{search_term}' within the"
            f"\n\tavailable '{task}' pipeline functions."
        )
        
    # Check required arguments are supplied
    for req_arg in func_row['req_arg_names'][0]:
        if not req_arg in kwarg_keys:
            raise ValueError(
                f"\n\t{calling_func}: To use the '{search_term}' as a '{task}' pipeline"
                f"\n\tfunction, you must specify '{req_arg}' as a keyword argument."
            )
    return(func_row['func'][0], func_row['arg_names'][0])
```

```python
# Examples
# get_pipeline_func('x_edges', 'min_max')
# get_pipeline_func('x_edges', 'x_edges_interval')
# get_pipeline_func('x_point', 'foo', calling_func='from_here')  # Throws an error
get_pipeline_func('x_point', 'wgt_av', ['x_var'])
```

```python
def add_coords(
    df_agg_all,
    x_edges=None, x_point=None, x_label=None,
    y_quad=None,
    **kwargs,
):
    """
    Given a DataFrame where each row is a bucket, add x-axis 
    properties to be used for plotting. See pipe_funcs_df for 
    available options.
    
    x_edges: How to position the x-axis edges.
        Default: 'interval'
    x_point: Where to position each bucket point on the x-axis.
        Default: 'mid'
    x_label: Option for x-axis label.
        Default: 'none'
    y_quad: How to plot the histogram quads.
        Default: 'cum'
    **kwargs: Additional arguments to pass to the functions.
    """
    # Set variables for use throughout the function
    calling_func = 'add_coords'
    kwarg_keys = list(kwargs.keys())
    
    # Set defaults
    if x_edges is None:
        x_edges = 'interval'
    if x_point is None:
        x_point = 'mid'
    if x_label is None:
        x_label = 'none'
    if y_quad is None:
        y_quad = 'cum'
    
    # Get pipeline functions
    full_func, arg_names = get_pipeline_func('x_edges', x_edges, kwarg_keys, calling_func)
    x_edges_func = functools.partial(full_func, **{
        arg_name: kwargs[arg_name] for arg_name in set(arg_names).intersection(set(kwarg_keys))
    })
    
    full_func, arg_names = get_pipeline_func('x_point', x_point, kwarg_keys, calling_func)
    x_point_func = functools.partial(full_func, **{
        arg_name: kwargs[arg_name] for arg_name in set(arg_names).intersection(set(kwarg_keys))
    })

    full_func, arg_names = get_pipeline_func('x_label', x_label, kwarg_keys, calling_func)
    x_label_func = functools.partial(full_func, **{
        arg_name: kwargs[arg_name] for arg_name in set(arg_names).intersection(set(kwarg_keys))
    })
    
    full_func, arg_names = get_pipeline_func('y_quad', y_quad, kwarg_keys, calling_func)
    y_quad_func = functools.partial(full_func, **{
        arg_name: kwargs[arg_name] for arg_name in set(arg_names).intersection(set(kwarg_keys))
    })
    
    # Apply the functions
    res = df_agg_all.pipe(
        lambda df: x_edges_func(df)
    ).pipe(
        lambda df: x_point_func(df)
    ).pipe(
        lambda df: x_label_func(df)
    ).pipe(
        lambda df: y_quad_func(df)
    )
    return(res)
```

```python
def create_bplot(
    df_for_plt, stat_wgt, stat_vars,
    stack=None,
    cols=bokeh.palettes.Dark2[8],
):
    """Create bucket plot object from aggregated data"""
    # Set defaults
    if not isinstance(stack, list):
        stack = [stack, stack]
    if len(stack) != 2:
        raise ValueError("`stack` not correct")
    for idx, stk in enumerate(stack):
        if stk == 0:
            stack[idx] = None
        if stk == 1:
            stack[idx] = 1
    
    # Set up the figure
    bkp = bokeh.plotting.figure(
        title="Bucket plot", x_axis_label="X-axis name", y_axis_label=stat_wgt, 
        tools="reset,box_zoom,pan,wheel_zoom,save", background_fill_color="#fafafa",
        plot_width=800, plot_height=500
    )

    # Plot the histogram squares...
    df_for_quads = select_hierarchical_levels(df_for_plt, stack[0])
    bkp.quad(
        top=df_for_quads['quad_upr'], bottom=df_for_quads['quad_lwr'],
        left=df_for_quads['x_left'], right=df_for_quads['x_right'],
        fill_color="khaki", line_color="white", legend_label="Weight"
    )
    # ...at the bottom of the graph
    bkp.y_range = bokeh.models.ranges.Range1d(
        **expand_lims(df_for_quads[['quad_upr', 'quad_lwr']], 0, 1.2)
    )

    bkp.legend.location = "top_left"
    bkp.legend.click_policy="hide"

    # Plot the weight average statistic points joined by straight lines
    df_for_stat_vars = select_hierarchical_levels(df_for_plt, stack[1])
    # Add a second index level, if it does not have one already
    if len(df_for_stat_vars.index.names) == 1:
        df_for_stat_vars = df_for_stat_vars.assign(
            __dummy_level__='__all__'
        ).set_index([df_for_stat_vars.index, '__dummy_level__'])
    
    # Set up the secondary axis
    bkp.extra_y_ranges['y_range_2'] = bokeh.models.ranges.Range1d(
        **expand_lims(df_for_stat_vars[[stat_var + '_wgt_av' for stat_var in stat_vars]])
    )
    bkp.add_layout(bokeh.models.axes.LinearAxis(
        y_range_name='y_range_2',
        axis_label="Weighted average statistic"
    ), 'right')
    
    for var_num, stat_var in enumerate(stat_vars):
        for split_level in df_for_stat_vars.index.get_level_values(1).drop_duplicates():
            # The following parameters need to be passed to both circle() and line()
            stat_line_args = {
                'x': df_for_stat_vars.xs(split_level, level=1)['x_point'],
                'y': df_for_stat_vars.xs(split_level, level=1)[stat_var + '_wgt_av'],
                'y_range_name': 'y_range_2',
                'color': cols[var_num],
                'legend_label': stat_var,
            }
            bkp.circle(**stat_line_args, size=4)
            bkp.line(**stat_line_args)
    
    return(bkp)
```

```python
# Example lift chart: Split into data processing and then plotting
# Create data
bucket_var, bucket_wgt = 'Freq_pred_simple', 'Exposure'
x_var, stat_wgt, stat_vars = 'cum_' + bucket_wgt, bucket_wgt, ['Frequency', 'Freq_pred_simple']
tmp8_agg = df.pipe(
    weighted_quantiles, bucket_var, 10, bucket_wgt
).pipe(
    agg_combined, agg_wgt_av, stat_wgt, x_var, stat_vars,
    split_cols=['split', 'Area']
)
tmp8_agg.head()
```

```python
# Plot
bkp = tmp8_agg.pipe(
    add_coords, stat_wgt=stat_wgt, #y_quad='prop', x_edges='unit'
).pipe(
    create_bplot, stat_wgt, stat_vars,
    stack=["Area", "split"]
)
bokeh.plotting.show(bkp)
```

<div style="text-align: right"><a href="#Contents">Back to Contents</a></div>

# Worked examples

```python
help(add_coords)
```

### DrivAge

```python
x_var, stat_wgt, stat_vars = 'DrivAge', 'Exposure', ['Frequency', 'Freq_pred_simple', 'Freq_pred_veh']
bucket_var = 'DrivAge'
df_for_plt = df.pipe(
#     divide_n, bucket_var, 10
#     all_levels, bucket_var
#     custom_width, bucket_var, 3, 17.5
    custom_width, bucket_var, 3, 17.5, None, 68.5
).pipe(
    agg_combined, agg_wgt_av, stat_wgt, x_var, stat_vars,
    # split_cols='split'
).pipe(
    add_coords, stat_wgt=stat_wgt, bucket_col='bucket',
    y_quad='area', 
    # x_edges='unit'
)
bkp = create_bplot(df_for_plt, stat_wgt, stat_vars)
bkp.legend.location = "top_right"
bokeh.plotting.show(bkp)
```

### Density

```python
x_var, stat_wgt, stat_vars = 'Density', 'Exposure', ['Frequency', 'Freq_pred_simple', 'Freq_pred_veh']
bucket_var, bucket_wgt = x_var, stat_wgt
df_for_plt = df.pipe(
#     divide_n, bucket_var, 10
    custom_width, bucket_var, 100, 0.5, None, 1000
#     weighted_quantiles, bucket_var, 10, bucket_wgt
).pipe(
    agg_combined, agg_wgt_av, stat_wgt, x_var, stat_vars,
    split_cols='split'
).pipe(
    add_coords, stat_wgt=stat_wgt, bucket_col='bucket',
#     x_edges='min_max', x_point='wgt_av', x_var=x_var,
#     y_quad='area',
    x_edges='unit',
)
bkp = create_bplot(df_for_plt, stat_wgt, stat_vars)
#bkp.legend.location = "top_right"
bokeh.plotting.show(bkp)
```

### Area and Region

```python
x_var, stat_wgt, stat_vars = 'Area', 'Exposure', ['Frequency', 'Freq_pred_simple', 'Freq_pred_veh']
bucket_var, bucket_wgt = x_var, stat_wgt
df_for_plt = df.pipe(
    all_levels, bucket_var
).pipe(
    agg_combined, agg_wgt_av, stat_wgt, x_var, stat_vars,
    split_cols='split',
).pipe(
    add_coords, stat_wgt=stat_wgt,
)
bkp = create_bplot(df_for_plt, stat_wgt, stat_vars)
bokeh.plotting.show(bkp)
```

```python
# Interesting case: Group a nominal variable first by all_levels and then by 
# weighted_quantiles, to group the levels in order of increasing stat_wgt_av.
x_var, stat_wgt, stat_vars = 'Region', 'Exposure', ['Frequency', 'Freq_pred_simple', 'Freq_pred_veh']
bucket_var, bucket_wgt = x_var, stat_wgt
df_for_plt = df.pipe(
    all_levels, bucket_var, bucket_col='split'
).pipe(
    agg_wgt_av, stat_wgt, x_var, stat_vars, bucket_col='split'
).pipe(
    weighted_quantiles, 'Frequency_wgt_av', 5, stat_wgt
).pipe(
    agg_combined, agg_wgt_av, stat_wgt, 'Frequency_wgt_av',
    stat_vars=['Frequency_wgt_av', 'Freq_pred_simple_wgt_av'],
    # split_cols='split' # NOT CURRENTLY WORKING PROPERLY
).pipe(
    add_coords, stat_wgt=stat_wgt, bucket_col='bucket',
)
bkp = create_bplot(df_for_plt, stat_wgt, stat_vars=['Frequency_wgt_av', 'Freq_pred_simple_wgt_av'])
bokeh.plotting.show(bkp)
```

```python

```

<div style="text-align: right"><a href="#Contents">Back to Contents</a></div>

# Rough work

```python
# Example of allowing an additional var for grouping as missing_ind
miss_ind_grpd = pd.Series([0, 1, np.nan]).to_frame('val').assign(
    missing_ind=lambda df: df['val'].isna(),  # Get missing_ind
    val_filled=lambda df: df['val'].fillna(df['val'].min()),   # Fill missing vals
).pipe(
    lambda df: divide_n(df, 'val_filled', 3)
).groupby(['bucket', 'missing_ind']).agg(
    n_rows=('bucket', 'size'),
)
display(miss_ind_grpd)
miss_ind_grpd.unstack(fill_value=0)
```

<div style="text-align: right"><a href="#Contents">Back to Contents</a></div>
