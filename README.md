<a name="top"></a>

<!-- Remember to change this link to ensure it matches the current branch! -->
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/A-Breeze/bucketplot/setup?urlpath=lab)

# bucketplot for Python
Cut your data into buckets for effective visualisations.

<!--This table of contents is maintained *manually*-->
## Contents
1. [Setup](#Setup)
    - [Start Binder instance](#Start-Binder-instance)
    - [Development environment](#Development-environment)
1. [Structure of the repo](#Structure-of-the-repo)
1. [Tasks](#Tasks)
    - [Development installation](#Development-installation)
    - [Run automated tests](#Run-automated-tests)
    - [Build package](#Build-package)
    - [Compile development notebooks](#Compile-development-notebooks)
1. [Future ideas](#Future-ideas)
1. [Further notes and troubleshooting](#Further-notes-and-troubleshooting)

<p align="right"><a href="#top">Back to top</a></p>

## Setup
This document describes how to run the repo using JupyterLab on Binder. It *should* be possible to run the code in JupyterLab (or another IDE) from your own machine (i.e. not on Binder), but this hasn't been tested. Follow the bullet point to install it *Locally on Windows* in [Development environment](#Development-environment) below.

All console commands are **run from the root folder of this project** unless otherwise stated.

### Start Binder instance
Click the *Launch Binder* button at the [top](#top) of this page.

### Development environment
The development requirements consist of the package dependencies, plus IDE, plus extra packages useful during development. They can be automatically installed into a conda-env as follows.
- **Binder**: A conda-env is created automatically from `binder/environment.yml` in Binder is called `notebook` by default. Unless otherwise stated, the below console commands assume the conda-env is activated, i.e.:
    ```
    conda activate notebook
    ```
- **Locally** (on Windows):
    ```
    conda env create -f binder\environment.yml --force
    conda activate buckplot_dev_env
    ```

<p align="right"><a href="#top">Back to top</a></p>

## Structure of the repo
**TODO**: Describe the structure

<p align="right"><a href="#top">Back to top</a></p>

## Tasks
### Development installation
While developing the package, we can install it from the local code (without needing to build and then install) as follows:
```
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### Run automated tests
Ensure the package is installed.
```
pytest
```

### Build package
The following will create a *source* distribution and a *wheel* distribution out of the Python package (given it includes a `setup.py`), and puts the resulting files in `build/` (for some intermediate files from the wheel build) and `dist/` (for the final source and wheel distributions) subfolders.
```
python setup.py sdist bdist_wheel
```

### Install built package
```
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install ./dist/bucketplot-*.whl  # For the wheel (including dependencies)
# For the source, we first need to install dependencies
pip install -r requirements.txt
pip install ./dist/bucketplot-*.tar.gz
```

### Compile development notebooks
The development notebooks have been saved in `jupytext` markdown format, so they can be executed (to produce the outputs) and compiled (to `ipynb` format) as follows:
```
jupytext --to notebook --output development/compiled/bucketplot-motor-claims.ipynb --execute development/bucketplot-motor-claims.md
```

<p align="right"><a href="#top">Back to top</a></p>

## Further notes and troubleshooting
### Using Binder for development
- Advantage: This will run it in the browser, so there is no prerequisite of software installed on your computer (other than a compatible browser). 
- Disadvantages:
    - Security is *not* guaranteed within Binder (as per [here](https://mybinder.readthedocs.io/en/latest/faq.html#can-i-push-data-from-my-binder-session-back-to-my-repository)), so I'll be pushing Git from another location, which involves some manual copy-paste.
    - The package environment has to be restored each time, which takes some time.
    - On starting Binder for the first time for this project, the launch process failed *twice* (and the error shown in the log was not immediately obvious), but suceeded on the third attempt without making any changes. I'm not sure how Binder works (e.g. whether it stores previously built images), so this is difficult to debug, but clearly some instability going on.

<p align="right"><a href="#top">Back to top</a></p>

## Future ideas
Backlog of all possible ideas, big or small, high priority or low.

**Work in progress**: This section will develop over time.

<p align="right"><a href="#top">Back to top</a></p>
