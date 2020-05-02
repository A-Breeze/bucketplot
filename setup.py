"""
Describe the package for distribution

Sections:
1. Create the values you want to populate setup() with
2. Define setup() with the values you created
"""

#################
# Create values #
#################
import io
import os
from setuptools import setup

import bucketplot

# Package meta-data
NAME = 'bucketplot'
DESCRIPTION = 'Cut your data into buckets for effective visualisations.'
URL = 'project URL TBA'
EMAIL = 'maintainer@email.TBA'
AUTHOR = 'author name TBA'

# What packages are required for this module to be executed?
def list_reqs(fname='requirements.txt'):
    with open(fname) as fd:
        return fd.read().splitlines()

# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file
here = os.path.abspath(os.path.dirname(__file__))
try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

#######################
# Put it into setup() #
#######################
setup(
    # Package metadata
    name=NAME,
    version=bucketplot.__version__,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    url=URL,

    # Package contents and dependencies
    py_modules=[NAME],  # For a package that consists of just one module
    # packages=setuptools.find_packages(exclude=('tests',)),
    install_requires=list_reqs(),
    extras_require={},
    # package_data={NAME: ['VERSION',]},
    include_package_data=True,

    # Additional info
    license='MIT',
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
    ],
)
