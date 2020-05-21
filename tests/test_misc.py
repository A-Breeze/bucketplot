"""Tests on the package that don't fit in any other category"""

import bucketplot as bplt

def test_package_version():
    """
    Currently, the package version is coded in two seperate places:
    - Within the code module
    - In the setup.py
    This test is an additional place where the version is hard-coded
    to act as a prompt to remember to *manually* keep the other two
    locations in sync.
    """
    assert bplt.__version__ == '0.1.0'
