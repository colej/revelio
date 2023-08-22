# revelio
Code for determining the dominant frequency of variability in multi-filter photometric time series using non-parametric periodograms



## Installation

First, clone revelio into a location where your PYTHONPATH points.

Before installing, you must modify the contents of the yml file! At the bottom, the prefix variable currently says: /YOUR/PATH/TO/miniconda3... This needs to be changed to reflect the location of miniconda3 (or anaconda3) in your directory structure.

To install, use miniconda3 or anaconda3, and run: conda env create -f revelio.yml

    python setup.py build_ext --inplace
    python setup.py install
