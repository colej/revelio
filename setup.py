from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import numpy as np

# List your Cython extension modules

ext_aov = Extension('revelio.periodograms.aov',
                    sources=['revelio/periodograms/aov.pyx'],
                    include_dirs=[np.get_include()],
                    language='c',
                    libraries=[],
                    extra_compile_args=[]
                   )

ext_ce = Extension('revelio.periodograms.conditional_entropy',
                    sources=['revelio/periodograms/conditional_entropy.pyx'],
                    include_dirs=[np.get_include()],
                    language='c',
                    libraries=[],
                    extra_compile_args=[]
                   )

cython_extensions = [ ext_aov, ext_ce ]

# Automatically include all Python packages
packages = find_packages()

setup(
    name='revelio',
    version='0.1.0',
    description='Time-series analysis tools for multi-filter photometric astronomical datasets.',
    author='Cole Johnston',
    author_email='cole.johnston@ru.nl',
    url='https://github.com/colej/revelio',
    packages=packages,
    install_requires=[
        # List your project's dependencies here
    ],
    ext_modules=cythonize(cython_extensions),
    classifiers=[
        # Choose appropriate classifiers from:
        # https://pypi.org/classifiers/
    ],
)
