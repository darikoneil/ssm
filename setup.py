#! /usr/bin/env python
import os
import numpy as np
from Cython.Build import cythonize
from setuptools import setup
from setuptools.extension import Extension

# Ensure builds run relative to the project root
os.chdir(os.path.dirname(os.path.abspath(__file__)))

USE_OPENMP = os.environ.get("USE_OPENMP", "").lower() in ("1", "true", "yes")
print("USE_OPENMP", USE_OPENMP)

extensions = [
    Extension(
        "ssm.cstats",
        sources=["ssm/cstats.pyx"],
        language="c++",
        include_dirs=[np.get_include()],
        extra_compile_args=["-fopenmp"] if USE_OPENMP else [],
        extra_link_args=["-fopenmp"] if USE_OPENMP else [],
    )
]

# metadata in pyproject.toml will be used by setuptools.build_meta;
# we only provide ext_modules here
setup(ext_modules=cythonize(extensions))