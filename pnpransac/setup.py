import sys
import os
import numpy
from distutils.core import setup
from distutils.extension import Extension

from Cython.Distutils import build_ext
from Cython.Build import cythonize

# where to find opencv headers and libraries

conda_env = os.environ['CONDA_PREFIX']

cv_include_dir = conda_env + '/include/opencv4'
cv_library_dir = conda_env + '/lib/opencv4'

ext_modules = [
    Extension(
        "pnpransac",
        sources=["pnpransacpy.pyx"],
        language="c++",
        include_dirs=[cv_include_dir, numpy.get_include()],
        library_dirs=[cv_library_dir],
        libraries=['opencv_core','opencv_calib3d'],
        extra_compile_args=['-fopenmp','-std=c++11'],
    )
]

setup(
    name='pnpransac',
    cmdclass={'build_ext': build_ext},
    ext_modules=cythonize(ext_modules),
    )
