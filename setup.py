from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize(["OffAxisProjectionCythonized/PurePython.pyx",
                           "OffAxisProjectionCythonized/OffAxisProjectionCythonized.pyx"],
                          annotate=True),
    include_dirs=[numpy.get_include()]
)