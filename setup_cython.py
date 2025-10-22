from setuptools import setup
from setuptools.extension import Extension
import numpy as np
import os
import sys

# Check for Cython installation
try:
    from Cython.Build import cythonize
    USE_CYTHON = True
except ImportError:
    USE_CYTHON = False
    print("WARNING: Cython not found. Will attempt to compile from .c file if it exists.")

# Define the source file and extension setup
ext = '.pyx' if USE_CYTHON else '.c'
extensions = [
    Extension(
        name="cython_writer",
        sources=["cython_writer" + ext],
        # Include NumPy headers for efficient array handling in Cython
        include_dirs=[np.get_include()],
        # Additional libraries might be needed for GDAL/Rasterio linking
        # but for this simplified wrapper, only NumPy is strictly required for the array types.
    )
]

if USE_CYTHON:
    # Use cythonize to compile the .pyx file
    setup(
        name='cython_writer',
        ext_modules=cythonize(extensions, compiler_directives={'language_level': "3"}),
    )
else:
    # Fallback to pure C compilation if Cython is not installed (requires cython_writer.c to exist)
    setup(
        name='cython_writer',
        ext_modules=extensions,
    )

print("\n--- Cython setup complete. Run 'python setup.py build_ext --inplace' to compile. ---\n")
