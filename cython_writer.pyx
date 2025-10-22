# distutils: extra_link_args=-Wl,-rpath,$(VIRTUAL_ENV)/lib
# distutils: include_dirs=
# cython: language_level=3

import numpy as np
cimport numpy as np
from rasterio.windows import Window

# Declare the DatasetWriter object type from rasterio's C API (or, more safely,
# rely on the Python object's .write method, which is already C-optimized).

# The Python function signature we want to expose (6 arguments)
def fast_write_chunk(dst, 
                     np.ndarray data, 
                     int col_off, 
                     int row_off, 
                     int width, 
                     int height):
    """
    Fixed version: handles single-band writes correctly.
    """
    try:
        data_2d = np.squeeze(data)
        if data_2d.ndim > 2:
            raise ValueError(f"Input array for single-band write is still {data_2d.ndim}D after squeeze.")

        window = Window(col_off, row_off, width, height)

        # Instead of creating (1, H, W) and passing band=1,
        # just let Rasterio infer it's one band from shape.
        dst.write(data_2d, window=window)

    except Exception as e:
        shape_info = getattr(data, 'shape', 'Unknown Shape')
        raise RuntimeError(f"safe_write_with_optional_cy failed: source shape {shape_info} -> {e}")

# The function above correctly defines the 6-argument interface. 
# The issue in your environment was that the raw C function (which takes 14 args) 
# was being exposed instead of this Python wrapper.
