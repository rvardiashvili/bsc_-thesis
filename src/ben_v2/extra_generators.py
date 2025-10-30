"""
This file contains functions that generate extra data to be written to disk.
"""

import numpy as np
from .utils import NEW_LABELS

def calculate_ndvi(probabilities, **kwargs):
    """
    Calculates the Normalized Difference Vegetation Index (NDVI).

    This is just an example, a real NDVI calculation would require the original bands.
    """
    # This is a dummy NDVI calculation based on probabilities.
    # A real implementation would need access to the original band data.
    vegetation_prob = probabilities[:, :, NEW_LABELS.index('Natural grassland and sparsely vegetated areas')] 
    bare_prob = probabilities[:, :, NEW_LABELS.index('Urban fabric')]
    
    # Avoid division by zero
    denominator = vegetation_prob + bare_prob
    denominator[denominator == 0] = 1e-6
    
    ndvi = (vegetation_prob - bare_prob) / denominator
    return ndvi
