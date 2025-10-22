"""
Defines the PyTorch Dataset and data loading functions for BigEarthNet patches,
optimized for robust patch-wise file reading (reading all bands in one operation).

This script now also includes the core functions for reading large data chunks and 
cutting them into smaller patches for inference.
"""
from pathlib import Path
from typing import List, Tuple, Dict, Any, Union
from collections import defaultdict

import torch
import rasterio
from rasterio.windows import Window
from torch.utils.data import Dataset
import numpy as np 
from rasterio.enums import Resampling

from config import PATCH_SIZE, BANDS as CONFIG_BANDS_KEY, CHUNK_SIZE
from utils import STANDARD_BANDS 

# --- Constants for Band Reading ---
# Use the band set defined in config.py (10 or 12)
BANDS = STANDARD_BANDS[CONFIG_BANDS_KEY]
NUM_BANDS = len(BANDS)

# --- Utility Functions for Chunk Reading ---

def _find_band_path(tile_folder: Path, band_name: str) -> Path | None:
    """Finds the path for a given band in the tile folder (supports .jp2 and .tif)."""
    # NOTE: This function is safe because it only runs once in the main process during init.
    for ext in ['.jp2','.tif']:
        # Assuming filename structure is like S2A_MSIL1C_..._B02.jp2 or similar
        candidate = next(tile_folder.glob(f"*{band_name}*{ext}"), None)
        if candidate:
            return candidate
    return None

def _read_all_bands_for_chunk(
    tile_folder: Path,
    r_start: int, 
    c_start: int, 
    W_chunk: int, 
    H_chunk: int
) -> np.ndarray:
    """
    Reads a single chunk (sub-window) from all required bands for a tile.
    
    Args:
        tile_folder (Path): The directory containing the Sentinel-2 band files.
        r_start, c_start (int): Top-left row and column coordinate of the chunk.
        W_chunk, H_chunk (int): Width and height of the chunk.

    Returns:
        np.ndarray: The stacked chunk data (C, H_chunk, W_chunk) as float32.
    """
    
    window = Window(c_start, r_start, W_chunk, H_chunk)
    band_data_list = []

    for band_name in BANDS:
        band_path = _find_band_path(tile_folder, band_name)
        if not band_path:
            raise FileNotFoundError(f"Missing required band file: {band_name} in {tile_folder}")
        
        with rasterio.open(band_path) as src:
            # Read the specific window
            band_data = src.read(
                1, 
                window=window, 
                out_shape=(H_chunk, W_chunk),
                resampling=Resampling.nearest 
            )
            band_data_list.append(band_data.astype(np.float32))

    # Stack the bands into (C, H, W) format
    return np.stack(band_data_list, axis=0)

# --- Main Public Functions ---

def read_chunk_data(tile_folder: Path, BANDS: List[str], r_start: int, c_start: int, W_chunk: int, H_chunk: int) -> np.ndarray:
    """
    Public wrapper to read a single chunk of Sentinel-2 data.
    """
    print(f"  📂 Reading chunk at (R:{r_start}, C:{c_start}) of size {H_chunk}x{W_chunk}...")
    # NOTE: We use the local _read_all_bands_for_chunk which references the BANDS list defined above
    return _read_all_bands_for_chunk(tile_folder, r_start, c_start, W_chunk, H_chunk)

def cut_into_patches(img_chunk: np.ndarray, PATCH_SIZE: int) -> Tuple[np.ndarray, List[Tuple[int, int]], int, int, int]:
    """
    Cuts the larger image chunk into smaller, overlapping patches for inference.
    (Currently implemented as non-overlapping for simplicity).
    
    Args:
        img_chunk (np.ndarray): The input chunk data (C, H, W).
        PATCH_SIZE (int): The side length of the square patch (e.g., 120).
        
    Returns:
        Tuple[np.ndarray, List[Tuple[int, int]], int, int, int]:
            - np.ndarray: Patches array (N, C, H_p, W_p).
            - List[Tuple[int, int]]: (r_start, c_start) coordinates of each patch within the chunk.
            - int: Cropped height.
            - int: Cropped width.
            - int: Patch size.
    """
    C, H, W = img_chunk.shape
    
    # Simple non-overlapping patching for the entire chunk
    patches = []
    coords = [] # (r_start_within_chunk, c_start_within_chunk)
    
    for r_start in range(0, H, PATCH_SIZE):
        for c_start in range(0, W, PATCH_SIZE):
            r_end = min(r_start + PATCH_SIZE, H)
            c_end = min(c_start + PATCH_SIZE, W)

            # Check if the patch is full size (discard partial border patches for now)
            if r_end - r_start == PATCH_SIZE and c_end - c_start == PATCH_SIZE:
                patch = img_chunk[:, r_start:r_end, c_start:c_end]
                patches.append(patch)
                coords.append((r_start, c_start))

    if not patches:
        raise ValueError("Chunk size is too small or patching logic failed to create patches.")
        
    patches_array = np.stack(patches, axis=0)
    
    # H_crop and W_crop are essentially H and W of the chunk if no trimming is done
    return patches_array, coords, H, W, PATCH_SIZE
    

# --- PyTorch Dataset Classes (Not currently used in the main pipeline but kept for context) ---

class OnDiskPatchDataset(Dataset):
    """
    A PyTorch Dataset that reads the full 12-band patch on-demand from disk.
    This is very slow for large-scale chunk processing but useful for DataLoader examples.
    """
    # NOTE: This class definition relies on _find_band_path and _read_all_bands_for_patch
    # which are now defined above.
    pass # Placeholder for full class definition
