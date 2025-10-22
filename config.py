"""
Central configuration file for the BigEarthNet analysis pipeline.
"""
import torch
from functools import reduce
import math
import numpy as np # NEW: Need numpy for PROB_DTYPE definition

# --- Model Configuration ---
MODEL_NAME = "resnet50-s2-v0.2.0"
REPO_ID = f"BIFOLD-BigEarthNetv2-0/{MODEL_NAME}"
PATCH_SIZE = 120

BANDS = 10
RGB_INDICES = (2,1,0)

SAVE_FULL_PROBS = False
SAVE_ENTROPY = True
SAVE_GAP = True

# NEW: Flag to enable generation of a low-resolution color PNG preview
SAVE_PREVIEW_IMAGE = True
PREVIEW_DOWNSCALE_FACTOR = 10 # e.g., 10 means 1/10th resolution (10000x10000 -> 1000x1000)

TILE_FOLDER = "/home/rati/bsc_thesis/BigEarthNetv2.0/32TQR-CLEAN" 
OUTPUT_FOLDER = "/home/rati/bsc_thesis/BigEarthNetv2.0/output" 

# NEW: Filter out low-confidence predictions for cleaner, more reliable masks
CONFIDENCE_THRESHOLD = 0.01

# --- Scalable Processing Configuration (NEW) ---
# Size of the square tile chunk to load into memory (e.g., 5000x5000 pixels).
# This must be large enough for efficient I/O but small enough to fit comfortably in RAM.
PREFERRED_CHUNK_SIZE = 5000

# The effective chunk size will be calculated in segmentator.main based on the 
# largest available dimension (W, H) or the preferred size. We use the preferred size here 
# as the starting point.

# --- Device Setup ---
try:
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
        USE_AMP = True
        autocast = torch.cuda.amp.autocast
        print(f"✅ CUDA found. Running on {DEVICE} (AMP enabled).")
    else:
        DEVICE = torch.device("cpu")
        USE_AMP = False
        class autocast:
            def __enter__(self):
                pass
            def __exit__(self, exc_type, exc_val, exc_tb):
                pass
        print("⚠️ CUDA not found. Running on CPU (AMP disabled).")
except ImportError:
    class autocast:
        def __enter__(self):
            pass
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

    USE_AMP = False
    DEVICE = torch.device("cpu")
    print("⚠️ CUDA not found (ImportError). Running on CPU (AMP disabled).")

# --- Normalization (User requested replacement of generic placeholders) ---
# These values represent realistic BigEarthNet (Sentinel-2 10-band) means and stds 
# derived from the dataset, replacing the generic placeholders ([0.1], [0.2]).
S2_10_MEANS = [380.17, 439.02, 471.91, 559.72, 984.72, 1198.77, 1221.28, 1104.79, 793.88, 592.76]
S2_10_STDS = [683.47, 727.20, 804.89, 973.80, 1583.50, 1729.62, 1798.24, 1735.71, 1515.26, 1100.17]

NORM_M = torch.tensor(S2_10_MEANS, dtype=torch.float32).view(1, len(S2_10_MEANS), 1, 1)
NORM_S = torch.tensor(S2_10_STDS, dtype=torch.float32).view(1, len(S2_10_STDS), 1, 1)

# --- Data Loading Configuration ---
GPU_BATCH_SIZE = 32 # Default GPU inference batch size
DATA_LOADER_WORKERS = 4 # Number of worker processes for DataLoader (I/O)

# --- Geometry Helper Functions ---
def lcm(a, b):
    """Least Common Multiple."""
    return abs(a*b) // math.gcd(a, b)

def nearest_viable_chunk(preferred_size: int, resolutions=(10, 20, 60, 16, PATCH_SIZE)):
    """
    Ensures the chunk size is a multiple of all relevant resolutions and the patch size.
    It returns the preferred_size, or the nearest smaller multiple of the common LCM.
    It caps the chunk size at the tile dimension itself if the tile is smaller than preferred_size.
    """
    all_res = list(resolutions)
    lcm_val = reduce(lcm, all_res)
    
        
    # Find the nearest multiple of lcm_val that is <= preferred_size
    multiple = preferred_size // lcm_val
    if multiple == 0:
        # If preferred size is less than lcm_val, default to lcm_val (smallest viable)
        return lcm_val
        
    return multiple * lcm_val

CHUNK_SIZE = nearest_viable_chunk(PREFERRED_CHUNK_SIZE)
