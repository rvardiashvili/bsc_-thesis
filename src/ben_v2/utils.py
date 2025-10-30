"""
Utility functions, placeholder logic for external libraries, and visualization tools.
"""
import base64
import io
import time # Added time for run_gpu_inference
from typing import Dict, List, Any, Tuple
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from PIL import Image # Required for saving PNG preview
from scipy.ndimage import zoom # Required for downscaling the mask
from torch.utils.data import DataLoader

from .config import (
    DEVICE, PATCH_SIZE, BANDS, RGB_INDICES, SAVE_PREVIEW_IMAGE, 
    PREVIEW_DOWNSCALE_FACTOR, GPU_BATCH_SIZE, USE_AMP, autocast, 
    NORM_M, NORM_S
)

# ----------------------------------------------------------------------
# --- Placeholder/Required Module Variables ---
# These are initialized to ensure module-level existence for imports.
# ----------------------------------------------------------------------

STANDARD_BANDS: Dict[int, List[str]] = {}
NEW_LABELS: List[str] = []
BigEarthNetv2_0_ImageClassifier = None
FALLBACK_LABEL_KEY = 'No_Dominant_Class' 

# ----------------------------------------------------------------------
# --- Placeholder Logic for Missing Libraries ---
# ----------------------------------------------------------------------

try:
    # Attempt to import the real libraries
    from configilm.extra.BENv2_utils import STANDARD_BANDS, NEW_LABELS, stack_and_interpolate, means, stds
    from .model import BigEarthNetv2_0_ImageClassifier
    print("✅ Successfully loaded BENv2 libraries.")
    
except ImportError as e:
    print(f"⚠️ Warning: External BENv2 libraries not found or failed to load ({e}). Using robust placeholders.")
    
    # Define fallback data structures if import fails

    # STANDARD_BANDS 
    STANDARD_BANDS = {
        10: ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12'],
        12: ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12'],
    }

    # Fallback Labels (43 classes from BigEarthNet v1)
    NEW_LABELS = [
        'Annual-crops', 'Permanent-crops', 'Pastures', 'Complex-cultivation-patterns', 
        'Agro-forestry-areas', 'Broad-leaved-forest', 'Coniferous-forest', 'Mixed-forest', 
        'Natural-grasslands-and-sparsely-vegetated-areas', 'Moors-and-heathland', 
        'Sclerophyllous-vegetation', 'Transitional-woodland-shrub', 'Beaches-sandy-plains', 
        'Intertidal-flats', 'Bare-areas', 'Burnt-areas', 'Inland-wetlands', 
        'Coastal-wetlands', 'Continental-water', 'Marine-water', 'Glaciers-and-perpetual-snow', 
        'Non-irrigated-arable-land', 'Permanently-irrigated-land', 'Rice-fields', 
        'Vineyards', 'Fruit-trees-and-berry-plantations', 'Olive-groves', 
        'Annual-crops-with-associated-fallow-lands', 
        'Land-principally-occupied-by-agriculture-with-significant-areas-of-natural-vegetation', 
        'Broad-leaved-forest-evergreen', 'Broad-leaved-forest-deciduous', 
        'Coniferous-forest-evergreen', 'Coniferous-forest-deciduous', 'Mixed-forest', 
        'Natural-grasslands', 'Sparsely-vegetated-areas', 'Salt-marshes', 
        'Bogs-and-peatlands', 'Water-bodies', 'Snow-and-ice', 'Urban-fabric', 
        'Industrial-or-commercial-units', 'Road-and-rail-networks-and-associated-land'
    ]
    
    # Placeholder Model Class
    class BigEarthNetv2_0_ImageClassifier(torch.nn.Module):
        """Placeholder model class for running without the external library."""
        def __init__(self, num_classes: int):
            super().__init__()
            self.num_classes = num_classes
            print("⚠️ Using a dummy torch.nn.Linear model as BigEarthNetv2_0_ImageClassifier placeholder.")
            # Use 12 bands for the dummy model size, assuming BANDS will be correctly set later
            input_size = len(STANDARD_BANDS.get(12, [])) * PATCH_SIZE * PATCH_SIZE
            self.linear = torch.nn.Linear(input_size, num_classes) 

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Flatten C, H, W to a single vector for the linear layer
            return self.linear(x.flatten(start_dim=1))
            
    # Placeholder utility function (if needed by other files)
    def stack_and_interpolate(bands_dict: Dict[str, np.ndarray]) -> np.ndarray:
        # Dummy implementation: just stacks the bands provided by the BANDS constant
        stacked = np.stack([bands_dict[b] for b in STANDARD_BANDS[BANDS]], axis=-1)
        return stacked

# ----------------------------------------------------------------------
# Consistent Label Color Map Generation
# This section runs regardless of the try/except block to ensure LABEL_COLOR_MAP exists.
# ----------------------------------------------------------------------

# Fallback Label Color Map (for visualization)
# Mapping the labels to simple random colors for visualization
LABEL_COLOR_MAP: Dict[str, np.ndarray] = {
    label: np.random.randint(0, 256, 3, dtype=np.uint8) 
    for label in NEW_LABELS
}

# Add the Fallback key
LABEL_COLOR_MAP[FALLBACK_LABEL_KEY] = np.array([128,128,128], dtype=np.uint8)


# ----------------------------------------------------------------------
# Visualization Functions (Defined here so they can run even with placeholders)
# ----------------------------------------------------------------------

def generate_low_res_preview(mask_data: np.ndarray, output_path: Path, downscale_factor: int):
    """
    Generates a low-resolution color PNG preview of the classification mask.
    This runs in the main process after the GeoTIFFs are written.
    
    Args:
        mask_data (np.ndarray): The 2D classification mask (integer indices).
        output_path (Path): Path where the PNG image should be saved.
        downscale_factor (int): Factor to reduce resolution (e.g., 10 means 1/10th resolution).
    """
    if not SAVE_PREVIEW_IMAGE:
        return

    print(f"🎨 Generating low-res preview image (downscale={downscale_factor})...")

    # 1. Downscale the index mask using nearest-neighbor interpolation
    # Order=0 ensures we use nearest-neighbor to keep class indices discrete
    if downscale_factor > 1:
        downscaled_mask = zoom(mask_data, 1.0 / downscale_factor, order=0)
    else:
        downscaled_mask = mask_data
    
    # 2. Map indices to colors
    
    # We use the module-level LABEL_COLOR_MAP and NEW_LABELS
    color_map_array = np.array([LABEL_COLOR_MAP[label] for label in NEW_LABELS], dtype=np.uint8)
    
    # Use the mask indices to look up colors
    # Ensure indices are within bounds (0 to N-1 classes)
    max_idx = len(NEW_LABELS) - 1
    safe_mask = np.clip(downscaled_mask, 0, max_idx)
    
    rgb_image = color_map_array[safe_mask]
    
    # 3. Save as PNG
    try:
        # PIL expects H x W x C
        img_pil = Image.fromarray(rgb_image, 'RGB')
        img_pil.save(output_path, 'PNG')
        print(f"  Saved preview to {output_path.name}")
    except Exception as e:
        print(f"❌ Error saving preview image: {e}")

# If the placeholder class was used, we need to alias the function name for segmentator.py
try:
    save_color_mask_preview
except NameError:
    # If the import failed, save_color_mask_preview was not defined, so we create the alias
    save_color_mask_preview = generate_low_res_preview

# ----------------------------------------------------------------------
# Helper for GPU Inference (Moved from pipeline.py to utils.py for portability)
# ----------------------------------------------------------------------

def run_gpu_inference(patches: np.ndarray, model: torch.nn.Module) -> np.ndarray:
    """
    Performs batch inference on the GPU for a NumPy array of patches.

    Args:
        patches (np.ndarray): NumPy array of C, H, W patches.
        model (torch.nn.Module): The loaded classification model.

    Returns:
        np.ndarray: Concatenated array of sigmoid probabilities (N_patches, N_classes).
    """
    start_time = time.time()
    all_probs = []
    
    # Create an in-memory DataLoader for the patches
    class InMemoryPatchDataset(torch.utils.data.Dataset):
        def __init__(self, patches: np.ndarray):
            # patches should be in (N, C, H, W) format
            self.patches = patches

        def __len__(self):
            return len(self.patches)

        def __getitem__(self, idx):
            # Return the tensor patch
            return torch.as_tensor(self.patches[idx]).float()

    dataset = InMemoryPatchDataset(patches)
    # Using 0 workers for in-memory data to prevent copy overhead
    dataloader = DataLoader(dataset, batch_size=GPU_BATCH_SIZE, shuffle=False, num_workers=0)

    for tensor_cpu in dataloader:
        try:
            # Normalize and move to GPU
            tensor_gpu = tensor_cpu.to(DEVICE, non_blocking=True)
            tensor_gpu = (tensor_gpu - NORM_M.to(DEVICE)) / (NORM_S.to(DEVICE) + 1e-6)
            
            # Inference
            if USE_AMP and DEVICE.type == 'cuda':
                with autocast(device_type='cuda', dtype=torch.float16):
                    logits = model(tensor_gpu)
            else:
                logits = model(tensor_gpu)
                
            # Calculate probabilities and move back to CPU
            probs = torch.sigmoid(logits.float()).cpu().detach().numpy()
            all_probs.append(probs)
        except Exception as e:
            print(f"❌ GPU inference error on batch: {e}")
            # Append zeros for the size of the batch that failed to maintain shape integrity
            probs = np.zeros((tensor_cpu.shape[0], len(NEW_LABELS)), dtype=np.float32)
            all_probs.append(probs)

    end_time = time.time()
    return np.concatenate(all_probs, axis=0)
