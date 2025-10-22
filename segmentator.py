import numpy as np, torch, rasterio, time, gc, json, threading, queue, sys
from rasterio.enums import Resampling
from rasterio.windows import Window
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import Tuple, List
from config import (
    PATCH_SIZE, REPO_ID, DEVICE, GPU_BATCH_SIZE, USE_AMP, autocast, 
    CONFIDENCE_THRESHOLD, CHUNK_SIZE, BANDS, SAVE_FULL_PROBS, SAVE_PREVIEW_IMAGE, PREVIEW_DOWNSCALE_FACTOR
)
from utils import (
    NEW_LABELS, LABEL_COLOR_MAP, BigEarthNetv2_0_ImageClassifier,
    NORM_M, NORM_S, STANDARD_BANDS, save_color_mask_preview
)

# ------------------------------------------------------------
# Setup
# ------------------------------------------------------------
BANDS = STANDARD_BANDS[BANDS]
FALLBACK_LABEL = "No_Dominant_Class"
LABEL_COLOR_MAP[FALLBACK_LABEL] = np.array([128,128,128],dtype=np.uint8)

SAVE_CONFIDENCE = True
SAVE_ENTROPY = True
SAVE_GAP = True # Assuming this was intended to be defined here
PROB_DTYPE = np.float32
PROB_COMPRESS = "lzw"

# ------------------------------------------------------------
def _find_band_path(tile_folder: Path, band_name: str) -> Path | None:
    for ext in ['.jp2','.tif']:
        candidate = next(tile_folder.glob(f"*{band_name}*{ext}"), None)
        if candidate:
            return candidate
    return None

# ------------------------------------------------------------
def load_bands_in_window(tile_folder: Path, window: Window, out_shape: Tuple[int, int],
                         full_tile_shape: Tuple[int, int], r_start: int, c_start: int,
                         bands_to_load: List[str] = BANDS) -> np.ndarray:
    band_paths = [_find_band_path(tile_folder, band) for band in bands_to_load]
    bands_data = []
    H_full, W_full = full_tile_shape
    ref_band_path = _find_band_path(tile_folder, 'B02')
    if not ref_band_path:
        return np.array([])

    for i, (band_name, band_path) in enumerate(zip(bands_to_load, band_paths)):
        if band_path is None:
            bands_data.append(np.zeros(out_shape, dtype=np.float32))
            continue
        try:
            with rasterio.open(band_path) as src:
                nodata_value = src.nodata
                src_W = src.width
                ratio = src_W / W_full if W_full > 0 else 1.0
                band_window = (Window(int(round(window.col_off * ratio)),
                                      int(round(window.row_off * ratio)),
                                      int(round(window.width * ratio)),
                                      int(round(window.height * ratio)))
                               if abs(ratio - 1.0) > 1e-4 else window)
                data = src.read(1, window=band_window, out_shape=out_shape, resampling=Resampling.nearest)
                if data is None:
                    bands_data.append(np.zeros(out_shape, dtype=np.float32))
                    continue
                data_2d = np.asarray(data, dtype=np.float32)
                if nodata_value is not None:
                    data_2d[data_2d == nodata_value] = 0.0
                bands_data.append(data_2d)
        except Exception as e:
            print(f"Warning: failed reading {band_name} in {tile_folder}: {e}", file=sys.stderr)
            bands_data.append(np.zeros(out_shape, dtype=np.float32))
    if not bands_data:
        return np.array([])
    return np.stack(bands_data, axis=-1)

# ------------------------------------------------------------
def cut_into_patches(img: np.ndarray, patch_size:int):
    H_full, W_full, C = img.shape
    stride = patch_size // 2
    r_coords = list(range(0, H_full - stride + 1, stride))
    c_coords = list(range(0, W_full - stride + 1, stride))
    if r_coords and (r_coords[-1] + patch_size < H_full):
        r_coords.append(H_full - patch_size)
    if c_coords and (c_coords[-1] + patch_size < W_full):
        c_coords.append(W_full - patch_size)
    r_coords = sorted(set(r for r in r_coords if 0 <= r and r + patch_size <= H_full))
    c_coords = sorted(set(c for c in c_coords if 0 <= c and c + patch_size <= W_full))
    patches_list, coords = [], []
    for r0 in r_coords:
        for c0 in c_coords:
            patch = img[r0:r0+patch_size, c0:c0+patch_size, :]
            patches_list.append(patch)
            coords.append((r0, c0))
    if not patches_list:
        return torch.empty(0), [], H_full, W_full, img
    patches_tensor = torch.as_tensor(np.stack(patches_list)).permute(0,3,1,2).float()
    return patches_tensor, coords, H_full, W_full, img

# ------------------------------------------------------------
def run_gpu_inference(patches_tensor: torch.Tensor, model: torch.nn.Module) -> np.ndarray:
    if patches_tensor.numel() == 0:
        return np.zeros((0, len(NEW_LABELS)), dtype=np.float32)
    dataset = torch.utils.data.TensorDataset(patches_tensor)
    loader = DataLoader(dataset, batch_size=GPU_BATCH_SIZE, shuffle=False, num_workers=0)
    all_probs = []
    norm_m_device = NORM_M.to(DEVICE)
    norm_s_device = NORM_S.to(DEVICE)
    for batch in loader:
        tensor_cpu = batch[0]
        tensor_gpu = (tensor_cpu.to(DEVICE) - norm_m_device) / norm_s_device
        with torch.no_grad(), torch.inference_mode():
            try:
                if USE_AMP:
                    with autocast(dtype=torch.float16):
                        logits = model(tensor_gpu)
                else:
                    logits = model(tensor_gpu)
                probs = torch.sigmoid(logits.float()).cpu().numpy()
            except Exception as e:
                print(f"GPU inference error: {e}", file=sys.stderr)
                probs = np.zeros((tensor_cpu.shape[0], len(NEW_LABELS)), dtype=np.float32)
        all_probs.append(probs)
    if not all_probs:
        return np.zeros((0, len(NEW_LABELS)), dtype=np.float32)
    return np.concatenate(all_probs, axis=0)

# ------------------------------------------------------------
def accumulate_probs(results: np.ndarray, coords: List[Tuple[int, int]], H: int, W: int, patch_size: int, n_classes: int) -> np.ndarray:
    avg = np.zeros((H, W, n_classes), dtype=np.float32)
    count = np.zeros((H, W, 1), dtype=np.float32)
    window_2d = np.outer(np.sin(np.linspace(0, np.pi, patch_size))**2,
                         np.sin(np.linspace(0, np.pi, patch_size))**2).astype(np.float32)
    weight_mask = np.tile(window_2d[:,:,np.newaxis], (1, 1, n_classes))
    weight_mask_count = np.tile(window_2d[:,:,np.newaxis], (1, 1, 1))
    for i, (r0, c0) in enumerate(coords):
        prob = results[i]
        weighted_prob = prob[np.newaxis, np.newaxis, :] * weight_mask
        avg[r0:r0+patch_size, c0:c0+patch_size, :] += weighted_prob
        count[r0:r0+patch_size, c0:c0+patch_size, :] += weight_mask_count
    count[count == 0] = 1.0
    avg /= count
    return avg

# ------------------------------------------------------------
def main(tile_folder: str, crop_limit=None, output_directory: str | None = None):
    t0 = time.time()
    tile = Path(tile_folder)
    # Ensure output_directory is used correctly
    out_base = Path(output_directory or tile.parent)
    out_path = out_base / tile.name
    out_path.mkdir(parents=True, exist_ok=True)

    target_band_path = _find_band_path(tile, 'B02')
    if not target_band_path:
        print("❌ Error: Reference band B02 not found.")
        return

    with rasterio.open(target_band_path) as src:
        H_full_original, W_full_original = src.height, src.width
        # 1. Get the base profile, including geospatial info
        base_profile = src.profile
        full_tile_shape = (H_full_original, W_full_original)

    if crop_limit:
        H_full = min(H_full_original, crop_limit[0])
        W_full = min(W_full_original, crop_limit[1])
    else:
        H_full, W_full = H_full_original, W_full_original
    
    # --- Profile Generation (Fixes RasterBlockError and StripOffsets Error) ---
    
    # Keys that often cause conflict when changing from stripped/tiled or changing driver
    # We remove parameters that define the internal TIFF structure which we are redefining below.
    keys_to_remove = ['tiled', 'blockxsize', 'blockysize', 'interleave', 'compress', 'count', 'dtype', 'nodata', 'driver']
    
    # Start with a clean profile derived from the source
    profile_base = base_profile.copy()
    for key in keys_to_remove:
        profile_base.pop(key, None)
    
    # 2. Define the Tiled GeoTIFF profile for uint8 class masks
    profile_mask = profile_base.copy()
    profile_mask.update(
        driver="GTiff",
        height=H_full,  # Ensure final height is set
        width=W_full,   # Ensure final width is set
        dtype=rasterio.uint8,
        count=1,
        nodata=255, 
        compress='lzw',
        tiled=True, 
        blockxsize=CHUNK_SIZE, # CHUNK_SIZE must be a multiple of 16!
        blockysize=CHUNK_SIZE
    )
    
    # 3. Define the Tiled GeoTIFF profile for float (max_prob, entropy, gap)
    profile_float = profile_base.copy()
    profile_float.update(
        driver="GTiff",
        height=H_full,  # Ensure final height is set
        width=W_full,   # Ensure final width is set
        dtype=rasterio.float32,
        count=1, # For single-band outputs
        compress='lzw',
        tiled=True,
        blockxsize=CHUNK_SIZE,
        blockysize=CHUNK_SIZE
    )

    out_class_path = out_path / f"{tile.name}_class.tif"
    out_conf_path  = out_path / f"{tile.name}_maxprob.tif"
    out_prob_path  = out_path / f"{tile.name}_probs.tif"
    out_entropy_path = out_path / f"{tile.name}_entropy.tif"
    out_gap_path = out_path / f"{tile.name}_gap.tif"

    with open(out_path / f"{tile.name}_classmap.json", "w") as f:
        json.dump({i: lbl for i, lbl in enumerate(NEW_LABELS)}, f)

    model = BigEarthNetv2_0_ImageClassifier.from_pretrained(REPO_ID).to(DEVICE).eval()

    result_queue = queue.Queue(maxsize=2)
    stop_signal = object()
    total_chunks = ((H_full + CHUNK_SIZE - 1) // CHUNK_SIZE) * ((W_full + CHUNK_SIZE - 1) // CHUNK_SIZE)
    mosaic_pbar = tqdm(total=total_chunks, desc="Writing", position=1)
    fetch_pbar  = tqdm(total=total_chunks, desc="Inference", position=0)

    def mosaicking_worker(dst_class, dst_conf=None, dst_probs=None, dst_entropy=None, dst_gap=None):
        try:
            while True:
                item = result_queue.get()
                if item is stop_signal:
                    result_queue.task_done()
                    break
                (results, coords, H_crop, W_crop, patch_size, n_classes,
                 c_start, r_start, W_chunk, H_chunk) = item
                try:
                    # Accumulate probabilities
                    avg = accumulate_probs(results, coords, H_crop, W_crop, patch_size, n_classes)
                    
                    # Calculate outputs
                    dominant_idx = np.argmax(avg, axis=2).astype(np.uint8)
                    max_prob = np.max(avg, axis=2).astype(np.float32)
                    entropy = -np.sum(avg * np.log(np.clip(avg, 1e-6, 1.0)), axis=2).astype(np.float32)
                    
                    # Calculate Gap (Difference between max and second-max prob)
                    top2 = np.partition(avg, -2, axis=2)[:, :, -2:]
                    gap = (top2[:, :, 1] - top2[:, :, 0]).astype(np.float32)

                    # Write to GeoTIFFs
                    window_cropped = Window(col_off=c_start, row_off=r_start, width=W_chunk, height=H_chunk)
                    
                    dst_class.write(dominant_idx[np.newaxis,:,:], window=window_cropped)
                    if dst_conf is not None:
                        dst_conf.write(max_prob[np.newaxis,:,:], window=window_cropped)
                    if dst_entropy is not None:
                        dst_entropy.write(entropy[np.newaxis,:,:], window=window_cropped)
                    if dst_gap is not None:
                        dst_gap.write(gap[np.newaxis,:,:], window=window_cropped)
                    if dst_probs is not None:
                        # Write all probability bands (n_classes bands)
                        dst_probs.write(avg.transpose(2,0,1), window=window_cropped)
                        
                except Exception as e:
                    print(f"Error in worker: {e}", file=sys.stderr)
                finally:
                    del results, avg
                    gc.collect()
                    mosaic_pbar.update(1)
                    result_queue.task_done()
        except Exception as e:
            print(f"Fatal error in mosaicking_worker: {e}", file=sys.stderr)

    # Open all destination files before starting the threads
    with rasterio.open(out_class_path, "w", **profile_mask) as dst_class:
        
        # Open single-band float files
        dst_conf = rasterio.open(out_conf_path, "w", **profile_float)
        dst_entropy = rasterio.open(out_entropy_path, "w", **profile_float) if SAVE_ENTROPY else None
        dst_gap = rasterio.open(out_gap_path, "w", **profile_float) if SAVE_GAP else None

        # Open multi-band float file
        if SAVE_FULL_PROBS:
            profile_probs = profile_float.copy()
            profile_probs.update(count=len(NEW_LABELS))
            dst_probs = rasterio.open(out_prob_path, "w", **profile_probs)
        else:
            dst_probs = None

        worker_thread = threading.Thread(
            target=mosaicking_worker,
            args=(dst_class, dst_conf, dst_probs, dst_entropy, dst_gap),
            daemon=True
        )
        worker_thread.start()

        for r_start in range(0, H_full, CHUNK_SIZE):
            for c_start in range(0, W_full, CHUNK_SIZE):
                r_end = min(r_start + CHUNK_SIZE, H_full)
                c_end = min(c_start + CHUNK_SIZE, W_full)
                H_chunk, W_chunk = r_end - r_start, c_end - c_start
                window = Window(c_start, r_start, W_chunk, H_chunk)
                img_chunk = load_bands_in_window(tile, window, (H_chunk, W_chunk), full_tile_shape, r_start, c_start, BANDS)
                
                # Check for empty/nodata chunk
                if img_chunk.size == 0 or np.mean(img_chunk < 1.0) > 0.99:
                    dummy_idx = np.full((H_chunk, W_chunk), profile_mask.get('nodata', 255), dtype=np.uint8)
                    dst_class.write(dummy_idx[np.newaxis,:,:], window=window)
                    if dst_conf: dst_conf.write(np.full((H_chunk, W_chunk), 0.0, dtype=np.float32)[np.newaxis,:,:], window=window)
                    if dst_entropy: dst_entropy.write(np.full((H_chunk, W_chunk), 0.0, dtype=np.float32)[np.newaxis,:,:], window=window)
                    if dst_gap: dst_gap.write(np.full((H_chunk, W_chunk), 0.0, dtype=np.float32)[np.newaxis,:,:], window=window)
                    if dst_probs: dst_probs.write(np.full((len(NEW_LABELS), H_chunk, W_chunk), 0.0, dtype=np.float32), window=window)
                    fetch_pbar.update(1)
                    mosaic_pbar.update(1) # Ensure we count this skipped chunk
                    continue
                
                # Inference
                patches, coords, H_crop, W_crop, _ = cut_into_patches(img_chunk, PATCH_SIZE)
                del img_chunk; gc.collect()
                results = run_gpu_inference(patches, model=model)
                n_classes = results.shape[1] if results.ndim == 2 else len(NEW_LABELS)
                del patches; gc.collect()
                
                # Push results to worker queue
                result_queue.put((results, coords, H_crop, W_crop, PATCH_SIZE, n_classes,
                                  c_start, r_start, W_chunk, H_chunk))
                fetch_pbar.update(1)

        result_queue.put(stop_signal)
        result_queue.join()
        worker_thread.join()

        # Close all files
        dst_conf.close()
        if dst_entropy: dst_entropy.close()
        if dst_gap: dst_gap.close()
        if dst_probs: dst_probs.close()
        fetch_pbar.close(); mosaic_pbar.close()

    print(f"\n✅ Finished in {time.time()-t0:.2f}s")
    print(f"Class map: {out_class_path}")
    print(f"Max prob:  {out_conf_path}")
    if SAVE_ENTROPY: print(f"Entropy:  {out_entropy_path}")
    if SAVE_GAP: print(f"Gap:       {out_gap_path}")
    if SAVE_FULL_PROBS: print(f"Full probs: {out_prob_path}")
    if SAVE_PREVIEW_IMAGE:
        # Need to re-open the file as it was closed in the worker thread
        try:
            with rasterio.open(out_class_path) as src:
                class_mask = src.read(1)
                # Calling the new, correctly imported function
                save_color_mask_preview(
                    class_mask, 
                    out_path / "preview.png", 
                    downscale_factor=PREVIEW_DOWNSCALE_FACTOR
                )
        except Exception as e:
            print(f"❌ Error during final preview generation: {e}")

    fetch_pbar.close(); mosaic_pbar.close()