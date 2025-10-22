Scalable Sentinel-2 Segmentation Pipeline

This documentation details the architecture, methodology, and implementation of a high-throughput, scalable deep learning pipeline designed for semantic segmentation of large-scale Sentinel-2 (BigEarthNet) imagery. The system is engineered to handle gigapixel-scale images by managing memory and maximizing GPU utilization through a novel chunking and weighted mosaicking approach.

I. Architectural Overview

The core challenge in processing large remote sensing tiles ($10,980 \times 10,980$ pixels) is the memory constraint on both system RAM and GPU VRAM. This pipeline addresses this by adopting a Producer-Consumer model based on three key stages:

Chunking: Splitting the large image into manageable, sequential chunks that fit in RAM.

Overlapping Patching & Inference (Producer): Subdividing the current chunk into highly-overlapping patches for GPU inference. This is the GPU-bound stage.

Weighted Mosaicking & Writing (Consumer): Blending the overlapping patch predictions and writing the final results to disk in real-time. This is the CPU/I/O-bound stage, running in parallel with the producer.

II. Core Methodology

1. Chunking and Memory Management

To avoid loading the entire $10,980 \times 10,980$ image (which could exceed 5GB for 10 bands of $\text{uint16}$ data) into system memory, the image is processed in large, square Chunks.

PREFERRED_CHUNK_SIZE ($\text{e.g., } 5000$): Defined in config.py, this sets the approximate pixel dimension of the chunk.

Adaptive Sizing: The system calculates the final $\text{CHUNK\_SIZE}$ to ensure it is a multiple of all Sentinel-2 band resolutions (e.g., 10m, 20m, 60m) and the $\text{PATCH\_SIZE}$. This prevents resampling artifacts at chunk boundaries.

Sequential $\text{I/O}$: Only one chunk's worth of data is loaded from the multi-band $\text{GeoTIFF}$ files into $\text{NumPy}$ arrays at a time using $\text{Rasterio}$'s Window reading capabilities.

2. Overlapping Patch Generation

The neural network is trained on small patches (e.g., $120 \times 120$ pixels). Simple non-overlapping patching would create severe checkerboard artifacts due to prediction inconsistencies at patch boundaries.

Overlap Stride: Patches are extracted from the chunk using a stride of $\text{PATCH\_SIZE} // 2$ (half the patch size).

$50\%$ Overlap: This stride ensures that every pixel in the interior of the chunk is covered by four different patches. Pixels near the chunk edges are covered by two or one patch.

Purpose: This extensive overlap is the prerequisite for the Weighted Mosaicking process, allowing for smooth, artifact-free transitions between predicted patches.

3. Probability Calculation and Fusion (Weighted Mosaicking)

This is the most critical step for generating visually and numerically high-quality segmentation maps.

The Problem: Simply averaging the probabilities from overlapping patches still yields a blurred result with visible seams.

The Solution: Use a spatial weighting function to emphasize the prediction reliability near the center of the patch and de-emphasize the less reliable predictions at the patch edges.

Sinusoidal Weighting Mask: The system generates a 2D sinusoidal window function (np.sin(x)) that is:

Maximum (1.0) at the center of the patch.

Minimum (0.0) at the edges of the patch.

Accumulation:

$$P_{\text{blended}}(x, y) = \frac{\sum_{i \in \text{patches}(x,y)} P_{i}(x, y) \cdot W_{i}(x, y)}{\sum_{i \in \text{patches}(x,y)} W_{i}(x, y)}$$

Where:

$P_i(x, y)$ is the probability from patch $i$ at pixel location $(x, y)$.

$W_i(x, y)$ is the weight (sinusoidal mask value) from patch $i$ at pixel $(x, y)$.

This weighted averaging technique effectively smooths the transitions, making the final classification map appear continuous and homogeneous.

4. Uncertainty Quantification

Beyond the final classification mask, the pipeline generates several products for quantifying prediction confidence:

Maximum Probability ($\text{MaxProb}$): The probability of the final dominant class.

Entropy: A measure of the statistical uncertainty of the class prediction for a given pixel, calculated as:

$$E = - \sum_{c=1}^{N} P_c \log_2(P_c)$$

High entropy indicates the model is highly uncertain (probabilities are spread out).

Gap: The difference between the highest and second-highest probability ($\text{max} - \text{second\_max}$). A smaller gap suggests the pixel is close to a decision boundary.

III. Component Deep Dive

config.py

Parameter

Type

Details

PATCH_SIZE

$\text{int}$

Standardized input size for the deep learning model ($\text{e.g., } 120$).

BANDS

$\text{int}$

The number of Sentinel-2 bands to use ($\text{10}$ or $\text{12}$).

PREFERRED_CHUNK_SIZE

$\text{int}$

Determines the chunk dimensions before adaptive adjustment.

NORM_M, NORM_S

$\text{torch.Tensor}$

Pre-calculated $\text{mean}$ and $\text{standard deviation}$ for the $\text{10}$ bands, used for $\text{Z-score}$ normalization on the $\text{GPU}$.

GPU_BATCH_SIZE

$\text{int}$

Number of patches processed in one $\text{GPU}$ forward pass.

utils.py

Provides necessary setup, data normalization, and the core $\text{GPU}$ execution logic.

Dependency Fallback: Contains robust $\text{try/except}$ logic to define placeholder classes and constants (like STANDARD_BANDS, NEW_LABELS, and a dummy BigEarthNetv2_0_ImageClassifier) if external libraries are not available, ensuring the script is runnable even in limited environments.

run_gpu_inference: This is the function that is threaded into the producer-consumer model. It converts $\text{NumPy}$ data to $\text{PyTorch}$ $\text{tensors}$, applies normalization using $\text{config}$ values, runs the $\text{model.forward}$ pass, and applies torch.sigmoid to get probabilities.

save_color_mask_preview: Utility to convert the final $\text{GeoTIFF}$ class indices into a visually appealing $\text{RGB}$ $\text{PNG}$ preview using LABEL_COLOR_MAP.

data_loader.py

Focuses on robust $\text{I/O}$ for large $\text{GeoTIFF}$ files.

_read_all_bands_for_chunk: Handles the complexity of reading a single rectangular window (Window) that spans multiple band files with different spatial resolutions (10m, 20m, 60m).

It reads the highest-resolution band (B02) first to establish the target dimensions.

Lower-resolution bands are read and then resampled using bilinear interpolation (Resampling.bilinear) to match the $\text{10m}$ resolution, ensuring all bands align perfectly in the final chunk array.

segmentator.py (Orchestrator)

The main logic file. It defines the $\text{Producer-Consumer}$ threads and coordinates the entire workflow.

main(tile_folder, output_directory):

Initializes all output $\text{GeoTIFF}$ files using $\text{Rasterio}$ with appropriate profiles (tiled=True, compress='lzw').

Sets up the result_queue to pass $\text{GPU}$ results from the producer to the consumer.

Starts the mosaicking_worker thread.

Producer Loop (in main):

Iterates through all calculated $\text{Chunk}$ coordinates.

Calls $\text{I/O}$ ($\text{load\_bands\_in\_window}$) $\rightarrow$ $\text{Patching}$ ($\text{cut\_into\_patches}$) $\rightarrow$ $\text{Inference}$ ($\text{run\_gpu\_inference}$).

Pushes the results (probabilities, coordinates) to the result_queue.

Consumer Thread (mosaicking_worker):

Runs indefinitely, waiting for results in the queue.

Pulls results and calls the CPU-intensive accumulate_probs function (mosaicking).

Calculates $\text{MaxProb}$, $\text{Entropy}$, and $\text{Gap}$ from the blended probabilities.

Writes the final $\text{NumPy}$ chunk data to the opened $\text{GeoTIFF}$ files using $\text{Rasterio}$'s write method with the correct window coordinates.

main.py

This serves as the simple execution wrapper. It defines the input and output directories and calls segmentator.main, providing a clean entry point for the pipeline.

IV. Output Products

The pipeline produces multiple $\text{GeoTIFF}$ files for downstream GIS analysis, all sharing the original $\text{GeoTIFF}$ coordinate reference system (CRS) and geospatial transform.

Output File ($\text{GeoTIFF}$)

Data Type

Description

*_class.tif

$\text{uint8}$

The final, highest-confidence land cover classification mask (index of the class label).

*_maxprob.tif

$\text{float32}$

The confidence level (probability) of the dominant class assigned in *_class.tif.

*_entropy.tif

$\text{float32}$

The uncertainty measure, useful for highlighting areas where the model was confused.

*_gap.tif

$\text{float32}$

The margin of confidence between the top two classes.

*_probs.tif

$\text{float32}$

A multi-band $\text{GeoTIFF}$ containing the full probability vector for every class at every pixel (optional).

preview.png

$\text{PNG}$

A low-resolution color visualization of the classification map.