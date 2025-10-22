"""
Main entry point for the scalable BigEarthNet analysis pipeline.

This script orchestrates the large-scale segmentation process defined in 
segmentator.py for a single, user-specified Sentinel-2 tile folder.
"""
import time
from pathlib import Path

import config
# Import the scalable main function from the refactored segmentator
from segmentator import main as segmentator_main 

def main():
    """
    Orchestrates the scalable analysis of a single input scene.
    """
    print(f"\n--- Starting Scalable Geo Segmentation ({config.MODEL_NAME}) ---")

    start_time = time.time()
    
    # =========================================================================
    # USER INPUT: Define the single directory containing the Sentinel-2 tile bands
    # NOTE: This path MUST point to the folder containing B02.jp2, B03.jp2, etc.
    # Replace the example path with your actual data directory.
    # =========================================================================
    
    tile_path = Path(config.TILE_FOLDER)
    output_path = Path(config.OUTPUT_FOLDER)

    if not tile_path.is_dir():
        print(f"❌ Error: Input directory '{TILE_FOLDER}' not found or is not a directory.")
        print("Please ensure the TILE_FOLDER variable points to the correct scene directory.")
        return

    # Direct call to the scalable segmentation pipeline for the single tile
    print(f"\n=======================================================")
    print(f"  STARTING TILE: {tile_path.name}")
    print(f"=======================================================")
    
    # The segmentator_main function handles all I/O, chunking, inference, and saving.
    segmentator_main(tile_folder=str(tile_path), output_directory=str(output_path))

    total_time = time.time() - start_time

    print("\n--- Pipeline Complete ---")
    print(f"Total Wall Time: {total_time:.2f} seconds")

if __name__ == "__main__":
    # This is the standard entry point when the script is run directly
    main()