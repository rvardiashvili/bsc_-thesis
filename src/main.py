"""
Main entry point for the BigEarthNet v2.0 analysis pipeline.
"""

import argparse
from ben_v2.main import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BigEarthNet v2.0 Analysis Pipeline")
    parser.add_argument("--tile_folder", type=str, required=True, help="Path to the Sentinel-2 tile folder.")
    parser.add_argument("--output_directory", type=str, required=True, help="Path to the output directory.")
    args = parser.parse_args()

    main(args.tile_folder, args.output_directory)