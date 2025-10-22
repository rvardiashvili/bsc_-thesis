import rasterio
import numpy as np
import sys
from collections import Counter
from pathlib import Path

def check_tif_uniformity(filepath: str):
    """
    Checks a single-band GeoTIFF file for uniformity by reading the data 
    and counting the unique pixel values.
    """
    file_path = Path(filepath)
    if not file_path.exists():
        print(f"❌ Error: File not found at '{filepath}'")
        return

    print(f"\n--- Analyzing {file_path.name} ---")
    
    try:
        with rasterio.open(file_path) as src:
            # Read the first band (index mask)
            data = src.read(1)
            
            # Flatten the array and find unique values
            flat_data = data.flatten()
            unique_values, counts = np.unique(flat_data, return_counts=True)
            
            total_pixels = flat_data.size
            print(f"Total Pixels: {total_pixels}")
            print(f"Unique Class Indices Found: {len(unique_values)}")

            print("\nPixel Count per Index:")
            
            # Use Counter for cleaner display, especially for non-contiguous indices
            value_counts = dict(zip(unique_values, counts))
            
            # Sort by count descending to see the most dominant class
            sorted_counts = sorted(value_counts.items(), key=lambda item: item[1], reverse=True)
            
            is_uniform = len(unique_values) <= 1
            
            for value, count in sorted_counts:
                percentage = (count / total_pixels) * 100
                print(f"  Index {value}: {count:,} pixels ({percentage:.4f}%)")

            if is_uniform and 0 in unique_values:
                print("\n⚠️ RESULT: The GeoTIFF is effectively UNIFORM (all pixels are class 0 or nodata).")
            elif is_uniform:
                print(f"\n⚠️ RESULT: The GeoTIFF is uniform (all pixels are class {unique_values[0]}).")
            else:
                print("\n✅ RESULT: The GeoTIFF is NON-UNIFORM and contains multiple classes.")

    except rasterio.RasterioIOError as e:
        print(f"❌ Rasterio Error: Could not open or read the file. {e}")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Example path from your output
        example_path = "/home/rati/bsc_thesis/BigEarthNetv2.0/output/32TQR-CLEAN/32TQR-CLEAN_class.tif"
        print("Usage: python check_uniformity.py <path_to_class_tif>")
        print(f"Example: python check_uniformity.py {example_path}")
    else:
        check_tif_uniformity(sys.argv[1])
