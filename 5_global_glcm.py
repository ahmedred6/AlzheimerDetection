import numpy as np
import nibabel as nib
import time
from skimage.feature import graycomatrix, graycoprops
import os

# --- CONFIGURATION ---
# Path to one of your processed NIFTI files (Choose a real one)
FILE_PATH = "AlzhiemerDisease/Processed_Data/MCI/002_S_0729_20060802.nii.gz"
ROI_CENTER = (60, 128, 128) # Approximate center of Hippocampus (Left)
ROI_SIZE = 32               # Smaller size (32x32x32) for faster testing
LEVELS = 32                 # Quantization levels (0-31)

def quantize_image(image, levels):
    # Normalize to 0-1, then scale to 0-(levels-1)
    min_val = np.min(image)
    max_val = np.max(image)
    if max_val == min_val: return np.zeros(image.shape, dtype=np.uint8)
   
    norm = (image - min_val) / (max_val - min_val)
    quantized = np.floor(norm * (levels - 0.999)).astype(np.uint8)
    return quantized

def get_roi(data, center, size):
    r = size // 2
    x, y, z = center
    # Crop a cubic ROI around the center
    roi = data[x-r:x+r, y-r:y+r, z-r:z+r]
    return roi

# --- METHOD 1: GLOBAL GLCM ---
def run_global_glcm(roi):
    start_time = time.perf_counter()
   
    # 1. Compute ONE matrix for the entire 3D volume
    # (We flatten it slightly or treat it as a sequence of 2D slices for simplicity in standard libraries)
    # Ideally, 3D GLCM looks at neighbors in 13 directions.
    # For CPU benchmarking, a simple average of 3 directions (0, 45, 90) is enough to show speed.
    glcm = graycomatrix(roi[:, :, roi.shape[2]//2], # Middle slice for speed example
                        distances=[1],
                        angles=[0, np.pi/4, np.pi/2],
                        levels=LEVELS,
                        symmetric=True, normed=True)
   
    # 2. Extract Features
    contrast = graycoprops(glcm, 'contrast')
    energy = graycoprops(glcm, 'energy')
   
    end_time = time.perf_counter()
    return (end_time - start_time) * 1000  # Convert to ms

# --- METHOD 2: SLIDING WINDOW GLCM ---
def run_sliding_glcm(roi):
    start_time = time.perf_counter()
   
    window_size = 5 # 5x5 window
    offset = window_size // 2
    depth = roi.shape[2] // 2 # Just run on middle slice for now to save time
   
    # We will slide over every pixel in the slice
    h, w = roi.shape[0], roi.shape[1]
   
    # Loop over every pixel (The "Sliding" part)
    for i in range(offset, h - offset):
        for j in range(offset, w - offset):
            # Extract small window
            window = roi[i-offset:i+offset+1, j-offset:j+offset+1, depth]
           
            # Compute GLCM for this tiny window
            glcm = graycomatrix(window,
                                distances=[1],
                                angles=[0],
                                levels=LEVELS,
                                symmetric=True, normed=True)
           
            # Extract feature (just one for the benchmark)
            _ = graycoprops(glcm, 'contrast')

    end_time = time.perf_counter()
    return (end_time - start_time) * 1000 # Convert to ms

# --- MAIN EXECUTION ---
if os.path.exists(FILE_PATH):
    # 1. Load Data
    img = nib.load(FILE_PATH)
    data = img.get_fdata()
   
    # 2. Extract ROI & Quantize
    roi_raw = get_roi(data, ROI_CENTER, ROI_SIZE)
    roi_quant = quantize_image(roi_raw, LEVELS)
   
    print(f"ROI Shape: {roi_quant.shape}")
    print(f"Quantization Levels: {LEVELS}")
    print("-" * 30)

    # 3. Run Global Benchmark
    time_global = run_global_glcm(roi_quant)
    print(f"Method 1 (Global GLCM) Time:   {time_global:.2f} ms")

    # 4. Run Sliding Benchmark
    time_sliding = run_sliding_glcm(roi_quant)
    print(f"Method 2 (Sliding GLCM) Time:  {time_sliding:.2f} ms")
   
    # 5. The Argument for FPGA
    ratio = time_sliding / time_global
    print("-" * 30)
    print(f"Sliding Window is {ratio:.1f}x slower on CPU.")
    print("This High Latency is why we need FPGA acceleration.")

else:
    print("File not found. Please check the path.")
