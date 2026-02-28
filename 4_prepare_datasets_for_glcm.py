import nibabel as nib
import numpy as np
import os
import time
from tqdm import tqdm

# --- CONFIGURATION ---
INPUT_ROOT = "AlzhiemerDisease/Processed_Data"
OUTPUT_ROOT = "AlzhiemerDisease/Final_Dataset"  # The Unified Folder

# CROP CONFIGURATION (Fixed 64x64x64 Cube)
ROI_SIZE = 64
HALF_SIZE = ROI_SIZE // 2

# COORDINATES (From your previous code)
# Format: (x, y, z)
L_CENTER = (60, 128, 128)
R_CENTER = (110, 128, 128)

# QUANTIZATION CONFIGURATION
LEVELS = 32  # 0 to 31 (5-bit)

def quantize_scan(volume, levels):
    """
    Performs 'Per-Scan' Normalization and Quantization.
    1. Finds Min/Max of this specific crop.
    2. Stretches it to 0-1.
    3. Scales it to 0-(levels-1).
    """
    v_min, v_max = volume.min(), volume.max()
   
    # Safety: If the volume is empty (all zeros), return zeros
    if v_max - v_min == 0:
        return np.zeros_like(volume, dtype=np.uint8)
   
    # Normalize to 0-1 float
    norm = (volume - v_min) / (v_max - v_min)
   
    # Scale to integer range (e.g., 0-31)
    # We subtract 0.999 to ensure the max value maps to 31, not 32
    quantized = np.floor(norm * (levels - 0.001)).astype(np.uint8)
   
    return quantized

def process_dataset():
    groups = ['AD', 'NC', 'MCI']  # The subfolders in your Processed_Data
   
    print(f"=== MASTER PREPROCESSING STARTED ===")
    print(f"Target: {OUTPUT_ROOT}")
    print(f"Quantization: {LEVELS} Levels")
    print(f"Crop Size: {ROI_SIZE}x{ROI_SIZE}x{ROI_SIZE}")
    print("-" * 40)
   
    total_saved = 0
    total_skipped = 0
   
    for group in groups:
        input_group_path = os.path.join(INPUT_ROOT, group)
        output_group_path = os.path.join(OUTPUT_ROOT, group)
       
        # Create destination folder (e.g., Final_Dataset/AD)
        os.makedirs(output_group_path, exist_ok=True)
       
        if not os.path.exists(input_group_path):
            print(f"Skipping missing group: {group}")
            continue
           
        files = [f for f in os.listdir(input_group_path) if f.endswith(".nii.gz")]
        print(f"\nProcessing Group [{group}]: {len(files)} scans...")
       
        # Use TQDM for a nice progress bar
        for file_name in tqdm(files):
            try:
                # 1. LOAD NIfTI
                file_path = os.path.join(input_group_path, file_name)
                img = nib.load(file_path)
                data = img.get_fdata()
               
                # Basic Shape Check
                if data.shape[0] < 120 or data.shape[1] < 150 or data.shape[2] < 150:
                    # Skip massive failures (e.g., corrupted small files)
                    total_skipped += 1
                    continue

                # 2. PROCESS BOTH SIDES (Left & Right)
                for side, (cx, cy, cz) in [('L', L_CENTER), ('R', R_CENTER)]:
                   
                    # Safe Slicing (Handle edges if coordinate is too close to border)
                    x1, x2 = max(0, cx - HALF_SIZE), min(data.shape[0], cx + HALF_SIZE)
                    y1, y2 = max(0, cy - HALF_SIZE), min(data.shape[1], cy + HALF_SIZE)
                    z1, z2 = max(0, cz - HALF_SIZE), min(data.shape[2], cz + HALF_SIZE)
                   
                    crop = data[x1:x2, y1:y2, z1:z2]
                   
                    # Check if crop is actually 64x64x64 (Padding check)
                    # If it's smaller (edge case), we pad it with zeros
                    if crop.shape != (ROI_SIZE, ROI_SIZE, ROI_SIZE):
                        # Calculate padding needed
                        pad_x = ROI_SIZE - crop.shape[0]
                        pad_y = ROI_SIZE - crop.shape[1]
                        pad_z = ROI_SIZE - crop.shape[2]
                        crop = np.pad(crop, ((0, pad_x), (0, pad_y), (0, pad_z)), mode='constant')

                    # 3. QUANTIZE (0-31 Integers)
                    crop_quantized = quantize_scan(crop, LEVELS)
                   
                    # 4. SAFETY CHECK: Is it empty?
                    # If the center was wrong, we might have cropped empty air (all zeros).
                    if np.max(crop_quantized) == 0:
                        # Log it but maybe save it anyway so we know it failed later?
                        # For now, let's skip saving empty "air" scans.
                        continue

                    # 5. SAVE
                    # naming convention: SubjectID_Date_Side.npy
                    # We strip .nii.gz and add _L or _R
                    clean_name = file_name.replace(".nii.gz", "")
                    save_name = f"{clean_name}_{side}.npy"
                    save_path = os.path.join(output_group_path, save_name)
                   
                    np.save(save_path, crop_quantized)
                    total_saved += 1
                   
            except Exception as e:
                print(f"Error processing {file_name}: {e}")
                total_skipped += 1

    print("\n" + "="*40)
    print(f"DONE.")
    print(f"Successfully Created: {total_saved} quantized crops")
    print(f"Skipped/Failed:       {total_skipped}")
    print(f"Location:             {OUTPUT_ROOT}")
    print("="*40)

if __name__ == "__main__":
    process_dataset()
