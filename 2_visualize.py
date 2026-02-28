import nibabel as nib
import numpy as np
import os
import time

# --- CONFIGURATION ---
INPUT_BASE = "AlzhiemerDisease/Processed_Data"
FPGA_OUTPUT = "AlzhiemerDisease/Final_FPGA_Data"  # For PYNQ-Z2 (Quantized)
PC_OUTPUT = "AlzhiemerDisease/Final_PC_Baseline"  # For Laptop (Full Precision)

# THE CONFIRMED CALIBRATION COORDINATES
# Format: (x, y, z)
L_CENTER = (60, 128, 128)
R_CENTER = (110, 128, 128)
ROI_SIZE = 64
LEVELS = 32 # 5-bit Quantization for FPGA BRAM

def extract_and_process_batch():
    groups = ['MCI']
    r = ROI_SIZE // 2
   
    start_time = time.time()
    total_processed = 0
    total_skipped = 0
   
    print("=== STARTING BATCH PROCESSING ===")
    print(f"FPGA Output: {FPGA_OUTPUT}")
    print(f"PC Output:   {PC_OUTPUT}")
    print(f"Coordinates: Left{L_CENTER} | Right{R_CENTER}")
    print("-----------------------------------")
   
    for group in groups:
        input_dir = os.path.join(INPUT_BASE, group)
       
        # Create output folders for this group
        fpga_group_dir = os.path.join(FPGA_OUTPUT, group)
        pc_group_dir = os.path.join(PC_OUTPUT, group)
        os.makedirs(fpga_group_dir, exist_ok=True)
        os.makedirs(pc_group_dir, exist_ok=True)
       
        if not os.path.exists(input_dir):
            print(f"Skipping missing group: {group}")
            continue
       
        files = [f for f in os.listdir(input_dir) if f.endswith(".nii.gz")]
        print(f"\nProcessing Group [{group}]: {len(files)} scans found.")

        for i, file in enumerate(files):
            try:
                # Load NIfTI
                img = nib.load(os.path.join(input_dir, file))
                data = img.get_fdata()
               
                # Double Check Data Shape (Must be at least the size of our coordinates)
                if data.shape != (193, 229, 193):
                    # If dimensions are slightly different but still large enough, we proceed.
                    # If too small, we skip.
                    if data.shape[0] < 120 or data.shape[1] < 150 or data.shape[2] < 150:
                        print(f"  [Skip] {file}: Volume too small {data.shape}")
                        total_skipped += 1
                        continue

                # Process Both Sides (Left & Right)
                for side, (cx, cy, cz) in [('L', L_CENTER), ('R', R_CENTER)]:
                   
                    # 1. CUT THE CUBE
                    # We use standard slicing.
                    roi = data[cx-r:cx+r, cy-r:cy+r, cz-r:cz+r]
                   
                    # Final Shape Guard
                    if roi.shape != (64, 64, 64):
                        print(f"  [Error] {file} ({side}): ROI truncated {roi.shape}")
                        continue
                       
                    base_name = file.replace(".nii.gz", f"_{side}")

                    # 2. PATH A: FPGA (Quantize to 0-31 for Verilog)
                    # Normalize 0 to 1
                    roi_norm = (roi - np.min(roi)) / (np.max(roi) - np.min(roi) + 1e-8)
                    # Scale to 0-31 integer
                    quantized = (roi_norm * (LEVELS - 1)).astype(np.uint8)
                    # Save as raw binary
                    quantized.tofile(os.path.join(fpga_group_dir, f"{base_name}.bin"))

                    # 3. PATH B: PC (Save Full Precision Float for Benchmark)
                    np.save(os.path.join(pc_group_dir, f"{base_name}.npy"), roi)
               
                total_processed += 1
               
                # Progress Update every 100 scans
                if (i + 1) % 100 == 0:
                    print(f"  -> {i + 1}/{len(files)} done...")

            except Exception as e:
                print(f"  [Exception] {file}: {e}")
                total_skipped += 1

    duration = (time.time() - start_time) / 60
    print("\n===================================")
    print(f"BATCH COMPLETE in {duration:.2f} minutes.")
    print(f"Total Scans Processed: {total_processed}")
    print(f"Total Skipped: {total_skipped}")
    print("===================================")

if __name__ == "__main__":
    extract_and_process_batch()
