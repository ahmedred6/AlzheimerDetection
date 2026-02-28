import os
import numpy as np
import nibabel as nib
from tqdm import tqdm

INPUT_ROOT = "AlzhiemerDisease/Processed_Data"
groups = ['AD','MCI','NC']

LOWER_P = 1
UPPER_P = 99
total_skipped = 0
all_voxels = []

print("Computing global intensity percentiles...")

for group in groups:
    group_path = os.path.join(INPUT_ROOT, group)
    files = [f for f in os.listdir(group_path) if f.endswith(".nii.gz")]

    for file_name in tqdm(files):
        file_path = os.path.join(group_path, file_name)
        try:
            img = nib.load(file_path)
            data = img.get_fdata()
        except Exception as e:
            print(f"Corrupted or invalid file: {file_name} | {e}")
            total_skipped += 1
            continue

        # IMPORTANT: only use non-zero voxels
        voxels = data[data > 0]
        if len(voxels) > 0:
            all_voxels.append(voxels)

# Concatenate safely
all_voxels = np.concatenate(all_voxels)

global_min = np.percentile(all_voxels, LOWER_P)
global_max = np.percentile(all_voxels, UPPER_P)

print("GLOBAL_MIN =", global_min)
print("GLOBAL_MAX =", global_max)

np.save("AlzhiemerDisease/global_bounds.npy", np.array([global_min, global_max]))