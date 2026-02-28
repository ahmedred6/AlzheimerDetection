import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
from skimage.feature import graycomatrix, graycoprops
from tqdm import tqdm

# --- CONFIGURATION ---
INPUT_DIR = "AlzhiemerDisease/Final_Dataset"
OUTPUT_FILE = "AlzhiemerDisease/Full_Radiomics_Features.csv"
DEBUG_DIR = "AlzhiemerDisease/Debug_Corrupted" # New Debug Folder

# 3D SETTINGS
WINDOW_SIZE = 5      
LEVELS = 32          
STRIDE = 1          

# THE 13 DIRECTIONS
OFFSETS_3D = [
    (0, 0, 1), (0, 1, 0), (1, 0, 0), (0, 1, 1), (0, 1, -1),
    (1, 0, 1), (1, 0, -1), (1, 1, 0), (1, -1, 0),
    (1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1)
]

# Ensure debug folder exists
os.makedirs(DEBUG_DIR, exist_ok=True)

def save_debug_image(volume, filename, reason):
    """
    Saves a slice of the corrupted volume to inspect what went wrong.
    """
    plt.figure()
    # Show middle slice
    mid = volume.shape[2] // 2
    plt.imshow(volume[:, :, mid], cmap='gray')
    plt.title(f"CORRUPTED: {reason}")
    plt.colorbar()
   
    # Save to file
    save_path = os.path.join(DEBUG_DIR, f"FAIL_{filename}.png")
    plt.savefig(save_path)
    plt.close()

def calculate_entropy(glcm_norm):
    mask = glcm_norm > 0
    p = glcm_norm[mask]
    return -np.sum(p * np.log2(p))

def get_3d_features(volume, filename):
    D, H, W = volume.shape
    pad = WINDOW_SIZE // 2
   
    # 1. PRE-CHECK: Is the volume empty?
    if np.max(volume) == 0:
        save_debug_image(volume, filename, "Volume is All Zeros")
        return None # Return None to signal failure

    maps = {
        'Contrast': [], 'Dissimilarity': [], 'Homogeneity': [],
        'Energy': [], 'Correlation': [], 'Entropy': []
    }
   
    # --- SLIDING WINDOW LOOP ---
    for z in range(pad, D-pad, STRIDE):
        for y in range(pad, H-pad, STRIDE):
            for x in range(pad, W-pad, STRIDE):
               
                center_val = volume[z, y, x]
                if center_val == 0: continue

                window = volume[z-pad:z+pad+1, y-pad:y+pad+1, x-pad:x+pad+1]
               
                glcm = np.zeros((LEVELS, LEVELS), dtype=np.uint32)
                for dz, dy, dx in OFFSETS_3D:
                    neighbor = window[pad+dz, pad+dy, pad+dx]
                    glcm[center_val, neighbor] += 1
                    glcm[neighbor, center_val] += 1
               
                total = glcm.sum()
                if total == 0: continue
                glcm_norm = glcm / total
               
                glcm_expanded = glcm_norm[np.newaxis, np.newaxis, :, :]
               
                maps['Contrast'].append(graycoprops(glcm_expanded, 'contrast')[0, 0])
                maps['Dissimilarity'].append(graycoprops(glcm_expanded, 'dissimilarity')[0, 0])
                maps['Homogeneity'].append(graycoprops(glcm_expanded, 'homogeneity')[0, 0])
                maps['Energy'].append(graycoprops(glcm_expanded, 'energy')[0, 0])
                maps['Correlation'].append(graycoprops(glcm_expanded, 'correlation')[0, 0])
                maps['Entropy'].append(calculate_entropy(glcm_norm))

    # --- STATISTICAL AGGREGATION ---
    final_stats = {}
   
    # Check if we got ANY valid windows
    if len(maps['Contrast']) < 5:
        save_debug_image(volume, filename, "Too Few Valid Windows (Empty Crop?)")
        return None

    for name, values in maps.items():
        values = np.array(values)
        values = values[values > 1e-6]
        values = values[~np.isnan(values)]

        if len(values) < 5:
            final_stats[f'{name}_Mean'] = 0
            final_stats[f'{name}_Var'] = 0
            final_stats[f'{name}_Skew'] = 0
            final_stats[f'{name}_Kurt'] = 0
            final_stats[f'{name}_Max'] = 0
            final_stats[f'{name}_Range'] = 0
        else:
            variance = np.var(values)
           
            final_stats[f'{name}_Mean'] = np.mean(values)
            final_stats[f'{name}_Var'] = variance
            final_stats[f'{name}_Max'] = np.max(values)
            final_stats[f'{name}_Range'] = np.ptp(values)
           
            # SAFETY LOCK & DEBUG TRIGGER
            if variance < 1e-6:
                # If variance is 0, the texture is suspiciously flat.
                # Could be a processing error or just a very blurry scan.
                # We save a picture just in case, but we don't discard it yet.
                if name == 'Entropy': # Only save image if Entropy is flat (most suspicious)
                     save_debug_image(volume, filename, f"Flat {name} Map")
               
                final_stats[f'{name}_Skew'] = 0
                final_stats[f'{name}_Kurt'] = 0
            else:
                final_stats[f'{name}_Skew'] = skew(values)
                final_stats[f'{name}_Kurt'] = kurtosis(values)

    return final_stats

def process_batch():
    groups = ['AD', 'NC', 'MCI']
    data_rows = []
   
    print(f"=== ROBUST FEATURE EXTRACTION (WITH DEBUGGER) ===")
    print(f"Debug Images will be saved to: {DEBUG_DIR}")
    print("-" * 50)
   
    for group in groups:
        folder = os.path.join(INPUT_DIR, group)
        if not os.path.exists(folder): continue
       
        files = [f for f in os.listdir(folder) if f.endswith(".npy")]
        print(f"Processing {group} ({len(files)} scans)...")
       
        for file in tqdm(files):
            try:
                path = os.path.join(folder, file)
                vol = np.load(path)
               
                # Pass filename to function for debugging
                feats = get_3d_features(vol, file)
               
                if feats is not None:
                    feats['Label'] = group
                    feats['Filename'] = file
                    data_rows.append(feats)
                else:
                    # If None was returned, it was corrupted.
                    # We create a "Dummy" row so we track that it failed
                    dummy = {'Label': group, 'Filename': file, 'STATUS': 'CORRUPTED'}
                    data_rows.append(dummy)
               
            except Exception as e:
                print(f"Error {file}: {e}")

    df = pd.DataFrame(data_rows)
    df.to_csv(OUTPUT_FILE, index=False)
   
    print("\n" + "="*50)
    print(f"DONE. Check {DEBUG_DIR} for any corrupted images.")
    print("="*50)

if __name__ == "__main__":
    process_batch()
