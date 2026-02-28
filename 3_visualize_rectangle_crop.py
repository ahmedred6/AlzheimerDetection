import nibabel as nib
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# --- CONFIGURATION ---
INPUT_DIR = "AlzhiemerDisease/Processed_Data/MCI"
ROI_SIZE = 64

# The "Magic Numbers" from your calibration
# Format: (x, y, z)
L_CENTER = (60, 128, 128)
R_CENTER = (110, 128, 128)

def verify_first_20_safety():
    # 1. Get the list of files
    if not os.path.exists(INPUT_DIR):
        print(f"Error: Directory not found {INPUT_DIR}")
        return

    files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".nii.gz")]
    files.sort() # Sort to ensure consistent order
   
    # Limit to first 20
    check_list = files[:20]
   
    print(f"Starting Safety Check on {len(check_list)} files...")
    print(f"Left Target: {L_CENTER} | Right Target: {R_CENTER}")

    r = ROI_SIZE // 2

    for i, file in enumerate(check_list):
        path = os.path.join(INPUT_DIR, file)
        img = nib.load(path)
        data = img.get_fdata()
       
        # We define the slices based on our fixed centers
        # Sagittal: Slice at Left X (60)
        # Coronal: Slice at Y (128)
        # Axial: Slice at Z (128)
       
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
       
        # --- VIEW 1: SAGITTAL (Side) ---
        # Showing the Left Hippocampus depth
        axes[0].imshow(data[L_CENTER[0], :, :].T, cmap='gray', origin='lower')
        axes[0].set_title(f"Sagittal (Slice X={L_CENTER[0]})")
        # Box: Y vs Z
        rect_sag = patches.Rectangle((L_CENTER[1]-r, L_CENTER[2]-r), ROI_SIZE, ROI_SIZE,
                                     linewidth=2, edgecolor='red', facecolor='none')
        axes[0].add_patch(rect_sag)

        # --- VIEW 2: CORONAL (Front) ---
        # Showing both Left and Right side-by-side
        axes[1].imshow(data[:, L_CENTER[1], :].T, cmap='gray', origin='lower')
        axes[1].set_title(f"Coronal (Slice Y={L_CENTER[1]})")
        # Left Box: X vs Z
        rect_cor_l = patches.Rectangle((L_CENTER[0]-r, L_CENTER[2]-r), ROI_SIZE, ROI_SIZE,
                                       linewidth=2, edgecolor='red', facecolor='none')
        # Right Box: X vs Z
        rect_cor_r = patches.Rectangle((R_CENTER[0]-r, R_CENTER[2]-r), ROI_SIZE, ROI_SIZE,
                                       linewidth=2, edgecolor='lime', facecolor='none')
        axes[1].add_patch(rect_cor_l)
        axes[1].add_patch(rect_cor_r)

        # --- VIEW 3: AXIAL (Top) ---
        # Showing both Left and Right from top down
        axes[2].imshow(data[:, :, L_CENTER[2]].T, cmap='gray', origin='lower')
        axes[2].set_title(f"Axial (Slice Z={L_CENTER[2]})")
        # Left Box: X vs Y
        rect_ax_l = patches.Rectangle((L_CENTER[0]-r, L_CENTER[1]-r), ROI_SIZE, ROI_SIZE,
                                      linewidth=2, edgecolor='red', facecolor='none')
        # Right Box: X vs Y
        rect_ax_r = patches.Rectangle((R_CENTER[0]-r, R_CENTER[1]-r), ROI_SIZE, ROI_SIZE,
                                      linewidth=2, edgecolor='lime', facecolor='none')
        axes[2].add_patch(rect_ax_l)
        axes[2].add_patch(rect_ax_r)

        # Final Plot Setup
        for ax in axes: ax.axis('off')
        plt.suptitle(f"Patient {i+1}/20: {file}\nClose window to see next...", fontsize=14)
       
        # This command blocks code execution until you close the window
        plt.show()

    print("\nSafety Check Complete. If these 20 looked good, you are ready for the batch.")

if __name__ == "__main__":
    verify_first_20_safety()