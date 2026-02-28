import os
import ants
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from nilearn import datasets

# =============================
# CONFIG
# =============================
ROOT_DIR = "AlzhiemerDisease/Registered_MNI"
CLASSES = ["AD", "MCI", "NC"]

OUTPUT_NPY_DIR = "AlzhiemerDisease/Hippocampus_Crops"
QC_DIR = "AlzhiemerDisease/Hippocampus_QC"

CUBE_SIZE = 64
HALF = CUBE_SIZE // 2
MAX_QC = 30

os.makedirs(OUTPUT_NPY_DIR, exist_ok=True)
os.makedirs(QC_DIR, exist_ok=True)

# =============================
# Helper functions
# =============================
def ras_to_lps(mm_ras):
    x, y, z = mm_ras
    return (-x, -y, z)

def crop_cube(vol, center):
    cx, cy, cz = map(int, center)
    return vol[
        cx-HALF:cx+HALF,
        cy-HALF:cy+HALF,
        cz-HALF:cz+HALF
    ]

def draw_box(ax, center_xyz, plane):
    cx, cy, cz = map(int, center_xyz)

    if plane == "axial":
        anchor_x = cx - HALF
        anchor_y = cy - HALF
    elif plane == "coronal":
        anchor_x = cx - HALF
        anchor_y = cz - HALF
    elif plane == "sagittal":
        anchor_x = cy - HALF
        anchor_y = cz - HALF

    rect = plt.Rectangle(
        (anchor_x, anchor_y),
        CUBE_SIZE,
        CUBE_SIZE,
        edgecolor="red",
        facecolor="none",
        linewidth=2
    )
    ax.add_patch(rect)

# =============================
# 1️⃣ Load atlas & compute centroids (ONCE)
# =============================
print("Loading atlas...")
ho = datasets.fetch_atlas_harvard_oxford("sub-maxprob-thr25-2mm")
atlas_img = ho.maps
atlas_data = atlas_img.get_fdata().astype(np.int16)

labels = ho.labels
left_label = next(i for i, n in enumerate(labels) if "left hippocampus" in str(n).lower())
right_label = next(i for i, n in enumerate(labels) if "right hippocampus" in str(n).lower())

left_vox = np.argwhere(atlas_data == left_label)
right_vox = np.argwhere(atlas_data == right_label)

left_centroid_vox = left_vox.mean(axis=0)
right_centroid_vox = right_vox.mean(axis=0)

left_mni_mm_lps = ras_to_lps(nib.affines.apply_affine(atlas_img.affine, left_centroid_vox))
right_mni_mm_lps = ras_to_lps(nib.affines.apply_affine(atlas_img.affine, right_centroid_vox))

print("Atlas centroids computed.")

# =============================
# 2️⃣ Iterate over dataset
# =============================
qc_counter = 0

for cls in CLASSES:

    input_dir = os.path.join(ROOT_DIR, cls)
    output_dir = os.path.join(OUTPUT_NPY_DIR, cls)
    os.makedirs(output_dir, exist_ok=True)

    files = sorted([f for f in os.listdir(input_dir) if f.endswith(".nii.gz")])

    print(f"\nProcessing class {cls} ({len(files)} scans)")

    for fname in files:

        in_path = os.path.join(input_dir, fname)
        out_path = os.path.join(output_dir, fname.replace(".nii.gz", ".npy"))

        if os.path.exists(out_path):
            continue

        img = ants.image_read(in_path)
        vol = img.numpy()

        left_center = ants.transform_physical_point_to_index(img, left_mni_mm_lps)
        right_center = ants.transform_physical_point_to_index(img, right_mni_mm_lps)

        # Boundary safety
        sx, sy, sz = vol.shape
        for c in [left_center, right_center]:
            cx, cy, cz = map(int, c)
            if not (HALF <= cx < sx-HALF and
                    HALF <= cy < sy-HALF and
                    HALF <= cz < sz-HALF):
                print("Skipping (out of bounds):", fname)
                continue

        left_crop = crop_cube(vol, left_center)
        right_crop = crop_cube(vol, right_center)

        # Concatenate channels
        stacked = np.stack([left_crop, right_crop], axis=0)

        np.save(out_path, stacked)

        # =============================
        # QC IMAGE (first 30 only)
        # =============================
        if qc_counter < MAX_QC:

            lx, ly, lz = map(int, left_center)
            rx, ry, rz = map(int, right_center)

            fig, axes = plt.subplots(3, 2, figsize=(10, 12))

            axes[0,0].imshow(vol[:,:,lz].T, cmap="gray", origin="lower")
            draw_box(axes[0,0], left_center, "axial")
            axes[0,0].set_title("Axial - Left")
            axes[0,0].axis("off")

            axes[0,1].imshow(vol[:,:,rz].T, cmap="gray", origin="lower")
            draw_box(axes[0,1], right_center, "axial")
            axes[0,1].set_title("Axial - Right")
            axes[0,1].axis("off")

            axes[1,0].imshow(vol[:,ly,:].T, cmap="gray", origin="lower")
            draw_box(axes[1,0], left_center, "coronal")
            axes[1,0].set_title("Coronal - Left")
            axes[1,0].axis("off")

            axes[1,1].imshow(vol[:,ry,:].T, cmap="gray", origin="lower")
            draw_box(axes[1,1], right_center, "coronal")
            axes[1,1].set_title("Coronal - Right")
            axes[1,1].axis("off")

            axes[2,0].imshow(vol[lx,:,:].T, cmap="gray", origin="lower")
            draw_box(axes[2,0], left_center, "sagittal")
            axes[2,0].set_title("Sagittal - Left")
            axes[2,0].axis("off")

            axes[2,1].imshow(vol[rx,:,:].T, cmap="gray", origin="lower")
            draw_box(axes[2,1], right_center, "sagittal")
            axes[2,1].set_title("Sagittal - Right")
            axes[2,1].axis("off")

            qc_path = os.path.join(QC_DIR, fname.replace(".nii.gz", "_QC.png"))
            plt.tight_layout()
            plt.savefig(qc_path, dpi=200, bbox_inches="tight")
            plt.close(fig)

            qc_counter += 1

        print("Saved:", out_path)

print("\nDataset cropping complete.")