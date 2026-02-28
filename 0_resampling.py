import os
import nibabel as nib
import numpy as np
from monai.transforms import (
    Compose,
    LoadImage,
    EnsureChannelFirst,
    Orientation,
    Spacing,
    ScaleIntensityRange
)

INPUT_DIR = "AlzhiemerDisease/Processed_Data/NC"
OUTPUT_DIR = "AlzhiemerDisease/Resampled_Data/NC"

os.makedirs(OUTPUT_DIR, exist_ok=True)

transforms = Compose([
    # Changed to True: Returns a MetaTensor that holds both data AND metadata
    LoadImage(image_only=True), 
    EnsureChannelFirst(),
    Orientation(axcodes="RAS"),
    Spacing(pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
    ScaleIntensityRange(a_min=0, a_max=2000,
                        b_min=0.0, b_max=1.0, clip=True)
])

for filename in os.listdir(INPUT_DIR):
    if filename.endswith(".nii.gz"):
        in_path = os.path.join(INPUT_DIR, filename)
        out_path = os.path.join(OUTPUT_DIR, filename)

        if os.path.exists(out_path):
            print(f"[Skip] {filename} already resampled.")
            continue

        print(f"Resampling {filename}...")

        # Unpack just the image, which is now a MetaTensor
        img = transforms(in_path)

        # Remove channel dimension for saving back to NIfTI
        img_np = img.squeeze(0).numpy()

        # Save using the affine attached directly to the MetaTensor
        new_nifti = nib.Nifti1Image(img_np, img.affine)
        nib.save(new_nifti, out_path)

print("Done. Original data untouched.")