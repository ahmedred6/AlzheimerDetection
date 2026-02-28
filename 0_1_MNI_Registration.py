import os
import ants

INPUT_DIR = "AlzhiemerDisease/Resampled_Data/NC"
OUTPUT_DIR = "AlzhiemerDisease/Registered_MNI/NC"
TEMPLATE_PATH = "AlzhiemerDisease/MNI_Template/MNI152_T1_1mm.nii.gz"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load template once (very important)
fixed = ants.image_read(TEMPLATE_PATH)

files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".nii.gz")]
total = len(files)

print(f"Found {total} scans to process.\n")

for i, filename in enumerate(files, 1):

    in_path = os.path.join(INPUT_DIR, filename)
    out_path = os.path.join(OUTPUT_DIR, filename)

    # Skip if already registered
    if os.path.exists(out_path):
        print(f"[{i}/{total}] Skipping {filename} (already registered)")
        continue

    print(f"[{i}/{total}] Registering {filename}...")

    try:
        moving = ants.image_read(in_path)

        reg = ants.registration(
            fixed=fixed,
            moving=moving,
            type_of_transform="Affine"   # fast + sufficient for your atlas ROI
        )

        warped = reg["warpedmovout"]
        ants.image_write(warped, out_path)

        print(f"Saved → {out_path}\n")

    except Exception as e:
        print(f"[ERROR] Failed on {filename}: {e}\n")

print("Registration complete.")