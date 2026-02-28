import matplotlib.pyplot as plt
import ants
import numpy as np
import os

TEMPLATE_PATH = "AlzhiemerDisease/MNI_Template/MNI152_T1_1mm.nii.gz"
REGISTERED_DIR = "AlzhiemerDisease/Registered_MNI/AD"
OUTPUT_DIR = "AlzhiemerDisease/registration_revised"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading template...")
template = ants.image_read(TEMPLATE_PATH)
template_np = template.numpy()

files = sorted([f for f in os.listdir(REGISTERED_DIR) if f.endswith(".nii.gz")])

print(f"Found {len(files)} registered scans.")
print("Saving first 20 QC images...\n")

for i, filename in enumerate(files[:20], 1):

    path = os.path.join(REGISTERED_DIR, filename)
    print(f"[{i}/20] Processing {filename}")

    registered = ants.image_read(path)
    registered_np = registered.numpy()

    z = template_np.shape[2] // 2

    fig = plt.figure(figsize=(14, 6))

    # Side-by-side
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.set_title("MNI Template")
    ax1.imshow(template_np[:, :, z], cmap='gray')
    ax1.axis("off")

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.set_title("Registered")
    ax2.imshow(registered_np[:, :, z], cmap='gray')
    ax2.axis("off")

    # Overlay
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.set_title("Overlay")
    ax3.imshow(template_np[:, :, z], cmap='gray')
    ax3.imshow(registered_np[:, :, z], cmap='hot', alpha=0.35)
    ax3.axis("off")

    plt.tight_layout()

    save_path = os.path.join(
        OUTPUT_DIR,
        filename.replace(".nii.gz", "_QC.png")
    )

    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)

    print(f"Saved → {save_path}\n")

print("QC image generation complete.")