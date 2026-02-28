import nibabel as nib
import os

folder = "AlzhiemerDisease/Resampled_Data/AD"

for f in os.listdir(folder)[:5]:
    img = nib.load(os.path.join(folder, f))
    print(f, img.header.get_zooms())
    
