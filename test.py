import os

file_path = "AlzhiemerDisease/Hippocampus_Cubes/AD/002_S_0816_20060929.npy"

import numpy as np

arr = np.load("AlzhiemerDisease/Hippocampus_Cubes/AD/002_S_0816_20060929.npy")

print("Array shape:", arr.shape)
print("Data type:", arr.dtype)
print("Memory used (bytes):", arr.nbytes)
print("Memory used (MB):", arr.nbytes / (1024**2))