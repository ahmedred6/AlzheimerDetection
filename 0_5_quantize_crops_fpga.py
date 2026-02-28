import os
import csv
import numpy as np
from tqdm import tqdm

# ==========================
# PLACEHOLDERS (EDIT THESE)
# ==========================
INPUT_NPY_ROOT = "AlzhiemerDisease/Hippocampus_Cubes"          # e.g., AlzhiemerDisease/Hippocampus_Crops
OUTPUT_ROOT    = "AlzhiemerDisease/Hippocampus_Crops_Quant32"  # new folder (do not overwrite old)

CLASSES = ["AD", "MCI", "NC"]

# Quantization
LEVELS = 32            # -> 0..31
P_LOW = 1.0
P_HIGH = 99.0
IGNORE_ZEROS = True

# Safety thresholds
MIN_NZ = 500
EPS_RANGE = 1e-6

# Outputs
OUT_NPY = os.path.join(OUTPUT_ROOT, "npy_uint8")
OUT_BIN = os.path.join(OUTPUT_ROOT, "bin_uint8")
OUT_QC  = os.path.join(OUTPUT_ROOT, "qc")
QC_CSV  = os.path.join(OUT_QC, "quant_qc.csv")


def ensure_dirs():
    os.makedirs(OUT_NPY, exist_ok=True)
    os.makedirs(OUT_BIN, exist_ok=True)
    os.makedirs(OUT_QC, exist_ok=True)
    for c in CLASSES:
        os.makedirs(os.path.join(OUT_NPY, c), exist_ok=True)
        os.makedirs(os.path.join(OUT_BIN, c), exist_ok=True)


def write_qc_header():
    if os.path.exists(QC_CSV):
        return
    with open(QC_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "class", "file",
            "status",
            "ch0_nz", "ch0_lo", "ch0_hi", "ch0_zero_frac",
            "ch1_nz", "ch1_lo", "ch1_hi", "ch1_zero_frac",
        ])


def quantize_channel(vol: np.ndarray):
    """vol: (64,64,64) float -> uint8 0..31"""
    v = vol.astype(np.float32)

    if IGNORE_ZEROS:
        nz = v[v > 0]
        zero_frac = float(np.mean(v == 0))
    else:
        nz = v.reshape(-1)
        zero_frac = 0.0

    if nz.size < MIN_NZ:
        return np.zeros_like(v, dtype=np.uint8), ("TOO_SPARSE", int(nz.size), None, None, zero_frac)

    lo = float(np.percentile(nz, P_LOW))
    hi = float(np.percentile(nz, P_HIGH))

    if (hi - lo) <= EPS_RANGE:
        return np.zeros_like(v, dtype=np.uint8), ("LOW_RANGE", int(nz.size), lo, hi, zero_frac)

    v = np.clip(v, lo, hi)
    v = (v - lo) / (hi - lo)  # [0,1]
    q = np.floor(v * (LEVELS - 1e-6)).astype(np.uint8)
    return q, ("OK", int(nz.size), lo, hi, zero_frac)


def main():
    ensure_dirs()
    write_qc_header()

    print("=== Quantize hippocampus crops for FPGA (per-crop p1/p99, ignore zeros) ===")
    print(f"INPUT : {INPUT_NPY_ROOT}")
    print(f"OUTPUT: {OUTPUT_ROOT}")
    print(f"LEVELS={LEVELS}, p={P_LOW}..{P_HIGH}, IGNORE_ZEROS={IGNORE_ZEROS}")
    print("-" * 70)

    for cls in CLASSES:
        in_dir = os.path.join(INPUT_NPY_ROOT, cls)
        if not os.path.isdir(in_dir):
            print(f"[WARN] Missing: {in_dir}")
            continue

        files = sorted([f for f in os.listdir(in_dir) if f.endswith(".npy")])
        print(f"\nClass {cls}: {len(files)} crops")

        for fn in tqdm(files):
            in_path = os.path.join(in_dir, fn)
            out_npy = os.path.join(OUT_NPY, cls, fn)  # same name
            out_bin = os.path.join(OUT_BIN, cls, fn.replace(".npy", ".bin"))

            # No overwrites
            if os.path.exists(out_npy) or os.path.exists(out_bin):
                continue

            x = np.load(in_path)  # expected (2,64,64,64)
            if x.ndim != 4 or x.shape[0] != 2:
                # If your pipeline later changes layout, catch it early
                status = "BAD_SHAPE"
                with open(QC_CSV, "a", newline="") as f:
                    csv.writer(f).writerow([cls, fn, status, "", "", "", "", "", "", "", ""])
                continue

            q0, info0 = quantize_channel(x[0])
            q1, info1 = quantize_channel(x[1])

            status = "OK" if (info0[0] == "OK" and info1[0] == "OK") else f"{info0[0]}|{info1[0]}"

            q = np.stack([q0, q1], axis=0)  # (2,64,64,64) uint8

            # Save
            np.save(out_npy, q)

            # FPGA bin: channel-major contiguous flatten (C-order)
            # Layout in file: [all ch0 voxels] then [all ch1 voxels]
            q.tofile(out_bin)

            # QC log
            with open(QC_CSV, "a", newline="") as f:
                csv.writer(f).writerow([
                    cls, fn,
                    status,
                    info0[1], info0[2], info0[3], info0[4],
                    info1[1], info1[2], info1[3], info1[4],
                ])

    print("\nDONE.")
    print(f"QC CSV: {QC_CSV}")


if __name__ == "__main__":
    main()