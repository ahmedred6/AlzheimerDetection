import os
import dicom2nifti
import shutil
import pydicom

# --- CONFIGURATION ---
# SAFETY CHECK: Ensure these are NOT the same folder
RAW_DATA_DIR = "AlzhiemerDisease/Dataset"        # Your Source (Read-Only)
OUTPUT_DIR = "AlzhiemerDisease/Processed_Data_AD"   # Your Destination (Write)

groups = ['AD']

def get_date_from_dicom(folder_path):
    """
    Reads the header of the first DICOM file found in the folder
    to extract the specific Study Date (YYYYMMDD).
    """
    try:
        for f in os.listdir(folder_path):
            if f.lower().endswith(".dcm"):
                full_path = os.path.join(folder_path, f)
                # Read only the header (stop_before_pixels=True) for speed/safety
                ds = pydicom.dcmread(full_path, stop_before_pixels=True)
                return ds.StudyDate
    except Exception as e:
        return "UnknownDate"
    return "UnknownDate"

def convert_and_rename(group_name):
    print(f"\n=== Starting Group: {group_name} ===")
   
    group_input_path = os.path.join(RAW_DATA_DIR, group_name)
    group_output_path = os.path.join(OUTPUT_DIR, group_name)
   
    # Create the output folder
    os.makedirs(group_output_path, exist_ok=True)

    # os.walk recursively digs through every folder level shown in your image
    for root, dirs, files in os.walk(group_input_path):
       
        # We only trigger if we find actual .dcm files (Level 6 in your image)
        if any(f.lower().endswith('.dcm') for f in files):
           
            # 1. Identify the Patient (Parse path for XXX_S_XXXX)
            path_parts = root.split(os.sep)
            subject_id = None
            for part in path_parts:
                if "_S_" in part and len(part) >= 10:
                    subject_id = part
                    break
           
            # 2. Identify the Date (Read the file header)
            scan_date = get_date_from_dicom(root)
           
            if subject_id:
                # 3. Create Unique Filename: ID + DATE
                # Example: 002_S_0729_20060802.nii.gz
                final_name = f"{subject_id}_{scan_date}.nii.gz"
                final_path = os.path.join(group_output_path, final_name)
               
                # Skip if already done
                if os.path.exists(final_path):
                    print(f"  [Skip] {final_name} exists.")
                    continue

                print(f"Processing: {subject_id} | Date: {scan_date} ...")
               
                # Create a temporary folder inside Processed_Data (Safe sandbox)
                temp_path = os.path.join(group_output_path, "temp_conversion")
                if os.path.exists(temp_path): shutil.rmtree(temp_path)
                os.makedirs(temp_path, exist_ok=True)
               
                try:
                    # CONVERT: Read from Raw, Write to Temp
                    dicom2nifti.convert_directory(root, temp_path, compression=True, reorient=True)
                   
                    # RENAME & MOVE
                    generated_files = os.listdir(temp_path)
                    if generated_files:
                        src = os.path.join(temp_path, generated_files[0])
                        shutil.move(src, final_path)
                        print(f"    -> Success! Saved as: {final_name}")
                       
                except Exception as e:
                    print(f"    -> Error: {e}")
               
                # Cleanup Temp
                if os.path.exists(temp_path): shutil.rmtree(temp_path)

# --- EXECUTE ---
for group in groups:
    if os.path.exists(os.path.join(RAW_DATA_DIR, group)):
        convert_and_rename(group)
    else:
        print(f"Warning: Group folder '{group}' not found inside {RAW_DATA_DIR}")

print("\nAll Done! Your original data was NOT touched.")
