import os
import shutil
from datetime import datetime

# Define the path to the root folder containing patient folders
root_folder = r'MiniProject\SkelLabel\'

# Define the start and end dates for modification
start_date = datetime(2024, 4, 1)
end_date = datetime(2024, 4, 6)

# Iterate through each patient folder
for patient_folder in os.listdir(root_folder):
    patient_folder_path = os.path.join(root_folder, patient_folder)
    
    # Check if it's a directory
    if os.path.isdir(patient_folder_path):
        # Get the modification date of gt.nii.gz file
        gt_file_path = os.path.join(patient_folder_path, 'gt.nii.gz')
        if os.path.exists(gt_file_path):
            modification_time = os.path.getmtime(gt_file_path)
            modification_date = datetime.fromtimestamp(modification_time)
            
            # Check if the modification date falls within the specified range
            if start_date <= modification_date <= end_date:
                # Create a new folder to store the modified gt.nii.gz file
                new_folder_path = os.path.join(root_folder, 'Modified', patient_folder)
                os.makedirs(new_folder_path, exist_ok=True)
                
                # Copy the modified gt.nii.gz file to the new folder
                shutil.copy(gt_file_path, new_folder_path)
