import os
import pandas as pd
from pathlib import Path
import shutil

# Define the path to the dataset
dataset_dir = Path("dataset/Patients_CT")

# Function to remove the 'bone' folder for each patient
def remove_bone_folders():
    for patient_id in range(49, 131):
        bone_dir = dataset_dir / f"{patient_id:03d}" / "bone"
        if bone_dir.exists() and bone_dir.is_dir():
            shutil.rmtree(bone_dir)
            print(f"Removed {bone_dir}")

# Function to remove images ending with '_HGE_Seg.jpg' in the 'brain' folder for each patient
def remove_segmentation_images():
    for patient_id in range(49, 131):
        brain_dir = dataset_dir / f"{patient_id:03d}" / "brain"
        if brain_dir.exists() and brain_dir.is_dir():
            for img_file in brain_dir.glob("*_HGE_Seg.jpg"):
                img_file.unlink()
                print(f"Removed {img_file}")

# Function to move contents of 'brain' folder to parent folder and remove 'brain' folder
def move_and_remove_brain_folders():
    for patient_id in range(49, 131):
        brain_dir = dataset_dir / f"{patient_id:03d}" / "brain"
        parent_dir = brain_dir.parent
        if brain_dir.exists() and brain_dir.is_dir():
            for item in brain_dir.iterdir():
                shutil.move(str(item), str(parent_dir))
                print(f"Moved {item} to {parent_dir}")
            shutil.rmtree(brain_dir)
            print(f"Removed {brain_dir}")

# Function to set labels per patient and organize them into 'Positives' and 'Negatives' folders
def set_labels_per_patient():
    hemorrhage_diagnosis = pd.read_csv("dataset/hemorrhage_diagnosis_per_patient.csv")
    positives_dir = dataset_dir.parent / "Positives"
    negatives_dir = dataset_dir.parent / "Negatives"
    
    positives_dir.mkdir(parents=True, exist_ok=True)
    negatives_dir.mkdir(parents=True, exist_ok=True)
    
    for _, row in hemorrhage_diagnosis.iterrows():
        patient_id = row['PatientNumber']
        diagnosis = row['has_hemorrhage']
        patient_dir = dataset_dir / f"{patient_id:03d}"
        
        if diagnosis == "1" or diagnosis == 1:
            if not (positives_dir / f"{patient_id:03d}").exists():
                shutil.move(str(patient_dir), str(positives_dir))
                print(f"Moved {patient_dir} to {positives_dir}")
        else:
            if not (negatives_dir / f"{patient_id:03d}").exists():
                shutil.move(str(patient_dir), str(negatives_dir))
                print(f"Moved {patient_dir} to {negatives_dir}")
    
    if dataset_dir.exists() and dataset_dir.is_dir():
        shutil.rmtree(dataset_dir)

# Function to count the number of folders in 'Positives' and 'Negatives' directories
def count_folders():
    positives_dir = dataset_dir.parent / "Positives"
    negatives_dir = dataset_dir.parent / "Negatives"
    
    positives_count = sum(1 for _ in positives_dir.iterdir() if _.is_dir())
    negatives_count = sum(1 for _ in negatives_dir.iterdir() if _.is_dir())
    
    print(f"Number of folders in Positives: {positives_count}")
    print(f"Number of folders in Negatives: {negatives_count}")

# Function to find the minimum number of images in the folders of all the patients and identify which patient
def find_min_images():
    positives_dir = dataset_dir.parent / "Positives"
    negatives_dir = dataset_dir.parent / "Negatives"
    
    min_images = float('inf')
    min_patient = None
    
    for patient_dir in positives_dir.iterdir():
        if patient_dir.is_dir():
            num_images = sum(1 for _ in patient_dir.glob("*.jpg"))
            if num_images < min_images:
                min_images = num_images
                min_patient = patient_dir.name
    
    for patient_dir in negatives_dir.iterdir():
        if patient_dir.is_dir():
            num_images = sum(1 for _ in patient_dir.glob("*.jpg"))
            if num_images < min_images:
                min_images = num_images
                min_patient = patient_dir.name
    
    print(f"Minimum number of images in any patient folder: {min_images}")
    print(f"Patient with minimum number of images: {min_patient}")

# Execute the function
remove_bone_folders()
remove_segmentation_images()
move_and_remove_brain_folders()
set_labels_per_patient()
count_folders()
find_min_images()