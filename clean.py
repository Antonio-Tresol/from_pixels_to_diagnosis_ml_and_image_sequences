import pandas as pd
from pathlib import Path
import shutil

dataset_dir: Path = Path("dataset/Patients_CT")


def remove_bone_folders() -> None:
    """Remove 'bone' folders from all patient directories."""
    for patient_id in range(49, 131):
        bone_dir: Path = dataset_dir / f"{patient_id:03d}" / "bone"
        if bone_dir.exists() and bone_dir.is_dir():
            shutil.rmtree(bone_dir)
            print(f"Removed {bone_dir}")


def remove_segmentation_images() -> None:
    """Remove segmentation images ending with '_HGE_Seg.jpg' from brain folders."""
    for patient_id in range(49, 131):
        brain_dir: Path = dataset_dir / f"{patient_id:03d}" / "brain"
        if brain_dir.exists() and brain_dir.is_dir():
            for img_file in brain_dir.glob("*_HGE_Seg.jpg"):
                img_file.unlink()
                print(f"Removed {img_file}")


def move_and_remove_brain_folders() -> None:
    """Move contents of brain folders to parent directories and remove brain folders."""
    for patient_id in range(49, 131):
        brain_dir: Path = dataset_dir / f"{patient_id:03d}" / "brain"
        parent_dir: Path = brain_dir.parent
        if brain_dir.exists() and brain_dir.is_dir():
            for item in brain_dir.iterdir():
                shutil.move(str(item), str(parent_dir))
                print(f"Moved {item} to {parent_dir}")
            shutil.rmtree(brain_dir)
            print(f"Removed {brain_dir}")


def set_labels_per_patient() -> None:
    """Organize patients into Positives and Negatives folders based on hemorrhage diagnosis.
    Reads diagnosis from CSV file and moves patient folders accordingly.
    """
    hemorrhage_diagnosis: pd.DataFrame = pd.read_csv(
        "dataset/hemorrhage_diagnosis_per_patient.csv"
    )
    positives_dir: Path = dataset_dir.parent / "Positives"
    negatives_dir: Path = dataset_dir.parent / "Negatives"

    positives_dir.mkdir(parents=True, exist_ok=True)
    negatives_dir.mkdir(parents=True, exist_ok=True)

    for _, row in hemorrhage_diagnosis.iterrows():
        patient_id: int = row["PatientNumber"]
        diagnosis: int = row["has_hemorrhage"]
        patient_dir: Path = dataset_dir / f"{patient_id:03d}"

        if diagnosis in {"1", 1}:
            if not (positives_dir / f"{patient_id:03d}").exists():
                shutil.move(str(patient_dir), str(positives_dir))
                print(f"Moved {patient_dir} to {positives_dir}")
        elif not (negatives_dir / f"{patient_id:03d}").exists():
            shutil.move(str(patient_dir), str(negatives_dir))
            print(f"Moved {patient_dir} to {negatives_dir}")

    if dataset_dir.exists() and dataset_dir.is_dir():
        shutil.rmtree(dataset_dir)


def count_folders() -> tuple[int, int]:
    """Count number of folders in Positives and Negatives directories.

    Returns
    -------
        tuple[int, int]: Count of positive and negative folders

    """
    positives_dir: Path = dataset_dir.parent / "Positives"
    negatives_dir: Path = dataset_dir.parent / "Negatives"

    positives_count: int = sum(1 for _ in positives_dir.iterdir() if _.is_dir())
    negatives_count: int = sum(1 for _ in negatives_dir.iterdir() if _.is_dir())

    print(f"Number of folders in Positives: {positives_count}")
    print(f"Number of folders in Negatives: {negatives_count}")
    return positives_count, negatives_count


def find_min_images() -> tuple[int, str]:
    """Find the minimum number of images across all patient folders.

    Returns
    -------
        tuple[int, str]: Minimum image count and corresponding patient ID

    """
    positives_dir: Path = dataset_dir.parent / "Positives"
    negatives_dir: Path = dataset_dir.parent / "Negatives"

    min_images: int = float("inf")
    min_patient: str = ""

    for patient_dir in positives_dir.iterdir():
        if patient_dir.is_dir():
            num_images: int = sum(1 for _ in patient_dir.glob("*.jpg"))
            if num_images < min_images:
                min_images = num_images
                min_patient = patient_dir.name

    for patient_dir in negatives_dir.iterdir():
        if patient_dir.is_dir():
            num_images: int = sum(1 for _ in patient_dir.glob("*.jpg"))
            if num_images < min_images:
                min_images = num_images
                min_patient = patient_dir.name

    print(f"Minimum number of images in any patient folder: {min_images}")
    print(f"Patient with minimum number of images: {min_patient}")
    return min_images, min_patient


def main() -> None:
    """Execute all dataset cleaning and organization functions."""
    remove_bone_folders()
    remove_segmentation_images()
    move_and_remove_brain_folders()
    set_labels_per_patient()
    count_folders()
    find_min_images()


if __name__ == "__main__":
    main()
