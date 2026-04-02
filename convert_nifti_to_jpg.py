"""Convert NIfTI CT scans to JPG images organized per patient for the project pipeline.

Reads ct_scans/*.nii from the PhysioNet download, applies brain windowing,
and saves JPGs into dataset/Patients_CT/{patient_id:03d}/brain/{slice_num}.jpg.
Also generates the two required CSVs:
  - dataset/hemorrhage_diagnosis_per_slice.csv
  - dataset/hemorrhage_diagnosis_per_patient.csv
"""

from pathlib import Path
import numpy as np
import nibabel as nib
import pandas as pd
from PIL import Image

PHYSIONET_DIR = Path("physionet.org/files/ct-ich/1.3.1")
DATASET_DIR = Path("dataset")
PATIENTS_CT_DIR = DATASET_DIR / "Patients_CT"

# Brain window parameters (from the original split_raw_data.py)
W_LEVEL = 40
W_WIDTH = 120


def window_ct(ct_scan: np.ndarray) -> np.ndarray:
    """Apply brain window to CT scan."""
    w_min = W_LEVEL - W_WIDTH / 2
    w_max = W_LEVEL + W_WIDTH / 2
    ct = ct_scan.copy().astype(np.float64)
    ct = (ct - w_min) * (255 / (w_max - w_min))
    ct = np.clip(ct, 0, 255)
    return ct.astype(np.uint8)


def convert_all():
    raw_csv = pd.read_csv(
        PHYSIONET_DIR / "hemorrhage_diagnosis_raw_ct.csv",
        encoding="utf-8-sig",
    )

    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    # Build per-slice and per-patient CSVs
    slice_rows = []
    patient_hemorrhage = {}

    ct_scans_dir = PHYSIONET_DIR / "ct_scans"

    for nii_file in sorted(ct_scans_dir.glob("*.nii")):
        patient_id = int(nii_file.stem)
        print(f"Processing patient {patient_id}...")

        # Load NIfTI
        ct_nifti = nib.load(str(nii_file))
        ct_data = ct_nifti.get_fdata()
        ct_windowed = window_ct(ct_data)

        # Create output directory
        brain_dir = PATIENTS_CT_DIR / f"{patient_id:03d}" / "brain"
        brain_dir.mkdir(parents=True, exist_ok=True)

        # Get slice info from CSV
        patient_slices = raw_csv[raw_csv["PatientNumber"] == patient_id]

        num_slices = ct_windowed.shape[2]

        # Determine if patient has any hemorrhage
        has_hemorrhage = 0
        if not patient_slices.empty:
            # No_Hemorrhage == 0 means hemorrhage IS present
            has_hemorrhage = int((patient_slices["No_Hemorrhage"] == 0).any())

        patient_hemorrhage[patient_id] = has_hemorrhage

        for slice_idx in range(num_slices):
            slice_img = ct_windowed[:, :, slice_idx]

            # Resize to 512x512 as in original script
            img = Image.fromarray(slice_img)
            img = img.resize((512, 512), Image.LANCZOS)

            # Save as JPG (1-indexed slice numbers to match project convention)
            slice_num = slice_idx + 1
            img.save(brain_dir / f"{slice_num}.jpg", "JPEG", quality=95)

            # Determine per-slice label
            slice_info = patient_slices[
                patient_slices["SliceNumber"] == slice_num
            ]
            if not slice_info.empty:
                no_hemorrhage = int(slice_info.iloc[0]["No_Hemorrhage"])
                positive_image = 1 - no_hemorrhage
            else:
                positive_image = 0

            slice_rows.append(
                {
                    "PatientNumber": patient_id,
                    "SliceNumber": slice_num,
                    "positive_image": positive_image,
                }
            )

        print(f"  -> {num_slices} slices saved, hemorrhage={has_hemorrhage}")

    # Write per-slice CSV
    slice_df = pd.DataFrame(slice_rows)
    slice_df.to_csv(
        DATASET_DIR / "hemorrhage_diagnosis_per_slice.csv", index=False
    )
    print(f"\nWrote hemorrhage_diagnosis_per_slice.csv ({len(slice_df)} rows)")

    # Write per-patient CSV
    patient_df = pd.DataFrame(
        [
            {"PatientNumber": pid, "has_hemorrhage": h}
            for pid, h in sorted(patient_hemorrhage.items())
        ]
    )
    patient_df.to_csv(
        DATASET_DIR / "hemorrhage_diagnosis_per_patient.csv", index=False
    )
    print(f"Wrote hemorrhage_diagnosis_per_patient.csv ({len(patient_df)} rows)")

    # Summary
    pos = sum(1 for v in patient_hemorrhage.values() if v == 1)
    neg = sum(1 for v in patient_hemorrhage.values() if v == 0)
    print(f"\nDataset summary: {pos} positive, {neg} negative patients")


if __name__ == "__main__":
    convert_all()
