# Imports

# Torch
import torch
from torchvision.io import read_image, ImageReadMode
from data_tools.sampling import sample_patient_images

# Utils
from pathlib import Path


def read_image_sequence(patient_path: str, indices: list[int], device:str) -> torch.Tensor:
    images = []
    for index in indices:
        image_path = f"{patient_path}/{index}.jpg"
        image = read_image(image_path, ImageReadMode.RGB).to(device)
        images.append(image)
    return torch.stack(images).to(device)


def get_all_images(root_dir: Path) -> list:
    return [patient.name for patient in root_dir.glob("*.jpg") if patient.is_file()]


def get_all_patients(root_dir: Path) -> list:
    return [patient.name for patient in root_dir.iterdir() if patient.is_dir()]


def retrieve_image_sequence(
    directory: str,
    patient_dir: str,
    patient_label: str,
    device: str,
) -> torch.Tensor:
    patient_path = f"{directory}/{patient_label}s/{patient_dir}"
    indices = sample_patient_images(
        directory=directory,
        image_num=18,
        patient_dir=patient_dir,
        label=patient_label,
        strategy="FIFO",
    )
    return read_image_sequence(
        patient_path=patient_path,
        indices=indices,
        device=device,
    )
