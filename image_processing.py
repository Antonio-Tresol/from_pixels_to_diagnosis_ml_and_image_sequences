from PIL import Image
import numpy as np
from pathlib import Path


def read_image_sequence(patient_path: str, indices: list[int]) -> np.ndarray:
    images = []
    for index in indices:
        image_path = f"{patient_path}/{index}.jpg"
        image = Image.open(image_path)
        image = image.convert("RGB")
        image_array = np.array(image)
        images.append(image_array)
        image.close()
    return np.stack(images)


def get_all_images(root_dir: Path) -> list:
    return [patient.name for patient in root_dir.glob("*.jpg") if patient.is_file()]


def get_all_patients(root_dir: Path) -> list:
    return [patient.name for patient in root_dir.iterdir() if patient.is_dir()]


def sample_patient_images(
    directory: str,
    image_num: int,
    patient_dir: str = "",
    label: str = "",
    strategy: str = "FIFO",
) -> list[int]:
    if strategy == "FIFO":
        return np.array(range(1, image_num + 1)).astype(np.int64)
    max_images = len(get_all_images(Path(f"{directory}/{label}s/{patient_dir}")))
    rng = np.random.default_rng()
    end_idx = rng.integers(17, max_images - 1)
    start_idx = max_images - end_idx
    indices = np.linspace(start_idx, end_idx, num=image_num)
    return np.clip(indices, start_idx, end_idx - 1).astype(np.int64)


def retrieve_image_sequence(
    directory: str,
    patient_dir: str,
    patient_label: str,
) -> np.ndarray:
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
    )