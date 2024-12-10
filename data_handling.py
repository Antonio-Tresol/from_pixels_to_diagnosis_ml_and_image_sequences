import numpy as np
from pathlib import Path
from PIL import Image
import torch
from transformers import VivitImageProcessor, ConvNextImageProcessor
from datasets import Dataset, load_from_disk

import config


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


def create_dataset_dictionary(
    directory: str,
    iterate_images: bool = False,
) -> tuple[list, list]:
    class_labels = ["Negative", "Positive"]
    all_image_sequences = []
    patients, labels = get_patients_from_dataset(
        directory=directory,
    )
    for patient_dir, patient_label in zip(patients, labels):
        image_sequence = retrieve_image_sequence(
            directory,
            patient_dir,
            class_labels[patient_label],
        )
        if iterate_images:
            all_image_sequences.extend(
                {
                    "video": image,
                    "labels": class_labels[patient_label],
                    "patient": patient_dir,
                } for image in image_sequence
            )
            print(patient_dir)
        else:
            all_image_sequences.append(
                {
                    "video": image_sequence,
                    "labels": class_labels[patient_label],
                },
            )
    return all_image_sequences, class_labels


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


def get_patients_from_dataset(directory: str) -> list:
    negative_patients = get_all_patients(Path(f"{directory}/Negatives/"))
    positive_patients = get_all_patients(Path(f"{directory}/Positives/"))
    negative_labels = [0] * len(negative_patients)
    positive_labels = [1] * len(positive_patients)

    patients = positive_patients + negative_patients
    labels = positive_labels + negative_labels

    return patients, labels


image_processor = VivitImageProcessor.from_pretrained(
    config.VIVIT_MODEL_NAME,
)

convnext_image_processor = ConvNextImageProcessor.from_pretrained(
    config.CONVNEXT_MODEL_NAME,
)


def process_convnext_sequence(example: dict) -> dict:
    inputs = convnext_image_processor(
        np.array(example["video"]),
        return_tensors="pt",
    )
    inputs["labels"] = example["labels"]
    return inputs


def process_vivit_sequence(example: dict) -> dict:
    inputs = image_processor(list(np.array(example["video"])), return_tensors="pt")
    inputs["labels"] = example["labels"]
    return inputs


def create_vivit_dataset(
    directory: str,
    test_size: float,
    seed: int,
    save_dataset: bool,
    dataset_name: str,
) -> Dataset:
    if not save_dataset:
        Path(dataset_name).mkdir(parents=True, exist_ok=True)
        return load_from_disk(dataset_name).train_test_split(test_size=test_size)
    dictionary, _ = create_dataset_dictionary(directory)
    dataset = Dataset.from_list(dictionary)
    dataset = dataset.class_encode_column("labels")
    processed_dataset = dataset.map(process_vivit_sequence, batched=False)
    processed_dataset = processed_dataset.remove_columns(["video"])
    shuffled_dataset = processed_dataset.shuffle(seed=seed)
    shuffled_dataset = shuffled_dataset.map(
        lambda x: {"pixel_values": torch.tensor(x["pixel_values"]).squeeze()},
        batched=False,
    )
    shuffled_dataset.save_to_disk(dataset_name)
    return shuffled_dataset.train_test_split(test_size=test_size)


def create_convnext_dataset(
    directory: str,
    test_size: float,
    seed: int,
    save_dataset: bool,
    dataset_name: str,
) -> Dataset:
    if not save_dataset:
        Path(dataset_name).mkdir(parents=True, exist_ok=True)
        dataset = load_from_disk(dataset_name)
        return dataset.train_test_split(test_size=test_size)
    dictionary, _ = create_dataset_dictionary(directory)

    dataset = Dataset.from_list(dictionary)
    dataset = dataset.class_encode_column("labels")
    processed_dataset = dataset.map(process_convnext_sequence, batched=False)
    processed_dataset = processed_dataset.remove_columns(["video"])
    print(processed_dataset.column_names)
    shuffled_dataset = processed_dataset.shuffle(seed=seed)
    shuffled_dataset = shuffled_dataset.map(
        lambda x: {"pixel_values": torch.tensor(x["pixel_values"]).squeeze()},
        batched=False
    )
    shuffled_dataset.save_to_disk(dataset_name)
    return shuffled_dataset.train_test_split(test_size=test_size)
