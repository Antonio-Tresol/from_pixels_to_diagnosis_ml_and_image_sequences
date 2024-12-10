# ========= ConvNext specific functions =========
from pathlib import Path

import numpy as np
import torch
from transformers import ConvNextImageProcessor
from datasets import Dataset, load_from_disk
from sklearn.model_selection import train_test_split
import config
import image_processing


def generate_patient_split(
    directory: str,
    test_size: float,
    seed: int,
) -> dict[str, dict]:
    negative_patients = image_processing.get_all_patients(
        Path(f"{directory}/Negatives/"),
    )
    positive_patients = image_processing.get_all_patients(
        Path(f"{directory}/Positives/"),
    )
    negative_labels = [0] * len(negative_patients)
    positive_labels = [1] * len(positive_patients)

    patients = positive_patients + negative_patients
    labels = positive_labels + negative_labels
    patients_train, patients_test, labels_train, labels_test = train_test_split(
        patients,
        labels,
        test_size=test_size,
        random_state=seed,
    )
    return {
        "train": {"patient_id": patients_train, "labels": labels_train},
        "test": {"patient_id": patients_test, "labels": labels_test},
    }


def get_patients_from_dataset(directory: str) -> tuple:
    negative_patients = image_processing.get_all_patients(
        Path(f"{directory}/Negatives/"),
    )
    positive_patients = image_processing.get_all_patients(
        Path(f"{directory}/Positives/"),
    )
    negative_labels = [0] * len(negative_patients)
    positive_labels = [1] * len(positive_patients)

    patients = positive_patients + negative_patients
    labels = positive_labels + negative_labels
    return patients, labels


convnext_image_processor = ConvNextImageProcessor.from_pretrained(
    config.CONVNEXT_MODEL_NAME,
)


def process_convnext_sequence(example: dict) -> dict:
    inputs = convnext_image_processor(
        np.array(example["image"]),
        return_tensors="pt",
    )
    inputs["labels"] = example["labels"]
    return inputs


def get_slice_label(patient_id: str, index: int) -> int:
    patient_id = int(patient_id)
    info = config.DATASET_INFO

    mask = (info["PatientNumber"] == patient_id) & (info["SliceNumber"] == index)
    result = info[mask]

    if result.empty:
        msg = f"No data found for patient {patient_id}, slice {index}"
        raise ValueError(msg)

    return int(result.iloc[0]["positive_image"])


def create_dataset_dictionary_for_convnext(directory: str) -> list:
    class_labels = ["Negative", "Positive"]
    all_images = []
    patients, labels = get_patients_from_dataset(directory)

    for patient_id, patient_label in zip(patients, labels):
        image_sequence = image_processing.retrieve_all_patient_images(
            directory,
            patient_id,
            class_labels[patient_label],
        )
        all_images.extend(
            {
                "image": image[0],
                "labels": get_slice_label(patient_id, image[1]),
                "id": patient_id,
            }
            for image in image_sequence
        )
    return all_images


def has_required_columns(dataset: Dataset) -> bool:
    required_columns = ("pixel_values", "labels", "id")
    return all(col in dataset.column_names for col in required_columns)


def create_convnext_dataset(
    directory: str,
    test_size: float,
    seed: int,
    dataset_name: str,
) -> Dataset:
    """Create a dataset for ConvNext model training and evaluation"""
    dataset_path = Path(dataset_name)
    if dataset_path.exists():
        try:
            dataset = load_from_disk(dataset_name)
            if has_required_columns(dataset):
                split = generate_patient_split(
                    directory,
                    test_size,
                    seed,
                )
                final_split = {
                    "train": dataset.filter(
                        lambda x: x["id"] in split["train"]["patient_id"],
                    ),
                    "test": dataset.filter(
                        lambda x: x["id"] in split["test"]["patient_id"],
                    ),
                }
                return final_split, split
            print(
                "Warning: Loaded dataset missing required columns."
                " Creating new dataset.",
            )
        except (ValueError, OSError) as e:
            print(f"Error loading dataset: {e}. Creating new dataset.")

    dictionary = create_dataset_dictionary_for_convnext(directory)

    dataset = Dataset.from_list(dictionary)
    dataset = dataset.class_encode_column("labels")

    processed_dataset = dataset.map(
        process_convnext_sequence,
        remove_columns=["image"],
    )

    final_dataset = processed_dataset.map(
        lambda x: {"pixel_values": torch.tensor(x["pixel_values"]).squeeze()},
    )
    split = generate_patient_split(
        directory,
        test_size,
        seed,
    )
    Path(dataset_name).mkdir(parents=True, exist_ok=True)
    final_dataset.save_to_disk(dataset_name)

    # Use the split made at the patient level
    final_split = {
        "train": final_dataset.filter(
            lambda x: x["id"] in split["train"]["patient_id"],
        ),
        "test": final_dataset.filter(
            lambda x: x["id"] in split["test"]["patient_id"],
        ),
    }
    return final_split, split
