# ========= ConvNext specific functions =========
from pathlib import Path

import numpy as np
import torch
from transformers import ConvNextImageProcessor
from datasets import Dataset, load_from_disk
from sklearn.model_selection import train_test_split
import config
import image_processing


def get_patients_from_dataset_and_split(
    directory: str,
    test_size: float,
    seed: int,
) -> dict[str, tuple[list, list]]:
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
        "train": (patients_train, labels_train),
        "test": (patients_test, labels_test),
    }


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


def create_dataset_dictionary_for_convnext(
    directory: str,
    test_size: float,
    seed: int,
) -> tuple[list, dict[str, tuple[list, list]], list[str]]:
    class_labels = ["Negative", "Positive"]
    all_images = []
    split = get_patients_from_dataset_and_split(directory, test_size, seed)

    for subset in ["train", "test"]:
        patients, labels = split[subset]
        for patient_dir, patient_label in zip(patients, labels):
            image_sequence = image_processing.retrieve_image_sequence(
                directory,
                patient_dir,
                class_labels[patient_label],
            )
            all_images.extend(
                {
                    "video": image,
                    "labels": patient_label,
                    "id": patient_dir,
                }
                for image in image_sequence
            )
    return all_images, split, class_labels


def has_required_columns(dataset: Dataset) -> bool:
    return all(col in dataset.column_names for col in ("pixel_values", "labels", "id"))


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
            dataset = dataset.shuffle(seed=seed)
            if has_required_columns(dataset):
                split = get_patients_from_dataset_and_split(
                    directory,
                    test_size,
                    seed,
                )
                return {
                    "train": dataset.filter(lambda x: x["id"] in split["train"][0]),
                    "test": dataset.filter(lambda x: x["id"] in split["test"][0]),
                }, split
            print(
                "Warning: Loaded dataset missing required columns."
                " Creating new dataset.",
            )
        except (ValueError, OSError) as e:
            print(f"Error loading dataset: {e}. Creating new dataset.")

    dictionary, split, class_labels = create_dataset_dictionary_for_convnext(
        directory,
        test_size=test_size,
        seed=seed,
    )

    dataset = Dataset.from_list(dictionary)
    dataset = dataset.class_encode_column("labels")

    processed_dataset = dataset.map(
        process_convnext_sequence,
        remove_columns=["video"],
    )

    shuffled_dataset = processed_dataset.shuffle(seed=seed)
    final_dataset = shuffled_dataset.map(
        lambda x: {"pixel_values": torch.tensor(x["pixel_values"]).squeeze()},
    )

    Path(dataset_name).mkdir(parents=True, exist_ok=True)
    final_dataset.save_to_disk(dataset_name)

    # Use the split made at the patient level
    return {
        "train": final_dataset.filter(lambda x: x["id"] in split["train"][0]),
        "test": final_dataset.filter(lambda x: x["id"] in split["test"][0]),
    }, split
