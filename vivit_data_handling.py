import numpy as np
from pathlib import Path
import torch
from transformers import VivitImageProcessor
from datasets import Dataset, load_from_disk
import config
import image_processing


# ========= Vivit specific functions =========
def get_patients_from_dataset(directory: str) -> list:
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


def create_dataset_dictionary_for_vivit(
    directory: str,
) -> tuple[list, list]:
    class_labels = ["Negative", "Positive"]
    all_image_sequences = []
    patients, labels = get_patients_from_dataset(
        directory=directory,
    )
    for patient_dir, patient_label in zip(patients, labels):
        image_sequence = image_processing.retrieve_image_sequence(
            directory,
            patient_dir,
            class_labels[patient_label],
        )
        all_image_sequences.append(
            {
                "video": image_sequence,
                "labels": class_labels[patient_label],
            },
        )
    return all_image_sequences, class_labels


image_processor = VivitImageProcessor.from_pretrained(
    config.VIVIT_MODEL_NAME,
)


def process_vivit_sequence(example: dict) -> dict:
    inputs = image_processor(list(np.array(example["video"])), return_tensors="pt")
    inputs["labels"] = example["labels"]
    return inputs


def create_vivit_dataset(
    directory: str,
    test_size: float,
    seed: int,
    dataset_name: str,
) -> Dataset:
    dataset_path = Path(dataset_name)
    if dataset_path.exists():
        try:
            return (
                load_from_disk(dataset_name)
                .shuffle(seed=seed)
                .train_test_split(test_size=test_size)
            )
        except Exception as e:
            print(f"Error loading dataset: {e}. Creating new dataset.")

    dictionary, _ = create_dataset_dictionary_for_vivit(directory)
    dataset = Dataset.from_list(dictionary)
    dataset = dataset.class_encode_column("labels")

    processed_dataset = dataset.map(process_vivit_sequence)

    processed_dataset = processed_dataset.remove_columns(["video"])
    shuffled_dataset = processed_dataset.shuffle(seed=seed)

    shuffled_dataset = shuffled_dataset.map(
        lambda x: {"pixel_values": torch.tensor(x["pixel_values"]).squeeze()},
    )
    shuffled_dataset.save_to_disk(dataset_name)
    return shuffled_dataset.train_test_split(test_size=test_size)


