# Torch
from torch.utils.data import Dataset
import torch
from typing import Optional

# Data tools
from data_tools.sampling import sample_patient_images
from data_tools.image_sequence_tools import (
    get_all_patients,
    retrieve_image_sequence,
)

# Utils
from pathlib import Path


def get_patients_from_dataset(directory: str) -> list:
    # get all the names of patients dir inside positives and negatives
    negative_patients = get_all_patients(Path(f"{directory}/Negatives/"))
    positive_patients = get_all_patients(Path(f"{directory}/Positives/"))
    # labels
    negative_labels = [0] * len(negative_patients)
    positive_labels = [1] * len(positive_patients)

    patients = positive_patients + negative_patients
    labels = positive_labels + negative_labels

    return patients, labels


class ImageSequenceListBuilder:
    @staticmethod
    def build_list(root_dir: str) -> tuple[list[str], list[int], list[int]]:
        """Build lists of image paths, corresponding labels, and class counts.

        Args:
        ----
            root_dir (str): The root directory containing image folders.
            folders (List[str]): List of folder names containing images.

        Returns:
        -------
            Tuple[List[str], List[int], List[int]]: A tuple containing lists of
            image sequences paths, labels, and class counts.

        """
        patients, labels = get_patients_from_dataset(Path(root_dir))
        class_counts = [labels.count(0), labels.count(1)]
        classes = ["Negative", "Positive"]
        return patients, labels, class_counts, classes


class ImageSequenceClassificationDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        transform: Optional[torch.nn.Module] = None,
    ) -> None:
        """Initialize the ImageFolderDataset.

        Args:
        ----
            root_dir (str): The root directory containing image folders.
            transform (optional): An optional transform to be applied to the images.

        """
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform

        # Build lists of images, labels, and class counts
        self.image_sequences, self.labels, self.class_counts, self.classes = (
            ImageSequenceListBuilder.build_list(root_dir=self.root_dir)
        )

    def __len__(self) -> int:
        """Get the total number of images in the dataset.

        Returns
        -------
            int: Total number of images.

        """
        return len(self.image_sequences)

    def __getitem__(self, idx: int) -> tuple:
        """Get an image sequence and its corresponding label at the specified index.

        Args:
        ----
            idx (int): Index of the image sequence to retrieve.

        Returns:
        -------
            tuple: A tuple containing the image sequence and its label.

        """
        image_sequence = retrieve_image_sequence(
            self.root_dir,
            self.image_sequences[idx],
            self.classes[self.labels[idx]],
        )
        label = self.labels[idx]
        if self.transform:
            image_sequence = self.transform(image_sequence)
        print(
            (
                f"patient {self.image_sequences[idx]} is "
                f"{self.classes[self.labels[idx]]} image sequence shape is "
                f"{image_sequence.shape}"
            ),
        )
        return image_sequence, label

    def __str__(self) -> str:
        """Return a string representation of the dataset.

        Returns
        -------
            str: A string representation of the dataset.

        """
        return (
            f"-> ImageClassificationFolderDataset\n- root_dir={self.root_dir},\n"
            f"-- num_patients={len(self.image_sequences)}, \n"
            f"-- class_counts={self.class_counts}, \n"
            f"-- classes={self.classes}),\n"
            f"-- patients={self.image_sequences}"
        )


def testing() -> None:
    from pl_vivit import get_vivit_transformations

    a = ImageSequenceClassificationDataset(
        "dataset",
        transform=get_vivit_transformations(),
    )
    print(a)

    a[0]

    directory = "dataset"
    image_num = 18
    patient_dir = "095"
    label = "Negative"
    indices = sample_patient_images(
        directory,
        image_num,
        patient_dir,
        label,
        strategy="random",
    )
    print(indices)


if __name__ == "__main__":
    testing()
