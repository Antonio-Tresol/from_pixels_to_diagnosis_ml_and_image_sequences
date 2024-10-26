import pickle
from pathlib import Path
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


# Utility class for managing indices
class IndexManager:
    @staticmethod
    def save_indices(indices: list[int], indices_path: str) -> None:
        """Save indices to a file.

        Args:
        ----
            indices (tuple): Tuple containing train and test indices.
            indices_path (str): Path to the file where indices will be saved.

        """
        with Path(indices_path).open("wb") as file:
            pickle.dump(indices, file)

    @staticmethod
    def load_indices(indices_path: str) -> tuple:
        """Load indices from a file.

        Args:
        ----
            indices_path (str): Path to the file containing saved indices.

        Returns:
        -------
            tuple: Tuple containing train and test indices.

        """
        with Path(indices_path).open("rb") as file:
            return pickle.load(file)

    @staticmethod
    def create_indices(
        dataset: Dataset,
        indices_path: str,
        train_size: float,
        test_size: float,
    ) -> tuple:
        """Create indices for training and testing.

        Args:
        ----
            dataset (Dataset): Dataset instance.
            indices_path (str): Path to the file where indices will be saved.
            train_size (float): Fraction of the data to reserve as training set.
            test_size (float): Fraction of the data to reserve as test set.

        Returns:
        -------
            tuple: Tuple containing train and test indices.

        """
        indices = train_test_split(
            range(len(dataset)),
            train_size=train_size,
            test_size=test_size,
            stratify=dataset.labels,
        )
        IndexManager.save_indices(indices, indices_path)
        return indices
