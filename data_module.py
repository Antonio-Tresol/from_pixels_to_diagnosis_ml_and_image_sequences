# Imports

# Utils
import numpy as np
from typing import Optional, Any
from pathlib import Path

# Torch
from torch.utils.data import Dataset, DataLoader, Subset
from pytorch_lightning import LightningDataModule

# Customs
from datasets import ImageSequenceClassificationDataset
from data_tools.sampling import Sampling, SamplerFactory
from data_tools.data_split import DataSplitter


class DataLoaderCreator:
    @staticmethod
    def create_dataloader(
        dataset: Dataset,
        batch_size: int,
        sampler: Any = None,
        shuffle: bool = False,
        num_workers: int = 1,
    ) -> DataLoader:
        """Create a DataLoader for a dataset.

        Args:
        ----
            dataset (Dataset): Dataset instance.
            batch_size (int): number of data instances to use per batch
            sampler (optional): Sampler used for sampling data. Default is None.
            shuffle (bool, optional): Flag indicating whether to shuffle the data.
              Default is False.
            num_workers (int, optional): Number of subprocesses to use for data loading.
              Default is 1.

        Returns:
        -------
            DataLoader: DataLoader instance.

        """
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True,
        )


class ImageSequencesDataModule(LightningDataModule):
    def __init__(
        self,
        dataset: str,
        root_dir: str,
        batch_size: int,
        train_folder_dataset: ImageSequenceClassificationDataset,
        test_folder_dataset: ImageSequenceClassificationDataset,
        train_size: float = 0.5,
        test_size: float = 0.5,
        use_index: bool = True,
        indices_dir: Optional[str] = None,
        preset_indices: Optional[list[int]] = None,
        sampling: Sampling = Sampling.NONE,
    ) -> None:
        """Initialize the ImageDataModule.

        Args:
        ----
            dataset (str): Name of the dataset.
            root_dir (str): Root directory of the dataset.
            batch_size (int): Batch size for data loaders.
            train_folder_dataset (Dataset): Dataset class to use for training.
            test_folder_dataset (Dataset): Dataset class to use for testing.
            train_size (float, optional): Fraction of data to use as training set.
              Default is 0.5.
            test_size (float, optional): Fraction of data to use as test set.
              Default is 0.5.
            use_index (bool, optional): Whether to use existing indices.
              Default is True.
            indices_dir (str, optional): Directory to save indices. Default is None.
            preset_indices (list, optional):
              List of indices to use for splitting data.
              Default is None.
            sampling (Sampling, optional): Sampling strategy.
              Default is Sampling.NONE.

        """
        super().__init__()
        self.save_hyperparameters()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.train_size = train_size
        self.test_size = test_size
        self.use_index = use_index
        self.sampling = sampling

        # Initialize training and test folders
        self.train_folder = train_folder_dataset
        self.test_folder = test_folder_dataset

        self.class_counts = self.train_folder.class_counts
        self.classes = self.train_folder.classes
        self.indices_path = Path(indices_dir) / f"{dataset}.pkl"
        self.preset_indices = preset_indices

    def prepare_data(self) -> None:
        """Prepare data for training and testing."""
        # Split train and test indices
        self.train_indices, self.test_indices = DataSplitter.split_data(
            self.train_folder,
            self.indices_path,
            self.train_size,
            self.test_size,
            self.use_index,
            self.preset_indices,
        )
        # Split the datasets
        self.train_dataset = Subset(self.train_folder, self.train_indices)
        self.test_dataset = Subset(self.test_folder, self.test_indices)
        train_labels = np.array(self.train_folder.labels)[self.train_indices]
        # Create a sampler (if needed)
        self.train_sampler = SamplerFactory.create_sampler(
            self.sampling,
            self.train_dataset,
            train_labels,
        )

    def create_data_loaders(self) -> None:
        """Create data loaders for training and testing."""
        # Shuffle flag
        shuffle = self.sampling == Sampling.NONE
        # Create data loaders
        self.train_loader = DataLoaderCreator.create_dataloader(
            self.train_dataset,
            self.batch_size,
            self.train_sampler,
            shuffle=shuffle,
            num_workers=8,
        )
        self.test_loader = DataLoaderCreator.create_dataloader(
            self.test_dataset,
            self.batch_size,
            num_workers=8,
        )

    def train_dataloader(self) -> None:
        """Get the training data loader.

        Returns
        -------
            DataLoader: Training data loader.

        """
        return self.train_loader

    def val_dataloader(self) -> DataLoader:
        """Get the validation data loader (same as test data loader).

        Returns
        -------
            DataLoader: Validation data loader.

        """
        return self.test_loader

    def test_dataloader(self) -> DataLoader:
        """Get the test data loader.

        Returns
        -------
            DataLoader: Test data loader.

        """
        return self.test_loader
