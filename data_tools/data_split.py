# Utility class for splitting data into train and test sets
from data_tools.index_manager import IndexManager
from torch.utils.data import Dataset


class DataSplitter:
    @staticmethod
    def split_data(
        dataset: Dataset,
        indices_path: str,
        train_size: float,
        test_size: float,
        use_index: bool,
    ) -> tuple:
        """Split data into train and test indices.

        Args:
        ----
            dataset (Dataset): Dataset instance.
            indices_path (str): Path to the file where indices will be saved.
            train_size (float): fraction of the data use for training
            test_size (float): Fraction of the data to reserve as test set.
            use_index (bool): Flag indicating whether to use existing indices.

        Returns:
        -------
            tuple: Tuple containing train and test indices.

        """
        if use_index:
            try:
                indices = IndexManager.load_indices(indices_path)
            except:
                indices = IndexManager.create_indices(
                    dataset,
                    indices_path,
                    train_size,
                    test_size,
                )
            return indices
        # else
        return IndexManager.create_indices(
            dataset,
            indices_path,
            train_size,
            test_size,
        )
