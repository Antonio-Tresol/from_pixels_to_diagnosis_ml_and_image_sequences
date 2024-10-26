import os


def count_files(root_dir: str) -> dict:
    """Count the files in each folder within the specified root directory.

    Args:
    ----
        root_dir (str): The root directory to scan.

    Returns:
    -------
        dict: A dictionary where keys are folder names and values are the number
        of files in each folder.

    """
    folder_counts = {}
    for folder in os.scandir(root_dir):
        if folder.is_dir():
            folder_counts[folder.name] = sum(
                1 for _ in os.scandir(folder) if _.is_file()
            )

    # Sort folders based on file counts
    return sorted(folder_counts, key=folder_counts.get)

