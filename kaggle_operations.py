import os
from kaggle.api.kaggle_api_extended import KaggleApi


def download_kaggle_dataset(dataset, path=os.getcwd(), unzip=True):
    """
    Download a Kaggle dataset using Kaggle API.

    This function checks for required environment variables, authenticates with
    Kaggle API, and downloads specified dataset to given directory.

    Args:
        dataset (str): Name of dataset to download (username/dataset_name).
        path (str): Directory where dataset will be downloaded to.
        unzip (bool, optional): Whether to unzip downloaded files. Defaults to True.

    Raises:
        EnvironmentError: If any required environment variables are missing.

    Returns:
        None
    """
    # Check for required environment variable
    required_env_vars = ['KAGGLE_USERNAME', 'KAGGLE_KEY']
    missing_env_vars = [var for var in required_env_vars if os.getenv(var) is None]

    # Print all missing environment variables
    if missing_env_vars:
        raise EnvironmentError(f"Missing environment variables: {', '.join(missing_env_vars)}")

    # Initialize Kaggle API
    KaggleApi().authenticate()

    KaggleApi().dataset_download_files(
        dataset=dataset,
        path=path,
        unzip=unzip,
    )
    print(f"Successfully downloaded dataset: {dataset}")
