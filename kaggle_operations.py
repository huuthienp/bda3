import os
import subprocess
# from kaggle.api.kaggle_api_extended import KaggleApi


def download_dataset(api, dataset, path=os.getcwd(), unzip=True):
    """
    Download a Kaggle dataset using Kaggle API.

    This function checks for required environment variables, uses the provided
    KaggleApi object, and downloads specified dataset to given directory.

    Args:
        api (KaggleApi): An authenticated KaggleApi object.
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

    # Use the provided KaggleApi object
    api.dataset_download_files(
        dataset=dataset,
        path=path,
        unzip=unzip,
    )
    print(f"Successfully downloaded dataset: {dataset}")


def download_output(kernel_owner, kernel_slug, output_file, output_path='.'):
    """
    Download a specific output file from a Kaggle kernel using the Kaggle CLI.

    Args:
        kernel_owner (str): The username of the kernel owner.
        kernel_slug (str): The slug of the kernel.
        output_file (str): The name of the file to download.
        output_path (str): The path where the file will be saved.

    Returns:
        bool: True if download is successful, False otherwise.

    Raises:
        subprocess.CalledProcessError: If the Kaggle CLI command fails.
        FileNotFoundError: If the Kaggle CLI is not installed or not in PATH.
        Exception: For any other unexpected errors.

    Note:
        Requires Kaggle CLI to be installed and configured with valid credentials.
    """
    command = [
        'kaggle', 'kernels', 'output',
        f'{kernel_owner}/{kernel_slug}',
        '-p', output_path,
        '-f', output_file
    ]

    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(f'Successfully downloaded {output_file} to {output_path}')
        return True
    except subprocess.CalledProcessError as e:
        print(e.stderr.strip())
        return False
    except FileNotFoundError:
        print('Error: Kaggle CLI not found. Make sure it is installed and in your PATH.')
        return False
    except Exception as e:
        print('Unexpected error:')
        print(str(e).strip())
        return False
