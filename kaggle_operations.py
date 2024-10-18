import os
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



def download_output(api, kernel_owner, kernel_slug, output_file, download_path='.'):
    """
    Download a specific output file from a Kaggle kernel.

    Args:
        api (KaggleApi): Authenticated Kaggle API instance.
        kernel_owner (str): Kernel owner's username.
        kernel_slug (str): Kernel slug.
        output_file (str): Name of the output file to download.
        download_path (str, optional): Download destination. Defaults to current directory.

    Returns:
        str: Full path of the downloaded file, or None if download fails.
    """
    full_kernel_slug = f'{kernel_owner}/{kernel_slug}'

    try:
        api.kernel_output_download(
            kernel_owner_slug=full_kernel_slug,
            path=download_path,
            file_name=output_file
        )

        full_file_path = os.path.join(download_path, output_file)
        print(f'Successfully downloaded: {full_file_path}')
        return full_file_path

    except Exception as e:
        print(str(e).strip())
        return None
