import os
import sys
import subprocess

import socket
from IPython import get_ipython


def check_platform():
    """
    Determine the current execution platform.

    Returns:
    str: The name of the platform ('Google Colab', 'Kaggle', or 'Unknown')
    """
    # Check if running in Google Colab
    if 'google.colab' in sys.modules:
        return 'Google Colab'
    # Check if running in Kaggle
    elif 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
        return 'Kaggle'
    # If neither Colab nor Kaggle, return 'Unknown'
    else:
        return 'Unknown'


def is_jupyter_on_localhost():
    # Check if we're running in a Jupyter environment
    try:
        ipython = get_ipython()
        if 'IPKernelApp' not in ipython.config:
            return False  # We're not in a Jupyter environment
    except:
        return False  # get_ipython() is not available

    # Get the IP address Jupyter is running on
    jupyter_ip = ipython.config.get('ServerApp', {}).get('ip', '')

    # Check if the IP is localhost or an empty string (which defaults to localhost)
    if jupyter_ip in ['', 'localhost', '127.0.0.1']:
        return True

    # Check if the IP is a local IP address
    try:
        return ipaddress.ip_address(jupyter_ip).is_private
    except:
        return False


def pip_install(requirements=None, return_installed=False):
    """
    Install Python packages using pip.

    Args:
        requirements (list): List of package names to install.
        return_installed (bool): If True, return list of successfully installed packages.

    Returns:
        list or None: List of installed packages if return_installed is True, else None.
    """
    # Check if requirements list is empty or None
    if not requirements:
        print("Error: No requirements provided. Aborting installation.")
        return None

    filename = 'requirements.txt'
    # Prepare the base command for pip install
    command = [sys.executable, '-m', 'pip', 'install', '--quiet']
    installed_packages = []

    print("Starting package installation...")
    for r in requirements:
        try:
            # Attempt to install each package
            print(f'Installing {r}...', end=' ', flush=True)
            result = subprocess.run(command + [r], check=True, capture_output=True, text=True)
            print('Success')
            # If successful, add to the list of installed packages
            installed_packages.append(r)
        except subprocess.CalledProcessError as e:
            # Handle pip installation errors
            print('Failed')
            print(e.stderr.strip())
            continue
        except Exception as e:
            # Handle any unexpected errors
            print('Failed')
            print('Unexpected error:')
            print(str(e))

    # Write successfully installed packages to file
    if installed_packages:
        with open(filename, 'w') as file:
            for p in installed_packages:
                file.write(f'{p}\n')
        print(f'List of {len(installed_packages)} installed packages written to: {filename}')
    else:
        print('No packages were successfully installed.')

    # Return installed packages list if requested
    if return_installed:
        return installed_packages
    return None
