import os
import urllib.request
import zipfile

# SRFE-200K Dataset URL
url = 'https://dl.fbaipublicfiles.com/opencatalystproject/data/srfe_200k.tar.gz'  # 请替换为实际的下载链接

# Function to download and extract SRFE-200K dataset
def download_srfe200k(datadir):
    """
    Download and extract SRFE-200K dataset.
    """
    save_path = os.path.join(datadir, 'srfe_200k.tar.gz')

    # Download the dataset
    if not os.path.exists(save_path):
        print(f"Downloading SRFE-200K dataset from {url}...")
        urllib.request.urlretrieve(url, save_path)
        print(f"Downloaded {save_path}")
    else:
        print(f"SRFE-200K dataset already exists at {save_path}")

    # Extract the dataset
    dataset_folder = os.path.join(datadir, 'SRFE-200K')
    if not os.path.exists(dataset_folder):
        print(f"Extracting SRFE-200K dataset to {dataset_folder}...")
        with zipfile.ZipFile(save_path, 'r') as zip_ref:
            zip_ref.extractall(dataset_folder)
        print(f"Extraction completed to {dataset_folder}")
    else:
        print(f"SRFE-200K dataset already extracted to {dataset_folder}")

# Example usage
download_srfe200k(datadir='./')
