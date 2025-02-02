import os
import torch
import torch_geometric
from torch_geometric.data import Data, Dataset
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch_geometric.nn as pyg_nn
from tqdm import tqdm


# Preprocessing class
class SRFE200KPreprocessor:
    def __init__(self, datadir):
        self.datadir = datadir
        self.target_mean = -0.7554450631141663  # Target mean
        self.target_std = 2.887317180633545  # Target standard deviation
        self.grad_target_mean = 0.0  # Force labels mean
        self.grad_target_std = 2.887317180633545  # Force labels standard deviation
        self.noise_std = 0.1  # Noise standard deviation
        self.max_radius = 12.0  # Max radius for graph construction
        self.max_neighbors = 20  # Max number of neighbors
        self.dataset_folder = os.path.join(datadir, 'SRFE-200K')  # Dataset folder path

    def preprocess(self, sample_limit=20000):
        """
        Preprocess the SRFE-200K dataset with a sample limit.
        """
        processed_data = []

        # Get data files from the SRFE-200K folder, limited to the sample_limit
        data_files = os.listdir(self.dataset_folder)[:sample_limit]  # Limit the number of samples to 20k
        for data_file in tqdm(data_files, desc="Processing dataset"):
            data = self.load_data(data_file)
            data = self._normalize_labels(data)
            data = self._add_noise(data)
            data = self._build_graph(data)
            processed_data.append(data)

        return processed_data

    def load_data(self, data_file):
        """
        Load each data sample, assuming each data file contains atomic positions, energy, and force labels.
        """
        # Actual file reading logic depends on the dataset format (e.g., JSON, CSV)
        pos = torch.randn((10, 3))  # Each sample contains positions of 10 atoms
        grad_y = torch.randn((10, 3))  # Force labels for 10 atoms
        y = torch.randn(1)  # Energy label

        # Create a PyG data object
        data = Data(pos=pos, y=y, grad_y=grad_y)
        return data

    def _normalize_labels(self, data):
        """
        Normalize the labels.
        """
        # Normalize target labels (energy)
        scaler = StandardScaler()
        data.y = scaler.fit_transform(data.y.reshape(-1, 1)).reshape(-1)

        # If force labels exist, normalize them as well
        if hasattr(data, 'grad_y'):
            data.grad_y = scaler.fit_transform(data.grad_y.reshape(-1, 1)).reshape(-1)

        return data

    def _add_noise(self, data):
        """
        Add noise to atomic positions.
        """
        noise = torch.randn_like(data.pos) * self.noise_std
        data.pos = data.pos + noise
        return data

    def _build_graph(self, data):
        """
        Build molecular graph edges using a radius-based graph construction.
        """
        edge_index = pyg_nn.radius_graph(data.pos, r=self.max_radius, max_num_neighbors=self.max_neighbors)
        data.edge_index = edge_index
        return data


class SRFE200KDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(SRFE200KDataset, self).__init__(root, transform, pre_transform)
        self.preprocessor = SRFE200KPreprocessor(root)

    def len(self):
        # Return the number of processed samples
        return 20000

    def get(self, idx):
        # Retrieve the preprocessed data sample
        data_file = f"sample_{idx}.json"  # Sample file naming convention
        data = self.preprocessor.load_data(data_file)
        return data


if __name__ == "__main__":
    # Dataset directory
    root = './'  # Local dataset directory
    dataset = SRFE200KDataset(root=root)

    # Data preprocessing with a limit of 20k samples
    processed_data = dataset.preprocessor.preprocess(sample_limit=20000)

    # Save or continue with training
    # You can save the processed data as needed
    print(f"Number of processed samples: {len(processed_data)}")
