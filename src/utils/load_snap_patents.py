import os

import gdown
import numpy as np
import scipy.io
import torch
from torch_geometric.data import Data


class DatasetLoader:
    def __init__(self, n_classes=5, root="dataset/"):
        """
        Initialize the DatasetLoader.

        Args:
            n_classes (int): Number of classes for quantile labels.
            root (str): Root directory for dataset storage.
        """
        self.n_classes = n_classes
        self.root = root
        self.dataset_drive_url = {"snap-patents": "1ldh23TSY1PwXia6dU0MYcpyEgX-w3Hia"}
        self.splits_drive_url = {"snap-patents": "12xbBRqd8mtG_XkNLH8dRRNZJvVM4Pw-N"}

    def load_snap_patents_mat(self):
        """
        Load the SNAP Patents dataset and return a list of Data objects.

        Returns:
            list: A list containing a single Data object.
        """
        self._build_dataset_folder()
        self._download_data()
        fulldata = scipy.io.loadmat(f"{self.root}snap_patents/snap_patents.mat")
        edge_index = torch.tensor(fulldata["edge_index"], dtype=torch.long)
        node_feat = torch.tensor(fulldata["node_feat"].todense(), dtype=torch.float)
        num_nodes = int(fulldata["num_nodes"])
        years = fulldata["years"].flatten()
        label = self.even_quantile_labels(years, verbose=False)
        label = torch.tensor(label, dtype=torch.long)

        self._download_splits()
        splits_lst = np.load(
            f"{self.root}snap_patents/snap-patents-splits.npy", allow_pickle=True
        )
        train_mask, val_mask, test_mask = self.process_fixed_splits(
            splits_lst, num_nodes
        )
        data = Data(
            x=node_feat,
            edge_index=edge_index,
            y=label,
            num_nodes=num_nodes,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            num_classes=self.n_classes,
        )

        return [data]

    def even_quantile_labels(self, vals, verbose=True):
        """
        Assign labels based on even quantile intervals.

        Args:
            vals (numpy.ndarray): Input values.
            verbose (bool): Whether to print class label intervals.

        Returns:
            numpy.ndarray: Assigned labels.
        """
        label = -1 * np.ones(vals.shape[0], dtype=np.int64)
        interval_lst = []
        lower = -np.inf
        for c in range(self.n_classes - 1):
            upper = np.nanquantile(vals, (c + 1) / self.n_classes)
            interval_lst.append((lower, upper))
            inds = (vals >= lower) * (vals < upper)
            label[inds] = c
            lower = upper
        label[vals >= lower] = self.n_classes - 1
        interval_lst.append((lower, np.inf))
        if verbose:
            print("Class Label Intervals:")
            for class_idx, interval in enumerate(interval_lst):
                print(f"Class {class_idx}: [{interval[0]}, {interval[1]})]")
        return label

    def process_fixed_splits(self, splits_lst, num_nodes):
        """
        Process fixed splits and return masks for training, validation, and testing.

        Args:
            splits_lst (list): List of splits.
            num_nodes (int): Number of nodes in the dataset.

        Returns:
            tuple: Tuple containing train, validation, and test masks.
        """
        n_splits = len(splits_lst)
        train_mask = torch.zeros(num_nodes, n_splits, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, n_splits, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, n_splits, dtype=torch.bool)
        for i in range(n_splits):
            train_mask[splits_lst[i]["train"], i] = 1
            val_mask[splits_lst[i]["valid"], i] = 1
            test_mask[splits_lst[i]["test"], i] = 1
        return train_mask, val_mask, test_mask

    def _build_dataset_folder(self):
        """
        Build the dataset folder if it doesn't exist.
        """
        if not os.path.exists(f"{self.root}snap_patents"):
            os.mkdir(f"{self.root}snap_patents")

    def _download_data(self):
        """
        Download the dataset if it doesn't exist.
        """
        if not os.path.exists(f"{self.root}snap_patents/snap_patents.mat"):
            gdown.download(
                id=self.dataset_drive_url["snap-patents"],
                output=f"{self.root}snap_patents/snap_patents.mat",
                quiet=False,
            )

    def _download_splits(self):
        """
        Download splits if they don't exist.
        """
        name = "snap-patents"
        if not os.path.exists(f"{self.root}snap_patents/{name}-splits.npy"):
            assert name in self.splits_drive_url.keys()
            gdown.download(
                id=self.splits_drive_url[name],
                output=f"{self.root}snap_patents/{name}-splits.npy",
                quiet=False,
            )


# Example Usage:
loader = DatasetLoader()
dataset = loader.load_snap_patents_mat()
