import sys
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Iterator, Optional
import h5py

# Import pc_normalize from ModelNetDataLoader
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pointnet_project', 'Pointnet_Pointnet2_pytorch'))
from data_utils.ModelNetDataLoader import pc_normalize

# Import helper for train/val splitting
from .dataset_loader import _split_train_val


class ScanObjectNNDataset(Dataset):
    """Dataset for ScanObjectNN HDF5 files."""
    
    def __init__(self, h5_path: str, num_points: int = 1024, 
                 use_background: bool = True, normal_channel: bool = False):
        """
        Initialize ScanObjectNN dataset.
        
        Args:
            h5_path: Path to HDF5 file.
            num_points: Number of points to sample from each point cloud.
            use_background: If False, filter out background points using mask.
            normal_channel: Not used for ScanObjectNN (only XYZ data), kept for API compatibility.
        """
        self.h5_path = h5_path
        self.num_points = num_points
        self.use_background = use_background
        self.normal_channel = normal_channel
        
        # Load data from HDF5 file
        with h5py.File(h5_path, 'r') as f:
            # Data shape: (N, 2048, 3)
            self.points = f['data'][:].astype(np.float32)
            # Labels shape: (N,)
            self.labels = f['label'][:].astype(np.int64)
            # Mask shape: (N, 2048), 1 for foreground, 0 for background
            self.mask = f['mask'][:].astype(np.int32)
        
        self.num_samples = len(self.labels)
        
        # Warn if normal_channel requested but ScanObjectNN has only XYZ
        if normal_channel:
            import warnings
            warnings.warn(
                "ScanObjectNN dataset only contains XYZ coordinates. "
                "normal_channel=True will return points with shape (num_points, 3)."
            )
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int):
        # Get original points (2048, 3)
        points = self.points[idx]  # (2048, 3)
        label = self.labels[idx]
        mask = self.mask[idx]  # (2048,)
        
        if not self.use_background:
            # Filter to foreground points only (mask == 1)
            foreground_mask = (mask == 1)
            if not np.any(foreground_mask):
                # No foreground points (should not happen), fall back to all points
                foreground_mask = np.ones_like(mask, dtype=bool)
            
            points = points[foreground_mask]
        
        # Sample num_points
        if len(points) >= self.num_points:
            # Randomly sample num_points
            indices = np.random.choice(len(points), self.num_points, replace=False)
            points = points[indices]
        else:
            # Not enough points, repeat sampling
            indices = np.random.choice(len(points), self.num_points, replace=True)
            points = points[indices]
        
        # Normalize points (centroid to origin, scale to unit sphere)
        points = pc_normalize(points)
        
        # Convert to torch tensors
        points = torch.from_numpy(points).float()  # (num_points, 3)
        label = torch.tensor(label).long()
        
        return points, label


class ScanObjectNNLoader:
    """Data loader adapter for ScanObjectNN dataset."""
    
    def __init__(self, root_dir: str, split: str = "train", num_points: int = 1024,
                 normal_channel: bool = False, batch_size: int = 32, num_workers: int = 0,
                 use_background: bool = True, split_name: str = "main_split",
                 val_ratio: float = 0.2, seed: int = 42):
        """
        Initialize the ScanObjectNN data loader.
        
        Args:
            root_dir: Path to the dataset root directory (should contain scanobjectnn/h5_files/).
            split: Dataset split, either "train", "val", or "test".
            num_points: Number of points to sample from each point cloud.
            normal_channel: Whether to include normal vectors (ignored for ScanObjectNN).
            batch_size: Batch size for data loading.
            num_workers: Number of worker processes for data loading (default 0 for safety).
            use_background: If False, filter out background points using mask.
            split_name: Which split to use (e.g., "main_split", "split1", etc.).
            val_ratio: Proportion of training data to use for validation (0 <= val_ratio < 1).
            seed: Random seed for reproducible train/validation split.
        """
        if not 0 <= val_ratio < 1:
            raise ValueError(f"val_ratio must be between 0 (inclusive) and 1 (exclusive), got {val_ratio}")
        
        self.root_dir = root_dir
        self.split = split
        self.num_points = num_points
        self.normal_channel = normal_channel
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_background = use_background
        self.split_name = split_name
        self.val_ratio = val_ratio
        self.seed = seed
        
        # Build HDF5 file path for training data (used for train/val splits)
        if split == "train" or split == "val":
            h5_filename = "training_objectdataset_augmented25_norot.h5"
        elif split == "test":
            h5_filename = "test_objectdataset_augmented25_norot.h5"
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'.")
        
        h5_path = os.path.join(
            root_dir, "scanobjectnn", "h5_files", split_name, h5_filename
        )
        
        if not os.path.exists(h5_path):
            raise FileNotFoundError(
                f"ScanObjectNN HDF5 file not found: {h5_path}\n"
                f"Make sure root_dir points to the directory containing scanobjectnn/h5_files/"
            )
        
        # Create dataset from HDF5 file
        full_dataset = ScanObjectNNDataset(
            h5_path=h5_path,
            num_points=num_points,
            use_background=use_background,
            normal_channel=normal_channel
        )
        
        if split == "test":
            self.dataset = full_dataset
        else:
            self.dataset = _split_train_val(full_dataset, val_ratio, seed, split)
        
        # Create data loader
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers
        )
    
    def __iter__(self) -> Iterator:
        """Return an iterator over the data loader."""
        return iter(self.dataloader)
    
    def __len__(self) -> int:
        """Return the number of batches in the data loader."""
        # Calculate number of batches: ceil(len(dataset) / batch_size)
        # Using drop_last=False by default
        dataset_len = len(self.dataset)
        batch_size = self.batch_size
        return (dataset_len + batch_size - 1) // batch_size  # ceil division