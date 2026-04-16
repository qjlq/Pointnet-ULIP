import sys
import os
import torch
from torch.utils.data import DataLoader
from typing import Iterator, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pointnet_project', 'Pointnet_Pointnet2_pytorch'))
from data_utils.ModelNetDataLoader import ModelNetDataLoader as OriginalModelNetDataLoader


def _split_train_val(full_dataset, val_ratio: float, seed: int, split: str):
    """
    Split a full dataset into train and validation portions.
    
    Args:
        full_dataset: The full dataset to split
        val_ratio: Proportion to use for validation (0 <= val_ratio < 1)
        seed: Random seed for reproducibility
        split: Which split to return, either "train" or "val"
        
    Returns:
        Dataset for the requested split
        
    Raises:
        ValueError: If split is not "train" or "val", or val_ratio is invalid
    """
    if not 0 <= val_ratio < 1:
        raise ValueError(f"val_ratio must be between 0 (inclusive) and 1 (exclusive), got {val_ratio}")
    
    if split not in ("train", "val"):
        raise ValueError(f"split must be 'train' or 'val', got {split}")
    
    if val_ratio == 0 and split == "val":
        raise ValueError("Cannot create validation split with val_ratio=0")
    
    if val_ratio == 0:
        # No validation split, return full dataset for train
        return full_dataset
    
    val_size = int(val_ratio * len(full_dataset))
    train_size = len(full_dataset) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size], generator=generator
    )
    
    return train_dataset if split == "train" else val_dataset


class ModelNet40Loader:
    """Data loader adapter for ModelNet40 dataset using PointNet's original loader."""
    def __init__(self, root_dir: str, split: str = "train", num_points: int = 1024, 
                 normal_channel: bool = False, batch_size: int = 32, num_workers: int = 0,
                 val_ratio: float = 0.2, seed: int = 42):
        """
        Initialize the ModelNet40 data loader.
        
        Args:
            root_dir: Path to the dataset root directory.
            split: Dataset split, either "train", "val", or "test".
            num_points: Number of points to sample from each point cloud.
            normal_channel: Whether to include normal vectors.
            batch_size: Batch size for data loading.
            num_workers: Number of worker processes for data loading (default 0 for safety).
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
        self.val_ratio = val_ratio
        self.seed = seed
        
        if split == "test":
            self.dataset = OriginalModelNetDataLoader(
                root=root_dir,
                split=split,
                npoint=num_points,
                normal_channel=normal_channel,
                process_data=True
            )
        else:
            # For "train" or "val" splits, load full training dataset
            full_train_dataset = OriginalModelNetDataLoader(
                root=root_dir,
                split="train",
                npoint=num_points,
                normal_channel=normal_channel,
                process_data=True
            )
            self.dataset = _split_train_val(full_train_dataset, val_ratio, seed, split)
        
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


def get_dataset_loader(dataset_type: str = "modelnet40", val_ratio: float = 0.2, 
                      seed: int = 42, **kwargs):
    """
    Factory function to get dataset loader based on dataset type.
    
    Args:
        dataset_type: Type of dataset, either "modelnet40" or "scanobjectnn".
        val_ratio: Proportion of training data to use for validation (0 <= val_ratio < 1).
        seed: Random seed for reproducible train/validation split.
        **kwargs: Arguments passed to the loader constructor.
        
    Returns:
        An instance of ModelNet40Loader or ScanObjectNNLoader.
        
    Raises:
        ValueError: If dataset_type is unknown.
    """
    if dataset_type == "modelnet40":
        return ModelNet40Loader(val_ratio=val_ratio, seed=seed, **kwargs)
    elif dataset_type == "scanobjectnn":
        # Import locally to avoid circular import issues
        from .scanobjectnn_loader import ScanObjectNNLoader
        return ScanObjectNNLoader(val_ratio=val_ratio, seed=seed, **kwargs)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
