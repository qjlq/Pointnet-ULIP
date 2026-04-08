import sys
import os
from torch.utils.data import DataLoader
from typing import Iterator, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'pointnet_project', 'Pointnet_Pointnet2_pytorch'))
from data_utils.ModelNetDataLoader import ModelNetDataLoader as OriginalModelNetDataLoader

class ModelNet40Loader:
    """Data loader adapter for ModelNet40 dataset using PointNet's original loader."""
    def __init__(self, root_dir: str, split: str = "train", num_points: int = 1024, 
                 normal_channel: bool = False, batch_size: int = 32, num_workers: int = 0):
        """
        Initialize the ModelNet40 data loader.
        
        Args:
            root_dir: Path to the dataset root directory.
            split: Dataset split, either "train" or "test".
            num_points: Number of points to sample from each point cloud.
            normal_channel: Whether to include normal vectors.
            batch_size: Batch size for data loading.
            num_workers: Number of worker processes for data loading (default 0 for safety).
        """
        self.root_dir = root_dir
        self.split = split
        self.num_points = num_points
        self.normal_channel = normal_channel
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.dataset = OriginalModelNetDataLoader(
            root=root_dir,
            split=split,
            npoint=num_points,
            normal_channel=normal_channel,
            process_data=True
        )
        
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
