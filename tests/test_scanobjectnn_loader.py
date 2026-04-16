import os
import sys
import pytest
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from libs.scanobjectnn_loader import ScanObjectNNLoader, ScanObjectNNDataset


class TestScanObjectNNDataset:
    """Test ScanObjectNNDataset class."""
    
    @pytest.fixture
    def dataset_path(self):
        """Path to a ScanObjectNN HDF5 file for testing."""
        # Use the training file from main_split
        root_dir = "pointnet_project/Pointnet_Pointnet2_pytorch/data"
        split_name = "main_split"
        h5_path = os.path.join(
            root_dir, "scanobjectnn", "h5_files", split_name,
            "training_objectdataset_augmented25_norot.h5"
        )
        if not os.path.exists(h5_path):
            pytest.skip(f"ScanObjectNN HDF5 file not found: {h5_path}")
        return h5_path
    
    def test_dataset_initialization(self, dataset_path):
        """Test that dataset can be initialized."""
        dataset = ScanObjectNNDataset(
            h5_path=dataset_path,
            num_points=512,
            use_background=True,
            normal_channel=False
        )
        assert len(dataset) > 0
        assert dataset.num_points == 512
        assert dataset.use_background == True
    
    def test_dataset_get_item(self, dataset_path):
        """Test that __getitem__ returns correct shapes."""
        dataset = ScanObjectNNDataset(
            h5_path=dataset_path,
            num_points=256,
            use_background=True,
            normal_channel=False
        )
        
        # Get a sample
        points, label = dataset[0]
        
        # Check shapes
        assert points.shape == (256, 3)
        assert isinstance(label, int) or (isinstance(label, np.integer) or hasattr(label, 'item'))
        
        # Points should be normalized (centered around origin)
        # Mean should be close to 0, max distance should be <= 1
        points_np = points.numpy() if hasattr(points, 'numpy') else points
        centroid = np.mean(points_np, axis=0)
        distances = np.sqrt(np.sum(points_np**2, axis=1))
        max_dist = np.max(distances)
        
        assert np.allclose(centroid, [0, 0, 0], atol=1e-4)
        assert max_dist <= 1.0 + 1e-6  # Allow small floating point error
    
    def test_dataset_without_background(self, dataset_path):
        """Test dataset when filtering out background points."""
        dataset = ScanObjectNNDataset(
            h5_path=dataset_path,
            num_points=128,
            use_background=False,
            normal_channel=False
        )
        
        points, label = dataset[0]
        assert points.shape == (128, 3)


class TestScanObjectNNLoader:
    """Test ScanObjectNNLoader class."""
    
    @pytest.fixture
    def root_dir(self):
        """Root directory containing scanobjectnn data."""
        root = "pointnet_project/Pointnet_Pointnet2_pytorch/data"
        scanobjectnn_dir = os.path.join(root, "scanobjectnn", "h5_files", "main_split")
        if not os.path.exists(os.path.join(scanobjectnn_dir, "training_objectdataset_augmented25_norot.h5")):
            pytest.skip("ScanObjectNN data not found")
        return root
    
    def test_loader_initialization(self, root_dir):
        """Test that loader can be initialized."""
        loader = ScanObjectNNLoader(
            root_dir=root_dir,
            split="train",
            num_points=1024,
            normal_channel=False,
            batch_size=4,
            use_background=True,
            split_name="main_split"
        )
        
        assert loader.split == "train"
        assert loader.num_points == 1024
        assert loader.batch_size == 4
        assert loader.use_background == True
        assert len(loader) > 0
    
    def test_loader_iteration(self, root_dir):
        """Test that loader yields batches with correct shapes."""
        loader = ScanObjectNNLoader(
            root_dir=root_dir,
            split="train",
            num_points=512,
            normal_channel=False,
            batch_size=8,
            use_background=True,
            split_name="main_split"
        )
        
        # Get first batch
        batch_count = 0
        for points, labels in loader:
            # Check batch shapes
            assert points.shape == (8, 512, 3)
            assert labels.shape == (8,)
            
            batch_count += 1
            if batch_count >= 2:
                break
        
        assert batch_count > 0
    
    def test_loader_without_background(self, root_dir):
        """Test loader when filtering out background points."""
        loader = ScanObjectNNLoader(
            root_dir=root_dir,
            split="train",
            num_points=256,
            normal_channel=False,
            batch_size=4,
            use_background=False,
            split_name="main_split"
        )
        
        points, labels = next(iter(loader))
        assert points.shape == (4, 256, 3)
        assert labels.shape == (4,)
    
    def test_loader_length(self, root_dir):
        """Test that __len__ returns correct number of batches."""
        loader = ScanObjectNNLoader(
            root_dir=root_dir,
            split="train",
            num_points=1024,
            normal_channel=False,
            batch_size=32,
            use_background=True,
            split_name="main_split"
        )
        
        # Length should be ceil(num_samples / batch_size)
        num_samples = len(loader.dataset)
        expected_len = (num_samples + 32 - 1) // 32
        assert len(loader) == expected_len


def test_get_dataset_loader_factory():
    """Test the factory function in dataset_loader.py."""
    from libs.dataset_loader import get_dataset_loader
    
    # Test with modelnet40 (should return ModelNet40Loader)
    # This test doesn't require actual data files
    try:
        loader = get_dataset_loader(
            dataset_type="modelnet40",
            root_dir="dummy",
            split="train",
            num_points=1024,
            normal_channel=False,
            batch_size=32
        )
        # Just check it doesn't crash
        assert loader is not None
    except (FileNotFoundError, OSError):
        # Expected if dummy path doesn't exist
        pass
    
    # Test with scanobjectnn (should return ScanObjectNNLoader)
    try:
        loader = get_dataset_loader(
            dataset_type="scanobjectnn",
            root_dir="pointnet_project/Pointnet_Pointnet2_pytorch/data",
            split="train",
            num_points=1024,
            normal_channel=False,
            batch_size=32,
            use_background=True,
            split_name="main_split"
        )
        # Just check it doesn't crash
        assert loader is not None
    except (FileNotFoundError, OSError) as e:
        # Skip if data not found
        pytest.skip(f"ScanObjectNN data not found: {e}")
    
    # Test unknown dataset type
    with pytest.raises(ValueError, match="Unknown dataset type"):
        get_dataset_loader(dataset_type="unknown")