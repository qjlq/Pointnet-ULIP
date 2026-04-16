import sys
import pytest
from itertools import islice
sys.path.insert(0, '.')
from libs.dataset_loader import ModelNet40Loader

def test_dataset_loader_initialization():
    import os
    dataset_path = "pointnet_project/Pointnet_Pointnet2_pytorch/data/modelnet40_normal_resampled"
    if not os.path.exists(dataset_path):
        pytest.skip("ModelNet40 dataset not found")
    
    loader = ModelNet40Loader(
        root_dir=dataset_path,
        split="train",
        num_points=1024
    )
    assert loader.split == "train"

def test_dataset_loader_iteration():
    """Test that the data loader can iterate without errors."""
    import os
    dataset_path = "pointnet_project/Pointnet_Pointnet2_pytorch/data/modelnet40_normal_resampled"
    if not os.path.exists(dataset_path):
        pytest.skip("ModelNet40 dataset not found")
    
    loader = ModelNet40Loader(
        root_dir=dataset_path,
        split="test",
        num_points=1024,
        batch_size=4,
        num_workers=0
    )
    # Iterate over first 3 batches to ensure no FileNotFoundError
    for batch in islice(loader, 3):
        assert batch is not None
        # batch is tuple (points, label) as per PointNet's dataset
        assert len(batch) == 2
        points, label = batch
        assert points.shape[0] == 4  # batch size
        assert points.shape[2] == 3 or points.shape[2] == 6  # xyz or xyz+normals
        break  # just test one batch to avoid long test
    assert True  # if we reached here, iteration succeeded

def test_modelnet40_loader_val_split():
    """Test that validation split works and is reproducible."""
    import os
    dataset_path = "pointnet_project/Pointnet_Pointnet2_pytorch/data/modelnet40_normal_resampled"
    if not os.path.exists(dataset_path):
        pytest.skip("ModelNet40 dataset not found")
    
    loader = ModelNet40Loader(
        root_dir=dataset_path,
        split="val",
        val_ratio=0.2,
        seed=42,
        batch_size=16
    )
    assert len(loader) > 0
    # Verify same split with same seed produces same indices
    loader2 = ModelNet40Loader(
        root_dir=dataset_path,
        split="val",
        val_ratio=0.2,
        seed=42,
        batch_size=16
    )
    # Compare dataset indices (random_split creates Subset with indices attribute)
    assert loader.dataset.indices == loader2.dataset.indices
    # Verify val split is smaller than full train
    full_train_loader = ModelNet40Loader(
        root_dir=dataset_path,
        split="train",
        val_ratio=0.2,
        seed=42,
        batch_size=16
    )
    # Train split should be complement of validation split
    assert len(loader.dataset) + len(full_train_loader.dataset) <= 9843  # total train samples
    # Ensure no overlap (indices should be disjoint)
    val_indices = set(loader.dataset.indices)
    train_indices = set(full_train_loader.dataset.indices)
    assert val_indices.isdisjoint(train_indices)

def test_scanobjectnn_loader_val_split():
    """Test that validation split works for ScanObjectNN."""
    # Skip if dataset not available
    import os
    root_dir = "pointnet_project/Pointnet_Pointnet2_pytorch/data"
    h5_path = os.path.join(root_dir, "scanobjectnn", "h5_files", "main_split", "training_objectdataset_augmented25_norot.h5")
    if not os.path.exists(h5_path):
        pytest.skip("ScanObjectNN dataset not found")
    
    from libs.scanobjectnn_loader import ScanObjectNNLoader
    loader = ScanObjectNNLoader(
        root_dir=root_dir,
        split="val",
        val_ratio=0.2,
        seed=42,
        batch_size=16
    )
    assert len(loader) > 0
    # Verify same split with same seed produces same indices
    loader2 = ScanObjectNNLoader(
        root_dir=root_dir,
        split="val",
        val_ratio=0.2,
        seed=42,
        batch_size=16
    )
    assert loader.dataset.indices == loader2.dataset.indices
    # Verify val split is smaller than full train
    full_train_loader = ScanObjectNNLoader(
        root_dir=root_dir,
        split="train",
        val_ratio=0.2,
        seed=42,
        batch_size=16
    )
    # Ensure no overlap
    val_indices = set(loader.dataset.indices)
    train_indices = set(full_train_loader.dataset.indices)
    assert val_indices.isdisjoint(train_indices)

def test_val_ratio_zero():
    """Test that val_ratio=0.0 returns full train dataset."""
    import os
    dataset_path = "pointnet_project/Pointnet_Pointnet2_pytorch/data/modelnet40_normal_resampled"
    if not os.path.exists(dataset_path):
        pytest.skip("ModelNet40 dataset not found")
    
    loader = ModelNet40Loader(
        root_dir=dataset_path,
        split="train",
        val_ratio=0.0,
        seed=42,
        batch_size=32
    )
    # Load full train dataset separately to compare length
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pointnet_project', 'Pointnet_Pointnet2_pytorch'))
    from data_utils.ModelNetDataLoader import ModelNetDataLoader as OriginalModelNetDataLoader
    full_train_dataset = OriginalModelNetDataLoader(
        root=dataset_path,
        split="train",
        npoint=1024,
        normal_channel=False,
        process_data=True
    )
    assert len(loader.dataset) == len(full_train_dataset)

def test_val_ratio_edge_cases():
    """Test invalid val_ratio values raise ValueError."""
    import os
    dataset_path = "pointnet_project/Pointnet_Pointnet2_pytorch/data/modelnet40_normal_resampled"
    if not os.path.exists(dataset_path):
        pytest.skip("ModelNet40 dataset not found")
    
    with pytest.raises(ValueError, match="val_ratio must be between 0.*and 1"):
        ModelNet40Loader(
            root_dir=dataset_path,
            split="train",
            val_ratio=-0.1,
            seed=42
        )
    with pytest.raises(ValueError, match="val_ratio must be between 0.*and 1"):
        ModelNet40Loader(
            root_dir=dataset_path,
            split="train",
            val_ratio=1.0,
            seed=42
        )
    with pytest.raises(ValueError, match="val_ratio must be between 0.*and 1"):
        ModelNet40Loader(
            root_dir=dataset_path,
            split="train",
            val_ratio=2.0,
            seed=42
        )

def test_test_split_ignores_val_ratio():
    """Test that test split ignores val_ratio parameter."""
    import os
    dataset_path = "pointnet_project/Pointnet_Pointnet2_pytorch/data/modelnet40_normal_resampled"
    if not os.path.exists(dataset_path):
        pytest.skip("ModelNet40 dataset not found")
    
    loader1 = ModelNet40Loader(
        root_dir=dataset_path,
        split="test",
        val_ratio=0.0,
        batch_size=32
    )
    loader2 = ModelNet40Loader(
        root_dir=dataset_path,
        split="test",
        val_ratio=0.5,
        batch_size=32
    )
    # Both should have same length (full test dataset)
    assert len(loader1.dataset) == len(loader2.dataset)
    # Verify it's the original test dataset size
    assert len(loader1.dataset) == 2468  # known test size for ModelNet40

def test_get_dataset_loader_with_val_ratio():
    """Test that get_dataset_loader passes val_ratio and seed."""
    import os
    dataset_path = "pointnet_project/Pointnet_Pointnet2_pytorch/data/modelnet40_normal_resampled"
    if not os.path.exists(dataset_path):
        pytest.skip("ModelNet40 dataset not found")
    
    from libs.dataset_loader import get_dataset_loader
    loader = get_dataset_loader(
        dataset_type="modelnet40",
        root_dir=dataset_path,
        split="train",
        val_ratio=0.3,
        seed=123,
        batch_size=32
    )
    assert loader.val_ratio == 0.3
    assert loader.seed == 123
    # Verify the split is applied (dataset should be 70% of full train)
    full_train_loader = get_dataset_loader(
        dataset_type="modelnet40",
        root_dir=dataset_path,
        split="train",
        val_ratio=0.0,
        seed=123,
        batch_size=32
    )
    assert len(loader.dataset) < len(full_train_loader.dataset)

def test_val_split_with_zero_val_ratio():
    """Test that val split with val_ratio=0 raises ValueError."""
    import os
    dataset_path = "pointnet_project/Pointnet_Pointnet2_pytorch/data/modelnet40_normal_resampled"
    if not os.path.exists(dataset_path):
        pytest.skip("ModelNet40 dataset not found")
    
    with pytest.raises(ValueError, match="Cannot create validation split with val_ratio=0"):
        ModelNet40Loader(
            root_dir=dataset_path,
            split="val",
            val_ratio=0.0,
            seed=42
        )

def test_scanobjectnn_val_split_with_zero_val_ratio():
    """Test that ScanObjectNN val split with val_ratio=0 raises ValueError."""
    import os
    root_dir = "pointnet_project/Pointnet_Pointnet2_pytorch/data"
    h5_path = os.path.join(root_dir, "scanobjectnn", "h5_files", "main_split", "training_objectdataset_augmented25_norot.h5")
    if not os.path.exists(h5_path):
        pytest.skip("ScanObjectNN dataset not found")
    
    from libs.scanobjectnn_loader import ScanObjectNNLoader
    with pytest.raises(ValueError, match="Cannot create validation split with val_ratio=0"):
        ScanObjectNNLoader(
            root_dir=root_dir,
            split="val",
            val_ratio=0.0,
            seed=42
        )
