"""Integration tests for ScanObjectNN pipeline."""
import os
import sys
import pytest
import numpy as np
import tempfile

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_scanobjectnn_end_to_end_smoke():
    """Smoke test for ScanObjectNN pipeline components."""
    # Test imports
    from libs.dataset_loader import get_dataset_loader
    
    # Test factory function
    root_dir = "pointnet_project/Pointnet_Pointnet2_pytorch/data"
    scanobjectnn_dir = os.path.join(root_dir, "scanobjectnn", "h5_files", "main_split")
    if not os.path.exists(os.path.join(scanobjectnn_dir, "training_objectdataset_augmented25_norot.h5")):
        pytest.skip("ScanObjectNN data not found")
    
    loader = get_dataset_loader(
        dataset_type="scanobjectnn",
        root_dir=root_dir,
        split="train",
        use_background=True,
        split_name="main_split",
        batch_size=4,
        num_points=1024
    )
    assert loader is not None
    
    # Test batch iteration
    batch_count = 0
    for points, labels in loader:
        assert points.shape[0] == 4
        assert points.shape[1] == 1024
        assert points.shape[2] == 3  # ScanObjectNN only has XYZ coordinates
        assert labels.shape[0] == 4
        batch_count += 1
        if batch_count > 2:
            break
    assert batch_count > 0

def test_scanobjectnn_feature_extraction():
    """Test ScanObjectNN feature extraction pipeline."""
    from unittest.mock import patch
    
    # Mock heavy dependencies to avoid loading models during test
    with patch('libs.pointnet_extractor.PointNetExtractor'):
        with patch('libs.ulip_extractor.ULIPExtractor'):
            # Test argument parsing
            from scripts.extract_features import parse_args
            args = parse_args([
                '--dataset_type', 'scanobjectnn',
                '--config', 'config/scanobjectnn_config.yaml',
                '--output_dir', 'test_output',
                '--batch_size', '4'
            ])
            assert args.dataset_type == 'scanobjectnn'
            assert args.batch_size == 4
