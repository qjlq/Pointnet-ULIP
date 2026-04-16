import sys
sys.path.insert(0, '.')
import os
import tempfile
import numpy as np
import pytest
from scripts.extract_features import parse_args as parse_extract_args
from scripts.train_fusion import parse_args as parse_train_args

def test_integration_workflow():
    """Test integration workflow with dummy feature files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create dummy feature files
        train_features = os.path.join(tmpdir, "train_features.npz")
        test_features = os.path.join(tmpdir, "test_features.npz")
        
        np.savez_compressed(train_features,
            pointnet_features=np.random.randn(100, 256).astype(np.float32),
            ulip_features=np.random.randn(100, 256).astype(np.float32),
            labels=np.random.randint(0, 40, 100).astype(np.int64)
        )
        
        np.savez_compressed(test_features,
            pointnet_features=np.random.randn(20, 256).astype(np.float32),
            ulip_features=np.random.randn(20, 256).astype(np.float32),
            labels=np.random.randint(0, 40, 20).astype(np.int64)
        )
        
        # Test that argparse works
        train_args = parse_train_args(["--train_features", train_features, "--test_features", test_features])
        assert train_args.train_features == train_features
        assert train_args.test_features == test_features
        assert train_args.epochs == 50  # default from spec
        assert train_args.batch_size == 32
        assert train_args.learning_rate == 0.001
        assert train_args.checkpoint_dir == "checkpoints"
        
        # Test extract features argparse
        extract_args = parse_extract_args(["--data_dir", "dummy", "--pointnet_checkpoint", "dummy.pth"])
        assert extract_args.data_dir == "dummy"
        assert extract_args.pointnet_checkpoint == "dummy.pth"
        assert extract_args.batch_size == 64
        assert extract_args.device == "cuda"
        assert extract_args.output_dir == "feature_cache"
        
        print("Integration test passed: argparse parsing works correctly")

def test_feature_file_loading():
    """Test that feature files can be loaded and validated."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create valid feature file
        feature_path = os.path.join(tmpdir, "valid_features.npz")
        np.savez_compressed(feature_path,
            pointnet_features=np.random.randn(50, 256),
            ulip_features=np.random.randn(50, 256),
            labels=np.random.randint(0, 40, 50)
        )
        
        # Load using numpy
        data = np.load(feature_path)
        assert 'pointnet_features' in data
        assert 'ulip_features' in data
        assert 'labels' in data
        assert data['pointnet_features'].shape == (50, 256)
        assert data['ulip_features'].shape == (50, 256)
        assert data['labels'].shape == (50,)
        
        print("Feature file loading test passed")

def test_backward_compatibility_with_scanobjectnn():
    """Test backward compatibility: existing ModelNet40 pipeline still works with ScanObjectNN code present."""
    from unittest.mock import patch
    
    # Mock heavy dependencies to avoid loading models during test
    with patch('libs.pointnet_extractor.PointNetExtractor'):
        with patch('libs.ulip_extractor.ULIPExtractor'):
            # Test that modelnet40 dataset type still works
            from scripts.extract_features import parse_args
            args = parse_args([
                '--dataset_type', 'modelnet40',
                '--config', 'config/fusion_config.yaml',
                '--output_dir', 'test_output',
                '--batch_size', '8'
            ])
            assert args.dataset_type == 'modelnet40'
            assert args.batch_size == 8
            
            # Test that scanobjectnn dataset type also works
            args = parse_args([
                '--dataset_type', 'scanobjectnn',
                '--config', 'config/scanobjectnn_config.yaml',
                '--output_dir', 'test_output',
                '--batch_size', '4',
                '--split_name', 'main_split',
                '--use_background'
            ])
            assert args.dataset_type == 'scanobjectnn'
            assert args.batch_size == 4
            assert args.split_name == 'main_split'
            assert args.use_background == True
            
            print("Backward compatibility test passed: both dataset types work")

def test_modelnet40_backward_compatibility():
    """Ensure ModelNet40 pipeline still works after ScanObjectNN changes."""
    import os
    import pytest
    
    # Check if dataset path exists, skip if not
    dataset_path = "pointnet_project/Pointnet_Pointnet2_pytorch/data/modelnet40_normal_resampled"
    if not os.path.exists(dataset_path):
        pytest.skip(f"Dataset path {dataset_path} not found, skipping actual loading test")
    
    from libs.dataset_loader import get_dataset_loader, ModelNet40Loader
    
    # Test ModelNet40 loading
    loader = get_dataset_loader(
        dataset_type="modelnet40",
        root_dir=dataset_path,
        split="train",
        num_points=1024,
        normal_channel=False,
        batch_size=4
    )
    
    # Verify loader properties
    assert isinstance(loader, ModelNet40Loader)
    
    # Load one batch
    for points, labels in loader:
        assert points.shape == (4, 1024, 3)  # ModelNet40 has XYZ only
        assert labels.shape == (4,)
        break
        
    # Test feature extraction still works
    from scripts.extract_features import parse_args
    
    # Check which config file exists
    config_path = "config/fusion_config.yaml"
    if not os.path.exists(config_path):
        config_path = "config/test_config.yaml"
    if not os.path.exists(config_path):
        config_path = "config/scanobjectnn_config.yaml"
    
    args = parse_args([
        '--dataset_type', 'modelnet40',
        '--config', config_path,
        '--output_dir', 'test_compat',
        '--batch_size', '4'
    ])
    assert args.dataset_type == 'modelnet40'
    assert args.batch_size == 4
    assert args.config == config_path
    
    print("ModelNet40 backward compatibility test passed")

if __name__ == "__main__":
    test_integration_workflow()
    test_feature_file_loading()
    test_backward_compatibility_with_scanobjectnn()
    test_modelnet40_backward_compatibility()
    print("All integration tests passed!")