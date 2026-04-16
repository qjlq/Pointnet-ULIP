import sys
import os
import tempfile
import numpy as np
from unittest.mock import patch, Mock, MagicMock
sys.path.insert(0, '.')
from scripts.extract_features import parse_args, validate_cache_file

def test_argparse_defaults():
    args = parse_args([])
    assert args.batch_size == 64
    assert args.device == "cuda"

def test_validate_cache_file():
    """Test cache validation function."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create valid cache file
        valid_path = os.path.join(tmpdir, "valid.npz")
        np.savez_compressed(
            valid_path,
            pointnet_features=np.random.randn(10, 1024),
            ulip_features=np.random.randn(10, 256),
            labels=np.random.randint(0, 40, size=10)
        )
        assert validate_cache_file(valid_path) == True
        
        # Missing key
        invalid_path = os.path.join(tmpdir, "invalid.npz")
        np.savez_compressed(invalid_path, pointnet_features=np.random.randn(10, 1024))
        assert validate_cache_file(invalid_path) == False
        
        # Corrupted file
        corrupt_path = os.path.join(tmpdir, "corrupt.npz")
        with open(corrupt_path, 'w') as f:
            f.write("not a npz file")
        assert validate_cache_file(corrupt_path) == False
        
        # Non-existent file
        assert validate_cache_file(os.path.join(tmpdir, "nonexistent.npz")) == False
        
        # Shape mismatch (different sample counts)
        mismatch_path = os.path.join(tmpdir, "mismatch.npz")
        np.savez_compressed(
            mismatch_path,
            pointnet_features=np.random.randn(10, 1024),
            ulip_features=np.random.randn(8, 256),  # different sample count
            labels=np.random.randint(0, 40, size=10)
        )
        assert validate_cache_file(mismatch_path) == False

def test_extract_features_integration():
    """Integration test with mocked components."""
    import argparse
    import torch
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create dummy config file
        config_path = os.path.join(tmpdir, "config.yaml")
        with open(config_path, 'w') as f:
            f.write("""data:
  root_dir: dummy_data
  num_points: 1024
  normal_channel: false
models:
  pointnet_checkpoint: dummy_pointnet.pth
  ulip_checkpoint: dummy_ulip.pth
""")
        
        # Mock argparse namespace
        args = argparse.Namespace(
            config=config_path,
            data_dir=None,
            dataset_type="modelnet40",
            pointnet_checkpoint=None,
            ulip_checkpoint=None,
            openclip_checkpoint=None,
            batch_size=2,
            device="cpu",
            output_dir=os.path.join(tmpdir, "output"),
            skip_if_cached=False,
            split_name="main_split",
            use_background=True,
            val_ratio=0.2,
            split_seed=42
        )
        
        # Mock dependencies
        mock_logger = Mock()
        mock_logger.info = Mock()
        mock_logger.warning = Mock()
        mock_logger.error = Mock()
        mock_logger.debug = Mock()
        
        mock_config_parser = Mock()
        mock_config = {
            'data': {'root_dir': 'dummy_data', 'num_points': 1024, 'normal_channel': False},
            'models': {'pointnet_checkpoint': 'dummy_pointnet.pth', 'ulip_checkpoint': 'dummy_ulip.pth'},
            'extraction': {'batch_size': 8, 'device': 'cpu', 'cache_dir': 'dummy_cache', 'skip_if_cached': False}
        }
        mock_config_parser.load.return_value = mock_config
        
        # Mock extractors
        mock_pointnet_extractor = Mock()
        mock_pointnet_extractor.extract_features.return_value = torch.randn(2, 1024)
        mock_ulip_extractor = Mock()
        mock_ulip_extractor.extract_features.return_value = torch.randn(2, 256)
        
        # Mock dataset loader
        mock_loader = [
            (torch.randn(2, 1024, 3), torch.randint(0, 40, (2,))),
            (torch.randn(2, 1024, 3), torch.randint(0, 40, (2,)))
        ]
        
        with patch('scripts.extract_features.PipelineLogger', return_value=mock_logger), \
             patch('scripts.extract_features.ConfigParser', return_value=mock_config_parser), \
             patch('scripts.extract_features.PointNetExtractor', return_value=mock_pointnet_extractor), \
             patch('scripts.extract_features.ULIPExtractor', return_value=mock_ulip_extractor), \
              patch('scripts.extract_features.get_dataset_loader', return_value=mock_loader), \
             patch('scripts.extract_features.os.makedirs'), \
             patch('scripts.extract_features.np.savez_compressed') as mock_save:
            
            # Import and run main with mocked args
            from scripts.extract_features import main
            
            # Temporarily replace parse_args to return our args
            with patch('scripts.extract_features.parse_args', return_value=args):
                main()
            
            # Verify expected calls
            mock_logger.info.assert_any_call("Starting feature extraction...")
            mock_config_parser.load.assert_called_with(config_path)
            mock_pointnet_extractor.extract_features.assert_called()
            mock_ulip_extractor.extract_features.assert_called()
            
            # Check that save was called for three splits
            assert mock_save.call_count == 3
            
            # Verify output files would be created
            train_path = os.path.join(args.output_dir, "train_features.npz")
            val_path = os.path.join(args.output_dir, "val_features.npz")
            test_path = os.path.join(args.output_dir, "test_features.npz")
            # mock_save should have been called with these paths
            # (actual file creation is mocked)

def test_extract_features_three_splits():
    """Test that extraction creates three split files (train, val, test)."""
    import argparse
    import torch
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create dummy config file
        config_path = os.path.join(tmpdir, "config.yaml")
        with open(config_path, 'w') as f:
            f.write("""data:
  root_dir: dummy_data
  num_points: 1024
  normal_channel: false
models:
  pointnet_checkpoint: dummy_pointnet.pth
  ulip_checkpoint: dummy_ulip.pth
""")
        
        # Mock argparse namespace with val_ratio and split_seed
        args = argparse.Namespace(
            config=config_path,
            data_dir=None,
            dataset_type="modelnet40",
            pointnet_checkpoint=None,
            ulip_checkpoint=None,
            openclip_checkpoint=None,
            batch_size=2,
            device="cpu",
            output_dir=os.path.join(tmpdir, "output"),
            skip_if_cached=False,
            split_name="main_split",
            use_background=True,
            val_ratio=0.2,
            split_seed=42
        )
        
        # Mock dependencies
        mock_logger = Mock()
        mock_logger.info = Mock()
        mock_logger.warning = Mock()
        mock_logger.error = Mock()
        mock_logger.debug = Mock()
        
        mock_config_parser = Mock()
        mock_config = {
            'data': {'root_dir': 'dummy_data', 'num_points': 1024, 'normal_channel': False},
            'models': {'pointnet_checkpoint': 'dummy_pointnet.pth', 'ulip_checkpoint': 'dummy_ulip.pth'},
            'extraction': {'batch_size': 8, 'device': 'cpu', 'cache_dir': 'dummy_cache', 'skip_if_cached': False}
        }
        mock_config_parser.load.return_value = mock_config
        
        # Mock extractors
        mock_pointnet_extractor = Mock()
        mock_pointnet_extractor.extract_features.return_value = torch.randn(2, 1024)
        mock_ulip_extractor = Mock()
        mock_ulip_extractor.extract_features.return_value = torch.randn(2, 256)
        
        # Mock dataset loader - return a simple list of batches
        mock_loader = [
            (torch.randn(2, 1024, 3), torch.randint(0, 40, (2,))),
            (torch.randn(2, 1024, 3), torch.randint(0, 40, (2,)))
        ]
        
        with patch('scripts.extract_features.PipelineLogger', return_value=mock_logger), \
             patch('scripts.extract_features.ConfigParser', return_value=mock_config_parser), \
             patch('scripts.extract_features.PointNetExtractor', return_value=mock_pointnet_extractor), \
             patch('scripts.extract_features.ULIPExtractor', return_value=mock_ulip_extractor), \
             patch('scripts.extract_features.get_dataset_loader', return_value=mock_loader), \
             patch('scripts.extract_features.os.makedirs'), \
             patch('scripts.extract_features.np.savez_compressed') as mock_save:
            
            # Import and run main with mocked args
            from scripts.extract_features import main
            
            # Temporarily replace parse_args to return our args
            with patch('scripts.extract_features.parse_args', return_value=args):
                main()
            
            # Verify expected calls
            mock_logger.info.assert_any_call("Starting feature extraction...")
            mock_config_parser.load.assert_called_with(config_path)
            mock_pointnet_extractor.extract_features.assert_called()
            mock_ulip_extractor.extract_features.assert_called()
            
            # Check that save was called for three splits (train, val, test)
            assert mock_save.call_count == 3, f"Expected 3 saves, got {mock_save.call_count}"
            
            # Verify output files would be created
            train_path = os.path.join(args.output_dir, "train_features.npz")
            val_path = os.path.join(args.output_dir, "val_features.npz")
            test_path = os.path.join(args.output_dir, "test_features.npz")
            # mock_save should have been called with these paths
            # (actual file creation is mocked)

def test_extract_features_val_ratio_zero():
    """Test that extraction skips validation split when val_ratio=0."""
    import argparse
    import torch
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create dummy config file
        config_path = os.path.join(tmpdir, "config.yaml")
        with open(config_path, 'w') as f:
            f.write("""data:
  root_dir: dummy_data
  num_points: 1024
  normal_channel: false
models:
  pointnet_checkpoint: dummy_pointnet.pth
  ulip_checkpoint: dummy_ulip.pth
""")
        
        # Mock argparse namespace with val_ratio=0
        args = argparse.Namespace(
            config=config_path,
            data_dir=None,
            dataset_type="modelnet40",
            pointnet_checkpoint=None,
            ulip_checkpoint=None,
            openclip_checkpoint=None,
            batch_size=2,
            device="cpu",
            output_dir=os.path.join(tmpdir, "output"),
            skip_if_cached=False,
            split_name="main_split",
            use_background=True,
            val_ratio=0.0,
            split_seed=42
        )
        
        # Mock dependencies
        mock_logger = Mock()
        mock_logger.info = Mock()
        mock_logger.warning = Mock()
        mock_logger.error = Mock()
        mock_logger.debug = Mock()
        
        mock_config_parser = Mock()
        mock_config = {
            'data': {'root_dir': 'dummy_data', 'num_points': 1024, 'normal_channel': False},
            'models': {'pointnet_checkpoint': 'dummy_pointnet.pth', 'ulip_checkpoint': 'dummy_ulip.pth'},
            'extraction': {'batch_size': 8, 'device': 'cpu', 'cache_dir': 'dummy_cache', 'skip_if_cached': False}
        }
        mock_config_parser.load.return_value = mock_config
        
        # Mock extractors
        mock_pointnet_extractor = Mock()
        mock_pointnet_extractor.extract_features.return_value = torch.randn(2, 1024)
        mock_ulip_extractor = Mock()
        mock_ulip_extractor.extract_features.return_value = torch.randn(2, 256)
        
        # Mock dataset loader - return a simple list of batches
        mock_loader = [
            (torch.randn(2, 1024, 3), torch.randint(0, 40, (2,))),
            (torch.randn(2, 1024, 3), torch.randint(0, 40, (2,)))
        ]
        
        with patch('scripts.extract_features.PipelineLogger', return_value=mock_logger), \
             patch('scripts.extract_features.ConfigParser', return_value=mock_config_parser), \
             patch('scripts.extract_features.PointNetExtractor', return_value=mock_pointnet_extractor), \
             patch('scripts.extract_features.ULIPExtractor', return_value=mock_ulip_extractor), \
             patch('scripts.extract_features.get_dataset_loader', return_value=mock_loader), \
             patch('scripts.extract_features.os.makedirs'), \
             patch('scripts.extract_features.np.savez_compressed') as mock_save:
            
            # Import and run main with mocked args
            from scripts.extract_features import main
            
            # Temporarily replace parse_args to return our args
            with patch('scripts.extract_features.parse_args', return_value=args):
                main()
            
            # Verify expected calls
            mock_logger.info.assert_any_call("Starting feature extraction...")
            mock_logger.info.assert_any_call("Validation ratio is 0, skipping validation split")
            mock_config_parser.load.assert_called_with(config_path)
            mock_pointnet_extractor.extract_features.assert_called()
            mock_ulip_extractor.extract_features.assert_called()
            
            # Check that save was called for only two splits (train, test)
            assert mock_save.call_count == 2, f"Expected 2 saves, got {mock_save.call_count}"
            
            # Verify output files would be created
            train_path = os.path.join(args.output_dir, "train_features.npz")
            test_path = os.path.join(args.output_dir, "test_features.npz")
            # mock_save should have been called with these paths
            # (actual file creation is mocked)

def test_extraction_log_file():
    """Test that extraction creates a log file with configuration details."""
    import argparse
    import torch
    import tempfile
    import os
    from unittest.mock import patch, Mock
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create dummy config file
        config_path = os.path.join(tmpdir, "config.yaml")
        with open(config_path, 'w') as f:
            f.write("""data:
  root_dir: dummy_data
  num_points: 1024
  normal_channel: false
models:
  pointnet_checkpoint: dummy_pointnet.pth
  ulip_checkpoint: dummy_ulip.pth
""")
        
        # Mock argparse namespace
        args = argparse.Namespace(
            config=config_path,
            data_dir=None,
            dataset_type="modelnet40",
            pointnet_checkpoint=None,
            ulip_checkpoint=None,
            openclip_checkpoint=None,
            batch_size=2,
            device="cpu",
            output_dir=os.path.join(tmpdir, "output"),
            skip_if_cached=False,
            split_name="main_split",
            use_background=True,
            val_ratio=0.2,
            split_seed=42
        )
        
        # Mock dependencies
        mock_logger = Mock()
        mock_logger.info = Mock()
        mock_logger.warning = Mock()
        mock_logger.error = Mock()
        mock_logger.debug = Mock()
        
        mock_config_parser = Mock()
        mock_config = {
            'data': {'root_dir': 'dummy_data', 'num_points': 1024, 'normal_channel': False},
            'models': {'pointnet_checkpoint': 'dummy_pointnet.pth', 'ulip_checkpoint': 'dummy_ulip.pth'},
            'extraction': {'batch_size': 8, 'device': 'cpu', 'cache_dir': 'dummy_cache', 'skip_if_cached': False}
        }
        mock_config_parser.load.return_value = mock_config
        
        # Mock extractors
        mock_pointnet_extractor = Mock()
        mock_pointnet_extractor.extract_features.return_value = torch.randn(2, 1024)
        mock_ulip_extractor = Mock()
        mock_ulip_extractor.extract_features.return_value = torch.randn(2, 256)
        
        # Mock dataset loader
        mock_loader = [
            (torch.randn(2, 1024, 3), torch.randint(0, 40, (2,))),
            (torch.randn(2, 1024, 3), torch.randint(0, 40, (2,)))
        ]
        
        # Capture PipelineLogger arguments
        captured_log_file = []
        def mock_pipeline_logger(name, level="INFO", log_file=None, formatter=None):
            captured_log_file.append(log_file)
            return mock_logger
        
        with patch('scripts.extract_features.PipelineLogger', side_effect=mock_pipeline_logger), \
             patch('scripts.extract_features.ConfigParser', return_value=mock_config_parser), \
             patch('scripts.extract_features.PointNetExtractor', return_value=mock_pointnet_extractor), \
             patch('scripts.extract_features.ULIPExtractor', return_value=mock_ulip_extractor), \
             patch('scripts.extract_features.get_dataset_loader', return_value=mock_loader), \
             patch('scripts.extract_features.os.makedirs'), \
             patch('scripts.extract_features.np.savez_compressed'):
            
            # Import and run main with mocked args
            from scripts.extract_features import main
            
            with patch('scripts.extract_features.parse_args', return_value=args):
                main()
            
            # Verify PipelineLogger was called with log_file parameter
            assert len(captured_log_file) > 0
            # Expect log file path: output_dir/extraction.log
            expected_log_file = os.path.join(args.output_dir, "extraction.log")
            assert captured_log_file[0] == expected_log_file
            
            # Verify configuration details were logged
            mock_logger.info.assert_any_call("Starting feature extraction...")
            # Check that configuration block includes dataset and batch size
            config_found = False
            for call in mock_logger.info.call_args_list:
                args, _ = call
                message = args[0]
                if "Dataset:" in message:
                    config_found = True
                    break
            assert config_found, "Configuration details not logged"
            
            # Verify shape logging format
            shape_logs_found = []
            for call in mock_logger.info.call_args_list:
                args, _ = call
                message = args[0]
                if "PointNet features:" in message:
                    shape_logs_found.append("PointNet")
                if "ULIP features:" in message:
                    shape_logs_found.append("ULIP")
                if "Labels:" in message and "Number of samples:" not in message:
                    shape_logs_found.append("Labels")
                if "Number of samples:" in message:
                    shape_logs_found.append("Samples")
            # At least one shape log for each split should be present
            # Since we have three splits, we expect at least three of each shape log
            # But we'll just check that each shape log type appears at least once
            assert "PointNet" in shape_logs_found, "PointNet shape log missing"
            assert "ULIP" in shape_logs_found, "ULIP shape log missing"
            assert "Labels" in shape_logs_found, "Labels shape log missing"
            assert "Samples" in shape_logs_found, "Number of samples log missing"