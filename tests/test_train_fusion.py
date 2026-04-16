import sys
import os
import tempfile
import numpy as np
sys.path.insert(0, '.')
from scripts.train_fusion import parse_args, load_features

def test_train_argparse_defaults():
    # Provide dummy required arguments to avoid argparse error
    args = parse_args(["--train_features", "dummy.npz", "--test_features", "dummy.npz"])
    # Defaults are hardcoded as per spec
    assert args.epochs == 50
    assert args.batch_size == 32
    assert args.learning_rate == 0.001
    assert args.checkpoint_dir == "checkpoints"
    assert args.output_dir == "training_output"

def test_load_features():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create valid npz file
        file_path = os.path.join(tmpdir, "test.npz")
        pointnet_features = np.random.randn(10, 1024)
        ulip_features = np.random.randn(10, 512)
        labels = np.random.randint(0, 40, size=10)
        np.savez_compressed(file_path,
                            pointnet_features=pointnet_features,
                            ulip_features=ulip_features,
                            labels=labels)
        
        data = load_features(file_path)
        assert 'pointnet' in data
        assert 'ulip' in data
        assert 'labels' in data
        assert data['pointnet'].shape == (10, 1024)
        assert data['ulip'].shape == (10, 512)
        assert data['labels'].shape == (10,)
        
        # Test missing file
        missing_path = os.path.join(tmpdir, "missing.npz")
        try:
            load_features(missing_path)
            assert False, "Expected FileNotFoundError"
        except FileNotFoundError:
            pass
        
        # Test missing key
        invalid_path = os.path.join(tmpdir, "invalid.npz")
        np.savez_compressed(invalid_path, pointnet_features=pointnet_features)
        try:
            load_features(invalid_path)
            assert False, "Expected KeyError"
        except KeyError:
            pass

def test_config_file_valid():
    """Test that default config file exists and has required training keys."""
    from project_utils.config_parser import ConfigParser
    config_parser = ConfigParser()
    config = config_parser.load("config/fusion_config.yaml")
    assert 'training' in config
    training = config['training']
    assert 'epochs' in training
    assert 'batch_size' in training
    assert 'learning_rate' in training
    assert 'checkpoint_dir' in training
    assert 'save_interval' in training
    # Validate types
    assert isinstance(training['epochs'], int)
    assert isinstance(training['batch_size'], int)
    assert isinstance(training['learning_rate'], float)
    assert isinstance(training['checkpoint_dir'], str)
    assert isinstance(training['save_interval'], int)