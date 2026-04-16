# tests/test_run_fusion_pipeline.py
import sys
import os
import tempfile
import json
import copy
import pytest
sys.path.insert(0, '.')
from scripts.run_fusion_pipeline import parse_args, create_experiment_dir, save_config, save_command, validate_config


def test_pipeline_argparse_defaults():
    args = parse_args([])
    assert args.mode == "full"
    assert args.config == "config/fusion_config.yaml"


def test_create_experiment_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test with custom base directory
        experiment_dir = create_experiment_dir(base_dir=tmpdir)
        assert os.path.exists(experiment_dir)
        assert "fusion_pipeline_" in os.path.basename(experiment_dir)
        assert experiment_dir.startswith(tmpdir)
        
        # Test directory creation is idempotent (timestamps may be same in rapid succession)
        experiment_dir2 = create_experiment_dir(base_dir=tmpdir)
        # Directory should exist, timestamp format should be present
        assert os.path.exists(experiment_dir2)
        assert "fusion_pipeline_" in os.path.basename(experiment_dir2)


def test_save_config():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = {"data": {"root_dir": "/test/path"}, "training": {"epochs": 10}}
        save_config(config, tmpdir)
        
        config_path = os.path.join(tmpdir, "config.json")
        assert os.path.exists(config_path)
        
        with open(config_path, 'r') as f:
            loaded_config = json.load(f)
        assert loaded_config == config


def test_save_command():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Mock args object
        class MockArgs:
            pass
        args = MockArgs()
        
        # Save command
        save_command(args, tmpdir)
        
        command_path = os.path.join(tmpdir, "command.txt")
        assert os.path.exists(command_path)
        
        with open(command_path, 'r') as f:
            command = f.read()
        assert len(command) > 0  # Should contain sys.argv


def test_validate_config():
    # Valid config template
    def create_valid_config():
        return {
            'data': {'root_dir': '/path', 'num_points': 1024, 'normal_channel': False},
            'models': {'pointnet_checkpoint': '/path/checkpoint.pth'},
            'extraction': {'batch_size': 64, 'device': 'cuda'},
            'training': {'epochs': 50, 'batch_size': 32, 'learning_rate': 0.001, 
                         'checkpoint_dir': 'checkpoints', 'save_interval': 10}
        }
    
    # Mock logger
    class MockLogger:
        def __init__(self):
            self.errors = []
        def error(self, msg):
            self.errors.append(msg)
    
    # Test valid config
    logger = MockLogger()
    valid_config = create_valid_config()
    assert validate_config(valid_config, logger) == True
    assert len(logger.errors) == 0
    
    # Test missing section
    logger = MockLogger()
    invalid_config = create_valid_config()
    del invalid_config['data']
    assert validate_config(invalid_config, logger) == False
    assert any("Missing required config section" in err for err in logger.errors)
    
    # Test missing field
    logger = MockLogger()
    invalid_config = create_valid_config()
    del invalid_config['data']['root_dir']
    assert validate_config(invalid_config, logger) == False
    assert any("Missing required field" in err for err in logger.errors)
    
    # Test invalid num_points
    logger = MockLogger()
    invalid_config = create_valid_config()
    invalid_config['data']['num_points'] = -1
    result = validate_config(invalid_config, logger)
    assert result == False
    assert any("Invalid num_points" in err for err in logger.errors), f"Expected 'Invalid num_points' in errors: {logger.errors}"
    
    # Test invalid batch_size
    logger = MockLogger()
    invalid_config = create_valid_config()
    invalid_config['extraction']['batch_size'] = 0
    assert validate_config(invalid_config, logger) == False
    assert any("Invalid extraction batch_size" in err for err in logger.errors)
    
    # Test invalid learning_rate
    logger = MockLogger()
    invalid_config = create_valid_config()
    invalid_config['training']['learning_rate'] = 0.0
    assert validate_config(invalid_config, logger) == False
    assert any("Invalid learning_rate" in err for err in logger.errors)


def test_parse_args_with_custom_values():
    # Test custom mode and config
    args = parse_args(["--mode", "extract_only", "--config", "custom_config.yaml"])
    assert args.mode == "extract_only"
    assert args.config == "custom_config.yaml"
    
    # Test output_dir
    args = parse_args(["--output_dir", "/custom/output"])
    assert args.output_dir == "/custom/output"
    assert args.mode == "full"  # Default


if __name__ == "__main__":
    pytest.main([__file__, "-v"])