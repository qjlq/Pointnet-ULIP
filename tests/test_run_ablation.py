"""Unit tests for ablation study script."""
import sys
import os
import tempfile
import numpy as np
import pytest
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from project_utils.logger import PipelineLogger
from scripts.run_ablation import parse_args, load_features, run_pointnet_only_experiment, run_ulip_only_experiment, run_fusion_experiment


def test_parse_args_with_validation():
    """Test that argument parser accepts validation features argument."""
    # Test with all three feature files
    test_args = [
        '--train_features', 'train.npz',
        '--val_features', 'val.npz',
        '--test_features', 'test.npz',
        '--output_dir', 'output',
        '--epochs', '50',
        '--batch_size', '32',
        '--learning_rate', '0.001'
    ]
    args = parse_args(test_args)
    
    # Check that val_features is parsed and stored
    assert args.val_features == 'val.npz'
    assert args.train_features == 'train.npz'
    assert args.test_features == 'test.npz'
    assert args.output_dir == 'output'
    assert args.epochs == 50
    assert args.batch_size == 32
    assert args.learning_rate == 0.001


def test_load_features():
    """Test loading features from NPZ files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create dummy feature files
        train_path = os.path.join(tmpdir, 'train.npz')
        val_path = os.path.join(tmpdir, 'val.npz')
        test_path = os.path.join(tmpdir, 'test.npz')
        
        n_samples = 10
        pointnet_dim = 256
        ulip_dim = 256
        
        # Create train features
        np.savez(train_path,
                 pointnet_features=np.random.randn(n_samples, pointnet_dim),
                 ulip_features=np.random.randn(n_samples, ulip_dim),
                 labels=np.random.randint(0, 40, n_samples))
        
        # Create val features (same structure)
        np.savez(val_path,
                 pointnet_features=np.random.randn(n_samples, pointnet_dim),
                 ulip_features=np.random.randn(n_samples, ulip_dim),
                 labels=np.random.randint(0, 40, n_samples))
        
        # Create test features
        np.savez(test_path,
                 pointnet_features=np.random.randn(n_samples, pointnet_dim),
                 ulip_features=np.random.randn(n_samples, ulip_dim),
                 labels=np.random.randint(0, 40, n_samples))
        
        # Load each
        train_data = load_features(train_path)
        val_data = load_features(val_path)
        test_data = load_features(test_path)
        
        assert 'pointnet' in train_data
        assert 'ulip' in train_data
        assert 'labels' in train_data
        assert train_data['pointnet'].shape == (n_samples, pointnet_dim)
        assert train_data['ulip'].shape == (n_samples, ulip_dim)
        assert train_data['labels'].shape == (n_samples,)
        
        # Ensure val and test have same keys
        assert 'pointnet' in val_data
        assert 'ulip' in val_data
        assert 'labels' in val_data


def test_experiment_with_validation():
    """Test that experiment functions run with validation data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create dummy feature files
        train_path = os.path.join(tmpdir, 'train.npz')
        val_path = os.path.join(tmpdir, 'val.npz')
        test_path = os.path.join(tmpdir, 'test.npz')
        
        n_samples = 20
        pointnet_dim = 256
        ulip_dim = 256
        num_classes = 10
        
        # Create train features
        np.savez(train_path,
                 pointnet_features=np.random.randn(n_samples, pointnet_dim),
                 ulip_features=np.random.randn(n_samples, ulip_dim),
                 labels=np.random.randint(0, num_classes, n_samples))
        
        # Create val features
        np.savez(val_path,
                 pointnet_features=np.random.randn(n_samples, pointnet_dim),
                 ulip_features=np.random.randn(n_samples, ulip_dim),
                 labels=np.random.randint(0, num_classes, n_samples))
        
        # Create test features
        np.savez(test_path,
                 pointnet_features=np.random.randn(n_samples, pointnet_dim),
                 ulip_features=np.random.randn(n_samples, ulip_dim),
                 labels=np.random.randint(0, num_classes, n_samples))
        
        # Load data
        train_data = load_features(train_path)
        val_data = load_features(val_path)
        test_data = load_features(test_path)
        
        # Create args namespace
        args = argparse.Namespace(
            checkpoint_dir=None,
            epochs=1,
            batch_size=2,
            learning_rate=0.001,
            lr_scheduler='none',
            early_stopping_patience=None,
            save_metrics=False,
            save_training_curves=False,
            save_interval=10,
            seed=42
        )
        
        # Create logger
        logger = PipelineLogger(name='test', log_file=None)
        
        # Run each experiment (should not raise exceptions)
        exp_dir = os.path.join(tmpdir, 'exp')
        os.makedirs(exp_dir, exist_ok=True)
        
        # PointNet-only
        final_acc, best_acc = run_pointnet_only_experiment(
            train_data, val_data, test_data, exp_dir, args, logger)
        assert isinstance(final_acc, float)
        assert isinstance(best_acc, float)
        
        # ULIP-only
        final_acc, best_acc = run_ulip_only_experiment(
            train_data, val_data, test_data, exp_dir, args, logger)
        assert isinstance(final_acc, float)
        assert isinstance(best_acc, float)
        
        # Fusion
        final_acc, best_acc = run_fusion_experiment(
            train_data, val_data, test_data, exp_dir, args, logger)
        assert isinstance(final_acc, float)
        assert isinstance(best_acc, float)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])