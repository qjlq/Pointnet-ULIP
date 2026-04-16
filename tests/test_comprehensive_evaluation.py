#!/usr/bin/env python3
"""Unit tests for comprehensive evaluation module."""
import sys
import os
import tempfile
import numpy as np
import torch
import pytest
import json
import time

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import the module to test (will be created)
try:
    from scripts import comprehensive_evaluation
    HAVE_MODULE = True
except ImportError:
    HAVE_MODULE = False


def create_dummy_features(n_samples=100, pointnet_dim=256, ulip_dim=256, num_classes=40):
    """Create dummy feature and label arrays for testing."""
    pointnet_features = np.random.randn(n_samples, pointnet_dim).astype(np.float32)
    ulip_features = np.random.randn(n_samples, ulip_dim).astype(np.float32)
    labels = np.random.randint(0, num_classes, size=(n_samples,))
    return pointnet_features, ulip_features, labels


@pytest.mark.skipif(not HAVE_MODULE, reason="comprehensive_evaluation module not available")
class TestDataSplitting:
    """Test data splitting functions."""
    
    def test_split_data_indices(self):
        """Test that split_data_indices returns correct sizes."""
        n_samples = 1000
        train_ratio = 0.7
        val_ratio = 0.2
        test_ratio = 0.1
        
        train_idx, val_idx, test_idx = comprehensive_evaluation.split_data_indices(
            n_samples=n_samples,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=42
        )
        
        # Check total size
        assert len(train_idx) + len(val_idx) + len(test_idx) == n_samples
        
        # Check approximate ratios (allow small rounding differences)
        assert abs(len(train_idx) / n_samples - train_ratio) < 0.01
        assert abs(len(val_idx) / n_samples - val_ratio) < 0.01
        assert abs(len(test_idx) / n_samples - test_ratio) < 0.01
        
        # Check no overlap
        train_set = set(train_idx.tolist())
        val_set = set(val_idx.tolist())
        test_set = set(test_idx.tolist())
        assert train_set.isdisjoint(val_set)
        assert train_set.isdisjoint(test_set)
        assert val_set.isdisjoint(test_set)
        
        # Test reproducibility with same seed
        train_idx2, val_idx2, test_idx2 = comprehensive_evaluation.split_data_indices(
            n_samples=n_samples,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=42
        )
        assert np.array_equal(train_idx, train_idx2)
        assert np.array_equal(val_idx, val_idx2)
        assert np.array_equal(test_idx, test_idx2)
        
        # Different seed should produce different splits
        train_idx3, val_idx3, test_idx3 = comprehensive_evaluation.split_data_indices(
            n_samples=n_samples,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=43
        )
        # Not guaranteed to be different, but likely
        # We'll just check that the function doesn't crash
        
    def test_split_data_indices_defaults(self):
        """Test split_data_indices with default ratios."""
        n_samples = 500
        train_idx, val_idx, test_idx = comprehensive_evaluation.split_data_indices(
            n_samples=n_samples,
            seed=123
        )
        # Default ratios: train=0.8, val=0.1, test=0.1
        assert len(train_idx) == 400  # 0.8 * 500
        assert len(val_idx) == 50     # 0.1 * 500
        assert len(test_idx) == 50    # 0.1 * 500


@pytest.mark.skipif(not HAVE_MODULE, reason="comprehensive_evaluation module not available")
class TestTimingMemory:
    """Test timing and memory measurement functions."""
    
    def test_measure_training_time(self):
        """Test that training time measurement works."""
        # Create a dummy model and data
        from fusion_model import FusionModel
        model = FusionModel(pointnet_dim=256, ulip_dim=256, num_classes=40)
        pointnet_features = torch.randn(64, 256)
        ulip_features = torch.randn(64, 256)
        labels = torch.randint(0, 40, (64,))
        
        # We'll test that the timing decorator can be applied
        # For now, just ensure the function exists
        assert hasattr(comprehensive_evaluation, 'measure_time')
        assert callable(comprehensive_evaluation.measure_time)
        
    def test_measure_gpu_memory(self):
        """Test GPU memory measurement (skips if CUDA not available)."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        # Just test that function exists
        assert hasattr(comprehensive_evaluation, 'measure_gpu_memory')
        assert callable(comprehensive_evaluation.measure_gpu_memory)


@pytest.mark.skipif(not HAVE_MODULE, reason="comprehensive_evaluation module not available")
class TestResultsAggregation:
    """Test results aggregation functions."""
    
    def test_aggregate_results(self):
        """Test aggregation of results across seeds."""
        # Create dummy results for 3 seeds using TrainingMetrics objects
        results_per_seed = [
            comprehensive_evaluation.TrainingMetrics(
                accuracy=0.85,
                training_time=120.5,
                inference_speed=1000.0,
                memory_allocated=1024 * 1024 * 1024,  # 1 GB
                per_epoch_times=[1.2, 1.1],
                train_losses=[0.5, 0.4],
                test_accuracies=[0.8, 0.85],
                seed=42
            ),
            comprehensive_evaluation.TrainingMetrics(
                accuracy=0.87,
                training_time=118.2,
                inference_speed=1100.0,
                memory_allocated=1024 * 1024 * 1024 * 2,  # 2 GB
                per_epoch_times=[1.3, 1.0],
                train_losses=[0.6, 0.3],
                test_accuracies=[0.82, 0.87],
                seed=43
            ),
            comprehensive_evaluation.TrainingMetrics(
                accuracy=0.86,
                training_time=122.0,
                inference_speed=1050.0,
                memory_allocated=1024 * 1024 * 1024 * 1.5,  # 1.5 GB
                per_epoch_times=[1.1, 1.2],
                train_losses=[0.55, 0.35],
                test_accuracies=[0.81, 0.86],
                seed=44
            )
        ]
        
        aggregated = comprehensive_evaluation.aggregate_results(results_per_seed)
        
        # Check that mean and std are computed
        assert 'accuracy_mean' in aggregated
        assert 'accuracy_std' in aggregated
        assert 'training_time_mean' in aggregated
        assert 'training_time_std' in aggregated
        assert 'memory_allocated_mean' in aggregated
        assert 'memory_allocated_std' in aggregated
        
        # Check values
        assert abs(aggregated['accuracy_mean'] - np.mean([0.85, 0.87, 0.86])) < 1e-9
        assert abs(aggregated['accuracy_std'] - np.std([0.85, 0.87, 0.86])) < 1e-9
        
        # Check that individual results are also included
        assert 'results_per_seed' in aggregated
        
    def test_generate_table_formats(self, tmp_path):
        """Test table generation functions."""
        aggregated_results = {
            'dataset': 'modelnet40',
            'fusion_type': 'concat',
            'num_seeds': 3,
            'accuracy_mean': 0.86,
            'accuracy_std': 0.01,
            'training_time_mean': 120.5,
            'training_time_std': 5.2,
            'inference_speed_mean': 1000.0,
            'inference_speed_std': 50.0,
            'memory_allocated_mean': 1073741824,
            'memory_allocated_std': 1024
        }
        
        # Test JSON generation
        json_str = comprehensive_evaluation.generate_json_table([aggregated_results])
        data = json.loads(json_str)
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]['dataset'] == 'modelnet40'
        
        # Test CSV generation
        csv_path = tmp_path / "results.csv"
        comprehensive_evaluation.generate_csv_table([aggregated_results], str(csv_path))
        assert csv_path.exists()
        
        # Test Markdown generation
        md_str = comprehensive_evaluation.generate_markdown_table([aggregated_results])
        assert isinstance(md_str, str)
        assert '|' in md_str  # Should contain table markers


if __name__ == '__main__':
    pytest.main([__file__, '-v'])