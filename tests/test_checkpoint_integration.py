import sys
sys.path.insert(0, '.')
import os
import tempfile
import numpy as np
import torch
from fusion_model import FusionModel
from train_utils import train_fusion_model

def test_checkpoint_saving():
    """Test that checkpoints are saved when checkpoint_dir is provided."""
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = os.path.join(tmpdir, "checkpoints")
        
        # Create dummy features
        n_train = 100
        n_test = 20
        pointnet_dim = 256
        ulip_dim = 256
        
        train_pointnet = np.random.randn(n_train, pointnet_dim).astype(np.float32)
        train_ulip = np.random.randn(n_train, ulip_dim).astype(np.float32)
        train_labels = np.random.randint(0, 40, n_train).astype(np.int64)
        
        test_pointnet = np.random.randn(n_test, pointnet_dim).astype(np.float32)
        test_ulip = np.random.randn(n_test, ulip_dim).astype(np.float32)
        test_labels = np.random.randint(0, 40, n_test).astype(np.int64)
        
        model = FusionModel(pointnet_dim=pointnet_dim, ulip_dim=ulip_dim, num_classes=40)
        
        # Train for 3 epochs with checkpoint_dir
        train_losses, test_accs = train_fusion_model(
            model=model,
            train_features_pointnet=train_pointnet,
            train_features_ulip=train_ulip,
            train_labels=train_labels,
            test_features_pointnet=test_pointnet,
            test_features_ulip=test_ulip,
            test_labels=test_labels,
            epochs=3,
            batch_size=16,
            learning_rate=0.001,
            checkpoint_dir=checkpoint_dir,
            save_interval=2
        )
        
        # Check that checkpoint directory exists
        assert os.path.exists(checkpoint_dir), f"Checkpoint directory {checkpoint_dir} not created"
        
        # Check that at least one checkpoint file exists (epoch 2 should be saved)
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_epoch_')]
        assert len(checkpoint_files) > 0, f"No checkpoint files found in {checkpoint_dir}"
        
        # Verify that epoch 2 checkpoint exists (save_interval=2)
        expected_checkpoint = os.path.join(checkpoint_dir, "checkpoint_epoch_2.pth")
        assert os.path.exists(expected_checkpoint), f"Expected checkpoint {expected_checkpoint} not found"
        
        # Load checkpoint and verify contents
        checkpoint = torch.load(expected_checkpoint, map_location='cpu')
        assert checkpoint['epoch'] == 2
        assert 'model_state_dict' in checkpoint
        assert 'optimizer_state_dict' in checkpoint
        assert 'loss' in checkpoint
        assert 'accuracy' in checkpoint
        
        # Verify that best model may exist (if accuracy improved)
        best_checkpoint = os.path.join(checkpoint_dir, "best_model.pth")
        if os.path.exists(best_checkpoint):
            best = torch.load(best_checkpoint, map_location='cpu')
            assert best['epoch'] >= 1
        
        print(f"Checkpoint test passed: {len(checkpoint_files)} checkpoint files created")

def test_no_checkpoint_saving():
    """Test that no checkpoints are saved when checkpoint_dir is None."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Change to temporary directory to isolate from existing checkpoints/
        original_cwd = os.getcwd()
        os.chdir(tmpdir)
        try:
            # Create dummy features
            n_train = 50
            n_test = 10
            pointnet_dim = 256
            ulip_dim = 256
            
            train_pointnet = np.random.randn(n_train, pointnet_dim).astype(np.float32)
            train_ulip = np.random.randn(n_train, ulip_dim).astype(np.float32)
            train_labels = np.random.randint(0, 40, n_train).astype(np.int64)
            
            test_pointnet = np.random.randn(n_test, pointnet_dim).astype(np.float32)
            test_ulip = np.random.randn(n_test, ulip_dim).astype(np.float32)
            test_labels = np.random.randint(0, 40, n_test).astype(np.int64)
            
            model = FusionModel(pointnet_dim=pointnet_dim, ulip_dim=ulip_dim, num_classes=40)
            
            # Train for 2 epochs without checkpoint_dir
            train_losses, test_accs = train_fusion_model(
                model=model,
                train_features_pointnet=train_pointnet,
                train_features_ulip=train_ulip,
                train_labels=train_labels,
                test_features_pointnet=test_pointnet,
                test_features_ulip=test_ulip,
                test_labels=test_labels,
                epochs=2,
                batch_size=16,
                learning_rate=0.001,
                checkpoint_dir=None  # Explicitly None
            )
            
            # Ensure no checkpoint directory was created in the temporary directory
            # Since we didn't provide checkpoint_dir, no checkpoints should be saved
            assert not os.path.exists('checkpoints'), "Default checkpoint directory should not be created"
            
            print("No checkpoint test passed")
        finally:
            os.chdir(original_cwd)

if __name__ == "__main__":
    test_checkpoint_saving()
    test_no_checkpoint_saving()
    print("All checkpoint integration tests passed!")