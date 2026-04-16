#!/usr/bin/env python
"""Test fusion training pipeline.

Verifies:
1. Feature loading from cache
2. Fusion model initialization
3. Training loop (forward/backward)
4. Checkpoint saving
"""

import sys
import os
from pathlib import Path
import numpy as np
import torch
import tempfile
import shutil

# Add project root to path for imports
project_root = str(Path(__file__).parent.parent.parent.absolute())
sys.path.insert(0, project_root)

from fusion_model import FusionModel
from train_utils import train_fusion_model


def print_header(title):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def validate_features(pointnet_features, ulip_features, labels):
    """Validate feature arrays."""
    if pointnet_features.shape[0] != ulip_features.shape[0] or pointnet_features.shape[0] != labels.shape[0]:
        print(f"❌ Sample count mismatch: pointnet {pointnet_features.shape[0]}, ulip {ulip_features.shape[0]}, labels {labels.shape[0]}")
        return False
    if pointnet_features.shape[1] != 1024:
        print(f"⚠ PointNet feature dimension mismatch: expected 1024, got {pointnet_features.shape[1]}")
    if ulip_features.shape[1] != 256:
        print(f"⚠ ULIP feature dimension mismatch: expected 256, got {ulip_features.shape[1]}")
    return True


def test_feature_loading():
    """Test loading features from cache files."""
    print_header("1. Feature Loading Test")
    
    train_path = Path("feature_cache/train_features.npz")
    test_path = Path("feature_cache/test_features.npz")
    
    if not train_path.exists():
        print(f"❌ Train feature cache not found: {train_path}")
        return False, None, None
    if not test_path.exists():
        print(f"❌ Test feature cache not found: {test_path}")
        return False, None, None
    
    try:
        train_data = np.load(train_path)
        test_data = np.load(test_path)
        
        required_keys = {'pointnet_features', 'ulip_features', 'labels'}
        for data, name in [(train_data, 'train'), (test_data, 'test')]:
            missing = required_keys - set(data.keys())
            if missing:
                print(f"❌ {name} cache missing keys: {missing}")
                return False, None, None
        
        print(f"✓ Train samples: {train_data['pointnet_features'].shape[0]}")
        print(f"✓ Test samples: {test_data['pointnet_features'].shape[0]}")
        
        # Validate dimensions
        if not validate_features(train_data['pointnet_features'], train_data['ulip_features'], train_data['labels']):
            return False, None, None
        if not validate_features(test_data['pointnet_features'], test_data['ulip_features'], test_data['labels']):
            return False, None, None
        
        return True, train_data, test_data
        
    except Exception as e:
        print(f"❌ Error loading features: {e}")
        return False, None, None


def test_fusion_model_initialization(pointnet_dim, ulip_dim):
    """Test fusion model initialization and forward pass."""
    print_header("2. Fusion Model Initialization Test")
    
    try:
        model = FusionModel(pointnet_dim=pointnet_dim, ulip_dim=ulip_dim, num_classes=40)
        print(f"✓ FusionModel initialized")
        print(f"  PointNet dimension: {pointnet_dim}")
        print(f"  ULIP dimension: {ulip_dim}")
        print(f"  Total parameters: {sum(p.numel() for p in model.parameters())}")
        
        # Test forward pass with random data
        batch_size = 4
        f_geo = torch.randn(batch_size, pointnet_dim)
        f_vlm = torch.randn(batch_size, ulip_dim)
        
        with torch.no_grad():
            outputs = model(f_geo, f_vlm)
        
        print(f"✓ Forward pass successful")
        print(f"  Input shapes: geo {f_geo.shape}, vlm {f_vlm.shape}")
        print(f"  Output shape: {outputs.shape}")
        
        if outputs.shape != (batch_size, 40):
            print(f"❌ Output shape mismatch: expected ({batch_size}, 40), got {outputs.shape}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Model initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_loop(train_data, test_data, subset_size=100):
    """Test training loop with small subset of data."""
    print_header("3. Training Loop Test")
    
    # Use small subset for quick test
    subset_size = min(subset_size, train_data['pointnet_features'].shape[0], test_data['pointnet_features'].shape[0])
    
    train_subset = {
        'pointnet': train_data['pointnet_features'][:subset_size],
        'ulip': train_data['ulip_features'][:subset_size],
        'labels': train_data['labels'][:subset_size]
    }
    test_subset = {
        'pointnet': test_data['pointnet_features'][:subset_size],
        'ulip': test_data['ulip_features'][:subset_size],
        'labels': test_data['labels'][:subset_size]
    }
    
    print(f"Using subset: {subset_size} samples")
    print(f"  Train shapes: pointnet {train_subset['pointnet'].shape}, ulip {train_subset['ulip'].shape}, labels {train_subset['labels'].shape}")
    print(f"  Test shapes: pointnet {test_subset['pointnet'].shape}, ulip {test_subset['ulip'].shape}, labels {test_subset['labels'].shape}")
    
    # Initialize model with correct dimensions
    pointnet_dim = train_subset['pointnet'].shape[1]
    ulip_dim = train_subset['ulip'].shape[1]
    model = FusionModel(pointnet_dim=pointnet_dim, ulip_dim=ulip_dim, num_classes=40)
    
    # Create temporary directory for checkpoints
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = Path(tmpdir) / "checkpoints"
        
        try:
            print(f"Starting training (2 epochs)...")
            train_losses, test_accs = train_fusion_model(
                model=model,
                train_features_pointnet=train_subset['pointnet'],
                train_features_ulip=train_subset['ulip'],
                train_labels=train_subset['labels'],
                test_features_pointnet=test_subset['pointnet'],
                test_features_ulip=test_subset['ulip'],
                test_labels=test_subset['labels'],
                epochs=2,
                batch_size=16,
                learning_rate=0.001,
                checkpoint_dir=str(checkpoint_dir),
                save_interval=1
            )
            
            print(f"✓ Training completed successfully")
            print(f"  Train losses: {train_losses}")
            print(f"  Test accuracies: {test_accs}")
            
            # Check that checkpoint files were created
            checkpoint_files = list(checkpoint_dir.glob("*.pth"))
            if checkpoint_files:
                print(f"✓ Checkpoints saved: {len(checkpoint_files)} files")
                for f in checkpoint_files:
                    print(f"  - {f.name}")
            else:
                print(f"⚠ No checkpoint files saved (maybe save_interval > epochs)")
            
            # Verify loss decreased (not required but good indicator)
            if len(train_losses) > 1 and train_losses[1] < train_losses[0]:
                print(f"✓ Training loss decreased: {train_losses[0]:.4f} → {train_losses[1]:.4f}")
            else:
                print(f"⚠ Training loss did not decrease (may need more epochs)")
            
            return True
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Main verification function."""
    print("Fusion Training Pipeline Verification")
    print("=" * 60)
    
    success = True
    
    # Step 1: Load features
    load_ok, train_data, test_data = test_feature_loading()
    if not load_ok:
        success = False
        # Cannot proceed
        print_header("VERIFICATION SUMMARY")
        print("❌ Feature loading failed. Cannot proceed with training test.")
        return 1
    
    # Step 2: Test model initialization
    pointnet_dim = train_data['pointnet_features'].shape[1]
    ulip_dim = train_data['ulip_features'].shape[1]
    if not test_fusion_model_initialization(pointnet_dim, ulip_dim):
        success = False
    
    # Step 3: Test training loop with subset
    if success:
        if not test_training_loop(train_data, test_data, subset_size=100):
            success = False
    
    # Summary
    print_header("VERIFICATION SUMMARY")
    if success:
        print("✅ All checks passed!")
        print("Fusion training pipeline is working correctly.")
        return 0
    else:
        print("❌ Some checks failed.")
        print("Please address the issues above before using the fusion training pipeline.")
        return 1


if __name__ == "__main__":
    sys.exit(main())