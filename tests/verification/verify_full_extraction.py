#!/usr/bin/env python
"""Test full feature extraction pipeline (PointNet2 + ULIP-2).

Verifies integration of:
1. Dataset loading
2. PointNet2 feature extraction
3. ULIP-2 feature extraction
4. Feature consistency and validation
"""

import sys
import os
from pathlib import Path
import torch
import numpy as np

# Add project root to path for imports
project_root = str(Path(__file__).parent.parent.parent.absolute())
sys.path.insert(0, project_root)

from libs.pointnet_extractor import PointNetExtractor
from libs.ulip_extractor import ULIPExtractor
from libs.dataset_loader import ModelNet40Loader

# Configuration
import pathlib
PROJECT_ROOT = pathlib.Path.cwd()
POINTNET_CHECKPOINT = PROJECT_ROOT / "pointnet_project/Pointnet_Pointnet2_pytorch/log/classification/pointnet2_ssg_wo_normals/checkpoints/best_model.pth"
ULIP_CHECKPOINT = PROJECT_ROOT / "checkpoints/ulip2_pointbert_weights.pth"
DATA_DIR = PROJECT_ROOT / "pointnet_project/Pointnet_Pointnet2_pytorch/data/modelnet40_normal_resampled"
BATCH_SIZE = 2
NUM_BATCHES = 2  # Limit number of batches for quick test


def print_header(title):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def validate_features(features, name="features"):
    """Validate extracted features for NaN, infinite values, and reasonable ranges."""
    if not torch.is_tensor(features):
        print(f"❌ {name} is not a tensor")
        return False
    
    if torch.isnan(features).any():
        print(f"❌ {name} contains NaN values")
        return False
    
    if torch.isinf(features).any():
        print(f"❌ {name} contains infinite values")
        return False
    
    # Check for reasonable range (features should not be extremely large)
    feature_max = features.abs().max().item()
    if feature_max > 1e6:
        print(f"⚠ {name} has unusually large values (max abs: {feature_max:.2f})")
    
    # Check for all zeros (possible failure)
    if features.abs().max().item() < 1e-8:
        print(f"⚠ {name} are all near-zero (possible extraction failure)")
    
    return True


def check_gpu():
    """Check GPU availability and CUDA version."""
    print_header("1. GPU Availability Check")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    
    if cuda_available:
        device_count = torch.cuda.device_count()
        print(f"Number of CUDA devices: {device_count}")
        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            print(f"  Device {i}: {props.name}")
            print(f"    Compute capability: {props.major}.{props.minor}")
            print(f"    Total memory: {props.total_memory / 1e9:.2f} GB")
        cuda_version = torch.version.cuda
        print(f"CUDA version (PyTorch built with): {cuda_version}")
        return True, "cuda"
    else:
        print("⚠ CUDA not available, falling back to CPU")
        return False, "cpu"


def test_full_extraction(device):
    """Test full feature extraction pipeline."""
    print_header("2. Full Feature Extraction Pipeline Test")
    
    try:
        # Load extractors
        print(f"Loading PointNetExtractor from {POINTNET_CHECKPOINT}")
        pointnet_extractor = PointNetExtractor(
            checkpoint_path=POINTNET_CHECKPOINT,
            device=device,
            normal_channel=False,
            num_class=40
        )
        print(f"✓ PointNetExtractor loaded (device: {pointnet_extractor.device})")
        
        print(f"Loading ULIPExtractor from {ULIP_CHECKPOINT if ULIP_CHECKPOINT.exists() else 'dummy'}")
        ulip_extractor = ULIPExtractor(
            checkpoint_path=str(ULIP_CHECKPOINT) if ULIP_CHECKPOINT.exists() else None,
            device=device
        )
        print(f"✓ ULIPExtractor loaded (device: {ulip_extractor.device}, dummy: {ulip_extractor.use_dummy})")
        
        # Load dataset (train split, small batch size)
        print(f"\nLoading ModelNet40 dataset from {DATA_DIR}")
        if not DATA_DIR.exists():
            print(f"❌ Data directory not found: {DATA_DIR}")
            return False
        loader = ModelNet40Loader(
            root_dir=str(DATA_DIR),
            split="train",
            num_points=1024,
            normal_channel=False,
            batch_size=BATCH_SIZE,
            num_workers=0
        )
        print(f"✓ Dataset loaded: {len(loader)} total batches")
        
        # Process limited number of batches
        print(f"Processing first {NUM_BATCHES} batches...")
        batch_count = 0
        all_pointnet_features = []
        all_ulip_features = []
        all_labels = []
        
        for batch_idx, (points, labels) in enumerate(loader):
            if batch_idx >= NUM_BATCHES:
                break
            print(f"\n  Batch {batch_idx}: points shape {points.shape}, labels shape {labels.shape}")
            
            # Extract features
            try:
                features_pn = pointnet_extractor.extract_features(points)
                features_ulip = ulip_extractor.extract_features(points)
                
                print(f"    PointNet features shape: {features_pn.shape}")
                print(f"    ULIP features shape: {features_ulip.shape}")
                
                # Validate features
                if not validate_features(features_pn, f"Batch {batch_idx} PointNet"):
                    return False
                if not validate_features(features_ulip, f"Batch {batch_idx} ULIP"):
                    return False
                
                # Check dimensions
                if features_pn.shape[0] != BATCH_SIZE:
                    print(f"❌ PointNet batch size mismatch: {features_pn.shape[0]} != {BATCH_SIZE}")
                    return False
                if features_ulip.shape[0] != BATCH_SIZE:
                    print(f"❌ ULIP batch size mismatch: {features_ulip.shape[0]} != {BATCH_SIZE}")
                    return False
                if features_pn.shape[1] != 1024:
                    print(f"❌ PointNet feature dimension mismatch: {features_pn.shape[1]} != 1024")
                    return False
                if features_ulip.shape[1] != 256:
                    print(f"❌ ULIP feature dimension mismatch: {features_ulip.shape[1]} != 256")
                    return False
                
                # Store features
                all_pointnet_features.append(features_pn.numpy())
                all_ulip_features.append(features_ulip.numpy())
                all_labels.append(labels.numpy())
                
                batch_count += 1
                
            except Exception as e:
                print(f"❌ Feature extraction failed for batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                return False
        
        print(f"\n✓ Successfully processed {batch_count} batches")
        
        if batch_count > 0:
            # Concatenate features
            pointnet_features = np.concatenate(all_pointnet_features, axis=0)
            ulip_features = np.concatenate(all_ulip_features, axis=0)
            labels = np.concatenate(all_labels, axis=0)
            
            print(f"\n  Total samples: {pointnet_features.shape[0]}")
            print(f"  PointNet feature matrix: {pointnet_features.shape}")
            print(f"  ULIP feature matrix: {ulip_features.shape}")
            print(f"  Labels: {labels.shape}")
            
            # Verify label range
            unique_labels = np.unique(labels)
            print(f"  Unique labels: {unique_labels}")
            if labels.max() >= 40 or labels.min() < 0:
                print(f"⚠ Label values out of expected range [0, 39]")
            
            # Check that features are not identical across samples (no trivial collapse)
            pointnet_std = pointnet_features.std(axis=0).mean()
            ulip_std = ulip_features.std(axis=0).mean()
            print(f"  Average feature std across samples: PointNet={pointnet_std:.4f}, ULIP={ulip_std:.4f}")
            if pointnet_std < 1e-6:
                print(f"⚠ PointNet features have near-zero variance (possible collapse)")
            if ulip_std < 1e-6:
                print(f"⚠ ULIP features have near-zero variance (possible collapse)")
        
        # Cleanup
        del pointnet_extractor, ulip_extractor
        if device == "cuda":
            torch.cuda.empty_cache()
            print(f"\n  CUDA cache cleared")
        
        return True
        
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        return False
    except RuntimeError as e:
        print(f"❌ Runtime error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main verification function."""
    print("Full Feature Extraction Pipeline Verification")
    print("=" * 60)
    
    success = True
    
    # Step 1: Check GPU
    gpu_available, device = check_gpu()
    
    # Step 2: Test full extraction pipeline
    if not test_full_extraction(device):
        success = False
    
    # Summary
    print_header("VERIFICATION SUMMARY")
    if success:
        print("✅ All checks passed!")
        print("Full feature extraction pipeline is working correctly.")
        if gpu_available:
            print("GPU acceleration is enabled.")
        else:
            print("Running on CPU (GPU not available).")
        return 0
    else:
        print("❌ Some checks failed.")
        print("Please address the issues above before using the full pipeline.")
        return 1


if __name__ == "__main__":
    sys.exit(main())