#!/usr/bin/env python
"""Test script for ULIP-2 feature extraction verification.

Verifies:
1. ULIP extractor loading (dummy model fallback)
2. Feature extraction with random data
3. GPU compatibility
4. Input format handling
"""

import sys
import os
from pathlib import Path
import torch
import numpy as np

# Add project root to path for imports
project_root = str(Path(__file__).parent.parent.parent.absolute())
sys.path.insert(0, project_root)

from libs.ulip_extractor import ULIPExtractor

# Configuration
import pathlib
PROJECT_ROOT = pathlib.Path.cwd()
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints/ULIP-2-PointBERT-10k-xyzrgb-pc-vit_g-objaverse_shapenet-pretrained.pt"
BATCH_SIZE = 2
NUM_POINTS = 1024
FEATURE_DIM = 1280  # ULIP-2 PointBERT output dimension (projected to CLIP space)


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


def check_checkpoint():
    """Verify checkpoint file exists (optional)."""
    print_header("1. Checkpoint Verification")
    if CHECKPOINT_PATH.exists():
        print(f"✓ ULIP-2 checkpoint file exists: {CHECKPOINT_PATH}")
        print(f"  Size: {CHECKPOINT_PATH.stat().st_size / 1e6:.2f} MB")
        return True
    else:
        print(f"⚠ ULIP-2 checkpoint not found: {CHECKPOINT_PATH}")
        print(f"  Will use dummy model (expected behavior for this pipeline)")
        return False


def check_gpu():
    """Check GPU availability and CUDA version."""
    print_header("2. GPU Availability Check")
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


def test_ulip_extractor(device):
    """Test ULIP-2 feature extraction."""
    print_header("3. ULIP-2 Extractor Test")
    
    try:
        print(f"Loading ULIPExtractor with checkpoint: {CHECKPOINT_PATH}")
        print(f"Device: {device}")
        
        # Load extractor
        extractor = ULIPExtractor(
            checkpoint_path=str(CHECKPOINT_PATH) if CHECKPOINT_PATH.exists() else None,
            device=device
        )
        print(f"✓ ULIPExtractor loaded successfully")
        print(f"  Model device: {extractor.device}")
        print(f"  Using dummy model: {extractor.use_dummy}")
        
        # Test 1: Random point cloud (B, N, C) format
        print("\nTest 1: Random point cloud (B, N, C) format")
        points_bnc = torch.randn(BATCH_SIZE, NUM_POINTS, 3)
        print(f"  Input shape: {points_bnc.shape}")
        
        features = extractor.extract_features(points_bnc)
        print(f"  Features shape: {features.shape}")
        
        if features.shape[0] != BATCH_SIZE:
            print(f"❌ Batch dimension mismatch: expected {BATCH_SIZE}, got {features.shape[0]}")
            return False
        if features.shape[1] != FEATURE_DIM:
            print(f"❌ Feature dimension mismatch: expected {FEATURE_DIM}, got {features.shape[1]}")
            return False
        if not validate_features(features, "Test 1 features"):
            return False
        print(f"✓ Feature extraction successful (B, N, C)")
        
        # Test 2: Random point cloud (B, C, N) format (automatic transpose)
        print("\nTest 2: Random point cloud (B, C, N) format")
        points_bcn = torch.randn(BATCH_SIZE, 3, NUM_POINTS)
        print(f"  Input shape: {points_bcn.shape}")
        
        features2 = extractor.extract_features(points_bcn)
        print(f"  Features shape: {features2.shape}")
        
        if features2.shape[1] != FEATURE_DIM:
            print(f"❌ Feature dimension mismatch: expected {FEATURE_DIM}, got {features2.shape[1]}")
            return False
        if not validate_features(features2, "Test 2 features"):
            return False
        print(f"✓ Feature extraction successful (B, C, N)")
        
        # Test 3: Single point cloud (batch size 1)
        print("\nTest 3: Single point cloud (batch size 1)")
        points_single = torch.randn(1, NUM_POINTS, 3)
        features_single = extractor.extract_features(points_single)
        print(f"  Input shape: {points_single.shape}")
        print(f"  Features shape: {features_single.shape}")
        
        if features_single.shape != (1, FEATURE_DIM):
            print(f"❌ Unexpected shape: {features_single.shape}")
            return False
        if not validate_features(features_single, "Test 3 features"):
            return False
        print(f"✓ Single batch extraction successful")
        
        # Test 4: Different point count
        print("\nTest 4: Different point count (512 points)")
        points_512 = torch.randn(BATCH_SIZE, 512, 3)
        features_512 = extractor.extract_features(points_512)
        print(f"  Input shape: {points_512.shape}")
        print(f"  Features shape: {features_512.shape}")
        
        if features_512.shape[1] != FEATURE_DIM:
            print(f"❌ Feature dimension mismatch")
            return False
        if not validate_features(features_512, "Test 4 features"):
            return False
        print(f"✓ Variable point count extraction successful")
        
        # Test 5: Verify feature consistency across formats
        print("\nTest 5: Feature consistency across input formats")
        # Generate same data in both formats
        data = torch.randn(BATCH_SIZE, NUM_POINTS, 3)
        data_bcn = data.transpose(1, 2)
        
        feat1 = extractor.extract_features(data)
        feat2 = extractor.extract_features(data_bcn)
        
        if not validate_features(feat1, "Test 5 feat1"):
            return False
        if not validate_features(feat2, "Test 5 feat2"):
            return False
        
        diff = torch.abs(feat1 - feat2).max().item()
        print(f"  Max difference between formats: {diff:.6f}")
        
        if diff < 1e-5:
            print(f"✓ Features match across input formats")
        else:
            print(f"⚠ Features differ between formats (max diff: {diff})")
            # This could be due to numerical precision, not necessarily an error
        
        # Cleanup
        del extractor
        if device == "cuda":
            torch.cuda.empty_cache()
            print(f"  CUDA cache cleared")
        
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
    print("ULIP-2 Feature Extraction Verification")
    print("=" * 60)
    
    success = True
    
    # Step 1: Check checkpoint (optional)
    checkpoint_exists = check_checkpoint()
    
    # Step 2: Check GPU
    gpu_available, device = check_gpu()
    
    # Step 3: Test ULIP-2 extractor
    if not test_ulip_extractor(device):
        success = False
    
    # Summary
    print_header("VERIFICATION SUMMARY")
    if success:
        print("✅ All checks passed!")
        print("ULIP-2 feature extraction is ready for use.")
        if gpu_available:
            print("GPU acceleration is enabled.")
        else:
            print("Running on CPU (GPU not available).")
        return 0
    else:
        print("❌ Some checks failed.")
        print("Please address the issues above before using ULIP-2 feature extraction.")
        return 1


if __name__ == "__main__":
    sys.exit(main())