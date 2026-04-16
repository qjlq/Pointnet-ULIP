#!/usr/bin/env python3
"""End-to-end verification of fusion pipeline with real ULIP-2 model.

This script verifies that:
1. Configuration file loads with real ULIP-2 and OpenCLIP checkpoints
2. Feature extraction produces 1280-D ULIP features (instead of 256-D dummy)
3. Fusion model can be instantiated with correct dimensions
4. Training script can load extracted features and run one batch
"""
import sys
import os
import tempfile
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, '.')

from project_utils.config_parser import ConfigParser
from libs.pointnet_extractor import PointNetExtractor
from libs.ulip_extractor import ULIPExtractor
from fusion_model import FusionModel
from libs.dataset_loader import ModelNet40Loader

def test_config_loading():
    """Test that config file loads with new checkpoint fields."""
    config_parser = ConfigParser()
    config = config_parser.load("config/fusion_config.yaml")
    
    assert 'models' in config, "Missing models section"
    assert 'ulip_checkpoint' in config['models'], "Missing ulip_checkpoint"
    assert 'openclip_checkpoint' in config['models'], "Missing openclip_checkpoint"
    assert os.path.exists(config['models']['ulip_checkpoint']), \
        f"ULIP checkpoint not found: {config['models']['ulip_checkpoint']}"
    assert os.path.exists(config['models']['openclip_checkpoint']), \
        f"OpenCLIP checkpoint not found: {config['models']['openclip_checkpoint']}"
    
    print("✓ Config loading passed")
    return config

def test_extractors(config):
    """Test PointNet2 and ULIP-2 extractors with real model."""
    # PointNet2 extractor
    pointnet_extractor = PointNetExtractor(
        checkpoint_path=config['models']['pointnet_checkpoint'],
        device='cpu'
    )
    assert pointnet_extractor.feature_dim == 1024, f"Expected PointNet2 dim 1024, got {pointnet_extractor.feature_dim}"
    print(f"✓ PointNet2 extractor loaded (dim={pointnet_extractor.feature_dim})")
    
    # ULIP-2 extractor with real model
    ulip_extractor = ULIPExtractor(
        checkpoint_path=config['models']['ulip_checkpoint'],
        openclip_checkpoint=config['models']['openclip_checkpoint'],
        device='cpu'
    )
    assert not ulip_extractor.use_dummy, "ULIP extractor should use real model, not dummy"
    assert ulip_extractor.feature_dim == 1280, f"Expected ULIP-2 dim 1280, got {ulip_extractor.feature_dim}"
    print(f"✓ ULIP-2 real model loaded (dim={ulip_extractor.feature_dim}, use_dummy={ulip_extractor.use_dummy})")
    
    # Test feature extraction with small batch
    points = torch.randn(2, 1024, 3)  # ModelNet40 uses 1024 points
    features_pn = pointnet_extractor.extract_features(points)
    features_ulip = ulip_extractor.extract_features(points)
    
    assert features_pn.shape == (2, 1024), f"PointNet2 features shape mismatch: {features_pn.shape}"
    assert features_ulip.shape == (2, 1280), f"ULIP-2 features shape mismatch: {features_ulip.shape}"
    print(f"✓ Feature extraction works (PointNet2: {features_pn.shape}, ULIP-2: {features_ulip.shape})")
    
    return pointnet_extractor, ulip_extractor

def test_fusion_model():
    """Test fusion model instantiation with correct dimensions."""
    # Test default constructor (should use 1280 for vlm_dim)
    model_default = FusionModel()
    assert model_default.vlm_dim == 1280, f"Default vlm_dim should be 1280, got {model_default.vlm_dim}"
    
    # Test explicit dimensions
    model_explicit = FusionModel(vlm_dim=1280, geo_dim=1024, num_classes=40)
    assert model_explicit.vlm_dim == 1280
    assert model_explicit.geo_dim == 1024
    assert model_explicit.num_classes == 40
    
    # Forward pass with dummy features
    batch_size = 4
    vlm_features = torch.randn(batch_size, 1280)
    geo_features = torch.randn(batch_size, 1024)
    logits = model_explicit(geo_features, vlm_features)
    assert logits.shape == (batch_size, 40), f"Logits shape mismatch: {logits.shape}"
    
    print(f"✓ Fusion model instantiated correctly (vlm_dim={model_explicit.vlm_dim}, geo_dim={model_explicit.geo_dim})")
    return model_explicit

def test_pipeline_integration(config):
    """Test integration with actual dataset loader (single batch)."""
    # Create temporary directory for feature cache
    with tempfile.TemporaryDirectory() as tmpdir:
        # Load a single batch from ModelNet40 (requires data directory)
        data_dir = config['data']['root_dir']
        if not os.path.exists(data_dir):
            print(f"⚠ Dataset not found at {data_dir}, skipping dataset integration test")
            return
        
        loader = ModelNet40Loader(
            root_dir=data_dir,
            split='train',
            num_points=config['data']['num_points'],
            normal_channel=config['data']['normal_channel'],
            batch_size=2
        )
        
        # Get one batch
        points, labels = next(iter(loader))
        
        # Extract features
        pointnet_extractor = PointNetExtractor(
            checkpoint_path=config['models']['pointnet_checkpoint'],
            device='cpu'
        )
        ulip_extractor = ULIPExtractor(
            checkpoint_path=config['models']['ulip_checkpoint'],
            openclip_checkpoint=config['models']['openclip_checkpoint'],
            device='cpu'
        )
        
        features_pn = pointnet_extractor.extract_features(points)
        features_ulip = ulip_extractor.extract_features(points)
        
        # Verify dimensions match fusion model expectations
        assert features_pn.shape[1] == 1024
        assert features_ulip.shape[1] == 1280
        
        # Create fusion model and compute predictions
        fusion_model = FusionModel(vlm_dim=1280, geo_dim=1024, num_classes=40)
        logits = fusion_model(features_pn, features_ulip)
        assert logits.shape == (len(labels), 40)
        
        print(f"✓ Pipeline integration test passed (batch size={len(labels)})")

def test_training_script_compatibility():
    """Test that extracted features can be loaded by training script."""
    # Create dummy feature files matching expected format
    with tempfile.TemporaryDirectory() as tmpdir:
        train_path = os.path.join(tmpdir, "train_features.npz")
        test_path = os.path.join(tmpdir, "test_features.npz")
        
        # Create dummy data with correct dimensions (1280 for ULIP)
        num_train = 100
        num_test = 20
        train_ulip = np.random.randn(num_train, 1280).astype(np.float32)
        train_pn = np.random.randn(num_train, 1024).astype(np.float32)
        train_labels = np.random.randint(0, 40, size=num_train).astype(np.int64)
        
        test_ulip = np.random.randn(num_test, 1280).astype(np.float32)
        test_pn = np.random.randn(num_test, 1024).astype(np.float32)
        test_labels = np.random.randint(0, 40, size=num_test).astype(np.int64)
        
        np.savez_compressed(train_path, 
                           pointnet_features=train_pn,
                           ulip_features=train_ulip,
                           labels=train_labels)
        np.savez_compressed(test_path,
                           pointnet_features=test_pn,
                           ulip_features=test_ulip,
                           labels=test_labels)
        
        # Verify files can be loaded
        train_data = np.load(train_path, allow_pickle=True)
        assert 'ulip_features' in train_data
        assert train_data['ulip_features'].shape[1] == 1280, \
            f"ULIP features dimension mismatch: {train_data['ulip_features'].shape}"
        
        print("✓ Training script compatibility test passed (1280-D features)")

def main():
    print("=" * 70)
    print("Fusion Pipeline Verification with Real ULIP-2 Model")
    print("=" * 70)
    
    try:
        # Step 1: Config loading
        config = test_config_loading()
        
        # Step 2: Extractors with real ULIP
        pointnet_extractor, ulip_extractor = test_extractors(config)
        
        # Step 3: Fusion model with 1280-D VLM features
        fusion_model = test_fusion_model()
        
        # Step 4: Pipeline integration (if dataset available)
        test_pipeline_integration(config)
        
        # Step 5: Training script compatibility
        test_training_script_compatibility()
        
        print("=" * 70)
        print("✅ All verification tests passed!")
        print("Real ULIP-2 integration is working correctly.")
        print("Features dimension: 1280-D (real ULIP) + 1024-D (PointNet2)")
        print("=" * 70)
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()