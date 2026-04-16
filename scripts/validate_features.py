#!/usr/bin/env python
# scripts/validate_features.py
import numpy as np
import torch
import sys

def validate_features():
    """验证特征文件的完整性和正确性"""
    print("Validating feature files...")
    
    # 1. 检查文件存在性
    train_path = "feature_cache_full/train_features.npz"
    test_path = "feature_cache_full/test_features.npz"
    
    for path in [train_path, test_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Feature file not found: {path}")
    
    # 2. 加载并检查训练特征
    train_data = np.load(train_path)
    print(f"Train features loaded. Keys: {list(train_data.keys())}")
    
    pointnet_train = train_data['pointnet_features']
    ulip_train = train_data['ulip_features']
    labels_train = train_data['labels']
    
    print(f"Train PointNet features: {pointnet_train.shape} (expected: (9843, 1024))")
    print(f"Train ULIP features: {ulip_train.shape} (expected: (9843, 1280) for real model)")
    print(f"Train labels: {labels_train.shape} (expected: (9843,))")
    
    # 3. 检查ULIP特征维度（必须为1280，非dummy）
    if ulip_train.shape[1] != 1280:
        raise ValueError(f"ULIP features are dummy (256D) or corrupted. Got {ulip_train.shape[1]}D, expected 1280D")
    
    # 4. 加载并检查测试特征
    test_data = np.load(test_path)
    print(f"Test features loaded. Keys: {list(test_data.keys())}")
    
    pointnet_test = test_data['pointnet_features']
    ulip_test = test_data['ulip_features']
    labels_test = test_data['labels']
    
    print(f"Test PointNet features: {pointnet_test.shape} (expected: (2468, 1024))")
    print(f"Test ULIP features: {ulip_test.shape} (expected: (2468, 1280))")
    print(f"Test labels: {labels_test.shape} (expected: (2468,))")
    
    # 5. 检查GPU内存
    if torch.cuda.is_available():
        free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
        free_memory_gb = free_memory / (1024**3)
        print(f"GPU free memory: {free_memory_gb:.2f} GB")
        
        if free_memory_gb < 2.0:
            print("WARNING: Low GPU memory available (< 2GB). Training may fail.")
            torch.cuda.empty_cache()
    else:
        print("WARNING: CUDA not available. Training will be very slow.")
    
    print("✓ Feature validation passed!")
    return True

if __name__ == "__main__":
    import os
    try:
        validate_features()
        sys.exit(0)
    except Exception as e:
        print(f"✗ Feature validation failed: {e}")
        sys.exit(1)