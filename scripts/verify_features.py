#!/usr/bin/env python
"""Verify extracted feature files."""
import os
import sys
import numpy as np
import argparse

def verify_file(filepath, expected_samples=None):
    """Verify a feature file."""
    if not os.path.exists(filepath):
        print(f"ERROR: File not found: {filepath}")
        return False
    
    try:
        data = np.load(filepath, allow_pickle=True)
        print(f"\nVerifying: {filepath}")
        print(f"  File size: {os.path.getsize(filepath) / (1024*1024):.2f} MB")
        
        # Check required keys
        required_keys = {'pointnet_features', 'ulip_features', 'labels'}
        missing_keys = required_keys - set(data.keys())
        if missing_keys:
            print(f"  ERROR: Missing keys: {missing_keys}")
            return False
        
        print(f"  Available keys: {list(data.keys())}")
        
        # Get shapes
        pn_shape = data['pointnet_features'].shape
        ulip_shape = data['ulip_features'].shape
        labels_shape = data['labels'].shape
        
        print(f"  PointNet features shape: {pn_shape}")
        print(f"  ULIP features shape: {ulip_shape}")
        print(f"  Labels shape: {labels_shape}")
        
        # Check consistency
        if pn_shape[0] != ulip_shape[0] or pn_shape[0] != labels_shape[0]:
            print(f"  ERROR: Inconsistent sample count: {pn_shape[0]}, {ulip_shape[0]}, {labels_shape[0]}")
            return False
        
        if expected_samples and pn_shape[0] != expected_samples:
            print(f"  WARNING: Expected {expected_samples} samples, got {pn_shape[0]}")
        
        # Check label range for ScanObjectNN (0-14)
        unique_labels = np.unique(data['labels'])
        print(f"  Unique labels: {unique_labels}")
        print(f"  Label range: {unique_labels.min()} - {unique_labels.max()}")
        
        # Check for NaN or Inf
        pn_nan = np.isnan(data['pointnet_features']).any()
        ulip_nan = np.isnan(data['ulip_features']).any()
        
        if pn_nan or ulip_nan:
            print(f"  WARNING: NaN values found in features")
        
        # Feature statistics
        print(f"  PointNet feature stats: min={data['pointnet_features'].min():.4f}, "
              f"max={data['pointnet_features'].max():.4f}, "
              f"mean={data['pointnet_features'].mean():.4f}")
        print(f"  ULIP feature stats: min={data['ulip_features'].min():.4f}, "
              f"max={data['ulip_features'].max():.4f}, "
              f"mean={data['ulip_features'].mean():.4f}")
        
        return True
        
    except Exception as e:
        print(f"  ERROR: Failed to load/verify file: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Verify extracted feature files")
    parser.add_argument("--train_file", type=str, help="Train features file (.npz)")
    parser.add_argument("--test_file", type=str, help="Test features file (.npz)")
    parser.add_argument("--expected_train_samples", type=int, default=11481,
                       help="Expected number of training samples (ScanObjectNN: 11481)")
    parser.add_argument("--expected_test_samples", type=int, default=2894,
                       help="Expected number of test samples (ScanObjectNN: 2894)")
    args = parser.parse_args()
    
    success = True
    
    if args.train_file:
        if not verify_file(args.train_file, args.expected_train_samples):
            success = False
    
    if args.test_file:
        if not verify_file(args.test_file, args.expected_test_samples):
            success = False
    
    # If no files specified, check common directories
    if not args.train_file and not args.test_file:
        print("Checking common ScanObjectNN feature directories...")
        directories = [
            "scanobjectnn_feature_cache_mini",
            "scanobjectnn_feature_cache_full_cpu", 
            "scanobjectnn_feature_cache_cpu",
            "scanobjectnn_feature_cache_quick",
            "scanobjectnn_feature_cache_test",
            "scanobjectnn_feature_cache_partial",
            "scanobjectnn_feature_cache_dummy"
        ]
        
        for dirname in directories:
            train_path = os.path.join(dirname, "train_features.npz")
            test_path = os.path.join(dirname, "test_features.npz")
            
            if os.path.exists(dirname):
                print(f"\n{'='*60}")
                print(f"Directory: {dirname}")
                print(f"{'='*60}")
                
                if os.path.exists(train_path):
                    verify_file(train_path)
                else:
                    print(f"  No train features found")
                
                if os.path.exists(test_path):
                    verify_file(test_path)
                else:
                    print(f"  No test features found")
    
    if success:
        print("\n✓ Verification completed successfully!")
        return 0
    else:
        print("\n✗ Verification failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())