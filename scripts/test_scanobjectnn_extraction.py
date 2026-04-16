#!/usr/bin/env python
"""Quick test of ScanObjectNN feature extraction pipeline."""
import os
import sys
import torch
import numpy as np
import h5py

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from libs.scanobjectnn_loader import ScanObjectNNDataset
from libs.pointnet_extractor import PointNetExtractor
from libs.ulip_extractor import ULIPExtractor

def test_loader():
    """Test the loader works correctly."""
    print("Testing ScanObjectNN loader...")
    
    # Path to a ScanObjectNN HDF5 file
    root_dir = "pointnet_project/Pointnet_Pointnet2_pytorch/data"
    split_name = "main_split"
    h5_path = os.path.join(
        root_dir, "scanobjectnn", "h5_files", split_name,
        "training_objectdataset_augmented25_norot.h5"
    )
    
    if not os.path.exists(h5_path):
        print(f"Error: HDF5 file not found: {h5_path}")
        return False
    
    print(f"Loading dataset from: {h5_path}")
    
    # Test dataset
    dataset = ScanObjectNNDataset(
        h5_path=h5_path,
        num_points=256,  # Smaller for testing
        use_background=True,
        normal_channel=False
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Get a few samples
    for i in range(min(3, len(dataset))):
        points, label = dataset[i]
        print(f"Sample {i}: points shape={points.shape}, label={label}")
    
    return True

def test_extractors():
    """Test that extractors can be loaded and run."""
    print("\nTesting feature extractors...")
    
    try:
        # Test PointNet extractor
        pointnet_checkpoint = "pointnet_project/Pointnet_Pointnet2_pytorch/log/classification/pointnet2_ssg_wo_normals/checkpoints/best_model.pth"
        print(f"Loading PointNet2 extractor from: {pointnet_checkpoint}")
        
        pointnet_extractor = PointNetExtractor(
            checkpoint_path=pointnet_checkpoint,
            device="cpu"
        )
        print("PointNet2 extractor loaded successfully")
        
        # Test ULIP extractor
        ulip_checkpoint = "checkpoints/ULIP-2-PointBERT-10k-xyzrgb-pc-vit_g-objaverse_shapenet-pretrained.pt"
        openclip_checkpoint = "checkpoints/open_clip_pytorch_model.bin"
        print(f"Loading ULIP-2 extractor from: {ulip_checkpoint}")
        
        ulip_extractor = ULIPExtractor(
            checkpoint_path=ulip_checkpoint,
            openclip_checkpoint=openclip_checkpoint,
            device="cpu"
        )
        print("ULIP-2 extractor loaded successfully")
        
        # Test extraction on dummy data
        dummy_points = torch.randn(2, 256, 3)
        print(f"Testing extraction on dummy points: {dummy_points.shape}")
        
        with torch.no_grad():
            features_pn = pointnet_extractor.extract_features(dummy_points)
            features_ulip = ulip_extractor.extract_features(dummy_points)
            
        print(f"PointNet features shape: {features_pn.shape}")
        print(f"ULIP features shape: {features_ulip.shape}")
        
        return True
        
    except Exception as e:
        print(f"Error testing extractors: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_end_to_end():
    """Test end-to-end extraction on a few samples."""
    print("\nTesting end-to-end extraction...")
    
    try:
        # Load extractors
        pointnet_checkpoint = "pointnet_project/Pointnet_Pointnet2_pytorch/log/classification/pointnet2_ssg_wo_normals/checkpoints/best_model.pth"
        ulip_checkpoint = "checkpoints/ULIP-2-PointBERT-10k-xyzrgb-pc-vit_g-objaverse_shapenet-pretrained.pt"
        openclip_checkpoint = "checkpoints/open_clip_pytorch_model.bin"
        
        pointnet_extractor = PointNetExtractor(
            checkpoint_path=pointnet_checkpoint,
            device="cpu"
        )
        
        ulip_extractor = ULIPExtractor(
            checkpoint_path=ulip_checkpoint,
            openclip_checkpoint=openclip_checkpoint,
            device="cpu"
        )
        
        # Load a few samples
        root_dir = "pointnet_project/Pointnet_Pointnet2_pytorch/data"
        split_name = "main_split"
        h5_path = os.path.join(
            root_dir, "scanobjectnn", "h5_files", split_name,
            "training_objectdataset_augmented25_norot.h5"
        )
        
        dataset = ScanObjectNNDataset(
            h5_path=h5_path,
            num_points=256,
            use_background=True,
            normal_channel=False
        )
        
        # Process first 5 samples
        all_features_pn = []
        all_features_ulip = []
        all_labels = []
        
        for i in range(min(5, len(dataset))):
            points, label = dataset[i]
            points = points.unsqueeze(0)  # Add batch dimension
            
            with torch.no_grad():
                features_pn = pointnet_extractor.extract_features(points)
                features_ulip = ulip_extractor.extract_features(points)
            
            all_features_pn.append(features_pn.numpy())
            all_features_ulip.append(features_ulip.numpy())
            all_labels.append(label)
            
            print(f"Processed sample {i}: label={label}")
        
        # Concatenate
        if all_features_pn:
            features_pn = np.concatenate(all_features_pn, axis=0)
            features_ulip = np.concatenate(all_features_ulip, axis=0)
            labels = np.array(all_labels)
            
            print(f"\nExtraction successful!")
            print(f"PointNet features: {features_pn.shape}")
            print(f"ULIP features: {features_ulip.shape}")
            print(f"Labels: {labels.shape}")
            print(f"Unique labels: {np.unique(labels)}")
            
            # Save test output
            output_dir = "scanobjectnn_test_output"
            os.makedirs(output_dir, exist_ok=True)
            
            output_path = os.path.join(output_dir, "test_features.npz")
            np.savez_compressed(
                output_path,
                pointnet_features=features_pn,
                ulip_features=features_ulip,
                labels=labels
            )
            
            print(f"Saved test features to: {output_path}")
            
            # Verify loading
            data = np.load(output_path)
            print(f"Verified load: pointnet={data['pointnet_features'].shape}, "
                  f"ulip={data['ulip_features'].shape}, labels={data['labels'].shape}")
            
            return True
        
    except Exception as e:
        print(f"Error in end-to-end test: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("=" * 60)
    print("ScanObjectNN Feature Extraction Pipeline Test")
    print("=" * 60)
    
    # Test 1: Loader
    if not test_loader():
        print("\nLoader test failed!")
        return 1
    
    # Test 2: Extractors
    if not test_extractors():
        print("\nExtractors test failed!")
        return 1
    
    # Test 3: End-to-end
    if not test_end_to_end():
        print("\nEnd-to-end test failed!")
        return 1
    
    print("\n" + "=" * 60)
    print("All tests passed! Pipeline is ready for full extraction.")
    print("=" * 60)
    
    # Estimate time for full extraction
    print("\nFull dataset extraction estimate:")
    print("- Train samples: 11481")
    print("- Test samples: 2882")
    print("- Batch size 32 -> ~359 train batches, ~90 test batches")
    print("- Approximate time: 1-2 hours (CPU) or 20-30 minutes (GPU)")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())