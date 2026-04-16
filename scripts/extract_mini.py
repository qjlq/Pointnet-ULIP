#!/usr/bin/env python
"""Extract a minimal set of features for testing."""
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from project_utils.logger import PipelineLogger
from libs.scanobjectnn_loader import ScanObjectNNDataset
from libs.pointnet_extractor import PointNetExtractor
from libs.ulip_extractor import ULIPExtractor

def main():
    logger = PipelineLogger("extract_mini")
    
    # Configuration
    data_root = "pointnet_project/Pointnet_Pointnet2_pytorch/data"
    split_name = "main_split"
    use_background = True
    num_points = 1024
    batch_size = 2
    device = "cpu"
    limit_samples = 10
    
    logger.info(f"Extracting {limit_samples} samples from ScanObjectNN")
    
    # Paths to HDF5 files
    train_h5_path = os.path.join(
        data_root, "scanobjectnn", "h5_files", split_name,
        "training_objectdataset_augmented25_norot.h5"
    )
    test_h5_path = os.path.join(
        data_root, "scanobjectnn", "h5_files", split_name,
        "test_objectdataset_augmented25_norot.h5"
    )
    
    if not os.path.exists(train_h5_path):
        logger.error(f"Train HDF5 file not found: {train_h5_path}")
        return 1
    if not os.path.exists(test_h5_path):
        logger.error(f"Test HDF5 file not found: {test_h5_path}")
        return 1
    
    # Create datasets
    train_dataset = ScanObjectNNDataset(
        h5_path=train_h5_path,
        num_points=num_points,
        use_background=use_background,
        normal_channel=False
    )
    
    test_dataset = ScanObjectNNDataset(
        h5_path=test_h5_path,
        num_points=num_points,
        use_background=use_background,
        normal_channel=False
    )
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Test dataset size: {len(test_dataset)}")
    
    # Limit datasets
    from torch.utils.data import Subset
    if len(train_dataset) > limit_samples:
        train_dataset = Subset(train_dataset, list(range(limit_samples)))
    if len(test_dataset) > limit_samples:
        test_dataset = Subset(test_dataset, list(range(limit_samples)))
    
    logger.info(f"Limited train dataset size: {len(train_dataset)}")
    logger.info(f"Limited test dataset size: {len(test_dataset)}")
    
    # Initialize extractors
    pointnet_checkpoint = "pointnet_project/Pointnet_Pointnet2_pytorch/log/classification/pointnet2_ssg_wo_normals/checkpoints/best_model.pth"
    ulip_checkpoint = "checkpoints/ULIP-2-PointBERT-10k-xyzrgb-pc-vit_g-objaverse_shapenet-pretrained.pt"
    openclip_checkpoint = "checkpoints/open_clip_pytorch_model.bin"
    
    logger.info("Initializing PointNet2 extractor...")
    pointnet_extractor = PointNetExtractor(checkpoint_path=pointnet_checkpoint, device=device)
    
    logger.info("Initializing ULIP-2 extractor...")
    ulip_extractor = ULIPExtractor(
        checkpoint_path=ulip_checkpoint,
        openclip_checkpoint=openclip_checkpoint,
        device=device
    )
    
    # Extract features
    output_dir = "scanobjectnn_feature_cache_mini"
    os.makedirs(output_dir, exist_ok=True)
    
    # Function to extract features from dataset
    def extract_features(dataset, split_name):
        pointnet_features = []
        ulip_features = []
        labels = []
        
        from torch.utils.data import DataLoader
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        logger.info(f"Extracting features for {split_name} ({len(dataset)} samples)...")
        
        for batch_idx, (points, batch_labels) in enumerate(dataloader):
            logger.info(f"Processing batch {batch_idx+1}/{len(dataloader)}")
            
            # PointNet features
            pointnet_feats = pointnet_extractor.extract_features(points)
            pointnet_features.append(pointnet_feats.detach().cpu().numpy())
            
            # ULIP features
            ulip_feats = ulip_extractor.extract_features(points)
            ulip_features.append(ulip_feats.detach().cpu().numpy())
            
            labels.append(batch_labels.numpy())
            
            # Break early if we have enough samples
            if (batch_idx + 1) * batch_size >= limit_samples * 2:  # *2 for safety
                break
        
        # Concatenate
        pointnet_features = np.concatenate(pointnet_features, axis=0)
        ulip_features = np.concatenate(ulip_features, axis=0)
        labels = np.concatenate(labels, axis=0)
        
        return pointnet_features, ulip_features, labels
    
    # Extract for train and test
    train_pointnet, train_ulip, train_labels = extract_features(train_dataset, "train")
    test_pointnet, test_ulip, test_labels = extract_features(test_dataset, "test")
    
    # Save
    train_path = os.path.join(output_dir, "train_features.npz")
    test_path = os.path.join(output_dir, "test_features.npz")
    
    np.savez_compressed(train_path, 
                       pointnet_features=train_pointnet,
                       ulip_features=train_ulip,
                       labels=train_labels)
    
    np.savez_compressed(test_path,
                       pointnet_features=test_pointnet,
                       ulip_features=test_ulip,
                       labels=test_labels)
    
    logger.info(f"Saved train features to {train_path}")
    logger.info(f"Saved test features to {test_path}")
    
    # Verify
    logger.info("Verifying saved files...")
    train_data = np.load(train_path)
    test_data = np.load(test_path)
    
    logger.info(f"Train: {len(train_data['labels'])} samples, "
               f"PointNet shape: {train_data['pointnet_features'].shape}, "
               f"ULIP shape: {train_data['ulip_features'].shape}")
    
    logger.info(f"Test: {len(test_data['labels'])} samples, "
               f"PointNet shape: {test_data['pointnet_features'].shape}, "
               f"ULIP shape: {test_data['ulip_features'].shape}")
    
    logger.info("Mini extraction completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())