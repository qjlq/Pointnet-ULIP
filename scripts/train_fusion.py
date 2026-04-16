#!/usr/bin/env python
# scripts/train_fusion.py
import os
import sys
import argparse
import random
import numpy as np
import torch
from typing import Dict, Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from project_utils.logger import PipelineLogger
from project_utils.config_parser import ConfigParser
from fusion_model import FusionModel
from train_utils import train_fusion_model

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train fusion model on extracted features")
    parser.add_argument("--config", type=str, default="config/fusion_config.yaml", help="Path to config file")
    parser.add_argument("--train_features", type=str, required=True, help="Path to train features .npz file")
    parser.add_argument("--test_features", type=str, required=True, help="Path to test features .npz file")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory for checkpoints")
    parser.add_argument("--output_dir", type=str, default="training_output", help="Output directory")
    parser.add_argument("--fusion_type", type=str, default="concat",
                        choices=["concat", "normalized", "gated"],
                        help="Type of fusion mechanism: concat (original), normalized, or gated")
    return parser.parse_args(argv)

def load_features(feature_path: str) -> Dict[str, np.ndarray]:
    """Load features from .npz file.
    
    Expected keys: 'pointnet_features', 'ulip_features', 'labels'
    
    Args:
        feature_path: Path to .npz file
        
    Returns:
        Dictionary with keys 'pointnet', 'ulip', 'labels' containing numpy arrays
        
    Raises:
        FileNotFoundError: If file does not exist
        KeyError: If required keys are missing
    """
    if not os.path.exists(feature_path):
        raise FileNotFoundError(f"Feature file not found: {feature_path}")
    data = np.load(feature_path)
    required_keys = ['pointnet_features', 'ulip_features', 'labels']
    for key in required_keys:
        if key not in data:
            raise KeyError(f"Missing required key '{key}' in feature file {feature_path}")
    return {
        'pointnet': data['pointnet_features'],
        'ulip': data['ulip_features'],
        'labels': data['labels']
    }

def set_random_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    """Main training workflow.
    
    Loads configuration, sets random seeds, loads features, validates data,
    initializes fusion model, and trains with configurable hyperparameters.
    """
    args = parse_args()
    logger = PipelineLogger("train_fusion")
    
    # Load configuration for save_interval (only parameter not in argparse)
    config_parser = ConfigParser()
    config = config_parser.load(args.config)
    training_config = config.get('training', {})
    save_interval = training_config.get('save_interval', 10)
    
    # Use argparse defaults directly (following spec)
    epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    checkpoint_dir = args.checkpoint_dir
    output_dir = args.output_dir
    
    # Set random seeds for reproducibility
    set_random_seed(42)
    
    # Log hyperparameters
    logger.info("Hyperparameters:")
    logger.info(f"  Epochs: {epochs}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Learning rate: {learning_rate}")
    logger.info(f"  Checkpoint directory: {checkpoint_dir}")
    logger.info(f"  Save interval: {save_interval}")
    logger.info(f"  Output directory: {output_dir}")
    
    # Load features
    logger.info("Loading features...")
    try:
        train_data = load_features(args.train_features)
        test_data = load_features(args.test_features)
    except Exception as e:
        logger.error(f"Failed to load features: {e}")
        sys.exit(1)
    
    logger.info(f"Train samples: {len(train_data['labels'])}")
    logger.info(f"Test samples: {len(test_data['labels'])}")
    
    # Dimension validation
    if train_data['pointnet'].shape[1] != test_data['pointnet'].shape[1]:
        logger.error(f"PointNet feature dimension mismatch: train {train_data['pointnet'].shape[1]}, test {test_data['pointnet'].shape[1]}")
        sys.exit(1)
    if train_data['ulip'].shape[1] != test_data['ulip'].shape[1]:
        logger.error(f"ULIP feature dimension mismatch: train {train_data['ulip'].shape[1]}, test {test_data['ulip'].shape[1]}")
        sys.exit(1)
    
    pointnet_dim = train_data['pointnet'].shape[1]
    ulip_dim = train_data['ulip'].shape[1]
    
    # Infer number of classes from labels
    train_labels = train_data['labels']
    test_labels = test_data['labels']
    
    # Check for negative labels
    if train_labels.min() < 0:
        logger.warning(f"Train labels have negative values: min={train_labels.min()}")
    if test_labels.min() < 0:
        logger.warning(f"Test labels have negative values: min={test_labels.min()}")
    
    # Calculate number of classes
    num_classes = int(max(train_labels.max(), test_labels.max())) + 1
    logger.info(f"Inferred number of classes: {num_classes}")
    logger.info(f"Train labels range: {train_labels.min()} to {train_labels.max()}")
    logger.info(f"Test labels range: {test_labels.min()} to {test_labels.max()}")
    
    # Initialize model based on fusion type
    if args.fusion_type == "concat":
        from fusion_model import FusionModel
        model = FusionModel(pointnet_dim=pointnet_dim, ulip_dim=ulip_dim, num_classes=num_classes)
    elif args.fusion_type == "normalized":
        from fusion_model import NormalizedFusionHead
        model = NormalizedFusionHead(geo_dim=pointnet_dim, vlm_dim=ulip_dim,
                                     hidden_dim=256, num_classes=num_classes,
                                     norm_mode='l2')
    elif args.fusion_type == "gated":
        from fusion_model import GatedFusionHead
        model = GatedFusionHead(geo_dim=pointnet_dim, vlm_dim=ulip_dim,
                                hidden_dim=256, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown fusion type: {args.fusion_type}")
    
    # Create directories
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # Train model
    train_fusion_model(
        model=model,
        train_features_pointnet=train_data['pointnet'],
        train_features_ulip=train_data['ulip'],
        train_labels=train_data['labels'],
        test_features_pointnet=test_data['pointnet'],
        test_features_ulip=test_data['ulip'],
        test_labels=test_data['labels'],
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        checkpoint_dir=checkpoint_dir,
        save_interval=save_interval
    )
    
    logger.info("Training completed!")

if __name__ == "__main__":
    main()