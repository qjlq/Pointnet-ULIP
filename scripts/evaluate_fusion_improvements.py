#!/usr/bin/env python3
"""Evaluate fusion improvements on ModelNet40 and ScanObjectNN."""
import argparse
import json
import os
import sys
import numpy as np
import torch
import torch.nn as nn

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from train_utils import train_fusion_model

def load_features(feature_path):
    """Load features from .npz file."""
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

def evaluate_fusion(dataset_name, fusion_type, output_dir, epochs=5,
                    lr_scheduler=None, early_stopping_patience=None,
                    save_metrics=False, save_training_curves=False):
    """Evaluate a fusion type on a specific dataset.
    
    Args:
        dataset_name: Name of dataset ('modelnet40' or 'scanobjectnn')
        fusion_type: Type of fusion ('concat', 'normalized', 'gated')
        output_dir: Directory for output files
        epochs: Number of training epochs
        lr_scheduler: Learning rate scheduler ('cosine', 'plateau', or None)
        early_stopping_patience: Patience for early stopping (None to disable)
        save_metrics: If True, save comprehensive metrics to JSON file
        save_training_curves: If True, save training curves to NPZ file
    
    Returns:
        Final test accuracy (float)
    """
    # Determine feature paths based on dataset
    if dataset_name == "modelnet40":
        train_feature_path = "./feature_cache/train_features.npz"
        test_feature_path = "./feature_cache/test_features.npz"
    elif dataset_name == "scanobjectnn":
        train_feature_path = "./scanobjectnn_feature_cache/train_features.npz"
        test_feature_path = "./scanobjectnn_feature_cache/test_features.npz"
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Load features
    train_data = load_features(train_feature_path)
    test_data = load_features(test_feature_path)
    
    # Infer dimensions
    pointnet_dim = train_data['pointnet'].shape[1]
    ulip_dim = train_data['ulip'].shape[1]
    num_classes = int(max(train_data['labels'].max(), test_data['labels'].max())) + 1
    
    # Initialize model based on fusion type
    if fusion_type == "concat":
        from fusion_model import FusionModel
        model = FusionModel(pointnet_dim=pointnet_dim, ulip_dim=ulip_dim, num_classes=num_classes)
    elif fusion_type == "normalized":
        from fusion_model import NormalizedFusionHead
        model = NormalizedFusionHead(geo_dim=pointnet_dim, vlm_dim=ulip_dim,
                                     hidden_dim=256, num_classes=num_classes,
                                     norm_mode='l2')
    elif fusion_type == "gated":
        from fusion_model import GatedFusionHead
        model = GatedFusionHead(geo_dim=pointnet_dim, vlm_dim=ulip_dim,
                                hidden_dim=256, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown fusion type: {fusion_type}")
    
    # Create checkpoint directory for this run
    checkpoint_dir = os.path.join(output_dir, f"checkpoints_{dataset_name}_{fusion_type}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Prepare paths for metrics and curves if requested
    metrics_path = None
    curves_path = None
    if save_metrics:
        metrics_path = os.path.join(checkpoint_dir, 'metrics.json')
    if save_training_curves:
        curves_path = os.path.join(checkpoint_dir, 'training_curves.npz')
    
    # Train with limited epochs for evaluation
    train_losses, test_accuracies = train_fusion_model(
        model=model,
        train_features_pointnet=train_data['pointnet'],
        train_features_ulip=train_data['ulip'],
        train_labels=train_data['labels'],
        test_features_pointnet=test_data['pointnet'],
        test_features_ulip=test_data['ulip'],
        test_labels=test_data['labels'],
        epochs=epochs,
        batch_size=32,
        learning_rate=0.001,
        checkpoint_dir=checkpoint_dir,
        save_interval=epochs + 1,  # Don't save intermediate checkpoints
        lr_scheduler=lr_scheduler,
        early_stopping_patience=early_stopping_patience,
        save_metrics=metrics_path,
        save_training_curves=curves_path
    )
    
    # Return final test accuracy
    return test_accuracies[-1]

def main():
    parser = argparse.ArgumentParser(description="Evaluate fusion improvements")
    parser.add_argument("--output_dir", default="improvement_results", 
                       help="Output directory for results")
    parser.add_argument("--dataset", choices=['modelnet40', 'scanobjectnn', 'all'], default='all',
                       help="Dataset to evaluate (modelnet40, scanobjectnn, or all)")
    parser.add_argument("--epochs", type=int, default=50,
                       help="Number of maximum training epochs")
    parser.add_argument("--lr_scheduler", choices=['cosine', 'plateau', 'none'], default='cosine',
                       help="Learning rate scheduler (cosine, plateau, or none)")
    parser.add_argument("--early_stopping_patience", type=int, default=10,
                       help="Patience for early stopping (epochs without improvement)")
    parser.add_argument("--save_metrics", action='store_true',
                       help="Save comprehensive metrics (per-class accuracy, confusion matrix, F1 scores)")
    parser.add_argument("--save_training_curves", action='store_true',
                       help="Save training curves (loss/accuracy vs epochs)")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Test configurations
    if args.dataset == 'all':
        datasets = ["modelnet40", "scanobjectnn"]
    else:
        datasets = [args.dataset]
    fusion_types = ["concat", "normalized", "gated"]
    
    results = {}
    
    for dataset in datasets:
        results[dataset] = {}
        for fusion_type in fusion_types:
            print(f"Evaluating {dataset} with {fusion_type} fusion...")
            try:
                lr_scheduler = None if args.lr_scheduler == 'none' else args.lr_scheduler
                accuracy = evaluate_fusion(dataset, fusion_type, args.output_dir, epochs=args.epochs,
                                           lr_scheduler=lr_scheduler,
                                           early_stopping_patience=args.early_stopping_patience,
                                           save_metrics=args.save_metrics,
                                           save_training_curves=args.save_training_curves)
                results[dataset][fusion_type] = float(accuracy)
                print(f"  Accuracy: {accuracy:.4f}")
            except Exception as e:
                print(f"  ERROR: {e}")
                results[dataset][fusion_type] = None
    
    # Save results
    results_path = os.path.join(args.output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {results_path}")
    
    # Print summary
    print("\n" + "="*50)
    print("Fusion Improvement Evaluation Results")
    print("="*50)
    for dataset in results:
        print(f"\n{dataset.upper()}:")
        for fusion_type, acc in results[dataset].items():
            if acc is not None:
                print(f"  {fusion_type:12s}: {acc:.4f}")
            else:
                print(f"  {fusion_type:12s}: FAILED")

if __name__ == "__main__":
    main()