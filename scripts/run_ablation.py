#!/usr/bin/env python
# scripts/run_ablation.py
import os
import sys
import argparse
import json
import numpy as np
import torch
from typing import Dict, Any, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from project_utils.logger import PipelineLogger
from fusion_model import PointNetOnlyClassifier, ULIPOnlyClassifier, FusionModel
from train_utils import train_single_feature_model, train_fusion_model


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run ablation studies for feature fusion")
    parser.add_argument("--train_features", type=str, required=True, 
                       help="Path to train features .npz file")
    parser.add_argument("--val_features", type=str, required=True,
                       help="Path to validation features .npz file")
    parser.add_argument("--test_features", type=str, required=True,
                       help="Path to test features .npz file")
    parser.add_argument("--output_dir", type=str, default="ablation_results",
                       help="Output directory for ablation results")
    parser.add_argument("--epochs", type=int, default=50,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                       help="Learning rate")
    parser.add_argument("--checkpoint_dir", type=str,
                       help="Directory for checkpoints (default: output_dir/experiment_name/checkpoints)")
    parser.add_argument("--save_interval", type=int, default=10,
                       help="Save checkpoint every N epochs")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--lr_scheduler", choices=['cosine', 'plateau', 'none'], default='none',
                       help="Learning rate scheduler (cosine, plateau, or none)")
    parser.add_argument("--early_stopping_patience", type=int,
                       help="Patience for early stopping (disabled if not specified)")
    parser.add_argument("--save_metrics", action='store_true',
                       help="Save comprehensive metrics (per-class accuracy, confusion matrix, F1 scores)")
    parser.add_argument("--save_training_curves", action='store_true',
                       help="Save training curves (loss/accuracy vs epochs)")
    return parser.parse_args(argv)


def set_random_seed(seed: int):
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_features(feature_path: str) -> Dict[str, np.ndarray]:
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


def run_pointnet_only_experiment(train_data: Dict[str, np.ndarray],
                                 val_data: Dict[str, np.ndarray],
                                 test_data: Dict[str, np.ndarray],
                                 exp_dir: str,
                                 args: argparse.Namespace,
                                 logger: PipelineLogger) -> Tuple[float, float]:
    """Run experiment using only PointNet2 features."""
    logger.info("Running PointNet-only experiment...")
    
    # Prepare features
    train_features = torch.FloatTensor(train_data['pointnet'])
    train_labels = torch.LongTensor(train_data['labels'])
    val_features = torch.FloatTensor(val_data['pointnet'])
    val_labels = torch.LongTensor(val_data['labels'])
    test_features = torch.FloatTensor(test_data['pointnet'])
    test_labels = torch.LongTensor(test_data['labels'])
    
    # Create model
    input_dim = train_features.shape[1]
    num_classes = int(max(train_data['labels'].max(), val_data['labels'].max(), test_data['labels'].max())) + 1
    logger.info(f"Detected {num_classes} classes from labels")
    model = PointNetOnlyClassifier(input_dim=input_dim, num_classes=num_classes)
    
    # Train
    checkpoint_dir = args.checkpoint_dir or os.path.join(exp_dir, "checkpoints")
    train_losses, val_accs, test_accs = train_single_feature_model(
        model=model,
        features=train_features,
        labels=train_labels,
        val_features=val_features,
        val_labels=val_labels,
        test_features=test_features,
        test_labels=test_labels,
        epochs=args.epochs,
        lr=args.learning_rate,
        batch_size=args.batch_size,
        checkpoint_dir=checkpoint_dir,
        save_interval=args.save_interval,
        lr_scheduler=None if args.lr_scheduler == 'none' else args.lr_scheduler,
        early_stopping_patience=args.early_stopping_patience,
        save_metrics=os.path.join(exp_dir, "metrics.json") if args.save_metrics else None,
        save_training_curves=os.path.join(exp_dir, "training_curves.npz") if args.save_training_curves else None
    )
    
    # Save results
    final_accuracy = test_accs[-1]
    best_accuracy = max(test_accs)
    
    results = {
        'experiment': 'pointnet_only',
        'final_accuracy': float(final_accuracy),
        'best_accuracy': float(best_accuracy),
        'final_loss': float(train_losses[-1]),
        'train_losses': [float(l) for l in train_losses],
        'test_accuracies': [float(a) for a in test_accs],
        'input_dim': input_dim
    }
    
    with open(os.path.join(exp_dir, "results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"PointNet-only: final accuracy = {final_accuracy:.4f}, best = {best_accuracy:.4f}")
    return final_accuracy, best_accuracy


def run_ulip_only_experiment(train_data: Dict[str, np.ndarray],
                             val_data: Dict[str, np.ndarray],
                             test_data: Dict[str, np.ndarray],
                             exp_dir: str,
                             args: argparse.Namespace,
                             logger: PipelineLogger) -> Tuple[float, float]:
    """Run experiment using only ULIP-2 features."""
    logger.info("Running ULIP-only experiment...")
    
    # Prepare features
    train_features = torch.FloatTensor(train_data['ulip'])
    train_labels = torch.LongTensor(train_data['labels'])
    val_features = torch.FloatTensor(val_data['ulip'])
    val_labels = torch.LongTensor(val_data['labels'])
    test_features = torch.FloatTensor(test_data['ulip'])
    test_labels = torch.LongTensor(test_data['labels'])
    
    # Create model
    input_dim = train_features.shape[1]
    num_classes = int(max(train_data['labels'].max(), val_data['labels'].max(), test_data['labels'].max())) + 1
    logger.info(f"Detected {num_classes} classes from labels")
    model = ULIPOnlyClassifier(input_dim=input_dim, num_classes=num_classes)
    
    # Train
    checkpoint_dir = args.checkpoint_dir or os.path.join(exp_dir, "checkpoints")
    train_losses, val_accs, test_accs = train_single_feature_model(
        model=model,
        features=train_features,
        labels=train_labels,
        val_features=val_features,
        val_labels=val_labels,
        test_features=test_features,
        test_labels=test_labels,
        epochs=args.epochs,
        lr=args.learning_rate,
        batch_size=args.batch_size,
        checkpoint_dir=checkpoint_dir,
        save_interval=args.save_interval,
        lr_scheduler=None if args.lr_scheduler == 'none' else args.lr_scheduler,
        early_stopping_patience=args.early_stopping_patience,
        save_metrics=os.path.join(exp_dir, "metrics.json") if args.save_metrics else None,
        save_training_curves=os.path.join(exp_dir, "training_curves.npz") if args.save_training_curves else None
    )
    
    # Save results
    final_accuracy = test_accs[-1]
    best_accuracy = max(test_accs)
    
    results = {
        'experiment': 'ulip_only',
        'final_accuracy': float(final_accuracy),
        'best_accuracy': float(best_accuracy),
        'final_loss': float(train_losses[-1]),
        'train_losses': [float(l) for l in train_losses],
        'test_accuracies': [float(a) for a in test_accs],
        'input_dim': input_dim
    }
    
    with open(os.path.join(exp_dir, "results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"ULIP-only: final accuracy = {final_accuracy:.4f}, best = {best_accuracy:.4f}")
    return final_accuracy, best_accuracy


def run_fusion_experiment(train_data: Dict[str, np.ndarray],
                          val_data: Dict[str, np.ndarray],
                          test_data: Dict[str, np.ndarray],
                          exp_dir: str,
                          args: argparse.Namespace,
                          logger: PipelineLogger) -> Tuple[float, float]:
    """Run experiment using both features (fusion)."""
    logger.info("Running fusion experiment...")
    
    # Prepare features
    train_features_pointnet = torch.FloatTensor(train_data['pointnet'])
    train_features_ulip = torch.FloatTensor(train_data['ulip'])
    train_labels = torch.LongTensor(train_data['labels'])
    val_features_pointnet = torch.FloatTensor(val_data['pointnet'])
    val_features_ulip = torch.FloatTensor(val_data['ulip'])
    val_labels = torch.LongTensor(val_data['labels'])
    test_features_pointnet = torch.FloatTensor(test_data['pointnet'])
    test_features_ulip = torch.FloatTensor(test_data['ulip'])
    test_labels = torch.LongTensor(test_data['labels'])
    
    # Create model
    pointnet_dim = train_features_pointnet.shape[1]
    ulip_dim = train_features_ulip.shape[1]
    num_classes = int(max(train_data['labels'].max(), val_data['labels'].max(), test_data['labels'].max())) + 1
    logger.info(f"Detected {num_classes} classes from labels")
    model = FusionModel(pointnet_dim=pointnet_dim, ulip_dim=ulip_dim, num_classes=num_classes)
    
    # Train using the fusion training function
    checkpoint_dir = args.checkpoint_dir or os.path.join(exp_dir, "checkpoints")
    train_losses, val_accs, test_accs = train_fusion_model(
        model=model,
        train_features_pointnet=train_features_pointnet,
        train_features_ulip=train_features_ulip,
        train_labels=train_labels,
        val_features_pointnet=val_features_pointnet,
        val_features_ulip=val_features_ulip,
        val_labels=val_labels,
        test_features_pointnet=test_features_pointnet,
        test_features_ulip=test_features_ulip,
        test_labels=test_labels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        checkpoint_dir=checkpoint_dir,
        save_interval=args.save_interval,
        lr_scheduler=None if args.lr_scheduler == 'none' else args.lr_scheduler,
        early_stopping_patience=args.early_stopping_patience,
        save_metrics=os.path.join(exp_dir, "metrics.json") if args.save_metrics else None,
        save_training_curves=os.path.join(exp_dir, "training_curves.npz") if args.save_training_curves else None
    )
    
    # Save results
    final_accuracy = test_accs[-1]
    best_accuracy = max(test_accs)
    
    results = {
        'experiment': 'fusion',
        'final_accuracy': float(final_accuracy),
        'best_accuracy': float(best_accuracy),
        'final_loss': float(train_losses[-1]),
        'train_losses': [float(l) for l in train_losses],
        'test_accuracies': [float(a) for a in test_accs],
        'pointnet_dim': pointnet_dim,
        'ulip_dim': ulip_dim
    }
    
    with open(os.path.join(exp_dir, "results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Fusion: final accuracy = {final_accuracy:.4f}, best = {best_accuracy:.4f}")
    return final_accuracy, best_accuracy


def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logger
    logger = PipelineLogger(
        name="ablation_study",
        log_file=os.path.join(args.output_dir, "ablation.log")
    )
    
    logger.info("Starting ablation study...")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Train features: {args.train_features}")
    logger.info(f"Test features: {args.test_features}")
    logger.info(f"Epochs: {args.epochs}, Batch size: {args.batch_size}, LR: {args.learning_rate}")
    
    # Set random seed
    set_random_seed(args.seed)
    
    # Load features
    try:
        train_data = load_features(args.train_features)
        val_data = load_features(args.val_features)
        test_data = load_features(args.test_features)
    except Exception as e:
        logger.error(f"Failed to load features: {e}")
        sys.exit(1)
    
    logger.info(f"Train samples: {len(train_data['labels'])}")
    logger.info(f"Validation samples: {len(val_data['labels'])}")
    logger.info(f"Test samples: {len(test_data['labels'])}")
    logger.info(f"PointNet feature dim: {train_data['pointnet'].shape[1]}")
    logger.info(f"ULIP feature dim: {train_data['ulip'].shape[1]}")
    
    # Run experiments
    experiments = [
        ('pointnet_only', run_pointnet_only_experiment),
        ('ulip_only', run_ulip_only_experiment),
        ('fusion', run_fusion_experiment)
    ]
    
    results_summary = {}
    
    for exp_name, exp_func in experiments:
        exp_dir = os.path.join(args.output_dir, exp_name)
        os.makedirs(exp_dir, exist_ok=True)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Running experiment: {exp_name}")
        logger.info(f"{'='*60}")
        
        try:
            final_acc, best_acc = exp_func(train_data, val_data, test_data, exp_dir, args, logger)
            results_summary[exp_name] = {
                'final_accuracy': final_acc,
                'best_accuracy': best_acc,
                'status': 'success'
            }
        except Exception as e:
            logger.error(f"Experiment {exp_name} failed: {e}")
            results_summary[exp_name] = {
                'final_accuracy': 0.0,
                'best_accuracy': 0.0,
                'status': 'failed',
                'error': str(e)
            }
    
    # Save summary
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    logger.info("\n" + "="*60)
    logger.info("Ablation Study Summary")
    logger.info("="*60)
    for exp_name, res in results_summary.items():
        if res['status'] == 'success':
            logger.info(f"{exp_name:15s} | Final Acc: {res['final_accuracy']:.4f} | Best Acc: {res['best_accuracy']:.4f}")
        else:
            logger.info(f"{exp_name:15s} | FAILED: {res.get('error', 'Unknown error')}")
    
    logger.info(f"\nResults saved to: {args.output_dir}")
    logger.info(f"Summary file: {summary_path}")


if __name__ == "__main__":
    main()