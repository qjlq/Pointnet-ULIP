#!/usr/bin/env python3
"""Comprehensive evaluation of fusion methods with statistical robustness."""
import argparse
import json
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import time
import csv
import random
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from train_utils import train_fusion_model


def split_data_indices(n_samples: int, 
                       train_ratio: float = 0.8, 
                       val_ratio: float = 0.1,
                       seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split dataset indices into train, validation, and test sets.
    
    Args:
        n_samples: Total number of samples
        train_ratio: Proportion of samples for training (default: 0.8)
        val_ratio: Proportion of samples for validation (default: 0.1)
        seed: Random seed for reproducibility (default: None)
        
    Returns:
        Tuple of (train_indices, val_indices, test_indices) as numpy arrays
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Calculate sizes
    n_train = int(np.round(train_ratio * n_samples))
    n_val = int(np.round(val_ratio * n_samples))
    n_test = n_samples - n_train - n_val
    
    # Ensure non-negative test size (handle rounding errors)
    if n_test < 0:
        # Adjust train size to maintain ratios as close as possible
        n_train = int(np.floor(train_ratio * n_samples))
        n_val = int(np.floor(val_ratio * n_samples))
        n_test = n_samples - n_train - n_val
    
    # Generate random permutation of indices
    indices = np.random.permutation(n_samples)
    
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    
    return train_idx, val_idx, test_idx


def load_features(feature_path: str) -> Dict[str, np.ndarray]:
    """Load features from .npz file (compatible with existing evaluation).
    
    Args:
        feature_path: Path to .npz file
        
    Returns:
        Dictionary with keys 'pointnet', 'ulip', 'labels'
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


def split_features(features: Dict[str, np.ndarray], 
                   train_idx: np.ndarray, 
                   val_idx: np.ndarray,
                   test_idx: np.ndarray) -> Tuple[Dict, Dict, Dict]:
    """Split features into train, validation, and test sets.
    
    Args:
        features: Dictionary with keys 'pointnet', 'ulip', 'labels'
        train_idx: Training indices
        val_idx: Validation indices
        test_idx: Test indices
        
    Returns:
        Tuple of (train_features, val_features, test_features) each with same keys
    """
    train_features = {}
    val_features = {}
    test_features = {}
    
    for key, array in features.items():
        train_features[key] = array[train_idx]
        val_features[key] = array[val_idx]
        test_features[key] = array[test_idx]
    
    return train_features, val_features, test_features


def measure_time() -> None:
    """Context manager to measure execution time.
    
    Usage:
        with measure_time() as timer:
            # code
        elapsed = timer.elapsed
    """
    class Timer:
        def __enter__(self):
            self.start = time.perf_counter()
            return self
        
        def __exit__(self, *args):
            self.end = time.perf_counter()
            self.elapsed = self.end - self.start
    
    return Timer()


def measure_gpu_memory() -> Optional[int]:
    """Measure current GPU memory allocated (if CUDA available).
    
    Returns:
        Memory allocated in bytes, or None if CUDA not available.
    """
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated()
    return None


def measure_inference_speed(model: nn.Module, 
                           pointnet_features: torch.Tensor,
                           ulip_features: torch.Tensor,
                           num_runs: int = 100) -> float:
    """Measure inference speed in samples per second.
    
    Args:
        model: Fusion model
        pointnet_features: Geometric features tensor (N, geo_dim)
        ulip_features: Semantic features tensor (N, ulip_dim)
        num_runs: Number of forward passes to average
        
    Returns:
        Samples per second (higher is faster)
    """
    device = next(model.parameters()).device
    model.eval()
    
    # Warm-up
    with torch.no_grad():
        _ = model(pointnet_features.to(device), ulip_features.to(device))
    
    # Measure time for num_runs forward passes
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(pointnet_features.to(device), ulip_features.to(device))
    end = time.perf_counter()
    
    total_samples = pointnet_features.shape[0] * num_runs
    seconds = end - start
    return total_samples / seconds


@dataclass
class TrainingMetrics:
    """Container for training metrics."""
    accuracy: float
    training_time: float  # seconds
    inference_speed: float  # samples per second
    memory_allocated: Optional[float]  # bytes, None if CUDA not available
    per_epoch_times: List[float]
    train_losses: List[float]
    test_accuracies: List[float]
    seed: int


def evaluate_with_seed(dataset_name: str,
                       fusion_type: str,
                       seed: int,
                       output_dir: str,
                       epochs: int = 5,
                       lr_scheduler: Optional[str] = None,
                       early_stopping_patience: Optional[int] = None,
                       save_metrics: bool = False,
                       save_training_curves: bool = False) -> TrainingMetrics:
    """Evaluate a fusion type on a dataset with a specific random seed.
    
    Args:
        dataset_name: Name of dataset ('modelnet40' or 'scanobjectnn')
        fusion_type: Type of fusion ('concat', 'normalized', 'gated')
        seed: Random seed for reproducibility
        output_dir: Directory for output files
        epochs: Number of training epochs
        lr_scheduler: Learning rate scheduler ('cosine', 'plateau', or None)
        early_stopping_patience: Patience for early stopping (None to disable)
        save_metrics: If True, save comprehensive metrics to JSON file
        save_training_curves: If True, save training curves to NPZ file
        
    Returns:
        TrainingMetrics object with performance metrics
    """

    # Set random seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
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
    checkpoint_dir = os.path.join(output_dir, f"checkpoints_{dataset_name}_{fusion_type}_seed{seed}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Prepare paths for metrics and curves if requested
    metrics_path = None
    curves_path = None
    if save_metrics:
        metrics_path = os.path.join(checkpoint_dir, 'metrics.json')
    if save_training_curves:
        curves_path = os.path.join(checkpoint_dir, 'training_curves.npz')
    
    # Reset GPU memory stats before training
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    # Collect per-epoch timings
    per_epoch_times = []
    start_time = time.perf_counter()
    epoch_start_time = start_time
    
    def epoch_callback(epoch_idx, train_loss, test_accuracy, learning_rate):
        nonlocal epoch_start_time
        epoch_end_time = time.perf_counter()
        epoch_duration = epoch_end_time - epoch_start_time
        per_epoch_times.append(epoch_duration)
        epoch_start_time = epoch_end_time
    
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
        save_training_curves=curves_path,
        epoch_callback=epoch_callback
    )
    
    # Record final epoch time if needed (callback may have missed last epoch)
    if len(per_epoch_times) < len(train_losses):
        # Estimate last epoch time as time since last epoch_start_time
        final_time = time.perf_counter() - epoch_start_time
        per_epoch_times.append(final_time)
    
    total_training_time = time.perf_counter() - start_time
    
    # Measure inference speed on test set
    inference_speed = measure_inference_speed(
        model,
        torch.FloatTensor(test_data['pointnet']),
        torch.FloatTensor(test_data['ulip'])
    )
    
    # Measure GPU memory allocated
    memory_allocated = measure_gpu_memory()  # returns None if CUDA not available
    
    # Final accuracy is the last test accuracy
    final_accuracy = test_accuracies[-1]
    
    return TrainingMetrics(
        accuracy=final_accuracy,
        training_time=total_training_time,
        inference_speed=inference_speed,
        memory_allocated=memory_allocated,
        per_epoch_times=per_epoch_times,
        train_losses=train_losses,
        test_accuracies=test_accuracies,
        seed=seed
    )


def aggregate_results(results_per_seed: List[TrainingMetrics]) -> Dict[str, Any]:
    """Aggregate results across multiple random seeds.
    
    Computes mean and standard deviation for each metric.
    
    Args:
        results_per_seed: List of TrainingMetrics objects
        
    Returns:
        Dictionary with aggregated statistics
    """
    import numpy as np
    if not results_per_seed:
        return {}
    
    # Convert TrainingMetrics objects to dictionaries
    dicts = []
    for metrics in results_per_seed:
        d = {
            'accuracy': metrics.accuracy,
            'training_time': metrics.training_time,
            'inference_speed': metrics.inference_speed,
            'memory_allocated': metrics.memory_allocated,
            'per_epoch_times': metrics.per_epoch_times,
            'train_losses': metrics.train_losses,
            'test_accuracies': metrics.test_accuracies,
            'seed': metrics.seed
        }
        dicts.append(d)
    
    # Compute mean and std for numeric metrics
    aggregated = {
        'num_seeds': len(results_per_seed),
        'results_per_seed': dicts
    }
    
    # Metrics to aggregate (numeric)
    numeric_keys = ['accuracy', 'training_time', 'inference_speed']
    for key in numeric_keys:
        values = [d[key] for d in dicts if d[key] is not None]
        if values:
            aggregated[f'{key}_mean'] = float(np.mean(values))
            aggregated[f'{key}_std'] = float(np.std(values))
        else:
            aggregated[f'{key}_mean'] = None
            aggregated[f'{key}_std'] = None
    
    # Memory allocated may be None for CPU runs
    memory_values = [d['memory_allocated'] for d in dicts if d['memory_allocated'] is not None]
    if memory_values:
        aggregated['memory_allocated_mean'] = float(np.mean(memory_values))
        aggregated['memory_allocated_std'] = float(np.std(memory_values))
    else:
        aggregated['memory_allocated_mean'] = None
        aggregated['memory_allocated_std'] = None
    
    # Per epoch times aggregation (mean per epoch across seeds)
    epoch_times_list = [d['per_epoch_times'] for d in dicts]
    if epoch_times_list and all(len(t) == len(epoch_times_list[0]) for t in epoch_times_list):
        # Same length across seeds: compute mean per epoch
        mean_per_epoch = np.mean(epoch_times_list, axis=0).tolist()
        aggregated['mean_per_epoch_times'] = mean_per_epoch
    
    return aggregated


def generate_json_table(aggregated_results: List[Dict[str, Any]]) -> str:
    """Generate JSON representation of comparison table.
    
    Args:
        aggregated_results: List of aggregated results for each (dataset, fusion_type)
        
    Returns:
        JSON string
    """
    import json
    return json.dumps(aggregated_results, indent=2)


def generate_csv_table(aggregated_results: List[Dict[str, Any]], output_path: str) -> None:
    """Generate CSV file with comparison table.
    
    Args:
        aggregated_results: List of aggregated results
        output_path: Path to output CSV file
    """
    import csv
    # Define column order
    columns = [
        'dataset', 'fusion_type', 'num_seeds',
        'accuracy_mean', 'accuracy_std',
        'training_time_mean', 'training_time_std',
        'inference_speed_mean', 'inference_speed_std',
        'memory_allocated_mean', 'memory_allocated_std'
    ]
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        
        for result in aggregated_results:
            row = {col: result.get(col, '') for col in columns}
            writer.writerow(row)


def generate_markdown_table(aggregated_results: List[Dict[str, Any]]) -> str:
    """Generate Markdown representation of comparison table.
    
    Args:
        aggregated_results: List of aggregated results
        
    Returns:
        Markdown table string
    """
    # Define column order and headers
    columns = [
        ('dataset', 'Dataset'),
        ('fusion_type', 'Fusion Type'),
        ('num_seeds', 'Seeds'),
        ('accuracy_mean', 'Accuracy Mean'),
        ('accuracy_std', 'Accuracy Std'),
        ('training_time_mean', 'Train Time Mean (s)'),
        ('training_time_std', 'Train Time Std'),
        ('inference_speed_mean', 'Inference Speed Mean (samples/s)'),
        ('inference_speed_std', 'Inference Speed Std'),
        ('memory_allocated_mean', 'Memory Mean (bytes)'),
        ('memory_allocated_std', 'Memory Std')
    ]
    
    # Build header row
    header = '| ' + ' | '.join([col[1] for col in columns]) + ' |'
    separator = '| ' + ' | '.join(['---' for _ in columns]) + ' |'
    
    rows = []
    for result in aggregated_results:
        row_cells = []
        for col_key, _ in columns:
            val = result.get(col_key, '')
            if isinstance(val, float):
                # Format floats nicely
                if 'accuracy' in col_key:
                    row_cells.append(f'{val:.4f}')
                elif 'time' in col_key or 'speed' in col_key:
                    row_cells.append(f'{val:.2f}')
                elif 'memory' in col_key:
                    if val is None:
                        row_cells.append('N/A')
                    else:
                        # Convert bytes to MB for readability
                        row_cells.append(f'{val / 1024**2:.2f} MB')
                else:
                    row_cells.append(f'{val:.2f}')
            else:
                row_cells.append(str(val) if val is not None else 'N/A')
        rows.append('| ' + ' | '.join(row_cells) + ' |')
    
    # Combine
    table_lines = [header, separator] + rows
    return '\n'.join(table_lines)


def main():
    """Main function for comprehensive evaluation."""
    parser = argparse.ArgumentParser(
        description="Comprehensive evaluation of fusion methods with statistical robustness."
    )
    parser.add_argument("--output_dir", default="comprehensive_results",
                       help="Output directory for results (default: comprehensive_results)")
    parser.add_argument("--epochs", type=int, default=5,
                       help="Number of epochs per evaluation (for speed)")
    parser.add_argument("--seeds", type=str, default="42,43,44",
                       help="Comma-separated random seeds (default: 42,43,44)")
    parser.add_argument("--datasets", type=str, default="modelnet40,scanobjectnn",
                       help="Comma-separated dataset names (default: modelnet40,scanobjectnn)")
    parser.add_argument("--fusion_types", type=str, default="concat,normalized,gated",
                       help="Comma-separated fusion types (default: concat,normalized,gated)")
    parser.add_argument("--lr_scheduler", choices=['cosine', 'plateau', 'none'], default='none',
                       help="Learning rate scheduler (cosine, plateau, or none)")
    parser.add_argument("--early_stopping_patience", type=int,
                       help="Patience for early stopping (disabled if not specified)")
    parser.add_argument("--save_metrics", action='store_true',
                       help="Save comprehensive metrics per seed")
    parser.add_argument("--save_training_curves", action='store_true',
                       help="Save training curves per seed")
    parser.add_argument("--formats", type=str, default="json,csv,md",
                       help="Output table formats (json,csv,md) comma-separated")
    
    args = parser.parse_args()
    
    # Parse comma-separated lists
    seeds = [int(s.strip()) for s in args.seeds.split(',')]
    datasets = [d.strip() for d in args.datasets.split(',')]
    fusion_types = [ft.strip() for ft in args.fusion_types.split(',')]
    formats = [f.strip().lower() for f in args.formats.split(',')]
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Dictionary to collect aggregated results per (dataset, fusion_type)
    all_aggregated = []
    
    for dataset in datasets:
        for fusion_type in fusion_types:
            print(f"\n{'='*60}")
            print(f"Evaluating {dataset} with {fusion_type} fusion")
            print(f"Seeds: {seeds}")
            print(f"{'='*60}")
            
            results_per_seed = []
            for seed in seeds:
                print(f"\nSeed {seed}:")
                try:
                    metrics = evaluate_with_seed(
                        dataset_name=dataset,
                        fusion_type=fusion_type,
                        seed=seed,
                        output_dir=args.output_dir,
                        epochs=args.epochs,
                        lr_scheduler=None if args.lr_scheduler == 'none' else args.lr_scheduler,
                        early_stopping_patience=args.early_stopping_patience,
                        save_metrics=args.save_metrics,
                        save_training_curves=args.save_training_curves
                    )
                    results_per_seed.append(metrics)
                    print(f"  Accuracy: {metrics.accuracy:.4f}, Time: {metrics.training_time:.2f}s")
                except Exception as e:
                    print(f"  ERROR: {e}")
                    import traceback
                    traceback.print_exc()
                    # Continue with other seeds
            
            if not results_per_seed:
                print(f"  No successful runs for {dataset} {fusion_type}, skipping.")
                continue
            
            # Aggregate results across seeds
            aggregated = aggregate_results(results_per_seed)
            aggregated['dataset'] = dataset
            aggregated['fusion_type'] = fusion_type
            all_aggregated.append(aggregated)
            
            # Print summary for this combination
            print(f"\nSummary for {dataset} {fusion_type}:")
            print(f"  Accuracy: {aggregated.get('accuracy_mean', 0):.4f} ± {aggregated.get('accuracy_std', 0):.4f}")
            print(f"  Training time: {aggregated.get('training_time_mean', 0):.2f} ± {aggregated.get('training_time_std', 0):.2f}s")
            if aggregated.get('memory_allocated_mean'):
                print(f"  GPU memory: {aggregated['memory_allocated_mean'] / 1024**2:.2f} ± {aggregated['memory_allocated_std'] / 1024**2:.2f} MB")
    
    if not all_aggregated:
        print("No successful evaluations. Exiting.")
        return
    
    # Generate output tables in requested formats
    output_base = os.path.join(args.output_dir, "comparison_table")
    
    if 'json' in formats:
        json_str = generate_json_table(all_aggregated)
        json_path = output_base + ".json"
        with open(json_path, 'w') as f:
            f.write(json_str)
        print(f"\nJSON table saved to {json_path}")
    
    if 'csv' in formats:
        csv_path = output_base + ".csv"
        generate_csv_table(all_aggregated, csv_path)
        print(f"CSV table saved to {csv_path}")
    
    if 'md' in formats:
        md_str = generate_markdown_table(all_aggregated)
        md_path = output_base + ".md"
        with open(md_path, 'w') as f:
            f.write(md_str)
        print(f"Markdown table saved to {md_path}")
    
    print("\nComprehensive evaluation complete!")


if __name__ == "__main__":
    main()