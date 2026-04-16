#!/usr/bin/env python
# scripts/extract_features.py
import os
import sys
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from project_utils.logger import PipelineLogger
from project_utils.config_parser import ConfigParser
from libs.pointnet_extractor import PointNetExtractor
from libs.ulip_extractor import ULIPExtractor, calculate_safe_batch_size, check_gpu_memory_available
from libs.dataset_loader import get_dataset_loader

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Extract features from PointNet2 and ULIP-2")
    parser.add_argument("--config", type=str, default="config/fusion_config.yaml", help="Path to config file")
    parser.add_argument("--data_dir", type=str, help="Dataset root directory")
    parser.add_argument("--dataset_type", type=str, default="modelnet40", 
                        choices=["modelnet40", "scanobjectnn"], help="Dataset type")
    parser.add_argument("--pointnet_checkpoint", type=str, help="PointNet2 checkpoint path")
    parser.add_argument("--ulip_checkpoint", type=str, help="ULIP-2 checkpoint path")
    parser.add_argument("--openclip_checkpoint", type=str, help="OpenCLIP checkpoint path (local .bin file)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for extraction")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda/cpu)")
    parser.add_argument("--output_dir", type=str, default="feature_cache", help="Output directory for features")
    parser.add_argument("--skip_if_cached", action="store_true", help="Skip extraction if cached files exist")
    parser.add_argument("--val_ratio", type=float, default=0.2, 
                       help="Ratio of training data to use for validation")
    parser.add_argument("--split_seed", type=int, default=42,
                       help="Random seed for train/val split")
    # ScanObjectNN specific parameters
    parser.add_argument("--split_name", type=str, default="main_split", 
                        help="ScanObjectNN split name (e.g., main_split, split1, etc.)")
    parser.add_argument("--use_background", action="store_true", default=True,
                        help="Use background points for ScanObjectNN (default: True)")
    parser.add_argument("--no_background", action="store_false", dest="use_background",
                        help="Exclude background points for ScanObjectNN")
    parser.add_argument("--limit_samples", type=int, default=None,
                        help="Limit number of samples per split (for testing)")
    return parser.parse_args(argv)

def validate_cache_file(file_path):
    """Validate cached feature file has required keys and can be loaded.
    
    Returns:
        True if file exists, can be loaded, and contains required keys.
        False otherwise.
    """
    try:
        with np.load(file_path, allow_pickle=True) as data:
            required_keys = {'pointnet_features', 'ulip_features', 'labels'}
            if not all(key in data for key in required_keys):
                return False
            # Quick shape check (optional but helpful)
            if (data['pointnet_features'].ndim != 2 or 
                data['ulip_features'].ndim != 2 or 
                data['labels'].ndim != 1):
                return False
            # Shape consistency: all arrays must have same number of samples
            num_samples = data['pointnet_features'].shape[0]
            if (data['ulip_features'].shape[0] != num_samples or 
                data['labels'].shape[0] != num_samples):
                return False
            return True
    except Exception:
        return False

def main():
    """Main feature extraction workflow.
    
    Orchestrates loading of configuration, models, dataset, and extraction
    of features from PointNet2 and ULIP-2 models for ModelNet40 dataset.
    Saves extracted features as compressed numpy arrays for later fusion.
    """
    args = parse_args()
    log_file = os.path.join(args.output_dir, "extraction.log")
    logger = PipelineLogger("extract_features", log_file=log_file)
    
    try:
        logger.info("Starting feature extraction...")
        
        config_parser = ConfigParser()
        config = config_parser.load(args.config)
        
        # Use args value if provided, else config value (following spec pattern)
        data_dir = args.data_dir or config['data']['root_dir']
        pointnet_checkpoint = args.pointnet_checkpoint or config['models']['pointnet_checkpoint']
        ulip_checkpoint = args.ulip_checkpoint or config['models'].get('ulip_checkpoint')
        openclip_checkpoint = args.openclip_checkpoint or config['models'].get('openclip_checkpoint')
        
        # Use config value if using argparse default, otherwise use provided arg value
        batch_size = config['extraction']['batch_size'] if args.batch_size == 64 else args.batch_size
        skip_if_cached = config['extraction'].get('skip_if_cached', False) if not args.skip_if_cached else True
        output_dir = args.output_dir
        
        # Device selection: use config value if using argparse default, otherwise use provided arg
        # Still respect CUDA availability
        if args.device == "cuda":  # Using argparse default
            device_from_config = config['extraction']['device']
        else:
            device_from_config = args.device
        
        # CUDA availability fallback
        if device_from_config == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            device = "cpu"
        else:
            device = device_from_config
        
        logger.info(f"Loading PointNet2 extractor from {pointnet_checkpoint}...")
        pointnet_extractor = PointNetExtractor(
            checkpoint_path=pointnet_checkpoint,
            device=device
        )
        
        logger.info(f"Loading ULIP-2 extractor from {ulip_checkpoint or 'dummy'}...")
        ulip_extractor = ULIPExtractor(
            checkpoint_path=ulip_checkpoint,
            openclip_checkpoint=openclip_checkpoint,
            device=device
        )
        
        # Calculate safe batch size for logging
        num_points = config['data']['num_points']
        channels = 6 if config['data']['normal_channel'] else 3
        points_shape = (batch_size, num_points, channels)
        safe_batch_size = calculate_safe_batch_size(points_shape, device=device)
        
        logger.info("=" * 60)
        logger.info("Feature Extraction Configuration")
        logger.info("=" * 60)
        logger.info(f"Dataset: {args.dataset_type}")
        logger.info(f"Config file: {args.config}")
        logger.info(f"PointNet checkpoint: {pointnet_checkpoint}")
        logger.info(f"ULIP checkpoint: {ulip_checkpoint or 'dummy (random init)'}")
        logger.info(f"OpenCLIP checkpoint: {openclip_checkpoint or 'not specified'}")
        logger.info(f"Device: {device}")
        logger.info(f"Batch size: {batch_size}")
        logger.info(f"Safe batch size: {safe_batch_size}")
        logger.info(f"Validation ratio: {args.val_ratio}")
        logger.info(f"Split seed: {args.split_seed}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Skip if cached: {skip_if_cached}")
        logger.info("=" * 60)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Determine splits based on validation ratio
        if args.val_ratio == 0:
            splits = ["train", "test"]
            logger.info("Validation ratio is 0, skipping validation split")
        else:
            splits = ["train", "val", "test"]
            logger.info(f"Including validation split (ratio: {args.val_ratio})")
        
        for split in splits:
            output_path = os.path.join(output_dir, f"{split}_features.npz")
            if skip_if_cached and os.path.exists(output_path):
                if validate_cache_file(output_path):
                    logger.info(f"Skipping {split} split (valid cache exists: {output_path})")
                    continue
                else:
                    logger.warning(f"Cached file {output_path} corrupted or incomplete, re-extracting...")
            
            logger.info(f"Processing {split} split...")
            
            # Use the smaller of configured batch size and safe batch size
            final_batch_size = min(batch_size, safe_batch_size)
            if final_batch_size != batch_size:
                logger.warning(
                    f"GPU memory limited: reducing batch size from {batch_size} to {final_batch_size} "
                    f"(safe batch size: {safe_batch_size})"
                )
            else:
                logger.info(f"Using batch size {final_batch_size} (safe batch size: {safe_batch_size})")
            
            # Prepare loader arguments
            loader_kwargs = {
                'root_dir': data_dir,
                'split': split,
                'num_points': num_points,
                'normal_channel': config['data']['normal_channel'],
                'batch_size': final_batch_size
            }
            
            # Add dataset-specific parameters for ScanObjectNN
            if args.dataset_type == "scanobjectnn":
                loader_kwargs.update({
                    'split_name': args.split_name,
                    'use_background': args.use_background
                })
            
            loader = get_dataset_loader(
                dataset_type=args.dataset_type,
                val_ratio=args.val_ratio,
                seed=args.split_seed,
                **loader_kwargs
            )
            logger.info(f"  Split: {split}")
            logger.info(f"  Batch size: {final_batch_size}")
            logger.info(f"  Number of batches: {len(loader)}")
            logger.info(f"  Estimated samples: {len(loader) * final_batch_size}")
            
            all_features_pointnet = []
            all_features_ulip = []
            all_labels = []
            
            for batch_idx, (points, labels) in enumerate(loader):
                try:
                    # Memory safety check before processing batch
                    if torch.cuda.is_available():
                        total, allocated, free, free_pct = check_gpu_memory_available('cuda')
                        if free_pct is not None and free_pct < 15.0:  # Less than 15% free memory
                            logger.warning(
                                f"Low GPU memory before batch {batch_idx}: {free_pct:.1f}% free "
                                f"({free/1024**3:.2f} GB). Cleaning cache."
                            )
                            torch.cuda.empty_cache()
                    
                    features_pn = pointnet_extractor.extract_features(points)
                    features_ulip = ulip_extractor.extract_features(points)
                    
                    all_features_pointnet.append(features_pn.numpy())
                    all_features_ulip.append(features_ulip.numpy())
                    all_labels.append(labels.numpy())
                    
                    if batch_idx % 10 == 0:
                        logger.debug(f"Processed {batch_idx} batches for {split} split")
                        # Clean up CUDA cache periodically to prevent memory accumulation
                        if torch.cuda.is_available():
                            allocated_before = torch.cuda.memory_allocated() / 1024**2
                            torch.cuda.empty_cache()
                            allocated_after = torch.cuda.memory_allocated() / 1024**2
                            logger.debug(f"GPU memory: {allocated_before:.1f}MB -> {allocated_after:.1f}MB after cache cleanup")
                        
                except torch.cuda.OutOfMemoryError as oom:
                    logger.error(f"CUDA out of memory in batch {batch_idx}: {oom}")
                    if torch.cuda.is_available():
                        logger.error(f"Attempting recovery: clearing cache and retrying batch")
                        torch.cuda.empty_cache()
                    # Re-raise to fail fast, but could implement retry logic here
                    raise
                except Exception as e:
                    logger.error(f"Batch {batch_idx} extraction failed: {e}")
                    raise
            
            logger.info(f"Processed all {len(loader)} batches for {split} split")
            features_pointnet = np.concatenate(all_features_pointnet, axis=0)
            features_ulip = np.concatenate(all_features_ulip, axis=0)
            labels = np.concatenate(all_labels, axis=0)
            
            logger.info(f"Saving {split} features: {features_pointnet.shape}, {features_ulip.shape}, {labels.shape}")
            np.savez_compressed(
                output_path,
                pointnet_features=features_pointnet,
                ulip_features=features_ulip,
                labels=labels
            )
            logger.info(f"Saved {split} features: {output_path}")
            logger.info(f"  PointNet features: {features_pointnet.shape}")
            logger.info(f"  ULIP features: {features_ulip.shape}")
            logger.info(f"  Labels: {labels.shape}")
            logger.info(f"  Number of samples: {labels.shape[0]}")
            # Quick verification that file can be loaded
            if validate_cache_file(output_path):
                logger.debug(f"Saved file validation passed for {output_path}")
            else:
                logger.warning(f"Saved file validation failed for {output_path}, file may be corrupted")
        
        logger.info("Feature extraction completed!")
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        sys.exit(1)
    except (IOError, OSError) as e:
        logger.error(f"I/O error: {e}")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"Invalid value: {e}")
        sys.exit(1)
    except RuntimeError as e:
        logger.error(f"Runtime error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Feature extraction interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()