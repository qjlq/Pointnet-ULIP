#!/bin/bash
# Quick test of ScanObjectNN feature extraction (small subset)

set -e

# Configuration
DATASET_TYPE="scanobjectnn"
CONFIG_FILE="config/scanobjectnn_config.yaml"
BATCH_SIZE=16  # Smaller batch size
DEVICE="cuda"  # Use GPU
OUTPUT_DIR="scanobjectnn_feature_cache_quick"
LOG_FILE="scanobjectnn_extraction_quick.log"

echo "=================================================="
echo "Quick ScanObjectNN Feature Extraction Test"
echo "=================================================="
echo "Dataset type: $DATASET_TYPE"
echo "Config file: $CONFIG_FILE"
echo "Batch size: $BATCH_SIZE"
echo "Device: $DEVICE"
echo "Output directory: $OUTPUT_DIR"
echo "Log file: $LOG_FILE"
echo "=================================================="

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Start time
START_TIME=$(date +%s)

# Create a modified config file with limited batches for testing
QUICK_CONFIG="config/scanobjectnn_quick_config.yaml"
cat > "$QUICK_CONFIG" << EOF
data:
  root_dir: "pointnet_project/Pointnet_Pointnet2_pytorch/data"
  dataset_type: "scanobjectnn"
  split: "main_split"
  use_background: true
  train_split_name: "train"
  test_split_name: "test"
  num_points: 1024  # Use full 1024 points
  normal_channel: false
  num_classes: 15

models:
  pointnet_checkpoint: "pointnet_project/Pointnet_Pointnet2_pytorch/log/classification/pointnet2_ssg_wo_normals/checkpoints/best_model.pth"
  ulip_checkpoint: "checkpoints/ULIP-2-PointBERT-10k-xyzrgb-pc-vit_g-objaverse_shapenet-pretrained.pt"
  openclip_checkpoint: "checkpoints/open_clip_pytorch_model.bin"

extraction:
  batch_size: 16
  device: "cuda"
  cache_dir: "scanobjectnn_feature_cache_quick"
  skip_if_cached: false

# Note: Training config not needed for extraction
EOF

echo "Created quick config: $QUICK_CONFIG"

# Run feature extraction with limited batches
echo "Starting quick extraction at $(date)" | tee -a "$LOG_FILE"

# We'll run a Python script that extracts only first few batches
python -c "
import sys
import os
import torch
import numpy as np
sys.path.insert(0, '.')

from scripts.extract_features import parse_args, validate_cache_file
from project_utils.logger import PipelineLogger
from project_utils.config_parser import ConfigParser
from libs.pointnet_extractor import PointNetExtractor
from libs.ulip_extractor import ULIPExtractor, calculate_safe_batch_size, check_gpu_memory_available
from libs.dataset_loader import get_dataset_loader

def extract_limited():
    args = parse_args(['--dataset_type', 'scanobjectnn',
                      '--config', '$QUICK_CONFIG',
                      '--batch_size', '16',
                      '--device', 'cuda',
                      '--output_dir', '$OUTPUT_DIR',
                      '--split_name', 'main_split',
                      '--use_background'])
    
    logger = PipelineLogger('extract_features_quick')
    logger.info('Starting quick feature extraction (limited batches)...')
    
    config_parser = ConfigParser()
    config = config_parser.load(args.config)
    
    # Use args value if provided, else config value
    data_dir = args.data_dir or config['data']['root_dir']
    pointnet_checkpoint = args.pointnet_checkpoint or config['models']['pointnet_checkpoint']
    ulip_checkpoint = args.ulip_checkpoint or config['models'].get('ulip_checkpoint')
    openclip_checkpoint = args.openclip_checkpoint or config['models'].get('openclip_checkpoint')
    
    batch_size = config['extraction']['batch_size'] if args.batch_size == 64 else args.batch_size
    output_dir = args.output_dir
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f'Using device: {device}')
    
    # Load extractors
    logger.info(f'Loading PointNet2 extractor from {pointnet_checkpoint}...')
    pointnet_extractor = PointNetExtractor(
        checkpoint_path=pointnet_checkpoint,
        device=device
    )
    
    logger.info(f'Loading ULIP-2 extractor from {ulip_checkpoint or \"dummy\"}...')
    ulip_extractor = ULIPExtractor(
        checkpoint_path=ulip_checkpoint,
        openclip_checkpoint=openclip_checkpoint,
        device=device
    )
    
    os.makedirs(output_dir, exist_ok=True)
    
    for split in ['train', 'test']:
        output_path = os.path.join(output_dir, f'{split}_features.npz')
        logger.info(f'Processing {split} split...')
        
        # Prepare loader arguments
        loader_kwargs = {
            'root_dir': data_dir,
            'split': split,
            'num_points': config['data']['num_points'],
            'normal_channel': config['data']['normal_channel'],
            'batch_size': batch_size
        }
        
        # Add dataset-specific parameters for ScanObjectNN
        if args.dataset_type == 'scanobjectnn':
            loader_kwargs.update({
                'split_name': args.split_name,
                'use_background': args.use_background
            })
        
        loader = get_dataset_loader(
            dataset_type=args.dataset_type,
            **loader_kwargs
        )
        
        all_features_pointnet = []
        all_features_ulip = []
        all_labels = []
        
        # Process only first 10 batches for quick test
        max_batches = 10
        batch_count = 0
        
        for batch_idx, (points, labels) in enumerate(loader):
            if batch_count >= max_batches:
                break
                
            try:
                features_pn = pointnet_extractor.extract_features(points)
                features_ulip = ulip_extractor.extract_features(points)
                
                all_features_pointnet.append(features_pn.cpu().numpy())
                all_features_ulip.append(features_ulip.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                
                batch_count += 1
                logger.info(f'Processed batch {batch_idx} ({batch_count}/{max_batches})')
                
                # Clean GPU memory periodically
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                logger.error(f'Batch {batch_idx} extraction failed: {e}')
                raise
        
        if all_features_pointnet:
            features_pointnet = np.concatenate(all_features_pointnet, axis=0)
            features_ulip = np.concatenate(all_features_ulip, axis=0)
            labels = np.concatenate(all_labels, axis=0)
            
            logger.info(f'Saving {split} features: {features_pointnet.shape}, {features_ulip.shape}, {labels.shape}')
            np.savez_compressed(
                output_path,
                pointnet_features=features_pointnet,
                ulip_features=features_ulip,
                labels=labels
            )
            logger.info(f'Saved {split} features to {output_path}')
        
        logger.info(f'Processed {batch_count} batches for {split} split')
    
    logger.info('Quick extraction completed!')
    return True

if __name__ == '__main__':
    try:
        extract_limited()
    except Exception as e:
        print(f'Error: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)
" 2>&1 | tee -a "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

# End time
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo "=================================================="
echo "Quick extraction completed at $(date)" | tee -a "$LOG_FILE"
echo "Duration: $((DURATION / 3600))h $(( (DURATION % 3600) / 60 ))m $((DURATION % 60))s" | tee -a "$LOG_FILE"
echo "Exit code: $EXIT_CODE" | tee -a "$LOG_FILE"
echo "=================================================="

# Verify output files
if [ $EXIT_CODE -eq 0 ]; then
    echo "Verifying output files..." | tee -a "$LOG_FILE"
    
    TRAIN_FILE="$OUTPUT_DIR/train_features.npz"
    TEST_FILE="$OUTPUT_DIR/test_features.npz"
    
    if [ -f "$TRAIN_FILE" ] && [ -f "$TEST_FILE" ]; then
        echo "Output files exist: $TRAIN_FILE, $TEST_FILE" | tee -a "$LOG_FILE"
        
        python -c "
import numpy as np
import sys

try:
    print('Verifying train features...')
    train_data = np.load('$TRAIN_FILE')
    print(f'Train samples: {len(train_data[\"labels\"])}')
    print(f'PointNet features: {train_data[\"pointnet_features\"].shape}')
    print(f'ULIP features: {train_data[\"ulip_features\"].shape}')
    print(f'Unique labels: {np.unique(train_data[\"labels\"])}')
    
    print('\nVerifying test features...')
    test_data = np.load('$TEST_FILE')
    print(f'Test samples: {len(test_data[\"labels\"])}')
    print(f'PointNet features: {test_data[\"pointnet_features\"].shape}')
    print(f'ULIP features: {test_data[\"ulip_features\"].shape}')
    print(f'Unique labels: {np.unique(test_data[\"labels\"])}')
    
    print('\nQuick extraction successful!')
    print(f'Note: This is a quick test with limited batches.')
    print(f'For full extraction, run: ./scripts/run_scanobjectnn_extraction.sh')
except Exception as e:
    print(f'Error verifying files: {e}')
    sys.exit(1)
" 2>&1 | tee -a "$LOG_FILE"
    fi
fi

exit $EXIT_CODE