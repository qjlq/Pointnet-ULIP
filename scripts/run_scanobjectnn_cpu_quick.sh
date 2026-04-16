#!/bin/bash
# Run ScanObjectNN feature extraction on CPU with limited samples

set -e

# Configuration
DATASET_TYPE="scanobjectnn"
CONFIG_FILE="config/scanobjectnn_quick_config.yaml"
BATCH_SIZE=4  # Even smaller for CPU
DEVICE="cpu"
OUTPUT_DIR="scanobjectnn_feature_cache_cpu_quick"
LOG_FILE="scanobjectnn_extraction_cpu_quick.log"
LIMIT_SAMPLES=100  # Limit samples for quick test

echo "=================================================="
echo "ScanObjectNN Quick Feature Extraction (CPU)"
echo "=================================================="
echo "Dataset type: $DATASET_TYPE"
echo "Config file: $CONFIG_FILE"
echo "Batch size: $BATCH_SIZE"
echo "Device: $DEVICE"
echo "Output directory: $OUTPUT_DIR"
echo "Log file: $LOG_FILE"
echo "Limit samples: $LIMIT_SAMPLES"
echo "=================================================="

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Start time
START_TIME=$(date +%s)

# Run feature extraction with limited samples
echo "Starting CPU quick extraction at $(date)" | tee -a "$LOG_FILE"

# First, create a temporary modified extract_features.py that limits samples
cat > /tmp/extract_features_limited.py << 'EOF'
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from scripts.extract_features import main
import argparse

# Monkey-patch the extraction function to limit samples
original_extract = None

def limited_extract(self, dataset, split_name, output_path, device):
    global original_extract
    if original_extract is None:
        from libs.feature_extractor import FeatureExtractor
        original_extract = FeatureExtractor.extract_features
    
    # Limit dataset
    import torch
    from torch.utils.data import Subset
    limit = 100  # Limit to 100 samples
    if len(dataset) > limit:
        print(f"Limiting dataset from {len(dataset)} to {limit} samples")
        indices = list(range(limit))
        dataset = Subset(dataset, indices)
    
    # Call original extract
    return original_extract(self, dataset, split_name, output_path, device)

# Apply patch
import libs.feature_extractor as fe
fe.FeatureExtractor.extract_features = limited_extract

# Run main
if __name__ == "__main__":
    main()
EOF

python /tmp/extract_features_limited.py \
    --dataset_type "$DATASET_TYPE" \
    --config "$CONFIG_FILE" \
    --batch_size "$BATCH_SIZE" \
    --device "$DEVICE" \
    --output_dir "$OUTPUT_DIR" \
    --split_name "main_split" \
    --use_background \
    2>&1 | tee -a "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

# End time
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo "=================================================="
echo "CPU quick extraction completed at $(date)" | tee -a "$LOG_FILE"
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
    
    print('\nCPU quick extraction successful!')
except Exception as e:
    print(f'Error verifying files: {e}')
    sys.exit(1)
" 2>&1 | tee -a "$LOG_FILE"
        
        echo "Verification completed successfully!" | tee -a "$LOG_FILE"
    else
        echo "ERROR: Output files not found!" | tee -a "$LOG_FILE"
        EXIT_CODE=1
    fi
fi

exit $EXIT_CODE