#!/bin/bash
# Quick training test for ScanObjectNN fusion model (5 epochs)

set -e

# Configuration
CONFIG_FILE="config/scanobjectnn_config.yaml"
TRAIN_FEATURES="scanobjectnn_feature_cache_quick/train_features.npz"
TEST_FEATURES="scanobjectnn_feature_cache_quick/test_features.npz"
EPOCHS=5
BATCH_SIZE=32
LEARNING_RATE=0.001
CHECKPOINT_DIR="scanobjectnn_checkpoints_quick"
OUTPUT_DIR="scanobjectnn_training_output_quick"
LOG_FILE="scanobjectnn_training_quick.log"

echo "=================================================="
echo "ScanObjectNN Fusion Model Quick Training Test"
echo "=================================================="
echo "Config file: $CONFIG_FILE"
echo "Train features: $TRAIN_FEATURES"
echo "Test features: $TEST_FEATURES"
echo "Epochs: $EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "Learning rate: $LEARNING_RATE"
echo "Checkpoint directory: $CHECKPOINT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Log file: $LOG_FILE"
echo "=================================================="

# Create directories
mkdir -p "$CHECKPOINT_DIR"
mkdir -p "$OUTPUT_DIR"

# Start time
START_TIME=$(date +%s)

# Run training
echo "Starting quick training at $(date)" | tee -a "$LOG_FILE"

python scripts/train_fusion.py \
    --config "$CONFIG_FILE" \
    --train_features "$TRAIN_FEATURES" \
    --test_features "$TEST_FEATURES" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --learning_rate "$LEARNING_RATE" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    2>&1 | tee -a "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

# End time
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo "=================================================="
echo "Quick training completed at $(date)" | tee -a "$LOG_FILE"
echo "Duration: $((DURATION / 3600))h $(( (DURATION % 3600) / 60 ))m $((DURATION % 60))s" | tee -a "$LOG_FILE"
echo "Exit code: $EXIT_CODE" | tee -a "$LOG_FILE"
echo "=================================================="

# Check for results
if [ $EXIT_CODE -eq 0 ]; then
    echo "Checking training results..." | tee -a "$LOG_FILE"
    
    # Look for final metrics
    if [ -f "$LOG_FILE" ]; then
        echo "Extracting final metrics from log..." | tee -a "$LOG_FILE"
        grep -A5 -B5 "Test Accuracy\|Best\|Epoch.*5" "$LOG_FILE" | tail -20
    fi
    
    # Check for checkpoint files
    CHECKPOINT_COUNT=$(find "$CHECKPOINT_DIR" -name "*.pth" -type f | wc -l)
    echo "Number of checkpoints: $CHECKPOINT_COUNT" | tee -a "$LOG_FILE"
    
    if [ $CHECKPOINT_COUNT -gt 0 ]; then
        echo "Checkpoints created:" | tee -a "$LOG_FILE"
        find "$CHECKPOINT_DIR" -name "*.pth" -type f | tee -a "$LOG_FILE"
    fi
    
    echo "Quick training completed successfully!" | tee -a "$LOG_FILE"
else
    echo "Quick training failed with exit code $EXIT_CODE" | tee -a "$LOG_FILE"
fi

exit $EXIT_CODE