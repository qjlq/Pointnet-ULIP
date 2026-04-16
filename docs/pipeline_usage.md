# Pipeline Usage Guide

## Modes of Operation

1. **Full Pipeline**: Extract features + train fusion model
2. **Extract Only**: Only extract features (useful for caching)
3. **Train Only**: Train on pre-extracted features

## Output Structure

Each run creates a timestamped directory:
```
fusion_experiments/fusion_pipeline_YYYYMMDD_HHMMSS/
├── config.json          # Complete configuration
├── command.txt          # Executed command
├── pipeline.log         # Execution logs
├── features/            # Extracted features (.npz)
│   ├── train_features.npz
│   ├── val_features.npz
│   └── test_features.npz
├── checkpoints/         # Training checkpoints
└── training_output/     # Final models and results
```

## Configuration Parameters

### Data Configuration
- `data.root_dir`: Path to ModelNet40 dataset
- `data.num_points`: Number of points per sample (default: 1024)
- `data.normal_channel`: Whether to use normal channels (default: false)

### Model Checkpoints
- `models.pointnet_checkpoint`: Path to pre-trained PointNet2 checkpoint
- `models.ulip_checkpoint`: Path to pre-trained ULIP-2 checkpoint (download via `scripts/download_checkpoints.py`)

### Extraction Parameters
- `extraction.batch_size`: Batch size for feature extraction (default: 64)
- `extraction.device`: Device to use ("cuda" or "cpu")
- `extraction.cache_dir`: Directory for caching extracted features
- `extraction.val_ratio`: Proportion of training data to use for validation (default: 0.2)
- `extraction.split_seed`: Random seed for reproducible train/validation splits (default: 42)

### Training Parameters
- `training.epochs`: Number of training epochs (default: 50)
- `training.batch_size`: Batch size for training (default: 32)
- `training.learning_rate`: Learning rate (default: 0.001)
- `training.checkpoint_dir`: Directory for saving checkpoints
- `training.save_interval`: Save checkpoint every N epochs (default: 10)

### Logging
- `logging.level`: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `logging.format`: Log message format
- `logging.file`: Log file path (relative to experiment directory)

## Advanced Usage

### Custom Configuration
Create a custom YAML configuration file:
```yaml
data:
  root_dir: "path/to/modelnet40"
  num_points: 2048

models:
  pointnet_checkpoint: "path/to/pointnet2.pth"
  ulip_checkpoint: "path/to/ulip2.pth"

extraction:
  batch_size: 32
  device: "cuda"

training:
  epochs: 100
  batch_size: 64
  learning_rate: 0.0005
  save_interval: 5
```

Pass custom config to pipeline:
```bash
python scripts/run_fusion_pipeline.py --config my_config.yaml
```

### Checkpoint Management
Checkpoints are saved in the `checkpoints/` directory:
- `checkpoint_epoch_{N}.pth`: Checkpoint at epoch N
- `best_model.pth`: Best model based on test accuracy

Resume training from checkpoint:
```python
from utils.checkpoint_manager import CheckpointManager

checkpoint_manager = CheckpointManager("checkpoints")
checkpoint = checkpoint_manager.load_checkpoint("checkpoint_epoch_20.pth")
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
```

### Feature Caching
Extracted features are cached as `.npz` files. To force re-extraction:
```bash
python scripts/extract_features.py --skip_if_cached False
```

## Troubleshooting

### CUDA Out of Memory
- Reduce `extraction.batch_size` and `training.batch_size`
- Use `--device cpu` for CPU-only extraction

### Missing Checkpoint Files
- Ensure PointNet2 checkpoint path is correct
- ULIP‑2 checkpoint is included (`ULIP‑2‑PointBERT‑10k‑xyzrgb‑pc‑vit_g‑objaverse_shapenet‑pretrained.pt`); dummy model fallback if missing

### Feature Dimension Mismatch
- Verify that PointNet2 and ULIP-2 feature dimensions match the fusion model
- Default dimensions: PointNet2 (1024), ULIP-2 (1280)

## Performance Tips
- Use `--skip_if_cached` to avoid re-extracting features
- Set `extraction.device` to "cuda" for GPU acceleration
- Adjust `training.save_interval` based on epoch length
- Monitor `pipeline.log` for detailed execution logs

## See Also
- `docs/plans/2026-03-27-fusion-pipeline-integration-design.md` for design details
- `docs/plans/2026-03-27-fusion-pipeline-implementation.md` for implementation plan