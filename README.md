# ULIP-2 + PointNet2 Fusion Pipeline

A two-stage feature fusion pipeline for point cloud classification.

## Results

Real ULIP‑2 integration achieved **93.19%** accuracy on ModelNet40 test set, a **+0.89%** improvement over the PointNet‑only baseline (92.30%). The pipeline now uses the full pre‑trained ULIP‑2 checkpoint (`ULIP‑2‑PointBERT‑10k‑xyzrgb‑pc‑vit_g‑objaverse_shapenet‑pretrained.pt`) for 1280‑D semantic features, fused with 1024‑D PointNet2 geometric features.

## Advanced Fusion Mechanisms

The pipeline now supports multiple fusion mechanisms beyond simple concatenation:

- **concat**: Original feature concatenation (baseline)
- **normalized**: Features normalized before fusion (L2 or layer normalization)
- **gated**: Learnable gating weights for adaptive feature combination

Use the `--fusion_type` parameter in `train_fusion.py` to select mechanism:

```bash
python scripts/train_fusion.py --fusion_type normalized ...
```

See [docs/IMPROVEMENT_RESULTS.md](docs/IMPROVEMENT_RESULTS.md) for detailed evaluation results.

## Setup

For detailed setup instructions, including CUDA compatibility and conda environment configuration, see [docs/SETUP_GUIDE.md](docs/SETUP_GUIDE.md).

The pipeline has been verified with:
- CUDA 13 driver compatibility (PyTorch 1.10.1 with CUDA 12.1 toolkit)
- Pre-trained PointNet2 checkpoint (geometric features)
- Real ULIP‑2 checkpoint (`ULIP‑2‑PointBERT‑10k‑xyzrgb‑pc‑vit_g‑objaverse_shapenet‑pretrained.pt`) for 1280‑D semantic features
- ModelNet40 dataset (provided in pointnet_project/)

Run verification tests:
```bash
python tests/verification/verify_pointnet_extraction.py
python tests/verification/verify_full_extraction.py
python tests/verification/verify_fusion_training.py
```

## Quick Start

1. Install dependencies (recommended: use conda environment):

Using conda (recommended for CUDA compatibility):
```bash
conda env create -f environment.yml
conda activate fusion-pipeline
```

Or using pip:
```bash
pip install torch numpy scipy scikit-learn tqdm pyyaml matplotlib
```

2. Run end-to-end pipeline:
```bash
python scripts/run_fusion_pipeline.py \
  --data_dir "pointnet_project/Pointnet_Pointnet2_pytorch/data/modelnet40_normal_resampled" \
  --pointnet_checkpoint "pointnet_project/Pointnet_Pointnet2_pytorch/log/classification/pointnet2_ssg_wo_normals/checkpoints/best_model.pth" \
  --output_dir "my_experiment"
```

*Note:* The pipeline uses the real ULIP‑2 checkpoint by default (`checkpoints/ULIP‑2‑PointBERT‑10k‑xyzrgb‑pc‑vit_g‑objaverse_shapenet‑pretrained.pt`) for semantic feature extraction. Checkpoint files are git‑ignored; use `scripts/download_checkpoints.py` to obtain them.

## Pipeline Components

- `scripts/extract_features.py` - Extract PointNet2 and ULIP-2 features
- `scripts/train_fusion.py` - Train fusion classifier on extracted features
- `scripts/run_fusion_pipeline.py` - End-to-end pipeline with experiment tracking
- `scripts/download_checkpoints.py` - Download required checkpoint files

## Configuration

Edit `config/fusion_config.yaml` for default parameters.

## Project Structure

```
fusion_pipeline_project/
├── scripts/                    # Main pipeline scripts
├── libs/                       # Feature extractor adapters
├── utils/                      # Utilities (config parser, logger, checkpoint manager)
├── config/                     # Configuration files
├── tests/                      # Unit and integration tests
├── pointnet_project/           # Original PointNet2 codebase
├── ULIP/                       # Original ULIP-2 codebase
├── train_utils.py              # Training utilities with backward compatibility
├── fusion_model.py             # Fusion model definition
└── docs/                       # Documentation and design plans
```

## Usage Examples

### Extract Features Only
```bash
python scripts/extract_features.py \
  --data_dir "pointnet_project/Pointnet_Pointnet2_pytorch/data/modelnet40_normal_resampled" \
  --pointnet_checkpoint "pointnet_project/Pointnet_Pointnet2_pytorch/log/classification/pointnet2_ssg_wo_normals/checkpoints/best_model.pth" \
  --output_dir "my_features" \
  --val_ratio 0.2 \
  --split_seed 42
```

### Train Fusion Model Only
```bash
python scripts/train_fusion.py \
  --train_features "my_features/train_features.npz" \
  --test_features "my_features/test_features.npz" \
  --epochs 30 \
  --batch_size 64 \
  --checkpoint_dir "my_checkpoints"
```

*Note: The pipeline now supports three-split workflow (train/val/test). Validation features are automatically created when `--val_ratio` > 0 and can be used with `scripts/run_ablation.py` and `scripts/comprehensive_evaluation.py`.*

### Full Pipeline with Custom Config
```bash
python scripts/run_fusion_pipeline.py \
  --config config/fusion_config.yaml \
  --mode full \
  --output_dir "experiment_1"
```

## License

This pipeline integrates with original PointNet2 and ULIP-2 codebases. Please refer to their respective licenses for usage restrictions.
