# ULIP-2 + PointNet2 Fusion Pipeline - Submission Version

A two-stage feature fusion pipeline for point cloud classification, combining geometric features from PointNet2 with semantic features from ULIP-2.

## Results

Real ULIP‑2 integration achieved **93.19%** accuracy on ModelNet40 test set, a **+0.89%** improvement over the PointNet‑only baseline (92.30%).

## Setup

1. Create conda environment:
```bash
conda env create -f environment.yml
conda activate fusion-pipeline
```

2. Download required checkpoints (run once):
```bash
python scripts/download_checkpoints.py
```

## Quick Start

Run the full pipeline:
```bash
python scripts/run_fusion_pipeline.py \
  --data_dir "path/to/modelnet40_normal_resampled" \
  --output_dir "experiment_output"
```

## Pipeline Components

- `scripts/extract_features.py` - Extract PointNet2 and ULIP-2 features
- `scripts/train_fusion.py` - Train fusion classifier on extracted features
- `scripts/run_fusion_pipeline.py` - End-to-end pipeline
- `scripts/download_checkpoints.py` - Download checkpoints

## Project Structure

```
fusion_pipeline_submission/
├── scripts/           # Pipeline scripts
├── libs/             # Feature extractors
├── config/           # Configuration files
├── project_utils/    # Utility modules
├── docs/             # Documentation
├── train_utils.py    # Training utilities
├── fusion_model.py   # Fusion model definition
└── environment.yml   # Environment configuration
```

## License

This project integrates with PointNet2 and ULIP-2 codebases. Refer to their respective licenses.