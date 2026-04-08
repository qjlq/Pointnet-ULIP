# Quick Start Guide

## 1. Environment Setup
```bash
conda env create -f environment.yml
conda activate fusion-pipeline
```

## 2. Download Checkpoints
```bash
python scripts/download_checkpoints.py
```

## 3. Run Full Pipeline
```bash
python scripts/run_fusion_pipeline.py \
  --data_dir "path/to/modelnet40" \
  --output_dir "my_experiment"
```

## 4. Verify Installation
```bash
python scripts/validate_pipeline.py
```

## Expected Output
- Feature extraction: Creates `my_experiment/features/`
- Training: Creates `my_experiment/checkpoints/`
- Final accuracy printed to console

## Configuration Notes

The pipeline uses relative paths to access external resources:
- PointNet2 codebase: `../pointnet_project/`
- ULIP-2 codebase: `../ULIP/`
- Checkpoints: `../checkpoints/`

Make sure these directories exist relative to the submission folder.