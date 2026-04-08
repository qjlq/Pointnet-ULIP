# Project Structure Guide

This document explains the organization of the fusion pipeline project. The structure separates pipeline scripts, external codebases, configuration, experiments, and documentation.

## Root Directory

```
fusion_pipeline_project/
├── scripts/                    # Main pipeline scripts
├── libs/                       # Feature extractor adapters
├── project_utils/              # Utilities (config parser, logger, checkpoint manager)
├── config/                     # Configuration files
├── tests/                      # Unit and integration tests
├── pointnet_project/           # Original PointNet2 codebase
├── ULIP/                       # Original ULIP-2 codebase
├── train_utils.py              # Training utilities with backward compatibility
├── fusion_model.py             # Fusion model definition
├── verify_gpu.py               # GPU compatibility checker
├── environment.yml             # Conda environment specification
├── README.md                   # Project overview
└── docs/                       # Documentation (this folder)
```

## Scripts (`scripts/`)

Main entry points for the pipeline:

- `extract_features.py` – Extract PointNet2 and ULIP‑2 features from ModelNet40 point clouds.
- `train_fusion.py` – Train the fusion MLP on extracted features.
- `run_fusion_pipeline.py` – End‑to‑end pipeline (extraction + training).
- `run_ablation.py` – Run ablation studies comparing different feature combinations.
- `validate_pipeline.py` – Validation script for pipeline correctness.
- `verify_fusion_training_real.py`, `verify_ulip_real.py` – Verification scripts for real ULIP‑2 integration.
- `download_checkpoints.py` – Download required checkpoint files (ULIP‑2, OpenCLIP).

## Libraries (`libs/`)

Adapter modules that interface with external codebases:

- `pointnet_extractor.py` – Wrapper for PointNet2 feature extraction.
- `ulip_extractor.py` – Wrapper for ULIP‑2 feature extraction (supports real and dummy models).
- `dataset_loader.py` – Load ModelNet40 dataset in PointNet2 format.

## Utilities (`project_utils/`)

Shared utilities used across scripts:

- `config_parser.py` – Parse YAML configuration files.
- `logger.py` – Logging setup with consistent formatting.
- `checkpoint_manager.py` – Save/load training checkpoints.

## Configuration (`config/`)

YAML configuration files:

- `fusion_config.yaml` – Primary configuration for the fusion pipeline (data paths, model checkpoints, training hyperparameters).
- `full_train_config.yaml` – Configuration for full training (50 epochs).
- `test_config.yaml` – Configuration for testing/verification.

## Data and Checkpoints

- `pointnet_project/Pointnet_Pointnet2_pytorch/data/modelnet40_normal_resampled/` – ModelNet40 dataset (expected location).
- `checkpoints/` – **Git‑ignored** directory for pretrained model weights.
  - `ULIP‑2‑PointBERT‑10k‑xyzrgb‑pc‑vit_g‑objaverse_shapenet‑pretrained.pt` – Real ULIP‑2 checkpoint.
  - `open_clip_pytorch_model.bin` – Optional OpenCLIP checkpoint.
- `feature_cache/` – **Git‑ignored** cache for extracted features (saves time on repeated runs).
- `training_output/` – **Git‑ignored** training logs and intermediate checkpoints.
- `full_training_checkpoints/`, `full_training_output/` – Results of full 50‑epoch training.

## Experiments and Results

- `fusion_experiments/` – **Git‑ignored** output of pipeline runs (each subdirectory is a timestamped experiment).
- `ablation_features/` – **Git‑ignored** cached features for ablation studies.
- `ablation_results/` – **Git‑ignored** results of ablation studies (accuracy tables, plots).
- `results/` – **Git‑ignored** general results and historical comparisons.

## External Codebases

Two external projects are included as subdirectories:

- `pointnet_project/` – Original [PointNet2 PyTorch implementation](https://github.com/yanx27/Pointnet_Pointnet2_pytorch).
  - Used for geometric feature extraction.
  - Pre‑trained checkpoint included at `log/classification/pointnet2_ssg_wo_normals/checkpoints/best_model.pth`.
- `ULIP/` – Original [ULIP‑2 repository](https://github.com/salesforce/ULIP).
  - Used for semantic feature extraction.
  - No checkpoints included (must be downloaded separately).

**Note**: Both external codebases are kept as‑is to ensure compatibility. The pipeline adapters (`libs/`) provide a clean interface.

## Documentation (`docs/`)

- `SETUP_GUIDE.md` – Environment setup and troubleshooting.
- `FULL_FUSION_PIPELINE_GUIDE.md` – Detailed explanation of the fusion pipeline.
- `ULIP_TRAINING_GUIDE.md` – Guide for ULIP‑2 feature extraction and fine‑tuning.
- `pipeline_usage.md` – Concise usage instructions.
- `PROJECT_STRUCTURE.md` – This document.
- `backup/` – Historical backup of project state.
- `plans/` – Design and implementation plans.

## Testing (`tests/`)

- Unit tests for extractors, utilities, and fusion model.
- Verification scripts that ensure real ULIP‑2 integration works.
- `verification/` – Integration tests that require checkpoints and data.

## Git Ignored Items

The following are excluded from version control (see `.gitignore`):

- Checkpoint files (large, user‑specific)
- Feature caches (regenerated)
- Experiment outputs (reproducible via scripts)
- Python bytecode (`__pycache__/`)
- Conda environment folders

## Workflow

1. **Setup**: Create conda environment, download checkpoints, verify dataset.
2. **Extract features**: Run `scripts/extract_features.py` (cached in `feature_cache/`).
3. **Train fusion model**: Use `scripts/train_fusion.py` or `scripts/run_fusion_pipeline.py`.
4. **Evaluate**: Check accuracy in experiment output directories.
5. **Ablation**: Use `scripts/run_ablation.py` to compare different feature combinations.

---

*Last updated: April 7, 2026*  
*Project version: 1.0*