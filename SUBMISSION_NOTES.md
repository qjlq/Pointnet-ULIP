# Submission Notes

## What's Included
1. Core pipeline code (scripts/, libs/, config/, project_utils/)
2. Fusion model implementation (fusion_model.py, train_utils.py)
3. Documentation (docs/, README.md)
4. Environment configuration (environment.yml)

## What's Excluded
1. Checkpoint files (*.pth, *.pt, *.bin) - Use `scripts/download_checkpoints.py`
2. Data files (*.npz) - Generated during pipeline execution
3. Third-party code (pointnet_project/, ULIP/) - Referenced but not included
4. Experiment results and logs

## How to Run
1. Install environment: `conda env create -f environment.yml`
2. Download checkpoints: `python scripts/download_checkpoints.py`
3. Run pipeline: `python scripts/run_fusion_pipeline.py --data_dir <path> --output_dir <dir>`

## Dependencies
- Python 3.8+
- PyTorch 1.10+
- NumPy, SciPy, scikit-learn
- ModelNet40 dataset (not included)

## Path Configuration
The pipeline uses relative paths to access external resources:
- PointNet2 codebase: `../pointnet_project/`
- ULIP-2 codebase: `../ULIP/`
- Checkpoints: `../checkpoints/`

Make sure these directories exist relative to the submission folder.

## Contact
For questions about this submission, refer to the original project documentation.