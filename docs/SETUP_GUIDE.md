# Fusion Pipeline Setup Guide

## Environment Setup

### 1. Create Conda Environment

```bash
conda env create -f environment.yml
```

### 2. Activate Environment

```bash
conda activate fusion_pipeline
```

If the environment name differs (check `environment.yml`), use the correct name.

### 3. Verify GPU Setup

```bash
python verify_gpu.py
```

Expected output includes:
- PyTorch version
- CUDA availability (should be True)
- GPU device information
- Tensor operation tests passing

## Environment Details

### CUDA Compatibility

- **Environment uses**: CUDA 12.1 toolkit (PyTorch built for CUDA 12.1)
- **Compatible with**: CUDA 13 driver via forward compatibility
- **Verification**: `verify_gpu.py` checks CUDA availability and prints compatibility note

### PyTorch Version Choice

- **Selected version**: PyTorch 1.10.1
- **Reason**: Compromise between PointNet2 (requires 1.6.0) and ULIP (requires 1.10.1)
- **Alternative**: Pipeline also works with PyTorch 2.5.1 but 1.10.1 ensures subproject compatibility

### Package Verification

All required packages from Task 1 analysis are included in `environment.yml`:

| Package | Version | Required by |
|---------|---------|-------------|
| PyTorch | 1.10.1 | Core framework |
| torchvision | 0.11.2 | ULIP compatibility |
| torchaudio | 0.10.1 | ULIP compatibility |
| numpy | * | Scientific computing |
| scipy | * | Scientific computing |
| scikit-learn | * | Machine learning utilities |
| h5py | 3.6.0 | ULIP data handling |
| pyyaml | * | Configuration parsing |
| pandas | * | Data manipulation |
| matplotlib | * | Visualization |
| tqdm | * | Progress bars |
| termcolor | 1.1.0 | ULIP logging |
| lmdb | 1.3.0 (via pip) | ULIP dataset |
| open3d | 0.16.0 (via pip) | ULIP point cloud visualization |
| wandb | 0.13.3 | ULIP experiment tracking |
| ftfy | 6.0.1 | ULIP text cleaning |
| regex | 2022.10.31 | ULIP text processing |
| easydict | 1.9 | ULIP configuration |
| timm | 0.4.12 | ULIP vision transformer |
| open-clip-torch | 2.24.0 | ULIP CLIP model |
| pyyaml_env_tag | 0.1 | ULIP config parsing |

**Note**: Some packages are installed via pip due to conda availability.

## Data Preparation

### ModelNet40 Dataset

The pipeline expects the ModelNet40 dataset in PointNet2 format at:

```
pointnet_project/Pointnet_Pointnet2_pytorch/data/modelnet40_normal_resampled/
```

Required files:
- `modelnet40_train.txt`
- `modelnet40_test.txt`
- `modelnet40_train_1024pts.dat`
- `modelnet40_test_1024pts.dat`

If the dataset is missing, download from [ModelNet40](https://modelnet.cs.princeton.edu/) and preprocess using PointNet2's data preparation scripts.

### Checkpoint Files

- **PointNet2 checkpoint**: Already present at `pointnet_project/Pointnet_Pointnet2_pytorch/log/classification/pointnet2_ssg_wo_normals/checkpoints/best_model.pth`
- **ULIP-2 checkpoint**: Included. The real ULIP-2 checkpoint is located at `checkpoints/ULIP-2-PointBERT-10k-xyzrgb-pc-vit_g-objaverse_shapenet-pretrained.pt`. If missing, a dummy model will be used (fallback).

### Checkpoint Management

Checkpoint files are git‑ignored (see `.gitignore`) and must be obtained separately. Use the provided download script to fetch required checkpoints:

```bash
# Download ULIP‑2 checkpoint (and optional OpenCLIP checkpoint)
python scripts/download_checkpoints.py

# List required files
python scripts/download_checkpoints.py --list

# Force re‑download
python scripts/download_checkpoints.py --force

# Skip OpenCLIP (optional)
python scripts/download_checkpoints.py --skip_openclip
```

The script downloads from Hugging Face Hub (`salesforce/ULIP‑2`) and optionally uses `open_clip` to fetch OpenCLIP weights. If the script fails due to missing dependencies, install them with `pip install huggingface-hub requests tqdm`. If the script still fails, manually download:

- **ULIP‑2 checkpoint**: [ulip2_pointbert_weights.pth](https://huggingface.co/salesforce/ULIP-2/blob/main/ulip2_pointbert_weights.pth) → rename to `ULIP‑2‑PointBERT‑10k‑xyzrgb‑pc‑vit_g‑objaverse_shapenet‑pretrained.pt`
- **OpenCLIP checkpoint**: Use `open_clip.create_model_and_transforms('ViT‑g‑14', pretrained='laion2b_s12b_b42k')` and save the state dict as `checkpoints/open_clip_pytorch_model.bin`

After downloading, verify that `config/fusion_config.yaml` points to the correct paths.

## Troubleshooting

### Common Issues

1. **CUDA not available**
   - Ensure NVIDIA driver is installed (CUDA 13 driver recommended)
   - Run `nvidia-smi` to verify GPU visibility
   - Check that PyTorch CUDA version matches driver

2. **Environment creation fails**
   - Check conda channels and network connectivity
   - Ensure sufficient disk space
   - Try `conda env create -f environment.yml --verbose` for detailed logs

3. **Import errors**
   - Verify all packages installed correctly with `conda list`
   - Check Python version (3.8 recommended)
   - Ensure environment is activated

4. **ULIP dummy fallback warning**
   - Real ULIP functionality requires downloading pretrained weights
   - Run `python scripts/download_checkpoints.py` to fetch the ULIP‑2 checkpoint
   - Ensure the checkpoint file exists at `checkpoints/ULIP‑2‑PointBERT‑10k‑xyzrgb‑pc‑vit_g‑objaverse_shapenet‑pretrained.pt`

5. **Dataset not found errors**
   - Verify `data.root_dir` in config points to correct location
   - Check file permissions and paths

### GPU Verification Output

Successful `verify_gpu.py` output includes:

```
PyTorch version: 1.10.1
CUDA available: True
CUDA version (PyTorch built with): 12.4
GPU 0: NVIDIA GeForce RTX 4070 Ti SUPER
Tensor operation test passed
```

### Memory Issues

If encountering CUDA out of memory errors:

- Reduce `extraction.batch_size` in config (default 64)
- Reduce `training.batch_size` in config (default 32)
- Close other GPU-intensive applications

## Next Steps

After successful environment setup:

1. **Run quick verification tests:**
   ```bash
   python test_pointnet_extraction.py
   python test_ulip_extraction.py
   ```

2. **Test full pipeline:**
   ```bash
   python scripts/run_fusion_pipeline.py --config config/fusion_config.yaml --mode train_only
   ```

3. **Refer to [Full Fusion Pipeline Guide](FULL_FUSION_PIPELINE_GUIDE.md)** for detailed usage instructions.

4. **ULIP-2 pretrained weights** can be downloaded using the checkpoint download script (see Checkpoint Management section above).

---

*Last updated: April 7, 2026*  
*Environment version: 1.0*  
*Tested on: Ubuntu 22.04, CUDA 13, RTX 4070 Ti SUPER*