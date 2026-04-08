# Setup Guide Template for Fusion Pipeline

## Environment Setup

### 1. Create Conda Environment
```bash
conda env create -f environment.yml
```

### 2. Activate Environment
```bash
conda activate fusion_pipeline
```

### 3. Verify GPU Setup
```bash
python verify_gpu.py
```

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

## Troubleshooting

### Common Issues

1. **CUDA not available**: Ensure NVIDIA driver is installed (CUDA 13 driver recommended)
2. **Environment creation fails**: Check conda channels and network connectivity
3. **Import errors**: Verify all packages installed correctly
4. **ULIP dummy fallback**: Real ULIP functionality requires downloading pretrained weights

### GPU Verification Output
Expected successful output includes:
- PyTorch version: 1.10.x
- CUDA available: True
- CUDA version: 12.1 (PyTorch built with)
- GPU device information
- Tensor operation tests passing

## Next Steps

After environment setup:
1. Run full pipeline test: `python scripts/run_fusion_pipeline.py --config config/test_config.yaml`
2. Download ULIP-2 pretrained weights if full ULIP functionality needed
3. Refer to `docs/pipeline_usage.md` for detailed pipeline usage