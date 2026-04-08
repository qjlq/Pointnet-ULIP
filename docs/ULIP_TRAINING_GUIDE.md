# ULIP-2 Training Guide

## Overview

ULIP-2 (Unified Language-Image-Point Cloud Pretraining) learns unified representations across 3D point clouds, images, and text. This guide covers using ULIP-2 for point cloud feature extraction and optional fine-tuning.

## ULIP-2 in Fusion Pipeline

The fusion pipeline uses ULIP-2 for semantic feature extraction:

- **Real ULIP‑2**: 1280‑D semantic features from pretrained checkpoint (`ULIP‑2‑PointBERT‑10k‑xyzrgb‑pc‑vit_g‑objaverse_shapenet‑pretrained.pt`) – included by default
- **Dummy model**: 256‑D random features (fallback if real checkpoint missing)

## Checkpoint Availability

The real ULIP‑2 checkpoint (`ULIP‑2‑PointBERT‑10k‑xyzrgb‑pc‑vit_g‑objaverse_shapenet‑pretrained.pt`) is expected in the `checkpoints/` directory. If missing, download it using one of the options below. The pipeline includes a dummy fallback model for development.

If you need to download a different checkpoint:

### Option 1: Download Official Checkpoints
1. Visit [ULIP‑2 repository](https://github.com/salesforce/ULIP)
2. Download pretrained weights for point cloud encoder
3. Place in `checkpoints/` directory

### Option 2: Use Hugging Face
```python
from huggingface_hub import hf_hub_download
checkpoint = hf_hub_download(repo_id="salesforce/ULIP-2", filename="ulip2_pointbert_weights.pth")
```

### Option 3: Use Download Script
Run `python scripts/download_checkpoints.py` to automatically download the ULIP‑2 checkpoint (and optional OpenCLIP checkpoint). See [SETUP_GUIDE.md](SETUP_GUIDE.md) for details.

### Option 4: Manual Placement
Place downloaded checkpoint file in `fusion_pipeline_project/checkpoints/`

## Configuration Update

The configuration already points to the included ULIP‑2 checkpoint. If using a different checkpoint, update `config/fusion_config.yaml`:

```yaml
models:
  ulip_checkpoint: "checkpoints/ULIP‑2‑PointBERT‑10k‑xyzrgb‑pc‑vit_g‑objaverse_shapenet‑pretrained.pt"
```

## Feature Extraction with Real ULIP-2

With the included checkpoint, ULIP-2 extracts 1280-dimensional semantic features:

```bash
# Re-extract features (delete old cache or disable skip_if_cached)
rm -rf feature_cache/
python scripts/extract_features.py --skip_if_cached
```

**Expected feature dimensions:**
- PointNet2: (N, 1024)
- ULIP-2: (N, 1280)

## Fine-tuning ULIP-2 on ModelNet40 (Optional)

### 1. Setup ULIP Repository

```bash
cd ULIP
pip install -r requirements.txt
```

### 2. Prepare Dataset

Ensure ModelNet40 is in PointNet2 format and accessible.

### 3. Fine-tuning Command

Refer to ULIP repository instructions for fine-tuning. Example command:

```bash
python main.py \
  --config configs/pointbert_finetune_modelnet40.yaml \
  --pretrained_ckpt ../checkpoints/ulip2_pointbert_weights.pth \
  --output_dir outputs/ulip2_modelnet40_finetuned \
  --batch_size 32 \
  --epochs 100
```

### 4. Integration with Fusion Pipeline

After fine-tuning:

1. Update checkpoint path in config:
   ```yaml
   models:
     ulip_checkpoint: "outputs/ulip2_modelnet40_finetuned/best_model.pth"
   ```

2. Re-extract features
3. Retrain fusion model

## ULIP-2 Architecture Details

### Point Cloud Encoder

ULIP-2 uses PointBERT as point cloud encoder:
- Transformer-based architecture
- Input: 1024 points × 3 coordinates
- Output: 512-dimensional semantic features
- Pretrained with contrastive loss across modalities

### Dummy Model

When pretrained weights unavailable, the pipeline uses a dummy MLP:

```
Input: (B, N, 3) point clouds
Per-point MLP: 3 → 128 → 256
Global average pooling → 256D features
```

## Performance Expectations

| Model | Feature Dim | Accuracy Contribution | Notes |
|-------|-------------|----------------------|-------|
| Dummy ULIP | 256 | Minimal | Random features, little semantic value |
| Real ULIP‑2 (included) | 1280 | High (93.19%) | 1280‑D semantic features, +0.89% over baseline |
| Fine‑tuned ULIP‑2 | 1280 | Optimal | Best semantic representation after fine‑tuning |

Actual fusion accuracy improvements (ModelNet40 test set):
- PointNet2 alone: 92.30%
- PointNet2 + Dummy ULIP: 91.69% (–0.61%)
- PointNet2 + Real ULIP‑2: **93.19%** (**+0.89%**)

## Troubleshooting

### Common Issues

1. **ULIP checkpoint loading fails**
   - Verify checkpoint file exists and is not corrupted
   - Check PyTorch version compatibility (1.10.1 recommended)
   - Ensure ULIP repository is in Python path

2. **Feature dimension mismatch**
    - Real ULIP-2 outputs 1280 features, dummy outputs 256
   - Update fusion model `ulip_dim` parameter accordingly
   - Reinitialize fusion model with correct dimensions

3. **CUDA memory issues**
   - ULIP-2 requires significant GPU memory
   - Reduce extraction batch size in config
   - Use CPU extraction if needed (`device: cpu`)

4. **Import errors**
   - Ensure ULIP dependencies installed (see `environment.yml`)
   - Check Python path includes ULIP directory

### Log Messages

**Expected with dummy model:**
```
ULIP-2 loading failed, using dummy model: [error details]
Using dummy model: True
```

**Expected with real ULIP-2:**
```
Loading ULIP-2 extractor from checkpoints/ULIP‑2‑PointBERT‑10k‑xyzrgb‑pc‑vit_g‑objaverse_shapenet‑pretrained.pt
Using dummy model: False
```

## Advanced Usage

### Custom Point Cloud Encoders

To use different point cloud encoders with ULIP-2 framework:

1. Implement encoder in `ULIP/models/`
2. Update configuration files
3. Ensure output dimension matches fusion pipeline expectations

### Multi-modal Training

ULIP-2 supports training with image-text-point cloud triplets. Refer to original repository for multi-modal training instructions.

## References

- ULIP-2 Paper: [ULIP-2: Towards Scalable Multimodal Pre-training for 3D Understanding](https://arxiv.org/abs/2305.08275)
- Official Repository: [salesforce/ULIP](https://github.com/salesforce/ULIP)
- Hugging Face: [salesforce/ULIP-2](https://huggingface.co/salesforce/ULIP-2)

---

*Last updated: April 7, 2026*  
*Checkpoint: ULIP‑2‑PointBERT‑10k‑xyzrgb‑pc‑vit_g‑objaverse_shapenet‑pretrained.pt*