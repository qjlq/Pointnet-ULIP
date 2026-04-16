# Core Pipeline Commands for ModelNet40 and ScanObjectNN

This document contains all core commands for feature extraction, fusion training, and evaluation for both ModelNet40 and ScanObjectNN datasets. These commands use the core Python scripts directly rather than end-to-end wrapper scripts.

## Prerequisites

Ensure the following checkpoints and directories exist:

1. **PointNet2 checkpoint**:
   ```
   pointnet_project/Pointnet_Pointnet2_pytorch/log/classification/pointnet2_ssg_wo_normals/checkpoints/best_model.pth
   ```

2. **ULIP-2 checkpoint**:
   ```
   checkpoints/ULIP-2-PointBERT-10k-xyzrgb-pc-vit_g-objaverse_shapenet-pretrained.pt
   ```

3. **OpenCLIP checkpoint**:
   ```
   checkpoints/open_clip_pytorch_model.bin
   ```

4. **Dataset directories**:
   - ModelNet40: `pointnet_project/Pointnet_Pointnet2_pytorch/data/modelnet40_normal_resampled`
   - ScanObjectNN: `pointnet_project/Pointnet_Pointnet2_pytorch/data`

## Training with Validation Set

The pipeline now supports three-split workflow (train/val/test). Validation features are automatically created during feature extraction when `--val_ratio` > 0 (default: 0.2). Use the validation set for hyperparameter tuning and early stopping:

**Ablation study with validation set:**
```bash
python scripts/run_ablation.py \
  --train_features feature_cache/train_features.npz \
  --val_features feature_cache/val_features.npz \
  --test_features feature_cache/test_features.npz \
  --output_dir ablation_results \
  --epochs 50 \
  --batch_size 32 \
  --learning_rate 0.001 \
  --lr_scheduler cosine \
  --early_stopping_patience 10
```

**Comprehensive evaluation with validation split:**
```bash
python scripts/comprehensive_evaluation.py \
  --output_dir comprehensive_results \
  --epochs 50 \
  --seeds 42,43,44 \
  --datasets modelnet40,scanobjectnn \
  --fusion_types concat,normalized,gated \
  --lr_scheduler cosine \
  --early_stopping_patience 10 \
  --save_metrics \
  --save_training_curves \
  --formats json,csv,md
```

---
## Advanced Evaluation Commands

**Script naming clarification**: `evaluate_fusion_improvements.py` compares fusion improvements across datasets, while `comprehensive_evaluation.py` provides statistical robustness with multiple seeds.

### 1. Comprehensive Evaluation with Statistical Robustness

**Run with multiple random seeds:**
```bash
python scripts/comprehensive_evaluation.py \
  --output_dir comprehensive_results \
  --epochs 50 \
  --seeds 42,43,44 \
  --datasets modelnet40,scanobjectnn \
  --fusion_types concat,normalized,gated \
  --lr_scheduler cosine \
  --early_stopping_patience 10 \
  --save_metrics \
  --save_training_curves \
  --formats json,csv,md
```

### 2. Quick Validation Commands

**Verify feature extraction:**
```bash
python scripts/validate_features.py --dataset modelnet40
python scripts/validate_features.py --dataset scanobjectnn
```

**Verify training pipeline:**
```bash
python scripts/validate_pipeline.py --dataset modelnet40
python scripts/validate_pipeline.py --dataset scanobjectnn
```

---

## Ablation Experiments

### 1. Basic Ablation: Single Modality vs Fusion

**Using the ablation script (compares PointNet-only, ULIP-only, and fusion):**
```bash
python scripts/run_ablation.py \
  --train_features feature_cache/train_features.npz \
  --val_features feature_cache/val_features.npz \
  --test_features feature_cache/test_features.npz \
  --output_dir ablation_results \
  --epochs 50 \
  --batch_size 32 \
  --learning_rate 0.001 \
  --save_interval 10 \
  --seed 42
```

**Using shell script (includes feature extraction):**
```bash
./run_ablation.sh
```

### 2. Feature Dimension Ablation: Real vs Dummy ULIP Features

**With real ULIP features (1280-dim):**
```bash
./run_ablation_real.sh
```

**Compare results:**
```bash
# After running both scripts, compare results
python -c "
import json
with open('ablation_results_real/summary.json') as f:
    real = json.load(f)
with open('ablation_results/summary.json') as f:
    dummy = json.load(f)
print('Experiment           | Dummy Final | Real Final  | Improvement')
print('-' * 70)
for exp in ['pointnet_only', 'ulip_only', 'fusion']:
    d = dummy[exp]['final_accuracy'] * 100
    r = real[exp]['final_accuracy'] * 100
    delta = r - d
    print(f'{exp:20s} | {d:6.2f}%     | {r:6.2f}%     | {delta:+6.2f}%')
"
```

### 3. Fusion Strategy Ablation

**Already covered in comprehensive evaluation:**
```bash
# Compare concat, normalized, and gated fusion
python scripts/evaluate_fusion_improvements.py \
  --dataset modelnet40 \
  --output_dir fusion_strategy_ablation \
  --epochs 50 \
  --lr_scheduler cosine \
  --early_stopping_patience 10 \
  --save_metrics \
  --save_training_curves
```

### 4. Normalization Method Ablation

**Test different normalization modes in normalized fusion:**
```python
# Manual testing with different norm modes
import sys
sys.path.append('.')
from fusion_model import NormalizedFusionHead
import torch

# Test L2 normalization
model_l2 = NormalizedFusionHead(
    geo_dim=1024, vlm_dim=1280, hidden_dim=256, num_classes=40,
    norm_mode='l2', learnable_scale=True
)

# Test layer normalization  
model_layer = NormalizedFusionHead(
    geo_dim=1024, vlm_dim=1280, hidden_dim=256, num_classes=40,
    norm_mode='layer', learnable_scale=False
)

# Test batch normalization
model_batch = NormalizedFusionHead(
    geo_dim=1024, vlm_dim=1280, hidden_dim=256, num_classes=40,
    norm_mode='batch', learnable_scale=True
)
```

### 5. Dataset-Specific Ablation

**Compare performance across datasets:**
```bash
python scripts/comprehensive_evaluation.py \
  --output_dir cross_dataset_ablation \
  --epochs 50 \
  --seeds 42 \
  --datasets modelnet40,scanobjectnn \
  --fusion_types concat \
  --lr_scheduler cosine \
  --early_stopping_patience 10
```

### Expected Results

| Experiment | Expected ModelNet40 Accuracy | Expected ScanObjectNN Accuracy |
|------------|-----------------------------|--------------------------------|
| PointNet-only | ~90-91% | 69.94% (ablation) |
| ULIP-only (real) | ~91-92% | 87.87% (ablation) |
| Fusion (concat) | ~93-94% | 87.01% (fusion training) / 87.56% (ablation) |
| Fusion (normalized) | ~93-95% (+1-2%) | 87.73% (fusion training) |
| Fusion (gated) | ~93-95% | 87.98% (fusion training) |

### ScanObjectNN Ablation Results

**Run ablation study for ScanObjectNN:**
```bash
# Using Python script directly
python scripts/run_ablation.py \
  --train_features scanobjectnn_feature_cache/train_features.npz \
  --val_features scanobjectnn_feature_cache/val_features.npz \
  --test_features scanobjectnn_feature_cache/test_features.npz \
  --output_dir scanobjectnn_ablation_results \
  --epochs 50 \
  --batch_size 32 \
  --learning_rate 0.001 \
  --lr_scheduler cosine \
  --early_stopping_patience 10 \
  --save_metrics \
  --save_training_curves \
  --save_interval 10 \
  --seed 42

# Or using the shell script
./run_ablation_scanobjectnn.sh
```

**Actual Results (from completed ablation study):**
- **PointNet-only**: 69.94% final accuracy (71.60% best)
- **ULIP-only**: 87.87% final accuracy (88.49% best)
- **Fusion (concat)**: 87.56% final accuracy (88.08% best)

**Comparison with existing fusion training results:**
- Concat fusion (existing): 87.01% (from `scanobjectnn_results/`)
- Normalized fusion: 87.73%
- Gated fusion: 87.98%

**Key observation**: ULIP-only outperforms fusion on ScanObjectNN (87.87% vs 87.56%), suggesting ULIP features are already strong for this dataset and fusion provides limited additional benefit.

---

## Core Module Usage Examples

### Direct Import and Usage (for advanced users)

**Feature extraction using core modules:**
```python
import sys
sys.path.append('.')
from libs.pointnet_extractor import PointNetExtractor
from libs.ulip_extractor import ULIPExtractor
from libs.dataset_loader import get_dataset_loader

# Load dataset
loader = get_dataset_loader(
    dataset_type='modelnet40',
    data_root='pointnet_project/Pointnet_Pointnet2_pytorch/data/modelnet40_normal_resampled',
    split='train',
    num_points=1024
)

# Initialize extractors
pointnet_extractor = PointNetExtractor(
    checkpoint_path='pointnet_project/Pointnet_Pointnet2_pytorch/log/classification/pointnet2_ssg_wo_normals/checkpoints/best_model.pth',
    device='cuda'
)

ulip_extractor = ULIPExtractor(
    ulip_checkpoint='checkpoints/ULIP-2-PointBERT-10k-xyzrgb-pc-vit_g-objaverse_shapenet-pretrained.pt',
    openclip_checkpoint='checkpoints/open_clip_pytorch_model.bin',
    device='cuda'
)
```

**Fusion training using core modules:**
```python
import sys
sys.path.append('.')
from fusion_model import FusionModel, NormalizedFusionHead, GatedFusionHead
from train_utils import train_fusion_model
import numpy as np

# Load features
train_data = np.load('feature_cache/train_features.npz')
test_data = np.load('feature_cache/test_features.npz')

# Initialize model
model = FusionModel(
    pointnet_dim=train_data['pointnet_features'].shape[1],
    ulip_dim=train_data['ulip_features'].shape[1],
    num_classes=int(train_data['labels'].max()) + 1
)

# Train
train_losses, test_accuracies = train_fusion_model(
    model=model,
    train_features_pointnet=train_data['pointnet_features'],
    train_features_ulip=train_data['ulip_features'],
    train_labels=train_data['labels'],
    test_features_pointnet=test_data['pointnet_features'],
    test_features_ulip=test_data['ulip_features'],
    test_labels=test_data['labels'],
    epochs=50,
    batch_size=32,
    learning_rate=0.001,
    lr_scheduler='cosine',
    early_stopping_patience=10
)
```

---

## Troubleshooting Notes

1. **GPU Memory Issues**: Reduce batch size if encountering CUDA out of memory errors
   - Extraction: `--batch_size 4`
   - Training: `--batch_size 16`

2. **Checkpoint Verification**: Ensure checkpoints are properly downloaded and paths are correct

3. **Dataset Verification**: Verify dataset directories contain the expected files

4. **Cache Skipping**: Use `--skip_if_cached` to avoid re-extraction if features already exist

5. **CPU Execution**: Use `--device cpu` if GPU is not available

---

## Quick Reference

| Task | ModelNet40 Command | ScanObjectNN Command |
|------|-------------------|----------------------|
| **Feature Extraction** | `python scripts/extract_features.py --config config/fusion_config.yaml` | `python scripts/extract_features.py --config config/scanobjectnn_config.yaml` |
| **Concat Fusion Training** | `python scripts/train_fusion.py --train_features feature_cache/train_features.npz --test_features feature_cache/test_features.npz --fusion_type concat` | `python scripts/train_fusion.py --train_features scanobjectnn_feature_cache/train_features.npz --test_features scanobjectnn_feature_cache/test_features.npz --fusion_type concat` |
| **Normalized Fusion Training** | Same as above with `--fusion_type normalized` | Same as above with `--fusion_type normalized` |
| **Gated Fusion Training** | Same as above with `--fusion_type gated` | Same as above with `--fusion_type gated` |
| **Fusion Improvements Evaluation** | `python scripts/evaluate_fusion_improvements.py --dataset modelnet40 --output_dir modelnet40_results` | `python scripts/evaluate_fusion_improvements.py --dataset scanobjectnn --output_dir scanobjectnn_results` |
| **Ablation Study** | `python scripts/run_ablation.py --train_features feature_cache/train_features.npz --val_features feature_cache/val_features.npz --test_features feature_cache/test_features.npz --output_dir ablation_results --lr_scheduler cosine --early_stopping_patience 10 --save_metrics --save_training_curves` | `python scripts/run_ablation.py --train_features scanobjectnn_feature_cache/train_features.npz --val_features scanobjectnn_feature_cache/val_features.npz --test_features scanobjectnn_feature_cache/test_features.npz --output_dir scanobjectnn_ablation_results --lr_scheduler cosine --early_stopping_patience 10 --save_metrics --save_training_curves` |

---

*Last updated: 2026-04-14*