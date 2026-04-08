# PointNet2 Training Guide

## Overview

PointNet++ (PointNet2) is a hierarchical neural network for point cloud processing. This guide covers training PointNet2 on ModelNet40 for classification, which provides the geometric features used in the fusion pipeline.

## Prerequisites

- CUDA-capable GPU (recommended)
- PointNet2 repository (included in `pointnet_project/Pointnet_Pointnet2_pytorch`)
- ModelNet40 dataset (already prepared for fusion pipeline)

## Training Steps

### 1. Navigate to PointNet2 Directory

```bash
cd pointnet_project/Pointnet_Pointnet2_pytorch
```

### 2. Verify Dataset

Ensure ModelNet40 dataset is present:

```
data/modelnet40_normal_resampled/
├── modelnet40_train.txt
├── modelnet40_test.txt
├── modelnet40_train_1024pts.dat
└── modelnet40_test_1024pts.dat
```

### 3. Training Command

Train PointNet2 with single-scale grouping (SSG) without normals:

```bash
python train_classification.py \
  --model pointnet2_cls_ssg \
  --log_dir pointnet2_ssg_wo_normals \
  --normal_channel 0 \
  --batch_size 32 \
  --epoch 250 \
  --learning_rate 0.001 \
  --optimizer adam
```

**Parameters:**
- `--model`: Model architecture (`pointnet2_cls_ssg` or `pointnet2_cls_msg`)
- `--log_dir`: Directory for logs and checkpoints
- `--normal_channel`: 0 for XYZ only, 1 for XYZ+normals
- `--batch_size`: Adjust based on GPU memory
- `--epoch`: Total training epochs
- `--learning_rate`: Initial learning rate
- `--optimizer`: `adam` or `sgd`

### 4. Monitoring Training

- Logs saved to `log/classification/pointnet2_ssg_wo_normals/`
- TensorBoard visualization:
  ```bash
  tensorboard --logdir log/classification
  ```
- Checkpoints saved at `log/classification/pointnet2_ssg_wo_normals/checkpoints/`

### 5. Evaluation

Evaluate trained model:

```bash
python test_classification.py \
  --log_dir pointnet2_ssg_wo_normals \
  --normal_channel 0 \
  --batch_size 32
```

Expected accuracy: ~91% on ModelNet40 test set.

## Integration with Fusion Pipeline

### Checkpoint Location

The fusion pipeline expects the PointNet2 checkpoint at:

```
pointnet_project/Pointnet_Pointnet2_pytorch/log/classification/pointnet2_ssg_wo_normals/checkpoints/best_model.pth
```

### Feature Extraction

After training, the fusion pipeline uses `libs/pointnet_extractor.py` to extract 1024-dimensional features from PointNet2's global feature vector.

### Using Different Architectures

If you train a different PointNet2 variant:

1. Update `pointnet_extractor.py` to load the correct model architecture
2. Update checkpoint path in `config/fusion_config.yaml`
3. Re-extract features

## Hyperparameter Tuning

### Learning Rate Schedule
- Default: 0.001 with step decay at epochs 20, 60, 120, 160
- Adjust in `train_classification.py` or via `--scheduler` flag

### Data Augmentation
- Random rotation, scaling, and jittering enabled by default
- Modify `provider.py` for custom augmentations

### Regularization
- Dropout: 0.5 in classification head
- Weight decay: 1e-4 (configure via `--decay_rate`)

## Troubleshooting

### Common Issues

1. **CUDA out of memory**
   - Reduce `--batch_size` (default 32)
   - Use `--num_point 512` for fewer points (requires model adjustment)

2. **Training loss not decreasing**
   - Check learning rate (`--learning_rate`)
   - Verify dataset is loaded correctly
   - Ensure normalization is applied

3. **Checkpoint not found**
   - Verify `--log_dir` matches during training and testing
   - Check checkpoint file exists

4. **Low test accuracy**
   - Ensure `--normal_channel` matches training setting
   - Verify model architecture consistency

## Performance Expectations

| Model | Input | Accuracy | Checkpoint Size |
|-------|-------|----------|-----------------|
| PointNet2 SSG | XYZ only | ~91% | 17.8 MB |
| PointNet2 SSG | XYZ+normals | ~92% | 17.8 MB |
| PointNet2 MSG | XYZ only | ~92% | 18.2 MB |

## Advanced: Custom Datasets

To train PointNet2 on custom point cloud datasets:

1. Prepare data in ModelNet40 format or implement custom DataLoader
2. Update `data_utils/ModelNetDataLoader.py` or create new loader
3. Adjust number of classes in model definition
4. Modify training script for new class count

## References

- Original PointNet++ paper: [PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space](https://arxiv.org/abs/1706.02413)
- Official implementation: [charlesq34/pointnet2](https://github.com/charlesq34/pointnet2)
- ModelNet40 dataset: [ModelNet](https://modelnet.cs.princeton.edu/)

---

*Last updated: March 30, 2026*  
*Based on PointNet2 PyTorch implementation*