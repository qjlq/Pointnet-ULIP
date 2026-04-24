

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
