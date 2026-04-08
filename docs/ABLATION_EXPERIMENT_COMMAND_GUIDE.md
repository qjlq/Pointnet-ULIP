# 消融实验完整命令行指导

本指导提供从特征提取到特征融合训练的全过程命令行操作，用于复现ULIP-2 + PointNet2融合管道的消融实验。

## 环境与数据准备

### 1. 检查环境
```bash
# 验证Python和PyTorch版本
python --version
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"

# 安装必要依赖
pip install numpy tqdm scikit-learn
```

### 2. 验证数据文件
```bash
# 检查ModelNet40数据集
DATA_DIR="pointnet_project/Pointnet_Pointnet2_pytorch/data/modelnet40_normal_resampled"
ls -la "$DATA_DIR"

# 检查PointNet2检查点
CHECKPOINT="pointnet_project/Pointnet_Pointnet2_pytorch/log/classification/pointnet2_ssg_wo_normals/checkpoints/best_model.pth"
ls -la "$CHECKPOINT"
```

## 特征提取阶段

### 3. 提取真实ULIP特征（1280维）
**注意**：需要ULIP-2模型检查点。如果不可用，将自动使用dummy特征。

```bash
# 提取真实ULIP特征（需要ULIP检查点）
python scripts/extract_features.py \
  --data_dir "pointnet_project/Pointnet_Pointnet2_pytorch/data/modelnet40_normal_resampled" \
  --pointnet_checkpoint "pointnet_project/Pointnet_Pointnet2_pytorch/log/classification/pointnet2_ssg_wo_normals/checkpoints/best_model.pth" \
  --ulip_checkpoint "path/to/your/ulip_checkpoint.pth" \
  --batch_size 64 \
  --device cuda \
  --output_dir "feature_cache_full" \
  --skip_if_cached
```

### 4. 提取dummy ULIP特征（256维）
**注意**：如果没有ULIP检查点，将自动使用dummy特征。

```bash
# 提取dummy ULIP特征（自动回退）
python scripts/extract_features.py \
  --data_dir "pointnet_project/Pointnet_Pointnet2_pytorch/data/modelnet40_normal_resampled" \
  --pointnet_checkpoint "pointnet_project/Pointnet_Pointnet2_pytorch/log/classification/pointnet2_ssg_wo_normals/checkpoints/best_model.pth" \
  --batch_size 64 \
  --device cuda \
  --output_dir "ablation_features" \
  --skip_if_cached
```

### 5. 验证特征文件
```bash
# 检查真实特征维度
python -c "
import numpy as np
train = np.load('feature_cache_full/train_features.npz')
test = np.load('feature_cache_full/test_features.npz')
print('Real ULIP features (1280-dim):')
print(f'  Train: ULIP={train[\"ulip_features\"].shape}, PointNet={train[\"pointnet_features\"].shape}')
print(f'  Test: ULIP={test[\"ulip_features\"].shape}, PointNet={test[\"pointnet_features\"].shape}')
"

# 检查dummy特征维度
python -c "
import numpy as np
train = np.load('ablation_features/train_features.npz')
test = np.load('ablation_features/test_features.npz')
print('Dummy ULIP features (256-dim):')
print(f'  Train: ULIP={train[\"ulip_features\"].shape}, PointNet={train[\"pointnet_features\"].shape}')
print(f'  Test: ULIP={test[\"ulip_features\"].shape}, PointNet={test[\"pointnet_features\"].shape}')
"
```

## 消融实验训练阶段

### 6. 运行完整消融实验（使用真实特征）
```bash
# 使用真实ULIP特征（1280-dim）运行完整消融实验
python scripts/run_ablation.py \
  --train_features "feature_cache_full/train_features.npz" \
  --test_features "feature_cache_full/test_features.npz" \
  --output_dir "ablation_results_real" \
  --epochs 50 \
  --batch_size 32 \
  --learning_rate 0.001 \
  --save_interval 10 \
  --seed 42
```

### 7. 运行完整消融实验（使用dummy特征）
```bash
# 使用dummy ULIP特征（256-dim）运行完整消融实验
python scripts/run_ablation.py \
  --train_features "ablation_features/train_features.npz" \
  --test_features "ablation_features/test_features.npz" \
  --output_dir "ablation_results_dummy" \
  --epochs 50 \
  --batch_size 32 \
  --learning_rate 0.001 \
  --save_interval 10 \
  --seed 42
```

### 8. 分步运行消融实验（可选）
**注意**：`run_ablation.py`实际上一次性运行三个实验。如果需要单独运行：

```bash
# 创建单独的实验目录
mkdir -p ablation_results_separate

# PointNet-only实验
python -c "
import sys
sys.path.insert(0, '.')
from scripts.run_ablation import main
import shlex
args = shlex.split('''
  --train_features ablation_features/train_features.npz
  --test_features ablation_features/test_features.npz  
  --output_dir ablation_results_separate/pointnet_only
  --epochs 50
  --batch_size 32
  --learning_rate 0.001
  --seed 42
''')
sys.argv = ['run_ablation.py'] + args
main()
"

# ULIP-only实验
python -c "
import sys
sys.path.insert(0, '.')
from scripts.run_ablation import main
import shlex
args = shlex.split('''
  --train_features ablation_features/train_features.npz
  --test_features ablation_features/test_features.npz  
  --output_dir ablation_results_separate/ulip_only
  --epochs 50
  --batch_size 32
  --learning_rate 0.001
  --seed 42
''')
sys.argv = ['run_ablation.py'] + args
main()
"

# Fusion实验
python -c "
import sys
sys.path.insert(0, '.')
from scripts.run_ablation import main
import shlex
args = shlex.split('''
  --train_features ablation_features/train_features.npz
  --test_features ablation_features/test_features.npz  
  --output_dir ablation_results_separate/fusion
  --epochs 50
  --batch_size 32
  --learning_rate 0.001
  --seed 42
''')
sys.argv = ['run_ablation.py'] + args
main()
"
```

## 结果验证与分析阶段

### 9. 检查实验结果
```bash
# 查看真实特征实验结果
echo "=== Real ULIP features (1280-dim) results ==="
python -c "
import json
with open('ablation_results_real/summary.json') as f:
    data = json.load(f)
for exp, res in data.items():
    if res['status'] == 'success':
        print(f'{exp:20s}: Final={res[\"final_accuracy\"]:.4f}, Best={res[\"best_accuracy\"]:.4f}')
    else:
        print(f'{exp:20s}: FAILED')
"

# 查看dummy特征实验结果
echo "=== Dummy ULIP features (256-dim) results ==="
python -c "
import json
with open('ablation_results_dummy/summary.json') as f:
    data = json.load(f)
for exp, res in data.items():
    if res['status'] == 'success':
        print(f'{exp:20s}: Final={res[\"final_accuracy\"]:.4f}, Best={res[\"best_accuracy\"]:.4f}')
    else:
        print(f'{exp:20s}: FAILED')
"
```

### 10. 对比分析
```bash
# 生成对比报告
python -c "
import json
import os

def load_summary(path):
    with open(path) as f:
        return json.load(f)

real_path = 'ablation_results_real/summary.json'
dummy_path = 'ablation_results_dummy/summary.json'

if os.path.exists(real_path) and os.path.exists(dummy_path):
    real = load_summary(real_path)
    dummy = load_summary(dummy_path)
    
    print('=== Ablation Study Comparison ===')
    print('Real ULIP (1280-dim) vs Dummy ULIP (256-dim)')
    print()
    print('Experiment           | Dummy Final | Real Final  | Improvement')
    print('-' * 70)
    
    for exp in ['pointnet_only', 'ulip_only', 'fusion']:
        if real[exp]['status'] == 'success' and dummy[exp]['status'] == 'success':
            d_final = dummy[exp]['final_accuracy'] * 100
            r_final = real[exp]['final_accuracy'] * 100
            delta = r_final - d_final
            print(f'{exp:20s} | {d_final:6.2f}%     | {r_final:6.2f}%     | {delta:+6.2f}%')
    
    print()
    print('=== Key Observations ===')
    ulip_improvement = (real['ulip_only']['final_accuracy'] - dummy['ulip_only']['final_accuracy']) * 100
    fusion_improvement = (real['fusion']['final_accuracy'] - dummy['fusion']['final_accuracy']) * 100
    print(f'1. ULIP-only improvement: +{ulip_improvement:.2f}%')
    print(f'2. Fusion improvement: +{fusion_improvement:.2f}%')
    
    if real['fusion']['final_accuracy'] > real['ulip_only']['final_accuracy']:
        fusion_gain = (real['fusion']['final_accuracy'] - real['ulip_only']['final_accuracy']) * 100
        print(f'3. Fusion gain over ULIP-only: +{fusion_gain:.2f}%')
    else:
        print('3. Fusion not better than ULIP-only')
else:
    print('Error: Missing result files')
    print(f'Real features: {os.path.exists(real_path)}')
    print(f'Dummy features: {os.path.exists(dummy_path)}')
"
```

### 11. 验证训练结果完整性
```bash
# 检查实验目录结构
echo "=== Checking experiment directory structure ==="
ls -la ablation_results_real/
ls -la ablation_results_real/pointnet_only/
ls -la ablation_results_real/ulip_only/
ls -la ablation_results_real/fusion/

# 验证模型检查点
python -c "
import torch
import os

def check_checkpoints(exp_dir):
    checkpoint_dir = os.path.join(exp_dir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        print(f'  ✗ Checkpoint directory missing: {checkpoint_dir}')
        return False
    
    files = os.listdir(checkpoint_dir)
    required = ['best_model.pth', 'final_model.pth']
    missing = [f for f in required if f not in files]
    
    if missing:
        print(f'  ✗ Missing checkpoints: {missing}')
        return False
    
    # 尝试加载检查点
    try:
        model = torch.load(os.path.join(checkpoint_dir, 'best_model.pth'), map_location='cpu')
        print(f'  ✓ Checkpoints OK ({len(files)} files)')
        return True
    except Exception as e:
        print(f'  ✗ Failed to load checkpoint: {e}')
        return False

print('Real features experiments:')
for exp in ['pointnet_only', 'ulip_only', 'fusion']:
    print(f'{exp}: ', end='')
    check_checkpoints(f'ablation_results_real/{exp}')
"
```

## 高级分析与可视化

### 12. 生成详细分析报告
```bash
# 创建详细的对比报告
python -c "
import json
import os
from datetime import datetime

def create_report(real_path, dummy_path, output_file='ablation_comparison_report.md'):
    with open(real_path) as f:
        real = json.load(f)
    with open(dummy_path) as f:
        dummy = json.load(f)
    
    report = f'''# Ablation Study Comparison Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary

| Experiment | Dummy Features (256-dim) | Real Features (1280-dim) | Improvement |
|------------|--------------------------|--------------------------|-------------|
'''
    
    for exp in ['pointnet_only', 'ulip_only', 'fusion']:
        d_final = dummy[exp]['final_accuracy'] * 100
        r_final = real[exp]['final_accuracy'] * 100
        delta = r_final - d_final
        report += f'| {exp} | {d_final:.2f}% | {r_final:.2f}% | {delta:+.2f}% |\n'
    
    report += '''
## Key Findings

1. **ULIP Feature Dimensionality Impact**
   - Real 1280-dim ULIP features: {:.2f}% (ULIP-only)
   - Dummy 256-dim ULIP features: {:.2f}% (ULIP-only)
   - **Absolute improvement: {:.2f}%**

2. **Fusion Effectiveness**
   - With real features: {:.2f}% (fusion)
   - With dummy features: {:.2f}% (fusion)
   - **Improvement: {:.2f}%**

3. **Best Single Feature vs Fusion**
   - Real features: Fusion ({:.2f}%) vs Best single ({:.2f}%)
   - **Fusion gain: {:.2f}%**

## Conclusion
Real ULIP features (1280-dim) are essential for meaningful semantic representation. The fusion pipeline provides additive gains when using proper semantic features.
'''.format(
        real['ulip_only']['final_accuracy'] * 100,
        dummy['ulip_only']['final_accuracy'] * 100,
        (real['ulip_only']['final_accuracy'] - dummy['ulip_only']['final_accuracy']) * 100,
        real['fusion']['final_accuracy'] * 100,
        dummy['fusion']['final_accuracy'] * 100,
        (real['fusion']['final_accuracy'] - dummy['fusion']['final_accuracy']) * 100,
        real['fusion']['final_accuracy'] * 100,
        max(real['pointnet_only']['final_accuracy'], real['ulip_only']['final_accuracy']) * 100,
        (real['fusion']['final_accuracy'] - max(real['pointnet_only']['final_accuracy'], real['ulip_only']['final_accuracy'])) * 100
    )
    
    with open(output_file, 'w') as f:
        f.write(report)
    
    print(f'Report saved to: {output_file}')

create_report('ablation_results_real/summary.json', 'ablation_results_dummy/summary.json')
"
```

## 故障排除命令

### 13. 常见问题检查
```bash
# 检查CUDA可用性
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, Device count: {torch.cuda.device_count()}')"

# 检查Python路径
python -c "import sys; print('Python path:'); [print(p) for p in sys.path[:5]]"

# 检查模块导入
python -c "
import sys
sys.path.insert(0, '.')
try:
    from fusion_model import PointNetOnlyClassifier, ULIPOnlyClassifier, FusionModel
    from train_utils import train_single_feature_model, train_fusion_model
    print('✓ All required imports successful')
except Exception as e:
    print(f'✗ Import error: {e}')
"

# 检查特征文件完整性
python -c "
import numpy as np
import os

def check_features(path):
    if not os.path.exists(path):
        print(f'  ✗ File not found: {path}')
        return False
    try:
        data = np.load(path)
        required = ['pointnet_features', 'ulip_features', 'labels']
        missing = [k for k in required if k not in data]
        if missing:
            print(f'  ✗ Missing keys: {missing}')
            return False
        print(f'  ✓ Features OK: {data[\"pointnet_features\"].shape}, {data[\"ulip_features\"].shape}')
        return True
    except Exception as e:
        print(f'  ✗ Error loading: {e}')
        return False

print('Checking feature files:')
check_features('feature_cache_full/train_features.npz')
check_features('feature_cache_full/test_features.npz')
check_features('ablation_features/train_features.npz')
check_features('ablation_features/test_features.npz')
"
```

## 环境变量与配置

### 14. 设置环境变量（可选）
```bash
# 设置PyTorch环境变量

# 设置CUDA设备
export CUDA_VISIBLE_DEVICES=0

# 设置随机种子（全局）
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8  # 对于确定性CUDA操作
```

## 清理与重置

### 15. 清理实验数据（谨慎使用）
```bash
# 删除特征缓存（释放磁盘空间）
# rm -rf feature_cache_full/ ablation_features/

# 删除实验结果
# rm -rf ablation_results_real/ ablation_results_dummy/ ablation_results_separate/

# 删除日志文件
# rm -f ablation.log training.log
```

---

## 预期输出验证

运行完成后，检查以下关键指标：

1. **特征提取成功**：
   - `feature_cache_full/train_features.npz` 存在且ULIP特征维度为1280
   - `ablation_features/train_features.npz` 存在且ULIP特征维度为256

2. **消融实验成功**：
   - 所有实验状态为"success"
   - ULIP-only准确率：dummy ~49%，real ~92%
   - Fusion准确率：dummy ~90%，real ~93%

3. **融合有效性**：
   - 真实特征下，Fusion > ULIP-only（~1.7%增益）
   - dummy特征下，Fusion可能 <= PointNet-only

## 执行时间参考
- **特征提取**：~15-20分钟（GPU）
- **消融实验**：~10-15分钟（GPU）
- **总时间**：~25-35分钟（GPU）

## 命令执行顺序

建议按以下顺序执行命令：

1. **环境准备**：1-2
2. **特征提取**：3-5
3. **消融实验**：6-8（选择一种方式）
4. **结果验证**：9-11
5. **分析报告**：12
6. **故障排除**：13（如有问题）

**提示**：所有命令都设计为可复制粘贴执行。按顺序执行即可完成完整的消融实验复现。

---

*文档创建时间：2026-04-08*  
*基于项目：ULIP-2 + PointNet2 Fusion Pipeline*  
*位置：docs/ABLATION_EXPERIMENT_COMMAND_GUIDE.md*