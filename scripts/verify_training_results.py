#!/usr/bin/env python
# scripts/verify_training_results.py
import os
import torch
import json
import sys

def verify_training_results(experiment_dir):
    """验证训练结果完整性"""
    print(f"Verifying training results in: {experiment_dir}")
    
    # 1. 检查目录存在
    if not os.path.exists(experiment_dir):
        raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")
    
    # 2. 检查检查点文件
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    required_files = ["best_model.pth", "final_model.pth"]
    
    for file in required_files:
        file_path = os.path.join(checkpoint_dir, file)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Required checkpoint file missing: {file_path}")
        print(f"✓ Found checkpoint: {file}")
    
    # 3. 检查模型文件可加载性
    try:
        best_model = torch.load(os.path.join(checkpoint_dir, "best_model.pth"), map_location="cpu")
        final_model = torch.load(os.path.join(checkpoint_dir, "final_model.pth"), map_location="cpu")
        print("✓ Checkpoint files can be loaded")
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint files: {e}")
    
    # 4. 检查训练日志文件
    log_files = [f for f in os.listdir(experiment_dir) if f.endswith('.log')]
    if len(log_files) > 0:
        print(f"✓ Found {len(log_files)} log file(s)")
    else:
        print("⚠ No log files found")
    
    # 5. 检查配置文件副本
    config_files = [f for f in os.listdir(experiment_dir) if 'config' in f.lower()]
    if len(config_files) > 0:
        print(f"✓ Found config file(s): {config_files}")
    else:
        print("⚠ No config file found")
    
    # 6. 检查训练摘要（如果存在）
    summary_files = [f for f in os.listdir(experiment_dir) if 'summary' in f.lower()]
    if len(summary_files) > 0:
        print(f"✓ Found summary file(s): {summary_files}")
        
        # 尝试读取第一个摘要文件
        summary_path = os.path.join(experiment_dir, summary_files[0])
        try:
            with open(summary_path, 'r') as f:
                summary = json.load(f)
                if 'final_test_accuracy' in summary:
                    accuracy = summary['final_test_accuracy']
                    print(f"  Final test accuracy: {accuracy:.2%}")
                    
                    # 验证准确率 > 随机猜测（2.5%）
                    if accuracy > 0.025:
                        print("✓ Accuracy exceeds random guessing threshold (2.5%)")
                    else:
                        print("⚠ Accuracy below random guessing threshold")
        except Exception as e:
            print(f"  Could not read summary file: {e}")
    else:
        print("⚠ No summary file found")
    
    # 7. 检查总文件数量
    all_files = []
    for root, dirs, files in os.walk(experiment_dir):
        for file in files:
            all_files.append(os.path.join(root, file))
    
    print(f"Total files in experiment directory: {len(all_files)}")
    
    if len(all_files) >= 3:  # At least 2 checkpoints + something else
        print("✓ Experiment directory contains sufficient files")
    else:
        print("⚠ Experiment directory may be incomplete")
    
    print("\n✓ Training results verification completed successfully!")
    return True

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_dir", required=True, help="Path to experiment directory")
    args = parser.parse_args()
    
    try:
        verify_training_results(args.experiment_dir)
        sys.exit(0)
    except Exception as e:
        print(f"✗ Verification failed: {e}")
        sys.exit(1)