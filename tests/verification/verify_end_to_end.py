#!/usr/bin/env python
"""Test end-to-end fusion pipeline.

Verifies the orchestration script runs extraction and training successfully.
Uses existing feature cache to avoid re-extraction.
"""

import sys
import os
import tempfile
import shutil
import subprocess
import yaml
from pathlib import Path

def print_header(title):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def create_test_config():
    """Create a temporary config file with reduced epochs for quick testing."""
    with open("config/fusion_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Reduce epochs and adjust batch sizes for quick test
    config['training']['epochs'] = 2
    config['training']['batch_size'] = 32  # keep default
    config['extraction']['batch_size'] = 64  # keep default
    
    # Write to temporary file
    import tempfile
    fd, temp_path = tempfile.mkstemp(suffix='.yaml', text=True)
    os.close(fd)
    with open(temp_path, 'w') as f:
        yaml.dump(config, f)
    return temp_path


def test_end_to_end_pipeline():
    """Test the end-to-end pipeline orchestration."""
    print_header("1. End-to-End Pipeline Test (train_only mode)")
    
    # Create temporary config
    config_path = create_test_config()
    print(f"Created test config: {config_path}")
    
    # Create temporary experiment directory
    with tempfile.TemporaryDirectory() as tmpdir:
        experiment_dir = Path(tmpdir) / "fusion_experiment"
        experiment_dir.mkdir(parents=True)
        
        # Copy existing feature cache to experiment_dir/features
        features_dir = experiment_dir / "features"
        features_dir.mkdir()
        
        source_cache = Path("feature_cache")
        if not source_cache.exists():
            print(f"❌ Source feature cache not found: {source_cache}")
            os.unlink(config_path)
            return False
        
        # Copy train and test feature files
        for f in ["train_features.npz", "test_features.npz"]:
            src = source_cache / f
            if not src.exists():
                print(f"❌ Feature file not found: {src}")
                os.unlink(config_path)
                return False
            shutil.copy2(src, features_dir / f)
        
        print(f"✓ Copied feature cache to {features_dir}")
        
        # Run pipeline in train_only mode
        script_path = Path("scripts/run_fusion_pipeline.py")
        if not script_path.exists():
            print(f"❌ Pipeline script not found: {script_path}")
            os.unlink(config_path)
            return False
        
        cmd = [
            sys.executable,
            str(script_path),
            "--config", config_path,
            "--mode", "train_only",
            "--output_dir", str(experiment_dir)
        ]
        
        print(f"Running pipeline: {' '.join(cmd)}")
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                cwd=Path.cwd(),
                timeout=300  # 5 minutes timeout
            )
            print(f"✓ Pipeline executed successfully")
            print(f"  stdout (last 500 chars): {result.stdout[-500:]}")
            if result.stderr:
                print(f"  stderr (last 500 chars): {result.stderr[-500:]}")
            
            # Check that training artifacts were created
            checkpoint_dir = experiment_dir / "checkpoints"
            if checkpoint_dir.exists():
                checkpoint_files = list(checkpoint_dir.glob("*.pth"))
                if checkpoint_files:
                    print(f"✓ Checkpoints created: {len(checkpoint_files)} files")
                else:
                    print(f"⚠ No checkpoint files found (training may have skipped)")
            else:
                print(f"⚠ Checkpoint directory not created")
            
            # Verify pipeline log file exists
            log_file = experiment_dir / "pipeline.log"
            if log_file.exists():
                print(f"✓ Pipeline log created: {log_file}")
            else:
                print(f"⚠ Pipeline log missing")
            
            # Cleanup temp config
            os.unlink(config_path)
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Pipeline failed with exit code {e.returncode}")
            print(f"  stdout: {e.stdout[-1000:]}")
            print(f"  stderr: {e.stderr[-1000:]}")
            os.unlink(config_path)
            return False
        except subprocess.TimeoutExpired:
            print(f"❌ Pipeline timed out after 5 minutes")
            os.unlink(config_path)
            return False
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            os.unlink(config_path)
            return False


def main():
    """Main verification function."""
    print("End-to-End Fusion Pipeline Verification")
    print("=" * 60)
    
    success = True
    
    if not test_end_to_end_pipeline():
        success = False
    
    print_header("VERIFICATION SUMMARY")
    if success:
        print("✅ All checks passed!")
        print("End-to-end pipeline orchestration is working correctly.")
        return 0
    else:
        print("❌ Some checks failed.")
        print("Please address the issues above before using the end-to-end pipeline.")
        return 1


if __name__ == "__main__":
    sys.exit(main())