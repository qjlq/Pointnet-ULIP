#!/usr/bin/env python
"""Validate pipeline components and dependencies."""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from project_utils.logger import PipelineLogger

def main():
    logger = PipelineLogger("validation")
    logger.info("Starting pipeline validation...")
    
    # Check directory structure
    required_dirs = ["scripts", "libs", "utils", "config", "tests"]
    for dir_name in required_dirs:
        if not os.path.exists(dir_name):
            logger.error(f"Missing directory: {dir_name}")
            return False
        logger.info(f"✓ Directory exists: {dir_name}")
    
    # Check required files
    required_files = [
        "scripts/extract_features.py",
        "scripts/train_fusion.py",
        "scripts/run_fusion_pipeline.py",
        "config/fusion_config.yaml",
        "utils/config_parser.py",
        "utils/logger.py",
        "utils/checkpoint_manager.py",
        "libs/pointnet_extractor.py",
        "libs/ulip_extractor.py",
        "libs/dataset_loader.py",
        "train_utils.py",
        "fusion_model.py"
    ]
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            logger.error(f"Missing file: {file_path}")
            return False
        logger.info(f"✓ File exists: {file_path}")
    
    # Check Python dependencies
    try:
        import torch
        import numpy as np
        import yaml
        import tqdm
        import scipy
        import sklearn
        logger.info("✓ All Python dependencies are importable")
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        return False
    
    # Check torch version and CUDA availability
    logger.info(f"PyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        logger.info(f"CUDA is available: {torch.cuda.get_device_name(0)}")
    else:
        logger.warning("CUDA is not available - using CPU")
    
    # Check that original codebases exist
    original_codebases = [
        ("pointnet_project", "PointNet2 codebase"),
        ("ULIP", "ULIP-2 codebase")
    ]
    
    for dir_path, description in original_codebases:
        if not os.path.exists(dir_path):
            logger.warning(f"Original {description} not found at {dir_path}")
            logger.warning("Feature extraction may fail if checkpoints are missing")
        else:
            logger.info(f"✓ {description} exists: {dir_path}")
    
    # Validate config file structure
    try:
        from project_utils.config_parser import ConfigParser
        parser = ConfigParser()
        config = parser.load("config/fusion_config.yaml")
        required_config_sections = ["data", "models", "extraction", "training", "logging"]
        for section in required_config_sections:
            if section not in config:
                logger.warning(f"Config section '{section}' not found in config file")
            else:
                logger.info(f"✓ Config section exists: {section}")
    except Exception as e:
        logger.error(f"Failed to load config file: {e}")
        return False
    
    logger.info("All pipeline components validated successfully!")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)