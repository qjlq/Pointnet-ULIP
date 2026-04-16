#!/usr/bin/env python3
"""Validate ScanObjectNN pipeline end-to-end."""
import argparse
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from project_utils.logger import PipelineLogger

def validate_data_loading(logger):
    """Validate ScanObjectNN data loading."""
    try:
        from libs.dataset_loader import get_dataset_loader
        
        loader = get_dataset_loader(
            dataset_type="scanobjectnn",
            root_dir="pointnet_project/Pointnet_Pointnet2_pytorch/data",
            split="train",
            use_background=True,
            split_name="main_split",
            batch_size=8,
            num_points=1024,
            num_workers=0
        )
        
        # Load one batch
        for points, labels in loader:
            logger.info(f"Points shape: {points.shape}")
            logger.info(f"Labels shape: {labels.shape}")
            logger.info(f"Unique labels: {labels.unique().tolist()}")
            break
            
        return True
    except Exception as e:
        logger.error(f"Data loading failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Validate ScanObjectNN pipeline")
    parser.add_argument("--output_dir", default="validation_output", help="Output directory")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logger with file output
    log_file = os.path.join(args.output_dir, "validation.log")
    logger = PipelineLogger("scanobjectnn_validation", log_file=log_file)
    logger.info("Starting ScanObjectNN pipeline validation...")
    
    # Check ScanObjectNN specific files
    required_files = [
        "libs/scanobjectnn_loader.py",
        "config/scanobjectnn_config.yaml",
        "tests/test_scanobjectnn_loader.py",
        "tests/test_scanobjectnn_integration.py",
    ]
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            logger.error(f"Missing file: {file_path}")
            return False
        logger.info(f"✓ File exists: {file_path}")
    
    # Check ScanObjectNN data directory
    scanobjectnn_data_dir = "pointnet_project/Pointnet_Pointnet2_pytorch/data/scanobjectnn/h5_files/main_split"
    if not os.path.exists(scanobjectnn_data_dir):
        logger.error(f"ScanObjectNN data directory not found: {scanobjectnn_data_dir}")
        logger.error("Please download ScanObjectNN dataset and place it in the correct location.")
        return False
    logger.info(f"✓ ScanObjectNN data directory exists: {scanobjectnn_data_dir}")
    
    # Check for HDF5 files
    h5_files = [
        "training_objectdataset_augmented25_norot.h5",
        "test_objectdataset_augmented25_norot.h5"
    ]
    for h5_file in h5_files:
        h5_path = os.path.join(scanobjectnn_data_dir, h5_file)
        if not os.path.exists(h5_path):
            logger.error(f"ScanObjectNN HDF5 file not found: {h5_file}")
            logger.error("Required data file missing. Validation failed.")
            return False
        else:
            logger.info(f"✓ HDF5 file exists: {h5_file}")
    
    # Check Python dependencies
    try:
        import torch
        import numpy as np
        import h5py
        logger.info("✓ All Python dependencies are importable")
    except ImportError as e:
        logger.error(f"Missing Python dependency: {e}")
        logger.error("Install required packages: torch, numpy, h5py")
        return False
    
    # Test basic imports
    try:
        from libs.scanobjectnn_loader import ScanObjectNNLoader
        from libs.dataset_loader import get_dataset_loader
        logger.info("✓ ScanObjectNN modules are importable")
    except ImportError as e:
        logger.error(f"Failed to import ScanObjectNN modules: {e}")
        return False
    
    # Validate data loading
    logger.info("1. Validating data loading...")
    if not validate_data_loading(logger):
        logger.error("Data loading validation failed")
        return False
    logger.info("✓ Data loading validation passed")
    
    logger.info("ScanObjectNN pipeline validation completed successfully!")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)