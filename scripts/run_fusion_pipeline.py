#!/usr/bin/env python
# scripts/run_fusion_pipeline.py
import os
import sys
import argparse
import datetime
import json
import subprocess
import yaml
from typing import Dict, Any, Optional, Tuple, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from project_utils.logger import PipelineLogger
from project_utils.config_parser import ConfigParser


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    """Parse command line arguments for the fusion pipeline.
    
    Args:
        argv: Argument list (defaults to sys.argv[1:]).
        
    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(description="End-to-end fusion pipeline")
    parser.add_argument("--config", type=str, default="config/fusion_config.yaml", help="Path to config file")
    parser.add_argument("--mode", type=str, default="full", choices=["full", "extract_only", "train_only"], help="Pipeline mode")
    parser.add_argument("--output_dir", type=str, help="Output directory (default: auto-generated)")
    return parser.parse_args(argv)


def create_experiment_dir(base_dir: str = "fusion_experiments") -> str:
    """Create a timestamped experiment directory.
    
    Args:
        base_dir: Base directory for experiments.
        
    Returns:
        Path to created experiment directory.
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(base_dir, f"fusion_pipeline_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)
    return experiment_dir


def save_config(config: Dict[str, Any], experiment_dir: str) -> None:
    """Save config dictionary as JSON for reproducibility.
    
    Args:
        config: Configuration dictionary.
        experiment_dir: Experiment directory path.
    """
    config_path = os.path.join(experiment_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)


def save_command(args: argparse.Namespace, experiment_dir: str) -> None:
    """Save the command used to run the pipeline.
    
    Args:
        args: Parsed arguments.
        experiment_dir: Experiment directory path.
    """
    command_path = os.path.join(experiment_dir, "command.txt")
    with open(command_path, 'w') as f:
        f.write(" ".join(sys.argv))


def validate_config(config: Dict[str, Any], logger: PipelineLogger) -> bool:
    """Validate configuration structure and required fields.
    
    Args:
        config: Configuration dictionary.
        logger: Logger instance.
        
    Returns:
        True if config is valid, False otherwise.
    """
    required_sections = {
        'data': ['root_dir', 'num_points', 'normal_channel'],
        'models': ['pointnet_checkpoint'],
        'extraction': ['batch_size', 'device'],
        'training': ['epochs', 'batch_size', 'learning_rate', 'checkpoint_dir', 'save_interval']
    }
    
    for section, required_fields in required_sections.items():
        if section not in config:
            logger.error(f"Missing required config section: {section}")
            return False
        
        for field in required_fields:
            if field not in config[section]:
                logger.error(f"Missing required field '{field}' in config section '{section}'")
                return False
    
    # Additional validation
    if config['data']['num_points'] <= 0:
        logger.error(f"Invalid num_points: {config['data']['num_points']}. Must be positive.")
        return False
    
    if config['extraction']['batch_size'] <= 0:
        logger.error(f"Invalid extraction batch_size: {config['extraction']['batch_size']}. Must be positive.")
        return False
    
    if config['training']['epochs'] <= 0:
        logger.error(f"Invalid training epochs: {config['training']['epochs']}. Must be positive.")
        return False
    
    if config['training']['batch_size'] <= 0:
        logger.error(f"Invalid training batch_size: {config['training']['batch_size']}. Must be positive.")
        return False
    
    if config['training']['learning_rate'] <= 0:
        logger.error(f"Invalid learning_rate: {config['training']['learning_rate']}. Must be positive.")
        return False
    
    return True


def run_extraction(config: Dict[str, Any], experiment_dir: str, logger: PipelineLogger) -> Tuple[bool, str]:
    """Run feature extraction script.
    
    Args:
        config: Configuration dictionary.
        experiment_dir: Experiment directory path.
        logger: Logger instance.
        
    Returns:
        Tuple of (success: bool, features_dir: str).
    """
    logger.info("Running feature extraction...")
    features_dir = os.path.join(experiment_dir, "features")
    os.makedirs(features_dir, exist_ok=True)
    
    # Build command arguments safely using list
    cmd_args = [
        sys.executable, "scripts/extract_features.py",
        "--data_dir", config.get('data', {}).get('root_dir', ''),
        "--pointnet_checkpoint", config.get('models', {}).get('pointnet_checkpoint', ''),
        "--batch_size", str(config.get('extraction', {}).get('batch_size', 64)),
        "--device", config.get('extraction', {}).get('device', 'cuda'),
        "--output_dir", features_dir
    ]
    
    if config.get('extraction', {}).get('skip_if_cached', True):
        cmd_args.append("--skip_if_cached")
    
    ulip_checkpoint = config.get('models', {}).get('ulip_checkpoint')
    if ulip_checkpoint:
        cmd_args.extend(["--ulip_checkpoint", ulip_checkpoint])
    
    openclip_checkpoint = config.get('models', {}).get('openclip_checkpoint')
    if openclip_checkpoint:
        cmd_args.extend(["--openclip_checkpoint", openclip_checkpoint])
    
    cmd_str = " ".join(cmd_args)
    logger.info(f"Executing: {cmd_str}")
    
    try:
        result = subprocess.run(
            cmd_args,
            check=True,
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        logger.debug(f"Extraction stdout: {result.stdout[:500]}..." if len(result.stdout) > 500 else f"Extraction stdout: {result.stdout}")
        if result.stderr:
            logger.warning(f"Extraction stderr: {result.stderr[:500]}..." if len(result.stderr) > 500 else f"Extraction stderr: {result.stderr}")
        logger.info("Feature extraction completed successfully")
        return True, features_dir
    except subprocess.CalledProcessError as e:
        logger.error(f"Feature extraction failed with exit code {e.returncode}")
        logger.error(f"stdout: {e.stdout[:1000]}..." if len(e.stdout) > 1000 else f"stdout: {e.stdout}")
        logger.error(f"stderr: {e.stderr[:1000]}..." if len(e.stderr) > 1000 else f"stderr: {e.stderr}")
        return False, features_dir
    except Exception as e:
        logger.error(f"Unexpected error during feature extraction: {e}")
        return False, features_dir


def run_training(config: Dict[str, Any], experiment_dir: str, features_dir: str, logger: PipelineLogger) -> bool:
    """Run fusion training script.
    
    Args:
        config: Configuration dictionary.
        experiment_dir: Experiment directory path.
        features_dir: Path to features directory.
        logger: Logger instance.
        
    Returns:
        success: bool indicating if training succeeded.
    """
    logger.info("Running fusion training...")
    train_features = os.path.join(features_dir, "train_features.npz")
    test_features = os.path.join(features_dir, "test_features.npz")
    
    # Verify feature files exist
    if not os.path.exists(train_features):
        logger.error(f"Train features not found: {train_features}")
        return False
    if not os.path.exists(test_features):
        logger.error(f"Test features not found: {test_features}")
        return False
    
    # Build command arguments safely using list
    cmd_args = [
        sys.executable, "scripts/train_fusion.py",
        "--train_features", train_features,
        "--test_features", test_features,
        "--epochs", str(config.get('training', {}).get('epochs', 50)),
        "--batch_size", str(config.get('training', {}).get('batch_size', 32)),
        "--learning_rate", str(config.get('training', {}).get('learning_rate', 0.001)),
        "--checkpoint_dir", os.path.join(experiment_dir, "checkpoints"),
        "--output_dir", os.path.join(experiment_dir, "training_output")
    ]
    
    cmd_str = " ".join(cmd_args)
    logger.info(f"Executing: {cmd_str}")
    
    try:
        result = subprocess.run(
            cmd_args,
            check=True,
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        logger.debug(f"Training stdout: {result.stdout[:500]}..." if len(result.stdout) > 500 else f"Training stdout: {result.stdout}")
        if result.stderr:
            logger.warning(f"Training stderr: {result.stderr[:500]}..." if len(result.stderr) > 500 else f"Training stderr: {result.stderr}")
        logger.info("Fusion training completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Fusion training failed with exit code {e.returncode}")
        logger.error(f"stdout: {e.stdout[:1000]}..." if len(e.stdout) > 1000 else f"stdout: {e.stdout}")
        logger.error(f"stderr: {e.stderr[:1000]}..." if len(e.stderr) > 1000 else f"stderr: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during fusion training: {e}")
        return False


def main() -> None:
    """Main entry point for the fusion pipeline."""
    args = parse_args()
    experiment_dir = args.output_dir or create_experiment_dir()
    
    logger = PipelineLogger(
        name="fusion_pipeline",
        log_file=os.path.join(experiment_dir, "pipeline.log")
    )
    
    logger.info(f"Starting fusion pipeline in {experiment_dir}")
    
    # Load configuration
    config_parser = ConfigParser()
    try:
        config = config_parser.load(args.config)
    except FileNotFoundError:
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)
    except yaml.YAMLError as e:
        logger.error(f"Invalid YAML in config file: {e}")
        sys.exit(1)
    
    # Validate configuration
    if not validate_config(config, logger):
        sys.exit(1)
    
    # Save reproducibility artifacts
    save_config(config, experiment_dir)
    save_command(args, experiment_dir)
    
    # Track successes and failures
    successes: List[str] = []
    failures: List[str] = []
    features_dir: Optional[str] = None
    
    # Run extraction if requested
    if args.mode in ["full", "extract_only"]:
        extraction_success, extracted_features_dir = run_extraction(config, experiment_dir, logger)
        features_dir = extracted_features_dir
        if extraction_success:
            successes.append("feature extraction")
        else:
            failures.append("feature extraction")
    
    # Run training if requested
    if args.mode in ["full", "train_only"]:
        if args.mode == "train_only":
            # Assume features already exist in experiment_dir/features
            features_dir = os.path.join(experiment_dir, "features")
            if not os.path.exists(features_dir):
                logger.error(f"Features directory not found: {features_dir}")
                failures.append("feature loading (train_only mode)")
                features_dir = None
        
        if features_dir:
            training_success = run_training(config, experiment_dir, features_dir, logger)
            if training_success:
                successes.append("fusion training")
            else:
                failures.append("fusion training")
        elif args.mode == "full":
            # In full mode, training depends on extraction
            logger.warning("Skipping training because feature extraction failed")
            failures.append("fusion training (dependency failed)")
    
    # Final status report
    if successes:
        logger.info(f"Successful steps: {', '.join(successes)}")
    if failures:
        logger.error(f"Failed steps: {', '.join(failures)}")
    
    if not failures:
        logger.info(f"Pipeline completed successfully! Results in {experiment_dir}")
        sys.exit(0)
    elif successes:
        logger.warning(f"Pipeline completed with {len(failures)} failure(s) and {len(successes)} success(es). Results in {experiment_dir}")
        sys.exit(1)
    else:
        logger.error(f"Pipeline failed completely! No steps succeeded. Check logs in {experiment_dir}")
        sys.exit(1)


if __name__ == "__main__":
    main()