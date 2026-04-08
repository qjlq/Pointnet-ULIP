import os
import yaml
import argparse
from typing import Dict, Any

class ConfigParser:
    def __init__(self):
        pass
    
    def load(self, config_path: str) -> Dict[str, Any]:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def merge_with_args(self, config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
        merged = config.copy()
        for key, value in vars(args).items():
            if value is not None:
                merged[key] = value
        return merged