# libs/pointnet_extractor.py
import sys
import os
from pathlib import Path
from typing import Optional, Tuple, Union, List
import torch
import torch.nn as nn

# Get absolute path to pointnet_project relative to this file
_CURRENT_DIR = Path(__file__).parent.absolute()
_POINTNET_ROOT = _CURRENT_DIR / ".." / ".." / "pointnet_project" / "Pointnet_Pointnet2_pytorch"
_POINTNET_ROOT = _POINTNET_ROOT.resolve()

# Add to sys.path (only if not already present)
def _add_pointnet_to_path():
    root_str = str(_POINTNET_ROOT)
    models_str = str(_POINTNET_ROOT / "models")
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    if models_str not in sys.path:
        sys.path.insert(0, models_str)

_add_pointnet_to_path()

from models.pointnet2_cls_ssg import get_model


class PointNetExtractor:
    """PointNet2 feature extractor for geometric feature extraction.
    
    This class loads a pre-trained PointNet2 model and extracts 1024-dimensional
    global geometric features from point clouds.
    
    Args:
        checkpoint_path: Path to the pre-trained checkpoint file (.pth)
        device: Device to load the model on ('cuda' or 'cpu')
        num_class: Number of classes for the model (default 40)
        normal_channel: Whether point clouds have normal channels (6D input)
    
    Raises:
        FileNotFoundError: If checkpoint file does not exist
        RuntimeError: If checkpoint loading fails
    """
    
    def __init__(self, checkpoint_path: str, device: str = 'cuda', 
                 num_class: int = 40, normal_channel: bool = False):
        self.device = device if torch.cuda.is_available() and device == 'cuda' else 'cpu'
        self.normal_channel = normal_channel
        self.num_class = num_class
        self.model = self.load_model(checkpoint_path)
        self.model.eval()

    @property
    def feature_dim(self) -> int:
        return 1024
        
    def load_model(self, checkpoint_path: str) -> nn.Module:
        """Load PointNet2 model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Loaded PointNet2 model in evaluation mode
            
        Raises:
            FileNotFoundError: If checkpoint file does not exist
            RuntimeError: If checkpoint loading or model initialization fails
        """
        # Check file existence
        checkpoint_file = Path(checkpoint_path)
        if not checkpoint_file.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file}")
        
        # Initialize model
        model = get_model(num_class=self.num_class, normal_channel=self.normal_channel)
        
        try:
            # Load checkpoint (weights_only=False due to numpy scalars in checkpoint)
            checkpoint = torch.load(str(checkpoint_file), map_location=torch.device(self.device), weights_only=False)
            
            # Extract state dict
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
                
            # Handle DataParallel prefix
            if any(k.startswith('module.') for k in state_dict.keys()):
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                
            model.load_state_dict(state_dict)
            
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint from {checkpoint_file}: {e}")
        
        model = model.to(self.device)
        return model
    
    def extract_features(self, point_clouds: torch.Tensor) -> torch.Tensor:
        """Extract global geometric features from point clouds.
        
        Args:
            point_clouds: Point cloud tensor of shape (B, C, N) or (B, N, C) where
                         C = 3 (xyz) or 6 (xyz+normals) depending on normal_channel setting.
                         N is the number of points (typically 1024).
        
        Returns:
            Global features tensor of shape (B, 1024)
            
        Raises:
            ValueError: If points tensor has invalid shape or dimensions
            RuntimeError: If feature extraction fails
        """
        # Validate input dimensions
        if point_clouds.dim() != 3:
            raise ValueError(f"Points tensor must be 3D (B, C, N) or (B, N, C), got shape {point_clouds.shape}")
        
        # Determine expected channel dimension based on normal_channel
        expected_channels = 6 if self.normal_channel else 3
        
        # Handle different input formats
        if point_clouds.shape[1] == expected_channels:
            # (B, C, N) format, PointNet2 expected format
            pass
        elif point_clouds.shape[2] == expected_channels:
            # (B, N, C) format, convert to (B, C, N)
            point_clouds = point_clouds.transpose(1, 2)
        else:
            raise ValueError(
                f"Invalid point cloud shape {point_clouds.shape}. "
                f"Expected channels={expected_channels} in dim 1 or 2. "
                f"normal_channel={self.normal_channel}"
            )
        
        # Validate point count (optional but recommended for performance)
        # PointNet2 typically expects 1024 points but can handle others
        num_points = point_clouds.shape[2]
        if num_points != 1024:
            print(f"⚠️ Warning: PointNet2 expects 1024 points, got {num_points}. "
                  f"Model may need resampling for optimal performance.")
        
        try:
            with torch.no_grad():
                # Forward pass: model returns (logits, features)
                _, features = self.model(point_clouds.to(self.device))
                
                # Handle different feature tensor shapes
                if features.dim() == 3:
                    # Shape: (B, 1024, 1) -> squeeze to (B, 1024)
                    features = features.squeeze(-1)
                elif features.dim() != 2:
                    raise RuntimeError(
                        f"Unexpected features shape {features.shape}. "
                        f"Expected (B, 1024) or (B, 1024, 1)."
                    )
                
                # Final validation
                if features.shape[1] != 1024:
                    raise RuntimeError(
                        f"Unexpected feature dimension {features.shape[1]}. "
                        f"Expected 1024."
                    )
                    
        except Exception as e:
            raise RuntimeError(f"Feature extraction failed: {e}")
        
        return features.cpu()