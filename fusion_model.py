import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureNormalizer(nn.Module):
    """Normalize features with optional learnable scaling.
    
    Modes:
        'l2': L2 normalization (divides by L2 norm)
        'layer': LayerNorm (learnable affine)
        'none': identity (no normalization)
    """
    def __init__(self, mode='l2', scale=True, dim=None):
        super().__init__()
        self.mode = mode
        self.scale = scale
        
        if scale:
            self.scale_param = nn.Parameter(torch.ones(1))
        
        if mode == 'layer':
            self.layernorm = nn.LayerNorm(dim) if dim else nn.LayerNorm(1)
    
    def forward(self, x):
        if self.mode == 'l2':
            # L2 normalize along feature dimension
            norm = torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=1e-8)
            x = x / norm
            
            if self.scale:
                x = x * self.scale_param
        elif self.mode == 'layer':
            x = self.layernorm(x)
        # 'none' does nothing
        
        return x
    
    def extra_repr(self):
        return f"mode={self.mode}, scale={self.scale}"


class NormalizedFusionHead(nn.Module):
    """Fusion head with feature normalization before concatenation."""
    def __init__(self, geo_dim, vlm_dim, hidden_dim=256, num_classes=40, 
                 dropout_rate=0.3, norm_mode='l2'):
        super().__init__()
        
        # Feature normalizers
        self.geo_norm = FeatureNormalizer(mode=norm_mode, scale=True, dim=geo_dim)
        self.vlm_norm = FeatureNormalizer(mode=norm_mode, scale=True, dim=vlm_dim)
        
        # Fusion MLP
        self.fc1 = nn.Linear(geo_dim + vlm_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, geo_features, vlm_features):
        # Normalize features
        geo_norm = self.geo_norm(geo_features)
        vlm_norm = self.vlm_norm(vlm_features)
        
        # Concatenate normalized features
        fused = torch.cat([geo_norm, vlm_norm], dim=1)
        
        # MLP classification
        out = self.fc1(fused)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out


class GatedFusionHead(nn.Module):
    """Fusion head with gated feature weighting before concatenation."""
    def __init__(self, geo_dim, vlm_dim, hidden_dim=256, num_classes=40, 
                 dropout_rate=0.3, gate_hidden_dim=128):
        super().__init__()
        self.geo_dim = geo_dim
        self.vlm_dim = vlm_dim
        
        # Gate networks: take concatenated features, output gate vectors for each feature type
        self.gate_network = nn.Sequential(
            nn.Linear(geo_dim + vlm_dim, gate_hidden_dim),
            nn.ReLU(),
            nn.Linear(gate_hidden_dim, geo_dim + vlm_dim),
            nn.Sigmoid()
        )
        
        # Fusion MLP (same as original)
        self.fc1 = nn.Linear(geo_dim + vlm_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, geo_features, vlm_features):
        # Concatenate features for gate computation
        combined = torch.cat([geo_features, vlm_features], dim=1)
        
        # Compute gates (values between 0 and 1)
        gates = self.gate_network(combined)
        
        # Split gates for geo and vlm features
        geo_gate = gates[:, :self.geo_dim]
        vlm_gate = gates[:, self.geo_dim:self.geo_dim + self.vlm_dim]
        
        # Apply gating
        gated_geo = geo_features * geo_gate
        gated_vlm = vlm_features * vlm_gate
        
        # Concatenate gated features
        fused = torch.cat([gated_geo, gated_vlm], dim=1)
        
        # MLP classification
        out = self.fc1(fused)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out


class FeatureFusionHead(nn.Module):
    """Feature fusion head combining geometric and semantic features.
    
    This module concatenates PointNet2 geometric features (1024D) with
    ULIP-2 semantic features (1280D for real model, 256D for dummy) and passes them through a single
    hidden layer MLP for classification.
    
    Architecture:
        Input: 1024 (geo) + vlm_dim (default 1280) = 2304D
        FC1: (geo_dim + vlm_dim) → hidden_dim (default 256) (with BatchNorm + ReLU + Dropout)
        FC2: hidden_dim → num_classes (output logits)
    
    Args:
        geo_dim: Dimension of geometric features (default: 1024)
        vlm_dim: Dimension of semantic features (default: 512)
        hidden_dim: Dimension of hidden layer (default: 256)
        num_classes: Number of output classes (default: 40 for ModelNet40)
        dropout_rate: Dropout rate after hidden layer (default: 0.3)
    """
    
    def __init__(self, geo_dim: int = 1024, vlm_dim: int = 1280, 
                 hidden_dim: int = 256, num_classes: int = 40, 
                 dropout_rate: float = 0.3):
        super().__init__()
        self.geo_dim = geo_dim
        self.vlm_dim = vlm_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        self.fc1 = nn.Linear(geo_dim + vlm_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Kaiming normal for linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, f_geo: torch.Tensor, f_vlm: torch.Tensor) -> torch.Tensor:
        """Forward pass of the fusion head.
        
        Args:
            f_geo: Geometric features tensor of shape (B, geo_dim)
            f_vlm: Semantic features tensor of shape (B, vlm_dim)
        
        Returns:
            Classification logits tensor of shape (B, num_classes)
            
        Raises:
            ValueError: If feature dimensions do not match expected dimensions
        """
        # Validate input dimensions
        if f_geo.shape[1] != self.geo_dim:
            raise ValueError(
                f"Geometric feature dimension mismatch: expected {self.geo_dim}, "
                f"got {f_geo.shape[1]}"
            )
        if f_vlm.shape[1] != self.vlm_dim:
            raise ValueError(
                f"Semantic feature dimension mismatch: expected {self.vlm_dim}, "
                f"got {f_vlm.shape[1]}"
            )
        if f_geo.shape[0] != f_vlm.shape[0]:
            raise ValueError(
                f"Batch size mismatch: f_geo batch {f_geo.shape[0]}, "
                f"f_vlm batch {f_vlm.shape[0]}"
            )
        
        # Concatenate features along channel dimension
        combined = torch.cat([f_geo, f_vlm], dim=-1)  # (B, geo_dim + vlm_dim)
        
        # Forward through MLP
        x = F.relu(self.bn1(self.fc1(combined)))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def get_feature_dimensions(self) -> dict:
        """Get the feature dimensions of the fusion head.
        
        Returns:
            Dictionary with geo_dim, vlm_dim, hidden_dim, num_classes
        """
        return {
            'geo_dim': self.geo_dim,
            'vlm_dim': self.vlm_dim,
            'hidden_dim': self.hidden_dim,
            'num_classes': self.num_classes,
            'total_input_dim': self.geo_dim + self.vlm_dim
        }


class FusionModel(FeatureFusionHead):
    """Alias for FeatureFusionHead with parameter name mapping for pipeline compatibility.
    
    Maps:
        pointnet_dim -> geo_dim (geometric features from PointNet)
        ulip_dim -> vlm_dim (semantic features from ULIP-2)
    
    All other parameters (hidden_dim, num_classes, dropout_rate) pass through unchanged.
    """
    
    def __init__(self, pointnet_dim: int = 1024, ulip_dim: int = 1280, **kwargs):
        """Initialize fusion model with pipeline parameter names.
        
        Args:
            pointnet_dim: Dimension of PointNet geometric features
            ulip_dim: Dimension of ULIP semantic features
            **kwargs: Additional arguments passed to FeatureFusionHead
                (hidden_dim, num_classes, dropout_rate)
        """
        # Remove geo_dim and vlm_dim from kwargs to avoid duplicate arguments
        kwargs.pop('geo_dim', None)
        kwargs.pop('vlm_dim', None)
        super().__init__(geo_dim=pointnet_dim, vlm_dim=ulip_dim, **kwargs)


class PointNetOnlyClassifier(nn.Module):
    """Classifier using only PointNet geometric features.
    
    Architecture:
        Input: input_dim (PointNet feature dimension)
        FC1: input_dim → hidden_dim (with BatchNorm + ReLU + Dropout)
        FC2: hidden_dim → num_classes (output logits)
    
    Args:
        input_dim: Dimension of PointNet features (default: 1024)
        hidden_dim: Dimension of hidden layer (default: 256)
        num_classes: Number of output classes (default: 40 for ModelNet40)
        dropout_rate: Dropout rate after hidden layer (default: 0.3)
    """
    
    def __init__(self, input_dim: int = 1024, hidden_dim: int = 256, 
                 num_classes: int = 40, dropout_rate: float = 0.3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Kaiming normal for linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass of the PointNet-only classifier.
        
        Args:
            features: PointNet features tensor of shape (B, input_dim)
        
        Returns:
            Classification logits tensor of shape (B, num_classes)
        """
        if features.shape[1] != self.input_dim:
            raise ValueError(
                f"Feature dimension mismatch: expected {self.input_dim}, "
                f"got {features.shape[1]}"
            )
        
        x = F.relu(self.bn1(self.fc1(features)))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def get_feature_dimensions(self) -> dict:
        """Get the feature dimensions of the classifier.
        
        Returns:
            Dictionary with input_dim, hidden_dim, num_classes
        """
        return {
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'num_classes': self.num_classes
        }


class ULIPOnlyClassifier(nn.Module):
    """Classifier using only ULIP semantic features.
    
    Architecture:
        Input: input_dim (ULIP feature dimension)
        FC1: input_dim → hidden_dim (with BatchNorm + ReLU + Dropout)
        FC2: hidden_dim → num_classes (output logits)
    
    Args:
        input_dim: Dimension of ULIP features (default: 1280 for real model)
        hidden_dim: Dimension of hidden layer (default: 256)
        num_classes: Number of output classes (default: 40 for ModelNet40)
        dropout_rate: Dropout rate after hidden layer (default: 0.3)
    """
    
    def __init__(self, input_dim: int = 1280, hidden_dim: int = 256, 
                 num_classes: int = 40, dropout_rate: float = 0.3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Kaiming normal for linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass of the ULIP-only classifier.
        
        Args:
            features: ULIP features tensor of shape (B, input_dim)
        
        Returns:
            Classification logits tensor of shape (B, num_classes)
        """
        if features.shape[1] != self.input_dim:
            raise ValueError(
                f"Feature dimension mismatch: expected {self.input_dim}, "
                f"got {features.shape[1]}"
            )
        
        x = F.relu(self.bn1(self.fc1(features)))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def get_feature_dimensions(self) -> dict:
        """Get the feature dimensions of the classifier.
        
        Returns:
            Dictionary with input_dim, hidden_dim, num_classes
        """
        return {
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'num_classes': self.num_classes
        }