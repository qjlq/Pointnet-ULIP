"""Unit tests for fusion improvement modules."""

import torch
import torch.nn as nn
import pytest


def test_feature_normalizer_l2():
    """Test L2 normalization with learnable scaling."""
    # Import will fail initially
    from fusion_model import FeatureNormalizer
    
    norm = FeatureNormalizer(mode='l2', scale=True)
    x = torch.randn(4, 1024) * 5.0
    out = norm(x)
    
    # Output should have unit L2 norm per sample
    norms = torch.norm(out, p=2, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), rtol=1e-4)
    
    # Scaling should be learnable
    assert hasattr(norm, 'scale_param')


def test_feature_normalizer_l2_no_scale():
    """Test L2 normalization without scaling."""
    from fusion_model import FeatureNormalizer
    
    norm = FeatureNormalizer(mode='l2', scale=False)
    x = torch.randn(4, 1024) * 5.0
    out = norm(x)
    
    # Output should have unit L2 norm per sample
    norms = torch.norm(out, p=2, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), rtol=1e-4)
    # No scale_param should exist
    assert not hasattr(norm, 'scale_param')


def test_feature_normalizer_layer():
    """Test layer normalization mode."""
    from fusion_model import FeatureNormalizer
    
    norm = FeatureNormalizer(mode='layer', dim=1024)
    x = torch.randn(4, 1024) * 5.0
    out = norm(x)
    
    # Output shape should match input
    assert out.shape == x.shape
    # Should have layernorm attribute
    assert hasattr(norm, 'layernorm')


def test_feature_normalizer_none():
    """Test no normalization mode."""
    from fusion_model import FeatureNormalizer
    
    norm = FeatureNormalizer(mode='none')
    x = torch.randn(4, 1024) * 5.0
    out = norm(x)
    
    # Output should be identical to input
    assert torch.allclose(out, x, rtol=1e-6)


def test_feature_normalizer_gradient():
    """Test that gradients flow through normalization."""
    from fusion_model import FeatureNormalizer
    
    norm = FeatureNormalizer(mode='l2', scale=True)
    # Create leaf tensor
    x = torch.randn(4, 1024) * 5.0
    x.requires_grad_(True)
    out = norm(x)
    
    # Compute gradient
    loss = out.sum()
    loss.backward()
    
    # Check gradient exists (x is leaf)
    assert x.grad is not None
    assert not torch.allclose(x.grad, torch.zeros_like(x.grad))
    
    # Scale parameter gradient should exist
    if hasattr(norm, 'scale_param'):
        assert norm.scale_param.grad is not None


def test_normalized_fusion_head():
    """Test normalized fusion head with L2 normalization."""
    from fusion_model import NormalizedFusionHead
    
    head = NormalizedFusionHead(
        geo_dim=1024,
        vlm_dim=256,
        hidden_dim=256,
        num_classes=40,
        dropout_rate=0.3,
        norm_mode='l2'
    )
    
    # Test forward pass
    geo_features = torch.randn(8, 1024)
    vlm_features = torch.randn(8, 256)
    
    output = head(geo_features, vlm_features)
    assert output.shape == (8, 40)


def test_normalized_fusion_head_layer_norm():
    """Test normalized fusion head with layer normalization."""
    from fusion_model import NormalizedFusionHead
    
    head = NormalizedFusionHead(
        geo_dim=1024,
        vlm_dim=256,
        hidden_dim=256,
        num_classes=40,
        dropout_rate=0.3,
        norm_mode='layer'
    )
    
    geo_features = torch.randn(8, 1024)
    vlm_features = torch.randn(8, 256)
    
    output = head(geo_features, vlm_features)
    assert output.shape == (8, 40)
    # Check that normalizers are layer norm
    assert head.geo_norm.mode == 'layer'
    assert head.vlm_norm.mode == 'layer'


def test_normalized_fusion_head_no_norm():
    """Test normalized fusion head with no normalization."""
    from fusion_model import NormalizedFusionHead
    
    head = NormalizedFusionHead(
        geo_dim=1024,
        vlm_dim=256,
        hidden_dim=256,
        num_classes=40,
        dropout_rate=0.3,
        norm_mode='none'
    )
    
    geo_features = torch.randn(8, 1024)
    vlm_features = torch.randn(8, 256)
    
    output = head(geo_features, vlm_features)
    assert output.shape == (8, 40)
    # Normalizers should be in 'none' mode
    assert head.geo_norm.mode == 'none'
    assert head.vlm_norm.mode == 'none'


def test_normalized_fusion_head_dimension_validation():
    """Test that dimension mismatches raise errors."""
    from fusion_model import NormalizedFusionHead
    
    head = NormalizedFusionHead(
        geo_dim=1024,
        vlm_dim=256,
        hidden_dim=256,
        num_classes=40
    )
    
    # Wrong geo dimension
    geo_features = torch.randn(8, 512)  # Should be 1024
    vlm_features = torch.randn(8, 256)
    
    with pytest.raises(RuntimeError):
        # This will fail in linear layer due to dimension mismatch
        output = head(geo_features, vlm_features)


def test_gated_fusion_head():
    """Test gated fusion head forward pass."""
    from fusion_model import GatedFusionHead
    
    head = GatedFusionHead(
        geo_dim=1024,
        vlm_dim=256,
        hidden_dim=256,
        num_classes=40
    )
    
    geo_features = torch.randn(8, 1024)
    vlm_features = torch.randn(8, 256)
    
    output = head(geo_features, vlm_features)
    assert output.shape == (8, 40)
    # Check that gate network exists
    assert hasattr(head, 'gate_network')


def test_gated_fusion_head_gradient():
    """Test gradients flow through gated fusion head."""
    from fusion_model import GatedFusionHead
    
    head = GatedFusionHead(
        geo_dim=1024,
        vlm_dim=256,
        hidden_dim=256,
        num_classes=40
    )
    
    geo_features = torch.randn(8, 1024, requires_grad=True)
    vlm_features = torch.randn(8, 256, requires_grad=True)
    
    output = head(geo_features, vlm_features)
    loss = output.sum()
    loss.backward()
    
    # Gradients should exist
    assert geo_features.grad is not None
    assert vlm_features.grad is not None
    assert not torch.allclose(geo_features.grad, torch.zeros_like(geo_features.grad))
    assert not torch.allclose(vlm_features.grad, torch.zeros_like(vlm_features.grad))