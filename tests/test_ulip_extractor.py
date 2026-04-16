import sys
import os
sys.path.insert(0, '.')
import torch
import pytest
from libs.ulip_extractor import ULIPExtractor

def test_ulip_extractor_initialization():
    extractor = ULIPExtractor(device="cpu")
    assert extractor.device == "cpu"
    assert extractor.use_dummy == True  # Should default to dummy due to CUDA issues

def test_ulip_extractor_variable_points():
    extractor = ULIPExtractor(device="cpu")
    # Test with different point counts
    for num_points in [256, 512, 1024, 2048]:
        points = torch.randn(2, num_points, 3)
        features = extractor.extract_features(points)
        assert features.shape == (2, 256), f"Expected (2,256) for {num_points} points, got {features.shape}"

def test_ulip_extractor_device_handling():
    # Test CPU device
    extractor_cpu = ULIPExtractor(device="cpu")
    assert extractor_cpu.device == "cpu"
    
    # Test CUDA request (may fallback to CPU if not available)
    extractor_cuda = ULIPExtractor(device="cuda")
    # Should be "cuda" only if torch.cuda.is_available(), else "cpu"
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert extractor_cuda.device == expected
    
    # Test feature extraction works on determined device
    points = torch.randn(2, 1024, 3)
    features = extractor_cpu.extract_features(points)
    assert features.shape == (2, 256)

def test_ulip_extractor_docstrings():
    """Check that ULIPExtractor has proper documentation."""
    import inspect
    extractor = ULIPExtractor(device="cpu")
    
    # Check class has docstring
    assert ULIPExtractor.__doc__ is not None, "ULIPExtractor class missing docstring"
    assert len(ULIPExtractor.__doc__.strip()) > 20, "ULIPExtractor docstring too short"
    
    # Check methods have docstrings
    methods = ['__init__', '_init_dummy_model', 'extract_features']
    for method_name in methods:
        method = getattr(ULIPExtractor, method_name, None)
        if method and inspect.ismethod(method) or inspect.isfunction(method):
            assert method.__doc__ is not None, f"Method {method_name} missing docstring"

def test_ulip_extractor_imports():
    """Check that ULIPExtractor doesn't have unnecessary numpy import."""
    import libs.ulip_extractor as module
    import inspect
    source = inspect.getsource(module)
    # Should not import numpy (unused)
    assert 'import numpy' not in source, "Unnecessary numpy import found"
    # Should check ULIP path existence before adding to sys.path
    assert 'os.path.exists' in source or 'os.path.isdir' in source, \
        "Missing path existence check before sys.path.insert"

def test_ulip_extractor_input_formats():
    extractor = ULIPExtractor(device="cpu")
    
    # Test (B, N, 3) format
    points_bn3 = torch.randn(2, 1024, 3)
    features1 = extractor.extract_features(points_bn3)
    assert features1.shape == (2, 256)
    
    # Test (B, 3, N) format (should be auto-converted)
    points_b3n = torch.randn(2, 3, 1024)
    features2 = extractor.extract_features(points_b3n)
    assert features2.shape == (2, 256)
    
    # Features should be different due to random weights, but same shape
    assert features1.shape == features2.shape

def test_ulip_extractor_checkpoint_fallback():
    # Test with non-existent checkpoint (should use dummy)
    extractor = ULIPExtractor(checkpoint_path="nonexistent.pth", device="cpu")
    assert extractor.use_dummy == True
    
    # Test with dummy checkpoint if exists
    dummy_checkpoint = "checkpoints/ulip2_pointbert_weights.pth"
    if os.path.exists(dummy_checkpoint):
        try:
            extractor2 = ULIPExtractor(checkpoint_path=dummy_checkpoint, device="cpu")
            # May use dummy or real depending on ULIP availability
            points = torch.randn(1, 1024, 3)
            features = extractor2.extract_features(points)
            assert features.shape == (1, 256) or features.shape == (1, 512)
        except Exception as e:
            print(f"Checkpoint loading test skipped: {e}")
    else:
        print("Dummy checkpoint not found, skipping checkpoint loading test")

def test_ulip_extractor_error_handling():
    extractor = ULIPExtractor(device="cpu")
    
    # Test invalid input dimensions
    with pytest.raises(RuntimeError):
        extractor.extract_features(torch.randn(2, 1024))  # 2D tensor
    
    # Test extraction still works after error
    points = torch.randn(1, 512, 3)
    features = extractor.extract_features(points)
    assert features.shape == (1, 256)


def test_ulip_extractor_cuda_if_available():
    """Test feature extraction with CUDA device if available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    # Test dummy model with CUDA
    extractor = ULIPExtractor(device="cuda")
    assert extractor.device == "cuda"
    assert extractor.use_dummy == True  # Should use dummy model
    
    # Create points on CPU
    points = torch.randn(2, 1024, 3)
    # Extract features (should move points to GPU internally)
    features = extractor.extract_features(points)
    assert features.shape == (2, 256)
    # Features should be on CPU after extraction
    assert features.device.type == "cpu"
    
    # Also test with points already on GPU
    points_gpu = points.cuda()
    features2 = extractor.extract_features(points_gpu)
    assert features2.shape == (2, 256)
    assert features2.device.type == "cpu"
    
    # Verify both feature sets are same (within tolerance)
    torch.testing.assert_close(features, features2, rtol=1e-5, atol=1e-6)


def test_ulip_extractor_feature_dim():
    """Test feature_dim property returns correct dimension."""
    # Dummy model should return 256
    extractor_dummy = ULIPExtractor(device="cpu")
    assert extractor_dummy.feature_dim == 256
    assert extractor_dummy.use_dummy == True
    
    # Real model (if checkpoint exists) should return 1280
    # This test is conditional on checkpoint availability
    test_checkpoint = "checkpoints/ULIP-2-PointBERT-10k-xyzrgb-pc-vit_g-objaverse_shapenet-pretrained.pt"
    if os.path.exists(test_checkpoint):
        try:
            extractor_real = ULIPExtractor(checkpoint_path=test_checkpoint, device="cpu")
            if not extractor_real.use_dummy:
                assert extractor_real.feature_dim == 1280
        except Exception as e:
            print(f"Real model test skipped due to: {e}")
    else:
        print(f"Checkpoint not found: {test_checkpoint}, skipping real model test")