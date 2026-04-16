#!/usr/bin/env python
"""Test script for OpenCLIP local loading verification.

Verifies that OpenCLIP can load local weights file instead of downloading from internet.
This is critical for ULIP-2 real model integration because ULIP-2 uses OpenCLIP internally.
"""

import os
import sys
from pathlib import Path
import torch
import open_clip

# Device detection moved inside test function to test both CPU and GPU


def get_checkpoint_path():
    """Get path to OpenCLIP checkpoint with fallback to environment variable."""
    # Allow override via environment variable
    env_path = os.environ.get("OPENCLIP_CHECKPOINT_PATH")
    if env_path:
        checkpoint_path = Path(env_path)
    else:
        # Default location relative to project root (assumes script is in tests/verification/)
        checkpoint_path = Path(__file__).parent.parent.parent / "checkpoints" / "open_clip_pytorch_model.bin"
    return checkpoint_path


def test_openclip_local_loading():
    """Test that OpenCLIP can load local weights file."""
    checkpoint_path = get_checkpoint_path()
    assert checkpoint_path.exists(), f"OpenCLIP weights not found: {checkpoint_path}"
    
    # Determine devices to test: always CPU, plus GPU if available
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    
    for device in devices:
        print(f"\nTesting on device: {device}")
        
        try:
            model, _, preprocess = open_clip.create_model_and_transforms(
                'ViT-bigG-14',
                pretrained=str(checkpoint_path)
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load OpenCLIP model from {checkpoint_path}. "
                f"Ensure the checkpoint is compatible with ViT-bigG-14. "
                f"Original error: {e}"
            )
        
        assert model is not None
        assert hasattr(model, 'encode_image')
        assert hasattr(model, 'encode_text')
        model.to(device)
        
        # Quick inference test with batch size 1
        model.eval()
        dummy_image = torch.randn(1, 3, 224, 224).to(device)
        with torch.no_grad():
            features = model.encode_image(dummy_image)
        
        assert features.shape == (1, 1280), f"Expected 1280D features, got {features.shape}"
        # Sanity check: features should not be all zeros or NaNs
        assert not torch.all(features == 0), "Features are all zeros"
        assert not torch.any(torch.isnan(features)), "Features contain NaN values"
        
        # Test encode_text method with dummy text
        dummy_text = ["a photo of a cat"]
        with torch.no_grad():
            text_features = model.encode_text(open_clip.tokenize(dummy_text).to(device))
        
        assert text_features.shape == (1, 1280), f"Expected 1280D text features, got {text_features.shape}"
        assert not torch.all(text_features == 0), "Text features are all zeros"
        assert not torch.any(torch.isnan(text_features)), "Text features contain NaN values"
        
        # Test with batch size > 1 to ensure batching works
        dummy_image_batch = torch.randn(3, 3, 224, 224).to(device)
        with torch.no_grad():
            batch_features = model.encode_image(dummy_image_batch)
        assert batch_features.shape == (3, 1280), f"Batch features shape mismatch: {batch_features.shape}"
        
        # Clean up GPU memory
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        
        print(f"  ✓ Device {device} test passed")
    
    print("\n✓ OpenCLIP local loading test passed on all devices")
    
if __name__ == "__main__":
    test_openclip_local_loading()
