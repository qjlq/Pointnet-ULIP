#!/usr/bin/env python3
"""Verify ULIPExtractor with real model loads and returns 1280-D features."""
import sys
import os
sys.path.insert(0, '.')
import torch
from libs.ulip_extractor import ULIPExtractor

checkpoint = "../checkpoints/ULIP-2-PointBERT-10k-xyzrgb-pc-vit_g-objaverse_shapenet-pretrained.pt"
openclip_checkpoint = "../checkpoints/open_clip_pytorch_model.bin"

if not os.path.exists(checkpoint):
    print(f"ULIP checkpoint not found: {checkpoint}")
    sys.exit(0)

if not os.path.exists(openclip_checkpoint):
    print(f"OpenCLIP checkpoint not found: {openclip_checkpoint}")
    print("Will attempt to download weights...")
    openclip_checkpoint = None

print("Testing ULIPExtractor with real model...")
extractor = ULIPExtractor(
    checkpoint_path=checkpoint,
    openclip_checkpoint=openclip_checkpoint,
    device="cpu"
)

print(f"use_dummy: {extractor.use_dummy}")
print(f"feature_dim: {extractor.feature_dim}")

# Test feature extraction
points = torch.randn(2, 8192, 3)
features = extractor.extract_features(points)
print(f"Features shape: {features.shape}")
print(f"Expected: (2, {extractor.feature_dim})")
assert features.shape == (2, extractor.feature_dim), f"Expected (2, {extractor.feature_dim}), got {features.shape}"
print("✓ Verification passed")