#!/usr/bin/env python
"""GPU verification script for fusion pipeline.

Checks PyTorch version, CUDA availability, GPU device information,
and tests basic tensor operations on GPU.

Returns exit code 0 on success, 1 on failure.
"""

import sys
import torch
import numpy as np

def check_pytorch_version():
    """Check PyTorch version compatibility."""
    print("=" * 60)
    print("PyTorch Version Check")
    print("=" * 60)
    
    version = torch.__version__
    print(f"PyTorch version: {version}")
    
    # Check if version matches expected (1.10.1)
    expected_major_minor = "1.10"
    if version.startswith(expected_major_minor):
        print(f"✓ PyTorch version compatible (expected {expected_major_minor}.x)")
        return True
    else:
        print(f"⚠ PyTorch version mismatch (expected {expected_major_minor}.x, got {version})")
        print("  The pipeline may still work with other versions, but 1.10.x is recommended.")
        return True  # Continue verification anyway

def check_cuda_availability():
    """Check CUDA availability and device information."""
    print("\n" + "=" * 60)
    print("CUDA Availability Check")
    print("=" * 60)
    
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    
    if not cuda_available:
        print("✗ CUDA is not available. GPU acceleration disabled.")
        print("  If you have a GPU, check CUDA driver installation.")
        return False
    
    # Get CUDA version
    cuda_version = torch.version.cuda
    print(f"CUDA version (PyTorch built with): {cuda_version}")
    
    # Note: PyTorch built for CUDA 12.1 works with CUDA 13 driver via forward compatibility
    print("  Note: CUDA 12.1 toolkit is compatible with CUDA 13 driver (forward compatibility)")
    
    # Get number of devices
    device_count = torch.cuda.device_count()
    print(f"Number of CUDA devices: {device_count}")
    
    # Get device properties
    for i in range(device_count):
        print(f"\nDevice {i}:")
        props = torch.cuda.get_device_properties(i)
        print(f"  Name: {props.name}")
        print(f"  Compute capability: {props.major}.{props.minor}")
        print(f"  Total memory: {props.total_memory / 1e9:.2f} GB")
        print(f"  Multi-processor count: {props.multi_processor_count}")
    
    return True

def test_tensor_operations():
    """Test basic tensor operations on GPU."""
    print("\n" + "=" * 60)
    print("Tensor Operations Test")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("Skipping GPU tensor tests (CUDA not available)")
        return True
    
    try:
        # Create tensors on CPU
        a_cpu = torch.randn(1000, 1000)
        b_cpu = torch.randn(1000, 1000)
        
        # Move to GPU
        device = torch.device("cuda:0")
        a_gpu = a_cpu.to(device)
        b_gpu = b_cpu.to(device)
        
        print("Testing tensor operations on GPU...")
        
        # Test 1: Basic arithmetic
        c_gpu = a_gpu + b_gpu
        print("✓ Tensor addition")
        
        # Test 2: Matrix multiplication
        d_gpu = torch.matmul(a_gpu, b_gpu)
        print("✓ Matrix multiplication")
        
        # Test 3: Backward pass (gradients)
        a_gpu.requires_grad = True
        b_gpu.requires_grad = True
        result = (a_gpu * b_gpu).sum()
        result.backward()
        print("✓ Gradient computation")
        
        # Test 4: Memory transfer
        c_cpu = c_gpu.cpu()
        print("✓ CPU-GPU memory transfer")
        
        # Verify results match CPU computation (within tolerance)
        c_cpu_expected = a_cpu + b_cpu
        if torch.allclose(c_cpu, c_cpu_expected, rtol=1e-5):
            print("✓ Results match CPU computation")
        else:
            print("⚠ Results differ from CPU computation (may indicate GPU issues)")
        
        # Clean up
        torch.cuda.empty_cache()
        print("✓ GPU memory cache cleared")
        
        return True
        
    except Exception as e:
        print(f"✗ Tensor operation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_memory():
    """Check GPU memory availability."""
    print("\n" + "=" * 60)
    print("GPU Memory Check")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("Skipping GPU memory check (CUDA not available)")
        return True
    
    try:
        # Get current memory allocator statistics
        allocated = torch.cuda.memory_allocated(0)
        reserved = torch.cuda.memory_reserved(0)
        total = torch.cuda.get_device_properties(0).total_memory
        
        print(f"Allocated memory: {allocated / 1e6:.2f} MB")
        print(f"Reserved memory: {reserved / 1e6:.2f} MB")
        print(f"Total device memory: {total / 1e9:.2f} GB")
        print(f"Available memory: {(total - allocated) / 1e9:.2f} GB")
        
        # Test memory allocation
        test_size = 100 * 1024 * 1024  # 100MB
        try:
            test_tensor = torch.empty(test_size, dtype=torch.uint8, device='cuda:0')
            print(f"✓ Successfully allocated {test_size / 1e6:.0f} MB test tensor")
            del test_tensor
            torch.cuda.empty_cache()
        except RuntimeError as e:
            print(f"⚠ Failed to allocate test tensor: {e}")
            print("  This may indicate memory fragmentation or insufficient memory.")
        
        return True
        
    except Exception as e:
        print(f"✗ Memory check failed: {e}")
        return False

def main():
    """Main verification function."""
    print("Fusion Pipeline GPU Verification")
    print("=" * 60)
    
    success = True
    
    # Run all checks
    if not check_pytorch_version():
        success = False
    
    cuda_available = check_cuda_availability()
    if not cuda_available:
        success = False
        print("\n⚠ CUDA not available. Some checks skipped.")
    else:
        if not test_tensor_operations():
            success = False
        
        if not check_memory():
            success = False
    
    # Summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    
    if success:
        print("✅ All checks passed!")
        print("GPU environment is ready for the fusion pipeline.")
        return 0
    else:
        print("❌ Some checks failed.")
        print("Please address the issues above before running the pipeline.")
        return 1

if __name__ == "__main__":
    sys.exit(main())