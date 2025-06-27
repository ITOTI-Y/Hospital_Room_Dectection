"""
Test script to verify GPU compatibility for the RL optimization system.
"""

import torch
import sys
import pathlib

project_root = pathlib.Path(__file__).parent
sys.path.append(str(project_root))

def test_cuda_availability():
    """Test basic CUDA availability"""
    print("=== GPU Compatibility Test ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name()}")
        
        memory_allocated = torch.cuda.memory_allocated() / 1024**3
        memory_reserved = torch.cuda.memory_reserved() / 1024**3
        memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        print(f"GPU Memory - Allocated: {memory_allocated:.2f} GB")
        print(f"GPU Memory - Reserved: {memory_reserved:.2f} GB") 
        print(f"GPU Memory - Total: {memory_total:.2f} GB")
        
        print("\n=== Testing GPU Tensor Operations ===")
        try:
            device = torch.device("cuda")
            x = torch.randn(1000, 1000, device=device)
            y = torch.randn(1000, 1000, device=device)
            z = torch.matmul(x, y)
            print("✅ GPU tensor operations successful")
            print(f"Result tensor shape: {z.shape}")
            print(f"Result tensor device: {z.device}")
        except Exception as e:
            print(f"❌ GPU tensor operations failed: {e}")
            return False
    else:
        print("❌ CUDA not available - will use CPU fallback")
        return False
    
    return True

def test_gpu_optimizer_import():
    """Test importing the GPU optimizer"""
    print("\n=== Testing GPU Optimizer Import ===")
    try:
        from src.analysis.rl_layout_optimizer_gpu import GPULayoutOptimizer, check_gpu_availability
        print("✅ GPU optimizer import successful")
        
        gpu_info = check_gpu_availability()
        print("GPU Info:")
        for key, value in gpu_info.items():
            print(f"  {key}: {value}")
        
        return True
    except Exception as e:
        print(f"❌ GPU optimizer import failed: {e}")
        return False

if __name__ == "__main__":
    cuda_ok = test_cuda_availability()
    import_ok = test_gpu_optimizer_import()
    
    print("\n=== Summary ===")
    if cuda_ok and import_ok:
        print("✅ GPU compatibility test PASSED")
        print("The system can use GPU acceleration for RL optimization")
    elif import_ok:
        print("⚠️  GPU compatibility test PARTIAL")
        print("The system will work but fall back to CPU")
    else:
        print("❌ GPU compatibility test FAILED")
        print("There are issues with the GPU implementation")
