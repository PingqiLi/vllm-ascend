"""
Test int32 matmul on NPU device.

Run: python tools/test_npu_int32_matmul.py
"""

import torch
import torch_npu

def test_int32_matmul():
    torch_npu.npu.set_device(0)
    
    M, K, N = 128, 512, 256
    
    # Create int8 tensors
    x_int8 = torch.randint(-8, 8, (M, K), dtype=torch.int8)
    w_int8 = torch.randint(-8, 8, (K, N), dtype=torch.int8)
    
    # Test on CPU first
    print("=" * 50)
    print("Testing on CPU...")
    x_cpu = x_int8.clone()
    w_cpu = w_int8.clone()
    
    try:
        # int32 matmul on CPU
        result_cpu = torch.matmul(x_cpu.to(torch.int32), w_cpu.to(torch.int32))
        print(f"CPU int32 matmul: SUCCESS")
        print(f"  Result shape: {result_cpu.shape}, dtype: {result_cpu.dtype}")
    except Exception as e:
        print(f"CPU int32 matmul: FAILED - {e}")
    
    # Test on NPU
    print("=" * 50)
    print("Testing on NPU...")
    x_npu = x_int8.npu()
    w_npu = w_int8.npu()
    
    # Test 1: Direct int32 matmul
    print("\n1. Direct int32 matmul:")
    try:
        result_npu = torch.matmul(x_npu.to(torch.int32), w_npu.to(torch.int32))
        print(f"   SUCCESS - shape: {result_npu.shape}, dtype: {result_npu.dtype}")
    except Exception as e:
        print(f"   FAILED - {e}")
    
    # Test 2: Float16 matmul (should work)
    print("\n2. Float16 matmul:")
    try:
        result_f16 = torch.matmul(x_npu.to(torch.float16), w_npu.to(torch.float16))
        print(f"   SUCCESS - shape: {result_f16.shape}, dtype: {result_f16.dtype}")
    except Exception as e:
        print(f"   FAILED - {e}")
    
    # Test 3: npu_quant_matmul (if available)
    print("\n3. torch_npu.npu_quant_matmul:")
    try:
        # Prepare inputs for npu_quant_matmul
        # See: https://www.hiascend.com/document/detail/zh/Pytorch/710/apiref/torchnpuCustomsapi/context/torch_npu-npu_quant_matmul.md
        
        # x: (M, K) int8
        # weight: (N, K) int8 (transposed)
        # scale: (N,) float32 or (1, N)
        
        scale = torch.ones(N, dtype=torch.float32).npu()
        offset = torch.zeros(N, dtype=torch.float32).npu()
        
        # npu_quant_matmul expects: x1 (int8), x2 (int8), scale, offset (optional)
        # weight needs to be (K, N) -> transpose to (N, K) for npu_quant_matmul? Check docs
        
        result_quant = torch_npu.npu_quant_matmul(
            x_npu,  # (M, K) int8
            w_npu,  # (K, N) int8
            scale,  # (N,) float32
        )
        print(f"   SUCCESS - result type: {type(result_quant)}")
        if isinstance(result_quant, tuple):
            print(f"   Result[0] shape: {result_quant[0].shape}, dtype: {result_quant[0].dtype}")
        else:
            print(f"   Result shape: {result_quant.shape}, dtype: {result_quant.dtype}")
    except Exception as e:
        print(f"   FAILED - {e}")
    
    # Test 4: Alternative - convert to float, matmul, then apply scale
    print("\n4. Float matmul with post-scale (workaround):")
    try:
        x_f16 = x_npu.to(torch.float16)
        w_f16 = w_npu.to(torch.float16)
        result = torch.matmul(x_f16, w_f16)
        print(f"   SUCCESS - shape: {result.shape}, dtype: {result.dtype}")
    except Exception as e:
        print(f"   FAILED - {e}")
    
    print("=" * 50)
    print("Test completed.")


if __name__ == "__main__":
    test_int32_matmul()
