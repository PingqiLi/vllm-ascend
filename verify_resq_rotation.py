
import torch
from unittest.mock import MagicMock
import sys
import os

# Mock modules to avoid loading full vllm/ascend dependencies
sys.modules['vllm_ascend.ascend_config'] = MagicMock()
try:
    import torch_npu
except ImportError:
    sys.modules['torch_npu'] = MagicMock()
sys.modules['vllm.model_executor.model_loader.weight_utils'] = MagicMock()

# Import the modified file
from vllm_ascend.quantization.w4a4_resq_dynamic import apply_rotation

def test_apply_rotation():
    print("Testing apply_rotation...")
    
    # Test 1: Hadamard (no matrix)
    x = torch.randn(2, 4) # Power of 2
    out_had = apply_rotation(x, None)
    assert out_had.shape == x.shape
    print("  [Pass] Hadamard fallback")

    # Test 2: Learned Matrix (R3)
    rot_mat = torch.eye(4)
    out_rot = apply_rotation(x, rot_mat)
    assert torch.allclose(out_rot, x)
    print("  [Pass] Learned rotation (Identity)")

    # Test 3: Learned Matrix (Random)
    rot_mat = torch.randn(4, 4)
    out_rot = apply_rotation(x, rot_mat)
    expected = torch.matmul(x, rot_mat)
    assert torch.allclose(out_rot, expected)
    print("  [Pass] Learned rotation (Random)")

    # Test 4: Block-wise Hadamard (27648 dim case)
    # Using smaller scale for testing but non-power-of-2 multiple
    # 24 = 3 * 8. Block size 8.
    x_blk = torch.randn(2, 24)
    # out_blk = apply_rotation(x_blk, None)
    # # Manual calculation
    # x_reshaped = x_blk.view(2, 3, 8)
    # # apply_hadamard_transform handles last dim
    # from vllm_ascend.quantization.w4a4_resq_dynamic import apply_hadamard_transform
    # out_manual = apply_hadamard_transform(x_reshaped) * (1.0 / (8 ** 0.5))
    # out_manual = out_manual.view(2, 24)
    # assert torch.allclose(out_blk, out_manual)
    # print("  [Pass] Block-wise Hadamard (Dim=24, Block=8)")
    
    # Test 5: Block-wise Hadamard (768 dim case, 3*256)
    # Simulated with 12 = 3 * 4
    # x_moe = torch.randn(2, 12)
    # out_moe = apply_rotation(x_moe, None)
    # x_reshaped_moe = x_moe.view(2, 3, 4)
    # out_manual_moe = apply_hadamard_transform(x_reshaped_moe) * (1.0 / (4 ** 0.5))
    # out_manual_moe = out_manual_moe.view(2, 12)
    # assert torch.allclose(out_moe, out_manual_moe)
    # print("  [Pass] Block-wise Hadamard (Dim=12, Block=4)")

    # Test 6: Explicit R4 vs Implicit
    # If we pass an R4 matrix, it should take precedence
    R4_learned = torch.randn(24, 24)
    out_explicit = apply_rotation(x_blk, R4_learned)
    expected_explicit = torch.matmul(x_blk, R4_learned)
    assert torch.allclose(out_explicit, expected_explicit)
    # If we don't pass it, it should fallback to block-wise
    out_implicit = apply_rotation(x_blk, None)
    # They should NOT be equal (very unlikely random R4 == Hadamard)
    assert not torch.allclose(out_explicit, out_implicit)
    print("  [Pass] Explicit R4 Precedence")

if __name__ == "__main__":
    test_apply_rotation()
