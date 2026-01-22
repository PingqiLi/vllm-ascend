import torch
import math

def is_pow2(n: int) -> bool:
    return (n & (n - 1) == 0) and (n > 0)

def hadamard_transform_recursive(u: torch.Tensor) -> torch.Tensor:
    """Fast Hadamard transform using butterfly algorithm (unnormalized)."""
    n = u.shape[-1]
    assert is_pow2(n), f"Last dimension must be power of 2, got {n}"
    
    original_shape = u.shape
    x = u.reshape(-1, n).clone()
    
    h = 1
    while h < n:
        x = x.view(-1, n // (2 * h), 2, h)
        a = x[:, :, 0, :]
        b = x[:, :, 1, :]
        x = torch.stack([a + b, a - b], dim=2)
        x = x.view(-1, n)
        h *= 2
    
    return x.view(original_shape)

def test_hadamard_matrix_equivalence():
    print("Testing Hadamard Matrix Equivalence...")
    
    blocksize = 256
    
    # 1. Generate random input
    x = torch.randn(10, 100, blocksize) # [batch, K, blocksize]
    
    # 2. Compute using recursive algorithm
    output_recursive = hadamard_transform_recursive(x)
    
    # 3. Compute using matrix multiplication
    # Pre-compute H matrix
    eye = torch.eye(blocksize)
    h_matrix = hadamard_transform_recursive(eye)
    
    # Apply matrix multiplication
    output_matmul = torch.matmul(x, h_matrix)
    
    # 4. Compare
    diff = (output_recursive - output_matmul).abs().max()
    mse = (output_recursive - output_matmul).pow(2).mean()
    
    print(f"Max Diff: {diff.item()}")
    print(f"MSE: {mse.item()}")
    
    if diff < 1e-4:
        print("PASS: Matrix multiplication is equivalent to recursive algorithm.")
    else:
        print("FAIL: Significant difference found.")
        exit(1)

if __name__ == "__main__":
    test_hadamard_matrix_equivalence()
