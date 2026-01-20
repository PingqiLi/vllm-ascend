"""
ResQ Quantized MatMul Operations for Debugging

Supports both CPU (int32 matmul) and NPU (npu_quant_matmul or float16 fallback).

Reference: vllm_ascend/ops/resq_quant_matmul.py, msmodelslim reference_op_impl.py
"""

import torch

try:
    import torch_npu
    HAS_NPU = True
except ImportError:
    HAS_NPU = False


def is_npu_tensor(x: torch.Tensor) -> bool:
    """Check if tensor is on NPU device."""
    return HAS_NPU and x.device.type == 'npu'


def quant_matmul_core(
    x: torch.Tensor, 
    weight: torch.Tensor, 
    per_channel_scale: torch.Tensor, 
    per_token_scale: torch.Tensor,
    out_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Core quantized matmul: y = (x @ W) * per_channel_scale * per_token_scale
    
    Args:
        x: (M, K) int8, quantized activation
        weight: (K, N) int8, quantized weight
        per_channel_scale: (1, N) or (N,) float32, weight scale
        per_token_scale: (M,) float32, activation scale
        out_dtype: output dtype
        
    Returns:
        (M, N) tensor in out_dtype
        
    Note:
        - CPU: uses int32 matmul for precision
        - NPU: uses npu_quant_matmul or float16 matmul fallback
    """
    M = x.shape[0]
    
    # Ensure per_channel_scale is (1, N) for broadcasting with (M, N) matmul result
    # Checkpoint stores scale as (out_features, 1), so we need to handle both cases
    if per_channel_scale.dim() == 1:
        per_channel_scale = per_channel_scale.unsqueeze(0)  # (N,) -> (1, N)
    elif per_channel_scale.dim() == 2 and per_channel_scale.shape[1] == 1:
        per_channel_scale = per_channel_scale.T  # (N, 1) -> (1, N)
    
    if is_npu_tensor(x):
        # NPU path
        if hasattr(torch_npu, 'npu_quant_matmul'):
            # Use npu_quant_matmul: y = x @ weight * scale
            scale = per_channel_scale.flatten().to(torch.float32)
            result = torch_npu.npu_quant_matmul(x, weight, scale)
            if isinstance(result, tuple):
                y = result[0]
            else:
                y = result
            # Match reference: fp16 -> fp32
            y = y.to(torch.float16).to(torch.float32)
        else:
            # Fallback: use float16 matmul
            y = torch.matmul(x.to(torch.float16), weight.to(torch.float16)).to(torch.float32)
            y = torch.mul(y, per_channel_scale).to(torch.float16).to(torch.float32)
    else:
        # CPU path: int32 matmul
        y = torch.matmul(x.to(torch.int32), weight.to(torch.int32))
        y = y.to(torch.float32)
        # Match reference: fp16 -> fp32 conversion for consistency
        y = torch.mul(y, per_channel_scale).to(torch.float16).to(torch.float32)
    
    # Apply per-token scale
    y = torch.mul(y, per_token_scale.reshape(M, 1))
    
    return y.to(out_dtype)


def resq_quant_matmul(
    x: torch.Tensor,
    weight_low: torch.Tensor,
    weight_high: torch.Tensor, 
    scale_low: torch.Tensor,
    scale_high: torch.Tensor,
    x_scale_low: torch.Tensor,
    x_scale_high: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    ResQ mixed-precision matmul (simplified interface for debugging).
    
    Computes: y = x_low @ W_low.T * scale_low * x_scale_low 
                + x_high @ W_high.T * scale_high * x_scale_high
    
    Args:
        x: (M, K) float, unquantized activation (will be quantized internally)
        weight_low: (out_dim, in_low) int8, int4 part (values in [-8, 7])
        weight_high: (out_dim, in_high) int8, int8 part (values in [-128, 127])
        scale_low: (out_dim,) or (out_dim, 1) float32, weight scale for int4 part
        scale_high: (out_dim,) or (out_dim, 1) float32, weight scale for int8 part
        x_scale_low: (M,) float32, per-token activation scale for int4 part
        x_scale_high: (M,) float32, per-token activation scale for int8 part
        out_dtype: output dtype
        
    Returns:
        (M, out_dim) tensor
    """
    in_low = weight_low.shape[1]
    in_high = weight_high.shape[1]
    
    # Split input
    x_low = x[:, :in_low]
    x_high = x[:, in_low:in_low + in_high]
    
    # Dynamic quantization of activations
    # int4 part: clamp to [-8, 7]
    x_low_int8 = torch.round(x_low / x_scale_low.unsqueeze(-1)).clamp(-8, 7).to(torch.int8)
    # int8 part: clamp to [-128, 127]
    x_high_int8 = torch.round(x_high / x_scale_high.unsqueeze(-1)).clamp(-128, 127).to(torch.int8)
    
    # Ensure scales are proper shape for broadcasting
    if scale_low.dim() == 2 and scale_low.shape[1] == 1:
        # (out_dim, 1) -> (1, out_dim) for per_channel_scale
        scale_low = scale_low.squeeze(-1)
    if scale_high.dim() == 2 and scale_high.shape[1] == 1:
        scale_high = scale_high.squeeze(-1)
    
    # int4 part: x_low @ W_low.T
    # weight_low is (out_dim, in_low), need to transpose to (in_low, out_dim)
    y_low = quant_matmul_core(
        x_low_int8, 
        weight_low.T.contiguous(), 
        scale_low, 
        x_scale_low,
        torch.float32
    )
    
    # int8 part: x_high @ W_high.T  
    y_high = quant_matmul_core(
        x_high_int8, 
        weight_high.T.contiguous(), 
        scale_high, 
        x_scale_high,
        torch.float32
    )
    
    # Sum
    output = y_low + y_high
    return output.to(out_dtype)


def dynamic_quantize_activation(
    x: torch.Tensor,
    split_k: int,
    int4_qmax: int = 7,
    int8_qmax: int = 127,
) -> tuple:
    """
    Dynamically quantize activation tensor.
    
    Args:
        x: (M, K) float tensor
        split_k: position to split K dimension (first split_k channels are int4)
        int4_qmax: max value for int4 part (default 7)
        int8_qmax: max value for int8 part (default 127)
        
    Returns:
        x_quant: (M, K) int8 quantized tensor
        x_scale_low: (M,) float32 per-token scale for int4 part
        x_scale_high: (M,) float32 per-token scale for int8 part
    """
    x_low = x[:, :split_k]
    x_high = x[:, split_k:]
    
    # Compute per-token scales
    # scale = max(abs(x)) / qmax, with eps for stability
    eps = 1e-8
    x_scale_low = x_low.abs().max(dim=-1).values / int4_qmax + eps
    x_scale_high = x_high.abs().max(dim=-1).values / int8_qmax + eps
    
    # Quantize
    x_low_quant = torch.round(x_low / x_scale_low.unsqueeze(-1)).clamp(-int4_qmax-1, int4_qmax).to(torch.int8)
    x_high_quant = torch.round(x_high / x_scale_high.unsqueeze(-1)).clamp(-int8_qmax-1, int8_qmax).to(torch.int8)
    
    x_quant = torch.cat([x_low_quant, x_high_quant], dim=-1)
    
    return x_quant, x_scale_low, x_scale_high


def resq_linear_forward(
    x: torch.Tensor,
    weight_low: torch.Tensor,
    weight_high: torch.Tensor,
    scale_low: torch.Tensor,
    scale_high: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    ResQ linear layer forward pass with true quantization.
    
    This is a convenience wrapper that:
    1. Dynamically quantizes the input activation
    2. Performs mixed-precision matmul
    
    Args:
        x: (M, K) float tensor, unquantized input
        weight_low: (out_dim, in_low) int8, int4 weight part
        weight_high: (out_dim, in_high) int8, int8 weight part  
        scale_low: (out_dim,) float32, weight scale for int4 part
        scale_high: (out_dim,) float32, weight scale for int8 part
        out_dtype: output dtype
        
    Returns:
        (M, out_dim) tensor
    """
    in_low = weight_low.shape[1]
    in_high = weight_high.shape[1]
    
    # Ensure x has correct size
    expected_k = in_low + in_high
    if x.shape[-1] != expected_k:
        raise ValueError(f"Input has {x.shape[-1]} features, expected {expected_k} (low={in_low}, high={in_high})")
    
    # Flatten for matmul if needed
    orig_shape = x.shape
    if x.dim() > 2:
        x = x.view(-1, x.shape[-1])
    
    # Dynamic quantization
    _, x_scale_low, x_scale_high = dynamic_quantize_activation(x, in_low)
    
    # Quantized matmul
    output = resq_quant_matmul(
        x, weight_low, weight_high, scale_low, scale_high,
        x_scale_low, x_scale_high, out_dtype
    )
    
    # Restore shape if needed
    if len(orig_shape) > 2:
        output = output.view(*orig_shape[:-1], -1)
    
    return output
