"""
ResQ debug wrappers for replacing linear layers in the original model.

Two modes:
- Fake quant (CPU/GPU): dequantize at load time, float matmul in forward.
- True quant (NPU): keep int4/int8 weights, use npu_dynamic_quant + npu_quant_matmul.

Reference implementations:
- ResQ: vllm_ascend/quantization/resq_linear.py (ResQLinearMethod)
- W8A8: vllm_ascend/quantization/w8a8_dynamic.py (AscendW8A8DynamicLinearMethod)

Usage:
    For ckpt A (ResQ quantized model):
    - down_proj -> W8A8Linear[TrueQuant]Wrapper (W8A8_DYNAMIC)
    - q/k/v/o_proj, gate/up_proj -> ResQLinear[TrueQuant]Wrapper (RESQ)
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Ud rotation helpers (from vllm_ascend/quantization/resq_linear.py)
# ---------------------------------------------------------------------------

def _is_pow2(n: int) -> bool:
    return (n & (n - 1) == 0) and (n > 0)


def _hadamard_transform(u: torch.Tensor) -> torch.Tensor:
    """Fast Hadamard transform using butterfly algorithm (unnormalized)."""
    n = u.shape[-1]
    assert _is_pow2(n), f"Last dimension must be power of 2, got {n}"
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


def apply_ud_rotation(
    x: torch.Tensor,
    Pd: torch.Tensor,
    Hd: Optional[torch.Tensor],
    K: int,
    blocksize: int,
    h_butterfly: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Apply Ud rotation before down_proj for ResQ inference.

    Ud = block_diag(Pd) @ (Hd kron H_blocksize).

    Args:
        h_butterfly: Pre-computed Hadamard matrix (blocksize, blocksize).
            If provided, uses matmul; otherwise uses butterfly algorithm.
    """
    original_shape = x.shape
    n = x.shape[-1]

    if n != K * blocksize:
        raise ValueError(
            f"Dimension mismatch: intermediate_size={n} != "
            f"K*blocksize={K}*{blocksize}={K * blocksize}"
        )

    original_dtype = x.dtype
    x = x.float()
    x = x.reshape(*original_shape[:-1], K, blocksize)

    # Step 1: block_diag(Pd)
    Pd_f32 = Pd.to(device=x.device, dtype=torch.float32)
    x = torch.matmul(x, Pd_f32)

    # Step 2: Hadamard on blocksize dimension
    if h_butterfly is not None:
        x = torch.matmul(
            x, h_butterfly.to(device=x.device, dtype=torch.float32)
        )
    else:
        x = _hadamard_transform(x.contiguous())

    # Step 3: Hadamard on K dimension (via Hd)
    if Hd is not None and K > 1:
        batch_shape = x.shape[:-2]
        batch_size = 1
        for d in batch_shape:
            batch_size *= d
        x = x.reshape(batch_size, K, blocksize)

        Hd_f32 = Hd.to(device=x.device, dtype=torch.float32)
        x = x.transpose(-1, -2)
        x = torch.matmul(x, Hd_f32.t())
        x = x.transpose(-1, -2)

        x = x.reshape(*batch_shape, K, blocksize)

    # Normalize
    x = x * K / math.sqrt(n)

    return x.reshape(original_shape).to(original_dtype)


# ---------------------------------------------------------------------------
# o_proj column reorder helper (shared between fake/true quant)
# ---------------------------------------------------------------------------

def compute_o_proj_column_order(
    in_dim: int,
    high_fraction: torch.Tensor,
    head_dim: int = 128,
) -> torch.Tensor:
    """Compute column reorder indices for o_proj: original -> [mid | high].

    Reference: ResQLinearMethod._setup_o_proj_column_order
    """
    hf = high_fraction.item() if high_fraction.numel() > 0 else 0.125
    high_bits_length = int(in_dim * hf)
    num_heads = in_dim // head_dim
    if num_heads == 0:
        return torch.arange(in_dim)
    high_per_head = high_bits_length // num_heads

    chunk_starts = torch.arange(0, in_dim, head_dim)
    high_cols = (
        chunk_starts.unsqueeze(1)
        + torch.arange(head_dim - high_per_head, head_dim)
    ).flatten()

    all_cols = torch.arange(in_dim)
    mask = torch.ones(in_dim, dtype=torch.bool)
    mask[high_cols] = False
    mid_cols = all_cols[mask]

    return torch.cat([mid_cols, high_cols])


# ---------------------------------------------------------------------------
# Fake-quant wrappers (CPU/GPU)
# ---------------------------------------------------------------------------

class ResQLinearWrapper(nn.Module):
    """CPU/GPU fake-quant wrapper for ResQ quantized linear layers.

    Dequantizes at init, standard float matmul in forward.
    For o_proj: inverts column reorder in the weight.
    """

    def __init__(
        self,
        weight_high: torch.Tensor,
        weight_low: torch.Tensor,
        scale_high: torch.Tensor,
        scale_low: torch.Tensor,
        high_fraction: torch.Tensor,
        is_o_proj: bool = False,
        head_dim: int = 128,
    ):
        super().__init__()

        W_low = weight_low.float() * scale_low.float()
        W_high = weight_high.float() * scale_high.float()
        W_full = torch.cat([W_low, W_high], dim=1)

        if is_o_proj:
            in_dim = W_full.shape[1]
            column_order = compute_o_proj_column_order(
                in_dim, high_fraction, head_dim
            )
            inverse_order = torch.empty_like(column_order)
            inverse_order[column_order] = torch.arange(len(column_order))
            W_full = W_full[:, inverse_order]

        self.register_buffer('weight', W_full.to(torch.bfloat16))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight)


class W8A8LinearWrapper(nn.Module):
    """CPU/GPU fake-quant wrapper for W8A8 dynamic quantized linear layers.

    Dequantizes at init, standard float matmul in forward.
    For down_proj: applies Ud rotation to input.
    """

    def __init__(
        self,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        weight_offset: torch.Tensor,
        rotation_Pd: Optional[torch.Tensor] = None,
        rotation_Hd: Optional[torch.Tensor] = None,
    ):
        super().__init__()

        # weight_offset not used in npu_quant_matmul apply path
        W_float = weight.float() * weight_scale.float()
        self.register_buffer('weight', W_float.to(torch.bfloat16))

        self.has_rotation = rotation_Pd is not None
        if self.has_rotation:
            self.register_buffer('rotation_Pd', rotation_Pd.float())
            if rotation_Hd is not None:
                self.register_buffer('rotation_Hd', rotation_Hd.float())
                self.K = rotation_Hd.shape[0]
            else:
                self.rotation_Hd = None
                self.K = 1
            self.blocksize = rotation_Pd.shape[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.has_rotation:
            x = apply_ud_rotation(
                x,
                self.rotation_Pd,
                getattr(self, 'rotation_Hd', None),
                K=self.K,
                blocksize=self.blocksize,
            )
        return F.linear(x, self.weight)


# ---------------------------------------------------------------------------
# True-quant wrappers (NPU) — mirrors production ResQLinearMethod.apply()
#                               and AscendW8A8DynamicLinearMethod.apply()
# ---------------------------------------------------------------------------

class ResQLinearTrueQuantWrapper(nn.Module):
    """NPU true-quant wrapper for ResQ mixed-precision linear layers.

    Uses npu_dynamic_quant + npu_quant_matmul with actual int4/int8 weights.
    Mirrors ResQLinearMethod.process_weights_after_loading + apply.
    """

    def __init__(
        self,
        weight_high: torch.Tensor,    # (out, k_high) int8
        weight_low: torch.Tensor,     # (out, k_low) int8
        scale_high: torch.Tensor,     # (out, 1)
        scale_low: torch.Tensor,      # (out, 1)
        high_fraction: torch.Tensor,
        is_o_proj: bool = False,
        head_dim: int = 128,
    ):
        super().__init__()
        import torch_npu

        self.is_o_proj = is_o_proj

        # --- process_weights_after_loading equivalent ---

        # int4 packing: (out, k_low) -> packed
        w_low = weight_low.to(torch.int32).npu()
        self.register_buffer(
            'weight_low_packed',
            torch_npu.npu_convert_weight_to_int4pack(w_low),
        )

        # int8 transpose: (out, k_high) -> (k_high, out)
        w_high = weight_high.transpose(0, 1).contiguous().npu()
        self.register_buffer('weight_high', w_high)

        # scales -> 1D float32
        self.register_buffer(
            'scale_low', scale_low.flatten().to(torch.float32).npu()
        )
        self.register_buffer(
            'scale_high', scale_high.flatten().to(torch.float32).npu()
        )

        # o_proj column reorder
        if is_o_proj:
            in_dim = weight_low.shape[1] + weight_high.shape[1]
            order = compute_o_proj_column_order(
                in_dim, high_fraction, head_dim
            )
            self.register_buffer('o_proj_column_order', order.npu())
        else:
            self.register_buffer(
                'o_proj_column_order', torch.empty(0, dtype=torch.long)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        import torch_npu

        if not x.is_contiguous():
            x = x.contiguous()

        # o_proj column reorder
        if self.is_o_proj and self.o_proj_column_order.numel() > 0:
            x = x[..., self.o_proj_column_order]

        # Mixed precision matmul
        original_shape = x.shape
        x_2d = x.contiguous().view(-1, x.shape[-1]).float()

        # weight_high is (k_high, out) after transpose
        in_high = self.weight_high.shape[0]
        in_low = x_2d.shape[-1] - in_high

        x_low = x_2d[:, :in_low].to(torch.float16)
        x_high = x_2d[:, in_low:].to(torch.float16)

        x_low_quant, lx_scale = torch_npu.npu_dynamic_quant(
            x_low.contiguous(), dst_type=torch.quint4x2
        )
        x_high_quant, rx_scale = torch_npu.npu_dynamic_quant(
            x_high, dst_type=torch.int8
        )

        output_low = torch_npu.npu_quant_matmul(
            x_low_quant,
            self.weight_low_packed.t(),
            self.scale_low,
            pertoken_scale=lx_scale,
            output_dtype=torch.float16,
        )

        output_high = torch_npu.npu_quant_matmul(
            x_high_quant,
            self.weight_high,
            self.scale_high,
            pertoken_scale=rx_scale,
            output_dtype=torch.float16,
        )

        output = torch.add(output_low, output_high)

        output_shape = list(original_shape[:-1]) + [output.shape[-1]]
        return output.view(output_shape).to(torch.bfloat16)


class W8A8LinearTrueQuantWrapper(nn.Module):
    """NPU true-quant wrapper for W8A8 dynamic quantized linear layers.

    Uses npu_dynamic_quant + npu_quant_matmul with actual int8 weights.
    For down_proj: applies Ud rotation before quantized matmul.
    Mirrors AscendW8A8DynamicLinearMethod.process_weights_after_loading + apply.
    """

    def __init__(
        self,
        weight: torch.Tensor,          # (out, in) int8
        weight_scale: torch.Tensor,    # (out, 1)
        weight_offset: torch.Tensor,   # (out, 1) int8
        rotation_Pd: Optional[torch.Tensor] = None,
        rotation_Hd: Optional[torch.Tensor] = None,
    ):
        super().__init__()

        # --- process_weights_after_loading equivalent ---

        # int8 transpose: (out, in) -> (in, out)
        w = weight.transpose(0, 1).contiguous().npu()
        self.register_buffer('weight', w)

        # scale -> 1D float32
        self.register_buffer(
            'weight_scale', weight_scale.flatten().to(torch.float32).npu()
        )

        # Ud rotation
        self.has_rotation = rotation_Pd is not None
        if self.has_rotation:
            self.register_buffer('rotation_Pd', rotation_Pd.float().npu())
            if rotation_Hd is not None:
                self.register_buffer('rotation_Hd', rotation_Hd.float().npu())
                self.K = rotation_Hd.shape[0]
            else:
                self.rotation_Hd = None
                self.K = 1
            self.blocksize = rotation_Pd.shape[0]

            # Pre-compute butterfly Hadamard matrix
            eye = torch.eye(self.blocksize, dtype=torch.float32)
            h_matrix = _hadamard_transform(eye)
            self.register_buffer('h_butterfly', h_matrix.npu())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        import torch_npu

        if not x.is_contiguous():
            x = x.contiguous()

        # Ud rotation for down_proj
        if self.has_rotation:
            x = apply_ud_rotation(
                x,
                self.rotation_Pd,
                getattr(self, 'rotation_Hd', None),
                K=self.K,
                blocksize=self.blocksize,
                h_butterfly=self.h_butterfly,
            )

        # Reshape to 2D for npu quant ops (HF model uses 3D: batch, seq, hidden)
        original_shape = x.shape
        x = x.view(-1, x.shape[-1])

        # W8A8 quantized matmul
        quantized_x, dynamic_scale = torch_npu.npu_dynamic_quant(x)

        output = torch_npu.npu_quant_matmul(
            quantized_x,
            self.weight,
            self.weight_scale,
            pertoken_scale=dynamic_scale,
            output_dtype=torch.bfloat16,
        )

        output_shape = list(original_shape[:-1]) + [output.shape[-1]]
        return output.view(output_shape)
