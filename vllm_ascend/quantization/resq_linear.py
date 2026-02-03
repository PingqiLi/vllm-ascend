#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
"""ResQ mixed-precision quantization linear method."""

import math
from typing import Any, Dict, List, Optional
import numpy as np

import torch
import torch_npu
from vllm.model_executor.layers.linear import LinearMethodBase
from vllm.model_executor.utils import set_weight_attrs

from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ, is_enable_nz


def _is_pow2(n: int) -> bool:
    """Check if n is a power of 2."""
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


def _apply_ud_rotation(
    x: torch.Tensor,
    Pd: torch.Tensor,
    Hd: Optional[torch.Tensor],
    h_butterfly: Optional[torch.Tensor],
    K: int,
    blocksize: int,
) -> torch.Tensor:
    """Apply Ud rotation before down_proj for ResQ inference."""
    original_shape = x.shape
    n = x.shape[-1]

    if n != K * blocksize:
        raise ValueError(
            f"Dimension mismatch: intermediate_size={n} != "
            f"K*blocksize={K}*{blocksize}={K*blocksize}"
        )

    original_dtype = x.dtype
    x = x.float()

    # Reshape: (..., n) -> (..., K, blocksize)
    x = x.reshape(*original_shape[:-1], K, blocksize)

    # Step 1: Apply block_diag(Pd) block-wise
    Pd_f32 = Pd.to(device=x.device, dtype=torch.float32)
    x = torch.matmul(x, Pd_f32)

    # Step 2: Apply H = Hd ⊗ H_butterfly
    if h_butterfly is not None:
        h_butterfly_f32 = h_butterfly.to(device=x.device, dtype=torch.float32)
        x = torch.matmul(x, h_butterfly_f32)
    else:
        x = _hadamard_transform(x.contiguous())

    # Apply Hd on the K dimension
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


def _pack_int4_to_int8_signed(x: torch.Tensor) -> torch.Tensor:
    """
    x: int8 tensor, shape (E, K, N)，值域 ∈ [-8, 7]
    return: int8 tensor, shape (E, K, N/2)，每个元素打包两个有符号 int4
    """
    assert x.dtype == torch.int8
    K, N = x.shape
    assert N % 2 == 0
    
    # 转成无符号补码 [0, 15]
    x_unsigned = torch.where(x < 0, x + 16, x).to(torch.int32)

    low = x_unsigned[..., 0::2]   # 偶数 -> 低 4 位
    high = x_unsigned[..., 1::2]  # 奇数 -> 高 4 位

    out = (low | (high << 4)).to(torch.int8)
    return out


def _convert_scales(scales):
    N = scales.shape[0]
    scaleUint32 = scales.cpu().to(torch.float32).clone().numpy().astype(np.float32).reshape(1, N)
    scaleUint32.dtype = np.uint32
    scaleUint64 = np.zeros((1, N * 2), dtype=np.uint32)
    scaleUint64[...,::2] = scaleUint32
    scaleUint64.dtype = np.int64
    scale = torch.from_numpy(scaleUint64).npu()
    return scale


class ResQLinearMethod(LinearMethodBase):
    """Linear method for ResQ mixed-precision quantization.

    ResQ uses a mixed-precision approach with:
    - weight_low: int4 weights (stored as int8, packed during processing)
    - weight_high: int8 weights
    - scale_low, scale_high: per-channel scales
    - rotation_Pd, rotation_Hd: MLP rotation matrices (for down_proj only)
    """

    def __init__(
        self,
        quant_config: Dict[str, Any],
        prefix: str,
        packed_modules_mapping: Dict[str, Any],
    ):
        self.quant_config = quant_config
        self.prefix = prefix
        self.packed_modules_mapping = packed_modules_mapping
        self.is_down_proj = "down_proj" in prefix

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        """Create ResQ weight parameters."""
        if hasattr(layer, "weight"):
            del layer.weight

        def register_resq_param(name: str, dtype: torch.dtype = torch.float32):
            param = torch.nn.Parameter(
                torch.empty(0, dtype=dtype), requires_grad=False
            )
            layer.register_parameter(name, param)
            set_weight_attrs(param, extra_weight_attrs)
            setattr(param, "weight_loader", self.weight_loader)

        register_resq_param("weight_high", torch.int8)
        register_resq_param("weight_low", torch.int8)
        register_resq_param("scale_high", torch.float32)
        register_resq_param("scale_low", torch.float32)

        if self.is_down_proj:
            register_resq_param("rotation_Pd", torch.float32)
            register_resq_param("rotation_Hd", torch.float32)
            h_butterfly = torch.nn.Parameter(
                torch.empty(0, dtype=torch.float32), requires_grad=False
            )
            layer.register_parameter("h_butterfly", h_butterfly)
            set_weight_attrs(h_butterfly, extra_weight_attrs)

        # Register high_fraction from checkpoint, default to 0.0 scalar
        high_fraction = torch.nn.Parameter(
            torch.tensor(0.0, dtype=torch.float32), requires_grad=False
        )
        layer.register_parameter("high_fraction", high_fraction)
        set_weight_attrs(high_fraction, extra_weight_attrs)
        setattr(high_fraction, "weight_loader", self.weight_loader)

    def weight_loader(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        shard_id: Optional[int] = None,
        **kwargs,
    ):
        """Load weights, buffering shards for merged layers."""
        if shard_id is None:
            shard_id = 0

        if not hasattr(param, "_shards"):
            param._shards = {}

        param._shards[shard_id] = loaded_weight

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Process weights after all shards are loaded."""
        for name in [
            "weight_low",
            "weight_high",
            "scale_low",
            "scale_high",
            "rotation_Pd",
            "rotation_Hd",
        ]:
            if hasattr(layer, name):
                param = getattr(layer, name)
                if hasattr(param, "_shards"):
                    def shard_sort_key(k):
                        if isinstance(k, str):
                            k_lower = k.lower()
                            if "q" in k_lower:
                                return 0
                            if "k" in k_lower:
                                return 1
                            if "v" in k_lower:
                                return 2
                        return k

                    sorted_keys = sorted(param._shards.keys(), key=shard_sort_key)
                    sorted_shards = [param._shards[k] for k in sorted_keys]

                    if len(sorted_shards) > 0:
                        full_weight = torch.cat(sorted_shards, dim=0)
                        param.data = full_weight.to(param.device, dtype=param.dtype)

                    del param._shards

        # Process high_fraction (scalar, no concatenation needed)
        if hasattr(layer, "high_fraction"):
            param = layer.high_fraction
            if hasattr(param, "_shards") and len(param._shards) > 0:
                first_shard = list(param._shards.values())[0]
                param.data = first_shard.to(param.device, dtype=param.dtype)
                del param._shards

        # Pre-compute butterfly Hadamard matrix
        if (
            self.is_down_proj
            and hasattr(layer, "rotation_Pd")
            and layer.rotation_Pd.numel() > 0
        ):
            blocksize = layer.rotation_Pd.shape[0]
            if layer.h_butterfly.numel() == 0 and blocksize > 0:
                eye = torch.eye(blocksize, dtype=torch.float32)
                h_matrix = _hadamard_transform(eye)
                layer.h_butterfly.data = h_matrix.to(layer.rotation_Pd.device)

        # TODO: .squeeze(0) operations need to be removed for MoE model
        # Process weight_low (int4): use NPU native packing
        # weight_low shape: (n, k_low) -> pack to (n, k_low//8)
        w_low = layer.weight_low.data
        # w_low_packed = torch_npu.npu_convert_weight_to_int4pack(w_low)
        w_low_packed = _pack_int4_to_int8_signed(w_low).contiguous().npu()
        # Transpose to (k_low//8, n) and make contiguous to match weight_high's layout status
        layer.register_buffer("weight_low_packed", w_low_packed)

        # Process weight_high (int8): transpose to (k_high, n) + NZ format
        w_high = layer.weight_high.data.npu()
        if is_enable_nz():
            w_high = torch_npu.npu_format_cast(w_high, ACL_FORMAT_FRACTAL_NZ)
            w_low_packed = torch_npu.npu_format_cast(w_low_packed, ACL_FORMAT_FRACTAL_NZ).view(torch.int32)
        layer.weight_high.data = w_high
        layer.weight_low_packed.data = w_low_packed

        # Process scales: flatten to 1D float32
        layer.scale_low.data = _convert_scales(layer.scale_low.data.flatten().to(torch.float32).npu())
        layer.scale_high.data = _convert_scales(layer.scale_high.data.flatten().to(torch.float32).npu())


    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        tp_rank: int = 0,
        **kwargs,
        ) -> torch.Tensor:
            """Apply ResQ quantized linear transformation."""
            if not x.is_contiguous():
                x = x.contiguous()

            # Apply Ud rotation for down_proj
            if self.is_down_proj and layer.rotation_Pd.numel() > 0:
                Hd = layer.rotation_Hd if layer.rotation_Hd.numel() > 0 else None
                K = Hd.shape[0] if Hd is not None else 1
                blocksize = layer.rotation_Pd.shape[0]

                x = x.contiguous()
                x = _apply_ud_rotation(
                    x,
                    Pd=layer.rotation_Pd,
                    Hd=Hd,
                    h_butterfly=layer.h_butterfly if layer.h_butterfly.numel() > 0 else None,
                    K=K,
                    blocksize=blocksize,
                )

            # Mixed precision matmul
            original_shape = x.shape
            x_2d = x.contiguous().view(-1, x.shape[-1]).float()

            in_high = layer.weight_high.shape[-1]
            in_low = x_2d.shape[-1] - in_high

            x_low = x_2d[:, :in_low].to(torch.float16).npu()
            x_high = x_2d[:, in_low:].to(torch.float16).npu()

            x_low_quant, lx_scale = torch_npu.npu_dynamic_quant(
                x_low.contiguous(), dst_type=torch.quint4x2
            )
            x_high_quant, rx_scale = torch_npu.npu_dynamic_quant(
                x_high, dst_type=torch.int8
            )
            # int4 matmul: weight_low_packed (n, k_low//8) -> transpose to (k_low//8, n)
            # int8 matmul: weight_high already (k_high, n) with NZ format
            output = torch_npu.npu_mixprecise_quant_matmul(x_low_quant, layer.weight_low_packed.transpose(-1,-2),
                                                        rx=x_high_quant, hweight=layer.weight_high.transpose(-1,-2),
                                                        bias=None, lscale=layer.scale_low, hscale=layer.scale_high, 
                                                        lper_token_scale=lx_scale, rper_token_scale=rx_scale,
                                                        output_dtype=torch.float16, mix_type=0, split_kpos=in_low)

            # TODO: this could be an issue if bias is not None, as so far 
            # we do not consider how and if bias should be reordered during quantization
            if bias is not None:
                output = output + bias

            output_shape = list(original_shape[:-1]) + [output.shape[-1]]
            return output.view(output_shape).to(torch.bfloat16)
