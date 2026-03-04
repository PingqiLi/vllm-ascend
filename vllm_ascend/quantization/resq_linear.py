#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
"""ResQ mixed-precision quantization linear method."""

from typing import Any, Dict, List, Optional

import numpy as np

import torch
import torch_npu
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.model_executor.layers.linear import LinearMethodBase
from vllm.model_executor.utils import set_weight_attrs

from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ, is_enable_nz


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

        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()

        self.h_dim = None
        self.l_dim = None

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
        # high_fraction is present in all RESQ layers
        register_resq_param("high_fraction", torch.float32)

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
            "high_fraction",
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
                        # Special handling for scalar parameters (like high_fraction)
                        # For merged layers, just use the first shard (they should all be the same)
                        if name == "high_fraction" and sorted_shards[0].dim() == 0:
                            full_weight = sorted_shards[0]
                        else:
                            full_weight = torch.cat(sorted_shards, dim=0)
                        param.data = full_weight.to(param.device, dtype=param.dtype)

                    del param._shards

        self.h_dim = layer.weight_high.shape[1]
        self.l_dim = layer.weight_low.shape[1]

        assert self.h_dim % self.tp_size == 0
        assert self.l_dim % self.tp_size == 0

        # Process weight_low (int4): use NPU native packing
        # weight_low shape: (n, k_low) -> pack to (n, k_low//8)
        w_low = layer.weight_low.data
        w_low_packed = _pack_int4_to_int8_signed(w_low).contiguous().npu()
        layer.register_buffer("weight_low_packed", w_low_packed)

        # Process weight_high (int8): keep (n, k_high) layout.
        # We transpose in apply() so low/high weights share the same transpose status.
        w_high = layer.weight_high.data.contiguous().npu()
        layer.weight_high.data = w_high
        layer.weight_low_packed.data = w_low_packed

        # Process scales: flatten to 1D float32
        layer.scale_low.data = _convert_scales(layer.scale_low.data.flatten().to(torch.float32).npu())
        layer.scale_high.data = _convert_scales(layer.scale_high.data.flatten().to(torch.float32).npu())

        # TP
        if self.prefix.endswith("qkv_proj") or self.prefix.endswith("gate_up_proj"):
            # _convert_scales stores scales as [1, N] int64 payload.
            # TP slicing for qkv/gate_up should therefore index the last dim.
            assert layer.weight_low_packed.data.shape[0] == layer.scale_low.data.shape[-1]
            assert layer.weight_high.data.shape[0] == layer.scale_high.data.shape[-1]
            n = layer.weight_low_packed.data.shape[0]

            if self.prefix.endswith("gate_up_proj"):
                assert n % 2 == 0
                shard_offsets = [0, n // 2]
                shard_sizes = [n // 2, n // 2]
            elif self.prefix.endswith("qkv_proj"):
                shard_offsets = [0, layer.output_sizes[0], layer.output_sizes[0] + layer.output_sizes[1]]
                shard_sizes = layer.output_sizes

            weight_low_packed = []
            scale_low = []
            weight_high = []
            scale_high = []

            for shard_offset, shard_size in zip(shard_offsets, shard_sizes):
                assert shard_size % self.tp_size == 0
                chunk_size = shard_size // self.tp_size
                begin = shard_offset + self.tp_rank * chunk_size
                end = begin + chunk_size
                weight_low_packed.append(layer.weight_low_packed.data[begin: end, :])
                weight_high.append(layer.weight_high.data[begin: end, :])
                scale_low.append(layer.scale_low.data[..., begin: end])
                scale_high.append(layer.scale_high.data[..., begin: end])

            layer.weight_low_packed.data = torch.cat(weight_low_packed)
            layer.weight_high.data = torch.cat(weight_high, dim=0)
            layer.scale_low.data = torch.cat(scale_low, dim=-1)
            layer.scale_high.data = torch.cat(scale_high, dim=-1)

        elif self.prefix.endswith("o_proj"):
            # Split low branch by the actual packed width before NZ/int32 view.
            # This keeps TP logic correct regardless of packing implementation
            # (e.g. int8 packed width = k/2, then optionally viewed to int32).
            packed_cols = layer.weight_low_packed.data.shape[1]
            assert packed_cols % self.tp_size == 0
            assert self.h_dim % self.tp_size == 0
            chunk_size_l = packed_cols // self.tp_size
            chunk_size_h = self.h_dim // self.tp_size
            begin_l = chunk_size_l * self.tp_rank 
            end_l = chunk_size_l * (self.tp_rank + 1)
            begin_h = chunk_size_h * self.tp_rank
            end_h = chunk_size_h * (self.tp_rank + 1)

            layer.weight_low_packed.data = layer.weight_low_packed.data[:, begin_l: end_l].clone()
            layer.weight_high.data = layer.weight_high.data[:, begin_h: end_h].clone()

        # Keep TP slicing/cat on default contiguous layout first, then cast to NZ.
        if is_enable_nz():
            layer.weight_high.data = torch_npu.npu_format_cast(
                layer.weight_high.data, ACL_FORMAT_FRACTAL_NZ
            )
            layer.weight_low_packed.data = torch_npu.npu_format_cast(
                layer.weight_low_packed.data, ACL_FORMAT_FRACTAL_NZ
            ).view(torch.int32)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Apply ResQ quantized linear transformation."""
        if not x.is_contiguous():
            x = x.contiguous()

        # Mixed precision matmul
        original_shape = x.shape
        x_2d = x.contiguous().view(-1, x.shape[-1])

        # weight_high is (n, k_high) before matmul; matmul uses transposed view.
        in_high = layer.weight_high.shape[-1]
        in_low = x_2d.shape[-1] - in_high

        x_low = x_2d[:, :in_low].npu()
        x_high = x_2d[:, in_low:].npu()

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
                                                    output_dtype=torch.bfloat16, mix_type=0, split_kpos=in_low)

        # TODO: this could be an issue if bias is not None, as so far 
        # we do not consider how and if bias should be reordered during quantization
        if bias is not None:
            raise NotImplementedError()
            output = output + bias

        output_shape = list(original_shape[:-1]) + [output.shape[-1]]
        return output.view(output_shape)
