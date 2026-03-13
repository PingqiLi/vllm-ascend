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
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import LinearMethodBase
from vllm.model_executor.utils import set_weight_attrs

from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ, is_enable_nz

logger = init_logger(__name__)


def _pack_int4_to_int8_signed(x: torch.Tensor) -> torch.Tensor:
    """Pack int8 tensor (values in [-8, 7]) into int4 pairs.

    Args:
        x: int8 tensor, shape (K, N), values in [-8, 7].

    Returns:
        int8 tensor, shape (K, N/2), each element packs two
        signed int4 values.
    """
    assert x.dtype == torch.int8
    K, N = x.shape
    assert N % 2 == 0

    x_unsigned = torch.where(x < 0, x + 16, x).to(torch.int32)

    low = x_unsigned[..., 0::2]
    high = x_unsigned[..., 1::2]

    out = (low | (high << 4)).to(torch.int8)
    return out


def _convert_scales(scales):
    """Convert float32 per-channel scales to the uint64-packed
    format expected by ``npu_mixprecise_quant_matmul``."""
    N = scales.shape[0]
    scale_u32 = (scales.cpu().to(torch.float32).clone().numpy().astype(np.float32).reshape(1, N))
    scale_u32.dtype = np.uint32
    scale_u64 = np.zeros((1, N * 2), dtype=np.uint32)
    scale_u64[..., ::2] = scale_u32
    scale_u64.dtype = np.int64
    scale = torch.from_numpy(scale_u64).npu()
    return scale


def _hadamard_transform(u: torch.Tensor) -> torch.Tensor:
    """Fast Hadamard transform (unnormalized, butterfly)."""
    n = u.shape[-1]
    x = u.reshape(-1, n).clone()
    h = 1
    while h < n:
        x = x.view(-1, n // (2 * h), 2, h)
        a = x[:, :, 0, :]
        b = x[:, :, 1, :]
        x = torch.stack([a + b, a - b], dim=2)
        x = x.view(-1, n)
        h *= 2
    return x.view(u.shape)


class ResQLinearMethod(LinearMethodBase):
    """Linear method for ResQ mixed-precision quantization.

    ResQ uses a mixed-precision approach with:
    - weight_low: int4 weights (stored as int8, packed during
      processing)
    - weight_high: int8 weights
    - scale_low, scale_high: per-channel scales

    Supports two checkpoint modes:
    - **Adaptive**: o_proj is ResQ (needs column reordering).
    - **Ua-only**: all layers are ResQ with a fixed split.

    When ``rd_block_size`` is present in the checkpoint for a
    down_proj layer (perm_rd mode), an online block-Hadamard
    rotation Rd is applied to the activation before the
    mixed-precision split.
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
        self.is_o_proj = "o_proj" in prefix
        self.is_down_proj = "down_proj" in prefix

        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()

        self.h_dim = None
        self.l_dim = None
        self._rd_block_size = 0

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

        def register_resq_param(
            name: str,
            dtype: torch.dtype = torch.float32,
        ):
            param = torch.nn.Parameter(torch.empty(0, dtype=dtype),
                                       requires_grad=False)
            layer.register_parameter(name, param)
            set_weight_attrs(param, extra_weight_attrs)
            setattr(param, "weight_loader", self.weight_loader)

        register_resq_param("weight_high", torch.int8)
        register_resq_param("weight_low", torch.int8)
        register_resq_param("scale_high", torch.float32)
        register_resq_param("scale_low", torch.float32)
        register_resq_param("high_fraction", torch.float32)

        if self.is_down_proj:
            register_resq_param("rd_block_size", torch.int32)

        if self.is_o_proj:
            layer.register_buffer(
                "o_proj_column_order",
                torch.empty(0, dtype=torch.long),
            )

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

    # ----------------------------------------------------------
    # process_weights_after_loading
    # ----------------------------------------------------------

    def process_weights_after_loading(
        self,
        layer: torch.nn.Module,
    ) -> None:
        """Process weights after all shards are loaded."""
        for name in [
                "weight_low",
                "weight_high",
                "scale_low",
                "scale_high",
                "high_fraction",
                "rd_block_size",
        ]:
            if not hasattr(layer, name):
                continue
            param = getattr(layer, name)
            if not hasattr(param, "_shards"):
                continue

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

            sorted_keys = sorted(
                param._shards.keys(),
                key=shard_sort_key,
            )
            sorted_shards = [param._shards[k] for k in sorted_keys]

            if len(sorted_shards) > 0:
                if (name in ("high_fraction", "rd_block_size")
                        and sorted_shards[0].dim() == 0):
                    full_weight = sorted_shards[0]
                else:
                    full_weight = torch.cat(sorted_shards, dim=0)
                param.data = full_weight.to(param.device,
                                            dtype=param.dtype)

            del param._shards

        self.h_dim = layer.weight_high.shape[1]
        self.l_dim = layer.weight_low.shape[1]

        assert self.h_dim % self.tp_size == 0
        assert self.l_dim % self.tp_size == 0

        if self.is_o_proj:
            self._setup_o_proj_column_order(layer)

        # perm_rd: extract rd_block_size and pre-compute H_b
        self._rd_block_size = 0
        if (self.is_down_proj
                and hasattr(layer, "rd_block_size")
                and layer.rd_block_size.numel() > 0):
            bs = int(layer.rd_block_size.item())
            if bs > 0:
                self._rd_block_size = bs
                eye = torch.eye(bs, dtype=torch.float32)
                h_b = _hadamard_transform(eye) / math.sqrt(bs)
                layer.register_buffer(
                    "h_block", h_b.npu())

        w_low_packed = _pack_int4_to_int8_signed(
            layer.weight_low.data).contiguous().npu()
        layer.register_buffer("weight_low_packed", w_low_packed)

        layer.weight_high.data = (
            layer.weight_high.data.contiguous().npu())
        layer.weight_low_packed.data = w_low_packed

        layer.scale_low.data = _convert_scales(
            layer.scale_low.data.flatten().to(
                torch.float32).npu())
        layer.scale_high.data = _convert_scales(
            layer.scale_high.data.flatten().to(
                torch.float32).npu())

        self._tp_slice(layer)

        if is_enable_nz():
            layer.weight_high.data = (
                torch_npu.npu_format_cast(
                    layer.weight_high.data,
                    ACL_FORMAT_FRACTAL_NZ,
                ))
            layer.weight_low_packed.data = (
                torch_npu.npu_format_cast(
                    layer.weight_low_packed.data,
                    ACL_FORMAT_FRACTAL_NZ,
                ).view(torch.int32))

    # ----------------------------------------------------------
    # TP slicing
    # ----------------------------------------------------------

    def _tp_slice(self, layer: torch.nn.Module) -> None:
        """Tensor-parallel slicing for qkv / gate_up / o_proj."""
        if self.tp_size <= 1:
            return

        if (self.prefix.endswith("qkv_proj")
                or self.prefix.endswith("gate_up_proj")):
            self._tp_slice_row_parallel(layer)
        elif self.prefix.endswith("o_proj"):
            self._tp_slice_o_proj(layer)

    def _tp_slice_row_parallel(
        self,
        layer: torch.nn.Module,
    ) -> None:
        """TP slicing for row-parallel (qkv_proj, gate_up_proj).

        Weights are (n, k); scales are [1, N] int64 payload.
        We slice along the output (n) dimension."""
        assert (layer.weight_low_packed.data.shape[0]
                == layer.scale_low.data.shape[-1])
        assert (layer.weight_high.data.shape[0]
                == layer.scale_high.data.shape[-1])
        n = layer.weight_low_packed.data.shape[0]

        if self.prefix.endswith("gate_up_proj"):
            assert n % 2 == 0
            shard_offsets = [0, n // 2]
            shard_sizes = [n // 2, n // 2]
        elif self.prefix.endswith("qkv_proj"):
            shard_offsets = [
                0,
                layer.output_sizes[0],
                layer.output_sizes[0] + layer.output_sizes[1],
            ]
            shard_sizes = layer.output_sizes

        wlp, sl, whp, sh = [], [], [], []

        for shard_offset, shard_size in zip(
                shard_offsets, shard_sizes):
            assert shard_size % self.tp_size == 0
            chunk = shard_size // self.tp_size
            b = shard_offset + self.tp_rank * chunk
            e = b + chunk
            wlp.append(layer.weight_low_packed.data[b:e, :])
            sl.append(layer.scale_low.data[..., b:e])
            whp.append(layer.weight_high.data[b:e, :])
            sh.append(layer.scale_high.data[..., b:e])

        layer.weight_low_packed.data = torch.cat(wlp)
        layer.scale_low.data = torch.cat(sl, dim=-1)
        layer.weight_high.data = torch.cat(whp, dim=0)
        layer.scale_high.data = torch.cat(sh, dim=-1)

    def _tp_slice_o_proj(
        self,
        layer: torch.nn.Module,
    ) -> None:
        """TP slicing for o_proj (column-parallel on input).

        Weights are (n, k); we split along the k dimension."""
        packed_cols = layer.weight_low_packed.data.shape[1]
        assert packed_cols % self.tp_size == 0
        assert self.h_dim % self.tp_size == 0
        chunk_l = packed_cols // self.tp_size
        chunk_h = self.h_dim // self.tp_size
        b_l = chunk_l * self.tp_rank
        e_l = chunk_l * (self.tp_rank + 1)
        b_h = chunk_h * self.tp_rank
        e_h = chunk_h * (self.tp_rank + 1)

        layer.weight_low_packed.data = (layer.weight_low_packed.data[:, b_l:e_l].clone())
        layer.weight_high.data = (layer.weight_high.data[:, b_h:e_h].clone())

    # ----------------------------------------------------------
    # o_proj column reordering
    # ----------------------------------------------------------

    def _get_new_column_order(
        self,
        in_dim: int,
        head_dim: int,
        high_per_head: int,
        device,
    ) -> torch.Tensor:
        """Compute column reorder indices.

        Original layout: ``[head0_cols, head1_cols, ...]``.
        Each head has ``head_dim`` columns.  We move the last
        ``high_per_head`` columns of each head to the end.

        Returns:
            ``[remaining (low) | high]`` index tensor.
        """
        chunk_starts = torch.arange(0, in_dim, head_dim, device=device)
        high_cols = torch.arange(
            head_dim - high_per_head,
            head_dim,
            device=device,
        )
        columns_to_end = (chunk_starts.unsqueeze(1) + high_cols).flatten()

        all_columns = torch.arange(in_dim, device=device)
        mask = torch.ones(in_dim, dtype=torch.bool, device=device)
        mask[columns_to_end] = False
        remaining = all_columns[mask]

        return torch.cat([remaining, columns_to_end])

    def _setup_o_proj_column_order(
        self,
        layer: torch.nn.Module,
    ) -> None:
        """Setup o_proj input column reorder indices.

        The ResQ quantization reorders o_proj weights to
        ``[low | high]`` layout.  We reorder the input
        (``attn_output``) columns to match.

        ``high_per_head`` is derived directly from the weight
        shapes (``in_high // num_heads``) rather than from
        ``high_fraction`` to avoid floating-point rounding
        mismatches with the aligned splits used by msit.
        """
        in_low = layer.weight_low.shape[1]
        in_high = layer.weight_high.shape[1]
        in_dim = in_low + in_high

        head_dim = 128  # Qwen3-32B head_dim
        num_heads = in_dim // head_dim
        high_per_head = in_high // num_heads

        layer.o_proj_column_order = (
            self._get_new_column_order(
                in_dim // self.tp_size,
                head_dim,
                high_per_head,
                layer.weight_low.device,
            ))

    # ----------------------------------------------------------
    # apply (forward)
    # ----------------------------------------------------------

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        tp_rank: int = 0,
    ) -> torch.Tensor:
        """Apply ResQ quantized linear transformation."""
        if not x.is_contiguous():
            x = x.contiguous()

        if (self.is_o_proj
                and hasattr(layer, "o_proj_column_order")
                and layer.o_proj_column_order.numel() > 0):
            x = x[..., layer.o_proj_column_order]

        original_shape = x.shape
        x_2d = x.contiguous().view(-1, x.shape[-1])

        # perm_rd: online block Hadamard before split
        if (self._rd_block_size > 0
                and hasattr(layer, "h_block")):
            bs = self._rd_block_size
            orig_dtype = x_2d.dtype
            M, D = x_2d.shape
            x_2d = x_2d.float().view(M, D // bs, bs)
            x_2d = torch.matmul(x_2d, layer.h_block)
            x_2d = x_2d.view(M, D).to(orig_dtype)

        in_high = layer.weight_high.shape[-1]
        in_low = x_2d.shape[-1] - in_high

        x_low = x_2d[:, :in_low].npu()
        x_high = x_2d[:, in_low:].npu()

        x_low_quant, lx_scale = (
            torch_npu.npu_dynamic_quant(
                x_low.contiguous(),
                dst_type=torch.quint4x2,
            ))
        x_high_quant, rx_scale = (
            torch_npu.npu_dynamic_quant(
                x_high, dst_type=torch.int8,
            ))

        output = torch_npu.npu_mixprecise_quant_matmul(
            x_low_quant,
            layer.weight_low_packed.transpose(-1, -2),
            rx=x_high_quant,
            hweight=layer.weight_high.transpose(-1, -2),
            bias=None,
            lscale=layer.scale_low,
            hscale=layer.scale_high,
            lper_token_scale=lx_scale,
            rper_token_scale=rx_scale,
            output_dtype=torch.bfloat16,
            mix_type=0,
            split_kpos=in_low,
        )

        if bias is not None:
            raise NotImplementedError(
                "Bias not yet supported for ResQ linear")

        output_shape = (
            list(original_shape[:-1]) + [output.shape[-1]])
        return output.view(output_shape)
