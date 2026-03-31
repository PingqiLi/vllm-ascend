#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""ResQ mixed-precision quantization linear method."""

import math
from typing import Any

import torch
import torch_npu
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import LinearMethodBase
from vllm.model_executor.utils import set_weight_attrs

from vllm_ascend.utils import maybe_trans_nz

logger = init_logger(__name__)


def _hadamard_transform(u: torch.Tensor) -> torch.Tensor:
    """Fast Hadamard transform (unnormalized, butterfly)."""
    n = u.shape[-1]
    x = u.reshape(-1, n).clone()
    h = 1
    while h < n:
        x = x.view(-1, n // (2 * h), 2, h)
        a, b = x[:, :, 0, :], x[:, :, 1, :]
        x = torch.stack([a + b, a - b], dim=2).view(-1, n)
        h *= 2
    return x.view(u.shape)


def _block_hadamard(x: torch.Tensor, block_size: int,
                    h_matrix: torch.Tensor) -> torch.Tensor:
    """Apply block-diagonal Hadamard transform.

    Args:
        x: input tensor [M, D], D must be divisible by block_size.
        block_size: Hadamard block size (e.g. 32).
        h_matrix: pre-computed normalized H matrix [block_size, block_size].

    Returns:
        Transformed tensor, same shape as x.
    """
    M, D = x.shape
    return torch.matmul(
        x.view(M, D // block_size, block_size), h_matrix
    ).view(M, D)


class ResQLinearMethod(LinearMethodBase):
    """Linear method for ResQ mixed-precision quantization.

    ResQ uses a mixed-precision approach with:
    - weight_low: int4 weights (stored as int8, packed during processing)
    - weight_high: int8 weights
    - scale_low, scale_high: per-channel scales

    For down_proj layers with perm_rd mode (``rd_block_size`` and
    ``perm_group_size`` present in checkpoint), an online per-group
    split-then-Hadamard rotation is applied before quantized matmul:
    each group of ``perm_group_size`` channels is split into [low | high],
    then block Hadamard H_{block_size} is applied to each part independently.
    """

    SCALAR_PARAMS = ("high_fraction", "rd_block_size", "perm_group_size")

    def __init__(self, quant_config: dict[str, Any], prefix: str,
                 packed_modules_mapping: dict[str, Any]):
        self.quant_config = quant_config
        self.prefix = prefix
        self.packed_modules_mapping = packed_modules_mapping
        self.is_o_proj = "o_proj" in prefix
        self.is_down_proj = "down_proj" in prefix

        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()

        self.high_dim = None
        self.low_dim = None
        self._rd_block_size = 0
        self._perm_group_size = 0

    # ------------------------------------------------------------------
    # Weight creation & loading
    # ------------------------------------------------------------------

    def create_weights(self, layer: torch.nn.Module,
                       input_size_per_partition: int,
                       output_partition_sizes: list[int],
                       input_size: int, output_size: int,
                       params_dtype: torch.dtype,
                       **extra_weight_attrs) -> None:
        """Create ResQ weight parameters."""
        if hasattr(layer, "weight"):
            del layer.weight

        def register(name: str, dtype: torch.dtype = torch.float32):
            param = torch.nn.Parameter(
                torch.empty(0, dtype=dtype), requires_grad=False)
            layer.register_parameter(name, param)
            set_weight_attrs(param, extra_weight_attrs)
            setattr(param, "weight_loader", self.weight_loader)

        register("weight_high", torch.int8)
        register("weight_low", torch.int8)
        register("scale_high", torch.float32)
        register("scale_low", torch.float32)
        register("high_fraction", torch.float32)

        if self.is_down_proj:
            register("rd_block_size", torch.int32)
            register("perm_group_size", torch.int32)

        if self.is_o_proj:
            layer.register_buffer(
                "o_proj_column_order",
                torch.empty(0, dtype=torch.long))

    def weight_loader(self, param: torch.nn.Parameter,
                      loaded_weight: torch.Tensor,
                      shard_id: int | None = None,
                      **kwargs):
        """Load weights, buffering shards for merged layers."""
        if shard_id is None:
            shard_id = 0
        if not hasattr(param, "_shards"):
            param._shards = {}
        param._shards[shard_id] = loaded_weight

    # ------------------------------------------------------------------
    # Post-loading processing
    # ------------------------------------------------------------------

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Merge shards, TP-slice on raw tensors, then convert formats."""
        self._merge_shards(layer)

        self.high_dim = layer.weight_high.shape[1]
        self.low_dim = layer.weight_low.shape[1]
        assert self.high_dim % self.tp_size == 0
        assert self.low_dim % self.tp_size == 0

        if self.is_o_proj:
            self._setup_o_proj_column_order(layer)

        if self.is_down_proj:
            self._setup_perm_rd(layer)

        # Flatten scales to 1D float32
        layer.scale_low.data = layer.scale_low.data.flatten().to(torch.float32)
        layer.scale_high.data = layer.scale_high.data.flatten().to(torch.float32)

        # TP slice on raw tensors (before format conversion)
        self._tp_slice(layer)

        # Pack int4 weights using NPU native packing
        w_low = layer.weight_low.data.to(torch.int32).npu()
        weight_low_packed = torch_npu.npu_convert_weight_to_int4pack(w_low)
        layer.register_buffer("weight_low_packed", weight_low_packed)

        # Transpose weight_high to (K, N) and apply NZ format
        w_high = layer.weight_high.data.transpose(0, 1).contiguous().npu()
        layer.weight_high.data = maybe_trans_nz(w_high)

        # Move scales to NPU
        layer.scale_low.data = layer.scale_low.data.npu()
        layer.scale_high.data = layer.scale_high.data.npu()

    def _merge_shards(self, layer: torch.nn.Module) -> None:
        """Merge buffered weight shards into single tensors."""
        param_names = [
            "weight_low", "weight_high", "scale_low", "scale_high",
            "high_fraction", "rd_block_size", "perm_group_size",
        ]

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

        for name in param_names:
            if not hasattr(layer, name):
                continue
            param = getattr(layer, name)
            if not hasattr(param, "_shards"):
                continue

            sorted_keys = sorted(param._shards.keys(), key=shard_sort_key)
            shards = [param._shards[k] for k in sorted_keys]

            if shards:
                if name in self.SCALAR_PARAMS and shards[0].dim() == 0:
                    merged = shards[0]
                else:
                    merged = torch.cat(shards, dim=0)
                param.data = merged.to(param.device, dtype=param.dtype)

            del param._shards

    def _setup_perm_rd(self, layer: torch.nn.Module) -> None:
        """Extract perm_rd parameters and pre-compute Hadamard matrix."""
        self._rd_block_size = 0
        self._perm_group_size = 0

        if (hasattr(layer, "rd_block_size")
                and layer.rd_block_size.numel() > 0):
            block_size = int(layer.rd_block_size.item())
            if block_size > 0:
                self._rd_block_size = block_size
                eye = torch.eye(block_size, dtype=torch.float32)
                h_matrix = _hadamard_transform(eye) / math.sqrt(block_size)
                layer.register_buffer("h_block", h_matrix.npu())

        if (hasattr(layer, "perm_group_size")
                and layer.perm_group_size.numel() > 0):
            group_size = int(layer.perm_group_size.item())
            if group_size > 0:
                self._perm_group_size = group_size

    # ------------------------------------------------------------------
    # TP slicing
    # ------------------------------------------------------------------

    def _tp_slice(self, layer: torch.nn.Module) -> None:
        """Tensor-parallel slicing for all ResQ layers."""
        if self.tp_size <= 1:
            return
        if (self.prefix.endswith("qkv_proj")
                or self.prefix.endswith("gate_up_proj")):
            self._tp_slice_output_dim(layer)
        elif (self.prefix.endswith("o_proj")
              or self.prefix.endswith("down_proj")):
            self._tp_slice_input_dim(layer)

    def _tp_slice_output_dim(self, layer: torch.nn.Module) -> None:
        """TP slicing for column-parallel layers (qkv_proj, gate_up_proj).

        Weights are raw (N, K); scales are flat 1D (N,).
        We slice along the output (N) dimension.
        """
        total_rows = layer.weight_low.data.shape[0]

        if self.prefix.endswith("gate_up_proj"):
            assert total_rows % 2 == 0
            half = total_rows // 2
            shard_offsets = [0, half]
            shard_sizes = [half, half]
        elif self.prefix.endswith("qkv_proj"):
            shard_offsets = [
                0,
                layer.output_sizes[0],
                layer.output_sizes[0] + layer.output_sizes[1],
            ]
            shard_sizes = layer.output_sizes

        low_parts, scale_low_parts = [], []
        high_parts, scale_high_parts = [], []
        for offset, size in zip(shard_offsets, shard_sizes):
            assert size % self.tp_size == 0
            chunk = size // self.tp_size
            start = offset + self.tp_rank * chunk
            end = start + chunk
            low_parts.append(layer.weight_low.data[start:end, :])
            scale_low_parts.append(layer.scale_low.data[start:end])
            high_parts.append(layer.weight_high.data[start:end, :])
            scale_high_parts.append(layer.scale_high.data[start:end])

        layer.weight_low.data = torch.cat(low_parts, dim=0)
        layer.scale_low.data = torch.cat(scale_low_parts, dim=0)
        layer.weight_high.data = torch.cat(high_parts, dim=0)
        layer.scale_high.data = torch.cat(scale_high_parts, dim=0)

    def _tp_slice_input_dim(self, layer: torch.nn.Module) -> None:
        """TP slicing along input dimension (o_proj, down_proj).

        Weights are raw (N, K); we split along the K dimension.
        Scales are per-output-channel so they remain unchanged.
        """
        assert self.low_dim % self.tp_size == 0
        assert self.high_dim % self.tp_size == 0
        chunk_low = self.low_dim // self.tp_size
        chunk_high = self.high_dim // self.tp_size

        start_low = chunk_low * self.tp_rank
        start_high = chunk_high * self.tp_rank
        layer.weight_low.data = layer.weight_low.data[
            :, start_low:start_low + chunk_low].clone()
        layer.weight_high.data = layer.weight_high.data[
            :, start_high:start_high + chunk_high].clone()

    # ------------------------------------------------------------------
    # o_proj column reordering
    # ------------------------------------------------------------------

    def _get_column_reorder(self, in_dim: int, head_dim: int,
                            high_per_head: int,
                            device) -> torch.Tensor:
        """Compute column reorder indices for o_proj.

        Original layout: [head0_cols, head1_cols, ...].
        Each head has ``head_dim`` columns.  The last ``high_per_head``
        columns of each head are moved to the end.

        Returns:
            [remaining (low) | high] index tensor.
        """
        chunk_starts = torch.arange(
            0, in_dim, head_dim, device=device)
        high_cols = torch.arange(
            head_dim - high_per_head, head_dim, device=device)
        columns_to_end = (
            chunk_starts.unsqueeze(1) + high_cols).flatten()

        all_columns = torch.arange(in_dim, device=device)
        mask = torch.ones(in_dim, dtype=torch.bool, device=device)
        mask[columns_to_end] = False
        return torch.cat([all_columns[mask], columns_to_end])

    def _setup_o_proj_column_order(self, layer: torch.nn.Module) -> None:
        """Setup o_proj input column reorder indices.

        ``high_per_head`` is derived from weight shapes
        (``in_high // num_heads``) rather than from ``high_fraction``
        to avoid rounding mismatches with aligned splits from msit.
        """
        in_low = layer.weight_low.shape[1]
        in_high = layer.weight_high.shape[1]
        in_dim = in_low + in_high
        head_dim = 128  # Qwen3-32B head_dim
        num_heads = in_dim // head_dim
        high_per_head = in_high // num_heads

        layer.o_proj_column_order = self._get_column_reorder(
            in_dim // self.tp_size, head_dim, high_per_head,
            layer.weight_low.device)

    # ------------------------------------------------------------------
    # Forward (apply)
    # ------------------------------------------------------------------

    def apply(self, layer: torch.nn.Module, x: torch.Tensor,
              bias: torch.Tensor | None = None,
              tp_rank: int = 0) -> torch.Tensor:
        """Apply ResQ quantized linear transformation."""
        if not x.is_contiguous():
            x = x.contiguous()

        # o_proj: reorder activation columns to match [low | high] weight
        if (self.is_o_proj
                and hasattr(layer, "o_proj_column_order")
                and layer.o_proj_column_order.numel() > 0):
            x = x[..., layer.o_proj_column_order]

        original_shape = x.shape
        x_2d = x.contiguous().view(-1, x.shape[-1]).float()

        # Split activation into low/high parts
        x_low, x_high = self._split_activation(layer, x_2d)

        # Quantize activations
        x_low_quant, low_act_scale = torch_npu.npu_dynamic_quant(
            x_low.contiguous(), dst_type=torch.quint4x2)
        x_high_quant, high_act_scale = torch_npu.npu_dynamic_quant(
            x_high, dst_type=torch.int8)

        # int4 matmul: weight_low_packed is NPU int4pack format
        output_low = torch_npu.npu_quant_matmul(
            x_low_quant,
            layer.weight_low_packed.t(),
            layer.scale_low,
            pertoken_scale=low_act_scale,
            output_dtype=torch.float16,
        )

        # int8 matmul: weight_high is (K, N) with NZ format
        output_high = torch_npu.npu_quant_matmul(
            x_high_quant,
            layer.weight_high,
            layer.scale_high,
            pertoken_scale=high_act_scale,
            output_dtype=torch.float16,
        )

        output = torch.add(output_low, output_high)

        if bias is not None:
            output = output + bias

        output_shape = list(original_shape[:-1]) + [output.shape[-1]]
        return output.view(output_shape).to(torch.bfloat16)

    def _split_activation(
        self, layer: torch.nn.Module, x_2d: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Split activation into low-precision and high-precision parts.

        For down_proj with perm_rd: per-group split -> block Hadamard
        on each part independently -> flatten to [all_low | all_high].

        For other layers: simple column split at the boundary.

        Returns:
            (x_low, x_high) activation tensors cast to float16.
        """
        # weight_high is (K_high, N) after transpose
        in_high = layer.weight_high.shape[0]

        # perm_rd down_proj: per-group split + block Hadamard
        if self._perm_group_size > 0 and self._rd_block_size > 0:
            return self._split_perm_rd(layer, x_2d, in_high)

        # Default: simple [low | high] column split
        in_low = x_2d.shape[-1] - in_high
        x_low = x_2d[:, :in_low].to(torch.float16).npu()
        x_high = x_2d[:, in_low:].to(torch.float16).npu()
        return x_low, x_high

    def _split_perm_rd(
        self, layer: torch.nn.Module, x_2d: torch.Tensor,
        in_high: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """perm_rd activation split with online block Hadamard.

        1. Reshape activation into per-group view
        2. Split each group into [low | high] channels
        3. Flatten across groups: [grp0_low, grp1_low, ...] etc.
        4. Apply block Hadamard to low and high independently
        """
        block_size = self._rd_block_size
        group_size = self._perm_group_size
        token_count, total_dim = x_2d.shape

        num_groups = total_dim // group_size
        high_per_group = in_high // num_groups
        low_per_group = group_size - high_per_group

        # [M, num_groups, group_size] -> split -> flatten
        x_grouped = x_2d.view(token_count, num_groups, group_size)
        x_low = x_grouped[:, :, :low_per_group].contiguous().reshape(
            token_count, -1)
        x_high = x_grouped[:, :, low_per_group:].contiguous().reshape(
            token_count, -1)

        # Block Hadamard on each part independently
        x_low = _block_hadamard(
            x_low, block_size, layer.h_block).to(torch.float16).npu()
        x_high = _block_hadamard(
            x_high, block_size, layer.h_block).to(torch.float16).npu()

        return x_low, x_high
