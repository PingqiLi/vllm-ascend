#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
"""ResQ mixed-precision quantization linear method."""

import math
from typing import Any, Dict, List, Optional

import torch
import torch_npu
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
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
        self.is_o_proj = "o_proj" in prefix

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

        if self.is_down_proj:
            register_resq_param("rotation_Pd", torch.float32)
            register_resq_param("rotation_Hd", torch.float32)
            h_butterfly = torch.nn.Parameter(
                torch.empty(0, dtype=torch.float32), requires_grad=False
            )
            layer.register_parameter("h_butterfly", h_butterfly)
            set_weight_attrs(h_butterfly, extra_weight_attrs)

        if self.is_o_proj:
            # Buffer for column reorder indices (computed in process_weights_after_loading)
            layer.register_buffer("o_proj_column_order", torch.empty(0, dtype=torch.long))

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

        # Compute o_proj column reorder indices based on high_fraction
        # Always setup for o_proj - use high_fraction from checkpoint or default 0.125
        if self.is_o_proj:
            self._setup_o_proj_column_order(layer)

        # Process weight_low (int4): use NPU native packing
        # weight_low shape: (n, k_low) -> pack to (n, k_low//8)
        w_low = layer.weight_low.data.to(torch.int32).npu()
        w_low_packed = torch_npu.npu_convert_weight_to_int4pack(w_low)
        layer.register_buffer("weight_low_packed", w_low_packed)

        # Process weight_high (int8): transpose to (k_high, n) + NZ format
        w_high = layer.weight_high.data.transpose(0, 1).contiguous().npu()
        if is_enable_nz():
            w_high = torch_npu.npu_format_cast(w_high, ACL_FORMAT_FRACTAL_NZ)
        layer.weight_high.data = w_high

        # Process scales: flatten to 1D float32
        layer.scale_low.data = layer.scale_low.data.flatten().to(torch.float32).npu()
        layer.scale_high.data = layer.scale_high.data.flatten().to(torch.float32).npu()

        # TP
        if self.prefix.endswith("qkv_proj") or self.prefix.endswith("gate_up_proj"):
            assert layer.weight_low_packed.data.shape[0] == layer.scale_low.data.shape[0]
            assert layer.weight_high.data.shape[1] == layer.scale_high.data.shape[0]
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
                weight_high.append(layer.weight_high.data[:, begin: end])
                scale_low.append(layer.scale_low.data[begin: end])
                scale_high.append(layer.scale_high.data[begin: end])

            layer.weight_low_packed.data = torch.cat(weight_low_packed)
            layer.weight_high.data = torch.cat(weight_high, dim=-1)
            layer.scale_low.data = torch.cat(scale_low)
            layer.scale_high.data = torch.cat(scale_high)
            
        elif self.prefix.endswith("down_proj"):
            PACK_COUNT = 8
            chunk_size = (self.l_dim + self.h_dim) // self.tp_size
            assert chunk_size % PACK_COUNT == 0
            begin = chunk_size * self.tp_rank
            end = chunk_size * (self.tp_rank + 1)

            if begin < self.l_dim:
                layer.weight_low_packed.data = layer.weight_low_packed.data[:, begin // PACK_COUNT: min(end // PACK_COUNT, layer.weight_low_packed.data.shape[1])].clone()
            else:
                layer.weight_low_packed.data = torch.empty(0, 0)
            
            if end - self.l_dim > 0:
                layer.weight_high.data = layer.weight_high.data[max(0, begin - self.l_dim): end - self.l_dim, :].clone()
            else:
                layer.weight_high.data = torch.empty(0, 0)

        elif self.prefix.endswith("o_proj"):
            PACK_COUNT = 8
            assert self.l_dim % (self.tp_size * PACK_COUNT) == 0
            assert self.h_dim % self.tp_size == 0
            chunk_size_l = self.l_dim // (self.tp_size * PACK_COUNT) 
            chunk_size_h = self.h_dim // self.tp_size
            begin_l = chunk_size_l * self.tp_rank 
            end_l = chunk_size_l * (self.tp_rank + 1)
            begin_h = chunk_size_h * self.tp_rank
            end_h = chunk_size_h * (self.tp_rank + 1)

            layer.weight_low_packed.data = layer.weight_low_packed.data[:, begin_l: end_l].clone()
            layer.weight_high.data = layer.weight_high.data[begin_h: end_h, :].clone()

    def _get_new_column_order(self, in_dim, head_dim, high_per_head, device):
        # Compute column reorder indices
        # Original layout: [head0_cols, head1_cols, ..., headN_cols]
        # Each head has head_dim columns
        # We move the last high_per_head columns of each head to the end

        chunk_starts = torch.arange(0, in_dim, head_dim, device=device)
        high_precision_columns = torch.arange(head_dim - high_per_head, head_dim, device=device)

        # Columns to move to end (high precision columns from each head)
        columns_to_end = (chunk_starts.unsqueeze(1) + high_precision_columns).flatten()

        # All columns
        all_columns = torch.arange(in_dim, device=device)

        # Remaining columns (mid precision)
        mask = torch.ones(in_dim, dtype=torch.bool, device=device)
        mask[columns_to_end] = False
        remaining_columns = all_columns[mask]

        # New order: [remaining (mid) | high]
        new_column_order = torch.cat([remaining_columns, columns_to_end])

        return new_column_order

    def _setup_o_proj_column_order(self, layer: torch.nn.Module) -> None:
        """Setup o_proj input column reorder indices based on high_fraction.

        The ResQ quantization reorders o_proj weights to [mid | high] layout.
        We need to reorder the input (attn_output) columns to match this layout.

        The reordering moves high-precision columns (last `high_per_head` columns
        of each head) to the end of the tensor.
        """
        from vllm.logger import logger

        # Get high_fraction value - use checkpoint value if available, else default 0.125
        # Reference implementation hardcodes 0.125
        DEFAULT_HIGH_FRACTION = 0.125
        if hasattr(layer, "high_fraction") and layer.high_fraction.numel() > 0:
            high_fraction = layer.high_fraction.item()
        else:
            high_fraction = DEFAULT_HIGH_FRACTION

        # Get input dimension from weight shape
        # weight_low shape: (out_features, in_low), weight_high shape: (out_features, in_high)
        # Note: weight_high has NOT been transposed yet at this point
        in_low = layer.weight_low.shape[1] if layer.weight_low.numel() > 0 else 0
        in_high = layer.weight_high.shape[1] if layer.weight_high.numel() > 0 else 0
        in_dim = in_low + in_high

        if in_dim == 0:
            logger.warning(f"[ResQ] o_proj {self.prefix}: in_dim is 0, skipping column reorder setup")
            return


        # For o_proj, we need to know head_dim and num_heads
        # We can infer this from the weight dimensions and high_fraction
        # num_heads * head_dim = in_dim
        # Assuming head_dim is typically 128 for Qwen3
        head_dim = 128  # TODO: Get this from config if needed
        num_heads = in_dim // head_dim
        if num_heads == 0:
            logger.warning(f"[ResQ] o_proj {self.prefix}: num_heads is 0, skipping column reorder setup")
            return
        high_per_head = int(head_dim * high_fraction)

        device = layer.weight_low.device if layer.weight_low.numel() > 0 else "cpu"

        layer.o_proj_column_order = self._get_new_column_order(in_dim // self.tp_size, head_dim, high_per_head, device)

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

        # Apply column reordering for o_proj input
        if self.is_o_proj and layer.o_proj_column_order.numel() > 0:
            x = x[..., layer.o_proj_column_order]

        # Apply Ud rotation for down_proj
        if self.is_down_proj and layer.rotation_Pd.numel() > 0:
            if self.tp_size > 1:
                raise NotImplementedError()
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

        # weight_high is now (k_high, n) after preprocessing
        in_high = layer.weight_high.shape[0]
        in_low = x_2d.shape[-1] - in_high

        x_low = x_2d[:, :in_low].to(torch.float16).npu()
        x_high = x_2d[:, in_low:].to(torch.float16).npu()

        x_low_quant, lx_scale = torch_npu.npu_dynamic_quant(
            x_low.contiguous(), dst_type=torch.quint4x2
        )
        x_high_quant, rx_scale = torch_npu.npu_dynamic_quant(
            x_high, dst_type=torch.int8
        )

        output_low = 0
        output_high = 0
        
        if layer.weight_low_packed.numel():
            # int4 matmul: weight_low_packed (n, k_low//8) -> transpose to (k_low//8, n)
            output_low = torch_npu.npu_quant_matmul(
                x_low_quant,
                layer.weight_low_packed.t(),
                layer.scale_low,
                pertoken_scale=lx_scale,
                output_dtype=torch.float16,
            )

        if layer.weight_high.numel():
            # int8 matmul: weight_high already (k_high, n) with NZ format
            output_high = torch_npu.npu_quant_matmul(
                x_high_quant,
                layer.weight_high,
                layer.scale_high,
                pertoken_scale=rx_scale,
                output_dtype=torch.float16,
            )

        output = torch.add(output_low, output_high)

        # TODO: this could be an issue if bias is not None, as so far 
        # we do not consider how and if bias should be reordered during quantization
        if bias is not None:
            raise NotImplementedError()
            output = output + bias

        output_shape = list(original_shape[:-1]) + [output.shape[-1]]
        return output.view(output_shape).to(torch.bfloat16)
