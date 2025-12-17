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

from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torch_npu
from vllm.config import CompilationLevel, get_current_vllm_config
from vllm.distributed import get_ep_group
from vllm.forward_context import get_forward_context

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.distributed.parallel_state import get_mc2_group
from vllm_ascend.ops.moe.experts_selector import select_experts
from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ, is_enable_nz

from .w4a4_flatquant_dynamic import pack_int4_weights


def unpack_int32_to_int4_signed(x: torch.Tensor) -> torch.Tensor:
    assert x.dtype == torch.int32
    E, K, N_ = x.shape  # N_ = N/8

    # 取出 8 个 4bit
    out = torch.stack([(x >> (4 * i)) & 0xF for i in range(8)], dim=-1)  # (E,K,M,8)

    # 转成有符号 int4
    out = out.to(torch.int8)
    out = torch.where(out >= 8, out - 16, out)  # [-8,7]

    out = out.reshape(E, K, N_ * 8).to(torch.int8)  # (E, K, N)
    return out


def pack_int4_to_int8_signed(x: torch.Tensor) -> torch.Tensor:
    assert x.dtype == torch.int8
    E, K, N = x.shape
    assert N % 2 == 0
    
    # 转成无符号补码 [0, 15]
    x_unsigned = torch.where(x < 0, x + 16, x).to(torch.int32)

    low = x_unsigned[..., 0::2]   # 偶数 -> 低 4 位
    high = x_unsigned[..., 1::2]  # 奇数 -> 高 4 位

    out = (low | (high << 4)).to(torch.int8)
    return out


def to_nz(weight_i32):
    assert weight_i32.to(torch.float32).abs().sum() > 0
    weight = unpack_int32_to_int4_signed(weight_i32)
    assert weight.to(torch.float32).abs().sum() > 0
    weight_i8 = pack_int4_to_int8_signed(weight)
    assert weight_i8.to(torch.float32).abs().sum() > 0
    weight_nz = torch_npu.npu_format_cast(weight_i8.npu(), 29).view(torch.int32)
    # assert weight_nz.to(torch.float32).abs().sum() > 0
    return weight_nz

def pack_to_int32_moe(weight: torch.Tensor):
    assert -8 <= weight.min()
    assert weight.max() <= 7
    return torch_npu.npu_quantize(weight.to(torch.float32), torch.tensor([1.]).npu(), None, torch.quint4x2, -1, False)

def quantize(
    x: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    x_quantized_int4, activation_scale = torch_npu.npu_dynamic_quant(x, dst_type=torch.quint4x2)
    return x_quantized_int4, activation_scale

def manual_w8a8_dynamic_quant(x: torch.Tensor):
    # x: [num_tokens, hidden_dim]
    # per-token quantization (symmetric)
    # scale: [num_tokens, 1]
    
    # Calculate scale: max(abs(x)) / 127
    scale = x.abs().max(dim=-1, keepdim=True)[0] / 127.0
    # scale = scale.to(x.dtype) # match input dtype (e.g. bf16) -> GMM requires float32 scale
    scale = scale.to(torch.float32)

    # Quantize: x / scale
    # avoid division by zero
    scale_safe = torch.where(scale == 0, torch.ones_like(scale), scale)
    x_quant = (x / scale_safe).round().clamp(-127, 127).to(torch.int8)
    
    return x_quant, scale.squeeze(-1)

def fused_experts(x,
                  w1,
                  w1_scale,
                  w2,
                  w2_scale,
                  topk_weights,
                  topk_ids,
                  global_num_experts
):
    original_dtype = x.dtype    

    expanded_x, expanded_row_idx, expert_token_count, _ = torch_npu.npu_moe_init_routing_v2(
        x,
        topk_ids,
        scale=None,
        active_num=topk_ids.numel(),
        expert_capacity=-1,
        expert_num=global_num_experts,
        drop_pad_mode=0,
        expert_tokens_num_type=1,
        expert_tokens_num_flag=True,
        quant_mode=-1,
        active_expert_range=[0, global_num_experts],
        row_idx_type=0,
    )
    expert_token_count = expert_token_count.to(torch.int64)
    print(f"DEBUG_W4A4: fused_experts start. x shape: {x.shape}, expert_token_count: {expert_token_count.tolist()}")
    print(f"DEBUG_W4A4: global_num_experts: {global_num_experts}, topk_weights shape: {topk_weights.shape}")

    if w1_scale is not None:
        x_quantized, pertoken_scale = quantize(expanded_x)
        expanded_x = torch_npu.npu_grouped_matmul(
            x=[x_quantized],
            weight=[w1],
            scale=[w1_scale],
            bias=None,
            per_token_scale=[pertoken_scale],
            split_item=2,
            group_list_type=1,
            group_type=0,
            group_list=expert_token_count,
            output_dtype=original_dtype)[0]
    else:
        # Float execution for w1
        expanded_x = torch_npu.npu_grouped_matmul(
            x=[expanded_x],
            weight=[w1],
            scale=[],
            bias=None,
            per_token_scale=[],
            split_item=2,
            group_list_type=1,
            group_type=0,
            group_list=expert_token_count,
            output_dtype=original_dtype)[0]

    # act_fn: swiglu
    expanded_x = torch_npu.npu_swiglu(expanded_x)

    w2_is_int8 = False
    if w2.dtype == torch.int8: 
        # Heuristic: if shape matches w8a8 (not packed)
        # Packed w4a4 (int8 storage) usually has halved dimensions. 
        # w2 (2048, 768) -> packed (1024, 384).
        # We can check global_experts or just consistency.
        # But simpler: check if we attached a flag or infer from scale?
        # Let's assume w8a8 if scale is not None and we can't pack it?
        # Actually, simpler: if w2.shape[1] == 768 (full size) vs 384 (packed).
        # intermediate_size/2 = 384. 
        # expert_token_count logic doesn't change weight shape.
        pass

    if w2_scale is not None:
        if w2_scale.dtype == torch.float32 and original_dtype == torch.bfloat16:
            w2_scale = w2_scale.to(original_dtype)
        # Check if w2 is W8A8 (by checking dtype)
        # W8A8 weights are stored as int8, while W4A4 weights are packed into int32
        is_w8a8_w2 = w2.shape[1] == expanded_x.shape[-1]
        
        if is_w8a8_w2:
             # DEBUG: Log before quantization
             print(f"DEBUG_W4A4: Doing W8A8 dynamic quantization for w2. expanded_x shape: {expanded_x.shape}, min: {expanded_x.min()}, max: {expanded_x.max()}")
             x_quantized, pertoken_scale = manual_w8a8_dynamic_quant(expanded_x)
             print(f"DEBUG_W4A4: W8A8 quantized x shape: {x_quantized.shape}, pertoken_scale shape: {pertoken_scale.shape}")
        else:
             print(f"DEBUG_W4A4: Doing W4A4 dynamic quantization for w2. expanded_x shape: {expanded_x.shape}")
             x_quantized, pertoken_scale = quantize(expanded_x)

        expanded_x = torch_npu.npu_grouped_matmul(
            x=[x_quantized],
            weight=[w2],
            scale=[w2_scale],
            bias=None,
            per_token_scale=[pertoken_scale],
            split_item=2,
            group_list_type=1,
            group_type=0,
            group_list=expert_token_count,
            output_dtype=original_dtype)[0]
    else:
        # Float execution for w2
        expanded_x = torch_npu.npu_grouped_matmul(
            x=[expanded_x],
            weight=[w2],
            scale=[],
            bias=None,
            per_token_scale=[],
            split_item=2,
            group_list_type=1,
            group_type=0,
            group_list=expert_token_count,
            output_dtype=original_dtype)[0]

    x = torch_npu.npu_moe_finalize_routing(
        expanded_x,
        skip1=None,
        skip2=None,
        bias=None,
        scales=topk_weights.to(original_dtype),
        expanded_src_to_dst_row=expanded_row_idx,
        export_for_source_row=topk_ids,
        drop_pad_mode=2,
    )
    return x



class AscendW4A4DynamicLinearMethod:
    input_size = 0

    def __init__(self):
        self.sym = True
        vllm_config = get_current_vllm_config()
        ascend_config = get_ascend_config()
        self.use_aclgraph = (
            vllm_config.compilation_config.level == CompilationLevel.PIECEWISE
            and not vllm_config.model_config.enforce_eager
            and not ascend_config.torchair_graph_config.enabled)

    @staticmethod
    def get_weight(input_size: int, output_size: int,
                   params_dtype: torch.dtype) -> Dict[str, Any]:
        if input_size % 8 != 0:
            raise ValueError(
                f"input_size ({input_size}) must be divisible by 8 for int4 packing"
            )
        AscendW4A4DynamicLinearMethod.input_size = input_size
        params_dict = {
            "weight": torch.empty(output_size, input_size, dtype=torch.int8)
        }
        return params_dict

    @staticmethod
    def get_pertensor_param(params_dtype: torch.dtype) -> Dict[str, Any]:
        params_dict = {}
        return params_dict

    @staticmethod
    def get_perchannel_param(
        output_size: int,
        params_dtype: torch.dtype,
    ) -> Dict[str, Any]:
        params_dict = {}
        params_dict["weight_scale"] = torch.empty(output_size,
                                                  1,
                                                  dtype=torch.float32)
        params_dict["weight_offset"] = torch.empty(output_size,
                                                   1,
                                                   dtype=torch.float32)
        return params_dict

    def get_pergroup_param(self,
                           input_size: int,
                           output_size: int,
                           params_dtype: torch.dtype,
                           layer_type: Optional[str] = None) -> Dict[str, Any]:
        return {}

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        tp_rank: Optional[int] = 0,
    ) -> torch.Tensor:
        input_shape = x.shape
        original_dtype = x.dtype
        x_quantized_reshaped, pertoken_scale = quantize(x)
        
        output = torch_npu.npu_quant_matmul(x_quantized_reshaped,
                                            layer.weight_packed.t(),
                                            layer.weight_scale.data,
                                            pertoken_scale=pertoken_scale,
                                            bias=None,
                                            output_dtype=original_dtype)
        output = output.view(*input_shape[:-1], -1)
        if bias is not None:
            output = output + bias.to(original_dtype)
        return output

    def process_weights_after_loading(self, layer):
        weight_packed = pack_int4_weights(layer.weight.data)
        layer.weight_scale.data = layer.weight_scale.data.view(-1).to(torch.float32)

        layer.register_parameter(
            'weight_packed',
            torch.nn.Parameter(weight_packed, requires_grad=False))
        del layer.weight
        layer.weight_offset.data = layer.weight_offset.data.to(torch.float32)


class AscendW4A4DynamicFusedMoEMethod:
    """FusedMoe method for Ascend W4A4_DYNAMIC.
    """

    def __init__(self):
        self.transpose_weight = True
        self.ep_group = get_ep_group()
        vllm_config = get_current_vllm_config()
        self.quant_config = vllm_config.quant_config
        self.group_size = vllm_config.quant_config.quant_description.get("group_size", 0)
        # NOTE: the weights are quantized from bf16 to int4 through a per-channel quantization process
        self.is_per_channel_weight = self.group_size == 0
        # NOTE: new quantize weights: 2 int4 pack into int8
        self.tp_size = 1 if vllm_config.parallel_config.enable_expert_parallel else self.ep_group.world_size
        ascend_config = get_ascend_config()
        self.dynamic_eplb = ascend_config.dynamic_eplb or ascend_config.expert_map_record_path

        try:
            device_group = get_mc2_group().device_group
            # TODO: Try local_rank = ep_group.rank_in_group
            local_rank = torch.distributed.get_rank(group=device_group)
            backend = device_group._get_backend(torch.device("npu"))
            self.moe_all_to_all_group_name = backend.get_hccl_comm_name(
                local_rank)
        except AttributeError:
            self.moe_all_to_all_group_name = ""


    def _detect_quant_types(self):
        # Default assume quantized
        self.w13_is_float = False
        self.w2_is_float = False
        self.w2_is_int8 = False

        # Detect W8A8 for down_proj using global config if specific check fails or as fallback
        if hasattr(self.quant_config, "quant_description"):
            down_proj_type = self.quant_config.quant_description.get(
                "model.layers.*.mlp.experts.*.down_proj.weight")
            if down_proj_type == "W8A8_DYNAMIC":
                self.w2_is_int8 = True

        if hasattr(self, "prefix") and hasattr(self, "packed_modules_mapping") and hasattr(self, "quant_config"):
            proj_name = self.prefix.split(".")[-1]
            if proj_name in self.packed_modules_mapping:
                shard_list = self.packed_modules_mapping[proj_name]
                # Assume last one is w2 (down_proj), others are w13 (gate/up)
                if len(shard_list) >= 2:
                    # Check w13 parts
                    w13_shards = shard_list[:-1] # All except last
                    w13_is_float = True
                    # If any part of w13 is NOT float, we assume whole w13 is NOT float (or handle mixed? usually consistent)
                    for suffix in w13_shards:
                        full_name = self.prefix.replace(proj_name, suffix)
                        quant_type = self.quant_config.quant_description.get(full_name + ".weight")
                        if quant_type != "FLOAT" and quant_type is not None:
                            w13_is_float = False
                            break
                    self.w13_is_float = w13_is_float

                    # Check w2 part
                    w2_suffix = shard_list[-1]
                    w2_full_name = self.prefix.replace(proj_name, w2_suffix)
                    w2_quant_type = self.quant_config.quant_description.get(w2_full_name + ".weight")
                    if w2_quant_type == "FLOAT" or w2_quant_type is None:
                        self.w2_is_float = True
                    elif w2_quant_type == "W8A8_DYNAMIC":
                        self.w2_is_int8 = True
                    elif w2_quant_type == "W8A8_DYNAMIC":
                        self.w2_is_int8 = True
        
        print(f"DEBUG_W4A4: _detect_quant_types result. w13_is_float: {self.w13_is_float}, w2_is_float: {self.w2_is_float}, w2_is_int8: {self.w2_is_int8}")

    def get_weight(self, num_experts: int,
                   intermediate_size_per_partition: int, hidden_sizes: int,
                   params_dtype: torch.dtype) -> Dict[str, Any]:
        param_dict = {}

        self._detect_quant_types()

        # Always use full unpacked sizes for loading
        w13_output_size = 2 * intermediate_size_per_partition
        w2_output_size = hidden_sizes
        w2_input_size = intermediate_size_per_partition

        if self.w13_is_float:
             param_dict["w13_weight"] = torch.empty(num_experts,
                                               w13_output_size,
                                               hidden_sizes,
                                               dtype=params_dtype)
        else:
            param_dict["w13_weight"] = torch.empty(num_experts,
                                                w13_output_size,
                                                hidden_sizes,
                                                dtype=torch.int8)

        if self.w2_is_float:
            param_dict["w2_weight"] = torch.empty(num_experts,
                                                  w2_output_size,
                                                  w2_input_size,
                                                  dtype=params_dtype)
        elif getattr(self, "w2_is_int8", False):
             param_dict["w2_weight"] = torch.empty(num_experts,
                                                   w2_output_size,
                                                   w2_input_size, 
                                                   dtype=torch.int8)
        else:
            param_dict["w2_weight"] = torch.empty(num_experts,
                                                  w2_output_size,
                                                  w2_input_size,
                                                  dtype=torch.int8)
        return param_dict

    def get_dynamic_quant_param(self, num_experts: int,
                                intermediate_size_per_partition: int,
                                hidden_sizes: int,
                                params_dtype: torch.dtype) -> Dict[str, Any]:
        param_dict = {}
        
        # Consistent with get_weight
        w13_output_size = 2 * intermediate_size_per_partition
        w2_output_size = hidden_sizes
        w2_input_size = intermediate_size_per_partition

        # Per-channel quantization scales have last dimension 1
        w13_scale_dim = 1
        w2_scale_dim = 1

        param_dict["w13_weight_scale"] = torch.empty(
            (num_experts, w13_output_size, w13_scale_dim),
            dtype=torch.float32)

        param_dict["w13_weight_offset"] = torch.empty(
            (num_experts, w13_output_size, w13_scale_dim),
            dtype=torch.float32)

        param_dict["w2_weight_scale"] = torch.empty(
            (num_experts, w2_output_size, w2_scale_dim),
            dtype=torch.float32)

        param_dict["w2_weight_offset"] = torch.empty(
            (num_experts, w2_output_size, w2_scale_dim),
            dtype=torch.float32)
        
        if getattr(self, "w2_is_int8", False):
            # Override for W8A8 dynamic shapes
            param_dict["w2_weight_scale"] = torch.empty(
                    num_experts,
                    hidden_sizes,
                    1,
                    dtype=torch.float32)
            param_dict["w2_weight_offset"] = torch.empty(
                num_experts,
                hidden_sizes,
                1,
                dtype=torch.float32)

        if not self.is_per_channel_weight:
            param_dict["w13_weight_scale_second"] = torch.empty(
                num_experts,
                w13_output_size,
                hidden_sizes // self.group_size,
                dtype=torch.float32)
            param_dict["w13_weight_offset_second"] = torch.empty(
                num_experts,
                w13_output_size,
                hidden_sizes // self.group_size,
                dtype=torch.float32)

            param_dict["w2_weight_scale_second"] = torch.empty(
                num_experts,
                w2_output_size,
                w2_input_size // self.group_size,
                dtype=torch.float32)
            param_dict["w2_weight_offset_second"] = torch.empty(
                num_experts,
                w2_output_size,
                w2_input_size // self.group_size,
                dtype=torch.float32)

        if getattr(self, "w13_is_float", False):
            # Remove w13 scale params
             keys_to_remove = [k for k in param_dict.keys() if 'w13_' in k]
             for k in keys_to_remove:
                 del param_dict[k]

        if getattr(self, "w2_is_float", False):
             # Remove w2 scale params
             keys_to_remove = [k for k in param_dict.keys() if 'w2_' in k]
             for k in keys_to_remove:
                 del param_dict[k]

        return param_dict

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        e_score_correction_bias: Optional[torch.Tensor] = None,
        is_prefill: bool = True,
        enable_force_load_balance: bool = True,
        log2phy: torch.Tensor = None,
        global_redundant_expert_num: int = 0,
        shared_experts: Optional[Any] = None,
        quantized_x_for_share: Optional[Any] = None,
        dynamic_scale_for_share: Optional[Any] = None,
        **kwargs,
    ) -> torch.Tensor:
        assert router_logits.shape[
            1] == global_num_experts - global_redundant_expert_num, "Number of global experts mismatch (excluding redundancy)"

        # NOTE: now npu_moe_gating_top_k can only support `group_count=256` pattern
        topk_weights, topk_ids = select_experts(
            hidden_states=x,
            router_logits=router_logits,
            top_k=top_k,
            use_grouped_topk=use_grouped_topk,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            e_score_correction_bias=e_score_correction_bias,
            global_num_experts=global_num_experts)

        # this is a naive implementation for experts load balance so as
        # to avoid accumulating too much tokens on a single rank.
        # currently it is only activated when doing profile runs.
        if enable_force_load_balance:
            topk_ids = torch.randint_like(topk_ids, 0, global_num_experts)

        topk_weights = topk_weights.to(x.dtype)

        # Quantize input for shared experts (and other backend that support it)
        x_quantized, pertoken_scale = quantize(x)

        return fused_experts(x=x,
                             w1=layer.w13_weight,
                             w1_scale=getattr(layer, 'w13_weight_scale', None),
                             w2=layer.w2_weight,
                             w2_scale=getattr(layer, 'w2_weight_scale', None),
                             topk_weights=topk_weights,
                             topk_ids=topk_ids,
                             global_num_experts=global_num_experts)

    def process_scale(self, weight: torch.Tensor, scale, per_group_scale):
        # DEBUG: Log scale info
        print(f"DEBUG_W4A4: process_scale called. Scale shape: {scale.shape}, dtype: {scale.dtype}, min: {scale.min()}, max: {scale.max()}")
        
        scale = scale.transpose(1, 2).contiguous()
        if self.is_per_channel_weight:
            # Match w4a4_flatquant_dynamic_ref.py convert_scales logic EXACTLY
            # convert_scales: 
            # E, N = scales.shape
            # scaleUint32 = scales.cpu().to(torch.float32).clone().numpy().astype(np.float32).reshape(E, 1, N)
            # scaleUint32.dtype = np.uint32
            # scaleUint64 = np.zeros((E, 1, N * 2), dtype=np.uint32)
            # scaleUint64[...,::2] = scaleUint32
            # scaleUint64.dtype = np.int64
            # scale = torch.from_numpy(scaleUint64).npu()

            E, _, N = scale.shape  # scale is (E, 1, Out)
            scale = scale.view(E, N) # Flatten middle dim if it is 1
            
            scale_fp32 = scale.to(torch.float32).cpu()
            scale_fp32_np = scale_fp32.numpy().astype(np.float32)
            
            # View as uint32
            scale_uint32 = scale_fp32_np.view(np.uint32).reshape(E, 1, N)
            
            # Pack into uint64 by interleaving with zeros
            scale_uint64 = np.zeros((E, 1, N * 2), dtype=np.uint32)
            scale_uint64[..., ::2] = scale_uint32
            
            scale_uint64 = scale_uint64.view(np.int64) # Interpret as int64
            scale_uint64_tensor = torch.from_numpy(scale_uint64).npu()
            
            return scale_uint64_tensor, None

        # Handle group quantization
        per_group_scale = per_group_scale.transpose(1, 2).contiguous()
        group_num, k, n = weight.shape
        n = n * 2
        per_group_scale = per_group_scale.reshape(group_num, -1, n)
        group_num, quantgroup_num, n = per_group_scale.shape
        
        bias = None

        scale_fp32 = (scale * per_group_scale).to(torch.float16).to(torch.float32)
        scale_fp32_np = scale_fp32.cpu().numpy()
        scale_fp32_np.dtype = np.uint32
        
        sscale_uint64 = np.zeros((group_num, quantgroup_num, n * 2), dtype=np.uint32)
        sscale_uint64[..., ::2] = scale_fp32_np
        
        sscale_uint64.dtype = np.int64
        sscale_uint64_tensor = torch.from_numpy(sscale_uint64).reshape(
            group_num, quantgroup_num, n)
        return sscale_uint64_tensor.npu(), bias

    def update_bias(self, layer, w13_bias, w2_bias):
        if self.group_size != 0:
            if hasattr(layer, "w13_scale_bias"):
                layer.w13_scale_bias.data = layer.w13_scale_bias.data.transpose(
                    1, 2).contiguous().sum(axis=1)

            if hasattr(layer, "w2_scale_bias"):
                layer.w2_scale_bias.data = layer.w2_scale_bias.data.transpose(
                    1, 2).contiguous().sum(axis=1)
        else:
            if w13_bias is not None:
                w13_scale_bias = torch.nn.Parameter(w13_bias, requires_grad=False)
                layer.register_parameter("w13_scale_bias", w13_scale_bias)
            if w2_bias is not None:
                w2_scale_bias = torch.nn.Parameter(w2_bias, requires_grad=False)
                layer.register_parameter("w2_scale_bias", w2_scale_bias)

    def pack_to_int32(self, weight: torch.Tensor):
        if weight.dim() == 3:
            E, N, K = weight.shape
            weight = weight.reshape(-1, K)
            packed = torch_npu.npu_convert_weight_to_int4pack(weight.to(torch.int32))
            return packed.reshape(E, N, -1)
        return torch_npu.npu_convert_weight_to_int4pack(weight.to(torch.int32))

    def process_weights_after_loading(self, layer):
        print(f"DEBUG_W4A4: process_weights_after_loading start. Layer w2 shape: {layer.w2_weight.shape} dtype: {layer.w2_weight.dtype}")
        if hasattr(layer, "w13_weight"):
             print(f"DEBUG_W4A4: w13 shape: {layer.w13_weight.shape} dtype: {layer.w13_weight.dtype}")
       # Only pack if quantized
        if not getattr(self, "w13_is_float", False):
            layer.w13_weight.data = layer.w13_weight.data.transpose(1, 2).contiguous()
            layer.w13_weight.data = pack_to_int32_moe(layer.w13_weight.data)
            
            w13_weight_scale_second = layer.w13_weight_scale_second.data if hasattr(
                layer, "w13_weight_scale_second") else None
            layer.w13_weight_scale.data, w13_bias = self.process_scale(
                layer.w13_weight, layer.w13_weight_scale.data, w13_weight_scale_second)
        else:
            # Float execution also wants NZ format for w13?
            if is_enable_nz():
                 layer.w13_weight.data = torch_npu.npu_format_cast(
                    layer.w13_weight.data, ACL_FORMAT_FRACTAL_NZ)
            w13_bias = None

        
        # Only pack if quantized
        if not getattr(self, "w2_is_float", False):
            if getattr(self, "w2_is_int8", False):
                 # W8A8 specific processing
                 layer.w2_weight.data = layer.w2_weight.data.transpose(1, 2).contiguous()
                 if is_enable_nz():
                     layer.w2_weight.data = torch_npu.npu_format_cast(
                         layer.w2_weight.data, ACL_FORMAT_FRACTAL_NZ)
                 # Handle Scale
                 layer.w2_weight_scale.data = layer.w2_weight_scale.data.view(
                        layer.w2_weight_scale.data.shape[0], -1)
                 # W8A8 usually wants flat scale or specific shape?
                 # W8A8DynamicFusedMoEMethod uses view(.., -1).
                 # And offsets. 
                 layer.w2_weight_offset.data = layer.w2_weight_offset.data.view(
                        layer.w2_weight_offset.data.shape[0], -1)
                 
                 w2_bias = None 
            else:
                # W4A4 packing
                layer.w2_weight.data = layer.w2_weight.data.transpose(1, 2).contiguous()
                layer.w2_weight.data = pack_to_int32_moe(layer.w2_weight.data)
    
                w2_weight_scale_second = layer.w2_weight_scale_second.data if hasattr(
                    layer, "w2_weight_scale_second") else None
    
                layer.w2_weight_scale.data, w2_bias = self.process_scale(
                    layer.w2_weight, layer.w2_weight_scale.data, w2_weight_scale_second)
        else:
             if hasattr(layer, "w2_weight_scale"):
                del layer.w2_weight_scale
             if is_enable_nz():
                 layer.w2_weight.data = torch_npu.npu_format_cast(
                    layer.w2_weight.data, ACL_FORMAT_FRACTAL_NZ)
             w2_bias = None


        # Cleanup
        if hasattr(layer, "w13_weight_scale_second"):
            del layer.w13_weight_scale_second
        if hasattr(layer, "w2_weight_scale_second"):
            del layer.w2_weight_scale_second
        if hasattr(layer, "w13_weight_offset_second"):
            del layer.w13_weight_offset_second
        if hasattr(layer, "w2_weight_offset_second"):
            del layer.w2_weight_offset_second

        self.update_bias(layer, w13_bias, w2_bias)

        if is_enable_nz():
            if not getattr(self, "w13_is_float", False):
                layer.w13_weight.data = to_nz(layer.w13_weight.data)

            if not getattr(self, "w2_is_float", False) and not getattr(self, "w2_is_int8", False):
                layer.w2_weight.data = to_nz(layer.w2_weight.data)
