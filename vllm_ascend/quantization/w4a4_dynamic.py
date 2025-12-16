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


def fused_experts(x,
                         w1,
                         w1_scale,
                         w2,
                         w2_scale,
                         topk_weights,
                         topk_ids,
                         global_num_experts):
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

    x_quantized, pertoken_scale = quantize(expanded_x)
    print(f"DEBUG: fused_experts x_quantized: {x_quantized.shape}, w1: {w1.shape}, w1_scale: {w1_scale.shape}")

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

    # act_fn: swiglu
    expanded_x = torch_npu.npu_swiglu(expanded_x)

    if w2_scale is not None:
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
        self.group_size = vllm_config.quant_config.quant_description.get("group_size", 0)
        # NOTE: the weights are quantized from bf16 to int4 through a per-channel quantization process
        self.is_per_channel_weight = self.group_size == 0
        quant_version = vllm_config.quant_config.quant_description.get("version", "0")
        # NOTE: new quantize weights: 2 int4 pack into int8
        self.new_quant_version = quant_version == "1.0.0"
        self.tp_size = 1 if vllm_config.parallel_config.enable_expert_parallel else self.ep_group.world_size
        ascend_config = get_ascend_config()
        self.dynamic_eplb = ascend_config.dynamic_eplb or ascend_config.expert_map_record_path
        if self.new_quant_version and self.tp_size > 16:
            raise ValueError(
                "The current weight does not support moe part tp>16.")

        try:
            device_group = get_mc2_group().device_group
            # TODO: Try local_rank = ep_group.rank_in_group
            local_rank = torch.distributed.get_rank(group=device_group)
            backend = device_group._get_backend(torch.device("npu"))
            self.moe_all_to_all_group_name = backend.get_hccl_comm_name(
                local_rank)
        except AttributeError:
            self.moe_all_to_all_group_name = ""

    def get_weight(self, num_experts: int,
                   intermediate_size_per_partition: int, hidden_sizes: int,
                   params_dtype: torch.dtype) -> Dict[str, Any]:
        param_dict = {}
        if self.new_quant_version:
            w13_output_size = intermediate_size_per_partition
            w2_output_size = hidden_sizes // 2
            w2_input_size = intermediate_size_per_partition // 2
        else:
            w13_output_size = 2 * intermediate_size_per_partition
            w2_output_size = hidden_sizes
            w2_input_size = intermediate_size_per_partition

        param_dict["w13_weight"] = torch.empty(num_experts,
                                               w13_output_size,
                                               hidden_sizes,
                                               dtype=torch.int8)
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
        
        if self.new_quant_version:
            w13_output_size = intermediate_size_per_partition
            w2_output_size = hidden_sizes // 2
            w2_input_size = intermediate_size_per_partition // 2
        else:
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

        if self.new_quant_version:
            param_dict["w13_scale_bias"] = torch.empty(
                num_experts,
                w13_output_size,
                1,
                dtype=torch.float32)
            param_dict["w2_scale_bias"] = torch.empty(num_experts,
                                                      w2_output_size,
                                                      16 // self.tp_size,
                                                      dtype=torch.float32)

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
                             w1_scale=layer.w13_weight_scale,
                             w2=layer.w2_weight,
                             w2_scale=getattr(layer, 'w2_weight_scale', None),
                             topk_weights=topk_weights,
                             topk_ids=topk_ids,
                             global_num_experts=global_num_experts)

    def process_scale(self, weight: torch.Tensor, scale, per_group_scale):
        scale = scale.transpose(1, 2).contiguous()
        if self.is_per_channel_weight:
            scale_np = scale.cpu().numpy()
            scale_np.dtype = np.uint32
            scale_uint64_tensor = torch.from_numpy(scale_np.astype(
                np.int64)).npu()
            return scale_uint64_tensor, None

        # Handle group quantization
        per_group_scale = per_group_scale.transpose(1, 2).contiguous()
        group_num, k, n = weight.shape
        if self.new_quant_version:
            n = n * 2
        per_group_scale = per_group_scale.reshape(group_num, -1, n)
        group_num, quantgroup_num, n = per_group_scale.shape
        
        bias = None
        if not self.new_quant_version:
            # Reconstruct high precision weight for bias calculation
            # Note: This part might be computationally expensive
            weight_high = weight.to(torch.float32).reshape([group_num, quantgroup_num, -1, n]) * \
                per_group_scale.reshape([group_num, quantgroup_num, 1, n])
            weight_high = weight_high.reshape([group_num, k, n])
            bias = 8 * (weight_high.to(torch.float32) * scale).sum(axis=1)

        scale_fp32 = (scale * per_group_scale).to(torch.float16).to(torch.float32)
        scale_fp32_np = scale_fp32.cpu().numpy()
        scale_fp32_np.dtype = np.uint32
        
        sscale_uint64 = np.zeros((group_num, quantgroup_num, n * 2), dtype=np.uint32)
        sscale_uint64[..., ::2] = scale_fp32_np
        
        sscale_uint64_buffer = np.frombuffer(sscale_uint64.tobytes(), dtype=np.int64).copy()
        sscale_uint64_tensor = torch.from_numpy(sscale_uint64_buffer).reshape(
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
        print(f"DEBUG: Transposing weights. Pre-transpose w13: {layer.w13_weight.data.shape}")
        layer.w13_weight.data = layer.w13_weight.data.transpose(1, 2).contiguous()
        layer.w2_weight.data = layer.w2_weight.data.transpose(1, 2).contiguous()
        print(f"DEBUG: Post-transpose w13: {layer.w13_weight.data.shape}")

        layer.w13_weight.data = pack_to_int32_moe(layer.w13_weight.data)
        
        is_w2_float = getattr(layer, "is_w2_float", False)
        if not is_w2_float and hasattr(layer, "w2_weight_scale"):
             layer.w2_weight.data = pack_to_int32_moe(layer.w2_weight.data)




        w13_weight_scale_second = layer.w13_weight_scale_second.data if hasattr(
            layer, "w13_weight_scale_second") else None

        layer.w13_weight_scale.data, w13_bias = self.process_scale(
            layer.w13_weight, layer.w13_weight_scale.data, w13_weight_scale_second)


        # is_w2_float has been checked above
        if not is_w2_float:
            w2_weight_scale_second = layer.w2_weight_scale_second.data if hasattr(
                layer, "w2_weight_scale_second") else None
            layer.w2_weight_scale.data, w2_bias = self.process_scale(
                layer.w2_weight, layer.w2_weight_scale.data, w2_weight_scale_second)
        else:
            w2_bias = None
            if hasattr(layer, "w2_weight_scale"):
                print("DEBUG: Deleting w2_weight_scale because is_w2_float is True")
                del layer.w2_weight_scale

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
            layer.w13_weight.data = to_nz(layer.w13_weight.data)
            if not is_w2_float:
                layer.w2_weight.data = to_nz(layer.w2_weight.data)
            else:
                 # Assuming float weights also need NZ format for grouped matmul
                layer.w2_weight.data = torch_npu.npu_format_cast(
                    layer.w2_weight.data, ACL_FORMAT_FRACTAL_NZ)
