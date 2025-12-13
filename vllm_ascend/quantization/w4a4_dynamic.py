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


def quantize(
    x: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    orig_shape = x.shape
    x2 = x.reshape(-1, x.shape[-1])
    qscale = torch.abs(x2).max(dim=-1, keepdim=True)[0].to(torch.float)
    ratio = torch.ones_like(qscale) * 7
    qscale2 = ratio / qscale
    x_quantized_int4 = \
        torch_npu.npu_quantize(x2, qscale2.to(x2.dtype), zero_points=None, dtype=torch.quint4x2, axis=-2, div_mode=False)
    activation_scale = torch.flatten(qscale / ratio)

    if len(x_quantized_int4.shape) != 2 and x_quantized_int4.shape[1] != orig_shape[-1] // 8:
        x_quantized_int4 = x_quantized_int4.reshape(-1, orig_shape[-1] // 8)
    return x_quantized_int4, activation_scale



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
        self.group_size = vllm_config.quant_config.quant_description.get("group_size", 256)
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
        else:
            w13_output_size = 2 * intermediate_size_per_partition
            w2_output_size = hidden_sizes

        param_dict["w13_weight"] = torch.empty(num_experts,
                                               w13_output_size,
                                               hidden_sizes,
                                               dtype=torch.int8)
        param_dict["w2_weight"] = torch.empty(num_experts,
                                              w2_output_size,
                                              intermediate_size_per_partition,
                                              dtype=torch.int8)
        return param_dict

    def get_dynamic_quant_param(self, num_experts: int,
                                intermediate_size_per_partition: int,
                                hidden_sizes: int,
                                params_dtype: torch.dtype) -> Dict[str, Any]:
        param_dict = {}
        # Per-channel quantization scales have last dimension 1
        w13_scale_dim = 1
        w2_scale_dim = 1

        param_dict["w13_weight_scale"] = torch.empty(
            (num_experts, 2 * intermediate_size_per_partition, w13_scale_dim),
            dtype=torch.float32)

        param_dict["w13_weight_offset"] = torch.empty(
            (num_experts, 2 * intermediate_size_per_partition, w13_scale_dim),
            dtype=torch.float32)

        param_dict["w2_weight_scale"] = torch.empty(
            (num_experts, hidden_sizes, w2_scale_dim),
            dtype=torch.float32)

        param_dict["w2_weight_offset"] = torch.empty(
            (num_experts, hidden_sizes, w2_scale_dim),
            dtype=torch.float32)

        if not self.is_per_channel_weight:
            param_dict["w13_weight_scale_second"] = torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_sizes // self.group_size,
                dtype=torch.float32)
            param_dict["w13_weight_offset_second"] = torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_sizes // self.group_size,
                dtype=torch.float32)

            param_dict["w2_weight_scale_second"] = torch.empty(
                num_experts,
                hidden_sizes,
                intermediate_size_per_partition // self.group_size,
                dtype=torch.float32)
            param_dict["w2_weight_offset_second"] = torch.empty(
                num_experts,
                hidden_sizes,
                intermediate_size_per_partition // self.group_size,
                dtype=torch.float32)

        if self.new_quant_version:
            param_dict["w13_scale_bias"] = torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                1,
                dtype=torch.float32)
            param_dict["w2_scale_bias"] = torch.empty(num_experts,
                                                      hidden_sizes,
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

        moe_comm_method = get_forward_context().moe_comm_method
        return moe_comm_method.fused_experts(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            w1_scale_bias=getattr(layer, 'w13_scale_bias', None),
            w2_scale_bias=getattr(layer, 'w2_scale_bias', None),
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            use_int4_w4a8=True,
            expert_map=expert_map,
            log2phy=log2phy,
            global_redundant_expert_num=global_redundant_expert_num,
            shared_experts=shared_experts,
            quantized_x_for_share=x_quantized,
            dynamic_scale_for_share=pertoken_scale,
            dynamic_eplb=self.dynamic_eplb)

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
        if self.new_quant_version:
            layer.w13_scale_bias.data = layer.w13_scale_bias.data.transpose(
                1, 2).contiguous().sum(axis=1)
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
        if self.new_quant_version:
            # pack 4 int8(int4*2) to int32, because in pytorch, we need to use int32 to represent int4
            assert weight.shape[
                -1] % 4 == 0, "the last dim of weight needs to be divided by 4"
            return weight.view(torch.int32).contiguous()
        else:
            return torch_npu.npu_quantize(weight.to(torch.float32),
                                          torch.tensor([1.]).npu(), None,
                                          torch.quint4x2, -1, False)

    def process_weights_after_loading(self, layer):
        if self.transpose_weight:
            layer.w13_weight.data = layer.w13_weight.data.transpose(
                1, 2).contiguous()
            layer.w2_weight.data = layer.w2_weight.data.transpose(
                1, 2).contiguous()

        w13_weight_scale_second = layer.w13_weight_scale_second.data if hasattr(
            layer, "w13_weight_scale_second") else None
        w2_weight_scale_second = layer.w2_weight_scale_second.data if hasattr(
            layer, "w2_weight_scale_second") else None

        layer.w13_weight_scale.data, w13_bias = self.process_scale(
            layer.w13_weight, layer.w13_weight_scale.data, w13_weight_scale_second)
            
        layer.w2_weight_scale.data, w2_bias = self.process_scale(
            layer.w2_weight, layer.w2_weight_scale.data, w2_weight_scale_second)

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
            layer.w13_weight.data = torch_npu.npu_format_cast(
                layer.w13_weight.data, ACL_FORMAT_FRACTAL_NZ)
            layer.w2_weight.data = torch_npu.npu_format_cast(
                layer.w2_weight.data, ACL_FORMAT_FRACTAL_NZ)
        layer.w13_weight.data = self.pack_to_int32(layer.w13_weight.data)
        layer.w2_weight.data = self.pack_to_int32(layer.w2_weight.data)
