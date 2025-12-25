from typing import Any, Dict, Optional

import torch
import torch_npu
from vllm.config import get_current_vllm_config
from vllm.logger import logger
from vllm_ascend.ascend_config import get_ascend_config
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from .w4a4_flatquant_dynamic import pack_int4_weights


# --- Hadamard Utils ---
class HadamardTransform(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        if x.shape[-1] == 1:
            return x
        n = x.shape[-1]
        out = x.clone()
        h = 1
        while h < n:
            out = out.view(x.shape[:-1] + (n // (2 * h), 2, h))
            x1 = out[..., 0, :]
            x2 = out[..., 1, :]
            sum_ = x1 + x2
            diff_ = x1 - x2
            out[..., 0, :] = sum_
            out[..., 1, :] = diff_
            h *= 2
        return out.view(x.shape)

    @staticmethod
    def backward(ctx, grad_output):
        return HadamardTransform.apply(grad_output)


def apply_hadamard_transform(x: torch.Tensor) -> torch.Tensor:
    return HadamardTransform.apply(x)


def is_pow2(n):
    return (n & (n - 1) == 0) and n > 0


def apply_rotation(x: torch.Tensor, rotation_matrix: Optional[torch.Tensor] = None) -> torch.Tensor:
    # 1. Learned Rotation (R3 / U_C)
    if rotation_matrix is not None:
        original_dtype = x.dtype
        # Ensure matrix multiplication matches types (usually FP16/BF16)
        return torch.matmul(x.to(rotation_matrix.dtype), rotation_matrix).to(original_dtype)
    
    # 2. Hadamard Rotation (R4 / U_D) - Parameterless
    n = x.shape[-1]
    scale = 1.0 / (n ** 0.5)
    
    if is_pow2(n):
        return apply_hadamard_transform(x) * scale
    else:
        logger.warning_once(f"ResQ: Dimension {n} is not power of 2. Skipping Hadamard rotation.")
    return x


# Reuse hybrid loader from w4a4_dynamic if needed, or redefine
def hybrid_weight_loader(param, loaded_weight):
    if param.data.shape != loaded_weight.shape:
        param.data = torch.empty_like(loaded_weight)
    default_weight_loader(param, loaded_weight)


class AscendResQW4A4DynamicLinearMethod:
    input_size = 0

    def __init__(self):
        self.sym = True
        vllm_config = get_current_vllm_config()
        ascend_config = get_ascend_config()

    @staticmethod
    def get_weight(input_size: int, output_size: int,
                   params_dtype: torch.dtype) -> Dict[str, Any]:
        AscendResQW4A4DynamicLinearMethod.input_size = input_size
        
        # Hybrid logic: We declare both int4 and int8 placeholders
        # They will be populated/resized by the loader based on the checkpoint keys
        params_dict = {
            "weight_int4": torch.empty(0, dtype=torch.int8),
            "weight_int8": torch.empty(0, dtype=torch.int8),
        }
        for p in params_dict.values():
            if not hasattr(p, "weight_loader"):
                 setattr(p, "weight_loader", hybrid_weight_loader)
        return params_dict

    @staticmethod
    def get_pertensor_param(params_dtype: torch.dtype) -> Dict[str, Any]:
        return {}

    @staticmethod
    def get_perchannel_param(
        output_size: int,
        params_dtype: torch.dtype,
    ) -> Dict[str, Any]:
        params_dict = {}
        # Force BF16 for scales/offsets
        params_dict["weight_scale_int4"] = torch.empty(output_size, 1, dtype=torch.bfloat16)
        params_dict["weight_offset_int4"] = torch.empty(output_size, 1, dtype=torch.bfloat16)
        params_dict["weight_scale_int8"] = torch.empty(output_size, 1, dtype=torch.bfloat16)
        params_dict["weight_offset_int8"] = torch.empty(output_size, 1, dtype=torch.bfloat16)
        
        for p in params_dict.values():
            if not hasattr(p, "weight_loader"):
                 setattr(p, "weight_loader", hybrid_weight_loader)
        return params_dict

    def get_pergroup_param(self, *args, **kwargs) -> Dict[str, Any]:
        return {}

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        tp_rank: Optional[int] = 0,
    ) -> torch.Tensor:
        
        # --- ResQ R4 Rotation ---
        # If this layer is flagged as needing rotation (e.g. down_proj)
        if getattr(layer, 'resq_apply_hadamard', False):
            rot_mat = None
            # Check if R4 is loaded
            if hasattr(layer, 'rotation_R4') and layer.rotation_R4.numel() > 0:
                rot_mat = layer.rotation_R4
            
            x = apply_rotation(x, rotation_matrix=rot_mat)
        # ---------------------------

        input_shape = x.shape
        original_dtype = x.dtype
        
        k_int8 = layer.weight_int8.shape[1]
        x_int8 = x[..., :k_int8]
        x_int4 = x[..., k_int8:]
        
        output_acc = None

        if x_int8.shape[-1] > 0:
             x8_quantized, x8_scale = torch_npu.npu_dynamic_quant(x_int8)
             out8 = torch_npu.npu_quant_matmul(x8_quantized,
                                               layer.weight_int8.t(),
                                               layer.weight_scale_int8.data,
                                               pertoken_scale=x8_scale,
                                               bias=None,
                                               output_dtype=original_dtype)
             output_acc = out8.view(*input_shape[:-1], -1)

        if x_int4.shape[-1] > 0:
             x4_quantized, x4_scale = torch_npu.npu_dynamic_quant(x_int4, dst_type=torch.quint4x2)
             out4 = torch_npu.npu_quant_matmul(x4_quantized,
                                               layer.weight_packed_int4.t(),
                                               layer.weight_scale_int4.data,
                                               pertoken_scale=x4_scale,
                                               bias=None,
                                               output_dtype=original_dtype)
             out4 = out4.view(*input_shape[:-1], -1)
             
             if output_acc is None:
                 output_acc = out4
             else:
                 output_acc += out4

        if bias is not None:
            output_acc = output_acc + bias.to(original_dtype)
            
        return output_acc

    def process_weights_after_loading(self, layer):
        input_dim = 0
        output_dim = 0
        if hasattr(layer, "weight_int8"):
             out_8, in_8 = layer.weight_int8.shape
             out_4, in_4 = layer.weight_int4.shape
             input_dim = in_8 + in_4
             output_dim = out_8

        # Packing logic
        if hasattr(layer, "weight_int4") and layer.weight_int4.numel() > 0:
            weight_packed_int4 = pack_int4_weights(layer.weight_int4.data)
            layer.register_parameter(
                'weight_packed_int4',
                torch.nn.Parameter(weight_packed_int4, requires_grad=False))
            del layer.weight_int4
            
            layer.weight_scale_int4.data = layer.weight_scale_int4.data.view(-1).to(torch.bfloat16)
            if hasattr(layer, "weight_offset_int4"):
                 layer.weight_offset_int4.data = layer.weight_offset_int4.data.to(torch.bfloat16)

        if hasattr(layer, "weight_int8") and layer.weight_int8.numel() > 0:
            layer.weight_scale_int8.data = layer.weight_scale_int8.data.view(-1).to(torch.bfloat16)
            if hasattr(layer, "weight_offset_int8"):
                 layer.weight_offset_int8.data = layer.weight_offset_int8.data.to(torch.bfloat16)
