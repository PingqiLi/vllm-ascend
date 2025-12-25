from typing import Any, Dict, Optional

import torch
# Note: torch_npu and pack_int4_weights are not needed for fake quantization
from vllm.config import get_current_vllm_config
from vllm.logger import logger
from vllm_ascend.ascend_config import get_ascend_config
from vllm.model_executor.model_loader.weight_utils import default_weight_loader


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
        """
        Fake Quantization Forward Pass
        """
        # --- ResQ R4 Rotation ---
        # If this layer is flagged as needing rotation (e.g. down_proj)
        if getattr(layer, 'resq_apply_hadamard', False):
            rot_mat = None
            # Check if R4 is loaded (learned rotation)
            if hasattr(layer, 'rotation_R4') and layer.rotation_R4.numel() > 0:
                rot_mat = layer.rotation_R4
            # Apply rotation (Hadamard if rot_mat is None, else MatMul)
            x = apply_rotation(x, rotation_matrix=rot_mat)
        # ---------------------------

        # Simple bfloat16 matmul with dequantized weight
        # layer.weight shape: [output_dim, input_dim]
        output = torch.matmul(x.to(layer.weight.dtype), layer.weight.t())

        if bias is not None:
            output = output + bias.to(output.dtype)
            
        return output

    def process_weights_after_loading(self, layer):
        """
        Fake Quantization: Dequantize Int4/Int8 weights to bfloat16.
        
        Dequantization formula (asymmetric): 
            float_weight = (int_weight - offset) * scale
        or (symmetric, offset=0):
            float_weight = int_weight * scale
        
        The Int8 channels come first, then Int4 channels (matching ResQ split order).
        """
        dequant_parts = []
        
        # 1. Dequantize Int8 part (high-precision outlier channels)
        if hasattr(layer, "weight_int8") and layer.weight_int8.numel() > 0:
            w_int8 = layer.weight_int8.float()  # [out_dim, in_8]
            s_int8 = layer.weight_scale_int8.data.float()
            # Reshape scale to [out_dim, 1] for broadcasting
            if s_int8.dim() == 1:
                s_int8 = s_int8.view(-1, 1)
            # Handle offset (if asymmetric quantization)
            if hasattr(layer, "weight_offset_int8") and layer.weight_offset_int8.numel() > 0:
                o_int8 = layer.weight_offset_int8.data.float()
                if o_int8.dim() == 1:
                    o_int8 = o_int8.view(-1, 1)
                dequant_int8 = (w_int8 - o_int8) * s_int8
            else:
                dequant_int8 = w_int8 * s_int8
            dequant_parts.append(dequant_int8)
            logger.info(f"Dequantized Int8 weight: shape={dequant_int8.shape}")
        
        # 2. Dequantize Int4 part (majority channels)
        if hasattr(layer, "weight_int4") and layer.weight_int4.numel() > 0:
            w_int4 = layer.weight_int4.float()  # [out_dim, in_4]
            s_int4 = layer.weight_scale_int4.data.float()
            if s_int4.dim() == 1:
                s_int4 = s_int4.view(-1, 1)
            if hasattr(layer, "weight_offset_int4") and layer.weight_offset_int4.numel() > 0:
                o_int4 = layer.weight_offset_int4.data.float()
                if o_int4.dim() == 1:
                    o_int4 = o_int4.view(-1, 1)
                dequant_int4 = (w_int4 - o_int4) * s_int4
            else:
                dequant_int4 = w_int4 * s_int4
            dequant_parts.append(dequant_int4)
            logger.info(f"Dequantized Int4 weight: shape={dequant_int4.shape}")
        
        # 3. Concatenate along input dimension: [out_dim, in_8 + in_4]
        if dequant_parts:
            full_weight = torch.cat(dequant_parts, dim=1).to(torch.bfloat16)
            layer.register_parameter(
                "weight",
                torch.nn.Parameter(full_weight, requires_grad=False)
            )
            logger.info(f"Created dequantized weight: shape={full_weight.shape}, dtype={full_weight.dtype}")
        
        # 4. Clean up quantized parameters (no longer needed)
        attrs_to_delete = [
            "weight_int4", "weight_int8",
            "weight_scale_int4", "weight_scale_int8",
            "weight_offset_int4", "weight_offset_int8",
        ]
        for attr in attrs_to_delete:
            if hasattr(layer, attr):
                delattr(layer, attr)
