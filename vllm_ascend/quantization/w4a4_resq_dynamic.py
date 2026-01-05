from typing import Any, Dict, Optional

import torch
# Note: torch_npu and pack_int4_weights are not needed for fake quantization
from vllm.config import get_current_vllm_config
from vllm.logger import logger
from vllm_ascend.ascend_config import get_ascend_config
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.utils import set_weight_attrs


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


# Custom loader to handle dynamic resizing of split weights
# Must accept optional shard_id for stacked params (qkv_proj, gate_up_proj)
def hybrid_weight_loader(param, loaded_weight, shard_id=None):
    """
    Hybrid weight loader for ResQ that handles:
    1. Dynamic resizing (empty params -> actual shape)
    2. Sharded loading for stacked params (q/k/v, gate/up)
    
    NOTE: We keep everything on CPU during loading to save NPU memory.
    The weights will be moved to NPU during process_weights_after_loading.
    
    shard_id can be:
    - None: Load the full weight (non-stacked params like o_proj, down_proj)
    - "q"/"k"/"v": QKV projection shards  
    - 0/1: gate_proj/up_proj shards
    """
    output_dim = getattr(param, "output_dim", 0)
    
    # Keep loaded_weight on CPU for memory efficiency
    loaded_weight_cpu = loaded_weight.cpu() if loaded_weight.device.type != 'cpu' else loaded_weight
    
    if shard_id is not None:
        # Stacked params: loading a shard into a merged param
        # Need to calculate offset and accumulate shards
        shard_size = loaded_weight_cpu.shape[output_dim]
        
        if param.data.numel() == 0 or param.data.shape[output_dim] == 0:
            # First shard - initialize on CPU
            if isinstance(shard_id, str):  # q/k/v
                # QKV pattern: start with first shard, will concat later
                param.data = loaded_weight_cpu.clone()
            else:  # 0/1 for gate/up
                # gate_up pattern: 2 equal shards, pre-allocate on CPU
                total_size = shard_size * 2
                new_shape = list(loaded_weight_cpu.shape)
                new_shape[output_dim] = total_size
                param.data = torch.zeros(new_shape, dtype=loaded_weight_cpu.dtype, device='cpu')
                # Place first shard at offset
                offset = shard_id * shard_size
                if output_dim == 0:
                    param.data[offset:offset+shard_size] = loaded_weight_cpu
                else:
                    param.data[:, offset:offset+shard_size] = loaded_weight_cpu
        else:
            # Subsequent shard - place at correct offset
            # Ensure param.data is on CPU
            if param.data.device.type != 'cpu':
                param.data = param.data.cpu()
            current_size = param.data.shape[output_dim]
            
            if isinstance(shard_id, str):  # q/k/v
                # Concatenate along output_dim on CPU
                param.data = torch.cat([param.data, loaded_weight_cpu], dim=output_dim)
            else:  # 0/1 for gate/up
                # Place at offset = shard_id * shard_size
                offset = shard_id * shard_size
                if output_dim == 0:
                    if offset + shard_size <= current_size:
                        param.data[offset:offset+shard_size] = loaded_weight_cpu
                    else:
                        # Need to expand
                        new_size = max(current_size, offset + shard_size)
                        new_shape = list(param.data.shape)
                        new_shape[output_dim] = new_size
                        new_data = torch.zeros(new_shape, dtype=param.data.dtype, device='cpu')
                        new_data[:current_size] = param.data
                        new_data[offset:offset+shard_size] = loaded_weight_cpu
                        param.data = new_data
                else:
                    if offset + shard_size <= current_size:
                        param.data[:, offset:offset+shard_size] = loaded_weight_cpu
                    else:
                        new_size = max(current_size, offset + shard_size)
                        new_shape = list(param.data.shape)
                        new_shape[output_dim] = new_size
                        new_data = torch.zeros(new_shape, dtype=param.data.dtype, device='cpu')
                        new_data[:, :current_size] = param.data
                        new_data[:, offset:offset+shard_size] = loaded_weight_cpu
                        param.data = new_data
    else:
        # Non-stacked params: simple resize and copy on CPU
        if param.data.shape != loaded_weight_cpu.shape:
            param.data = loaded_weight_cpu.clone()
        else:
            param.data.copy_(loaded_weight_cpu)


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
        # Set hybrid_weight_loader to handle empty->real shape resize during loading
        for p in params_dict.values():
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
        
        # Set hybrid_weight_loader to handle empty->real shape resize during loading
        for p in params_dict.values():
            setattr(p, "weight_loader", hybrid_weight_loader)
        return params_dict

    def get_pergroup_param(self, *args, **kwargs) -> Dict[str, Any]:
        return {}

    # NOTE: create_weights is NOT implemented here because AscendLinearMethod 
    # has its own create_weights that calls our get_weight/get_perchannel_param methods.
    # Our hybrid_weight_loader is preserved via AscendLinearMethod's original_weight_loader logic.

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
        # NOTE: All operations on CPU to avoid NPU OOM
        if hasattr(layer, "weight_int8") and layer.weight_int8.numel() > 0:
            w_int8 = layer.weight_int8.cpu().float()  # Move to CPU first
            s_int8 = layer.weight_scale_int8.data.cpu().float()
            logger.debug(f"Int8 shapes: weight={w_int8.shape}, scale={s_int8.shape}")
            
            # Reshape scale to [out_dim, 1] for broadcasting
            if s_int8.dim() == 1:
                s_int8 = s_int8.view(-1, 1)
            # Ensure scale matches weight's output dim
            if s_int8.shape[0] != w_int8.shape[0]:
                logger.warning(f"Scale shape mismatch: scale={s_int8.shape}, weight={w_int8.shape}. Truncating or broadcasting.")
                s_int8 = s_int8[:w_int8.shape[0]]
                
            # Handle offset (if asymmetric quantization)
            if hasattr(layer, "weight_offset_int8") and layer.weight_offset_int8.numel() > 0:
                o_int8 = layer.weight_offset_int8.data.cpu().float()
                logger.debug(f"Int8 offset shape: {o_int8.shape}")
                if o_int8.dim() == 1:
                    o_int8 = o_int8.view(-1, 1)
                # Ensure offset matches weight's output dim
                if o_int8.shape[0] != w_int8.shape[0]:
                    logger.warning(f"Offset shape mismatch: offset={o_int8.shape}, weight={w_int8.shape}. Truncating or broadcasting.")
                    o_int8 = o_int8[:w_int8.shape[0]]
                dequant_int8 = (w_int8 - o_int8) * s_int8
            else:
                dequant_int8 = w_int8 * s_int8
            dequant_parts.append(dequant_int8)
            logger.info(f"Dequantized Int8 weight: shape={dequant_int8.shape}")
        
        # 2. Dequantize Int4 part (majority channels)
        # NOTE: All operations on CPU to avoid NPU OOM
        if hasattr(layer, "weight_int4") and layer.weight_int4.numel() > 0:
            w_int4 = layer.weight_int4.cpu().float()  # Move to CPU first
            s_int4 = layer.weight_scale_int4.data.cpu().float()
            logger.debug(f"Int4 shapes: weight={w_int4.shape}, scale={s_int4.shape}")
            
            if s_int4.dim() == 1:
                s_int4 = s_int4.view(-1, 1)
            # Ensure scale matches weight's output dim
            if s_int4.shape[0] != w_int4.shape[0]:
                logger.warning(f"Int4 Scale shape mismatch: scale={s_int4.shape}, weight={w_int4.shape}. Truncating.")
                s_int4 = s_int4[:w_int4.shape[0]]
                
            if hasattr(layer, "weight_offset_int4") and layer.weight_offset_int4.numel() > 0:
                o_int4 = layer.weight_offset_int4.data.cpu().float()
                logger.debug(f"Int4 offset shape: {o_int4.shape}")
                if o_int4.dim() == 1:
                    o_int4 = o_int4.view(-1, 1)
                # Ensure offset matches weight's output dim
                if o_int4.shape[0] != w_int4.shape[0]:
                    logger.warning(f"Int4 Offset shape mismatch: offset={o_int4.shape}, weight={w_int4.shape}. Truncating.")
                    o_int4 = o_int4[:w_int4.shape[0]]
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
