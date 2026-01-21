from typing import Any, Dict, List, Optional

import torch
import torch_npu
from vllm.model_executor.layers.linear import LinearMethodBase
from vllm.model_executor.utils import set_weight_attrs

from vllm_ascend.quantization.resq_utils import (
    apply_ud_rotation,
    pack_int4_to_int8_signed,
    hadamard_transform
)


import os

def resq_log(msg: str):
    print(msg, flush=True)
    log_file = os.environ.get("RESQ_LOG_FILE")
    if log_file:
        try:
            with open(log_file, "a") as f:
                f.write(msg + "\n")
        except Exception as e:
            print(f"WARN: Failed to write to {log_file}: {e}")

class ResQLinearMethod(LinearMethodBase):
    """Linear method for ResQ mixed-precision quantization.
    
    Supports:
    - Mixed precision weights: weight_low (int4 as int8), weight_high (int8)
    - Scales: scale_low, scale_high
    - MLP Ud Rotation: Pd, Hd (for down_proj)
    """

    def __init__(self, quant_config, prefix: str, packed_modules_mapping: Dict[str, Any]):
        self.quant_config = quant_config
        self.prefix = prefix
        self.packed_modules_mapping = packed_modules_mapping
        self.is_down_proj = "down_proj" in prefix
        if self.is_down_proj:
            resq_log(f"DEBUG [ResQ] {prefix}")


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
        output_size_per_partition = sum(output_partition_sizes)
        
        # We replace the standard weight with our mixed precision parameters
        if hasattr(layer, "weight"):
            del layer.weight

        # Helpers to register parameter with custom weight_loader
        def register_resq_param(name, dtype=torch.float32):
            # Initialize as empty 1D tensor to allow any shape loading
            param = torch.nn.Parameter(torch.empty(0, dtype=dtype), requires_grad=False)
            layer.register_parameter(name, param)
            set_weight_attrs(param, extra_weight_attrs)
            # Bind our custom weight_loader to this parameter
            setattr(param, "weight_loader", self.weight_loader)

        # 1. Weight High (int8)
        register_resq_param("weight_high", torch.int8)
        
        # 2. Weight Low (int8 storage, will be packed later or during forward)
        register_resq_param("weight_low", torch.int8)
        
        # 3. Scales
        register_resq_param("scale_high", torch.float32)
        register_resq_param("scale_low", torch.float32)
        
        # 4. Rotation buffers (only for down_proj)
        if self.is_down_proj:
            register_resq_param("rotation_Pd", torch.float32)
            register_resq_param("rotation_Hd", torch.float32)
            # Buffer for pre-computed butterfly matrix (not loaded from checkpoint)
            h_butterfly = torch.nn.Parameter(torch.empty(0, dtype=torch.float32), requires_grad=False)
            layer.register_parameter("h_butterfly", h_butterfly)
            set_weight_attrs(h_butterfly, extra_weight_attrs)

    def weight_loader(self, param: torch.nn.Parameter, loaded_weight: torch.Tensor, shard_id: Optional[int] = None, **kwargs):
        """Custom weight loader for ResQ parameters."""
        # For merged layers (QKV, GateUp), weights are loaded in shards.
        # We buffer them and assemble in process_weights_after_loading.
        if shard_id is None:
            shard_id = 0
            
        if not hasattr(param, "_shards"):
            param._shards = {}
        
        # Store shard on CPU/NPU as loaded (usually CPU during load)
        param._shards[shard_id] = loaded_weight

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Step 0: Assemble sharded parameters (for QKV/GateUp)
        for name in ["weight_low", "weight_high", "scale_low", "scale_high", "rotation_Pd", "rotation_Hd"]:
             if hasattr(layer, name):
                 param = getattr(layer, name)
                 if hasattr(param, "_shards"):
                    # Custom sort for QKV strings if present
                    def shard_sort_key(k):
                        if isinstance(k, str):
                            k_lower = k.lower()
                            if 'q' in k_lower: return 0
                            if 'k' in k_lower: return 1
                            if 'v' in k_lower: return 2
                        return k

                    # Sort by shard_id to ensure correct order (Q, K, V)
                    # Use shard_sort_key to handle ['k', 'q', 'v'] case -> [0, 1, 2] -> Q, K, V
                    sorted_keys = sorted(param._shards.keys(), key=shard_sort_key)
                    sorted_shards = [param._shards[k] for k in sorted_keys]
                    
                    if "gate_up_proj" in self.prefix and name == "weight_low":
                        pass
                        # Debugging removed for clarity

                    if "qkv_proj" in self.prefix and name == "weight_low":
                        resq_log(f"DEBUG [ResQ] Assembly {self.prefix} {name}: Shards found: {list(param._shards.keys())}")
                        resq_log(f"DEBUG [ResQ] Assembly order: {sorted_keys}")
                    
                    if "gate_up_proj" in self.prefix and name == "weight_low":
                        resq_log(f"DEBUG [ResQ] Assembly {self.prefix} {name}: Shards found: {list(param._shards.keys())}")
                        resq_log(f"DEBUG [ResQ] Assembly order: {sorted_keys}")


                    # Concatenate along output dimension (dim 0)
                    # Ensure all shards are on same device/dtype (should be)
                    if len(sorted_shards) > 0:
                        full_weight = torch.cat(sorted_shards, dim=0)
                        # Move to correct device if needed, but usually kept on load device until inference
                        param.data = full_weight.to(param.device, dtype=param.dtype)
                    
                    # Cleanup
                    del param._shards

        # Optimization 1: Pre-compute h_butterfly if Pd/Hd are present
        if self.is_down_proj and hasattr(layer, "rotation_Pd") and layer.rotation_Pd.numel() > 0:
            blocksize = layer.rotation_Pd.shape[0] if len(layer.rotation_Pd.shape) > 0 else 256
            if layer.h_butterfly.numel() == 0 and blocksize > 0:
                 # Create identity matrix and apply transform to get the matrix form
                eye = torch.eye(blocksize, dtype=torch.float32)
                try:
                    h_matrix = hadamard_transform(eye)
                    layer.h_butterfly.data = h_matrix.to(layer.rotation_Pd.device)
                except Exception as e:
                    print(f"Failed to compute butterfly matrix: {e}")

        # Optimization 2: Pack weight_low (int8 [-8, 7]) into int4x2 format (stored as int32)
        # Check if weight_low is populated (loaded)
        if hasattr(layer, "weight_low") and layer.weight_low.numel() > 0:
            # pack_int4_to_int8_signed: packs 2 int4 into 1 int8.
            # view(int32): packs 4 int8 into 1 int32. Total 8 int4 per int32.
            w_low_packed = pack_int4_to_int8_signed(layer.weight_low)
            
            # Ensure tensor is on NPU before view/transpose if needed?
            # Usually keep on device.
            device = layer.weight_low.device
            w_low_packed = w_low_packed.to(device)
            # Ensure contiguous before view to avoid layout issues
            # WARNING: transpose(-1, -2) makes it non-contiguous again unless we call contiguous() after!
            # The previous code: contiguous().view(...).transpose(...) -> transpose result is NON-contiguous.
            # Correct flow for storage (if NPU expects contiguous buffer):
            # But wait, npu_quant_matmul typically expects KxN weight? 
            # Ref script: weight_low = weight_low.view(torch.int32).transpose(-1, -2).npu()
            # If we just buffer it, we should store it in the final format.
            # Let's ensure the final registered buffer is contiguous if that helps.
            # REVERT: Ref script DOES NOT use contiguous(). Adding it might change stride info that NPU op relies on.
            # UPDATE: Store as [N, K_packed] (Contiguous). 
            # We will transpose it in `apply` to get [K_packed, N] with [1, K] strides.
            # This ensures vLLM model moving doesn't mess up strides.
            w_low_packed = w_low_packed.view(torch.int32).contiguous()
            
            
            # Register buffer for packed weight
            layer.register_buffer("weight_low_packed", w_low_packed)


    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        tp_rank: int = 0, 
        **kwargs,
    ) -> torch.Tensor:
        
        should_log = ("layers.0." in self.prefix) and (self.is_down_proj or "qkv" in self.prefix)
        
        if should_log:
             resq_log(f"DEBUG [ResQ] {self.prefix} INPUT 'x' stats: min={x.min()}, max={x.max()}, mean={x.float().mean()}, std={x.float().std()}")
             resq_log(f"DEBUG [ResQ] {self.prefix} weight_low stats: mean={layer.weight_low.float().mean()}, dtype={layer.weight_low.dtype}")
             if hasattr(layer, 'scale_low'):
                 resq_log(f"DEBUG [ResQ] {self.prefix} scale_low stats: min={layer.scale_low.min()}, max={layer.scale_low.max()}, mean={layer.scale_low.mean()}")
             if hasattr(layer, 'scale_high'):
                 resq_log(f"DEBUG [ResQ] {self.prefix} scale_high stats: min={layer.scale_high.min()}, max={layer.scale_high.max()}, mean={layer.scale_high.mean()}")



        # CRITICAL FIX 1: Enforce input contiguousness at the start
        if not x.is_contiguous():
            x = x.contiguous()

        if self.is_down_proj and layer.rotation_Pd.numel() > 0:
            # We assume Hd is loaded or shared.
            # K is derived from Hd shape
            Hd = layer.rotation_Hd if layer.rotation_Hd.numel() > 0 else None
            K = Hd.shape[0] if Hd is not None else 1
            # Intermediate size is x.shape[-1]
            blocksize = layer.rotation_Pd.shape[0]
            
            # Ensure input is contiguous before rotation (critical for NPU custom kernels/views)
            x = x.contiguous()
            
            x = apply_ud_rotation(
                x,
                Pd=layer.rotation_Pd,
                Hd=Hd,
                h_butterfly=layer.h_butterfly if layer.h_butterfly.numel() > 0 else None,
                K=K,
                blocksize=blocksize
            )
            
        # 2. ResQ Mixed Precision Matmul
        original_shape = x.shape
        x_2d = x.contiguous().view(-1, x.shape[-1]).float()
        
        # Determine split size from weight shapes
        if layer.weight_high.numel() == 0:
            return x # Should not happen if loaded
            
        in_high = layer.weight_high.shape[1]
        in_low = x_2d.shape[-1] - in_high
        
        # DEBUG: Verify split
        # print(f"DEBUG [ResQ] {self.prefix} Split: x_dim={x_2d.shape[-1]}, in_low={in_low}, in_high={in_high}, weight_low.shape={layer.weight_low.shape}")
        
        # Split input
        # Note: input is float/bf16.
        # Use [:in_low] to match Ref exactly
        x_low = x_2d[:, :in_low].to(torch.float16).npu()
        x_high = x_2d[:, in_low:].to(torch.float16).npu()
        
        # if "gate_up_proj" in self.prefix:
        #      print(f"DEBUG [ResQ] gate_up_proj shape: {layer.weight_low.shape}")
        
        # Ensure x_low is contiguous for kernel safety
        x_low = x_low.contiguous()
        x_low_abs_max, lxScale = torch_npu.npu_dynamic_quant(x_low, dst_type=torch.quint4x2)
        x_high_abs_max, rxScale = torch_npu.npu_dynamic_quant(x_high, dst_type=torch.int8)
        
        # Revert to pre-packed buffer for performance
        # w_low_packed is stored as [N, K_packed]. 
        # But npu_quant_matmul logic apparently infers from Transposed view?
        # Ref: weight_low.view(int32).transpose(-1, -2).npu()
        # If Ref works, we need to match Ref's final state.
        # If Ref sends [560, 10240] with strides [1, 560].
        # We need to recreate that here.
        w_low_packed = layer.weight_low_packed.transpose(-1, -2)
        
        if should_log:
            resq_log(f"DEBUG [ResQ] {self.prefix} MatMul Shapes:")
            resq_log(f"  x_low: {x_low.shape} {x_low.dtype}")
            resq_log(f"  x_low_abs_max: {x_low_abs_max.shape} {x_low_abs_max.dtype}")
            resq_log(f"  w_low_packed: {w_low_packed.shape} {w_low_packed.dtype}")
            resq_log(f"  lxScale: {lxScale.shape}")
        
        output_low = torch_npu.npu_quant_matmul(
            x_low_abs_max, 
            w_low_packed, 
            layer.scale_low.to(torch.float).reshape(-1).npu(), 
            pertoken_scale=lxScale, 
            output_dtype=torch.float16
        )

        w_high = torch_npu.npu_format_cast(layer.weight_high.npu(), 29).transpose(-1,-2)
        output_high = torch_npu.npu_quant_matmul(
            x_high_abs_max, 
            w_high, 
            layer.scale_high.to(torch.float).reshape(-1).npu(), 
            pertoken_scale=rxScale, 
            output_dtype=torch.float16
        )

        if should_log:
             resq_log(f"DEBUG [ResQ] {self.prefix} Output Low stats: min={output_low.min()}, max={output_low.max()}, mean={output_low.float().mean()}, nan={torch.isnan(output_low).any()}")
             resq_log(f"DEBUG [ResQ] {self.prefix} Output High stats: min={output_high.min()}, max={output_high.max()}, mean={output_high.float().mean()}, nan={torch.isnan(output_high).any()}")

        
        output = torch.add(output_low, output_high)
        
        if bias is not None:
             # Check if bias seems valid (not all zero?)
             # Only print if bias is 0 (suspicious)? Or just print mean for QKV
            if should_log and "qkv" in self.prefix:
                 resq_log(f"DEBUG [ResQ] {self.prefix} Bias stats: min={bias.min()}, max={bias.max()}")
            output = output + bias
            
        output_shape = list(original_shape[:-1]) + [output.shape[-1]]
        return output.view(output_shape).to(torch.bfloat16)

