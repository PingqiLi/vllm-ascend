"""
Qwen3 ResQ Model with W8A8 Quantization

This module implements Qwen3 with:
1. W8A8 quantization using vllm-ascend's native quantization
2. ResQ online rotations: Uc (Q/K after RoPE), Ud (intermediate before down_proj)

Weight format (from convert_resq_to_w8a8.py):
- Standard W8A8 weights for all linear layers
- resq.layer.{i}.Uc: Q/K rotation after RoPE [head_dim, head_dim]
- resq.layer.{i}.Ud: intermediate rotation [blocksize, blocksize]

Usage:
    vllm serve /path/to/w8a8_checkpoint --quantization ascend

Environment variables:
    RESQ_DEBUG=1: Enable debug logging
    RESQ_SKIP_UC=1: Skip Uc rotation (for debugging)
    RESQ_SKIP_UD=1: Skip Ud rotation (for debugging)
"""

import os
import math
import logging
from typing import Optional, Iterable, Dict, Any

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.model_executor.models.qwen3 import (
    Qwen3Attention, Qwen3MLP, Qwen3DecoderLayer, 
    Qwen3Model, Qwen3ForCausalLM
)

logger = logging.getLogger(__name__)

# Configuration flags
RESQ_DEBUG = os.environ.get("RESQ_DEBUG", "0") == "1"
RESQ_SKIP_UC = os.environ.get("RESQ_SKIP_UC", "0") == "1"
RESQ_SKIP_UD = os.environ.get("RESQ_SKIP_UD", "0") == "1"


# ============================================================================
# Hadamard Transform Utilities (same as fake quant version)
# ============================================================================

def is_pow2(n: int) -> bool:
    return (n & (n - 1) == 0) and (n > 0)


def hadamard_transform(u: torch.Tensor) -> torch.Tensor:
    """Fast Hadamard transform using butterfly algorithm (unnormalized)."""
    n = u.shape[-1]
    assert is_pow2(n), f"Last dimension must be power of 2, got {n}"
    
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


# ============================================================================
# Rotation Functions
# ============================================================================

def apply_rotation(x: torch.Tensor, R: Optional[torch.Tensor]) -> torch.Tensor:
    """Apply block-wise rotation: x @ R for each head."""
    if R is None or R.numel() == 0:
        return x
    
    original_dtype = x.dtype
    K = R.shape[0]
    R_f32 = R.to(device=x.device, dtype=torch.float32)
    x_f32 = x.float()
    
    original_shape = x.shape
    N = original_shape[-1]
    
    if N == K:
        return torch.matmul(x_f32, R_f32).to(original_dtype)
    
    if N % K != 0:
        raise ValueError(f"Dim {N} must be divisible by block size {K}")
    
    num_blocks = N // K
    x_blocked = x_f32.reshape(*original_shape[:-1], num_blocks, K)
    x_rotated = torch.matmul(x_blocked, R_f32)
    return x_rotated.reshape(original_shape).to(original_dtype)


def apply_ud_rotation(
    x: torch.Tensor,
    Ud: Optional[torch.Tensor],
    Hd: Optional[torch.Tensor] = None,
    Hd_K: int = 1,
    blocksize: int = 256,
) -> torch.Tensor:
    """
    Apply Ud rotation before down_proj: x @ Ud where Ud = block_diag(Pd) @ H
    
    For TP=1 (this implementation):
    1. Apply Pd block-wise
    2. Apply H = Hd ⊗ H_butterfly
    
    Note: msmodelslim's Hd is normalized by 1/sqrt(K), need to compensate.
    """
    if Ud is None or Ud.numel() == 0:
        return x
    
    if RESQ_SKIP_UD:
        return x
    
    original_dtype = x.dtype
    original_shape = x.shape
    n = original_shape[-1]
    
    # Ud is Pd in the checkpoint (blocksize x blocksize)
    Pd = Ud
    num_blocks = n // blocksize
    
    x = x.float()
    
    # Step 1: Apply Pd block-wise
    Pd_f32 = Pd.to(device=x.device, dtype=torch.float32)
    x = x.reshape(*original_shape[:-1], num_blocks, blocksize)
    x = torch.matmul(x, Pd_f32.T)  # Pd.T as in msmodelslim
    
    # Step 2: Apply H = Hd ⊗ H_butterfly
    if Hd is not None and Hd_K > 1:
        # Apply H_butterfly to each block
        x = hadamard_transform(x.contiguous())
        x = x / math.sqrt(n)  # Normalize as in msmodelslim
        
        # Apply Hd
        Hd_f32 = Hd.to(device=x.device, dtype=torch.float32)
        batch_shape = x.shape[:-2]
        batch_size = 1
        for d in batch_shape:
            batch_size *= d
        x = x.reshape(batch_size, Hd_K, blocksize)
        x = torch.einsum('ij,bjk->bik', Hd_f32, x)
        
        # Compensate for normalized Hd (elements ~±1/sqrt(K))
        x = x * Hd_K
        
        x = x.reshape(*batch_shape, Hd_K, blocksize)
    
    return x.reshape(original_shape).to(original_dtype)


# ============================================================================
# Qwen3 ResQ Components
# ============================================================================

class Qwen3ResQW8A8Attention(Qwen3Attention):
    """Qwen3 Attention with Uc rotation after RoPE."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Uc rotation matrix [head_dim, head_dim]
        self.register_buffer('rotation_Uc', torch.empty(0))
    
    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        
        # QK norm
        q = self.q_norm(q.reshape(*q.shape[:-1], -1, self.head_dim)).reshape(q.shape)
        k = self.k_norm(k.reshape(*k.shape[:-1], -1, self.head_dim)).reshape(k.shape)
        
        # Apply RoPE
        q, k = self.rotary_emb(positions, q, k)
        
        # Apply Uc rotation after RoPE (ResQ key innovation)
        if not RESQ_SKIP_UC and self.rotation_Uc.numel() > 0:
            q = apply_rotation(q, self.rotation_Uc)
            k = apply_rotation(k, self.rotation_Uc)
        
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class Qwen3ResQW8A8MLP(Qwen3MLP):
    """Qwen3 MLP with Pd rotation before down_proj."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Pd rotation matrix [blocksize, blocksize]
        self.register_buffer('rotation_Pd', torch.empty(0))
        # Shared Hadamard (set after loading)
        self.shared_Hd: Optional[torch.Tensor] = None
        self.shared_Hd_K: int = 1
        self.blocksize: int = 256
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        gate, up = gate_up.chunk(2, dim=-1)
        intermediate = self.act_fn(gate) * up
        
        # Apply Pd rotation before down_proj (ResQ key innovation)
        if self.rotation_Pd.numel() > 0:
            intermediate = apply_ud_rotation(
                intermediate,
                Ud=self.rotation_Pd,
                Hd=self.shared_Hd,
                Hd_K=self.shared_Hd_K,
                blocksize=self.blocksize,
            )
        
        output, _ = self.down_proj(intermediate)
        return output


class Qwen3ResQW8A8DecoderLayer(Qwen3DecoderLayer):
    """Qwen3 Decoder Layer with ResQ rotations."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Replace attention and MLP with ResQ versions
        # Note: This is done by patching after standard init


class Qwen3ResQW8A8ForCausalLM(Qwen3ForCausalLM):
    """
    Qwen3 with W8A8 quantization and ResQ online rotations.
    
    Inherits from standard Qwen3ForCausalLM and adds:
    1. Uc rotation loading and application in attention
    2. Ud rotation loading and application in MLP
    
    Usage:
        vllm serve /path/to/w8a8_resq_checkpoint --quantization ascend
    """
    
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        
        # Global Hadamard parameters
        self.register_buffer('resq_Hd', torch.empty(0))
        self.resq_Hd_K = 1
        self.resq_blocksize = 256
        
        # Patch attention and MLP layers to add rotation buffers
        self._patch_layers_for_resq()
        
        if RESQ_DEBUG:
            logger.warning("[ResQ W8A8] Initialized with rotation support")
    
    def _patch_layers_for_resq(self):
        """Add rotation buffers to attention and MLP layers."""
        for layer in self.model.layers:
            # Add Uc buffer to attention
            if hasattr(layer, 'self_attn'):
                layer.self_attn.register_buffer('rotation_Uc', torch.empty(0))
            
            # Add Pd buffer and Hadamard refs to MLP
            if hasattr(layer, 'mlp'):
                layer.mlp.register_buffer('rotation_Pd', torch.empty(0))
                layer.mlp.shared_Hd = None
                layer.mlp.shared_Hd_K = 1
                layer.mlp.blocksize = 256
    
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights including ResQ rotation matrices.
        
        msmodelslim output keys:
        - resq.Hd: Hadamard matrix
        - resq.Hd_K: Hadamard size
        - resq.down_proj_blocksize: block size for Pd rotation
        - resq.intermediate_size: intermediate size (ignored)
        - resq.layer.{i}.Uc: Q/K rotation after RoPE
        - resq.layer.{i}.Pd: down_proj rotation
        """
        # Collect weights
        weights_dict: Dict[str, torch.Tensor] = {}
        for name, tensor in weights:
            weights_dict[name] = tensor
        
        # Load global ResQ parameters
        if 'resq.Hd' in weights_dict:
            self.resq_Hd = weights_dict.pop('resq.Hd')
        if 'resq.Hd_K' in weights_dict:
            self.resq_Hd_K = int(weights_dict.pop('resq.Hd_K').item())
        if 'resq.down_proj_blocksize' in weights_dict:
            self.resq_blocksize = int(weights_dict.pop('resq.down_proj_blocksize').item())
        if 'resq.intermediate_size' in weights_dict:
            weights_dict.pop('resq.intermediate_size')  # Not needed
        
        # Load per-layer rotation matrices
        for i, layer in enumerate(self.model.layers):
            # Load Uc for attention
            uc_key = f'resq.layer.{i}.Uc'
            if uc_key in weights_dict:
                layer.self_attn.rotation_Uc = weights_dict.pop(uc_key)
                if RESQ_DEBUG:
                    logger.warning(f"[ResQ] Loaded Uc for layer {i}: {layer.self_attn.rotation_Uc.shape}")
            
            # Load Pd for MLP
            pd_key = f'resq.layer.{i}.Pd'
            if pd_key in weights_dict:
                layer.mlp.rotation_Pd = weights_dict.pop(pd_key)
                if RESQ_DEBUG:
                    logger.warning(f"[ResQ] Loaded Pd for layer {i}: {layer.mlp.rotation_Pd.shape}")
            
            # Set shared Hadamard references
            layer.mlp.shared_Hd = self.resq_Hd if self.resq_Hd.numel() > 0 else None
            layer.mlp.shared_Hd_K = self.resq_Hd_K
            layer.mlp.blocksize = self.resq_blocksize
        
        # Remove any remaining resq.* keys
        for key in list(weights_dict.keys()):
            if key.startswith('resq.'):
                if RESQ_DEBUG:
                    logger.warning(f"[ResQ] Removing unhandled key: {key}")
                weights_dict.pop(key)
        
        # Load remaining weights using parent's loader
        remaining_weights = [(k, v) for k, v in weights_dict.items()]
        return super().load_weights(iter(remaining_weights))
