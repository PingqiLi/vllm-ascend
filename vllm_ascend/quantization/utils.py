from typing import Any, Dict, Optional, Type

from vllm.logger import logger

from .w4a4_flatquant_dynamic import AscendW4A4FlatQuantDynamicLinearMethod
from .w4a8_dynamic import (AscendW4A8DynamicFusedMoEMethod,
                           AscendW4A8DynamicLinearMethod)
from .w8a8 import (AscendC8KVCacheMethod, AscendW8A8FusedMoEMethod,
                   AscendW8A8LinearMethod)
from .w8a8_dynamic import (AscendW8A8DynamicFusedMoEMethod,
                           AscendW8A8DynamicLinearMethod)
from .resq_linear import ResQLinearMethod

ASCEND_QUANTIZATION_METHOD_MAP: Dict[str, Dict[str, Type[Any]]] = {
    "W4A8_DYNAMIC": {
        "linear": AscendW4A8DynamicLinearMethod,
        "moe": AscendW4A8DynamicFusedMoEMethod,
    },
    "W4A4_FLATQUANT_DYNAMIC": {
        "linear": AscendW4A4FlatQuantDynamicLinearMethod,
    },
    "W8A8": {
        "linear": AscendW8A8LinearMethod,
        "moe": AscendW8A8FusedMoEMethod,
        "attention": AscendC8KVCacheMethod,
    },
    "W8A8_DYNAMIC": {
        "linear": AscendW8A8DynamicLinearMethod,
        "moe": AscendW8A8DynamicFusedMoEMethod,
    },
    "C8": {
        "attention": AscendC8KVCacheMethod,
    },
    "RESQ": {
        "linear": ResQLinearMethod,
    },
}


def _get_layer_quant_type(quant_description: Dict[str, Any], prefix: str) -> str:
    """Get quant type for a single layer, handling both standard and RESQ formats."""
    # Try standard .weight key
    weight_key = prefix + '.weight'
    if weight_key in quant_description:
        return quant_description[weight_key]

    # Try RESQ format (.weight_low)
    weight_low_key = prefix + '.weight_low'
    if weight_low_key in quant_description:
        return quant_description[weight_low_key]

    raise KeyError(f"Cannot find quant type for {prefix}: "
                   f"neither {weight_key} nor {weight_low_key} found in quant_description")


def get_linear_quant_type(quant_description: Dict[str, Any], prefix: str,
                          packed_modules_mapping: Dict[str, Any]):
    proj_name = prefix.split(".")[-1]
    if proj_name in packed_modules_mapping:
        quant_type = None
        shard_prefixes = [
            prefix.replace(proj_name, shard_proj_name)
            for shard_proj_name in packed_modules_mapping[proj_name]
        ]
        for shard_prefix in shard_prefixes:
            shard_quant_type = _get_layer_quant_type(quant_description, shard_prefix)

            if quant_type is None:
                quant_type = shard_quant_type
            elif shard_quant_type != quant_type:
                raise ValueError(
                    f"Not all shards of {prefix} are quantized with same quant type."
                    f"Shard {proj_name} uses {shard_quant_type}, but another shard"
                    f"use {quant_type}. Please check quantization config.")
    elif "experts" in prefix:
        # For the experts' prefix (e.g., "model.layers.3.mlp.experts")
        # Assume all experts within the same MLP use the same quantization method
        experts_quant_description = set(quant_description[layer]
                                        for layer in quant_description
                                        if prefix in layer)
        if not len(experts_quant_description) == 1:
            raise RuntimeError(
                f"{prefix} has different quantization type: {experts_quant_description}."
            )
        quant_type = experts_quant_description.pop()
    else:
        quant_type = _get_layer_quant_type(quant_description, prefix)
    return quant_type


def get_quant_method(quant_description: Dict[str, Any],
                     prefix: str,
                     layer_type: str,
                     packed_modules_mapping: Optional[Dict[str, Any]] = None):
    logger.info_once("Using the vLLM Ascend Quantization now!")
    if packed_modules_mapping is None:
        packed_modules_mapping = dict()

    # Special handling for RESQ model type - check per-layer quant type
    model_quant_type = quant_description.get("model_quant_type", "")
    if model_quant_type.upper() == "RESQ" and layer_type == "linear":
        # Get the actual quant type for this specific layer
        layer_quant_type = get_linear_quant_type(quant_description, prefix,
                                                  packed_modules_mapping)
        # Only use ResQLinearMethod for layers that are actually RESQ
        # Other layers (e.g., down_proj with W8A8_DYNAMIC) fall through to standard path
        if layer_quant_type.upper() == "RESQ":
            return ResQLinearMethod(quant_description, prefix, packed_modules_mapping)

    # Attention
    if '.attn' in prefix and 'fa_quant_type' in quant_description.keys():
        quant_type = quant_description['fa_quant_type']
    # Use KVCache int8
    elif '.attn' in prefix and 'kv_quant_type' in quant_description.keys():
        quant_type = quant_description['kv_quant_type']
    # Linear
    else:
        quant_type = get_linear_quant_type(quant_description, prefix,
                                           packed_modules_mapping)
    if quant_type in ASCEND_QUANTIZATION_METHOD_MAP.keys():
        method_map = ASCEND_QUANTIZATION_METHOD_MAP[quant_type]
        if layer_type in method_map.keys():
            method_cls = method_map[layer_type]
            return method_cls()
        else:
            raise NotImplementedError(
                f"Currently, vLLM Ascend doesn't support {quant_type} for {layer_type}."
            )
    raise NotImplementedError("Currently, vLLM Ascend only supports following quant types:" \
                                f"{list(ASCEND_QUANTIZATION_METHOD_MAP.keys())}")
