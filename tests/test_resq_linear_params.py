
import sys
import os
import types
from unittest.mock import MagicMock

# 1. Setup Mocks BEFORE importing vllm_ascend
# This allows testing without installing vllm
mock_vllm = types.ModuleType("vllm")
sys.modules["vllm"] = mock_vllm

# Mock vllm.config
mock_vllm_config = types.ModuleType("vllm.config")
sys.modules["vllm.config"] = mock_vllm_config
mock_vllm.config = mock_vllm_config
class MockQuantizationConfig:
    def __init__(self, *args, **kwargs): pass
    @classmethod
    def get_name(cls): return "MOCK"
mock_vllm_config.QuantizationConfig = MockQuantizationConfig
mock_vllm_config.get_current_vllm_config = MagicMock()

# Mock vllm.model_executor
mock_vllm_executor = types.ModuleType("vllm.model_executor")
sys.modules["vllm.model_executor"] = mock_vllm_executor
mock_vllm.model_executor = mock_vllm_executor

# Mock vllm.model_executor.layers
mock_layers = types.ModuleType("vllm.model_executor.layers")
sys.modules["vllm.model_executor.layers"] = mock_layers
mock_vllm_executor.layers = mock_layers

# Mock vllm.model_executor.layers.linear
mock_linear = types.ModuleType("vllm.model_executor.layers.linear")
sys.modules["vllm.model_executor.layers.linear"] = mock_linear
class MockLinearMethodBase:
    def create_weights(self, layer, **kwargs): pass
    def process_weights_after_loading(self, layer): pass
    def apply(self, layer, x, bias=None): return x
mock_linear.LinearMethodBase = MockLinearMethodBase
class MockLinearBase: pass
mock_linear.LinearBase = MockLinearBase
class MockRowParallelLinear: pass
mock_linear.RowParallelLinear = MockRowParallelLinear
mock_layers.linear = mock_linear

# Ensure vllm acts as a package
mock_vllm.__path__ = []

# Mock vllm.distributed
mock_dist = types.ModuleType("vllm.distributed")
sys.modules["vllm.distributed"] = mock_dist
mock_dist.get_tensor_model_parallel_rank = lambda: 0
mock_dist.get_tensor_model_parallel_world_size = lambda: 1
mock_vllm.distributed = mock_dist

# Mock vllm.model_executor.layers.quantization
mock_quant = types.ModuleType("vllm.model_executor.layers.quantization")
sys.modules["vllm.model_executor.layers.quantization"] = mock_quant
mock_layers.quantization = mock_quant
def mock_register_quantization_config(name):
    def decorator(cls):
        return cls
    return decorator
mock_quant.register_quantization_config = mock_register_quantization_config

# Mock vllm.model_executor.layers.quantization.base_config
mock_base_config = types.ModuleType("vllm.model_executor.layers.quantization.base_config")
sys.modules["vllm.model_executor.layers.quantization.base_config"] = mock_base_config
mock_base_config.QuantizationConfig = MockQuantizationConfig
mock_base_config.QuantizeMethodBase = object # Dummy

# Mock vllm.model_executor.utils
mock_utils = types.ModuleType("vllm.model_executor.utils")
sys.modules["vllm.model_executor.utils"] = mock_utils
def mock_set_weight_attrs(weight, attrs):
    for k, v in attrs.items():
        setattr(weight, k, v)
mock_utils.set_weight_attrs = mock_set_weight_attrs

# Mock vllm_ascend.ops.linear (imported by utils.py)
# We need to make sure imports in utils.py work
mock_va_ops = types.ModuleType("vllm_ascend.ops")
sys.modules["vllm_ascend.ops"] = mock_va_ops
mock_va_linear = types.ModuleType("vllm_ascend.ops.linear")
sys.modules["vllm_ascend.ops.linear"] = mock_va_linear
mock_va_linear.AscendUnquantizedLinearMethod = MagicMock()
mock_va_linear.AscendLinearMethod = MagicMock() # Ensure this mocks what utils imports

# Mock other imports in utils.py
sys.modules["vllm_ascend.ops.common_fused_moe"] = MagicMock()
sys.modules["vllm.model_executor.layers.fused_moe"] = MagicMock()
sys.modules["vllm.model_executor.layers.quantization.kv_cache"] = MagicMock()
sys.modules["vllm.model_executor.layers.vocab_parallel_embedding"] = MagicMock()
sys.modules["vllm.model_executor.parameter"] = MagicMock()
sys.modules["vllm_ascend.distributed.parallel_state"] = MagicMock()
sys.modules["vllm_ascend.utils"] = MagicMock()
sys.modules["vllm.logger"] = MagicMock()

# Also prevent implicit import errors
sys.modules["vllm_ascend.quantization.w4a4_flatquant_dynamic"] = MagicMock()
sys.modules["vllm_ascend.quantization.w4a8_dynamic"] = MagicMock()
sys.modules["vllm_ascend.quantization.w8a8"] = MagicMock()
sys.modules["vllm_ascend.quantization.w8a8_dynamic"] = MagicMock()


import torch
import torch.nn as nn

# Mock torch_npu if missing (CPU env)
if not hasattr(torch, "npu") or not torch.npu.is_available():
    mock_npu = types.ModuleType("torch_npu")
    mock_npu.npu_dynamic_quant = lambda x, dst_type: (torch.zeros_like(x), torch.zeros(x.shape[0]))
    mock_npu.npu_quant_matmul = lambda *args, **kwargs: torch.zeros(1)
    mock_npu.npu_format_cast = lambda x, fmt: x
    sys.modules["torch_npu"] = mock_npu
    torch.npu = mock_npu
    # Add fake attributes
    if not hasattr(torch, "quint4x2"):
        torch.quint4x2 = "quint4x2"

# NOW we can import the code under test
# Import relatively or absolute
try:
    from vllm_ascend.quantization.resq_linear import ResQLinearMethod
    from vllm_ascend.quantization.quant_config import ResQQuantConfig
except ImportError:
    # If running from project root
    sys.path.append(os.getcwd())
    from vllm_ascend.quantization.resq_linear import ResQLinearMethod
    from vllm_ascend.quantization.quant_config import ResQQuantConfig

def test_resq_parameters():
    print("Testing ResQ Parameter Registration...")
    
    layer = nn.Linear(128, 64)
    quant_config_dict = {
        "quant_method": "RESQ",
        "weight_low": "INT4",
        "weight_high": "INT8"
    }
    
    config = ResQQuantConfig(quant_config_dict)
    
    # Instantiate method
    method = ResQLinearMethod(config, prefix="model.layers.0.mlp.down_proj", packed_modules_mapping={})
    
    # Call create_weights
    method.create_weights(
        layer=layer,
        input_size_per_partition=128,
        output_partition_sizes=[64],
        input_size=128,
        output_size=64,
        params_dtype=torch.float16
    )
    
    # Verify parameters
    expected_params = ["weight_low", "weight_high", "scale_low", "scale_high"]
    expected_resq = ["rotation_Pd", "rotation_Hd", "h_butterfly"]
    
    for name in expected_params:
        if not hasattr(layer, name):
            print(f"FAILED: Missing {name}")
            return
        print(f"OK: Found {name}")
        
    for name in expected_resq:
        if not hasattr(layer, name):
             print(f"FAILED: Missing {name} (Should be present for down_proj)")
             return
        print(f"OK: Found {name}")

    print("Testing weight_loader binding...")
    # Check if weight_loader is bound to parameters
    if not hasattr(layer.weight_low, "weight_loader"):
        print("FAILED: weight_low has no weight_loader bound")
        return
    
    # Test loader behavior (resizing)
    # layer.weight_low is initialized as empty(0)
    print(f"Initial size: {layer.weight_low.shape}")
    
    dummy_weight = torch.randn(10).to(torch.int8)
    # Invoke the bound loader
    layer.weight_low.weight_loader(layer.weight_low, dummy_weight)
    
    if layer.weight_low.numel() != 10:
        print(f"FAILED: weight_loader did not resize param. Size: {layer.weight_low.shape}")
        return
        
    print(f"OK: weight_loader resized param to {layer.weight_low.shape}")
    print("ResQLinearMethod Test Passed!")

if __name__ == "__main__":
    test_resq_parameters()
