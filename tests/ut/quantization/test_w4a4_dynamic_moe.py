from unittest.mock import Mock, patch
import torch
from tests.ut.base import TestBase
from vllm_ascend.quantization.w4a4_dynamic import AscendW4A4DynamicFusedMoEMethod

class TestAscendW4A4DynamicFusedMoEMethod(TestBase):
    experts = 8
    input_size = 16
    output_size = 56
    group_size = 128  # 模拟 Per-Channel 或 Group 场景

    @patch('vllm_ascend.quantization.w4a4_dynamic.get_ascend_config')
    @patch('vllm_ascend.quantization.w4a4_dynamic.get_current_vllm_config')
    @patch('vllm_ascend.quantization.w4a4_dynamic.get_ep_group')
    @patch('vllm_ascend.quantization.w4a4_dynamic.get_mc2_group')
    @patch('torch.distributed.get_rank', return_value=0)
    def setUp(self, mock_get_rank, mock_get_mc2_group, mock_get_ep_group,
              get_current_vllm_config, mock_get_ascend_config):
        # 1. Mock 配置
        mock_ascend_config = Mock()
        mock_ascend_config.dynamic_eplb = False
        mock_get_ascend_config.return_value = mock_ascend_config

        mock_vllm_config = Mock()
        mock_vllm_config.quant_config = Mock(quant_description={
            "group_size": self.group_size,
            "version": "0.0.0"
        })
        mock_vllm_config.parallel_config = Mock(enable_expert_parallel=True)
        # Mock scheduler config needed for init
        mock_vllm_config.scheduler_config = Mock(max_num_batched_tokens=2048,
                                                 max_model_len=2048,
                                                 enable_chunked_prefill=False)
        get_current_vllm_config.return_value = mock_vllm_config
        
        # 2. 初始化被测类
        self.quant_method = AscendW4A4DynamicFusedMoEMethod()

    def test_weight_shapes(self):
        """测试权重申请形状是否正确"""
        # Old Version
        param_dict = self.quant_method.get_weight(self.experts, self.input_size, self.output_size, torch.bfloat16)
        # Old version w13 shape should be [experts, 2*input, output]
        self.assertEqual(param_dict["w13_weight"].shape, (self.experts, 2 * self.input_size, self.output_size))
        
        # Power check: New Version
        self.quant_method.new_quant_version = True
        param_dict_new = self.quant_method.get_weight(self.experts, self.input_size, self.output_size, torch.bfloat16)
        # New version packs 2x, so w13 input dim is halved (or treated differently depending on logic)
        self.assertEqual(param_dict_new["w13_weight"].shape, (self.experts, self.input_size, self.output_size))

    def test_process_weights_execution(self):
        """测试权重处理流程 (需要在 NPU 环境运行)"""
        if not torch.cuda.is_available() and not hasattr(torch, 'npu'):
             print("Skipping execution test (No NPU found).")
             return

        # 构建一个模拟的 Layer
        layer = torch.nn.Module()
        # 初始化权重 (CPU Tensor, 会在 process 中被 .npu() )
        # 注意：W4A4 dynamic w13 是 int8 存储
        layer.w13_weight = torch.nn.Parameter(torch.randint(-8, 8, (self.experts, 2*self.input_size, self.output_size)).to(torch.int8), requires_grad=False)
        layer.w2_weight = torch.nn.Parameter(torch.randint(-8, 8, (self.experts, self.output_size, self.input_size)).to(torch.int8), requires_grad=False)
        
        # 初始化 Scale
        layer.w13_weight_scale = torch.nn.Parameter(torch.ones(self.experts, 2*self.input_size, 1), requires_grad=False)
        layer.w2_weight_scale = torch.nn.Parameter(torch.ones(self.experts, self.output_size, 1), requires_grad=False)
        
        # 强制设为 Per-Channel 模式以简化测试
        self.quant_method.is_per_channel_weight = True
        self.quant_method.new_quant_version = False

        # 执行! (如果 NPU 环境正常，这里应该能跑通)
        try:
            self.quant_method.process_weights_after_loading(layer)
        except Exception as e:
            self.fail(f"process_weights_after_loading failed with error: {e}")

        # 验证结果
        # 1. 权重应该被打包成 int32
        self.assertEqual(layer.w13_weight.dtype, torch.int32)
        self.assertEqual(layer.w2_weight.dtype, torch.int32)
        
        # 2. 应该生成了 scale_bias (根据老版本逻辑)
        self.assertTrue(hasattr(layer, "w13_scale_bias"))
        self.assertTrue(hasattr(layer, "w2_scale_bias"))