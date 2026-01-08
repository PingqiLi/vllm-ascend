"""
ResQ Mixed-Precision Quantized MatMul Operations

Reference: msmodelslim reference_op_impl.py

Note: On NPU, int32 matmul is not supported. We use:
- CPU: torch.matmul(int32, int32) 
- NPU: torch_npu.npu_quant_matmul (int8 + scale)

Usage:
    from vllm_ascend.ops.resq_quant_matmul import resq_quant_matmul
    
    output = resq_quant_matmul(
        x=x_int8,              # (M, K) int8, 已量化的activation
        weight=weight_int8,    # (E, K, N) int8, 已量化的weight
        lweightScale=lscale,   # (E, 1, N) float32, int4部分weight scale
        hweightScale=hscale,   # (E, 1, N) float32, int8部分weight scale
        lxScale=lx_scale,      # (M,) float32, int4部分activation scale
        rxScale=rx_scale,      # (M,) float32, int8部分activation scale
        splitKPos=split_k,     # int, K维度切分位置
        groupList=group_list,  # (E,) int64, 每个expert处理的token数，默认None表示E=1
    )
"""

import torch
from typing import Optional

try:
    import torch_npu
    HAS_NPU = True
except ImportError:
    HAS_NPU = False


def pack_int4_to_int8_signed(x: torch.Tensor) -> torch.Tensor:
    """
    x: int8 tensor, shape (E, K, N)，值域 ∈ [-8, 7]
    return: int8 tensor, shape (E, K, N/2)，每个元素打包两个有符号 int4
    """
    assert x.dtype == torch.int8
    E, K, N = x.shape
    assert N % 2 == 0
    
    x_unsigned = torch.where(x < 0, x + 16, x).to(torch.int32)
    low = x_unsigned[..., 0::2]
    high = x_unsigned[..., 1::2]
    out = (low | (high << 4)).to(torch.int8)
    return out


def _is_npu_tensor(x: torch.Tensor) -> bool:
    """Check if tensor is on NPU device."""
    return HAS_NPU and x.device.type == 'npu'


def MM(x: torch.Tensor, weight: torch.Tensor, perChannelScale: torch.Tensor, perTokenScale: torch.Tensor, 
       m: int, outDtype: torch.dtype, KNum_per_group: int, groupListType: int, dequantModle: int):
    """
    量化MatMul。
    
    x: (m, k) int8
    weight: (k, n) int8
    perChannelScale: (1, n) 或 (k//KNum_per_group, n)
    perTokenScale: (m,)
    
    Note: On NPU, uses npu_quant_matmul since int32 matmul is not supported.
    """
    K, N = x.shape[1], weight.shape[1]
    
    if dequantModle == 0:
        # Per-group mode - not optimized for NPU yet, use CPU fallback logic
        c_temp1 = torch.zeros(m, N, device=x.device).type(torch.float32)
        for k_idx in range(K // KNum_per_group):
            x_slice = x[:, k_idx*KNum_per_group:(k_idx+1)*KNum_per_group]
            w_slice = weight[k_idx*KNum_per_group:(k_idx+1)*KNum_per_group, :]
            
            if _is_npu_tensor(x):
                # NPU: use float16 matmul as fallback
                mm_result = torch.matmul(x_slice.to(torch.float16), w_slice.to(torch.float16)).to(torch.float32)
            else:
                # CPU: use int32 matmul
                mm_result = torch.matmul(x_slice.to(torch.int32), w_slice.to(torch.int32)).to(torch.float32)
            
            c_temp1 = c_temp1 + perTokenScale.reshape(m, 1) * perChannelScale[k_idx].reshape(1, N) * mm_result
        return c_temp1.type(outDtype)
    
    elif dequantModle == 1:
        # Per-channel mode
        if _is_npu_tensor(x):
            # NPU: use npu_quant_matmul
            # npu_quant_matmul(x1, x2, scale, *, offset=None, pertoken_scale=None, bias=None)
            # x1: (M, K) int8
            # x2: (K, N) int8  
            # scale: (N,) float32 - weight scale (per-channel)
            # Returns: (M, N) float16
            
            # Flatten perChannelScale to (N,)
            scale = perChannelScale.flatten().to(torch.float32)
            
            # Call npu_quant_matmul
            result = torch_npu.npu_quant_matmul(x, weight, scale)
            
            # Result is tuple, get first element
            if isinstance(result, tuple):
                c_temp1 = result[0].to(torch.float32)
            else:
                c_temp1 = result.to(torch.float32)
            
            # Apply per-token scale
            c_temp2 = torch.mul(c_temp1, perTokenScale.reshape(m, 1))
            return c_temp2.type(outDtype)
        else:
            # CPU: use int32 matmul (original implementation)
            c_temp1 = torch.matmul(x.type(torch.int32), weight.type(torch.int32))
            c_temp1 = c_temp1.type(torch.float32)
            c_temp2 = torch.mul(c_temp1, perChannelScale).type(torch.float16).type(torch.float32)
            c_temp3 = torch.mul(c_temp2, perTokenScale.reshape(m, 1))
            return c_temp3.type(outDtype)


def MIXPGMM(x: torch.Tensor, weight: torch.Tensor, perChannelScale: torch.Tensor, perTokenScale: torch.Tensor, 
            groupList: torch.Tensor, outDtype: torch.dtype, KNum_per_group: int, groupListType: int, dequantModle: int):
    """
    MoE分组MatMul。
    
    x: (M, K) int8
    weight: (E, K, N) int8
    perChannelScale: (E, 1, N) 或 (E, K//KNum_per_group, N)
    perTokenScale: (M,)
    groupList: (E,) 每个expert的token数
    """
    M, N = x.shape[0], weight.shape[2]
    Output = torch.zeros(M, N, dtype=outDtype, device=x.device)

    start_idx = 0
    preV = 0
    groupList = groupList.tolist()
    for i, v in enumerate(groupList):
        if groupListType == 0:
            currV = v
            tempV = currV - preV
            preV = currV
        elif groupListType == 1:
            tempV = v
        if (tempV > 0):
            Output[start_idx:start_idx + tempV] = \
                MM(x[start_idx:start_idx + tempV], 
                   weight[i], 
                   perChannelScale[i], 
                   perTokenScale[start_idx:start_idx + tempV], 
                   tempV, outDtype, KNum_per_group, groupListType, dequantModle)
        start_idx += tempV
    return Output.to(outDtype)


def resq_quant_matmul(
    x: torch.Tensor,
    weight: torch.Tensor,
    lweightScale: torch.Tensor,
    hweightScale: torch.Tensor,
    lxScale: torch.Tensor,
    rxScale: torch.Tensor,
    splitKPos: int,
    groupList: Optional[torch.Tensor] = None,
    outDtype: torch.dtype = torch.float16,
    groupListType: int = 1,
    dequantModle: int = 1,
) -> torch.Tensor:
    """
    ResQ混精MatMul对外接口。
    
    计算: output = x_low @ W_low * lweightScale * lxScale + x_high @ W_high * hweightScale * rxScale
    
    参数:
        x: 已量化的activation，shape (M, K)，dtype int8
           - x[:, :splitKPos] 值域 [-8, 7] (int4范围)
           - x[:, splitKPos:] 值域 [-128, 127] (int8范围)
           
        weight: 已量化的weight，shape (E, K, N) 或 (K, N)，dtype int8
           - weight[..., :splitKPos, :] 值域 [-8, 7] (int4范围)
           - weight[..., splitKPos:, :] 值域 [-128, 127] (int8范围)
           
        lweightScale: int4部分的weight scale
           - dequantModle=1: shape (E, 1, N) 或 (1, N) 或 (N,)
           - dequantModle=0: shape (E, K//KNum_per_group, N)
           
        hweightScale: int8部分的weight scale，shape同上
        
        lxScale: int4部分的per-token activation scale，shape (M,)，dtype float32
        
        rxScale: int8部分的per-token activation scale，shape (M,)，dtype float32
        
        splitKPos: K维度切分位置。前splitKPos个通道是int4，后面是int8
        
        groupList: 每个expert处理的token数，shape (E,)，dtype int64
                   None表示E=1，即所有M个token都由单个expert处理
                   
        outDtype: 输出dtype，默认torch.float16
        
        groupListType: groupList语义。0=cumsum模式, 1=count模式 (默认)
        
        dequantModle: 反量化模式。0=per-channel&&per-group, 1=纯per-channel (默认)
        
    返回:
        output: shape (M, N)，dtype outDtype
    """
    M, K = x.shape
    KNum_per_group = K
    
    # 处理weight shape: 2D -> 3D
    if weight.dim() == 2:
        weight = weight.unsqueeze(0)  # (K, N) -> (1, K, N)
    
    # 处理scale shape: 确保是 (E, ..., N)
    if lweightScale.dim() == 1:
        lweightScale = lweightScale.unsqueeze(0).unsqueeze(0)  # (N,) -> (1, 1, N)
    elif lweightScale.dim() == 2:
        lweightScale = lweightScale.unsqueeze(0)  # (1, N) -> (1, 1, N)
    
    if hweightScale.dim() == 1:
        hweightScale = hweightScale.unsqueeze(0).unsqueeze(0)
    elif hweightScale.dim() == 2:
        hweightScale = hweightScale.unsqueeze(0)
    
    # 处理groupList: None -> E=1
    if groupList is None:
        groupList = torch.tensor([M], dtype=torch.int64, device=x.device)

    # int8部分
    y_high = MIXPGMM(
        x[:, splitKPos:K], 
        weight[:, splitKPos:K, :], 
        hweightScale, 
        rxScale, 
        groupList, 
        torch.float16, 
        KNum_per_group, 
        groupListType, 
        dequantModle
    )
    
    # int4部分
    y_low = MIXPGMM(
        x[:, :splitKPos], 
        weight[:, :splitKPos, :], 
        lweightScale, 
        lxScale, 
        groupList, 
        torch.float16, 
        KNum_per_group, 
        groupListType, 
        dequantModle
    )
    
    # 相加
    output = y_low + y_high
    return output.to(outDtype)
