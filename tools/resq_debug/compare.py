"""
激活值比较工具

Checkpoint 格式:
    - CKPT_O: 原始 bf8 模型 (architectures: Qwen3ForCausalLM)
    - CKPT_A: ResQ 量化模型 (architectures: Qwen3ResQForCausalLM)
    - CKPT_B: 辅助矩阵目录 (无 config.json)

模式:
    - save-orig: 保存原始 bf8 模型激活
    - save-resq: 保存 ResQ 模型激活 (伪量化: load 时 dequant)
    - save-resq-true: 保存 ResQ 模型激活 (真量化: forward 时 int32 matmul)
    - compare: 比较两个激活文件

用法:
    # 保存原始模型激活
    python -m tools.resq_debug.compare save-orig \
        --model ${CKPT_O} --out orig.pt

    # 保存 ResQ 模型激活 (伪量化) CPU
    python -m tools.resq_debug.compare save-resq \
        --ckpt-a ${CKPT_A} --out resq.pt

    # 保存 ResQ 模型激活 (真量化) NPU
    python -m tools.resq_debug.compare save-resq-true \
        --ckpt-a ${CKPT_A} --out resq_true.pt

    # 比较伪量化和原始激活值
    python -m tools.resq_debug.compare compare \
        --orig orig.pt --resq resq.pt --ckpt-b ${CKPT_B}

    # 比较真量化和原始激活值
    python -m tools.resq_debug.compare compare \
        --orig orig.pt --resq resq_true.pt --ckpt-b ${CKPT_B}
"""

import argparse
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


class EarlyStopException(Exception):
    """用于提前终止前向传播的异常"""
    pass


DEFAULT_TEXT = """The quick brown fox jumps over the lazy dog. This is a sample text for testing language model activations. We need a reasonably long piece of text to fill the sequence length for proper comparison between the original and quantized models. The text content itself doesn't matter much, what matters is that both models receive exactly the same input tokens so we can compare their intermediate activations layer by layer. Machine learning models process text by first tokenizing it into smaller units called tokens, then passing these through multiple transformer layers. Each layer performs attention and feedforward operations, producing intermediate representations that we want to compare."""


def save_original(model_path: str, out_path: str, prompt: str, device: str, layers_arg: str, seq_len: int):
    """保存原始模型的激活值"""
    print(f"Loading {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # 解析 layers 参数
    num_layers = len(model.model.layers)
    if layers_arg == "all":
        layers = list(range(num_layers))
    else:
        layers = [int(x) for x in layers_arg.split(",")]
    max_layer = max(layers)
    early_stop = (max_layer < num_layers - 1)
    print(f"Saving activations for {len(layers)} layers (max_layer={max_layer}, early_stop={early_stop})")
    
    # 构造指定长度的输入（截断或 pad）
    text = prompt if prompt != "default" else DEFAULT_TEXT
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=seq_len, padding="max_length")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    print(f"Input sequence length: {input_ids.shape[1]} (actual tokens: {attention_mask.sum().item()})")
    acts = {}
    handles = []
    
    # 注册 hooks
    for i in layers:
        layer = model.model.layers[i]
        
        def make_hook(idx, name, is_last_hook=False):
            def hook(m, inp, out):
                key = f"L{idx}.{name}"
                acts[key] = (out[0] if isinstance(out, tuple) else out).cpu().clone()
                if is_last_hook:
                    raise EarlyStopException()
            return hook
        
        is_last_layer = (i == max_layer) and early_stop
        handles.append(layer.input_layernorm.register_forward_hook(make_hook(i, "input_ln")))
        handles.append(layer.self_attn.q_proj.register_forward_hook(make_hook(i, "q")))
        handles.append(layer.self_attn.k_proj.register_forward_hook(make_hook(i, "k")))
        handles.append(layer.self_attn.v_proj.register_forward_hook(make_hook(i, "v")))
        handles.append(layer.self_attn.o_proj.register_forward_hook(make_hook(i, "o")))
        handles.append(layer.mlp.gate_proj.register_forward_hook(make_hook(i, "gate")))
        handles.append(layer.mlp.up_proj.register_forward_hook(make_hook(i, "up")))
        # down_proj 是每层最后一个 hook，在最后一层时触发 early stop
        handles.append(layer.mlp.down_proj.register_forward_hook(make_hook(i, "down", is_last_hook=is_last_layer)))
    
    # 只有跑完整模型时才保存 logits
    if not early_stop:
        handles.append(model.lm_head.register_forward_hook(
            lambda m, inp, out: acts.__setitem__("logits", out.cpu().clone())))
    
    # Forward (使用 try-except 捕获早停)
    with torch.no_grad():
        try:
            model(input_ids, attention_mask=attention_mask)
        except EarlyStopException:
            pass  # 正常的早停
    
    for h in handles:
        h.remove()
    
    acts["input_ids"] = input_ids.cpu()
    acts["attention_mask"] = attention_mask.cpu()
    acts["prompt"] = prompt
    acts["seq_len"] = seq_len
    acts["layers"] = layers
    
    torch.save(acts, out_path)
    print(f"Saved {len(acts)} tensors to {out_path}")


def save_resq(ckpt_a: str, out_path: str, 
              prompt: str, device: str, layers_arg: str, seq_len: int):
    """保存 ResQ 模型的激活值 (只需 CKPT_A)"""
    from .modeling_qwen3_resq import Qwen3ResQForCausalLM
    
    print(f"Loading ResQ model from {ckpt_a}...")
    model = Qwen3ResQForCausalLM.from_resq_checkpoint(ckpt_a, device).eval()
    tokenizer = AutoTokenizer.from_pretrained(ckpt_a, trust_remote_code=True)
    
    # 解析 layers 参数
    num_layers = len(model.layers)
    if layers_arg == "all":
        layers = list(range(num_layers))
    else:
        layers = [int(x) for x in layers_arg.split(",")]
    max_layer = max(layers)
    early_stop = (max_layer < num_layers - 1)
    print(f"Saving activations for {len(layers)} layers (max_layer={max_layer}, early_stop={early_stop})")
    
    # 构造指定长度的输入（截断或 pad）
    text = prompt if prompt != "default" else DEFAULT_TEXT
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=seq_len, padding="max_length")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    print(f"Input sequence length: {input_ids.shape[1]} (actual tokens: {attention_mask.sum().item()})")
    acts = {}
    handles = []
    
    # 注册 hooks (与 save_original 相同结构)
    for i in layers:
        layer = model.layers[i]
        
        def make_hook(idx, name, is_last_hook=False):
            def hook(m, inp, out):
                key = f"L{idx}.{name}"
                acts[key] = (out[0] if isinstance(out, tuple) else out).cpu().clone()
                if is_last_hook:
                    raise EarlyStopException()
            return hook
        
        is_last_layer = (i == max_layer) and early_stop
        handles.append(layer.input_layernorm.register_forward_hook(make_hook(i, "input_ln")))
        handles.append(layer.self_attn.q_proj.register_forward_hook(make_hook(i, "q")))
        handles.append(layer.self_attn.k_proj.register_forward_hook(make_hook(i, "k")))
        handles.append(layer.self_attn.v_proj.register_forward_hook(make_hook(i, "v")))
        handles.append(layer.self_attn.o_proj.register_forward_hook(make_hook(i, "o")))
        handles.append(layer.mlp.gate_proj.register_forward_hook(make_hook(i, "gate")))
        handles.append(layer.mlp.up_proj.register_forward_hook(make_hook(i, "up")))
        handles.append(layer.mlp.down_proj.register_forward_hook(make_hook(i, "down", is_last_hook=is_last_layer)))
    
    # 只有跑完整模型时才保存 logits
    if not early_stop:
        handles.append(model.lm_head.register_forward_hook(
            lambda m, inp, out: acts.__setitem__("logits", out.cpu().clone())))
    
    # Forward (使用 try-except 捕获早停)
    with torch.no_grad():
        try:
            model(input_ids, attention_mask=attention_mask)
        except EarlyStopException:
            pass  # 正常的早停
    
    for h in handles:
        h.remove()
    
    acts["input_ids"] = input_ids.cpu()
    acts["attention_mask"] = attention_mask.cpu()
    acts["prompt"] = prompt
    acts["seq_len"] = seq_len
    acts["layers"] = layers
    acts["mode"] = "fake_quant"
    
    torch.save(acts, out_path)
    print(f"Saved {len(acts)} tensors to {out_path}")


def save_resq_true(ckpt_a: str, out_path: str, 
                   prompt: str, device: str, layers_arg: str, seq_len: int):
    """保存 ResQ 模型的激活值 (真量化: forward 时 int32 matmul, 只需 CKPT_A)"""
    from .modeling_qwen3_resq_truequant import Qwen3ResQTrueQuantForCausalLM
    
    print(f"Loading ResQ model (true quant) from {ckpt_a}...")
    model = Qwen3ResQTrueQuantForCausalLM.from_resq_checkpoint(ckpt_a, device).eval()
    tokenizer = AutoTokenizer.from_pretrained(ckpt_a, trust_remote_code=True)
    
    # 解析 layers 参数
    num_layers = len(model.layers)
    if layers_arg == "all":
        layers = list(range(num_layers))
    else:
        layers = [int(x) for x in layers_arg.split(",")]
    max_layer = max(layers)
    early_stop = (max_layer < num_layers - 1)
    print(f"Saving activations for {len(layers)} layers (max_layer={max_layer}, early_stop={early_stop})")
    
    # 构造指定长度的输入
    text = prompt if prompt != "default" else DEFAULT_TEXT
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=seq_len, padding="max_length")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    print(f"Input sequence length: {input_ids.shape[1]} (actual tokens: {attention_mask.sum().item()})")
    acts = {}
    handles = []
    
    # 注册 hooks
    for i in layers:
        layer = model.layers[i]
        
        def make_hook(idx, name, is_last_hook=False):
            def hook(m, inp, out):
                key = f"L{idx}.{name}"
                acts[key] = (out[0] if isinstance(out, tuple) else out).cpu().clone()
                if is_last_hook:
                    raise EarlyStopException()
            return hook
        
        is_last_layer = (i == max_layer) and early_stop
        handles.append(layer.input_layernorm.register_forward_hook(make_hook(i, "input_ln")))
        handles.append(layer.self_attn.q_proj.register_forward_hook(make_hook(i, "q")))
        handles.append(layer.self_attn.k_proj.register_forward_hook(make_hook(i, "k")))
        handles.append(layer.self_attn.v_proj.register_forward_hook(make_hook(i, "v")))
        handles.append(layer.self_attn.o_proj.register_forward_hook(make_hook(i, "o")))
        handles.append(layer.mlp.gate_proj.register_forward_hook(make_hook(i, "gate")))
        handles.append(layer.mlp.up_proj.register_forward_hook(make_hook(i, "up")))
        handles.append(layer.mlp.down_proj.register_forward_hook(make_hook(i, "down", is_last_hook=is_last_layer)))
    
    # 只有跑完整模型时才保存 logits
    if not early_stop:
        handles.append(model.lm_head.register_forward_hook(
            lambda m, inp, out: acts.__setitem__("logits", out.cpu().clone())))
    
    # Forward (使用 try-except 捕获早停)
    with torch.no_grad():
        try:
            model(input_ids, attention_mask=attention_mask)
        except EarlyStopException:
            pass  # 正常的早停
    
    for h in handles:
        h.remove()
    
    acts["input_ids"] = input_ids.cpu()
    acts["attention_mask"] = attention_mask.cpu()
    acts["prompt"] = prompt
    acts["seq_len"] = seq_len
    acts["layers"] = layers
    acts["mode"] = "true_quant"
    
    torch.save(acts, out_path)
    print(f"Saved {len(acts)} tensors to {out_path}")


def load_rotation_matrices(ckpt_b_path: str):
    """从 CKPT_B 加载旋转矩阵
    
    CKPT_B 包含:
    - resq.layer.*.P_a, R_a → Ua = P_a @ R_a (hidden_size x hidden_size)
    - resq.layer.*.P_b, R_b → Ub = P_b @ R_b (per-kv-head: [num_kv_heads, head_dim, head_dim])
    - resq.layer.*.P_c, R_c → Uc = P_c @ R_c (online Q/K rotation)
    - resq.layer.*.P_d, R_d_hadK → Ud (online MLP rotation)
    """
    from pathlib import Path
    from safetensors import safe_open
    
    path = Path(ckpt_b_path)
    result = {}
    
    if path.is_dir():
        for f in path.glob("*.safetensors"):
            with safe_open(f, framework="pt", device="cpu") as sf:
                for key in sf.keys():
                    result[key] = sf.get_tensor(key)
        for f in path.glob("*.pt"):
            data = torch.load(f, map_location="cpu")
            if isinstance(data, dict):
                result.update(data)
    elif path.suffix == '.pt':
        result = torch.load(path, map_location="cpu")
    
    print(f"Loaded {len(result)} matrices from {ckpt_b_path}")
    # Print available keys
    sample_keys = [k for k in list(result.keys())[:10]]
    print(f"  Sample keys: {sample_keys}")
    
    # 预计算 Ua = P_a @ R_a, Ub = P_b @ R_b for each layer
    computed = {}
    for key in result.keys():
        if 'P_a' in key:
            layer_prefix = key.replace('P_a', '')
            r_key = key.replace('P_a', 'R_a')
            if r_key in result:
                ua_key = key.replace('P_a', 'Ua')
                computed[ua_key] = result[key].float() @ result[r_key].float()
        elif 'P_b' in key:
            layer_prefix = key.replace('P_b', '')
            r_key = key.replace('P_b', 'R_b')
            if r_key in result:
                ub_key = key.replace('P_b', 'Ub')
                # P_b: [num_kv_heads, head_dim, head_dim], R_b: [head_dim, head_dim]
                # Ub[h] = P_b[h] @ R_b
                p_b = result[key].float()  # [num_kv_heads, head_dim, head_dim]
                r_b = result[r_key].float()  # [head_dim, head_dim]
                computed[ub_key] = torch.matmul(p_b, r_b)  # [num_kv_heads, head_dim, head_dim]
    
    result.update(computed)
    return result


def compare(orig_path: str, resq_path: str, ckpt_b_path: str = None):
    """比较两个激活文件
    
    如果提供 ckpt_b_path，会使用其中的旋转矩阵对 ResQ 激活进行 un-rotate，
    使其可以与原始激活直接比较。
    
    旋转矩阵:
    - Ua: 应用于 hidden states (embed 之后)
    - Ub: 应用于 V projection 输出
    """
    orig = torch.load(orig_path, map_location="cpu")
    resq = torch.load(resq_path, map_location="cpu")
    
    # Load rotation matrices if provided
    rotations = load_rotation_matrices(ckpt_b_path) if ckpt_b_path else {}
    
    print(f"Prompt: {orig.get('prompt', '?')}")
    print(f"Mode: {resq.get('mode', 'unknown')}")
    if rotations:
        print(f"Un-rotate: enabled (using {len(rotations)} matrices)")
    print("=" * 60)
    
    def get_rotation_matrix(key: str, rotations):
        """获取对应的旋转矩阵
        
        返回 (U, rotation_type) 其中 rotation_type 描述旋转类型
        
        旋转空间分析:
        - q, k, gate, up: 权重已融合 Ua.T，输入 Ua 空间的 h 经过后输出在原始空间 → 不变点
        - v: 权重融合 Ua.T @ Ub，输出在 Ub 空间 → 需要 @ Ub.T
        - input_ln: 在 Ua 空间 → 需要 @ Ua.T
        - o: 输出给 residual 在 Ua 空间 → 需要 @ Ua.T
        - down: 输出给 residual 在 Ua 空间 → 需要 @ Ua.T
        """
        import re
        match = re.match(r'L(\d+)\.(\w+)', key)
        if not match:
            return None, None
        layer_idx, suffix = int(match.group(1)), match.group(2)
        
        # Map suffix to rotation matrix
        if suffix in ('input_ln', 'o', 'down'):
            # 这些输出在 Ua 空间，需要 @ Ua.T 变回原始空间
            ua_key = f'resq.layer.{layer_idx}.Ua'
            if ua_key in rotations:
                return rotations[ua_key], 'Ua'
        elif suffix == 'v':
            # v 输出在 Ub 空间
            ub_key = f'resq.layer.{layer_idx}.Ub'
            if ub_key in rotations:
                return rotations[ub_key], 'Ub'  # [num_kv_heads, head_dim, head_dim]
        # q, k, gate, up: 不变点，不需要旋转
        return None, None
    
    def apply_inverse_rotation(x: torch.Tensor, U: torch.Tensor, rot_type: str) -> torch.Tensor:
        """应用逆旋转: x @ U.T (正交矩阵的逆是其转置)
        
        rot_type:
        - 'Ua': U is [hidden, hidden], x is [batch, seq, hidden]
        - 'Ub': U is [num_kv_heads, head_dim, head_dim], x is [batch, seq, num_kv_heads * head_dim]
        """
        if U is None:
            return x
        
        if rot_type == 'Ua':
            # Simple case: x @ U.T
            if x.shape[-1] == U.shape[0]:
                return torch.matmul(x.float(), U.T.float()).to(x.dtype)
        elif rot_type == 'Ub':
            # Per-head rotation: U is [num_kv_heads, head_dim, head_dim]
            # x is [batch, seq, num_kv_heads * head_dim]
            num_kv_heads, head_dim, _ = U.shape
            if x.shape[-1] == num_kv_heads * head_dim:
                # Reshape to [batch, seq, num_kv_heads, head_dim]
                x_reshaped = x.view(*x.shape[:-1], num_kv_heads, head_dim).float()
                # Apply per-head rotation: x[..., h, :] @ U[h].T
                result = torch.zeros_like(x_reshaped)
                for h in range(num_kv_heads):
                    result[..., h, :] = torch.matmul(x_reshaped[..., h, :], U[h].T.float())
                return result.view(x.shape).to(x.dtype)
        
        return x  # Shape mismatch, skip
    
    # 如果有旋转矩阵，需要 un-rotate 的激活才能与原始比较
    # 不变点 (q, k, gate, up) 可以直接比较
    INVARIANT_KEYS = {'q', 'k', 'gate', 'up', 'logits'}
    ROTATED_KEYS = {'input_ln', 'v', 'o', 'down'}
    
    orig_keys = {k for k, v in orig.items() if isinstance(v, torch.Tensor)}
    resq_keys = {k for k, v in resq.items() if isinstance(v, torch.Tensor)}
    
    # 按层序排序: L0.xxx, L1.xxx, ..., L39.xxx, logits
    def layer_sort_key(key):
        import re
        match = re.match(r'L(\d+)\.(.+)', key)
        if match:
            return (0, int(match.group(1)), match.group(2))
        return (1, 0, key)  # 非层激活(如 logits)排在最后
    
    common = sorted(orig_keys & resq_keys, key=layer_sort_key)
    
    passed, failed, skipped = 0, 0, 0
    failures = []
    
    for key in common:
        o, r = orig[key].float(), resq[key].float()
        
        # 检查 NaN/Inf
        if torch.isnan(r).any() or torch.isinf(r).any():
            failed += 1
            print(f"✗ {key}: NaN/Inf detected in ResQ tensor!")
            failures.append((key, float('nan'), float('nan'), float('nan')))
            continue
        if torch.isnan(o).any() or torch.isinf(o).any():
            failed += 1
            print(f"✗ {key}: NaN/Inf detected in original tensor!")
            continue
        
        # 尝试 un-rotate
        key_suffix = key.split('.')[-1] if '.' in key else key
        U, rot_type = get_rotation_matrix(key, rotations) if rotations else (None, None)
        if U is not None:
            r_before = r.clone()
            r = apply_inverse_rotation(r, U, rot_type)
            unrotated = True
            # Debug: check un-rotation effect
            print(f"  [debug] {key}: before_norm={r_before.norm():.4f}, after_norm={r.norm():.4f}, U_shape={U.shape}")
        else:
            unrotated = False
        
        o_flat, r_flat = o.flatten(), r.flatten()
        
        # 确定激活类型
        is_invariant = key_suffix in INVARIANT_KEYS or unrotated
        is_rotated = key_suffix in ROTATED_KEYS and not unrotated
        
        # 计算指标
        o_norm = o_flat.norm().item()
        r_norm = r_flat.norm().item()
        norm_ratio = r_norm / (o_norm + 1e-8)
        
        # Debug: L0 层详细信息
        if 'L0.' in key:
            print(f"  [debug] {key}: orig_norm={o_norm:.4f}, resq_norm={r_norm:.4f}, shape={o.shape}")
        
        if is_invariant:
            # 旋转不变点：可以直接比较值
            rel_err = (o_flat - r_flat).norm().item() / (o_norm + 1e-8)
            corr = F.cosine_similarity(o_flat.unsqueeze(0), r_flat.unsqueeze(0)).item()
            ok = rel_err < 0.15 and corr > 0.95
            status = "inv"
        elif is_rotated:
            # 旋转空间：只比较 norm (正交旋转保持 norm)
            rel_err = abs(norm_ratio - 1.0)
            corr = float('nan')  # 不适用
            ok = rel_err < 0.1
            status = "rot"
        else:
            # 未分类：使用 norm 比较
            rel_err = abs(norm_ratio - 1.0)
            corr = float('nan')
            ok = rel_err < 0.1
            status = "???"
        
        if ok:
            passed += 1
            if is_invariant:
                print(f"✓ {key} [{status}]: corr={corr:.4f}, rel_err={rel_err:.4f}, norm_ratio={norm_ratio:.4f}")
            else:
                print(f"✓ {key} [{status}]: norm_ratio={norm_ratio:.4f}")
        else:
            failed += 1
            if is_invariant:
                print(f"✗ {key} [{status}]: corr={corr:.4f}, rel_err={rel_err:.4f}, norm_ratio={norm_ratio:.4f}")
            else:
                print(f"✗ {key} [{status}]: norm_ratio={norm_ratio:.4f} (expected ~1.0)")
            failures.append((key, corr, rel_err, norm_ratio))
    
    print("=" * 60)
    print(f"Passed: {passed}, Failed: {failed}")
    print("\n[inv] = 旋转不变点或已 un-rotate，可比较 rel_err/corr")
    print("[rot] = 旋转空间 (未提供 --ckpt-b)，只能比较 norm")
    if rotations:
        print("       (已提供旋转矩阵，rotated keys 自动 un-rotate 后按 [inv] 比较)")
    
    if failures:
        print("\nFirst failure details:")
        key = failures[0][0]
        if key in orig and key in resq:
            o, r = orig[key].float(), resq[key].float()
            print(f"  {key}: shape={o.shape}")
            print(f"  orig: mean={o.mean():.4f}, std={o.std():.4f}, norm={o.norm():.4f}")
            print(f"  resq: mean={r.mean():.4f}, std={r.std():.4f}, norm={r.norm():.4f}")


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="cmd", required=True)
    
    # save-orig
    p1 = subparsers.add_parser("save-orig")
    p1.add_argument("--model", required=True)
    p1.add_argument("--out", required=True)
    p1.add_argument("--prompt", default="default", help="Text input or 'default' for built-in text")
    p1.add_argument("--device", default="cpu")
    p1.add_argument("--layers", default="all", help="'all' or comma-separated indices like '0,31,63'")
    p1.add_argument("--seq-len", type=int, default=4, help="Input sequence length")
    
    # save-resq (伪量化)
    p2 = subparsers.add_parser("save-resq", help="保存 ResQ 激活 (伪量化: load 时 dequant)")
    p2.add_argument("--ckpt-a", required=True, help="ResQ checkpoint A (含 config.json、量化权重、Uc/Pd)")
    p2.add_argument("--out", required=True)
    p2.add_argument("--prompt", default="default", help="Text input or 'default' for built-in text")
    p2.add_argument("--device", default="cpu")
    p2.add_argument("--layers", default="all", help="'all' or comma-separated indices")
    p2.add_argument("--seq-len", type=int, default=4, help="Input sequence length")
    
    # save-resq-true (真量化)
    p2t = subparsers.add_parser("save-resq-true", help="保存 ResQ 激活 (真量化: forward 时 int32 matmul)")
    p2t.add_argument("--ckpt-a", required=True, help="ResQ checkpoint A (含 config.json、量化权重、Uc/Pd)")
    p2t.add_argument("--out", required=True)
    p2t.add_argument("--prompt", default="default", help="Text input or 'default' for built-in text")
    p2t.add_argument("--device", default="npu")
    p2t.add_argument("--layers", default="all", help="'all' or comma-separated indices")
    p2t.add_argument("--seq-len", type=int, default=4, help="Input sequence length")
    
    # compare
    p3 = subparsers.add_parser("compare")
    p3.add_argument("--orig", required=True)
    p3.add_argument("--resq", required=True)
    p3.add_argument("--ckpt-b", default=None, help="ResQ checkpoint B (含旋转矩阵，用于 un-rotate)")
    
    args = parser.parse_args()
    
    if args.cmd == "save-orig":
        save_original(args.model, args.out, args.prompt, args.device, args.layers, args.seq_len)
    elif args.cmd == "save-resq":
        save_resq(args.ckpt_a, args.out, args.prompt, args.device, args.layers, args.seq_len)
    elif args.cmd == "save-resq-true":
        save_resq_true(args.ckpt_a, args.out, args.prompt, args.device, args.layers, args.seq_len)
    elif args.cmd == "compare":
        compare(args.orig, args.resq, args.ckpt_b)


if __name__ == "__main__":
    main()
