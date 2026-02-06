"""
Activation comparison tool using wrapper approach.

Instead of requiring custom model definitions (Qwen3ResQForCausalLM),
this tool loads the standard HuggingFace model and replaces linear layers
with ResQ/W8A8 wrappers that load quantized weights from ckpt A.

Modes:
    - run:       One-shot: save-orig + save-resq + compare in memory
    - save-orig: Save original bf16 model activations
    - save-resq: Save ResQ model activations (wrapper, fake/true quant)
    - compare:   Compare two activation files

Usage:
    # All-in-one (recommended)
    python -m tools.resq_debug.debug_compare run \
        --model ${CKPT_O} --ckpt-a ${CKPT_A} --ckpt-b ${CKPT_B} \
        --layers 0,1 --device npu --true-quant

    # Step-by-step
    python -m tools.resq_debug.debug_compare save-orig \
        --model ${CKPT_O} --out orig.pt
    python -m tools.resq_debug.debug_compare save-resq \
        --model ${CKPT_O} --ckpt-a ${CKPT_A} --out resq.pt \
        --device npu --true-quant
    python -m tools.resq_debug.debug_compare compare \
        --orig orig.pt --resq resq.pt --ckpt-b ${CKPT_B}
"""

import argparse
import gc
import re
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from .wrappers import (
    ResQLinearTrueQuantWrapper,
    ResQLinearWrapper,
    W8A8LinearTrueQuantWrapper,
    W8A8LinearWrapper,
)


class EarlyStopException(Exception):
    pass


DEFAULT_TEXT = (
    "The quick brown fox jumps over the lazy dog. This is a sample text for "
    "testing language model activations. We need a reasonably long piece of "
    "text to fill the sequence length for proper comparison between the "
    "original and quantized models."
)


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------

def load_safetensors(path: str) -> Dict[str, torch.Tensor]:
    """Eagerly load all tensors (use for small checkpoints like ckpt B)."""
    from safetensors import safe_open

    weights: Dict[str, torch.Tensor] = {}
    p = Path(path)
    for f in sorted(p.glob("*.safetensors")):
        with safe_open(str(f), framework="pt", device="cpu") as sf:
            for key in sf.keys():
                weights[key] = sf.get_tensor(key)
    print(f"Loaded {len(weights)} tensors from {path}")
    return weights


class SafetensorsIndex:
    """Lazy tensor loader: indexes safetensors files, loads on demand.

    Avoids loading all tensors into memory at once, preventing OOM
    when the full checkpoint is too large for single-device memory.
    """

    def __init__(self, path: str):
        from safetensors import safe_open

        p = Path(path)
        self._index: Dict[str, str] = {}  # key -> filename
        for f in sorted(p.glob("*.safetensors")):
            with safe_open(str(f), framework="pt", device="cpu") as sf:
                for key in sf.keys():
                    self._index[key] = str(f)
        print(f"Indexed {len(self._index)} tensors from {path}")

    def keys(self):
        return self._index.keys()

    def __contains__(self, key: str) -> bool:
        return key in self._index

    def __getitem__(self, key: str) -> torch.Tensor:
        from safetensors import safe_open

        with safe_open(self._index[key], framework="pt", device="cpu") as sf:
            return sf.get_tensor(key)

    def get(self, key: str, default=None):
        if key in self._index:
            return self[key]
        return default


# ---------------------------------------------------------------------------
# Wrapper application
# ---------------------------------------------------------------------------

def apply_resq_wrappers(
    model: torch.nn.Module,
    ckpt_a_weights: Union[Dict[str, torch.Tensor], SafetensorsIndex],
    target_layers: Optional[List[int]] = None,
    true_quant: bool = False,
    apply_uc: bool = True,
    head_dim: int = 128,
    device: str = "npu",
) -> torch.nn.Module:
    """Replace linear layers in model with ResQ/W8A8 wrappers.

    Args:
        ckpt_a_weights: Dict or SafetensorsIndex (lazy loader) for ckpt A.
            SafetensorsIndex loads tensors on demand to avoid OOM.
        target_layers: Layer indices to replace. None = all layers.
            Typically range(max_layer + 1) when using early_stop.
        true_quant: Use NPU true-quant wrappers (requires NPU).
                    False = fake-quant (CPU/GPU compatible).
        device: Target device for fake-quant wrappers and Uc hooks.
    """
    num_layers = len(model.model.layers)
    if target_layers is None:
        target_layers = list(range(num_layers))

    # --- Load ALL non-quantized weights from ckpt A ---
    # This includes embed_tokens, lm_head, layernorms, q_norm, k_norm, etc.
    # Quantized linear layers are handled separately by wrappers below.
    QUANT_SUFFIXES = (
        '.weight_high', '.weight_low', '.scale_high', '.scale_low',
        '.high_fraction', '.weight_scale', '.weight_offset',
    )
    loaded_plain = 0
    for key in list(ckpt_a_weights.keys()):
        if key.startswith('resq.'):
            continue
        if any(key.endswith(s) for s in QUANT_SUFFIXES):
            continue
        # Navigate model to find the parameter
        try:
            parts = key.split('.')
            obj = model
            for p in parts[:-1]:
                if p.isdigit():
                    obj = obj[int(p)]
                else:
                    obj = getattr(obj, p)
            param = getattr(obj, parts[-1])
            if hasattr(param, 'data'):
                param.data = ckpt_a_weights[key].to(
                    device=param.device, dtype=param.dtype
                )
                loaded_plain += 1
        except (AttributeError, IndexError):
            pass
    print(f"  Loaded {loaded_plain} non-quantized weights from ckpt A "
          f"(embed, layernorm, etc.)")

    ResQCls = ResQLinearTrueQuantWrapper if true_quant else ResQLinearWrapper
    W8A8Cls = W8A8LinearTrueQuantWrapper if true_quant else W8A8LinearWrapper

    for i in target_layers:
        layer = model.model.layers[i]

        # --- Attention projections: ResQ ---
        for name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
            prefix = f'model.layers.{i}.self_attn.{name}'
            wrapper = ResQCls(
                weight_high=ckpt_a_weights[f'{prefix}.weight_high'],
                weight_low=ckpt_a_weights[f'{prefix}.weight_low'],
                scale_high=ckpt_a_weights[f'{prefix}.scale_high'],
                scale_low=ckpt_a_weights[f'{prefix}.scale_low'],
                high_fraction=ckpt_a_weights[f'{prefix}.high_fraction'],
                is_o_proj=(name == 'o_proj'),
                head_dim=head_dim,
            )
            if not true_quant:
                wrapper = wrapper.to(device)
            setattr(layer.self_attn, name, wrapper)

        # --- MLP gate/up: ResQ ---
        for name in ['gate_proj', 'up_proj']:
            prefix = f'model.layers.{i}.mlp.{name}'
            wrapper = ResQCls(
                weight_high=ckpt_a_weights[f'{prefix}.weight_high'],
                weight_low=ckpt_a_weights[f'{prefix}.weight_low'],
                scale_high=ckpt_a_weights[f'{prefix}.scale_high'],
                scale_low=ckpt_a_weights[f'{prefix}.scale_low'],
                high_fraction=ckpt_a_weights[f'{prefix}.high_fraction'],
            )
            if not true_quant:
                wrapper = wrapper.to(device)
            setattr(layer.mlp, name, wrapper)

        # --- MLP down_proj: W8A8 ---
        prefix = f'model.layers.{i}.mlp.down_proj'
        wrapper = W8A8Cls(
            weight=ckpt_a_weights[f'{prefix}.weight'],
            weight_scale=ckpt_a_weights[f'{prefix}.weight_scale'],
            weight_offset=ckpt_a_weights[f'{prefix}.weight_offset'],
            rotation_Pd=ckpt_a_weights.get(f'resq.layer.{i}.Pd'),
            rotation_Hd=ckpt_a_weights.get('resq.Hd'),
        )
        if not true_quant:
            wrapper = wrapper.to(device)
        setattr(layer.mlp, 'down_proj', wrapper)

        # --- Uc rotation hooks on q_norm / k_norm ---
        if apply_uc:
            Uc_key = f'resq.layer.{i}.Uc'
            if Uc_key in ckpt_a_weights:
                Uc = ckpt_a_weights[Uc_key].float().to(device)

                def make_uc_hook(Uc_mat):
                    def hook(module, input, output):
                        return torch.matmul(
                            output.float(), Uc_mat.T
                        ).to(output.dtype)
                    return hook

                layer.self_attn.q_norm.register_forward_hook(
                    make_uc_hook(Uc)
                )
                layer.self_attn.k_norm.register_forward_hook(
                    make_uc_hook(Uc)
                )

        # Free CPU tensors loaded for this layer
        gc.collect()

        if (len(target_layers) > 1 and
                (target_layers.index(i) + 1) % 10 == 0) or i == target_layers[-1]:
            print(f"  Replaced layer {i + 1}/{num_layers}")

    mode = "true_quant" if true_quant else "fake_quant"
    print(f"Replaced {len(target_layers)}/{num_layers} layers with {mode} "
          f"wrappers (Uc rotation: {apply_uc})")
    return model


# ---------------------------------------------------------------------------
# Activation capture
# ---------------------------------------------------------------------------

def register_activation_hooks(
    model: torch.nn.Module,
    layers: List[int],
    acts: Dict[str, torch.Tensor],
    early_stop: bool,
    max_layer: int,
) -> list:
    handles = []

    def make_hook(name, is_last_hook=False):
        def hook(m, inp, out):
            t = out[0] if isinstance(out, tuple) else out
            if hasattr(t, 'device') and t.device.type == 'npu':
                import torch_npu
                torch_npu.npu.synchronize()
            acts[name] = t.detach().cpu().clone()
            if is_last_hook:
                raise EarlyStopException()
        return hook

    for i in layers:
        layer = model.model.layers[i]
        is_last_layer = (i == max_layer) and early_stop

        handles.append(layer.input_layernorm.register_forward_hook(
            make_hook(f"L{i}.input_ln")))
        handles.append(layer.self_attn.q_proj.register_forward_hook(
            make_hook(f"L{i}.q")))
        handles.append(layer.self_attn.k_proj.register_forward_hook(
            make_hook(f"L{i}.k")))
        handles.append(layer.self_attn.v_proj.register_forward_hook(
            make_hook(f"L{i}.v")))
        handles.append(layer.self_attn.o_proj.register_forward_hook(
            make_hook(f"L{i}.o")))
        handles.append(layer.mlp.gate_proj.register_forward_hook(
            make_hook(f"L{i}.gate")))
        handles.append(layer.mlp.up_proj.register_forward_hook(
            make_hook(f"L{i}.up")))
        handles.append(layer.mlp.down_proj.register_forward_hook(
            make_hook(f"L{i}.down", is_last_hook=is_last_layer)))

    if not early_stop:
        handles.append(model.lm_head.register_forward_hook(
            lambda m, inp, out: acts.__setitem__(
                "logits", out.cpu().clone())))

    return handles


def parse_layers(layers_arg: str, num_layers: int) -> List[int]:
    if layers_arg == "all":
        return list(range(num_layers))
    return [int(x) for x in layers_arg.split(",")]


def _move_to_device(model, device: str, max_layer: int, num_layers: int):
    """Move only embedding + layers[0..max_layer] to device.

    Keeps remaining layers on CPU to avoid OOM when the full model
    doesn't fit on a single device. Works with early_stop which
    skips layers beyond max_layer.
    """
    model.model.embed_tokens = model.model.embed_tokens.to(device)
    if hasattr(model.model, 'rotary_emb'):
        model.model.rotary_emb = model.model.rotary_emb.to(device)
    for i in range(max_layer + 1):
        model.model.layers[i] = model.model.layers[i].to(device)
    # If all layers needed, also move final norm and lm_head
    if max_layer >= num_layers - 1:
        if hasattr(model.model, 'norm'):
            model.model.norm = model.model.norm.to(device)
        if hasattr(model, 'lm_head'):
            model.lm_head = model.lm_head.to(device)
    moved = max_layer + 1
    print(f"Moved embed + {moved}/{num_layers} layers to {device}")


def _run_forward(model, input_ids, attention_mask, layers, early_stop, max_layer):
    """Run forward pass and return captured activations."""
    acts: Dict[str, torch.Tensor] = {}
    handles = register_activation_hooks(
        model, layers, acts, early_stop, max_layer
    )
    with torch.no_grad():
        try:
            model(input_ids, attention_mask=attention_mask)
        except EarlyStopException:
            pass
    for h in handles:
        h.remove()
    return acts


# ---------------------------------------------------------------------------
# save-orig
# ---------------------------------------------------------------------------

def save_original(
    model_path: str, out_path: str, prompt: str,
    device: str, layers_arg: str, seq_len: int,
):
    print(f"Loading original model from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True
    )

    num_layers = len(model.model.layers)
    layers = parse_layers(layers_arg, num_layers)
    max_layer = max(layers)
    early_stop = (max_layer < num_layers - 1)

    _move_to_device(model, device, max_layer, num_layers)
    print(f"Capturing {len(layers)} layers "
          f"(max_layer={max_layer}, early_stop={early_stop})")

    text = prompt if prompt != "default" else DEFAULT_TEXT
    inputs = tokenizer(
        text, return_tensors="pt",
        truncation=True, max_length=seq_len, padding="max_length",
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    print(f"Input seq_len={input_ids.shape[1]}, "
          f"actual_tokens={attention_mask.sum().item()}")

    acts = _run_forward(model, input_ids, attention_mask,
                        layers, early_stop, max_layer)

    acts["input_ids"] = input_ids.cpu()
    acts["attention_mask"] = attention_mask.cpu()
    acts["prompt"] = prompt
    acts["seq_len"] = seq_len
    acts["layers"] = layers

    torch.save(acts, out_path)
    print(f"Saved {len(acts)} entries to {out_path}")


# ---------------------------------------------------------------------------
# save-resq
# ---------------------------------------------------------------------------

def save_resq(
    model_path: str, ckpt_a_path: str, out_path: str, prompt: str,
    device: str, layers_arg: str, seq_len: int,
    true_quant: bool = False, apply_uc: bool = True, head_dim: int = 128,
):
    print(f"Loading base model from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True
    )

    num_layers = len(model.model.layers)
    layers = parse_layers(layers_arg, num_layers)
    max_layer = max(layers)
    early_stop = (max_layer < num_layers - 1)

    _move_to_device(model, device, max_layer, num_layers)

    print(f"Loading ckpt A (lazy) from {ckpt_a_path}...")
    ckpt_a_weights = SafetensorsIndex(ckpt_a_path)

    model = apply_resq_wrappers(
        model, ckpt_a_weights,
        target_layers=list(range(max_layer + 1)),
        true_quant=true_quant, apply_uc=apply_uc, head_dim=head_dim,
        device=device,
    )

    print(f"Capturing {len(layers)} layers "
          f"(max_layer={max_layer}, early_stop={early_stop})")

    text = prompt if prompt != "default" else DEFAULT_TEXT
    inputs = tokenizer(
        text, return_tensors="pt",
        truncation=True, max_length=seq_len, padding="max_length",
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    print(f"Input seq_len={input_ids.shape[1]}, "
          f"actual_tokens={attention_mask.sum().item()}")

    acts = _run_forward(model, input_ids, attention_mask,
                        layers, early_stop, max_layer)

    acts["input_ids"] = input_ids.cpu()
    acts["attention_mask"] = attention_mask.cpu()
    acts["prompt"] = prompt
    acts["seq_len"] = seq_len
    acts["layers"] = layers
    acts["mode"] = "true_quant" if true_quant else "fake_quant"

    torch.save(acts, out_path)
    print(f"Saved {len(acts)} entries to {out_path}")


# ---------------------------------------------------------------------------
# run (all-in-one)
# ---------------------------------------------------------------------------

def run_all(
    model_path: str, ckpt_a_path: str,
    ckpt_b_path: Optional[str], prompt: str,
    device: str, layers_arg: str, seq_len: int,
    true_quant: bool = False, apply_uc: bool = True, head_dim: int = 128,
    save_orig_path: Optional[str] = None,
    save_resq_path: Optional[str] = None,
):
    """All-in-one: load model once, capture orig + resq activations, compare."""

    # --- Load model ---
    print(f"Loading model from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True
    )

    num_layers = len(model.model.layers)
    layers = parse_layers(layers_arg, num_layers)
    max_layer = max(layers)
    early_stop = (max_layer < num_layers - 1)

    _move_to_device(model, device, max_layer, num_layers)
    print(f"Layers: {layers} (early_stop={early_stop})")

    # --- Prepare input ---
    text = prompt if prompt != "default" else DEFAULT_TEXT
    inputs = tokenizer(
        text, return_tensors="pt",
        truncation=True, max_length=seq_len, padding="max_length",
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    print(f"Input seq_len={input_ids.shape[1]}, "
          f"actual_tokens={attention_mask.sum().item()}")

    # --- Step 1: Original activations ---
    print("\n--- Step 1: Original model forward ---")
    orig_acts = _run_forward(model, input_ids, attention_mask,
                             layers, early_stop, max_layer)
    print(f"Captured {len(orig_acts)} original activations")

    if save_orig_path:
        orig_save = dict(orig_acts)
        orig_save.update(input_ids=input_ids.cpu(),
                         attention_mask=attention_mask.cpu(),
                         prompt=prompt, seq_len=seq_len, layers=layers)
        torch.save(orig_save, save_orig_path)
        print(f"Saved to {save_orig_path}")

    # --- Step 2: Replace with wrappers, ResQ forward ---
    print("\n--- Step 2: ResQ wrapper forward ---")
    print(f"Loading ckpt A (lazy) from {ckpt_a_path}...")
    ckpt_a_weights = SafetensorsIndex(ckpt_a_path)

    model = apply_resq_wrappers(
        model, ckpt_a_weights,
        target_layers=list(range(max_layer + 1)),
        true_quant=true_quant, apply_uc=apply_uc, head_dim=head_dim,
        device=device,
    )
    del ckpt_a_weights
    gc.collect()

    resq_acts = _run_forward(model, input_ids, attention_mask,
                             layers, early_stop, max_layer)
    print(f"Captured {len(resq_acts)} ResQ activations")

    if save_resq_path:
        resq_save = dict(resq_acts)
        mode = "true_quant" if true_quant else "fake_quant"
        resq_save.update(input_ids=input_ids.cpu(),
                         attention_mask=attention_mask.cpu(),
                         prompt=prompt, seq_len=seq_len, layers=layers,
                         mode=mode)
        torch.save(resq_save, save_resq_path)
        print(f"Saved to {save_resq_path}")

    # --- Step 3: Compare ---
    print("\n--- Step 3: Compare ---")
    rotations = load_rotation_matrices(ckpt_b_path) if ckpt_b_path else {}
    mode = "true_quant" if true_quant else "fake_quant"
    compare_acts(orig_acts, resq_acts, rotations, mode=mode)


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------

def load_rotation_matrices(ckpt_b_path: str) -> Dict[str, torch.Tensor]:
    """Load rotation matrices from ckpt B; pre-compute Ua, Ub."""
    from safetensors import safe_open

    path = Path(ckpt_b_path)
    result: Dict[str, torch.Tensor] = {}

    if path.is_dir():
        for f in path.glob("*.safetensors"):
            with safe_open(str(f), framework="pt", device="cpu") as sf:
                for key in sf.keys():
                    result[key] = sf.get_tensor(key)
        for f in path.glob("*.pt"):
            data = torch.load(f, map_location="cpu", weights_only=True)
            if isinstance(data, dict):
                result.update(data)
    elif path.suffix == '.pt':
        result = torch.load(path, map_location="cpu", weights_only=True)

    print(f"Loaded {len(result)} matrices from {ckpt_b_path}")

    computed: Dict[str, torch.Tensor] = {}
    for key in list(result.keys()):
        if 'P_a' in key:
            r_key = key.replace('P_a', 'R_a')
            if r_key in result:
                computed[key.replace('P_a', 'Ua')] = (
                    result[key].float() @ result[r_key].float()
                )
        elif 'P_b' in key:
            r_key = key.replace('P_b', 'R_b')
            if r_key in result:
                computed[key.replace('P_b', 'Ub')] = torch.matmul(
                    result[key].float(), result[r_key].float()
                )

    result.update(computed)
    return result


def _get_rotation(key: str, rotations: Dict[str, torch.Tensor]):
    match = re.match(r'L(\d+)\.(\w+)', key)
    if not match:
        return None, None
    layer_idx, suffix = int(match.group(1)), match.group(2)

    if suffix in ('input_ln', 'o', 'down'):
        ua_key = f'resq.layer.{layer_idx}.Ua'
        if ua_key in rotations:
            return rotations[ua_key], 'Ua'
    elif suffix == 'v':
        ub_key = f'resq.layer.{layer_idx}.Ub'
        if ub_key in rotations:
            return rotations[ub_key], 'Ub'
    return None, None


def _apply_inv_rotation(x: torch.Tensor, U: torch.Tensor, rot_type: str):
    if rot_type == 'Ua':
        if x.shape[-1] == U.shape[0]:
            return torch.matmul(x.float(), U.T.float()).to(x.dtype)
    elif rot_type == 'Ub':
        nkv, hd, _ = U.shape
        if x.shape[-1] == nkv * hd:
            x_r = x.view(*x.shape[:-1], nkv, hd).float()
            res = torch.zeros_like(x_r)
            for h in range(nkv):
                res[..., h, :] = torch.matmul(x_r[..., h, :], U[h].T)
            return res.view(x.shape).to(x.dtype)
    return x


def compare_acts(
    orig: Dict[str, torch.Tensor],
    resq: Dict[str, torch.Tensor],
    rotations: Dict[str, torch.Tensor],
    mode: str = "unknown",
):
    """Compare two activation dicts."""
    print(f"Mode: {mode}")
    if rotations:
        print(f"Un-rotate: enabled ({len(rotations)} matrices)")
    print("=" * 70)

    INVARIANT = {'q', 'k', 'gate', 'up', 'logits'}
    ROTATED = {'input_ln', 'v', 'o', 'down'}

    orig_keys = {k for k, v in orig.items() if isinstance(v, torch.Tensor)}
    resq_keys = {k for k, v in resq.items() if isinstance(v, torch.Tensor)}

    def sort_key(key):
        m = re.match(r'L(\d+)\.(.+)', key)
        return (0, int(m.group(1)), m.group(2)) if m else (1, 0, key)

    common = sorted(orig_keys & resq_keys, key=sort_key)

    passed, failed = 0, 0
    failures = []

    for key in common:
        o, r = orig[key].float(), resq[key].float()

        nan_r = torch.isnan(r)
        inf_r = torch.isinf(r)
        if nan_r.any() or inf_r.any():
            failed += 1
            print(f"X {key}: {nan_r.sum().item()} NaNs, "
                  f"{inf_r.sum().item()} Infs in ResQ")
            failures.append((key, float('nan'), float('nan'), float('nan')))
            continue
        if torch.isnan(o).any() or torch.isinf(o).any():
            failed += 1
            print(f"X {key}: NaN/Inf in original")
            continue

        suffix = key.split('.')[-1] if '.' in key else key
        U, rt = _get_rotation(key, rotations) if rotations else (None, None)
        if U is not None:
            r = _apply_inv_rotation(r, U, rt)
            unrotated = True
        else:
            unrotated = False

        of, rf = o.flatten(), r.flatten()
        o_norm = of.norm().item()
        r_norm = rf.norm().item()
        nr = r_norm / (o_norm + 1e-8)

        is_inv = suffix in INVARIANT or unrotated
        is_rot = suffix in ROTATED and not unrotated

        if is_inv:
            rel = (of - rf).norm().item() / (o_norm + 1e-8)
            corr = F.cosine_similarity(of.unsqueeze(0), rf.unsqueeze(0)).item()
            ok = rel < 0.15 and corr > 0.95
            tag = "inv"
        elif is_rot:
            rel = abs(nr - 1.0)
            corr = float('nan')
            ok = rel < 0.1
            tag = "rot"
        else:
            rel = abs(nr - 1.0)
            corr = float('nan')
            ok = rel < 0.1
            tag = "???"

        if ok:
            passed += 1
            if is_inv:
                print(f"  {key} [{tag}]: corr={corr:.4f}, "
                      f"rel_err={rel:.4f}, norm_ratio={nr:.4f}")
            else:
                print(f"  {key} [{tag}]: norm_ratio={nr:.4f}")
        else:
            failed += 1
            if is_inv:
                print(f"X {key} [{tag}]: corr={corr:.4f}, "
                      f"rel_err={rel:.4f}, norm_ratio={nr:.4f}")
            else:
                print(f"X {key} [{tag}]: norm_ratio={nr:.4f} (expected ~1.0)")
            failures.append((key, corr, rel, nr))

    print("=" * 70)
    print(f"Passed: {passed}, Failed: {failed}")
    print("\n[inv] = invariant / un-rotated  [rot] = rotated (norm-only)")

    if failures:
        key = failures[0][0]
        if key in orig and key in resq:
            o, r = orig[key].float(), resq[key].float()
            print(f"\nFirst failure: {key}  shape={o.shape}")
            print(f"  orig: mean={o.mean():.4f} std={o.std():.4f} "
                  f"norm={o.norm():.4f}")
            print(f"  resq: mean={r.mean():.4f} std={r.std():.4f} "
                  f"norm={r.norm():.4f}")
            ratio = r.norm() / (o.norm() + 1e-8)
            if ratio > 100 or ratio < 0.01:
                print("  [hint] Norm scale mismatch")


def compare(
    orig_path: str, resq_path: str,
    ckpt_b_path: Optional[str] = None,
):
    orig = torch.load(orig_path, map_location="cpu", weights_only=False)
    resq = torch.load(resq_path, map_location="cpu", weights_only=False)
    rotations = load_rotation_matrices(ckpt_b_path) if ckpt_b_path else {}
    print(f"Prompt: {orig.get('prompt', '?')}")
    compare_acts(orig, resq, rotations, mode=resq.get('mode', 'unknown'))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _add_common_args(p):
    p.add_argument("--prompt", default="default")
    p.add_argument("--layers", default="all",
                   help="'all' or comma-separated: '0,31,63'")
    p.add_argument("--seq-len", type=int, default=4)
    p.add_argument("--head-dim", type=int, default=128)


def main():
    parser = argparse.ArgumentParser(
        description="ResQ activation comparison tool (wrapper approach)"
    )
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    # --- run (all-in-one) ---
    p0 = subparsers.add_parser("run", help="All-in-one: orig + resq + compare")
    p0.add_argument("--model", required=True)
    p0.add_argument("--ckpt-a", required=True)
    p0.add_argument("--ckpt-b", default=None)
    p0.add_argument("--device", default="npu")
    p0.add_argument("--true-quant", action="store_true",
                    help="Use NPU true-quant (default: fake-quant)")
    p0.add_argument("--no-uc", action="store_true")
    p0.add_argument("--save-orig", default=None,
                    help="Optionally save orig activations to file")
    p0.add_argument("--save-resq", default=None,
                    help="Optionally save resq activations to file")
    _add_common_args(p0)

    # --- save-orig ---
    p1 = subparsers.add_parser("save-orig")
    p1.add_argument("--model", required=True)
    p1.add_argument("--out", required=True)
    p1.add_argument("--device", default="cpu")
    _add_common_args(p1)

    # --- save-resq ---
    p2 = subparsers.add_parser("save-resq")
    p2.add_argument("--model", required=True)
    p2.add_argument("--ckpt-a", required=True)
    p2.add_argument("--out", required=True)
    p2.add_argument("--device", default="cpu")
    p2.add_argument("--true-quant", action="store_true")
    p2.add_argument("--no-uc", action="store_true")
    _add_common_args(p2)

    # --- compare ---
    p3 = subparsers.add_parser("compare")
    p3.add_argument("--orig", required=True)
    p3.add_argument("--resq", required=True)
    p3.add_argument("--ckpt-b", default=None)

    args = parser.parse_args()

    if args.cmd == "run":
        run_all(
            args.model, args.ckpt_a, args.ckpt_b, args.prompt,
            args.device, args.layers, args.seq_len,
            true_quant=args.true_quant, apply_uc=not args.no_uc,
            head_dim=args.head_dim,
            save_orig_path=args.save_orig,
            save_resq_path=args.save_resq,
        )
    elif args.cmd == "save-orig":
        save_original(
            args.model, args.out, args.prompt,
            args.device, args.layers, args.seq_len,
        )
    elif args.cmd == "save-resq":
        save_resq(
            args.model, args.ckpt_a, args.out, args.prompt,
            args.device, args.layers, args.seq_len,
            true_quant=args.true_quant, apply_uc=not args.no_uc,
            head_dim=args.head_dim,
        )
    elif args.cmd == "compare":
        compare(args.orig, args.resq, args.ckpt_b)


if __name__ == "__main__":
    main()
