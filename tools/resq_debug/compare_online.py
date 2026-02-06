#!/usr/bin/env python3
"""
Compare online vllm-ascend activations with offline wrapper replay.

The online vllm-ascend inference saves per-layer activations to .pt files
(input/output of each projection). This script loads those files, replays
the same input through offline wrappers (using ckpt A weights), and compares.

Key comparison: online uses FUSED qkv_proj / gate_up_proj, while offline
uses separate q/k/v and gate/up projections. If the fused weight merging
is wrong, this comparison will catch it.

Usage:
    # 1. Run vllm-ascend (activations auto-saved)
    # 2. Compare all saved layers:
    python -m tools.resq_debug.compare_online \
        --ckpt-a /path/to/ckpt_a \
        --diag-dir /tmp/resq_online_acts

    # Compare specific layer:
    python -m tools.resq_debug.compare_online \
        --ckpt-a /path/to/ckpt_a \
        --diag-dir /tmp/resq_online_acts \
        --layers 0,31
"""

import argparse
import re
from pathlib import Path
from typing import Set

import torch
import torch.nn.functional as F

from .debug_compare import SafetensorsIndex
from .wrappers import ResQLinearWrapper, W8A8LinearWrapper


def compare_tensor(name: str, online: torch.Tensor, offline: torch.Tensor) -> bool:
    """Compare two tensors and print diagnostics."""
    o = online.float().flatten()
    r = offline.float().flatten()

    if o.numel() != r.numel():
        print(f"X {name}: shape mismatch online={online.shape} offline={offline.shape}")
        return False

    corr = F.cosine_similarity(o.unsqueeze(0), r.unsqueeze(0)).item()
    o_norm = o.norm().item()
    r_norm = r.norm().item()
    norm_ratio = r_norm / (o_norm + 1e-8)
    max_diff = (o - r).abs().max().item()
    rel_err = (o - r).norm().item() / (o_norm + 1e-8)

    ok = corr > 0.99 and rel_err < 0.15
    mark = "  " if ok else "X "
    print(f"{mark}{name}: corr={corr:.6f} rel_err={rel_err:.4f} "
          f"norm_ratio={norm_ratio:.4f} max_diff={max_diff:.4f}")

    if not ok:
        print(f"    online:  norm={o_norm:.4f} first5={online.flatten()[:5].tolist()}")
        print(f"    offline: norm={r_norm:.4f} first5={offline.flatten()[:5].tolist()}")

    return ok


def compare_fused_resq(
    name: str,
    online_input: torch.Tensor,
    online_output: torch.Tensor,
    ckpt: SafetensorsIndex,
    layer_idx: int,
    proj_names: list,
    head_dim: int,
) -> bool:
    """Compare fused online output vs concatenated separate offline outputs."""

    parts = []
    for pname in proj_names:
        prefix = f'model.layers.{layer_idx}.self_attn.{pname}' \
            if pname in ('q_proj', 'k_proj', 'v_proj', 'o_proj') \
            else f'model.layers.{layer_idx}.mlp.{pname}'

        wrapper = ResQLinearWrapper(
            weight_high=ckpt[f'{prefix}.weight_high'],
            weight_low=ckpt[f'{prefix}.weight_low'],
            scale_high=ckpt[f'{prefix}.scale_high'],
            scale_low=ckpt[f'{prefix}.scale_low'],
            high_fraction=ckpt[f'{prefix}.high_fraction'],
            is_o_proj=(pname == 'o_proj'),
            head_dim=head_dim,
        )

        x = online_input.to(wrapper.weight.dtype)
        with torch.no_grad():
            part = wrapper(x)
        parts.append(part)

    offline_output = torch.cat(parts, dim=-1)
    return compare_tensor(name, online_output, offline_output)


def compare_single_resq(
    name: str,
    online_input: torch.Tensor,
    online_output: torch.Tensor,
    ckpt: SafetensorsIndex,
    prefix: str,
    is_o_proj: bool,
    head_dim: int,
) -> bool:
    """Compare single (non-fused) ResQ projection."""
    wrapper = ResQLinearWrapper(
        weight_high=ckpt[f'{prefix}.weight_high'],
        weight_low=ckpt[f'{prefix}.weight_low'],
        scale_high=ckpt[f'{prefix}.scale_high'],
        scale_low=ckpt[f'{prefix}.scale_low'],
        high_fraction=ckpt[f'{prefix}.high_fraction'],
        is_o_proj=is_o_proj,
        head_dim=head_dim,
    )

    x = online_input.to(wrapper.weight.dtype)
    with torch.no_grad():
        offline_output = wrapper(x)

    return compare_tensor(name, online_output, offline_output)


def compare_w8a8(
    name: str,
    online_input: torch.Tensor,
    online_output: torch.Tensor,
    ckpt: SafetensorsIndex,
    prefix: str,
    layer_idx: int,
) -> bool:
    """Compare W8A8 down_proj."""
    wrapper = W8A8LinearWrapper(
        weight=ckpt[f'{prefix}.weight'],
        weight_scale=ckpt[f'{prefix}.weight_scale'],
        weight_offset=ckpt[f'{prefix}.weight_offset'],
        rotation_Pd=ckpt.get(f'resq.layer.{layer_idx}.Pd'),
        rotation_Hd=ckpt.get('resq.Hd'),
    )

    x = online_input.to(wrapper.weight.dtype)
    with torch.no_grad():
        offline_output = wrapper(x)

    return compare_tensor(name, online_output, offline_output)


def check_mlp_activation_chain(
    layer: int,
    diag_dir: Path,
) -> bool:
    """Verify down_proj.input == silu(gate) * up from gate_up_proj.output.

    This checks the non-linear MLP activation between gate_up_proj and down_proj.
    """
    gate_up_file = diag_dir / f"model_layers_{layer}_mlp_gate_up_proj.pt"
    down_file = diag_dir / f"model_layers_{layer}_mlp_down_proj.pt"

    if not gate_up_file.exists() or not down_file.exists():
        print(f"\n  (skipped MLP chain check — need both gate_up and down files)")
        return True  # skip, not a failure

    gate_up_data = torch.load(gate_up_file, map_location="cpu", weights_only=True)
    down_data = torch.load(down_file, map_location="cpu", weights_only=True)

    gate_up_out = gate_up_data['output'].float()
    down_in = down_data['input'].float()

    # gate_up_out = [gate | up], split in half
    half = gate_up_out.shape[-1] // 2
    gate_out = gate_up_out[..., :half]
    up_out = gate_up_out[..., half:]

    # Expected: silu(gate) * up
    expected_down_in = F.silu(gate_out) * up_out

    print(f"\n  MLP chain: gate_up_out={gate_up_out.shape} → silu(gate)*up → down_in={down_in.shape}")
    return compare_tensor(f"L{layer}.MLP chain (silu(gate)*up vs down_proj.input)",
                          down_in, expected_down_in)


def check_embedding_chain(
    diag_dir: Path,
    ckpt: SafetensorsIndex,
) -> bool:
    """Verify layer 0 qkv_proj.input == RMSNorm(embed_tokens(input_ids), weight=layernorm.weight).

    This checks the embedding → input_layernorm → qkv_proj.input chain.
    """
    ids_file = diag_dir / "input_ids.pt"
    qkv_file = diag_dir / "model_layers_0_self_attn_qkv_proj.pt"

    if not ids_file.exists():
        print("\n(skipped embedding chain check — input_ids.pt not found, re-run run_online.py)")
        return True
    if not qkv_file.exists():
        print("\n(skipped embedding chain check — layer 0 qkv_proj not saved)")
        return True

    ids_data = torch.load(ids_file, map_location="cpu", weights_only=True)
    qkv_data = torch.load(qkv_file, map_location="cpu", weights_only=True)

    input_ids = ids_data['input_ids']
    online_qkv_input = qkv_data['input']  # shape: (seq_len, hidden_size)

    # Load embed_tokens weight
    embed_w = ckpt.get('model.embed_tokens.weight')
    if embed_w is None:
        print("\n  SKIP embedding chain — model.embed_tokens.weight not in ckpt")
        return True

    # Compute embedding
    embeddings = embed_w[input_ids].float()  # (seq_len, hidden_size)

    # Load input_layernorm weight (should be all-1s for ResQ)
    ln_w = ckpt.get('model.layers.0.input_layernorm.weight')
    if ln_w is not None:
        ln_weight = ln_w.float()
    else:
        ln_weight = torch.ones(embeddings.shape[-1])

    # Apply RMSNorm: y = x / rms(x) * weight
    eps = 1e-6  # Qwen3 default
    variance = embeddings.pow(2).mean(-1, keepdim=True)
    normed = embeddings * torch.rsqrt(variance + eps)
    expected_qkv_input = normed * ln_weight

    print(f"\n{'=' * 70}")
    print(f"Embedding chain check:")
    print(f"  input_ids: {input_ids.tolist()}")
    print(f"  embed_tokens.weight: {embed_w.shape}")
    print(f"  input_layernorm.weight: norm={ln_weight.norm().item():.4f} "
          f"min={ln_weight.min().item():.4f} max={ln_weight.max().item():.4f}")
    print(f"  expected shape: {expected_qkv_input.shape}, online shape: {online_qkv_input.shape}")

    return compare_tensor("Embedding chain (embed→RMSNorm→qkv_input)",
                          online_qkv_input.float(), expected_qkv_input)


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Apply RMSNorm: y = x / rms(x) * weight."""
    variance = x.float().pow(2).mean(-1, keepdim=True)
    normed = x.float() * torch.rsqrt(variance + eps)
    return normed * weight.float()


def check_attention_residual_chain(
    diag_dir: Path,
    ckpt: SafetensorsIndex,
) -> bool:
    """Verify gate_up_proj.input == post_attn_layernorm(embed(input_ids) + o_proj.output).

    For layer 0, the residual before attention is just embed_tokens(input_ids).
    After attention + residual: hidden = embed(ids) + o_proj.output.
    Then: gate_up_proj.input = post_attn_layernorm(hidden).
    """
    ids_file = diag_dir / "input_ids.pt"
    o_file = diag_dir / "model_layers_0_self_attn_o_proj.pt"
    gate_up_file = diag_dir / "model_layers_0_mlp_gate_up_proj.pt"

    if not all(f.exists() for f in [ids_file, o_file, gate_up_file]):
        print("\n(skipped attention+residual chain — need input_ids.pt, layer 0 o_proj, gate_up_proj)")
        return True

    ids_data = torch.load(ids_file, map_location="cpu", weights_only=True)
    o_data = torch.load(o_file, map_location="cpu", weights_only=True)
    gate_up_data = torch.load(gate_up_file, map_location="cpu", weights_only=True)

    input_ids = ids_data['input_ids']
    o_proj_output = o_data['output'].float()
    online_gate_up_input = gate_up_data['input'].float()

    # residual = embed_tokens(input_ids)
    embed_w = ckpt.get('model.embed_tokens.weight')
    if embed_w is None:
        print("\n  SKIP — model.embed_tokens.weight not in ckpt")
        return True
    residual = embed_w[input_ids].float()

    # hidden = residual + o_proj.output
    hidden = residual + o_proj_output

    # post_attention_layernorm
    post_ln_w = ckpt.get('model.layers.0.post_attention_layernorm.weight')
    if post_ln_w is None:
        print("\n  SKIP — model.layers.0.post_attention_layernorm.weight not in ckpt")
        return True

    expected = _rms_norm(hidden, post_ln_w)

    print(f"\n{'=' * 70}")
    print(f"Attention + residual chain check (layer 0):")
    print(f"  residual (embed): {residual.shape}, o_proj.output: {o_proj_output.shape}")
    print(f"  post_attn_ln.weight: norm={post_ln_w.float().norm().item():.4f}")
    print(f"  expected gate_up_input: {expected.shape}, online: {online_gate_up_input.shape}")

    return compare_tensor("L0.attn+residual chain (post_ln(embed+o_proj.out) vs gate_up.input)",
                          online_gate_up_input, expected)


def check_lm_head_weight(ckpt: SafetensorsIndex):
    """Diagnose lm_head weight: existence, shape, relationship to embed_tokens."""
    print(f"\n{'=' * 70}")
    print("lm_head weight diagnostics:")
    print("-" * 70)

    lm_head_w = ckpt.get('lm_head.weight')
    embed_w = ckpt.get('model.embed_tokens.weight')
    norm_w = ckpt.get('model.norm.weight')

    if lm_head_w is None:
        print("  X lm_head.weight: NOT FOUND in checkpoint!")
        print("    vLLM may tie it to embed_tokens.weight — check config.json tie_word_embeddings")
    else:
        print(f"  lm_head.weight: shape={list(lm_head_w.shape)} dtype={lm_head_w.dtype}")

    if embed_w is not None:
        print(f"  embed_tokens.weight: shape={list(embed_w.shape)} dtype={embed_w.dtype}")
    else:
        print("  X embed_tokens.weight: NOT FOUND")

    if norm_w is not None:
        print(f"  model.norm.weight: shape={list(norm_w.shape)} dtype={norm_w.dtype} "
              f"norm={norm_w.float().norm().item():.4f}")
    else:
        print("  X model.norm.weight: NOT FOUND")

    # Check if lm_head == embed_tokens (weight tying)
    if lm_head_w is not None and embed_w is not None:
        if lm_head_w.shape == embed_w.shape:
            diff = (lm_head_w.float() - embed_w.float()).abs().max().item()
            corr = F.cosine_similarity(
                lm_head_w.float().flatten().unsqueeze(0),
                embed_w.float().flatten().unsqueeze(0),
            ).item()
            if diff == 0:
                print(f"  lm_head == embed_tokens: IDENTICAL (weight tied)")
            else:
                print(f"  lm_head vs embed_tokens: corr={corr:.6f} max_diff={diff:.6f} (separate weights)")
        else:
            print(f"  lm_head vs embed_tokens: different shapes, separate weights")


def detect_saved_layers(diag_dir: Path) -> Set[int]:
    """Auto-detect which layer indices have saved activation files."""
    layers = set()
    for f in diag_dir.glob("model_layers_*_*.pt"):
        m = re.search(r'model_layers_(\d+)_', f.name)
        if m:
            layers.add(int(m.group(1)))
    return layers


def compare_layer(
    layer: int,
    diag_dir: Path,
    ckpt: SafetensorsIndex,
    head_dim: int,
    check_chain: bool = False,
) -> tuple:
    """Compare all projections for a single layer. Returns (passed, failed)."""
    passed, failed = 0, 0

    # --- qkv_proj (fused: q+k+v merged into one ResQ layer online) ---
    qkv_file = diag_dir / f"model_layers_{layer}_self_attn_qkv_proj.pt"
    if qkv_file.exists():
        data = torch.load(qkv_file, map_location="cpu", weights_only=True)
        print(f"\n  qkv_proj: input={data['input'].shape} output={data['output'].shape}")

        ok = compare_fused_resq(
            f"L{layer}.qkv_proj (fused vs separate q+k+v)",
            data['input'], data['output'], ckpt, layer,
            ['q_proj', 'k_proj', 'v_proj'], head_dim,
        )
        if ok:
            passed += 1
        else:
            failed += 1

            q_out = ckpt[f'model.layers.{layer}.self_attn.q_proj.weight_high'].shape[0]
            k_out = ckpt[f'model.layers.{layer}.self_attn.k_proj.weight_high'].shape[0]
            v_out = ckpt[f'model.layers.{layer}.self_attn.v_proj.weight_high'].shape[0]
            online_out_dim = data['output'].shape[-1]
            print(f"    Expected: q={q_out} + k={k_out} + v={v_out} = {q_out+k_out+v_out}")
            print(f"    Online output dim: {online_out_dim}")

    # --- o_proj (not fused) ---
    o_file = diag_dir / f"model_layers_{layer}_self_attn_o_proj.pt"
    if o_file.exists():
        data = torch.load(o_file, map_location="cpu", weights_only=True)
        print(f"\n  o_proj: input={data['input'].shape} output={data['output'].shape}")

        prefix = f'model.layers.{layer}.self_attn.o_proj'
        ok = compare_single_resq(
            f"L{layer}.o_proj", data['input'], data['output'],
            ckpt, prefix, is_o_proj=True, head_dim=head_dim,
        )
        if ok:
            passed += 1
        else:
            failed += 1

    # --- gate_up_proj (fused: gate+up merged online) ---
    gate_up_file = diag_dir / f"model_layers_{layer}_mlp_gate_up_proj.pt"
    if gate_up_file.exists():
        data = torch.load(gate_up_file, map_location="cpu", weights_only=True)
        print(f"\n  gate_up_proj: input={data['input'].shape} output={data['output'].shape}")

        ok = compare_fused_resq(
            f"L{layer}.gate_up_proj (fused vs separate gate+up)",
            data['input'], data['output'], ckpt, layer,
            ['gate_proj', 'up_proj'], head_dim,
        )
        if ok:
            passed += 1
        else:
            failed += 1

    # --- down_proj (W8A8, not fused) ---
    down_file = diag_dir / f"model_layers_{layer}_mlp_down_proj.pt"
    if down_file.exists():
        data = torch.load(down_file, map_location="cpu", weights_only=True)
        print(f"\n  down_proj: input={data['input'].shape} output={data['output'].shape}")

        prefix = f'model.layers.{layer}.mlp.down_proj'
        ok = compare_w8a8(
            f"L{layer}.down_proj (W8A8)", data['input'], data['output'],
            ckpt, prefix, layer,
        )
        if ok:
            passed += 1
        else:
            failed += 1

    # --- MLP activation chain check ---
    if check_chain:
        ok = check_mlp_activation_chain(layer, diag_dir)
        if ok:
            passed += 1
        else:
            failed += 1

    return passed, failed


def main():
    parser = argparse.ArgumentParser(
        description="Compare online vllm-ascend activations with offline replay"
    )
    parser.add_argument("--ckpt-a", required=True,
                        help="Path to ckpt A (original or converted)")
    parser.add_argument("--diag-dir", default="/tmp/resq_online_acts",
                        help="Directory with saved online activations")
    parser.add_argument("--layers", default=None,
                        help="Layer indices to compare (comma-separated). "
                             "Default: auto-detect from saved files")
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--check-chain", action="store_true",
                        help="Also check MLP activation chain: down_proj.input == silu(gate)*up")
    parser.add_argument("--check-embedding", action="store_true",
                        help="Check embedding→RMSNorm→qkv_input chain for layer 0")
    args = parser.parse_args()

    diag_dir = Path(args.diag_dir)

    if not diag_dir.exists():
        print(f"Error: {diag_dir} not found. Run vllm-ascend first to save activations.")
        return

    # Determine which layers to compare
    if args.layers is not None:
        layers = sorted(int(x) for x in args.layers.split(',') if x.strip().isdigit())
    else:
        layers = sorted(detect_saved_layers(diag_dir))

    if not layers:
        print(f"Error: no activation files found in {diag_dir}")
        return

    print(f"Loading ckpt A from {args.ckpt_a}...")
    ckpt = SafetensorsIndex(args.ckpt_a)

    total_passed, total_failed = 0, 0

    for layer in layers:
        print(f"\n{'=' * 70}")
        print(f"Layer {layer}:")
        print("-" * 70)

        p, f = compare_layer(layer, diag_dir, ckpt, args.head_dim,
                              check_chain=args.check_chain)
        total_passed += p
        total_failed += f

    # --- Embedding chain check ---
    if args.check_embedding:
        ok = check_embedding_chain(diag_dir, ckpt)
        if ok:
            total_passed += 1
        else:
            total_failed += 1

        # --- Attention + residual chain check (layer 0) ---
        ok = check_attention_residual_chain(diag_dir, ckpt)
        if ok:
            total_passed += 1
        else:
            total_failed += 1

    # --- lm_head weight diagnostics ---
    check_lm_head_weight(ckpt)

    print(f"\n{'=' * 70}")
    print(f"Total: Passed={total_passed}, Failed={total_failed} "
          f"(across {len(layers)} layer(s): {layers})")

    if total_failed == 0 and total_passed > 0:
        print("\nAll checks passed.")
        if len(layers) == 1:
            print("Consider testing more layers (--diag-layers 0,15,31,47,63) "
                  "to narrow down the issue.")
    elif total_failed > 0:
        print("\nMismatch detected — see details above.")


if __name__ == "__main__":
    main()
