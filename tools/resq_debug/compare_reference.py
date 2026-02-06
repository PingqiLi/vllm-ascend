#!/usr/bin/env python3
"""
Compare bf16 reference activations (CKPT_O) with quantized online activations (CKPT_A).

Shows where quantization error accumulates across layers.
Supports rotation-aware comparison using CKPT_B rotation matrices.

bf16 (run_reference.py) saves separate projections:
    model_layers_63_self_attn_q_proj.pt
    model_layers_63_self_attn_k_proj.pt
    model_layers_63_self_attn_v_proj.pt
    ...

quantized (run_online.py) saves fused projections:
    model_layers_63_self_attn_qkv_proj.pt
    model_layers_63_mlp_gate_up_proj.pt
    ...

This script handles the mapping between them.

Rotation domains (ResQ):
    INVARIANT (directly comparable): q, k, gate, up outputs
    ROTATED by Ua: layer input, o_proj output, MLP input, down_proj output
    ROTATED by Ub: v output (per-head rotation)
    ROTATED (complex): o_proj input (attention output domain)

Usage:
    # Without rotation handling (norm-ratio only for rotated points):
    python -m tools.resq_debug.compare_reference \\
        --ref-dir /tmp/resq_ref_acts \\
        --online-dir /tmp/resq_online_acts \\
        --layers 63

    # With rotation handling (recommended, requires CKPT_B):
    python -m tools.resq_debug.compare_reference \\
        --ref-dir /tmp/resq_ref_acts \\
        --online-dir /tmp/resq_online_acts \\
        --ckpt-b ${CKPT_B} \\
        --layers 63
"""

import argparse
import re
from pathlib import Path
from typing import Dict, Set

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Rotation matrix loading (same logic as debug_compare.py)
# ---------------------------------------------------------------------------

def load_rotation_matrices(ckpt_b_path: str) -> Dict[str, torch.Tensor]:
    """Load rotation matrices from ckpt B; pre-compute Ua, Ub.

    Ua = Pa @ Ra  (hidden_dim rotation, applied to residual stream)
    Ub = Pb @ Rb  (per-head rotation, applied to v output)
    """
    from safetensors import safe_open

    path = Path(ckpt_b_path)
    result: Dict[str, torch.Tensor] = {}

    if path.is_dir():
        for f in sorted(path.glob("*.safetensors")):
            with safe_open(str(f), framework="pt", device="cpu") as sf:
                for key in sf.keys():
                    result[key] = sf.get_tensor(key)
        for f in sorted(path.glob("*.pt")):
            data = torch.load(f, map_location="cpu", weights_only=True)
            if isinstance(data, dict):
                result.update(data)
    elif path.suffix == '.pt':
        result = torch.load(path, map_location="cpu", weights_only=True)

    print(f"Loaded {len(result)} matrices from {ckpt_b_path}")

    # Pre-compute Ua = Pa @ Ra, Ub = Pb @ Rb
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


def apply_inv_rotation(
    x: torch.Tensor, U: torch.Tensor, rot_type: str,
) -> torch.Tensor:
    """Apply inverse rotation to bring tensor back to original domain.

    For Ua (hidden_dim rotation): x @ Ua^T
    For Ub (per-head rotation):   per-head x_h @ Ub_h^T
    """
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


# ---------------------------------------------------------------------------
# Tensor comparison
# ---------------------------------------------------------------------------

def compare_tensor(name: str, a: torch.Tensor, b: torch.Tensor,
                   label_a: str = "ref(O)", label_b: str = "online(A)",
                   domain: str = "inv") -> bool:
    """Compare two tensors with rotation-domain-aware diagnostics.

    Args:
        domain:
            "inv"   - invariant (same domain): compare cosine + rel_err
            "unrot" - unrotated via CKPT_B: compare cosine + rel_err
            "rot"   - rotated (no CKPT_B available): norm_ratio only
    """
    fa = a.float().flatten()
    fb = b.float().flatten()

    if fa.numel() != fb.numel():
        print(f"  X {name}: shape mismatch "
              f"{label_a}={list(a.shape)} {label_b}={list(b.shape)}")
        return False

    if fa.numel() == 0:
        print(f"  (skip {name}: empty tensor)")
        return True

    corr = F.cosine_similarity(fa.unsqueeze(0), fb.unsqueeze(0)).item()
    a_norm = fa.norm().item()
    b_norm = fb.norm().item()
    norm_ratio = b_norm / (a_norm + 1e-8)
    rel_err = (fa - fb).norm().item() / (a_norm + 1e-8)
    max_diff = (fa - fb).abs().max().item()

    if domain == "rot":
        # Rotated domain without unrotation: cosine is meaningless
        ok = 0.9 < norm_ratio < 1.1
        mark = "  " if ok else "X "
        print(f"{mark}{name} [rot]:")
        print(f"    norm_ratio={norm_ratio:.4f}  "
              f"(cosine={corr:.4f} -- not meaningful across rotation domains)")
        if not ok:
            print(f"    {label_a}: norm={a_norm:.4f}")
            print(f"    {label_b}: norm={b_norm:.4f}")
            if norm_ratio > 100 or norm_ratio < 0.01:
                print("    [hint] Extreme norm mismatch -- likely a real bug")
    else:
        # Invariant or unrotated: full comparison
        tag = "unrot" if domain == "unrot" else "inv"
        ok = corr > 0.95 and rel_err < 0.2
        mark = "  " if ok else "X "
        print(f"{mark}{name} [{tag}]:")
        print(f"    corr={corr:.6f}  rel_err={rel_err:.4f}  "
              f"norm_ratio={norm_ratio:.4f}  max_diff={max_diff:.4f}")
        if not ok:
            print(f"    {label_a}: norm={a_norm:.4f} "
                  f"first5={a.flatten()[:5].tolist()}")
            print(f"    {label_b}: norm={b_norm:.4f} "
                  f"first5={b.flatten()[:5].tolist()}")
            if domain == "inv" and corr < 0.5 and 0.9 < norm_ratio < 1.1:
                print("    [hint] Low corr but norm preserved -- "
                      "may need rotation handling (provide --ckpt-b)")

    return ok


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def detect_saved_layers(diag_dir: Path, prefix_pattern: str) -> Set[int]:
    """Auto-detect which layer indices have saved activation files."""
    layers = set()
    for f in diag_dir.glob("model_layers_*_*.pt"):
        m = re.search(r'model_layers_(\d+)_', f.name)
        if m:
            layers.add(int(m.group(1)))
    return layers


def load_act(diag_dir: Path, fname: str):
    """Load activation file, return None if not found."""
    path = diag_dir / fname
    if not path.exists():
        return None
    return torch.load(path, map_location="cpu", weights_only=True)


def _tally(ok: bool, passed: int, failed: int):
    """Helper to increment passed/failed counters."""
    return (passed + 1, failed) if ok else (passed, failed + 1)


# ---------------------------------------------------------------------------
# Per-layer comparison
# ---------------------------------------------------------------------------

def compare_layer(
    layer: int,
    ref_dir: Path,
    online_dir: Path,
    rotations: Dict[str, torch.Tensor],
) -> tuple:
    """Compare a single layer between bf16 ref and quantized online.

    Rotation domain handling (ResQ):
        INVARIANT:    q, k, gate, up outputs (rotation absorbed into weights)
        Ua-ROTATED:   layer input, o_proj output, MLP input, down_proj output
        Ub-ROTATED:   v output (per-head rotation)
        COMPLEX-ROT:  o_proj input (attention output, mixed rotation domain)

    Returns (passed, failed, skipped).
    """
    passed, failed, skipped = 0, 0, 0
    print(f"\n{'=' * 70}")
    print(f"Layer {layer}:")
    print("-" * 70)

    # Get rotation matrices for this layer
    Ua = rotations.get(f'resq.layer.{layer}.Ua')
    Ub = rotations.get(f'resq.layer.{layer}.Ub')

    if rotations:
        print(f"  Rotation: Ua={'loaded' if Ua is not None else 'MISSING'}, "
              f"Ub={'loaded' if Ub is not None else 'MISSING'}")
    else:
        print("  Rotation: not provided (--ckpt-b), "
              "rotated points use norm-ratio only")

    # Load activation files
    ref_q = load_act(ref_dir,
                     f"model_layers_{layer}_self_attn_q_proj.pt")
    ref_k = load_act(ref_dir,
                     f"model_layers_{layer}_self_attn_k_proj.pt")
    ref_v = load_act(ref_dir,
                     f"model_layers_{layer}_self_attn_v_proj.pt")
    ref_o = load_act(ref_dir,
                     f"model_layers_{layer}_self_attn_o_proj.pt")
    ref_gate = load_act(ref_dir,
                        f"model_layers_{layer}_mlp_gate_proj.pt")
    ref_up = load_act(ref_dir,
                      f"model_layers_{layer}_mlp_up_proj.pt")
    ref_down = load_act(ref_dir,
                        f"model_layers_{layer}_mlp_down_proj.pt")

    online_qkv = load_act(
        online_dir, f"model_layers_{layer}_self_attn_qkv_proj.pt")
    online_o = load_act(
        online_dir, f"model_layers_{layer}_self_attn_o_proj.pt")
    online_gate_up = load_act(
        online_dir, f"model_layers_{layer}_mlp_gate_up_proj.pt")
    online_down = load_act(
        online_dir, f"model_layers_{layer}_mlp_down_proj.pt")

    # ---- 1. Layer input (post-input_layernorm) -- Ua-ROTATED ----
    # In the ResQ model the residual stream is in the Ua-rotated domain,
    # so the post-layernorm hidden state fed to qkv_proj is also rotated.
    if ref_q is not None and online_qkv is not None:
        online_input = online_qkv['input']
        if Ua is not None:
            online_input = apply_inv_rotation(online_input, Ua, 'Ua')
            dom = "unrot"
        else:
            dom = "rot"
        print("\n  Layer input (post-input_layernorm) [Ua-rotated]:")
        ok = compare_tensor(
            f"L{layer} input (q_proj.in vs qkv_proj.in)",
            ref_q['input'], online_input, domain=dom,
        )
        passed, failed = _tally(ok, passed, failed)
    else:
        print("\n  (skip layer input -- missing files)")
        skipped += 1

    # ---- 2. Q/K/V outputs -- split from fused QKV ----
    # Q and K outputs are INVARIANT (rotation absorbed into weights).
    # V output is ROTATED by Ub (per-head rotation).
    # Previously these were compared as a single fused tensor, which
    # mixed invariant (q,k) and rotated (v) domains -- misleading.
    if (ref_q is not None and ref_k is not None and ref_v is not None
            and online_qkv is not None):
        q_dim = ref_q['output'].shape[-1]
        k_dim = ref_k['output'].shape[-1]

        online_out = online_qkv['output']
        online_q = online_out[..., :q_dim]
        online_k = online_out[..., q_dim:q_dim + k_dim]
        online_v = online_out[..., q_dim + k_dim:]

        print("\n  Q output [invariant]:")
        ok = compare_tensor(
            f"L{layer} q", ref_q['output'], online_q, domain="inv")
        passed, failed = _tally(ok, passed, failed)

        print("\n  K output [invariant]:")
        ok = compare_tensor(
            f"L{layer} k", ref_k['output'], online_k, domain="inv")
        passed, failed = _tally(ok, passed, failed)

        online_v_cmp = online_v
        if Ub is not None:
            online_v_cmp = apply_inv_rotation(online_v, Ub, 'Ub')
            dom = "unrot"
        else:
            dom = "rot"
        print("\n  V output [Ub per-head rotated]:")
        ok = compare_tensor(
            f"L{layer} v", ref_v['output'], online_v_cmp, domain=dom)
        passed, failed = _tally(ok, passed, failed)
    else:
        skipped += 3

    # ---- 3. o_proj ----
    if ref_o is not None and online_o is not None:
        # o_proj input: attention output, in a complex rotated domain
        # (involves Ub rotation from V path + attention weights).
        # Can only meaningfully compare norm ratio.
        print("\n  o_proj input [rotated -- attention output domain]:")
        ok = compare_tensor(
            f"L{layer} o_proj input",
            ref_o['input'], online_o['input'], domain="rot")
        passed, failed = _tally(ok, passed, failed)

        # o_proj output: Ua-ROTATED (enters residual stream)
        online_o_out = online_o['output']
        if Ua is not None:
            online_o_out = apply_inv_rotation(online_o_out, Ua, 'Ua')
            dom = "unrot"
        else:
            dom = "rot"
        print("\n  o_proj output [Ua-rotated, residual stream]:")
        ok = compare_tensor(
            f"L{layer} o_proj output",
            ref_o['output'], online_o_out, domain=dom)
        passed, failed = _tally(ok, passed, failed)
    else:
        skipped += 2

    # ---- 4. MLP input (post-post_attention_layernorm) -- Ua-ROTATED ----
    if ref_gate is not None and online_gate_up is not None:
        online_mlp_in = online_gate_up['input']
        if Ua is not None:
            online_mlp_in = apply_inv_rotation(online_mlp_in, Ua, 'Ua')
            dom = "unrot"
        else:
            dom = "rot"
        print("\n  MLP input (post-post_attn_layernorm) [Ua-rotated]:")
        ok = compare_tensor(
            f"L{layer} MLP input (gate.in vs gate_up.in)",
            ref_gate['input'], online_mlp_in, domain=dom,
        )
        passed, failed = _tally(ok, passed, failed)
    else:
        skipped += 1

    # ---- 5. Gate+Up output -- INVARIANT ----
    # gate and up projections absorb the Ua rotation into their weights,
    # so their outputs should be in the same domain as the original model.
    ref_up = load_act(ref_dir, f"model_layers_{layer}_mlp_up_proj.pt")

    if (ref_gate is not None and ref_up is not None
            and online_gate_up is not None):
        ref_gate_up_out = torch.cat([
            ref_gate['output'], ref_up['output'],
        ], dim=-1)
        print("\n  Gate+Up output [invariant]:")
        print(f"    ref: gate={list(ref_gate['output'].shape)} "
              f"up={list(ref_up['output'].shape)} -> "
              f"concat={list(ref_gate_up_out.shape)}")
        print(f"    online: gate_up="
              f"{list(online_gate_up['output'].shape)}")
        ok = compare_tensor(
            f"L{layer} Gate+Up output",
            ref_gate_up_out, online_gate_up['output'], domain="inv",
        )
        passed, failed = _tally(ok, passed, failed)
    else:
        skipped += 1

    # ---- 6. down_proj ----
    if ref_down is not None and online_down is not None:
        # down_proj input: the MLP intermediate value SiLU(gate)*up.
        # Since gate and up outputs are invariant, this should also be
        # approximately invariant (saved before Ud rotation in resq_linear).
        print("\n  down_proj input [~invariant, pre-Ud]:")
        ok = compare_tensor(
            f"L{layer} down_proj input",
            ref_down['input'], online_down['input'], domain="inv")
        passed, failed = _tally(ok, passed, failed)

        # down_proj output: Ua-ROTATED (enters residual stream)
        online_down_out = online_down['output']
        if Ua is not None:
            online_down_out = apply_inv_rotation(
                online_down_out, Ua, 'Ua')
            dom = "unrot"
        else:
            dom = "rot"
        print(f"\n  down_proj output [Ua-rotated, residual stream]:")
        ok = compare_tensor(
            f"L{layer} down_proj output",
            ref_down['output'], online_down_out, domain=dom)
        passed, failed = _tally(ok, passed, failed)
    else:
        skipped += 2

    return passed, failed, skipped


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare bf16 reference vs quantized online activations "
                    "(rotation-aware)"
    )
    parser.add_argument("--ref-dir", required=True,
                        help="Directory with bf16 reference activations "
                             "(from run_reference.py)")
    parser.add_argument("--online-dir", required=True,
                        help="Directory with quantized online activations "
                             "(from run_online.py)")
    parser.add_argument("--ckpt-b", default=None,
                        help="Path to CKPT_B (rotation matrices). "
                             "Enables proper unrotation before comparison. "
                             "Without this, rotated points use norm-ratio "
                             "only.")
    parser.add_argument("--layers", default=None,
                        help="Layer indices to compare (comma-separated). "
                             "Default: auto-detect from intersection of "
                             "both dirs")
    args = parser.parse_args()

    ref_dir = Path(args.ref_dir)
    online_dir = Path(args.online_dir)

    if not ref_dir.exists():
        print(f"Error: {ref_dir} not found. Run run_reference.py first.")
        return
    if not online_dir.exists():
        print(f"Error: {online_dir} not found. Run run_online.py first.")
        return

    # Load rotation matrices
    rotations: Dict[str, torch.Tensor] = {}
    if args.ckpt_b:
        rotations = load_rotation_matrices(args.ckpt_b)
    else:
        print("WARNING: --ckpt-b not provided. Rotated activation points "
              "will only compare norm_ratio (cosine is meaningless across "
              "rotation domains). Provide --ckpt-b for full comparison.\n")

    # Determine layers to compare
    if args.layers is not None:
        layers = sorted(int(x) for x in args.layers.split(',')
                        if x.strip().isdigit())
    else:
        ref_layers = detect_saved_layers(ref_dir, "model_layers_*")
        online_layers = detect_saved_layers(online_dir, "model_layers_*")
        layers = sorted(ref_layers & online_layers)
        if not layers:
            print(f"Error: no common layers found between "
                  f"{ref_dir} and {online_dir}")
            print(f"  ref layers: {sorted(ref_layers)}")
            print(f"  online layers: {sorted(online_layers)}")
            return

    print(f"Comparing bf16 reference vs quantized online")
    print(f"  ref-dir:    {ref_dir}")
    print(f"  online-dir: {online_dir}")
    print(f"  ckpt-b:     {args.ckpt_b or '(not provided)'}")
    print(f"  layers:     {layers}")

    # Check prompts match
    ref_ids = load_act(ref_dir, "input_ids.pt")
    online_ids = load_act(online_dir, "input_ids.pt")
    if ref_ids is not None and online_ids is not None:
        ref_prompt = ref_ids.get('prompt', '?')
        online_prompt = online_ids.get('prompt', '?')
        if ref_prompt != online_prompt:
            print(f"\n  WARNING: prompts differ!")
            print(f"    ref:    {ref_prompt!r}")
            print(f"    online: {online_prompt!r}")
        ref_tokens = ref_ids['input_ids']
        online_tokens = online_ids['input_ids']
        if not torch.equal(ref_tokens, online_tokens):
            print(f"\n  WARNING: input_ids differ!")
            print(f"    ref:    {ref_tokens.tolist()}")
            print(f"    online: {online_tokens.tolist()}")

    total_passed, total_failed, total_skipped = 0, 0, 0

    for layer in layers:
        p, f, s = compare_layer(layer, ref_dir, online_dir, rotations)
        total_passed += p
        total_failed += f
        total_skipped += s

    print(f"\n{'=' * 70}")
    print(f"Total: Passed={total_passed}, Failed={total_failed}, "
          f"Skipped={total_skipped}")
    print(f"  (across {len(layers)} layer(s): {layers})")

    print(f"\nDomain legend:")
    print(f"  [inv]   = invariant (same domain, directly comparable)")
    print(f"  [unrot] = unrotated via CKPT_B (proper comparison)")
    print(f"  [rot]   = rotated (no CKPT_B, norm-ratio only)")

    if total_failed == 0 and total_passed > 0:
        print("\nAll comparisons passed -- quantized layer outputs match "
              "bf16 reference (within rotation-adjusted tolerance).")
        print("If output is still garbled, the issue is likely in "
              "model.norm or lm_head.")
    elif total_failed > 0:
        print("\nMismatch detected -- see details above.")
        if not rotations:
            print("NOTE: Provide --ckpt-b for rotation-aware comparison "
                  "to avoid false positives on rotated points.")
        print("\nInterpretation guide:")
        print("  [inv]   failed -> quantization error or implementation bug")
        print("  [unrot] failed -> quantization error "
              "(rotation handled correctly)")
        print("  [rot]   failed -> norm not preserved (likely real bug)")


if __name__ == "__main__":
    main()
