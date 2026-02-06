#!/usr/bin/env python3
"""
Validate converted ResQ checkpoint completeness against original model.

Loads the original model's weight_map (from CKPT O) and checks that every
expected parameter exists in the converted checkpoint (CKPT A).

Mapping rules:
  - ResQ layers (q/k/v/o_proj, gate/up_proj):
      .weight → .weight_high, .weight_low, .scale_high, .scale_low, .high_fraction
  - W8A8 layers (down_proj):
      .weight → .weight, .weight_scale, .weight_offset
  - Non-quantized (embed_tokens, lm_head, layernorms, model.norm):
      kept as-is

Usage:
    python -m tools.resq_debug.check_completeness \
        --ckpt-o /path/to/original_bf16_model \
        --ckpt-a /path/to/converted_resq_ckpt
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Set, Tuple

import torch
import torch.nn.functional as F

from .debug_compare import SafetensorsIndex


# Projections that use ResQ quantization
RESQ_PROJ_NAMES = {'q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj'}

# Projections that use W8A8 dynamic quantization
W8A8_PROJ_NAMES = {'down_proj'}

# ResQ sub-parameters (replace .weight)
RESQ_SUFFIXES = ['weight_high', 'weight_low', 'scale_high', 'scale_low', 'high_fraction']

# W8A8 sub-parameters (replace .weight with these)
W8A8_SUFFIXES = ['weight', 'weight_scale', 'weight_offset']


def classify_param(name: str) -> str:
    """Classify a parameter name into: resq, w8a8, or passthrough."""
    # Check if it's a quantized linear weight
    if name.endswith('.weight'):
        for proj in RESQ_PROJ_NAMES:
            if f'.{proj}.weight' in name:
                return 'resq'
        for proj in W8A8_PROJ_NAMES:
            if f'.{proj}.weight' in name:
                return 'w8a8'
    # Everything else (bias, layernorm, embed_tokens, lm_head, etc.)
    return 'passthrough'


def get_expected_params(orig_name: str, category: str) -> List[str]:
    """Given an original param name, return the expected converted param names."""
    if category == 'resq':
        prefix = orig_name.rsplit('.weight', 1)[0]
        return [f'{prefix}.{s}' for s in RESQ_SUFFIXES]
    elif category == 'w8a8':
        prefix = orig_name.rsplit('.weight', 1)[0]
        return [f'{prefix}.{s}' for s in W8A8_SUFFIXES]
    else:
        return [orig_name]


def check_passthrough_values(orig_index: SafetensorsIndex, conv_index: SafetensorsIndex,
                             passthrough_params: List[str]) -> int:
    """Compare values of all passthrough (non-quantized) params between CKPT O and CKPT A.

    Returns number of mismatches.
    """
    print(f"\n{'=' * 80}")
    print(f"Value comparison for {len(passthrough_params)} passthrough parameters:")
    print(f"{'=' * 80}")

    mismatches = 0
    for name in passthrough_params:
        orig_w = orig_index.get(name)
        conv_w = conv_index.get(name)

        if orig_w is None or conv_w is None:
            continue

        o = orig_w.float().flatten()
        c = conv_w.float().flatten()

        if o.numel() != c.numel():
            print(f"  X {name}: shape mismatch orig={list(orig_w.shape)} conv={list(conv_w.shape)}")
            mismatches += 1
            continue

        if o.numel() == 0:
            continue

        max_diff = (o - c).abs().max().item()

        if max_diff == 0:
            continue  # identical, skip

        corr = F.cosine_similarity(o.unsqueeze(0), c.unsqueeze(0)).item()
        rel_err = (o - c).norm().item() / (o.norm().item() + 1e-8)

        ok = corr > 0.999 and max_diff < 0.01
        mark = "  " if ok else "X "
        print(f"{mark}{name}: corr={corr:.6f} rel_err={rel_err:.6f} max_diff={max_diff:.6f}")

        if not ok:
            print(f"    orig: norm={o.norm().item():.4f} first5={o[:5].tolist()}")
            print(f"    conv: norm={c.norm().item():.4f} first5={c[:5].tolist()}")
            mismatches += 1

    if mismatches == 0:
        print("  All passthrough values match.")
    else:
        print(f"\n  WARNING: {mismatches} value mismatch(es)!")

    return mismatches


def main():
    parser = argparse.ArgumentParser(
        description="Validate converted checkpoint completeness"
    )
    parser.add_argument("--ckpt-o", required=True,
                        help="Path to original bf16 model (has model.safetensors.index.json)")
    parser.add_argument("--ckpt-a", required=True,
                        help="Path to converted ResQ checkpoint")
    parser.add_argument("--check-values", action="store_true",
                        help="Also compare values of passthrough params between CKPT O and CKPT A")
    args = parser.parse_args()

    # Load original model's weight map
    orig_index_path = os.path.join(args.ckpt_o, "model.safetensors.index.json")
    if not os.path.exists(orig_index_path):
        print(f"Error: {orig_index_path} not found")
        return

    with open(orig_index_path) as f:
        orig_index = json.load(f)
    orig_params = sorted(orig_index["weight_map"].keys())
    print(f"Original model (CKPT O): {len(orig_params)} parameters")

    # Load converted checkpoint index
    conv_index = SafetensorsIndex(args.ckpt_a)
    conv_keys = set(conv_index._index.keys())
    print(f"Converted checkpoint (CKPT A): {len(conv_keys)} tensors")

    # Check each original param
    missing = []
    found = []
    extra_in_conv = set(conv_keys)  # track what's NOT accounted for

    print(f"\n{'=' * 80}")
    print("Checking parameter completeness:")
    print(f"{'=' * 80}")

    categories = {'resq': 0, 'w8a8': 0, 'passthrough': 0}
    passthrough_params = []

    for orig_name in orig_params:
        cat = classify_param(orig_name)
        categories[cat] += 1
        expected = get_expected_params(orig_name, cat)

        if cat == 'passthrough':
            passthrough_params.append(orig_name)

        missing_for_param = []
        for exp in expected:
            if exp in conv_keys:
                extra_in_conv.discard(exp)
                found.append(exp)
            else:
                missing_for_param.append(exp)

        if missing_for_param:
            print(f"\n  X {orig_name} ({cat}):")
            for m in missing_for_param:
                print(f"      MISSING: {m}")
            missing.extend(missing_for_param)

    # Summary
    print(f"\n{'=' * 80}")
    print("Summary:")
    print(f"{'=' * 80}")
    print(f"  Original params:  {len(orig_params)}")
    print(f"    ResQ layers:    {categories['resq']}")
    print(f"    W8A8 layers:    {categories['w8a8']}")
    print(f"    Passthrough:    {categories['passthrough']}")
    print(f"  Expected in CKPT A: {len(found) + len(missing)}")
    print(f"  Found:            {len(found)}")
    print(f"  MISSING:          {len(missing)}")

    if missing:
        print(f"\n  Missing parameters ({len(missing)}):")
        for m in sorted(missing):
            print(f"    - {m}")

    # Extra keys in converted checkpoint (not mapped from original)
    # Filter out known extra keys (rotation matrices, etc.)
    unexpected = sorted(k for k in extra_in_conv
                        if not k.startswith('resq.')
                        and 'rotation_Pd' not in k
                        and 'rotation_Hd' not in k
                        and 'h_butterfly' not in k)
    if unexpected:
        print(f"\n  Extra keys in CKPT A not mapped from original ({len(unexpected)}):")
        for k in unexpected:
            print(f"    + {k}")

    if not missing:
        print("\n  All parameters accounted for.")
    else:
        print(f"\n  WARNING: {len(missing)} missing parameter(s) — likely root cause of garbled output!")

    # Value comparison for passthrough params
    if args.check_values:
        orig_ckpt = SafetensorsIndex(args.ckpt_o)
        check_passthrough_values(orig_ckpt, conv_index, passthrough_params)


if __name__ == "__main__":
    main()
