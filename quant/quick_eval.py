"""
Phase 3 — Quick Evaluation
===========================
Run a mini validation set, compute per-block S_win gate trigger rate,
and assign INT8/INT4 to produce block_bits.json.

Uses the same hook mechanism as calibrate (scalars accumulated on GPU),
with calib_stats for normalisation bounds and per-stage tau.
"""

import json
import os
from typing import Dict

import numpy as np
import torch
import torch.nn as nn

from .calibrate import (
    _disable_checkpoint, _restore_checkpoint,
    _extract_single_view,
    ALPHA, BETA, EPS,
)
from .utils import (
    register_attention_capture_hooks,
    clear_all_calib_state,
    clear_all_hooks,
)

HIGH_RATIO_THRESHOLD = 0.30
FORCE_INT8_LAST_IN_STAGE = True


def quick_eval(
    model,
    mini_loader,
    calib_stats,
    n_eval_batches=50,
    save_path='output_pt/block_bits.json',
    high_ratio_threshold=HIGH_RATIO_THRESHOLD,
    verbose=True,
):
    """
    Compute per-block window gate trigger rate and assign bit-widths.
    """
    model.eval()
    backbone = model.module.backbone if hasattr(model, 'module') else model.backbone
    device = next(backbone.parameters()).device
    _disable_checkpoint(backbone)
    handles = register_attention_capture_hooks(backbone)

    n_stages = len(backbone.layers)
    # gate_counts[s][b] = {'high': int, 'total': int}
    gate_counts = [[{'high': 0, 'total': 0}
                    for _ in backbone.layers[s].blocks]
                   for s in range(n_stages)]

    if verbose:
        print(f'[QuickEval] Running {n_eval_batches} batches ...')

    with torch.no_grad():
        for batch_idx, data in enumerate(mini_loader):
            if batch_idx >= n_eval_batches:
                break
            if verbose and (batch_idx == 0 or (batch_idx + 1) % 50 == 0):
                print(f'  batch {batch_idx + 1}/{n_eval_batches}')

            # Direct backbone call on first view — skips cls_head and
            # max_testing_views chunking for maximum speed
            imgs = _extract_single_view(data, device)
            _ = backbone(imgs)

            for s_idx in range(n_stages):
                if s_idx not in calib_stats:
                    continue
                stage_st = calib_stats[s_idx]
                r_min = stage_st['r_min']
                r_max = stage_st['r_max']
                e_min = stage_st['e_min']
                e_max = stage_st['e_max']
                tau   = stage_st['tau']
                r_range = max(r_max - r_min, EPS)
                e_range = max(e_max - e_min, EPS)

                for b_idx, blk in enumerate(backbone.layers[s_idx].blocks):
                    r_list = getattr(blk, '_calib_r_list', [])
                    e_list = getattr(blk.attn, '_calib_e_list', [])
                    if not r_list:
                        continue
                    # Use the scalar from this batch (last appended value)
                    r = r_list[-1]
                    e = e_list[-1]
                    r_norm = max(0.0, min(1.0, (r - r_min) / r_range))
                    e_norm = max(0.0, min(1.0, (e - e_min) / e_range))
                    s_score = ALPHA * r_norm + BETA * (1.0 - e_norm)
                    z = 1 if s_score > tau else 0
                    gate_counts[s_idx][b_idx]['high'] += z
                    gate_counts[s_idx][b_idx]['total'] += 1

    clear_all_hooks(handles)
    clear_all_calib_state(backbone)
    _restore_checkpoint(backbone)

    if verbose:
        print('[QuickEval] Generating block_bits ...')

    block_bits = {}
    for s_idx in range(n_stages):
        n_blocks = len(backbone.layers[s_idx].blocks)
        for b_idx in range(n_blocks):
            counts = gate_counts[s_idx][b_idx]
            total = counts['total']
            high  = counts['high']
            ratio = high / total if total > 0 else 0.0

            is_last = FORCE_INT8_LAST_IN_STAGE and (b_idx == n_blocks - 1)
            bits = 'int8' if (ratio >= high_ratio_threshold or is_last) else 'int4'
            key = f'stage{s_idx}_block{b_idx}'
            block_bits[key] = bits

            if verbose:
                flag = ' (forced INT8)' if is_last else ''
                print(f'  {key}: high_ratio={ratio:.3f} -> {bits}{flag}')

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(block_bits, f, indent=2)
    if verbose:
        n8 = sum(1 for v in block_bits.values() if v == 'int8')
        n4 = sum(1 for v in block_bits.values() if v == 'int4')
        print(f'[QuickEval] Saved -> {save_path}  (INT8={n8}, INT4={n4})')

    return block_bits
