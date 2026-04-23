"""
Phase 2 — Calibration
=====================
Run N calibration clips through the FP32 model.

Speed design:
 - R and E are computed on GPU inside hooks as scalars (no large tensor copies).
 - Only the first batch's x_in (shape[:1]) is kept per block for lambda estimation.
 - lambda estimation uses those stored samples after the main loop — no extra full
   model forward passes.

calib_stats layout
------------------
{
  stage_idx (int): {
    "r_min": float, "r_max": float,
    "e_min": float, "e_max": float,
    "tau":   float,
    "lambda_per_block": {block_idx: float},
  },
  "block_stats": {
    (stage_idx, blk_idx): {
        "r_win_mean": float,
        "e_win_mean": float,
        "s_score":    float,
        "lambda":     float,
    }
  }
}
"""

import math
import os
import pickle
from typing import Dict

import numpy as np
import torch
import torch.nn as nn

from .utils import (
    register_attention_capture_hooks,
    restore_all_attention_patches,
    clear_all_calib_state,
    clear_all_hooks,
    symmetric_fake_quant,
)
from mmaction.models.backbones.swin_transformer import get_window_size, compute_mask

ALPHA = 0.5
BETA  = 0.5
TAU_PERCENTILE = 70
LAMBDA_BITS = 8
EPS = 1e-8


# ---------------------------------------------------------------------------
# lambda estimation
# ---------------------------------------------------------------------------

def _build_attn_mask(blk, D, H, W, device):
    window_size, shift_size = get_window_size(
        (D, H, W), blk.window_size, blk.shift_size)
    Dp = int(np.ceil(D / window_size[0])) * window_size[0]
    Hp = int(np.ceil(H / window_size[1])) * window_size[1]
    Wp = int(np.ceil(W / window_size[2])) * window_size[2]
    return compute_mask(Dp, Hp, Wp, window_size, shift_size, device)


def _estimate_lambda(
    blk,
    x_in_cpu,
    device,
    lambda_ratio_scale=1.0,
    lambda_min=0.1,
    lambda_max=0.5,
):
    """Two forward_part1 passes (FP32 vs INT8) on a single CPU-stored sample."""
    x = x_in_cpu.to(device)
    _, D, H, W, _ = x.shape
    attn_mask = _build_attn_mask(blk, D, H, W, device)

    with torch.no_grad():
        fp32_out = blk.forward_part1(x, attn_mask)

        linears = [blk.attn.qkv, blk.attn.proj, blk.mlp.fc1, blk.mlp.fc2]
        orig_ws = [l.weight.data.clone() for l in linears]
        for l in linears:
            l.weight.data.copy_(
                symmetric_fake_quant(l.weight.data, LAMBDA_BITS,
                                     per_channel=True, channel_dim=0))
        int8_out = blk.forward_part1(x, attn_mask)
        for l, w in zip(linears, orig_ws):
            l.weight.data.copy_(w)

    mse = (fp32_out - int8_out).pow(2).mean().item()
    ref = fp32_out.pow(2).mean().item() + EPS
    ratio = mse / ref
    raw = math.exp(-lambda_ratio_scale * ratio)
    lam = float(np.clip(raw, lambda_min, lambda_max))
    return lam, ratio, raw


# ---------------------------------------------------------------------------
# Checkpoint toggle
# ---------------------------------------------------------------------------

def _disable_checkpoint(backbone):
    for layer in backbone.layers:
        for blk in layer.blocks:
            blk._use_checkpoint_orig = blk.use_checkpoint
            blk.use_checkpoint = False


def _restore_checkpoint(backbone):
    for layer in backbone.layers:
        for blk in layer.blocks:
            if hasattr(blk, '_use_checkpoint_orig'):
                blk.use_checkpoint = blk._use_checkpoint_orig


# ---------------------------------------------------------------------------
# Helper: extract a single-view tensor from the data dict and move to device
# ---------------------------------------------------------------------------

def _extract_single_view(data, device):
    """
    data['imgs'] shape: (B, num_views, C, T, H, W)  or  (B, C, T, H, W)
    Returns: (B, C, T, H, W) on device using only the FIRST view.
    Each "batch" from the test dataloader is B=1 video, so we get exactly
    one clip — avoids max_testing_views chunking overhead entirely.
    """
    imgs = data['imgs'].to(device)
    if imgs.ndim == 6:
        imgs = imgs[:, 0]   # take first view: (B, C, T, H, W)
    return imgs


# ---------------------------------------------------------------------------
# Main calibration
# ---------------------------------------------------------------------------

def calibrate(
    model,
    calib_loader,
    n_calib_batches=32,
    save_path='output_pt/calib_stats.pkl',
    alpha=ALPHA,
    beta=BETA,
    tau_percentile=TAU_PERCENTILE,
    lambda_ratio_scale=1.0,
    lambda_min=0.1,
    lambda_max=0.5,
    verbose=True,
):
    if lambda_ratio_scale <= 0:
        raise ValueError('lambda_ratio_scale must be > 0')
    if lambda_min > lambda_max:
        raise ValueError('lambda_min must be <= lambda_max')

    model.eval()
    backbone = model.module.backbone if hasattr(model, 'module') else model.backbone
    device = next(backbone.parameters()).device
    _disable_checkpoint(backbone)
    handles = register_attention_capture_hooks(backbone)
    n_stages = len(backbone.layers)

    if verbose:
        print(f'[Calibrate] Running {n_calib_batches} calibration batches ...')

    with torch.no_grad():
        for batch_idx, data in enumerate(calib_loader):
            if batch_idx >= n_calib_batches:
                break
            if verbose and (batch_idx == 0 or (batch_idx + 1) % 8 == 0):
                print(f'  batch {batch_idx + 1}/{n_calib_batches}')
            # Call backbone directly on ONE view — 3x faster than going
            # through recognizer (skips max_testing_views chunking + cls_head)
            imgs = _extract_single_view(data, device)
            _ = backbone(imgs)
            # Scalar metrics are accumulated in hooks automatically.

    clear_all_hooks(handles)
    _restore_checkpoint(backbone)

    if verbose:
        print('[Calibrate] Computing per-stage statistics ...')
    calib_stats = {
        'block_stats': {},
        'lambda_cfg': {
            'ratio_scale': float(lambda_ratio_scale),
            'clip_min': float(lambda_min),
            'clip_max': float(lambda_max),
        },
    }

    for s_idx in range(n_stages):
        layer = backbone.layers[s_idx]
        all_r, all_e = [], []
        for blk in layer.blocks:
            r_list = getattr(blk, '_calib_r_list', [])
            e_list = getattr(blk.attn, '_calib_e_list', [])
            if r_list:
                all_r.append(float(np.mean(r_list)))
                all_e.append(float(np.mean(e_list)))

        if not all_r:
            continue

        r_min, r_max = float(min(all_r)), float(max(all_r))
        e_min, e_max = float(min(all_e)), float(max(all_e))
        r_range = max(r_max - r_min, EPS)
        e_range = max(e_max - e_min, EPS)
        s_scores = []
        lambda_per_block = {}

        for b_idx, blk in enumerate(layer.blocks):
            r_list = getattr(blk, '_calib_r_list', [])
            e_list = getattr(blk.attn, '_calib_e_list', [])
            if not r_list:
                continue
            r_mean = float(np.mean(r_list))
            e_mean = float(np.mean(e_list))
            r_norm = (r_mean - r_min) / r_range
            e_norm = (e_mean - e_min) / e_range
            s_score = alpha * r_norm + beta * (1.0 - e_norm)
            s_scores.append(s_score)
            lambda_per_block[b_idx] = 0.3
            calib_stats['block_stats'][(s_idx, b_idx)] = {
                'r_win_mean': r_mean, 'e_win_mean': e_mean,
                's_score': s_score, 'lambda': 0.3,
            }

        tau = float(np.percentile(s_scores, tau_percentile)) if s_scores else 0.5
        calib_stats[s_idx] = {
            'r_min': r_min, 'r_max': r_max,
            'e_min': e_min, 'e_max': e_max,
            'tau': tau, 'lambda_per_block': lambda_per_block,
        }
        if verbose:
            print(f'  Stage {s_idx}: r=[{r_min:.4f},{r_max:.4f}] '
                  f'e=[{e_min:.4f},{e_max:.4f}] tau={tau:.4f}')

    # lambda estimation using stored first-batch samples (no extra backbone forward)
    if verbose:
        print('[Calibrate] Estimating lambda per block ...')
    for s_idx in range(n_stages):
        for b_idx, blk in enumerate(backbone.layers[s_idx].blocks):
            x_in_cpu = getattr(blk, '_calib_x_in_sample', None)
            if x_in_cpu is None:
                continue
            try:
                lam, ratio, raw = _estimate_lambda(
                    blk,
                    x_in_cpu,
                    device,
                    lambda_ratio_scale=lambda_ratio_scale,
                    lambda_min=lambda_min,
                    lambda_max=lambda_max,
                )
            except Exception:
                lam = 0.3
                ratio = None
                raw = None
            if (s_idx, b_idx) in calib_stats['block_stats']:
                calib_stats['block_stats'][(s_idx, b_idx)]['lambda'] = lam
            if s_idx in calib_stats:
                calib_stats[s_idx]['lambda_per_block'][b_idx] = lam
            if verbose:
                if ratio is None:
                    print(f'    Stage {s_idx} Block {b_idx}: lambda={lam:.4f} (fallback)')
                else:
                    print(
                        f'    Stage {s_idx} Block {b_idx}: '
                        f'ratio={ratio:.3e} raw={raw:.4f} lambda={lam:.4f}'
                    )

    clear_all_calib_state(backbone)

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(calib_stats, f)
    if verbose:
        print(f'[Calibrate] Saved -> {save_path}')

    return calib_stats
