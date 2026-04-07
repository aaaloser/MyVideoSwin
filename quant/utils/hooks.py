"""
Forward-hook utilities for Video Swin Transformer PTQ calibration.

Design for speed:
 - R and E metrics are computed ON GPU inside the hook, only scalars are
   accumulated (no large tensor copies to CPU every batch).
 - Only the first batch's x_in (B=1 slice) is kept per block for λ estimation.
"""

from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce
from operator import mul

EPS = 1e-8


# ---------------------------------------------------------------------------
# Inline GPU metrics
# ---------------------------------------------------------------------------

def _residual_magnitude_gpu(x_in: torch.Tensor, ws: Tuple) -> float:
    """Compute R_win on GPU. x_in: (B, D, H, W, C)"""
    Wd, Wh, Ww = ws
    B, D, H, W, C = x_in.shape
    pad_d = (Wd - D % Wd) % Wd
    pad_h = (Wh - H % Wh) % Wh
    pad_w = (Ww - W % Ww) % Ww
    x = F.pad(x_in, (0, 0, 0, pad_w, 0, pad_h, 0, pad_d))
    Bd, Hd, Wd2 = x.shape[1] // Wd, x.shape[2] // Wh, x.shape[3] // Ww
    x = x.view(B, Bd, Wd, Hd, Wh, Wd2, Ww, C)
    x_win = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(
        -1, reduce(mul, ws), C)                  # (B*nW, N, C)
    anchor = x_win.mean(dim=1, keepdim=True)
    return (x_win - anchor).norm(dim=-1).mean().item()


def _attn_entropy_gpu(attn: torch.Tensor) -> float:
    """Compute E_3D on GPU. attn: (B*nW, nH, N, N) post-softmax"""
    p = attn.float().clamp(min=EPS)
    return (-(p * p.log2()).sum(dim=-1)).mean().item()


# ---------------------------------------------------------------------------
# Hook registration
# ---------------------------------------------------------------------------

def register_attention_capture_hooks(backbone: nn.Module) -> List:
    """
    For each SwinTransformerBlock3D:
      - Register a forward hook that computes R on GPU and appends a scalar
        to blk._calib_r_list. Also stores blk._calib_x_in_sample (first batch only).
      - Monkey-patch WindowAttention3D.forward to compute E on GPU and append
        a scalar to blk.attn._calib_e_list.

    Returns list of hook handles for later removal.
    """
    handles = []

    for stage_idx, layer in enumerate(backbone.layers):
        ws = layer.window_size
        for blk_idx, blk in enumerate(layer.blocks):
            # ---------- block x_in hook ----------
            def _make_xin_hook(b, window_size):
                def _hook(module, inputs, output):
                    x_in = inputs[0]   # (B, D, H, W, C) on GPU
                    with torch.no_grad():
                        r = _residual_magnitude_gpu(x_in, window_size)
                    if not hasattr(b, '_calib_r_list'):
                        b._calib_r_list = []
                    b._calib_r_list.append(r)
                    # Keep one sample for λ estimation
                    if not hasattr(b, '_calib_x_in_sample'):
                        b._calib_x_in_sample = x_in[:1].detach().cpu()
                return _hook
            h = blk.register_forward_hook(_make_xin_hook(blk, ws))
            handles.append(h)

            # ---------- attn entropy hook (monkey-patch) ----------
            _patch_window_attention(blk.attn)

    return handles


def _patch_window_attention(attn_module):
    """
    Monkey-patch WindowAttention3D.forward to capture post-softmax attn
    entropy as a scalar. Idempotent.
    """
    if getattr(attn_module, '_ptq_patched', False):
        return

    original_forward = attn_module.forward

    def _patched_forward(x, mask=None):
        B_, N, C = x.shape
        qkv = attn_module.qkv(x).reshape(
            B_, N, 3, attn_module.num_heads,
            C // attn_module.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * attn_module.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = attn_module.relative_position_bias_table[
            attn_module.relative_position_index[:N, :N].reshape(-1)
        ].reshape(N, N, -1).permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, attn_module.num_heads, N, N) \
                   + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, attn_module.num_heads, N, N)
            attn = attn_module.softmax(attn)
        else:
            attn = attn_module.softmax(attn)

        # Capture entropy as scalar (no CPU copy of the full attn tensor)
        with torch.no_grad():
            e = _attn_entropy_gpu(attn)
        if not hasattr(attn_module, '_calib_e_list'):
            attn_module._calib_e_list = []
        attn_module._calib_e_list.append(e)

        attn = attn_module.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = attn_module.proj(x)
        x = attn_module.proj_drop(x)
        return x

    attn_module.forward = _patched_forward
    attn_module._ptq_patched = True
    attn_module._original_forward = original_forward


def restore_window_attention(attn_module):
    if getattr(attn_module, '_ptq_patched', False):
        attn_module.forward = attn_module._original_forward
        attn_module._ptq_patched = False


def restore_all_attention_patches(backbone: nn.Module):
    for layer in backbone.layers:
        for blk in layer.blocks:
            restore_window_attention(blk.attn)


def clear_calib_buffers(backbone: nn.Module):
    """Delete per-batch buffers (scalar lists only; sample kept until end)."""
    # NOTE: We no longer need per-batch cleanup since we only store scalars.
    # This is intentionally a no-op now; call clear_all_calib_state() at end.
    pass


def clear_all_calib_state(backbone: nn.Module):
    """Remove all calibration state from blocks after calibrate() finishes."""
    for layer in backbone.layers:
        for blk in layer.blocks:
            for attr in ('_calib_r_list', '_calib_x_in_sample'):
                if hasattr(blk, attr):
                    delattr(blk, attr)
            for attr in ('_calib_e_list',):
                if hasattr(blk.attn, attr):
                    delattr(blk.attn, attr)


def clear_all_hooks(handles: List):
    for h in handles:
        h.remove()

