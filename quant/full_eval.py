"""
Phase 4 — Full Evaluation
==========================
Inject fake quantisation into every SwinTransformerBlock3D according to
`block_bits`, with inter-block error forwarding compensation, and run the
full test set through the model.

Quantisation strategy
---------------------
INT8 Block (W8A8):
  - Weight: real INT8 storage via RealQuantLinear (4× memory reduction).
            Dequantised to FP32 on-the-fly inside each forward().
  - Activation: per-tensor fake-quant (simulate INT8 precision)
  - Error carry: e_block = fp32_out - int8_out, forwarded to next same-stage block

INT4 Block (W4A16):
  - Weight: real INT4 nibble-packed storage via RealQuantLinear (8× memory reduction).
            Dequantised to FP32 on-the-fly inside each forward().
  - Activation: FP32 (no activation quantisation)
  - No carry — activations flow through unchanged.

Memory savings: every Linear weight is stored as int8/uint8 instead of float32.
The temporary FP32 weight tensor is created for each matmul and immediately freed.

Note on DDP: patches and linear replacements are applied after DDP wrapping via
model.module.backbone. carry state is local to each rank (correct for inference).
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import (
    symmetric_fake_quant,
    replace_linear_real_quant,
)
from .calibrate import _disable_checkpoint, _restore_checkpoint

# ---------------------------------------------------------------------------
# Helper: parse block_bits key
# ---------------------------------------------------------------------------

def _key(s_idx: int, b_idx: int) -> str:
    return f'stage{s_idx}_block{b_idx}'


def _get_bits(block_bits: Dict, s_idx: int, b_idx: int) -> int:
    val = block_bits.get(_key(s_idx, b_idx), 'int8')
    return 8 if val == 'int8' else 4


# ---------------------------------------------------------------------------
# Per-block patch state container  (carry only — W4A16 has no anchor)
# ---------------------------------------------------------------------------

class _BlockQuantState:
    def __init__(self):
        self.carry: Optional[torch.Tensor] = None   # INT8 error carry

    def reset(self):
        self.carry = None


# ---------------------------------------------------------------------------
# Patch injection
# ---------------------------------------------------------------------------

def inject_full_eval(
    model: nn.Module,
    block_bits: Dict,
    calib_stats: Dict,
    verbose: bool = True,
):
    """
    For every SwinTransformerBlock3D:
      1. Replace its 4 nn.Linear layers with RealQuantLinear (real int storage).
         FP32 weights are freed immediately; memory drops 4–8× per block.
      2. Patch block.forward() for INT8 activation fake-quant + carry.
    Also patches BasicLayer.forward() for carry propagation.

    Call remove_full_eval() to remove forward patches.
    Note: linear replacements are permanent (weights remain as int storage).
    """
    backbone = model.module.backbone if hasattr(model, 'module') else model.backbone
    _disable_checkpoint(backbone)

    depths = [len(layer.blocks) for layer in backbone.layers]

    for s_idx, layer in enumerate(backbone.layers):
        n_blocks = depths[s_idx]
        for b_idx, blk in enumerate(layer.blocks):
            bits = _get_bits(block_bits, s_idx, b_idx)
            lam = calib_stats.get('block_stats', {}).get((s_idx, b_idx), {}).get('lambda', 0.3)

            # Replace Linear weights with real int storage (frees FP32 immediately)
            _replace_block_linears(blk, bits)

            # Attach mutable carry state
            blk._qstate = _BlockQuantState()
            blk._ptq_bits = bits
            blk._ptq_lambda = lam
            blk._ptq_s_idx = s_idx
            blk._ptq_b_idx = b_idx
            blk._ptq_n_blocks = n_blocks

            _patch_block_forward(blk)

        _patch_layer_forward(layer, s_idx, verbose)

    # Free any lingering FP32 weight tensors
    torch.cuda.empty_cache()

    if verbose:
        total = sum(depths)
        n8 = sum(1 for v in block_bits.values() if v == 'int8')
        n4 = sum(1 for v in block_bits.values() if v == 'int4')
        print(f'[RealQuant] Replaced {total} blocks: {n8}×INT8, {n4}×INT4 '
              f'(weights now integer-stored)')


def _patch_block_forward(blk):
    """Replace blk.forward with a quantised version."""
    if getattr(blk, '_ptq_block_patched', False):
        return

    original_forward = blk.forward
    bits = blk._ptq_bits
    lam  = blk._ptq_lambda

    def _quant_forward(x, mask_matrix):
        qstate: _BlockQuantState = blk._qstate

        # Apply incoming carry from a preceding INT8 block
        if qstate.carry is not None:
            x = x + qstate.carry
            qstate.carry = None

        if bits == 8:
            # ---- W8A8 ----
            # Weights already stored as INT8 in RealQuantLinear;
            # dequantisation to FP32 happens inside each linear.forward().
            shortcut = x
            x_attn = blk.forward_part1(x, mask_matrix)
            x = shortcut + blk.drop_path(x_attn)
            x = x + blk.forward_part2(x)

            # Activation INT8 fake-quant + error carry
            x_sim = symmetric_fake_quant(x, 8, per_channel=False)
            blk._qstate_next_carry = (lam * (x - x_sim)).detach()
            return x_sim

        else:
            # ---- W4A16 ----
            # Weights stored as INT4 packed in RealQuantLinear;
            # activations remain FP32, no carry.
            shortcut = x
            x_attn = blk.forward_part1(x, mask_matrix)
            x = shortcut + blk.drop_path(x_attn)
            x = x + blk.forward_part2(x)
            return x

    blk.forward = _quant_forward
    blk._ptq_block_patched = True
    blk._original_block_forward = original_forward


def _patch_layer_forward(layer, s_idx: int, verbose: bool):
    """
    Wrap BasicLayer.forward to:
      1. Reset all carry/anchor state at the start of each Stage forward
      2. After each block, propagate carry/anchor to the next block
    """
    if getattr(layer, '_ptq_layer_patched', False):
        return

    original_layer_forward = layer.forward
    from einops import rearrange
    import numpy as np

    def _quant_layer_forward(x):
        from mmaction.models.backbones.swin_transformer import get_window_size, compute_mask
        B, C, D, H, W = x.shape
        window_size, shift_size = get_window_size(
            (D, H, W), layer.window_size, layer.shift_size)
        x = rearrange(x, 'b c d h w -> b d h w c')
        Dp = int(np.ceil(D / window_size[0])) * window_size[0]
        Hp = int(np.ceil(H / window_size[1])) * window_size[1]
        Wp = int(np.ceil(W / window_size[2])) * window_size[2]
        attn_mask = compute_mask(Dp, Hp, Wp, window_size, shift_size, x.device)

        # Reset all carry/anchor for this stage
        for blk in layer.blocks:
            blk._qstate.reset()

        for i, blk in enumerate(layer.blocks):
            x = blk(x, attn_mask)
            # Propagate INT8 carry to next block (INT4 blocks produce no carry)
            if i + 1 < len(layer.blocks):
                next_blk = layer.blocks[i + 1]
                carry = getattr(blk, '_qstate_next_carry', None)
                if carry is not None:
                    next_blk._qstate.carry = carry
                    blk._qstate_next_carry = None

        x = x.view(B, D, H, W, -1)
        if layer.downsample is not None:
            x = layer.downsample(x)
        x = rearrange(x, 'b d h w c -> b c d h w')
        return x

    layer.forward = _quant_layer_forward
    layer._ptq_layer_patched = True
    layer._original_layer_forward = original_layer_forward


# ---------------------------------------------------------------------------
# Linear replacement helpers
# ---------------------------------------------------------------------------

def _replace_block_linears(blk, bits: int):
    """
    Replace the 4 nn.Linear layers in a block with RealQuantLinear.
    The old FP32 Linear objects are deleted so their weight tensors can be
    freed from GPU memory (torch.cuda.empty_cache() flushes the allocator).
    """
    for parent_name, attr_name in [
        ('attn', 'qkv'), ('attn', 'proj'), ('mlp', 'fc1'), ('mlp', 'fc2')
    ]:
        parent = getattr(blk, parent_name)
        old_linear = getattr(parent, attr_name)
        new_linear = replace_linear_real_quant(old_linear, bits)
        setattr(parent, attr_name, new_linear)
        del old_linear


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

def remove_full_eval(model: nn.Module):
    """
    Remove forward-patch overrides (carry logic, activation fake-quant).
    Linear layers are NOT restored — they remain as RealQuantLinear with
    integer-stored weights (intentional: call save_quantized_model after this).
    """
    backbone = model.module.backbone if hasattr(model, 'module') else model.backbone

    for layer in backbone.layers:
        for blk in layer.blocks:
            if getattr(blk, '_ptq_block_patched', False):
                blk.forward = blk._original_block_forward
                blk._ptq_block_patched = False
                if hasattr(blk, '_qstate'):
                    del blk._qstate

        if getattr(layer, '_ptq_layer_patched', False):
            layer.forward = layer._original_layer_forward
            layer._ptq_layer_patched = False

    _restore_checkpoint(backbone)


# ---------------------------------------------------------------------------
# Save quantised model
# ---------------------------------------------------------------------------

def save_quantized_model(model: nn.Module, block_bits: Dict, save_path: str):
    """
    Save the quantised model to disk.

    Saved file contains:
      'state_dict' — model state dict with RealQuantLinear buffers
                     (weight_q: int8/uint8, scale: float32) instead of
                     the original float32 .weight parameters.
      'block_bits' — per-block bit-width config dict.

    To reload: reconstruct the model structure with RealQuantLinear in place,
    then load the state dict.
    """
    import os
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    payload = {
        'state_dict': model.state_dict(),
        'block_bits': block_bits,
    }
    torch.save(payload, save_path)
    size_mb = os.path.getsize(save_path) / 1024 / 1024
    print(f'[RealQuant] Saved quantised model → {save_path}  ({size_mb:.1f} MB)')
