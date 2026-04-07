"""
Phase 4 — Full Evaluation
==========================
Inject fake quantisation into every SwinTransformerBlock3D according to
`block_bits`, with inter-block error forwarding compensation, and run the
full test set through the model.

Quantisation strategy
---------------------
INT8 Block:
  - Weight: per-output-channel fake-quant all 4 Linear layers
  - Activation: fake-quant attn_windows output of forward_part1
  - Error carry: e_block = fp32_out - int8_out, forwarded to next same-stage block

INT4 Block (W4A16 — weight-only quantisation):
  - Weight: per-output-channel fake-quant (INT4)
  - Activation: FP32 (no activation quantisation)
  - No carry, no anchor — activations flow through unchanged.
  - At Stage boundaries, INT8 carry is reset.

Note on DDP: the patches are applied before wrapping in DDP, so they are
visible to all ranks. carry/anchor state is local to each rank (each rank
processes its own batch slice), which is correct.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import (
    symmetric_fake_quant,
    apply_fake_quant_weight_inplace,
    restore_weight_inplace,
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
    Monkey-patch every SwinTransformerBlock3D.forward() to inject:
      1. Per-block weight fake quantisation
      2. Activation fake quantisation
      3. Error forwarding (INT8 carry) / Residual reuse (INT4 anchor)

    Also patches BasicLayer.forward() to reset carry/anchor at Stage entry.

    Call remove_full_eval() to undo all patches.
    """
    backbone = model.module.backbone if hasattr(model, 'module') else model.backbone
    _disable_checkpoint(backbone)

    n_stages = len(backbone.layers)
    depths = [len(layer.blocks) for layer in backbone.layers]

    for s_idx, layer in enumerate(backbone.layers):
        n_blocks = depths[s_idx]
        for b_idx, blk in enumerate(layer.blocks):
            bits = _get_bits(block_bits, s_idx, b_idx)
            lam = calib_stats.get('block_stats', {}).get((s_idx, b_idx), {}).get('lambda', 0.3)

            # Attach mutable state
            blk._qstate = _BlockQuantState()
            blk._ptq_bits = bits
            blk._ptq_lambda = lam
            blk._ptq_s_idx = s_idx
            blk._ptq_b_idx = b_idx
            blk._ptq_n_blocks = n_blocks

            _patch_block_forward(blk)

        _patch_layer_forward(layer, s_idx, verbose)

    if verbose:
        total = sum(depths)
        n8 = sum(1 for v in block_bits.values() if v == 'int8')
        n4 = sum(1 for v in block_bits.values() if v == 'int4')
        print(f'[FullEval] Patched {total} blocks: {n8}×INT8, {n4}×INT4')


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
            _apply_weights(blk, 8)
            shortcut = x
            x_attn = blk.forward_part1(x, mask_matrix)
            x = shortcut + blk.drop_path(x_attn)
            x = x + blk.forward_part2(x)
            _restore_weights(blk)

            # Activation INT8 fake-quant + error carry
            x_sim = symmetric_fake_quant(x, 8, per_channel=False)
            blk._qstate_next_carry = (lam * (x - x_sim)).detach()
            return x_sim

        else:
            # ---- W4A16: weight-only INT4, activations stay FP32 ----
            _apply_weights(blk, 4)
            shortcut = x
            x_attn = blk.forward_part1(x, mask_matrix)
            x = shortcut + blk.drop_path(x_attn)
            x = x + blk.forward_part2(x)
            _restore_weights(blk)
            # No activation quant, no carry
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
# Weight helpers
# ---------------------------------------------------------------------------

def _apply_weights(blk, bits: int):
    """In-place per-channel weight fake-quant for all Linear in block."""
    for l in [blk.attn.qkv, blk.attn.proj, blk.mlp.fc1, blk.mlp.fc2]:
        apply_fake_quant_weight_inplace(l, bits)


def _restore_weights(blk):
    for l in [blk.attn.qkv, blk.attn.proj, blk.mlp.fc1, blk.mlp.fc2]:
        restore_weight_inplace(l)


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

def remove_full_eval(model: nn.Module):
    """Remove all PTQ patches from the model, restoring original forward()."""
    backbone = model.module.backbone if hasattr(model, 'module') else model.backbone

    for layer in backbone.layers:
        for blk in layer.blocks:
            if getattr(blk, '_ptq_block_patched', False):
                blk.forward = blk._original_block_forward
                blk._ptq_block_patched = False
                _restore_weights(blk)   # safety: ensure fp32 weights
                if hasattr(blk, '_qstate'):
                    del blk._qstate

        if getattr(layer, '_ptq_layer_patched', False):
            layer.forward = layer._original_layer_forward
            layer._ptq_layer_patched = False

    _restore_checkpoint(backbone)
