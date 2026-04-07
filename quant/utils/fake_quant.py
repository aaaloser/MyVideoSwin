"""
Symmetric fake (simulated) quantization utilities.

All operations are purely tensor-level and do NOT rely on
torch.quantization, keeping them fully decoupled from mmaction.
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Core symmetric quantisation primitives
# ---------------------------------------------------------------------------

def compute_scale(x: torch.Tensor, bits: int, per_channel: bool = False,
                  channel_dim: int = 0) -> torch.Tensor:
    """Compute per-tensor or per-channel absmax scale.

    scale = max(|x|) / (2^(bits-1) - 1)

    Args:
        x: Float tensor to be quantised.
        bits: Target bit-width (e.g. 8 or 4).
        per_channel: If True, compute scale per output-channel.
        channel_dim: Which dimension is the output-channel axis.

    Returns:
        scale: Scalar or 1-D tensor broadcastable to x.
    """
    q_max = float(2 ** (bits - 1) - 1)
    if per_channel:
        # move channel dim to front, flatten rest, take absmax
        x_perm = x.transpose(0, channel_dim).contiguous()
        n_ch = x_perm.shape[0]
        absmax = x_perm.view(n_ch, -1).abs().max(dim=1).values  # (n_ch,)
        absmax = absmax.clamp(min=1e-8)
        scale = absmax / q_max
        # reshape for broadcasting: (n_ch, 1, 1, ...)
        shape = [1] * x.dim()
        shape[channel_dim] = n_ch
        return scale.view(shape)
    else:
        absmax = x.abs().max().clamp(min=1e-8)
        return absmax / q_max


def symmetric_fake_quant(x: torch.Tensor, bits: int,
                         per_channel: bool = False,
                         channel_dim: int = 0) -> torch.Tensor:
    """Simulate quantisation: quantise then immediately dequantise.

    W_sim = clamp(round(x / scale), -q_max, q_max) * scale

    Args:
        x: Float tensor (weights or activations).
        bits: Target bit-width.
        per_channel: Per-channel scale for weights; per-tensor for activations.
        channel_dim: Output-channel axis when per_channel=True.

    Returns:
        x_sim: Same shape as x, dtype float, with quantisation noise injected.
    """
    q_max = float(2 ** (bits - 1) - 1)
    scale = compute_scale(x, bits, per_channel=per_channel,
                          channel_dim=channel_dim)
    x_q = torch.clamp(torch.round(x / scale), -q_max, q_max)
    return x_q * scale


# ---------------------------------------------------------------------------
# Weight fake-quantisation helper
# ---------------------------------------------------------------------------

def fake_quant_weight(layer: nn.Linear, bits: int) -> torch.Tensor:
    """Return W_sim for an nn.Linear weight (per-output-channel, symmetric).

    Does NOT modify layer.weight in-place; returns the simulated tensor.

    Args:
        layer: An nn.Linear module.
        bits: Target bit-width.

    Returns:
        w_sim: Tensor with same shape as layer.weight.
    """
    return symmetric_fake_quant(layer.weight.data, bits,
                                per_channel=True, channel_dim=0)


def apply_fake_quant_weight_inplace(layer: nn.Linear, bits: int) -> torch.Tensor:
    """Overwrite layer.weight.data with W_sim and return the quantisation error.

    Call restore_weight_inplace() to undo.

    Returns:
        error: layer.weight.data_original - W_sim  (for error forwarding)
    """
    w_fp = layer.weight.data.clone()
    w_sim = fake_quant_weight(layer, bits)
    layer.weight.data.copy_(w_sim)
    layer._weight_backup = w_fp          # stash original
    return w_fp - w_sim


def restore_weight_inplace(layer: nn.Linear):
    """Restore original FP32 weight after fake-quant inference."""
    if hasattr(layer, '_weight_backup'):
        layer.weight.data.copy_(layer._weight_backup)
        del layer._weight_backup
