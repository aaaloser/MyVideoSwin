"""
Real integer-storage weight quantisation utilities.

INT8  — weight stored as torch.int8,  per-output-channel scale (float32).
INT4  — two values packed into one uint8 byte (nibble packing),
        per-output-channel scale (float32).

Memory footprint vs FP32:
  INT8 :  1 byte / param  → 4× compression
  INT4 :  0.5 byte / param → 8× compression
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Scale helpers
# ---------------------------------------------------------------------------

def _absmax_scale(weight: torch.Tensor, max_val: float) -> torch.Tensor:
    """Per-output-channel absmax scale: scale = amax / max_val."""
    w = weight.view(weight.size(0), -1)
    amax = w.abs().max(dim=1).values.clamp(min=1e-8)
    return amax / max_val


# ---------------------------------------------------------------------------
# INT8 primitives
# ---------------------------------------------------------------------------

def quantize_to_int8(weight: torch.Tensor) -> tuple:
    """
    Symmetric per-output-channel quantisation to INT8.
    Returns (weight_q: int8 [out, in*...], scale: float32 [out]).
    """
    scale = _absmax_scale(weight, 127.0)
    w = weight.view(weight.size(0), -1)
    q = (w / scale.unsqueeze(1)).round().clamp(-128, 127).to(torch.int8)
    return q, scale


def dequantize_int8(weight_q: torch.Tensor, scale: torch.Tensor,
                    original_shape: torch.Size, dtype=torch.float32) -> torch.Tensor:
    w = weight_q.to(dtype) * scale.to(dtype).unsqueeze(1)
    return w.view(original_shape)


# ---------------------------------------------------------------------------
# INT4 primitives
# ---------------------------------------------------------------------------

def quantize_to_int4(weight: torch.Tensor) -> tuple:
    """
    Symmetric per-output-channel quantisation to INT4 (range [-8, 7]).
    Returns (weight_q: int8 [out, in*...], scale: float32 [out]).
    Values are in [-8, 7] — not yet packed.
    """
    scale = _absmax_scale(weight, 7.0)
    w = weight.view(weight.size(0), -1)
    q = (w / scale.unsqueeze(1)).round().clamp(-8, 7).to(torch.int8)
    return q, scale


def pack_int4(weight_int8: torch.Tensor) -> torch.Tensor:
    """
    Pack pairs of int8 values (each in [-8, 7]) into nibbles stored as uint8.

    Input : [out, in]  — int8, values in [-8, 7];  'in' must be even.
    Output: [out, in//2] — uint8.

    Packing layout: packed_byte = (lo & 0x0F) | ((hi & 0x0F) << 4)
    where lo = element at even index, hi = element at odd index.
    """
    assert weight_int8.shape[-1] % 2 == 0, \
        "INT4 packing requires even inner dimension"
    u = weight_int8.view(torch.uint8)          # same bits, reinterpreted
    lo = u[..., 0::2] & 0x0F                   # lower nibble (even indices)
    hi = u[..., 1::2] & 0x0F                   # upper nibble (odd  indices)
    return (lo | (hi << 4)).to(torch.uint8)


def unpack_int4(packed: torch.Tensor, original_shape: torch.Size) -> torch.Tensor:
    """
    Unpack uint8 nibble-packed tensor back to int8 in [-8, 7].

    Input : [out, in//2] — uint8.
    Output: [out, in]    — int8, values in [-8, 7].
    """
    lo = (packed & 0x0F).to(torch.int32)       # 0..15
    hi = ((packed >> 4) & 0x0F).to(torch.int32)

    # Sign-extend 4-bit → 32-bit via arithmetic shift
    lo_s = ((lo << 28) >> 28).to(torch.int8)   # -8..7
    hi_s = ((hi << 28) >> 28).to(torch.int8)

    out = torch.empty((*packed.shape[:-1], packed.shape[-1] * 2),
                      dtype=torch.int8, device=packed.device)
    out[..., 0::2] = lo_s
    out[..., 1::2] = hi_s
    return out.view(original_shape)


def dequantize_int4_packed(packed: torch.Tensor, scale: torch.Tensor,
                            original_shape: torch.Size,
                            dtype=torch.float32) -> torch.Tensor:
    q = unpack_int4(packed, original_shape)                # int8 [out, in]
    return q.to(dtype) * scale.to(dtype).unsqueeze(1)


# ---------------------------------------------------------------------------
# RealQuantLinear — drop-in replacement for nn.Linear
# ---------------------------------------------------------------------------

class RealQuantLinear(nn.Module):
    """
    Linear layer whose weights are stored as real integers (not FP32).

    INT8 : weight_q stored as int8  [out_features, in_features]
    INT4 : weight_q stored as uint8 [out_features, in_features // 2]  (nibble-packed)

    Dequantisation is performed on-the-fly inside forward(), so the
    temporary FP32 weight tensor is allocated, used for F.linear, and
    immediately freed — keeping peak VRAM proportional to activation
    size rather than weight size.
    """

    def __init__(self, bits: int, weight_fp32: torch.Tensor,
                 bias: Optional[torch.Tensor]):
        super().__init__()
        self.bits = bits
        self.weight_shape = weight_fp32.shape          # (out, in)
        self.in_features  = weight_fp32.size(1)
        self.out_features = weight_fp32.size(0)

        if bits == 8:
            weight_q, scale = quantize_to_int8(weight_fp32)
            self.register_buffer('weight_q', weight_q)     # int8
        else:  # 4
            q_i8, scale = quantize_to_int4(weight_fp32)
            weight_q = pack_int4(q_i8)
            self.register_buffer('weight_q', weight_q)     # uint8, half cols

        self.register_buffer('scale', scale)               # float32 [out]

        if bias is not None:
            self.register_buffer('bias_buf', bias.clone().float())
        else:
            self.bias_buf = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.bits == 8:
            w = dequantize_int8(self.weight_q, self.scale,
                                self.weight_shape, x.dtype)
        else:
            w = dequantize_int4_packed(self.weight_q, self.scale,
                                       self.weight_shape, x.dtype)
        bias = self.bias_buf.to(x.dtype) if self.bias_buf is not None else None
        return F.linear(x, w, bias)

    def extra_repr(self) -> str:
        return (f'in={self.in_features}, out={self.out_features}, '
                f'bits={self.bits}')


# ---------------------------------------------------------------------------
# Conversion helper
# ---------------------------------------------------------------------------

def replace_linear_real_quant(linear: nn.Linear, bits: int) -> RealQuantLinear:
    """
    Convert an nn.Linear to RealQuantLinear.

    The caller should delete the reference to `linear` afterwards so that
    its FP32 .weight is freed from GPU memory.
    """
    w = linear.weight.data.float()
    b = linear.bias.data if linear.bias is not None else None
    rql = RealQuantLinear(bits=bits, weight_fp32=w, bias=b)
    return rql.to(linear.weight.device)
