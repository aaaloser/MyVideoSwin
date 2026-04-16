"""
quant/utils/__init__.py
"""
from .fake_quant import (
    compute_scale,
    symmetric_fake_quant,
    fake_quant_weight,
    apply_fake_quant_weight_inplace,
    restore_weight_inplace,
)
from .hooks import (
    register_attention_capture_hooks,
    restore_all_attention_patches,
    clear_calib_buffers,
    clear_all_calib_state,
    clear_all_hooks,
)
from .real_quant import (
    RealQuantLinear,
    replace_linear_real_quant,
    quantize_to_int8,
    quantize_to_int4,
    pack_int4,
    unpack_int4,
    dequantize_int8,
    dequantize_int4_packed,
)

__all__ = [
    'compute_scale', 'symmetric_fake_quant', 'fake_quant_weight',
    'apply_fake_quant_weight_inplace', 'restore_weight_inplace',
    'register_attention_capture_hooks', 'restore_all_attention_patches',
    'clear_calib_buffers', 'clear_all_calib_state', 'clear_all_hooks',
    'RealQuantLinear', 'replace_linear_real_quant',
    'quantize_to_int8', 'quantize_to_int4',
    'pack_int4', 'unpack_int4', 'dequantize_int8', 'dequantize_int4_packed',
]
