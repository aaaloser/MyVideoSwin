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

__all__ = [
    'compute_scale', 'symmetric_fake_quant', 'fake_quant_weight',
    'apply_fake_quant_weight_inplace', 'restore_weight_inplace',
    'register_attention_capture_hooks', 'restore_all_attention_patches',
    'clear_calib_buffers', 'clear_all_calib_state', 'clear_all_hooks',
]
