import argparse
import csv
import json
import math
import os
import os.path as osp
import sys

import mmcv
import numpy as np
import torch
from mmcv import Config, DictAction
from mmcv.runner import load_checkpoint

# Ensure repo root is on PYTHONPATH when called from tools/analysis/
sys.path.insert(0, osp.abspath(osp.join(osp.dirname(__file__), '../..')))

from mmaction.datasets import build_dataloader, build_dataset
from mmaction.models import build_model
from quant.calibrate import (  # noqa: E402
    EPS,
    LAMBDA_BITS,
    _build_attn_mask,
    _disable_checkpoint,
    _extract_single_view,
    _restore_checkpoint,
)
from quant.utils import symmetric_fake_quant  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description='Observe lambda estimation stats without changing PTQ main flow')
    parser.add_argument('config', help='Test config file path')
    parser.add_argument('checkpoint', help='Checkpoint file path')
    parser.add_argument('--calib-batches', type=int, default=32,
                        help='Number of calibration batches to observe')
    parser.add_argument('--csv-path', default='output_pt/lambda_diag_calib32.csv',
                        help='Path to save per-block csv stats')
    parser.add_argument('--json-path', default='output_pt/lambda_diag_calib32_summary.json',
                        help='Path to save summary json stats')
    parser.add_argument('--device', default='cuda:0',
                        help='Device for observation, e.g. cuda:0 or cpu')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction, default={},
                        help='Override config settings (key=value)')
    return parser.parse_args()


def turn_off_pretrained(cfg):
    if 'pretrained' in cfg:
        cfg.pretrained = None
    for sub_cfg in cfg.values():
        if isinstance(sub_cfg, dict):
            turn_off_pretrained(sub_cfg)


def build_model_from_cfg(cfg, checkpoint, device):
    turn_off_pretrained(cfg.model)
    model = build_model(cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, checkpoint, map_location='cpu')
    model = model.to(device)
    model.eval()
    return model


def build_test_loader(cfg):
    dataset = build_dataset(cfg.data.test, dict(test_mode=True))
    dataloader_default = dict(
        videos_per_gpu=1,
        workers_per_gpu=1,
        dist=False,
        shuffle=False,
    )
    test_dataloader_cfg = cfg.data.get('test_dataloader', {})
    dataloader_cfg = {
        **dataloader_default,
        **test_dataloader_cfg,
        'dist': False,
        'shuffle': False,
    }
    loader = build_dataloader(dataset, **dataloader_cfg)
    return loader


def register_first_input_hooks(backbone):
    samples = {}
    handles = []

    for s_idx, layer in enumerate(backbone.layers):
        for b_idx, blk in enumerate(layer.blocks):
            key = (s_idx, b_idx)

            def _make_hook(k):
                def _hook(module, inputs, output):
                    if k in samples:
                        return
                    if not inputs:
                        return
                    x_in = inputs[0]
                    samples[k] = x_in[:1].detach().cpu()
                return _hook

            handles.append(blk.register_forward_hook(_make_hook(key)))

    return handles, samples


def remove_hooks(handles):
    for h in handles:
        h.remove()


def _compute_pair_metrics(fp32_out, int8_out):
    mse = (fp32_out - int8_out).pow(2).mean().item()
    ref = fp32_out.pow(2).mean().item() + EPS
    ratio = mse / ref
    raw = float(math.exp(-ratio))
    clip = float(np.clip(raw, 0.1, 0.5))
    return mse, ref, ratio, raw, clip


def observe_block_lambda(blk, x_in_cpu, device):
    x = x_in_cpu.to(device)
    _, d_sz, h_sz, w_sz, _ = x.shape
    attn_mask = _build_attn_mask(blk, d_sz, h_sz, w_sz, device)

    linears = [blk.attn.qkv, blk.attn.proj, blk.mlp.fc1, blk.mlp.fc2]
    orig_ws = [l.weight.data.clone() for l in linears]

    with torch.no_grad():
        fp32_part1 = blk.forward_part1(x, attn_mask)
        fp32_full = blk(x.clone(), attn_mask)

        for l in linears:
            l.weight.data.copy_(
                symmetric_fake_quant(
                    l.weight.data,
                    LAMBDA_BITS,
                    per_channel=True,
                    channel_dim=0))

        int8_part1 = blk.forward_part1(x, attn_mask)
        int8_full = blk(x.clone(), attn_mask)

        for l, w in zip(linears, orig_ws):
            l.weight.data.copy_(w)

    p1 = _compute_pair_metrics(fp32_part1, int8_part1)
    full = _compute_pair_metrics(fp32_full, int8_full)
    return p1, full


def _safe_stats(arr):
    if not arr:
        return dict(min=None, median=None, max=None)
    a = np.asarray(arr, dtype=np.float64)
    return dict(min=float(np.min(a)), median=float(np.median(a)), max=float(np.max(a)))


def build_summary(rows):
    raw_p1 = [r['lambda_raw_part1'] for r in rows]
    raw_full = [r['lambda_raw_full'] for r in rows]
    ratio_p1 = [r['mse_over_ref_part1'] for r in rows]
    ratio_full = [r['mse_over_ref_full'] for r in rows]

    sat_raw_p1 = sum(v > 0.5 for v in raw_p1)
    sat_raw_full = sum(v > 0.5 for v in raw_full)
    sat_clip_p1 = sum(abs(r['lambda_clip_part1'] - 0.5) < 1e-12 for r in rows)
    sat_clip_full = sum(abs(r['lambda_clip_full'] - 0.5) < 1e-12 for r in rows)

    summary = {
        'n_blocks': len(rows),
        'part1': {
            'raw_gt_0_5_count': sat_raw_p1,
            'raw_gt_0_5_ratio': float(sat_raw_p1 / max(len(rows), 1)),
            'clip_eq_0_5_count': sat_clip_p1,
            'clip_eq_0_5_ratio': float(sat_clip_p1 / max(len(rows), 1)),
            'lambda_raw_stats': _safe_stats(raw_p1),
            'mse_over_ref_stats': _safe_stats(ratio_p1),
        },
        'full': {
            'raw_gt_0_5_count': sat_raw_full,
            'raw_gt_0_5_ratio': float(sat_raw_full / max(len(rows), 1)),
            'clip_eq_0_5_count': sat_clip_full,
            'clip_eq_0_5_ratio': float(sat_clip_full / max(len(rows), 1)),
            'lambda_raw_stats': _safe_stats(raw_full),
            'mse_over_ref_stats': _safe_stats(ratio_full),
        },
    }
    return summary


def save_csv(rows, save_path):
    os.makedirs(osp.dirname(osp.abspath(save_path)), exist_ok=True)
    fields = [
        'stage',
        'block',
        'mse_part1',
        'ref_part1',
        'mse_over_ref_part1',
        'lambda_raw_part1',
        'lambda_clip_part1',
        'mse_full',
        'ref_full',
        'mse_over_ref_full',
        'lambda_raw_full',
        'lambda_clip_full',
    ]
    with open(save_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_json(summary, save_path):
    os.makedirs(osp.dirname(osp.abspath(save_path)), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(summary, f, indent=2)


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg.merge_from_dict(args.cfg_options)

    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA device requested but CUDA is not available.')

    print(f'[LambdaDiag] Building model on {device} ...')
    model = build_model_from_cfg(cfg, args.checkpoint, device)
    backbone = model.backbone

    print('[LambdaDiag] Building test loader ...')
    data_loader = build_test_loader(cfg)

    _disable_checkpoint(backbone)
    handles, samples = register_first_input_hooks(backbone)

    print(f'[LambdaDiag] Collecting first-input samples from {args.calib_batches} batches ...')
    with torch.no_grad():
        for batch_idx, data in enumerate(data_loader):
            if batch_idx >= args.calib_batches:
                break
            if batch_idx == 0 or (batch_idx + 1) % 8 == 0:
                print(f'  batch {batch_idx + 1}/{args.calib_batches}')
            imgs = _extract_single_view(data, device)
            _ = backbone(imgs)

    remove_hooks(handles)
    _restore_checkpoint(backbone)

    rows = []
    n_stages = len(backbone.layers)

    print('[LambdaDiag] Computing per-block diagnostics ...')
    for s_idx in range(n_stages):
        for b_idx, blk in enumerate(backbone.layers[s_idx].blocks):
            key = (s_idx, b_idx)
            if key not in samples:
                continue
            p1, full = observe_block_lambda(blk, samples[key], device)
            row = {
                'stage': s_idx,
                'block': b_idx,
                'mse_part1': p1[0],
                'ref_part1': p1[1],
                'mse_over_ref_part1': p1[2],
                'lambda_raw_part1': p1[3],
                'lambda_clip_part1': p1[4],
                'mse_full': full[0],
                'ref_full': full[1],
                'mse_over_ref_full': full[2],
                'lambda_raw_full': full[3],
                'lambda_clip_full': full[4],
            }
            rows.append(row)

    rows = sorted(rows, key=lambda x: (x['stage'], x['block']))
    summary = build_summary(rows)

    save_csv(rows, args.csv_path)
    save_json(summary, args.json_path)

    print(f'[LambdaDiag] Saved csv -> {args.csv_path}')
    print(f'[LambdaDiag] Saved json -> {args.json_path}')

    n_blocks = summary['n_blocks']
    p1_sat = summary['part1']['clip_eq_0_5_count']
    full_sat = summary['full']['clip_eq_0_5_count']
    print('[LambdaDiag] Summary:')
    print(f'  blocks={n_blocks}')
    print(f'  part1 clip==0.5: {p1_sat}/{n_blocks} '
          f'({summary["part1"]["clip_eq_0_5_ratio"]:.3f})')
    print(f'  full  clip==0.5: {full_sat}/{n_blocks} '
          f'({summary["full"]["clip_eq_0_5_ratio"]:.3f})')
    print(f'  part1 lambda_raw stats: {summary["part1"]["lambda_raw_stats"]}')
    print(f'  full  lambda_raw stats: {summary["full"]["lambda_raw_stats"]}')


if __name__ == '__main__':
    main()
