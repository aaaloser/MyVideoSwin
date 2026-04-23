import argparse
import csv
import json
import os
import os.path as osp
import sys

import numpy as np
import torch
from mmcv import Config, DictAction
from mmcv.runner import load_checkpoint

# Ensure repo root is on PYTHONPATH when called from tools/analysis/
sys.path.insert(0, osp.abspath(osp.join(osp.dirname(__file__), '../..')))

from mmaction.datasets import build_dataloader, build_dataset  # noqa: E402
from mmaction.models import build_model  # noqa: E402
from quant.calibrate import (  # noqa: E402
    EPS,
    _build_attn_mask,
    _disable_checkpoint,
    _extract_single_view,
    _restore_checkpoint,
)
from quant.utils import symmetric_fake_quant  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description='Diagnose per-block W8A8 vs W4A16 local distortion gap')
    parser.add_argument('config', help='Test config file path')
    parser.add_argument('checkpoint', help='Checkpoint file path')
    parser.add_argument('--collect-batches', type=int, default=2,
                        help='Max batches used to collect block input samples')
    parser.add_argument('--samples-per-block', type=int, default=1,
                        help='How many input samples to keep for each block')
    parser.add_argument('--device', default='cuda:0',
                        help='Device for diagnosis, e.g. cuda:0 or cpu')
    parser.add_argument('--block-bits-path', default='output_pt/block_bits_t020_f4b9_3g.json',
                        help='Optional block bits json to annotate current bit assignment')
    parser.add_argument('--csv-path', default='output_pt/bitgap_diag_3g.csv',
                        help='Path to save per-block diagnosis csv')
    parser.add_argument('--json-path', default='output_pt/bitgap_diag_3g_summary.json',
                        help='Path to save summary json')
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
    return build_dataloader(dataset, **dataloader_cfg)


def _block_forward_fp(blk, x, attn_mask):
    shortcut = x
    x_attn = blk.forward_part1(x, attn_mask)
    x = shortcut + blk.drop_path(x_attn)
    x = x + blk.forward_part2(x)
    return x


def _get_block_linears(blk):
    return [blk.attn.qkv, blk.attn.proj, blk.mlp.fc1, blk.mlp.fc2]


def _apply_weight_fake_quant(linears, bits):
    backup = [layer.weight.data.clone() for layer in linears]
    for layer in linears:
        layer.weight.data.copy_(
            symmetric_fake_quant(layer.weight.data, bits,
                                 per_channel=True, channel_dim=0)
        )
    return backup


def _restore_weight(linears, backup):
    for layer, weight in zip(linears, backup):
        layer.weight.data.copy_(weight)


def evaluate_sample_bitgap(blk, x_in_cpu, device):
    x = x_in_cpu.to(device)
    _, d, h, w, _ = x.shape
    attn_mask = _build_attn_mask(blk, d, h, w, device)

    with torch.no_grad():
        x_fp = _block_forward_fp(blk, x, attn_mask)

        linears = _get_block_linears(blk)

        bk8 = _apply_weight_fake_quant(linears, 8)
        x_w8 = _block_forward_fp(blk, x, attn_mask)
        _restore_weight(linears, bk8)

        x_w8a8 = symmetric_fake_quant(x_w8, 8, per_channel=False)

        bk4 = _apply_weight_fake_quant(linears, 4)
        x_w4a16 = _block_forward_fp(blk, x, attn_mask)
        _restore_weight(linears, bk4)

    ref = x_fp.pow(2).mean().item() + EPS

    mse_w8 = (x_fp - x_w8).pow(2).mean().item()
    mse_w8a8 = (x_fp - x_w8a8).pow(2).mean().item()
    mse_w4a16 = (x_fp - x_w4a16).pow(2).mean().item()
    mse_act_quant = (x_w8 - x_w8a8).pow(2).mean().item()

    return {
        'rel_w8': mse_w8 / ref,
        'rel_w8a8': mse_w8a8 / ref,
        'rel_w4a16': mse_w4a16 / ref,
        'rel_act_q': mse_act_quant / ref,
        'delta_rel_8_minus_4': (mse_w8a8 - mse_w4a16) / ref,
    }


def collect_block_samples(backbone, data_loader, collect_batches, samples_per_block, device):
    for stage in backbone.layers:
        for blk in stage.blocks:
            if hasattr(blk, '_bitgap_samples'):
                delattr(blk, '_bitgap_samples')

    handles = []

    def make_hook(blk):
        def _hook(module, inputs, output):
            if not hasattr(blk, '_bitgap_samples'):
                blk._bitgap_samples = []
            if len(blk._bitgap_samples) < samples_per_block:
                blk._bitgap_samples.append(inputs[0][:1].detach().cpu())
        return _hook

    for stage in backbone.layers:
        for blk in stage.blocks:
            handles.append(blk.register_forward_hook(make_hook(blk)))

    def all_full():
        for stage in backbone.layers:
            for blk in stage.blocks:
                if len(getattr(blk, '_bitgap_samples', [])) < samples_per_block:
                    return False
        return True

    with torch.no_grad():
        for batch_idx, data in enumerate(data_loader):
            if batch_idx >= collect_batches:
                break
            imgs = _extract_single_view(data, device)
            _ = backbone(imgs)
            if all_full():
                break

    for h in handles:
        h.remove()


def analyze_blocks(backbone, device, current_bits):
    rows = []

    for s_idx, stage in enumerate(backbone.layers):
        for b_idx, blk in enumerate(stage.blocks):
            key = 'stage{}_block{}'.format(s_idx, b_idx)
            samples = getattr(blk, '_bitgap_samples', [])
            if not samples:
                continue

            per_sample = []
            for x_cpu in samples:
                per_sample.append(evaluate_sample_bitgap(blk, x_cpu, device))

            rel_w8a8 = float(np.mean([d['rel_w8a8'] for d in per_sample]))
            rel_w4a16 = float(np.mean([d['rel_w4a16'] for d in per_sample]))
            rel_w8 = float(np.mean([d['rel_w8'] for d in per_sample]))
            rel_act_q = float(np.mean([d['rel_act_q'] for d in per_sample]))
            delta = float(np.mean([d['delta_rel_8_minus_4'] for d in per_sample]))

            local_choice = 'int8' if rel_w8a8 < rel_w4a16 else 'int4'
            current_bit = current_bits.get(key, '')

            rows.append({
                'block': key,
                'stage': s_idx,
                'block_idx': b_idx,
                'n_samples': len(samples),
                'current_bit': current_bit,
                'rel_w8a8': rel_w8a8,
                'rel_w4a16': rel_w4a16,
                'delta_rel_8_minus_4': delta,
                'rel_w8_weight_only': rel_w8,
                'rel_activation_quant': rel_act_q,
                'local_recommend': local_choice,
            })

    rows.sort(key=lambda x: (x['stage'], x['block_idx']))
    return rows


def summarize(rows):
    if not rows:
        return {
            'n_blocks': 0,
            'int8_better_count': 0,
            'int4_better_count': 0,
            'equal_count': 0,
            'top_int4_advantage_blocks': [],
            'top_int8_advantage_blocks': [],
            'candidate_force_int4': [],
            'candidate_force_int8': [],
        }

    eps = 1e-9
    int8_better = [r for r in rows if r['delta_rel_8_minus_4'] < -eps]
    int4_better = [r for r in rows if r['delta_rel_8_minus_4'] > eps]
    equal = [r for r in rows if abs(r['delta_rel_8_minus_4']) <= eps]

    top_int4_adv = sorted(int4_better, key=lambda r: r['delta_rel_8_minus_4'], reverse=True)
    top_int8_adv = sorted(int8_better, key=lambda r: r['delta_rel_8_minus_4'])

    cand_force_int4 = [
        r for r in top_int4_adv
        if r['current_bit'] == 'int8'
    ][:8]
    cand_force_int8 = [
        r for r in top_int8_adv
        if r['current_bit'] == 'int4'
    ][:8]

    def brief(items):
        out = []
        for r in items[:8]:
            out.append({
                'block': r['block'],
                'current_bit': r['current_bit'],
                'delta_rel_8_minus_4': r['delta_rel_8_minus_4'],
                'rel_w8a8': r['rel_w8a8'],
                'rel_w4a16': r['rel_w4a16'],
            })
        return out

    return {
        'n_blocks': len(rows),
        'int8_better_count': len(int8_better),
        'int4_better_count': len(int4_better),
        'equal_count': len(equal),
        'top_int4_advantage_blocks': brief(top_int4_adv),
        'top_int8_advantage_blocks': brief(top_int8_adv),
        'candidate_force_int4': brief(cand_force_int4),
        'candidate_force_int8': brief(cand_force_int8),
    }


def write_csv(rows, csv_path):
    os.makedirs(osp.dirname(osp.abspath(csv_path)), exist_ok=True)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                'block',
                'stage',
                'block_idx',
                'n_samples',
                'current_bit',
                'rel_w8a8',
                'rel_w4a16',
                'delta_rel_8_minus_4',
                'rel_w8_weight_only',
                'rel_activation_quant',
                'local_recommend',
            ]
        )
        writer.writeheader()
        writer.writerows(rows)


def load_current_bits(path):
    if not path or (not osp.exists(path)):
        return {}
    with open(path, 'r') as f:
        return json.load(f)


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    cfg.merge_from_dict(args.cfg_options)

    print('[BitGapDiag] Building model and loader...')
    model = build_model_from_cfg(cfg, args.checkpoint, args.device)
    data_loader = build_test_loader(cfg)

    backbone = model.backbone if not hasattr(model, 'module') else model.module.backbone

    _disable_checkpoint(backbone)

    print('[BitGapDiag] Collecting block input samples...')
    collect_block_samples(
        backbone=backbone,
        data_loader=data_loader,
        collect_batches=args.collect_batches,
        samples_per_block=args.samples_per_block,
        device=args.device,
    )

    current_bits = load_current_bits(args.block_bits_path)

    print('[BitGapDiag] Evaluating per-block W8A8 vs W4A16 gap...')
    rows = analyze_blocks(backbone, args.device, current_bits)

    summary = summarize(rows)

    os.makedirs(osp.dirname(osp.abspath(args.json_path)), exist_ok=True)
    with open(args.json_path, 'w') as f:
        json.dump(summary, f, indent=2)

    write_csv(rows, args.csv_path)

    _restore_checkpoint(backbone)

    print('[BitGapDiag] Saved CSV -> {}'.format(args.csv_path))
    print('[BitGapDiag] Saved JSON -> {}'.format(args.json_path))
    print('[BitGapDiag] int8_better={}, int4_better={}, equal={}'.format(
        summary['int8_better_count'], summary['int4_better_count'], summary['equal_count']))

    if summary['candidate_force_int4']:
        print('[BitGapDiag] Top candidate_force_int4: {}'.format(
            ', '.join([x['block'] for x in summary['candidate_force_int4'][:5]])))
    if summary['candidate_force_int8']:
        print('[BitGapDiag] Top candidate_force_int8: {}'.format(
            ', '.join([x['block'] for x in summary['candidate_force_int8'][:5]])))


if __name__ == '__main__':
    main()
