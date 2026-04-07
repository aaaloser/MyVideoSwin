"""
PTQ Test Script — Video Swin Transformer
=========================================
Pipeline:
  1. Build model + load FP32 checkpoint  (same as tools/test.py)
  2. [Phase 2] Calibrate          → output_pt/calib_stats.pkl
  3. [Phase 3] Quick evaluation   → output_pt/block_bits.json
  4. [Phase 4] Inject fake-quant and run full evaluation

Usage (single-node 4-GPU):
  bash tools/dist_test.sh \
    configs/recognition/swin/swin_small_patch244_window877_kinetics400_1k.py \
    /path/to/checkpoint.pth \
    4 --eval top_k_accuracy \
    --cfg-options \
      data.test.ann_file=/path/to/val.csv \
      data.test.data_prefix=/path/to/kinetics \
    --ptq \
    --calib-batches 32 \
    --quick-batches 200

Pass --ptq to enable PTQ pipeline; without it, falls back to standard FP32 test.
Pass --load-calib  to skip calibration and load existing calib_stats.pkl.
Pass --load-bits   to skip quick_eval and load existing block_bits.json.
"""

import argparse
import json
import os
import os.path as osp
import pickle
import sys
import warnings

# Ensure repo root is on the path when called from tools/
sys.path.insert(0, osp.abspath(osp.join(osp.dirname(__file__), '..')))

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint

from mmaction.datasets import build_dataloader, build_dataset
from mmaction.models import build_model
from mmaction.utils import register_module_hooks

try:
    from mmcv.engine import multi_gpu_test, single_gpu_test
except (ImportError, ModuleNotFoundError):
    from mmaction.apis import multi_gpu_test, single_gpu_test

from quant import calibrate, quick_eval, inject_full_eval, remove_full_eval


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='Video Swin Transformer PTQ Test')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', default=None,
                        help='output result file in pkl/yaml/json format')
    parser.add_argument('--eval', type=str, nargs='+',
                        help='eval metrics, e.g. top_k_accuracy')
    parser.add_argument('--gpu-collect', action='store_true')
    parser.add_argument('--tmpdir', default=None)
    parser.add_argument('--eval-options', nargs='+', action=DictAction, default={})
    parser.add_argument('--cfg-options', nargs='+', action=DictAction, default={})
    parser.add_argument('--average-clips', choices=['score', 'prob', None], default=None)
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'],
                        default='none')
    parser.add_argument('--local_rank', type=int, default=0)

    # PTQ-specific arguments
    parser.add_argument('--ptq', action='store_true',
                        help='Enable PTQ pipeline (calibrate + quick_eval + full_eval)')
    parser.add_argument('--calib-batches', type=int, default=32,
                        help='Number of batches for calibration (Phase 2)')
    parser.add_argument('--quick-batches', type=int, default=50,
                        help='Number of batches for quick evaluation (Phase 3)')
    parser.add_argument('--calib-stats-path', default='output_pt/calib_stats.pkl',
                        help='Path to save/load calibration statistics')
    parser.add_argument('--block-bits-path', default='output_pt/block_bits.json',
                        help='Path to save/load block bit-width assignments')
    parser.add_argument('--load-calib', action='store_true',
                        help='Skip calibration and load existing calib_stats.pkl')
    parser.add_argument('--load-bits', action='store_true',
                        help='Skip quick_eval and load existing block_bits.json')
    parser.add_argument('--high-ratio-threshold', type=float, default=0.30,
                        help='Fraction of windows above τ to assign INT8 (Phase 3)')

    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def turn_off_pretrained(cfg):
    if 'pretrained' in cfg:
        cfg.pretrained = None
    for sub_cfg in cfg.values():
        if isinstance(sub_cfg, dict):
            turn_off_pretrained(sub_cfg)


def build_model_from_cfg(args, cfg):
    if args.average_clips is not None:
        if cfg.model.get('test_cfg') is None and cfg.get('test_cfg') is None:
            cfg.model.setdefault('test_cfg', dict(average_clips=args.average_clips))
        else:
            if cfg.model.get('test_cfg') is not None:
                cfg.model.test_cfg.average_clips = args.average_clips
            else:
                cfg.test_cfg.average_clips = args.average_clips

    turn_off_pretrained(cfg.model)
    model = build_model(cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
    if len(cfg.get('module_hooks', [])) > 0:
        register_module_hooks(model, cfg.module_hooks)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    cfg.merge_from_dict(args.cfg_options)

    output_config = cfg.get('output_config', {})
    if args.out:
        output_config = Config._merge_a_into_b(dict(out=args.out), output_config)
    eval_config = cfg.get('eval_config', {})
    if args.eval:
        eval_config = Config._merge_a_into_b(dict(metrics=args.eval), eval_config)
    if args.eval_options:
        eval_config = Config._merge_a_into_b(args.eval_options, eval_config)

    assert output_config or eval_config, \
        'Please specify --out or --eval'

    # ---- distributed init ----
    distributed = args.launcher != 'none'
    if distributed:
        init_dist(args.launcher, backend='nccl')
    rank, world_size = get_dist_info()

    # ---- build dataset / dataloader ----
    # Replicate the exact logic from tools/test.py so that
    # test_dataloader overrides (videos_per_gpu=1 for max_testing_views) apply.
    dataset = build_dataset(cfg.data.test, dict(test_mode=True))
    # Start with test_dataloader-specific settings (highest priority), then
    # fall back to top-level data settings.
    test_dataloader_default = dict(
        videos_per_gpu=1,
        workers_per_gpu=1,
        dist=distributed,
        shuffle=False,
    )
    test_dataloader_cfg = cfg.data.get('test_dataloader', {})
    dataloader_cfg = {**test_dataloader_default, **test_dataloader_cfg,
                      'dist': distributed, 'shuffle': False}
    data_loader = build_dataloader(dataset, **dataloader_cfg)

    # ---- build model ----
    model = build_model_from_cfg(args, cfg)

    if not args.ptq:
        # ---- Standard FP32 test (fallback) ----
        if not distributed:
            model = MMDataParallel(model, device_ids=[0])
            outputs = single_gpu_test(model, data_loader)
        else:
            model = MMDistributedDataParallel(
                model.cuda(),
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False)
            outputs = multi_gpu_test(model, data_loader,
                                     args.tmpdir, args.gpu_collect)
    else:
        # ---- PTQ pipeline ----
        # Wrap model for single/multi GPU before PTQ phases
        if not distributed:
            model = MMDataParallel(model, device_ids=[0])
        else:
            model = MMDistributedDataParallel(
                model.cuda(),
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False)
        model.eval()

        # Phase 2: Calibration
        if args.load_calib and osp.exists(args.calib_stats_path):
            if rank == 0:
                print(f'[PTQ] Loading calibration stats from {args.calib_stats_path}')
            with open(args.calib_stats_path, 'rb') as f:
                calib_stats = pickle.load(f)
        else:
            if rank == 0:
                print('[PTQ] === Phase 2: Calibration ===')
            calib_stats = calibrate(
                model=model,
                calib_loader=data_loader,
                n_calib_batches=args.calib_batches,
                save_path=args.calib_stats_path,
                verbose=(rank == 0),
            )

        # Phase 3: Quick Evaluation
        if args.load_bits and osp.exists(args.block_bits_path):
            if rank == 0:
                print(f'[PTQ] Loading block_bits from {args.block_bits_path}')
            with open(args.block_bits_path, 'r') as f:
                block_bits = json.load(f)
        else:
            if rank == 0:
                print('[PTQ] === Phase 3: Quick Evaluation ===')
            block_bits = quick_eval(
                model=model,
                mini_loader=data_loader,
                calib_stats=calib_stats,
                n_eval_batches=args.quick_batches,
                save_path=args.block_bits_path,
                high_ratio_threshold=args.high_ratio_threshold,
                verbose=(rank == 0),
            )

        # Phase 4: Full Evaluation with fake quantisation
        if rank == 0:
            print('[PTQ] === Phase 4: Full Evaluation (Fake Quant) ===')
            n8 = sum(1 for v in block_bits.values() if v == 'int8')
            n4 = sum(1 for v in block_bits.values() if v == 'int4')
            print(f'[PTQ] Block allocation: {n8}×INT8, {n4}×INT4')

        inject_full_eval(model, block_bits, calib_stats, verbose=(rank == 0))

        if not distributed:
            outputs = single_gpu_test(model, data_loader)
        else:
            outputs = multi_gpu_test(model, data_loader,
                                     args.tmpdir, args.gpu_collect)

        remove_full_eval(model)

    # ---- Evaluation ----
    if rank == 0:
        if output_config.get('out'):
            out = output_config['out']
            mmcv.mkdir_or_exist(osp.dirname(out))
            mmcv.dump(outputs, out)
            print(f'\nOutput saved to {out}')

        if eval_config:
            eval_res = dataset.evaluate(outputs, **eval_config)
            for name, val in eval_res.items():
                print(f'{name}: {val:.04f}')


if __name__ == '__main__':
    main()
