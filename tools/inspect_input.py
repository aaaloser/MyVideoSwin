"""
inspect_input.py
================
可视化模型真实接收到的输入 tensor。

打印内容：
  - data['imgs'] 的 shape / dtype / 值域
  - 反归一化后每帧的 RGB 图像（按时间顺序排成一张大图保存）

用法：
  python tools/inspect_input.py \
    configs/recognition/swin/swin_small_patch244_window877_kinetics400_1k.py \
    --ann-file /data/liyifan24/Datasets/Kinetics-400/val.csv \
    --data-prefix /data/liyifan24/Datasets/Kinetics-400 \
    --n-samples 2 \
    --out-dir /data/liyifan24/Video-Swin-Transformer-master/output_pt/
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
from mmcv import Config
from mmaction.datasets import build_dataloader, build_dataset

try:
    from PIL import Image
except ImportError:
    Image = None


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def denormalize(tensor: torch.Tensor, mean, std) -> np.ndarray:
    """
    tensor: (C, T, H, W)  float32, normalised
    returns: (T, H, W, 3)  uint8
    """
    t = tensor.clone().cpu()  # C T H W
    for c, (m, s) in enumerate(zip(mean, std)):
        t[c] = t[c] * s + m
    t = t.clamp(0, 255).byte()            # C T H W
    t = t.permute(1, 2, 3, 0).numpy()    # T H W C   (RGB)
    return t


def make_grid(frames: np.ndarray, cols: int = 8) -> np.ndarray:
    """
    frames: (T, H, W, 3)  uint8
    Returns a single image with frames laid out in a grid.
    """
    T, H, W, C = frames.shape
    rows = (T + cols - 1) // cols
    grid = np.zeros((rows * H, cols * W, C), dtype=np.uint8)
    for i, frame in enumerate(frames):
        r, c = divmod(i, cols)
        grid[r * H:(r + 1) * H, c * W:(c + 1) * W] = frame
    return grid


def save_png(array: np.ndarray, path: str):
    if Image is None:
        print(f'  [skip] Pillow not installed — cannot save {path}')
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    Image.fromarray(array).save(path)
    print(f'  Saved  → {path}')


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('config', help='model config file (used only for pipeline/norm)')
    p.add_argument('--ann-file',    required=True, help='val.csv path')
    p.add_argument('--data-prefix', required=True, help='video root directory')
    p.add_argument('--n-samples',   type=int, default=2,
                   help='how many samples to inspect')
    p.add_argument('--out-dir',     default='/tmp/inspect_input',
                   help='directory to save output images')
    p.add_argument('--pipeline',    choices=['test', 'val'], default='test',
                   help='which pipeline to use (test = 4 clips × 3 crops)')
    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    # Apply CLI overrides
    cfg.data.test.ann_file    = args.ann_file
    cfg.data.test.data_prefix = args.data_prefix

    # Read normalisation params from config
    norm_cfg = cfg.get('img_norm_cfg', dict(
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375]))
    mean = norm_cfg['mean']
    std  = norm_cfg['std']

    # Build dataset with the chosen pipeline
    pipeline_key  = 'test_pipeline' if args.pipeline == 'test' else 'val_pipeline'
    cfg.data.test.pipeline = cfg.get(pipeline_key, cfg.data.test.pipeline)

    dataset = build_dataset(cfg.data.test, dict(test_mode=True))
    loader  = build_dataloader(
        dataset,
        videos_per_gpu=1,
        workers_per_gpu=0,   # no multiprocessing — cleaner debug output
        dist=False,
        shuffle=False,
    )

    print(f'\n{"="*60}')
    print(f'Dataset pipeline : {args.pipeline}')
    print(f'Total samples    : {len(dataset)}')
    print(f'Inspecting first : {args.n_samples} sample(s)')
    print(f'{"="*60}\n')

    for sample_idx, data in enumerate(loader):
        if sample_idx >= args.n_samples:
            break

        imgs = data['imgs']   # (B, N_clips*N_crops, C, T, H, W)  or  (B, C, T, H, W)

        print(f'--- Sample {sample_idx} ---')
        print(f'  imgs.shape  : {tuple(imgs.shape)}')
        print(f'  imgs.dtype  : {imgs.dtype}')
        print(f'  value range : min={imgs.min():.3f}  max={imgs.max():.3f}  '
              f'mean={imgs.mean():.3f}')

        # Determine layout
        if imgs.dim() == 6:
            # (B, num_views, C, T, H, W)  — test pipeline w/ multi-view
            B, N, C, T, H, W = imgs.shape
            print(f'  Layout      : B={B}, num_views={N}, C={C}, T={T}, H={H}, W={W}')
            # Take first sample, first view
            clip = imgs[0, 0]   # (C, T, H, W)
            n_views = N
        else:
            # (B, C, T, H, W)  — val / single-view
            B, C, T, H, W = imgs.shape
            print(f'  Layout      : B={B}, C={C}, T={T}, H={H}, W={W}')
            clip = imgs[0]      # (C, T, H, W)
            n_views = 1

        print(f'  num_views   : {n_views}')
        print(f'  clip shape  : {tuple(clip.shape)}  (C×T×H×W, first view)')

        # Per-frame pixel stats (after denorm)
        frames = denormalize(clip, mean, std)   # (T, H, W, 3)  uint8
        print(f'  After denorm: min={frames.min()}  max={frames.max()}  '
              f'mean={frames.mean():.1f}')

        # Save grid image
        grid = make_grid(frames, cols=8)
        out_path = os.path.join(args.out_dir,
                                f'sample{sample_idx:03d}_view0_T{T}.png')
        save_png(grid, out_path)

        # If multi-view, also save view grid (T=1 to show all views' first frame)
        if n_views > 1:
            first_frames = []
            for v in range(min(n_views, 12)):
                clip_v = imgs[0, v]
                frames_v = denormalize(clip_v, mean, std)
                first_frames.append(frames_v[0])   # first frame only
            view_grid = make_grid(np.stack(first_frames), cols=n_views)
            vpath = os.path.join(args.out_dir,
                                 f'sample{sample_idx:03d}_all_views_frame0.png')
            save_png(view_grid, vpath)

        print()

    print('Done.')


if __name__ == '__main__':
    main()
