import argparse
import csv
import datetime as dt
import os
import os.path as osp
import re
import subprocess
import sys
from typing import Dict, List, Optional


# Ensure repo root is on PYTHONPATH when called from tools/analysis/
REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), '../..'))
sys.path.insert(0, REPO_ROOT)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run PTQ experiment matrix and auto-collect top1/top5 from logs')
    parser.add_argument(
        '--config',
        default='configs/recognition/swin/swin_small_patch244_window877_kinetics400_1k.py',
        help='Model config path')
    parser.add_argument(
        '--checkpoint',
        default='/data/liyifan24/vit_video_pretrained/swin_small_patch244_window877_kinetics400_1k.pth',
        help='Checkpoint path')
    parser.add_argument(
        '--ann-file',
        default='/data/liyifan24/Datasets/Kinetics-400/val.csv',
        help='Annotation file for evaluation')
    parser.add_argument(
        '--data-prefix',
        default='/data/liyifan24/Datasets/Kinetics-400',
        help='Dataset prefix for evaluation')
    parser.add_argument(
        '--cuda-visible-devices',
        default='1,2,3',
        help='CUDA_VISIBLE_DEVICES value used for each run')
    parser.add_argument(
        '--num-gpus',
        type=int,
        default=3,
        help='World size argument passed to dist_ptq_test.sh')
    parser.add_argument(
        '--calib-stats-path',
        default='output_pt/calib_stats_anchor_3g.pkl',
        help='Calibration stats path used with --load-calib')
    parser.add_argument(
        '--quick-batches',
        type=int,
        default=400,
        help='Quick eval batches')
    parser.add_argument(
        '--results-csv',
        default='output_pt/ptq_matrix_results_3g.csv',
        help='CSV file to append experiment results')
    parser.add_argument(
        '--preset',
        default='b9_refine',
        choices=['b9_refine', 'b9_bitgap'],
        help='Built-in experiment preset')
    parser.add_argument(
        '--experiments',
        nargs='*',
        default=None,
        help='Optional subset of experiment names to run')
    parser.add_argument(
        '--rerun-existing',
        action='store_true',
        help='Rerun experiments already present in results CSV')
    parser.add_argument(
        '--stop-on-error',
        action='store_true',
        help='Stop immediately when one experiment fails')
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print commands only, do not execute')
    return parser.parse_args()


def get_preset(name: str) -> List[Dict]:
    if name == 'b9_refine':
        # This preset follows the current successful direction:
        # keep stage2_block9 forced INT4 and perform local threshold search.
        return [
            {
                'name': 't020_f4b9',
                'threshold': 0.20,
                'force_int8': [],
                'force_int4': ['stage2_block9'],
                'block_bits_path': 'output_pt/matrix_bits_3g/block_bits_t020_f4b9.json',
                'save_quant_model': 'output_pt/matrix_models_3g/swin_s_t020_f4b9.pth',
            },
            {
                'name': 't025_f4b9',
                'threshold': 0.25,
                'force_int8': [],
                'force_int4': ['stage2_block9'],
                'block_bits_path': 'output_pt/matrix_bits_3g/block_bits_t025_f4b9.json',
                'save_quant_model': 'output_pt/matrix_models_3g/swin_s_t025_f4b9.pth',
            },
            {
                'name': 't030_f4b9',
                'threshold': 0.30,
                'force_int8': [],
                'force_int4': ['stage2_block9'],
                'block_bits_path': 'output_pt/matrix_bits_3g/block_bits_t030_f4b9.json',
                'save_quant_model': 'output_pt/matrix_models_3g/swin_s_t030_f4b9.pth',
            },
            {
                'name': 't030_f4b9_f8b1',
                'threshold': 0.30,
                'force_int8': ['stage2_block1'],
                'force_int4': ['stage2_block9'],
                'block_bits_path': 'output_pt/matrix_bits_3g/block_bits_t030_f4b9_f8b1.json',
                'save_quant_model': 'output_pt/matrix_models_3g/swin_s_t030_f4b9_f8b1.pth',
            },
        ]

    if name == 'b9_bitgap':
        # Fix threshold at 0.20 and test bitgap-driven block overrides.
        return [
            {
                'name': 't020_f4b9',
                'threshold': 0.20,
                'force_int8': [],
                'force_int4': ['stage2_block9'],
                'block_bits_path': 'output_pt/matrix_bits_3g/block_bits_t020_f4b9.json',
                'save_quant_model': 'output_pt/matrix_models_3g/swin_s_t020_f4b9.pth',
            },
            {
                'name': 't020_f4b9_f4b11',
                'threshold': 0.20,
                'force_int8': [],
                'force_int4': ['stage2_block9', 'stage2_block11'],
                'block_bits_path': 'output_pt/matrix_bits_3g/block_bits_t020_f4b9_f4b11.json',
                'save_quant_model': 'output_pt/matrix_models_3g/swin_s_t020_f4b9_f4b11.pth',
            },
            {
                'name': 't020_f4b9_f4b13',
                'threshold': 0.20,
                'force_int8': [],
                'force_int4': ['stage2_block9', 'stage2_block13'],
                'block_bits_path': 'output_pt/matrix_bits_3g/block_bits_t020_f4b9_f4b13.json',
                'save_quant_model': 'output_pt/matrix_models_3g/swin_s_t020_f4b9_f4b13.pth',
            },
            {
                'name': 't020_f4b9_f4b11_f4b13',
                'threshold': 0.20,
                'force_int8': [],
                'force_int4': ['stage2_block9', 'stage2_block11', 'stage2_block13'],
                'block_bits_path': 'output_pt/matrix_bits_3g/block_bits_t020_f4b9_f4b11_f4b13.json',
                'save_quant_model': 'output_pt/matrix_models_3g/swin_s_t020_f4b9_f4b11_f4b13.pth',
            },
            {
                'name': 't020_f4b9_f4b11_f8b0',
                'threshold': 0.20,
                'force_int8': ['stage0_block0'],
                'force_int4': ['stage2_block9', 'stage2_block11'],
                'block_bits_path': 'output_pt/matrix_bits_3g/block_bits_t020_f4b9_f4b11_f8b0.json',
                'save_quant_model': 'output_pt/matrix_models_3g/swin_s_t020_f4b9_f4b11_f8b0.pth',
            },
        ]

    raise ValueError('Unknown preset: {}'.format(name))


def ensure_parent(path: str):
    parent = osp.dirname(osp.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def load_existing_names(csv_path: str) -> List[str]:
    if not osp.exists(csv_path):
        return []

    names = []
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            status = row.get('status', '').strip().lower()
            name = row.get('exp_name', '').strip()
            # Only skip already successful experiments.
            if name and status == 'ok':
                names.append(name)
    return names


def init_csv_if_needed(csv_path: str):
    if osp.exists(csv_path):
        return

    ensure_parent(csv_path)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                'timestamp',
                'exp_name',
                'threshold',
                'force_int8',
                'force_int4',
                'int8_blocks',
                'int4_blocks',
                'top1',
                'top5',
                'duration_sec',
                'status',
                'log_path',
                'command',
            ],
        )
        writer.writeheader()


def build_command(args, exp: Dict) -> List[str]:
    cmd = [
        'bash', 'tools/dist_ptq_test.sh',
        args.config,
        args.checkpoint,
        str(args.num_gpus),
        '--eval', 'top_k_accuracy',
        '--cfg-options',
        'data.test.ann_file={}'.format(args.ann_file),
        'data.test.data_prefix={}'.format(args.data_prefix),
        '--ptq',
        '--load-calib',
        '--calib-stats-path', args.calib_stats_path,
        '--quick-batches', str(args.quick_batches),
        '--high-ratio-threshold', '{:.3f}'.format(exp['threshold']),
        '--block-bits-path', exp['block_bits_path'],
        '--save-quant-model', exp['save_quant_model'],
    ]

    if exp.get('force_int8'):
        cmd += ['--force-int8-blocks'] + exp['force_int8']
    if exp.get('force_int4'):
        cmd += ['--force-int4-blocks'] + exp['force_int4']

    return cmd


def parse_metrics_from_log(log_path: str) -> Dict[str, Optional[str]]:
    top1 = None
    top5 = None
    int8_blocks = None
    int4_blocks = None

    r_top1 = re.compile(r'^top1_acc[:\t ]+([0-9.]+)')
    r_top5 = re.compile(r'^top5_acc[:\t ]+([0-9.]+)')
    r_alloc = re.compile(r'Block allocation:\s*(\d+)×INT8.*?,\s*(\d+)×INT4')

    with open(log_path, 'r') as f:
        for line in f:
            s = line.strip()
            m1 = r_top1.match(s)
            if m1:
                top1 = m1.group(1)
            m2 = r_top5.match(s)
            if m2:
                top5 = m2.group(1)
            ma = r_alloc.search(s)
            if ma:
                int8_blocks = ma.group(1)
                int4_blocks = ma.group(2)

    return {
        'top1': top1,
        'top5': top5,
        'int8_blocks': int8_blocks,
        'int4_blocks': int4_blocks,
    }


def resolve_log_path(log_rel_or_abs: Optional[str], start_time: float) -> Optional[str]:
    if log_rel_or_abs:
        if osp.isabs(log_rel_or_abs):
            p = log_rel_or_abs
        else:
            p = osp.join(REPO_ROOT, log_rel_or_abs)
        if osp.exists(p):
            return p

    logs_dir = osp.join(REPO_ROOT, 'logs')
    if not osp.isdir(logs_dir):
        return None

    cands = []
    for name in os.listdir(logs_dir):
        if not name.startswith('ptq_') or not name.endswith('.log'):
            continue
        p = osp.join(logs_dir, name)
        if osp.getmtime(p) >= start_time - 2.0:
            cands.append(p)

    if not cands:
        return None
    cands.sort(key=lambda x: osp.getmtime(x), reverse=True)
    return cands[0]


def run_one(args, exp: Dict) -> Dict:
    cmd = build_command(args, exp)
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = args.cuda_visible_devices

    print('')
    print('============================================================')
    print('[Matrix] Running {} | thr={} | force_int8={} | force_int4={}'.format(
        exp['name'], exp['threshold'], exp.get('force_int8', []), exp.get('force_int4', [])))
    print('[Matrix] Command: {}'.format(' '.join(cmd)))
    print('============================================================')

    if args.dry_run:
        return {
            'status': 'dry_run',
            'log_path': '',
            'top1': '',
            'top5': '',
            'int8_blocks': '',
            'int4_blocks': '',
            'duration_sec': '0',
            'command': 'CUDA_VISIBLE_DEVICES={} {}'.format(args.cuda_visible_devices, ' '.join(cmd)),
        }

    start_dt = dt.datetime.now()
    start_ts = start_dt.timestamp()
    log_path_hint = None

    proc = subprocess.Popen(
        cmd,
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )

    log_re = re.compile(r'\[ptq_test\] Log\s*→\s*(\S+\.log)')

    while True:
        line = proc.stdout.readline()
        if not line and proc.poll() is not None:
            break
        if not line:
            continue
        print(line, end='')
        m = log_re.search(line)
        if m:
            log_path_hint = m.group(1)

    ret = proc.wait()
    end_dt = dt.datetime.now()
    duration = int((end_dt - start_dt).total_seconds())

    log_path = resolve_log_path(log_path_hint, start_ts)
    if not log_path:
        status = 'no_log'
        metrics = {
            'top1': '',
            'top5': '',
            'int8_blocks': '',
            'int4_blocks': '',
        }
    else:
        metrics = parse_metrics_from_log(log_path)
        missing = (not metrics['top1']) or (not metrics['top5'])
        if ret != 0:
            status = 'failed'
        elif missing:
            status = 'missing_metrics'
        else:
            status = 'ok'

    if ret != 0 and status != 'failed':
        status = 'failed'

    return {
        'status': status,
        'log_path': log_path or '',
        'top1': metrics['top1'] or '',
        'top5': metrics['top5'] or '',
        'int8_blocks': metrics['int8_blocks'] or '',
        'int4_blocks': metrics['int4_blocks'] or '',
        'duration_sec': str(duration),
        'command': 'CUDA_VISIBLE_DEVICES={} {}'.format(args.cuda_visible_devices, ' '.join(cmd)),
    }


def append_csv_row(csv_path: str, row: Dict):
    init_csv_if_needed(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                'timestamp',
                'exp_name',
                'threshold',
                'force_int8',
                'force_int4',
                'int8_blocks',
                'int4_blocks',
                'top1',
                'top5',
                'duration_sec',
                'status',
                'log_path',
                'command',
            ],
        )
        writer.writerow(row)


def print_ranking(csv_path: str):
    rows = []
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                top1 = float(row.get('top1', ''))
            except ValueError:
                continue
            row['_top1f'] = top1
            rows.append(row)

    if not rows:
        print('[Matrix] No valid top1 rows in {}'.format(csv_path))
        return

    rows.sort(key=lambda r: r['_top1f'], reverse=True)
    print('')
    print('[Matrix] Top results (sorted by top1):')
    for i, r in enumerate(rows[:10], 1):
        print('  {}. {} | top1={} top5={} | INT8/INT4={}/{} | {}'.format(
            i,
            r.get('exp_name', ''),
            r.get('top1', ''),
            r.get('top5', ''),
            r.get('int8_blocks', ''),
            r.get('int4_blocks', ''),
            r.get('log_path', ''),
        ))


def main():
    args = parse_args()

    if not osp.isabs(args.config):
        args.config = osp.normpath(args.config)
    if not osp.isabs(args.results_csv):
        args.results_csv = osp.normpath(args.results_csv)

    exps = get_preset(args.preset)

    if args.experiments:
        allow = set(args.experiments)
        exps = [e for e in exps if e['name'] in allow]
        if not exps:
            raise ValueError('No experiments selected after filtering: {}'.format(args.experiments))

    ensure_parent(args.results_csv)
    ensure_parent('output_pt/matrix_bits_3g/dummy.txt')
    ensure_parent('output_pt/matrix_models_3g/dummy.txt')

    existing = set(load_existing_names(args.results_csv))

    print('[Matrix] Repo root: {}'.format(REPO_ROOT))
    print('[Matrix] Preset: {}'.format(args.preset))
    print('[Matrix] Result CSV: {}'.format(args.results_csv))
    print('[Matrix] Existing rows: {}'.format(len(existing)))

    for exp in exps:
        name = exp['name']
        if (name in existing) and (not args.rerun_existing):
            print('[Matrix] Skip existing experiment: {}'.format(name))
            continue

        outcome = run_one(args, exp)

        if args.dry_run:
            continue

        row = {
            'timestamp': dt.datetime.now().isoformat(timespec='seconds'),
            'exp_name': name,
            'threshold': '{:.3f}'.format(exp['threshold']),
            'force_int8': ' '.join(exp.get('force_int8', [])),
            'force_int4': ' '.join(exp.get('force_int4', [])),
            'int8_blocks': outcome['int8_blocks'],
            'int4_blocks': outcome['int4_blocks'],
            'top1': outcome['top1'],
            'top5': outcome['top5'],
            'duration_sec': outcome['duration_sec'],
            'status': outcome['status'],
            'log_path': outcome['log_path'],
            'command': outcome['command'],
        }
        append_csv_row(args.results_csv, row)

        if outcome['status'] != 'ok':
            print('[Matrix] Warning: {} finished with status={}'.format(name, outcome['status']))
            if args.stop_on_error:
                print('[Matrix] stop-on-error enabled, exiting')
                break

    print_ranking(args.results_csv)


if __name__ == '__main__':
    main()
