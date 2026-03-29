import argparse
import os
import os.path as osp
import warnings
import math
import pandas as pd
import numpy as np
import json
import time
from utils import AverageMeter

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.fileio.io import file_handlers
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from mmcv.runner.fp16_utils import wrap_fp16_model

from mmaction.datasets import build_dataloader, build_dataset
from mmaction.models import build_model
from mmaction.utils import register_module_hooks

# # TODO import test functions from mmcv and delete them from mmaction2
# try:
#     from mmcv.engine import multi_gpu_test, single_gpu_test
# except (ImportError, ModuleNotFoundError):
#     warnings.warn(
#         'DeprecationWarning: single_gpu_test, multi_gpu_test, '
#         'collect_results_cpu, collect_results_gpu from mmaction2 will be '
#         'deprecated. Please install mmcv through master branch.')
#     from mmaction.apis import multi_gpu_test, single_gpu_test

# python tools/adv_test.py --eval top_1_accuracy


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMAction2 test (and eval) a model')
    parser.add_argument('config', default='/home/tangchen/pangbo/Video-Swin-Transformer-master/configs/recognition/swin/swin_small_patch244_window877_kinetics400_1k.py',
                        help='test config file path')
    parser.add_argument('checkpoint', default='/data16t/tangchen/pangbo/vit_video_pretrained/swin_small_patch244_window877_kinetics400_1k.pth',
                        help='checkpoint file')
    parser.add_argument(
        '--out',
        default=None,
        help='output result file in pkl/yaml/json format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g.,'
        ' "top_k_accuracy", "mean_class_accuracy" for video dataset')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        default={},
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        default={},
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        default={},
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. For example, '
        "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'")
    parser.add_argument(
        '--average-clips',
        choices=['score', 'prob', None],
        default=None,
        help='average type when averaging test clips')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--onnx',
        action='store_true',
        help='Whether to test with onnx model or not')
    parser.add_argument(
        '--tensorrt',
        action='store_true',
        help='Whether to test with TensorRT engine or not')
    # adv_test
    parser.add_argument('--adv_path', type=str, default='/data16t/tangchen/pangbo/video_data/TT_nonlocal101_k400_True/', help='the path of adversarial examples.')
    parser.add_argument('--gpu', type=str, default='0', help='gpu device.')
    parser.add_argument('--batch_size', type=int, default=16, metavar='N',
                    help='input batch size for reference (default: 16)')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both '
            'specified, --options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args


def turn_off_pretrained(cfg):
    # recursively find all pretrained in the model config,
    # and set them None to avoid redundant pretrain steps for testing
    if 'pretrained' in cfg:
        cfg.pretrained = None

    # recursively turn off pretrained value
    for sub_cfg in cfg.values():
        if isinstance(sub_cfg, dict):
            turn_off_pretrained(sub_cfg)

def get_model(args, cfg, distributed):
    """Get predictions by pytorch models."""
    if args.average_clips is not None:
        # You can set average_clips during testing, it will override the
        # original setting
        if cfg.model.get('test_cfg') is None and cfg.get('test_cfg') is None:
            cfg.model.setdefault('test_cfg',
                                 dict(average_clips=args.average_clips))
        else:
            if cfg.model.get('test_cfg') is not None:
                cfg.model.test_cfg.average_clips = args.average_clips
            else:
                cfg.test_cfg.average_clips = args.average_clips

    # remove redundant pretrain steps for testing
    turn_off_pretrained(cfg.model)

    # build the model and load checkpoint
    model = build_model(
        cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))

    if len(cfg.module_hooks) > 0:
        register_module_hooks(model, cfg.module_hooks)

    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')

    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)

    return model


    # test adv ########################################

def accuracy(output, target):
    batch_size = target.size(0)

    _, pred = output.topk(1, 1, True, True)
    pred = pred.t() # batch_size, 1
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    correct_k = correct[:1].view(-1).float().sum(0)
    return correct_k.mul_(100.0 / batch_size), torch.squeeze(pred)

def generate_batch(batch_files, args):
    batches = []
    labels = []
    for file in batch_files:
        batches.append(torch.from_numpy(np.load(os.path.join(args.adv_path, file))).cuda())
        labels.append(int(file.split('-')[0]))
    labels = np.array(labels).astype(np.int32)
    labels = torch.from_numpy(labels)
    return torch.stack(batches), labels

def reference(model, files_batch, args):
    data_time = AverageMeter()
    top1 = AverageMeter()
    batch_time = AverageMeter()

    predictions = []
    labels = []
    
    # print(model)

    end = time.time()
    with torch.no_grad():
        for step, batch in enumerate(files_batch):
            data_time.update(time.time() - end)
            val_batch, val_label = generate_batch(batch, args)

            val_batch = val_batch.cuda()
            val_label = val_label.cuda()

            batch_size = val_label.size(0)
            outputs = model(val_batch)

            prec1a, preds = accuracy(outputs.data, val_label)

            predictions += list(preds.cpu().numpy())
            labels += list(val_label.cpu().numpy())

            top1.update(prec1a.item(), val_batch.size(0))   
            batch_time.update(time.time() - end)
            end = time.time()

            if step % 5 == 0:
                print('----validation----')
                print_string = 'Process: [{0}/{1}]'.format(step + 1, len(files_batch))
                print(print_string)
                print_string = 'data_time: {data_time:.3f}, batch time: {batch_time:.3f}'.format(
                    data_time=data_time.val,
                    batch_time=batch_time.val)
                print(print_string)
                print_string = 'top-1 accuracy: {top1_acc:.2f}%'.format(top1_acc = top1.avg)
                print (print_string)
    return predictions, labels, top1.avg

def main():
    args = parse_args()

    # if args.tensorrt and args.onnx:
    #     raise ValueError(
    #         'Cannot set onnx mode and tensorrt mode at the same time.')

    cfg = Config.fromfile(args.config)

    cfg.merge_from_dict(args.cfg_options)

    # Load output_config from cfg
    output_config = cfg.get('output_config', {})
    if args.out:
        # Overwrite output_config from args.out
        output_config = Config._merge_a_into_b(
            dict(out=args.out), output_config)

    # Load eval_config from cfg
    eval_config = cfg.get('eval_config', {})
    if args.eval:
        # Overwrite eval_config from args.eval
        eval_config = Config._merge_a_into_b(
            dict(metrics=args.eval), eval_config)
    if args.eval_options:
        # Add options from args.eval_options
        eval_config = Config._merge_a_into_b(args.eval_options, eval_config)

    assert output_config or eval_config, \
        ('Please specify at least one operation (save or eval the '
         'results) with the argument "--out" or "--eval"')

    dataset_type = cfg.data.test.type
    if output_config.get('out', None):
        if 'output_format' in output_config:
            # ugly workround to make recognition and localization the same
            warnings.warn(
                'Skip checking `output_format` in localization task.')
        else:
            out = output_config['out']
            # make sure the dirname of the output path exists
            mmcv.mkdir_or_exist(osp.dirname(out))
            _, suffix = osp.splitext(out)
            if dataset_type == 'AVADataset':
                assert suffix[1:] == 'csv', ('For AVADataset, the format of '
                                             'the output file should be csv')
            else:
                assert suffix[1:] in file_handlers, (
                    'The format of the output '
                    'file should be json, pickle or yaml')

    # set cudnn benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.data.test.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # The flag is used to register module's hooks
    cfg.setdefault('module_hooks', [])

    model = get_model(args, cfg, distributed)

    ###############################
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # loading adversarial examples.
    files = os.listdir(args.adv_path)
    files = [i for i in files if 'adv' in i]

    batch_times = math.ceil(len(files) / args.batch_size)
    files_batch = []
    for i in range(batch_times):
        batch = files[i*args.batch_size: min((i+1)*args.batch_size, len(files))]
        files_batch.append(batch)

    model_val_acc = {}
    model_attack_rate = {}
    attack_success_avg = 0
    info_df = pd.DataFrame()
    info_df['gt_label'] = [i for i in range(400)]
    
    device = torch.device(args.device)
    # checkpoint = torch.load(args.use_checkpoint, map_location=device)
    # model.load_state_dict(checkpoint)
    
    model.cuda()
    model.eval()
    preds, labels, top1_avg = reference(model, files_batch, args=args)
    
    attack_success_rate = round((100 - top1_avg), 1)
    attack_success_avg += top1_avg

    predd = np.zeros_like(preds)
    inds = np.argsort(labels)
    for i,ind in enumerate(inds):
        predd[ind] = preds[i]

    model_name = 'swin_vit'
    
    info_df['{}-pre'.format(model_name)] = predd
    model_val_acc[model_name] = top1_avg
    model_attack_rate[model_name] = attack_success_rate
    del model
    torch.cuda.empty_cache()

    attack_success_avg = round((100 - attack_success_avg) / 1, 1)
    model_attack_rate['Average Success Rate'] = attack_success_avg
    with open(os.path.join(args.adv_path, 'average_success_rate.json'), 'w') as opt:
        json.dump(model_attack_rate, opt)
    
    info_df.to_csv(os.path.join(args.adv_path, 'results_all_models_prediction.csv'), index=False)
    with open(os.path.join(args.adv_path, 'top1_acc_all_models.json'), 'w') as opt:
        json.dump(model_val_acc, opt)
        
    consist_data = {
        'gt_label': labels,  # 第一列：真实标签
        'swin_vit-pre': preds     # 第二列：预测值
    }
    df = pd.DataFrame(consist_data)

    # csv_file_path = "/home/tangchen/pangbo/VideoMamba-main/videomamba/video_sm/csv_data_analyze/true_real_output.csv"  # 输出文件路径
    # df.to_csv(csv_file_path, index=False)


if __name__ == '__main__':
    main()
