复现推理（原始 FP32）
```bash
bash tools/dist_test.sh \
configs/recognition/swin/swin_small_patch244_window877_kinetics400_1k.py \
/data/liyifan24/vit_video_pretrained/swin_small_patch244_window877_kinetics400_1k.pth \
4 --eval top_k_accuracy \
--cfg-options \
data.test.ann_file=/data/liyifan24/Datasets/Kinetics-400/val.csv \
data.test.data_prefix=/data/liyifan24/Datasets/Kinetics-400
```

PTQ 推理（完整流程：校准 → 快速评估 → 全量伪量化测试）
```bash
CUDA_VISIBLE_DEVICES=3 bash tools/dist_ptq_test.sh \
configs/recognition/swin/swin_small_patch244_window877_kinetics400_1k.py \
/data/liyifan24/vit_video_pretrained/swin_small_patch244_window877_kinetics400_1k.pth \
1 --eval top_k_accuracy \
--cfg-options \
data.test.ann_file=/data/liyifan24/Datasets/Kinetics-400/val.csv \
data.test.data_prefix=/data/liyifan24/Datasets/Kinetics-400 \
--ptq \
--calib-batches 32 \
--quick-batches 400 \
--save-quant-model output_pt/swin_small_244_k400_1k_h8l4.pth
```

PTQ 推理（跳过校准/快速评估，直接加载已有统计文件）
```bash
bash tools/dist_ptq_test.sh \
configs/recognition/swin/swin_small_patch244_window877_kinetics400_1k.py \
/data/liyifan24/vit_video_pretrained/swin_small_patch244_window877_kinetics400_1k.pth \
4 --eval top_k_accuracy \
--cfg-options \
data.test.ann_file=/data/liyifan24/Datasets/Kinetics-400/val.csv \
data.test.data_prefix=/data/liyifan24/Datasets/Kinetics-400 \
--ptq \
--load-calib \
--load-bits
```
CUDA_VISIBLE_DEVICES=1,2,3

PTQ4ViT
```bash
bash tools/dist_ptq_test.sh \
  configs/recognition/swin/swin_small_patch244_window877_kinetics400_1k.py \
  /data/liyifan24/vit_video_pretrained/swin_small_patch244_window877_kinetics400_1k.pth \
  4 --eval top_k_accuracy \
  --cfg-options \
  data.test.ann_file=/data/liyifan24/Datasets/Kinetics-400/val.csv \
  data.test.data_prefix=/data/liyifan24/Datasets/Kinetics-400 \
  --ptq4vit --ptq4vit-calib-batches 32 --ptq4vit-w-bit 8 --ptq4vit-a-bit 8 、
  --save-quant-model output_pt/swin_small_244_k400_1k_ptq4vit_w8a8.pth
```

FIMA-Q
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash tools/dist_ptq_test.sh \
    configs/recognition/swin/swin_small_patch244_window877_kinetics400_1k.py \
    /data/liyifan24/vit_video_pretrained/swin_small_patch244_window877_kinetics400_1k.pth \
    1 --eval top_k_accuracy \
    --cfg-options \
        data.test.ann_file=/data/liyifan24/Datasets/Kinetics-400/val.csv \
        data.test.data_prefix=/data/liyifan24/Datasets/Kinetics-400 \
    --fimaq \
    --fimaq-w-bit 8 --fimaq-a-bit 8 \
    --fimaq-calib-size 32 --fimaq-calib-batch-size 4 \
    --fimaq-skip-optim \
    --fimaq-calib-checkpoint output_pt/fimaq_calib.pth
```
nohup方式跑FIMA-Q评估（后台运行，输出日志到文件）
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
  nohup bash tools/dist_ptq_test.sh \
    configs/recognition/swin/swin_small_patch244_window877_kinetics400_1k.py \
    /data/liyifan24/vit_video_pretrained/swin_small_patch244_window877_kinetics400_1k.pth \
    4 \
    --eval top_k_accuracy \
    --cfg-options data.test.ann_file=/data/liyifan24/Datasets/Kinetics-400/val.csv \
        data.test.data_prefix=/data/liyifan24/Datasets/Kinetics-400 \
    --fimaq --fimaq-w-bit 8 --fimaq-a-bit 8 \
    --fimaq-calib-size 32 --fimaq-calib-batch-size 4 \
    --fimaq-skip-optim \
    --fimaq-calib-checkpoint output_pt/fimaq_calib.pth \
    > /tmp/fimaq_8bit.log 2>&1 &
echo "PID: $!"
```

load checkpoint跑推理
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash tools/dist_ptq_test.sh \
    configs/recognition/swin/swin_small_patch244_window877_kinetics400_1k.py \
    /data/liyifan24/vit_video_pretrained/swin_small_patch244_window877_kinetics400_1k.pth \
    4 \
    --eval top_k_accuracy \
    --cfg-options data.test.ann_file=/data/liyifan24/Datasets/Kinetics-400/val.csv \
        data.test.data_prefix=/data/liyifan24/Datasets/Kinetics-400 \
    --fimaq --fimaq-w-bit 4 --fimaq-a-bit 4 \
    --fimaq-calib-size 32 --fimaq-calib-batch-size 4 \
    --fimaq-skip-optim \
    --fimaq-calib-checkpoint output_pt/fimaq_calib.pth \
    --fimaq-load-calib
```