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
bash tools/dist_ptq_test.sh \
configs/recognition/swin/swin_small_patch244_window877_kinetics400_1k.py \
/data/liyifan24/vit_video_pretrained/swin_small_patch244_window877_kinetics400_1k.pth \
4 --eval top_k_accuracy \
--cfg-options \
data.test.ann_file=/data/liyifan24/Datasets/Kinetics-400/val.csv \
data.test.data_prefix=/data/liyifan24/Datasets/Kinetics-400 \
--ptq \
--calib-batches 32 \
--quick-batches 200
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

