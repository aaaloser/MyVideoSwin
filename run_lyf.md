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

