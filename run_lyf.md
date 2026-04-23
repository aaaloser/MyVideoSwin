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
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup  bash tools/dist_ptq_test.sh \
configs/recognition/swin/swin_small_patch244_window877_kinetics400_1k.py \
/data/liyifan24/vit_video_pretrained/swin_small_patch244_window877_kinetics400_1k.pth \
4 --eval top_k_accuracy \
--cfg-options \
data.test.ann_file=/data/liyifan24/Datasets/Kinetics-400/val.csv \
data.test.data_prefix=/data/liyifan24/Datasets/Kinetics-400 \
--ptq \
--calib-batches 32 \
--quick-batches 400 > /tmp/ptq_work1.log 2>&1 &
```

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash tools/dist_ptq_test.sh \
configs/recognition/swin/swin_small_patch244_window877_kinetics400_1k.py \
/data/liyifan24/vit_video_pretrained/swin_small_patch244_window877_kinetics400_1k.pth \
4 --eval top_k_accuracy \
--cfg-options \
data.test.ann_file=/data/liyifan24/Datasets/Kinetics-400/val.csv \
data.test.data_prefix=/data/liyifan24/Datasets/Kinetics-400 \
--ptq \
--calib-batches 32 \
--quick-batches 400 \
--high-ratio-threshold 0.23 \
--save-quant-model output_pt/swin_s_lyf_threshold0.25.pth
```

PTQ 推理（跳过校准/快速评估，直接加载已有统计文件）
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash tools/dist_ptq_test.sh \
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

lambda 诊断（仅观测，不改主流程，固定 calib-batches=32）
```bash
CUDA_VISIBLE_DEVICES=0 python tools/analysis/lambda_diagnose.py \
configs/recognition/swin/swin_small_patch244_window877_kinetics400_1k.py \
/data/liyifan24/vit_video_pretrained/swin_small_patch244_window877_kinetics400_1k.pth \
--calib-batches 32 \
--device cuda:0 \
--cfg-options \
data.test.ann_file=/data/liyifan24/Datasets/Kinetics-400/val.csv \
data.test.data_prefix=/data/liyifan24/Datasets/Kinetics-400 \
--csv-path output_pt/lambda_diag_calib32.csv \
--json-path output_pt/lambda_diag_calib32_summary.json
```

快速查看诊断摘要
```bash
cat output_pt/lambda_diag_calib32_summary.json
```

快速查看前10个block诊断值
```bash
head -n 11 output_pt/lambda_diag_calib32.csv
```

PTQ 推理（lambda 参数化：缓解 lambda=0.5 饱和）
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash tools/dist_ptq_test.sh \
configs/recognition/swin/swin_small_patch244_window877_kinetics400_1k.py \
/data/liyifan24/vit_video_pretrained/swin_small_patch244_window877_kinetics400_1k.pth \
4 --eval top_k_accuracy \
--cfg-options \
data.test.ann_file=/data/liyifan24/Datasets/Kinetics-400/val.csv \
data.test.data_prefix=/data/liyifan24/Datasets/Kinetics-400 \
--ptq \
--calib-batches 32 \
--quick-batches 400 \
--high-ratio-threshold 0.25 \
--lambda-ratio-scale 20000 \
--lambda-min 0.1 \
--lambda-max 0.9 \
--save-quant-model output_pt/swin_s_lamS2e4_lam0.1_0.9.pth
```

PTQ 推理（敏感块强制 INT8：优先修复 Stage2 分配盲区）
```bash
CUDA_VISIBLE_DEVICES=1,2,3 bash tools/dist_ptq_test.sh \
configs/recognition/swin/swin_small_patch244_window877_kinetics400_1k.py \
/data/liyifan24/vit_video_pretrained/swin_small_patch244_window877_kinetics400_1k.pth \
3 --eval top_k_accuracy \
--cfg-options \
data.test.ann_file=/data/liyifan24/Datasets/Kinetics-400/val.csv \
data.test.data_prefix=/data/liyifan24/Datasets/Kinetics-400 \
--ptq \
--calib-batches 32 \
--quick-batches 400 \
--high-ratio-threshold 0.999 \
--force-int8-blocks stage2_block10 stage2_block12 stage2_block14 stage2_block15 stage2_block16
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

77.99/93.40
```bash
CUDA_VISIBLE_DEVICES=1,2,3 bash tools/dist_ptq_test.sh \
    configs/recognition/swin/swin_small_patch244_window877_kinetics400_1k.py \
    /data/liyifan24/vit_video_pretrained/swin_small_patch244_window877_kinetics400_1k.pth \
    3 --eval top_k_accuracy \
    --cfg-options \
    data.test.ann_file=/data/liyifan24/Datasets/Kinetics-400/val.csv \
    data.test.data_prefix=/data/liyifan24/Datasets/Kinetics-400 \
    --ptq \
    --calib-batches 32 \
    --quick-batches 400 \
    --high-ratio-threshold 0.30 \
    --calib-stats-path output_pt/calib_stats_anchor_3g.pkl \
    --force-int4-blocks stage2_block9 \
    --block-bits-path output_pt/block_bits_t030_f4b9_3g.json \
    --save-quant-model output_pt/swin_s_t030_f4b9_3g.pth
```

78.05/93.39
```bash
CUDA_VISIBLE_DEVICES=1,2,3 bash tools/dist_ptq_test.sh \
    configs/recognition/swin/swin_small_patch244_window877_kinetics400_1k.py \
    /data/liyifan24/vit_video_pretrained/swin_small_patch244_window877_kinetics400_1k.pth \
    3 --eval top_k_accuracy \
    --cfg-options \
    data.test.ann_file=/data/liyifan24/Datasets/Kinetics-400/val.csv \
    data.test.data_prefix=/data/liyifan24/Datasets/Kinetics-400 \
    --ptq \
    --load-calib \
    --calib-stats-path output_pt/calib_stats_anchor_3g.pkl \
    --quick-batches 400 \
    --high-ratio-threshold 0.20 \
    --force-int4-blocks stage2_block9 \
    --block-bits-path output_pt/block_bits_t020_f4b9_3g.json \
    --save-quant-model output_pt/swin_s_t020_f4b9_3g.pth
```

自动化矩阵测试（推荐，免手工一条条跑）
```bash
CUDA_VISIBLE_DEVICES=1,2,3 python tools/analysis/run_ptq_matrix.py \
    --preset b9_refine \
    --calib-stats-path output_pt/calib_stats_anchor_3g.pkl \
    --results-csv output_pt/ptq_matrix_results_3g.csv
```

bitgap 候选矩阵（固定 t=0.20，自动测 stage2_block11/13 等组合）
```bash
CUDA_VISIBLE_DEVICES=1,2,3 python tools/analysis/run_ptq_matrix.py \
    --preset b9_bitgap \
    --calib-stats-path output_pt/calib_stats_anchor_3g.pkl \
    --results-csv output_pt/ptq_matrix_results_3g.csv
```

只跑指定实验（例如你已跑过 t020_f4b9 / t030_f4b9 时）
```bash
CUDA_VISIBLE_DEVICES=1,2,3 python tools/analysis/run_ptq_matrix.py \
    --preset b9_refine \
    --calib-stats-path output_pt/calib_stats_anchor_3g.pkl \
    --results-csv output_pt/ptq_matrix_results_3g.csv \
    --experiments t025_f4b9 t030_f4b9_f8b1
```

查看当前矩阵结果
```bash
cat output_pt/ptq_matrix_results_3g.csv
```

Block 级 8/4 差异诊断（新）
```bash
CUDA_VISIBLE_DEVICES=1,2,3 /home/liyifan24/miniconda3/bin/conda run -p /data/liyifan24/envs/videoswin --no-capture-output \
python tools/analysis/bitgap_diagnose.py \
    configs/recognition/swin/swin_small_patch244_window877_kinetics400_1k.py \
    /data/liyifan24/vit_video_pretrained/swin_small_patch244_window877_kinetics400_1k.pth \
    --collect-batches 8 \
    --samples-per-block 2 \
    --device cuda:0 \
    --block-bits-path output_pt/block_bits_t020_f4b9_3g.json \
    --cfg-options \
    data.test.ann_file=/data/liyifan24/Datasets/Kinetics-400/val.csv \
    data.test.data_prefix=/data/liyifan24/Datasets/Kinetics-400 \
    --csv-path output_pt/bitgap_diag_3g_b8s2.csv \
    --json-path output_pt/bitgap_diag_3g_b8s2_summary.json
```

候选（基于 output_pt/bitgap_diag_3g_b8s2_summary.json）
- candidate_force_int4: stage2_block11 stage2_block13 stage2_block15 stage3_block1 stage2_block16
- candidate_force_int8: stage0_block0 stage1_block0 stage2_block0 stage2_block2 stage2_block4

建议先做单变量验证（保持 stage2_block9 强制 INT4 不变）
```bash
# 1) + stage2_block11 INT4
CUDA_VISIBLE_DEVICES=1,2,3 /home/liyifan24/miniconda3/bin/conda run -p /data/liyifan24/envs/videoswin \
bash tools/dist_ptq_test.sh \
    configs/recognition/swin/swin_small_patch244_window877_kinetics400_1k.py \
    /data/liyifan24/vit_video_pretrained/swin_small_patch244_window877_kinetics400_1k.pth \
    3 --eval top_k_accuracy \
    --cfg-options \
    data.test.ann_file=/data/liyifan24/Datasets/Kinetics-400/val.csv \
    data.test.data_prefix=/data/liyifan24/Datasets/Kinetics-400 \
    --ptq --load-calib \
    --calib-stats-path output_pt/calib_stats_anchor_3g.pkl \
    --quick-batches 400 \
    --high-ratio-threshold 0.20 \
    --force-int4-blocks stage2_block9 stage2_block11 \
    --block-bits-path output_pt/block_bits_t020_f4b9_f4b11_3g.json \
    --save-quant-model output_pt/swin_s_t020_f4b9_f4b11_3g.pth

# 2) + stage2_block13 INT4
CUDA_VISIBLE_DEVICES=1,2,3 /home/liyifan24/miniconda3/bin/conda run -p /data/liyifan24/envs/videoswin \
bash tools/dist_ptq_test.sh \
    configs/recognition/swin/swin_small_patch244_window877_kinetics400_1k.py \
    /data/liyifan24/vit_video_pretrained/swin_small_patch244_window877_kinetics400_1k.pth \
    3 --eval top_k_accuracy \
    --cfg-options \
    data.test.ann_file=/data/liyifan24/Datasets/Kinetics-400/val.csv \
    data.test.data_prefix=/data/liyifan24/Datasets/Kinetics-400 \
    --ptq --load-calib \
    --calib-stats-path output_pt/calib_stats_anchor_3g.pkl \
    --quick-batches 400 \
    --high-ratio-threshold 0.20 \
    --force-int4-blocks stage2_block9 stage2_block13 \
    --block-bits-path output_pt/block_bits_t020_f4b9_f4b13_3g.json \
    --save-quant-model output_pt/swin_s_t020_f4b9_f4b13_3g.pth
```

单变量实测（2026-04-20）
- t020 + force-int4 stage2_block9 stage2_block11：78.96/93.86
- 日志：logs/ptq_calib32_f42_20260420_143107.log


78.96/93.86
force-int4 stage2_block9 + stage2_block11
high-ratio-threshold 0.20 
```bash
CUDA_VISIBLE_DEVICES=1,2,3 bash tools/dist_ptq_test.sh \
    configs/recognition/swin/swin_small_patch244_window877_kinetics400_1k.py \
    /data/liyifan24/vit_video_pretrained/swin_small_patch244_window877_kinetics400_1k.pth \
    3 --eval top_k_accuracy \
    --cfg-options \
    data.test.ann_file=/data/liyifan24/Datasets/Kinetics-400/val.csv \
    data.test.data_prefix=/data/liyifan24/Datasets/Kinetics-400 \
    --ptq \
    --load-calib \
    --calib-stats-path output_pt/calib_stats_anchor_3g.pkl \
    --quick-batches 400 \
    --high-ratio-threshold 0.20 \
    --force-int4-blocks stage2_block9 stage2_block11 \
    --block-bits-path output_pt/block_bits_t020_f4b9_f4b11_3g.json \
    --save-quant-model output_pt/swin_s_t020_f4b9_f4b11_3g.pth
```