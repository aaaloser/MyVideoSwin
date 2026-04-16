# FIMA-Q 移植至 Video Swin Transformer 实现清单

> 目标：在 Video Swin Transformer (Kinetics-400, Swin-S) 上复现 FIMA-Q PTQ 流程。  
> 参考：`fimaq/` 原始代码（针对 timm 2D Swin），`mmaction/models/backbones/swin_transformer.py`。

---

## 0. 整体架构差异对照

| 维度 | FIMA-Q (timm 2D Swin) | Video Swin (mmaction2) |
|---|---|---|
| Attention 类 | `timm.WindowAttention` (2D) | `WindowAttention3D` (3D，window=(8,7,7)) |
| Block 类 | `timm.SwinTransformerBlock` | `SwinTransformerBlock3D` |
| Block.forward 签名 | `forward(self, x)` | `forward(self, x, mask_matrix)` |
| PatchMerging | `timm.PatchMerging` (2D) | `PatchMerging` (Video Swin, 3D) |
| PatchEmbed | `timm.PatchEmbed` (2D Conv2d) | `PatchEmbed3D` (3D Conv3d) |
| DataLoader 输出 | `(image, label)` Tensor | `dict` with `imgs` key, shape `(B, num_clips, C, T, H, W)` |
| 运行框架 | 单 GPU，独立 Python 脚本 | mmaction2 + mmcv，可多 GPU |
| 分类头路径 | `model.head` | `model.cls_head.fc_cls` |
| 激活内存 | ~49 tokens/window | ~392 tokens/window (8×7×7) |

---

## 1. 新建 `quant/fimaq/` 工作目录

**所有方法（原始方法、PTQ4ViT、FIMA-Q）统一通过 `tools/dist_ptq_test.sh` → `tools/ptq_test.py` 执行**，通过互斥参数 `--ptq` / `--ptq4vit` / `--fimaq` 区分。不新建独立入口脚本。

创建如下文件结构：

```
quant/fimaq/
    __init__.py
    wrap_net.py          # 适配 Video Swin 的网络包装
    block_recon.py       # 适配 Video Swin 的 BlockReconstructor
    calibrator.py        # 适配 Video Swin DataLoader 的 QuantCalibrator
    config.py            # Video Swin 专用 Config 类
    fimaq_eval.py        # 校准/重建/保存载入逻辑（被 ptq_test.py 调用）
```

将 `fimaq/quant_layers/`、`fimaq/quantizers/` **原样复制**到 `quant/fimaq/quant_layers/` 和 `quant/fimaq/quantizers/`，不需要修改。

---

## 2. `wrap_net.py`：注入 MatMul + 替换量化模块

### 2.1 `video_swin_attn_forward`

替代 `fimaq/utils/wrap_net.py` 中的 `swin_attn_forward`，适配 `WindowAttention3D`。

**需实现**：
- 函数签名：`def video_swin_attn_forward(self, x, mask=None)`
- 在 `qkv` 之后分离 q/k/v，使用 `self.matmul1(q, k.transpose(-2, -1))` 替代 `q @ k.transpose(-2, -1)`
- 加入 relative_position_bias（保持不变）
- 加入 mask 分支（`nW = mask.shape[0]` 那段，保持不变）
- softmax + attn_drop
- 使用 `self.matmul2(attn, v)` 替代 `attn @ v`
- 注意：Video Swin 注入 `@` 的位置在 `forward` 内，与 timm 一致，但 forward 有额外 `mask` 入参

### 2.2 `wrap_modules_in_net`（主要修改）

原函数的两步逻辑保持，改变**检测条件**：

**Step 1（注入 MatMul）**：
```
原检测：isinstance(module, timm.models.vision_transformer.Attention)
         isinstance(module, timm.models.swin_transformer.WindowAttention)
新检测：isinstance(module, WindowAttention3D)
```
对检测到的 `WindowAttention3D`：
- `module.matmul1 = MatMul()`
- `module.matmul2 = MatMul()`
- `module.forward = MethodType(video_swin_attn_forward, module)`

**Step 2（替换 Linear/MatMul/Conv）**：
- `MatMul` → `AsymmetricallyBatchingQuantMatMul`（参数：`num_heads` 来自 `father_module.num_heads`）
- `nn.Linear` → `AsymmetricallyBatchingQuantLinear`（逻辑同原版，注意 `qkv` 用 `n_V=3`）
- `nn.Conv3d`（即 `PatchEmbed3D.proj`）：**暂时跳过**（或改为 W8A8 处理，见注5）
- `nn.Conv2d`：原版逻辑保留（Video Swin 不使用 Conv2d，实际不会触发）

**注意**：`n_V=3` 判断条件需检查 `'qkv' in name`（Video Swin 的 qkv linear 就叫 `qkv`，与 2D Swin 一致）。

---

## 3. `calibrator.py`：适配 Video Swin DataLoader

### 3.1 DataLoader 输出格式转换

FIMA-Q 的 `QuantCalibrator` 假设 `calib_loader` 产生 `(inp, target)` 对，`inp` 是图像 Tensor `(B, C, H, W)`。

Video Swin 的 DataLoader 产生 `dict`，需要在喂入前提取并 reshape：

```python
# 在 calibrator 循环中：
for data in calib_loader:
    inp = data['imgs']           # (B, num_clips, C, T, H, W)
    inp = inp[:, 0, ...]         # 取第一个 clip，(B, C, T, H, W)
    inp = inp.float().to(device)
    target = data['label']       # (B,)
    pred = model(inp)            # 经 mmaction2 backbone+head
```

### 3.2 模型前向接口

Video Swin 的完整前向（backbone + cls_head）：
```python
pred = model(inp)   # inp: (B, C, T, H, W)，model 为 mmaction2 Recognizer3D
```
**但 `Recognizer3D.forward` 的接口是 `forward(self, imgs, **kwargs)`**，在 `raw_mode` / `test_mode` 下的 API 与训练有别。

建议：**抽取 `backbone + cls_head`** 单独构建一个 `nn.Sequential`-like wrapper，使其接受 `(B, C, T, H, W)` 并输出 logits，以统一 calibrator 的前向接口。

### 3.3 `batching_quant_calib` 循环中的 Multi-GPU 注意事项

FIMA-Q 原版是单 GPU。如不做 DDP，直接在单卡上跑 128 条校准样本即可（Video Swin-S 在单 A100 下单样本显存约 2-3 GB，`calib_size=32` 是安全上界）。

---

## 4. `block_recon.py`：适配 Video Swin Block 类型

### 4.1 `BlockReconstructor.__init__` — 替换 Block 类型检测

```python
# 原代码
types_of_block = [
    timm.layers.patch_embed.PatchEmbed,
    timm.models.vision_transformer.Block,
    timm.models.swin_transformer.SwinTransformerBlock,
    timm.models.swin_transformer.PatchMerging,
]
for name, module in self.model.named_modules():
    if any(isinstance(module, t) for t in types_of_block) or name.split('.')[-1] == 'head':
        ...

# 新代码（Video Swin）
from mmaction.models.backbones.swin_transformer import (
    SwinTransformerBlock3D, PatchMerging, PatchEmbed3D
)
types_of_block = [SwinTransformerBlock3D, PatchMerging, PatchEmbed3D]
for name, module in self.model.named_modules():
    if any(isinstance(module, t) for t in types_of_block) or name == 'cls_head.fc_cls':
        ...
```

### 4.2 `_prepare_module_data_init` — 替换 forward 函数注入

```python
# 原代码针对各 timm 类型注入不同 forward
# 新代码：
if isinstance(module, SwinTransformerBlock3D):
    module.forward = MethodType(video_swin_block_forward, module)
elif isinstance(module, PatchMerging):
    module.forward = MethodType(video_swin_patch_merging_forward, module)
elif isinstance(module, PatchEmbed3D):
    module.forward = MethodType(video_swin_patch_embed_forward, module)
```

### 4.3 实现三个 perturb-aware 的 forward 函数

**`video_swin_block_forward(self, x, mask_matrix)`**：
- 逻辑完全照搬 `SwinTransformerBlock3D.forward`（复制原始实现）
- 在函数末尾加入 perturbation 分支：
  ```python
  if self.perturb:
      rand_perturb = torch.empty_like(x, dtype=torch.float).uniform_(1, 2) * self.r
      x = x + rand_perturb
  ```
- **关键**：函数签名必须包含 `mask_matrix`，因为 `BasicLayer.forward` 始终传两个参数

**`video_swin_patch_merging_forward(self, x)`**：
- 照抄 `PatchMerging.forward`
- 末尾加 perturb 分支

**`video_swin_patch_embed_forward(self, x)`**：
- 照抄 `PatchEmbed3D.forward`
- 末尾加 perturb 分支

### 4.4 Block 的 `forward_hook` 适配 — 双参数 forward

`SwinTransformerBlock3D.forward(x, mask_matrix)` 有 **两个输入参数**。`single_input_forward_hook` 使用 `inp[0]`（即 `x`），这部分**不需要修改**——钩子捕获的 `inp` 是一个 tuple，`inp[0]` 就是特征张量，`inp[1]` 是 mask_matrix，忽略即可。

**问题**：`reconstruct_single_block` 中调用 `block(cur_inp)` 只传了一个参数，会报错。

**解决方案**：在 `init_block_raw_inp_outp` 中，先保存 `mask_matrix` 到 `block._calib_mask_matrix`，然后用 `register_forward_pre_hook` 或直接覆写 forward 为 `lambda x: original_forward(x, block._calib_mask_matrix)`。更简洁的方式：

```python
# 在 init_block_raw_inp_outp 之前，先从一次前向获取 mask_matrix
# 方案：在 BasicLayer.forward 中 attn_mask 是从 compute_mask 计算的，
# 可以在 SwinTransformerBlock3D 中缓存 mask_matrix

# 或者：包一层 wrapper
block._orig_forward = block.forward
block.forward = lambda x: block._orig_forward(x, block._saved_mask)
# 在 hook 中额外捕获 mask：
def save_mask_hook(module, inp, out):
    module._saved_mask = inp[1] if len(inp) > 1 else None
```

### 4.5 `PatchEmbed3D` 无需 `quanted_input`

与 `timm.PatchEmbed` 相同处理：`block.quanted_input = block.raw_input`（因为 PatchEmbed3D 是第一层）。

---

## 5. `config.py`：Video Swin 专用配置

```python
class Config:
    def __init__(self):
        # 量化位宽
        self.w_bit = 4
        self.a_bit = 4
        self.qhead_a_bit = 8           # 分类头激活用 8bit
        self.qconv_a_bit = 8           # PatchEmbed3D Conv3d 激活用 8bit（若量化）
        
        # Calibration
        self.calib_size = 32           # 视频样本数（内存受限，从 128→32）
        self.calib_batch_size = 4      # 每批 4 条视频
        self.calib_metric = 'mse'
        
        # 搜索参数
        self.search_round = 3
        self.eq_n = 128
        self.matmul_head_channel_wise = True
        self.token_channel_wise = True
        
        # Optimization
        self.optim_size = 64           # 重建用样本数
        self.optim_batch_size = 4
        self.optim_metric = 'fisher_dplr'
        self.keep_gpu = False          # 视频激活大，不全保留在 GPU
        self.temp = 20
        
        # Fisher / DPLR
        self.k = 5
        self.p1 = 1.0
        self.p2 = 1.0
        self.dis_mode = 'q'
        
        # QDrop
        self.optim_mode = 'qdrop'
        self.drop_prob = 0.5
```

**关键调整说明**：
- `calib_size=32`（原为 128）：Video Swin 每条视频激活约为 2D Swin 的 8 倍（8 帧×7×7 窗口 vs 7×7）。
- `calib_batch_size=4`（原为 32）：更小的 batch 防止显存溢出。
- `keep_gpu=False`：激活张量改存 CPU，训练时再搬上 GPU。

---

## 6. `fimaq_eval.py`：校准 + 重建逻辑（被 `ptq_test.py` 调用）

实现以下函数，供 `tools/ptq_test.py` 的 `elif args.fimaq:` 分支调用：

```python
def inject_fimaq(model, calib_loader, args):
    """
    步骤：
    1. wrap_modules_in_net(model, cfg)  —— 注入量化模块
    2. QuantCalibrator.batching_quant_calib()  —— 校准 scale/zero_point
    3. （可选）BlockReconstructor.reconstruct_model()  —— Fisher 重建
    4. 保存/加载 checkpoint（state_dict）
    返回：无（in-place 修改 model）
    """
    ...

def remove_fimaq(model):
    """
    将所有量化模块的 mode 切换回 'raw'（可选，用于评测后还原）
    """
    ...
```

`tools/ptq_test.py` 中的调用方式（已预留 `elif args.fimaq:` 占位分支）：

```python
elif args.fimaq:
    from quant.fimaq.fimaq_eval import inject_fimaq, remove_fimaq
    inject_fimaq(model, data_loader, args)
    # 之后复用 single_gpu_test / multi_gpu_test 得到 outputs
    outputs = multi_gpu_test(model, data_loader, args.tmpdir, args.gpu_collect)
    remove_fimaq(model)
```

---

## 7. 统一入口：`tools/dist_ptq_test.sh` → `tools/ptq_test.py`

三种方法的调用命令格式完全相同，仅互斥参数不同：

**原始方法**（`--ptq`）：
```bash
bash tools/dist_ptq_test.sh \
    configs/recognition/swin/swin_small_patch244_window877_kinetics400_1k.py \
    /data/liyifan24/vit_video_pretrained/swin_small_patch244_window877_kinetics400_1k.pth \
    4 --eval top_k_accuracy \
    --cfg-options \
        data.test.ann_file=/data/liyifan24/Datasets/Kinetics-400/val.csv \
        data.test.data_prefix=/data/liyifan24/Datasets/Kinetics-400 \
    --ptq --calib-batches 32 --quick-batches 400 \
    --save-quant-model output_pt/swin_small_244_k400_1k_h8l4.pth
```

**FIMA-Q**（`--fimaq`，调试期间建议先单卡）：
```bash
bash tools/dist_ptq_test.sh \
    configs/recognition/swin/swin_small_patch244_window877_kinetics400_1k.py \
    /data/liyifan24/vit_video_pretrained/swin_small_patch244_window877_kinetics400_1k.pth \
    1 --eval top_k_accuracy \
    --cfg-options \
        data.test.ann_file=/data/liyifan24/Datasets/Kinetics-400/val.csv \
        data.test.data_prefix=/data/liyifan24/Datasets/Kinetics-400 \
    --fimaq \
    --fimaq-w-bit 4 --fimaq-a-bit 4 \
    --fimaq-calib-size 32 --fimaq-calib-batch-size 4 \
    --fimaq-optim-metric fisher_dplr \
    --fimaq-calib-checkpoint output_pt/fimaq_calib.pth \
    --fimaq-optim-checkpoint output_pt/fimaq_optim.pth
```

**PTQ4ViT**（`--ptq4vit`）：（已有，不变）

`tools/ptq_test.py` 中 `elif args.fimaq:` 占位分支实现后，将 `raise NotImplementedError` 替换为对 `inject_fimaq` / `remove_fimaq` 的调用（见第 6 节）。

---

## 8. 关键技术风险与注意事项

### 8.1 `MatMul` 激活形状与 head-channel-wise scale

FIMA-Q 的 `AsymmetricallyBatchingQuantMatMul` 假设输入形状 `(B, H, S, C)` / `(B, H, C, S)`，其中 H 是 head 维度（scale shape `[1, num_heads, 1, 1]`）。

Video Swin `WindowAttention3D` 的 matmul 形状：
- `matmul1(q, k.T)`：q 为 `(B*nW, nH, N, C//nH)`，k.T 为 `(B*nW, nH, C//nH, N)` → 结果 `(B*nW, nH, N, N)`
- `matmul2(attn, v)`：attn 为 `(B*nW, nH, N, N)`，v 为 `(B*nW, nH, N, C//nH)` → 结果 `(B*nW, nH, N, C//nH)`

其中 `B*nW` 是 batch × 窗口数，而非原始 batch。这导致 `num_heads` 维度出现在 dim-1 的位置——与 FIMA-Q 的 `(B, H, S, C)` 假设**一致**，无需修改 scale 形状。

但 `B*nW` 不是固定的（不同 clip 不同），`_initialize_calib_parameters` 中的显存估算需要注意，建议降低 `parallel_eq_n` 或显式指定为较小值。

### 8.2 `SwinTransformerBlock3D.forward` 双参数问题（最大风险）

`BlockReconstructor.reconstruct_single_block` 中 `block(cur_inp)` 只传一个参数。`SwinTransformerBlock3D.forward(x, mask_matrix)` 需要两个参数。

**推荐实现方案**：在 `_prepare_module_data_init` 时，将 block 的 forward 包装为：
```python
import functools
original_forward = block.forward
# 先做一次 dry run 获取 mask，保存到 block._attn_mask
block.forward = functools.partial(original_forward, mask_matrix=None)
# 在 init_block_raw_inp_outp 时，用 pre_hook 捕获并缓存真实 mask_matrix
```
更好的方案（无侵入）：注册 `register_forward_pre_hook` 在 `BasicLayer` 上，在调用每个 block 前将 attn_mask 保存到 `block._cached_mask`，然后 block 的 perturb-forward 调用 `self._orig_forward(x, self._cached_mask)`。

### 8.3 `PatchEmbed3D.proj` 是 `nn.Conv3d`，不是 `nn.Conv2d`

FIMA-Q 的 `wrap_modules_in_net` 会把 `nn.Conv2d` 换成 `AsymmetricallyBatchingQuantConv2d`，但 `Conv3d` 没有对应实现。

**推荐方案**：在 `wrap_modules_in_net` 的 step2 中，对 `nn.Conv3d` **直接跳过**（不量化），或单独实现一个 `QuantConv3d`（参考 `fimaq/quant_layers/conv.py` 改造）。跳过 PatchEmbed3D 的误差一般可忽略。

### 8.4 分类头路径

Video Swin 完整模型（`Recognizer3D`）中分类头路径为 `cls_head.fc_cls`，不是 `head`。

在 `BlockReconstructor` 的 block 检测中，将 `name.split('.')[-1] == 'head'` 改为 `name == 'cls_head.fc_cls'`。在 `LossFunction` 的 `'head' not in name` 条件改为 `name != 'cls_head.fc_cls'`。

### 8.5 内存压力

Video Swin-S 在 Kinetics-400 上的激活内存远大于 2D Swin-S：
- 单条 32-clip 视频输入 `(1, 3, 32, 224, 224)` 经 PatchEmbed3D 后为 `(1, 96, 8, 56, 56)`
- 每个 stage-1 window partition 产生约 `(8×8×8, 8*7*7, 96) = (4096, 392, 96)` 的 matmul 输入
- 原始 `calib_size=128` 会导致 OOM，建议**从 `calib_size=32`、`calib_batch_size=1`** 开始测试

### 8.6 `reparam`（reparameterized  linear）

`wrap_modules_in_net` 中 `reparam=True` 时，`qkv`/`fc1`/`reduction` 会使用 `AsymmetricallyChannelWiseBatchingQuantLinear`，需要 `prev_layer`（前层 norm）。Video Swin 的 norm 层名同为 `norm1`/`norm2`，与 timm 一致，此路径**可以保留**，但建议**先用 `reparam=False`** 调通基础流程。

---

## 9. 实现顺序建议

```
[x] Step 0. 复制 fimaq/quant_layers/ 和 fimaq/quantizers/ 到 quant/fimaq/
[ ] Step 1. 实现 wrap_net.py
    [ ] 1a. video_swin_attn_forward（注入 matmul1/matmul2，保留 3D bias 和 mask）
    [ ] 1b. wrap_modules_in_net（检测 WindowAttention3D，替换 Linear/MatMul）
[ ] Step 2. 实现 config.py（使用较小 calib_size/batch_size）
[ ] Step 3. 实现 calibrator.py
    [ ] 3a. 视频 DataLoader 适配（(B,C,T,H,W) 输入格式）
    [ ] 3b. 测试单次校准前向是否正常
[ ] Step 4. 实现 block_recon.py
    [ ] 4a. 三个 perturb-aware forward 函数
    [ ] 4b. BlockReconstructor 类检测替换
    [ ] 4c. SwinTransformerBlock3D 双参数 forward 问题解决
[ ] Step 5. 实现 fimaq_eval.py（保存/加载 checkpoint）
[ ] Step 6. 实现 tools/ptq_fimaq.py（命令行入口）
[ ] Step 7. 端到端测试（只跑 calibrate，不跑 optimize）
[ ] Step 8. 端到端测试（calibrate + optimize + evaluate）
```

---

## 10. 文件依赖图

```
tools/dist_ptq_test.sh
    └─ tools/ptq_test.py  (--ptq | --ptq4vit | --fimaq)
           ├─ [--ptq]      quant/{calibrate, quick_eval, inject_full_eval, ...}
           ├─ [--ptq4vit]  quant/ptq4vit/{inject_ptq4vit, remove_ptq4vit, ...}
           └─ [--fimaq]    quant/fimaq/fimaq_eval.py
                               ├─ inject_fimaq(model, calib_loader, args)
                               │       ├─ quant/fimaq/wrap_net.py
                               │       │       ├─ quant/fimaq/quant_layers/ (复制自 fimaq/quant_layers/)
                               │       │       └─ mmaction.models.backbones.swin_transformer.WindowAttention3D
                               │       ├─ quant/fimaq/calibrator.py
                               │       │       └─ quant/fimaq/quant_layers/
                               │       ├─ quant/fimaq/block_recon.py
                               │       │       ├─ mmaction.models.backbones.swin_transformer
                               │       │       │       .{SwinTransformerBlock3D, PatchMerging, PatchEmbed3D}
                               │       │       └─ quant/fimaq/calibrator.py
                               │       └─ quant/fimaq/config.py
                               └─ remove_fimaq(model)
```
