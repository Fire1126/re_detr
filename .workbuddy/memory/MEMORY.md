# 项目长期记忆 —— ai4rs (mmrotate fork)

## 环境
- 仓库即 mmrotate 的 fork，以 **editable 方式安装**（site-packages 中 mmrotate
  指向 `/home/l/detection/huanjing/ai4rs`）。改仓库源码即时生效。
- 运行 python 用 conda 环境：`/home/l/miniconda3/envs/ai4rs/bin/python`
  （mmdet 也装在该环境）。
- 自定义 dataset transform 放在 `mmrotate/datasets/transforms/transforms.py`，
  用 `from mmrotate.registry import TRANSFORMS` + `@TRANSFORMS.register_module()`
  注册，并在同目录 `__init__.py` 的 import 与 `__all__` 中导出。config 里可用
  不带前缀的 `type='Xxx'`（mmrotate 为默认 scope；mmdet 的要写 `mmdet.Xxx`）。

## 关键坑
- **bash 工具 shell 有 `ulimit -v 4GB` / `nproc 4096` 硬限制**，直接跑 python 导入 torch 会报
  `libnvJitLink.so.12: failed to map segment` 或 numpy `CPU dispatcher tracer already initlized`。
  绕法：`systemd-run --user --collect --wait --pipe bash -c '...'`（+`OPENBLAS_NUM_THREADS=4`）。
  这不是代码问题，用户自己的终端正常。
- `mmdet.PhotoMetricDistortion` 会把图像转成 **float32(0-255) 且不转回 uint8**。
  凡是按 0-255 尺度做像素操作的增强（如加噪），应放在它**之前**。
- `RotatedBoxes.clip_()` 只告警、不做任何裁剪。`mmdet.RandomShift` 直接用于
  旋转框会：① 每次告警；② 完全移出画面的框仍作为有效标签保留。需自定义平移。
- 数据集配置 `tailing.py` 中 `LoadAnnotations` / `ConvertBoxType` 必须在
  `Resize` 之前，否则 >1024 的大切片会框图错位。

## 数据集
- 尾矿库单类别 OBB 数据集，`dataset_type='DOTADataset'`，`data_root='data/tailing/'`，
  classes=('tailing_pond',)，train/val/test 各含 images/ 与 annfiles/。
- 训练权重 `my_chp/5strong.pth`（含 state_dict + ema_state_dict，optimizer 也在里面，697MB）。
  实测 test mAP：raw **0.8337** / ema 0.7715 → 一律用 **raw**。

## 推理/演示
- `deploy/infer.py`（CLI + `TailingDetector`）、`deploy/app.py`（stdlib HTTP 网页演示，
  默认端口 7860）、`deploy/README.md`。改模型/权重只需换 `--config` / `--checkpoint`。
- 推理 pipeline = `val_pipeline` 去掉标注；输出 bbox 为 `(cx,cy,w,h,rad)`、rad∈[0,π)。
- 默认 score_thr=0.5（test 集 F1 最优 0.903）。
