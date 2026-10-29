# deploy —— 尾矿库检测演示

用训练好的 Rotated RT-DETR 权重做**单图推理 + 可视化**：输入一张图片（本地路径 / 上传 / URL），
输出带**旋转框 + 类别 + 置信度**的结果图，并列出每个框的中心点、宽高、角度。

```
deploy/
├── infer.py      # 核心推理 + 绘图 + 命令行工具
├── app.py        # 浏览器界面（内置 HTTP 服务，零额外依赖）
├── outputs/      # CLI 默认结果目录（*_pred.jpg / *_pred.txt）
└── README.md
```

配置与权重默认指向：

- 配置：`projects/rotated_rtdetr/configs/o2_rtdetr_r50vd_2xb4_72e_tailing.py`
- 权重：`my_chp/5strong.pth`（`state_dict` + `ema_state_dict` 都在里面）

---

## 一、命令行：输入图片路径

```bash
cd /home/l/detection/huanjing/ai4rs

# 单张（结果存到 deploy/outputs/xxx_pred.jpg）
/home/l/miniconda3/envs/ai4rs/bin/python deploy/infer.py data/tailing/test/images/2016_105.png

# 指定阈值 / 输出目录，并额外导出 DOTA 格式 txt
python deploy/infer.py a.jpg -o /tmp/vis --score-thr 0.4 --save-txt

# 整个文件夹批量
python deploy/infer.py data/tailing/test/images/ --score-thr 0.5

# 只看终端结果，不落盘
python deploy/infer.py a.jpg --no-save
```

终端输出：

```
== data/tailing/test/images/2016_105.png  (3 个目标, 推理 20.9 ms) ==
 #  类别              置信度        cx     cy      w      h      角度
 1  tailing_pond  0.814     652.6  435.9  179.6  103.7   90.6°
 2  tailing_pond  0.799     842.4  486.0  177.9   86.5  123.3°
 3  tailing_pond  0.560     358.3  528.7  162.1  121.1   97.7°
   -> 已保存: .../deploy/outputs/2016_105_pred.jpg
```

`--save-txt` 的 txt 为 DOTA 格式：`x1 y1 x2 y2 x3 y3 x4 y4 class score`，可直接用
`tools/` 里的脚本算指标。

## 二、浏览器界面

```bash
python deploy/app.py                        # http://127.0.0.1:7860
python deploy/app.py --port 8000 --host 0.0.0.0   # 局域网/远程访问
```

页面支持三种输入方式，右侧实时显示结果图 + 统计 + 检测明细表：

1. **拖拽 / 点击选择**本机图片；
2. 在输入框粘贴**图片路径**（相对路径以仓库根目录为准），回车或点「检测」；
3. 粘贴 **http(s) 图片地址**，服务端下载后检测。

另外：阈值滑块（改动后自动重检）、「取一张测试图」随机抽 `data/tailing/test/images` 里的图。

关掉服务：`Ctrl+C`；如果是后台 `systemd` 起的：`systemctl --user stop tailing-demo`。

## 三、常用参数

| 参数 | 说明 |
| --- | --- |
| `--score-thr` | 置信度阈值，默认 `0.5`（见下方阈值实测） |
| `--weights` | `auto`(默认，用 raw)/`ema`/`raw` |
| `--device` | `auto`/`cuda:0`/`cpu` |
| `--config` `--checkpoint` | 换模型/换权重 |
| `--max-per-img` | 覆盖配置里的 `test_cfg.max_per_img`（默认 300） |
| `--json` | 把全部结果写成 json |

## 四、默认阈值的来历

在 **test 集 237 张** 上用 IoU≥0.5 贪心匹配统计（raw 权重）：

| thr | 0.2 | 0.3 | 0.4 | **0.5** | 0.6 | 0.7 |
| --- | --- | --- | --- | --- | --- | --- |
| Precision | 0.751 | 0.840 | 0.884 | **0.911** | 0.936 | 0.949 |
| Recall | 0.932 | 0.916 | 0.910 | **0.894** | 0.872 | 0.810 |
| F1 | 0.832 | 0.877 | 0.897 | **0.903** | 0.903 | 0.874 |

所以默认 `0.5`：画面上基本没有多余框；想尽量不漏（宁可多报）就调到 `0.25~0.3`。

## 五、raw 还是 EMA？

checkpoint 里两个权重都存了，实测 test/mAP：

| 权重 | mAP | AP50 |
| --- | --- | --- |
| `state_dict`（默认） | **0.8337** | 0.8340 |
| `ema_state_dict` | 0.7715 | 0.7720 |

EMA 反而更低（EMAHook 的 `momentum=1e-4` 相当于 ~1w 步的滑动窗口，72 epoch 训练末端还没跟上），
所以默认用 raw；`--weights ema` 可以自己对比。

## 六、常见问题

**① `OSError: [Errno 98] Address already in use`**
7860 被别的进程占了（比如之前开着的另一个 `app.py`）。现在会自动顺延到 7861/7862…，
日志里会打印 `[!] 端口 7860 已被占用, 换 7861 试试 ...`。也可以直接指定：
`python deploy/app.py --port 8000`。查是谁占的：`ss -ltnp | grep 7860`。

**② `SyntaxError: Non-UTF-8 code starting with '\x89' ... in .../2016_42.png`**
把图片当成脚本传给 python 了，漏了脚本名。正确写法：

```bash
# 错：python data/tailing/test/images/2016_42.png
python deploy/infer.py data/tailing/test/images/2016_42.png
python deploy/infer.py a.jpg b.png --no-save      # 多张一起也行
```

**③ 启动时那堆 torch/mmengine 警告**
已经压掉了（`TorchScript`/`weights_only`/`meshgrid` 等），现在启动日志只有：
`[*] 正在加载模型 ... → [load] ... → [*] 设备 → [*] 预热完成 → [+] 服务已启动`。

## 七、实现细节 / 排错

- **预处理与验证集完全一致**：`LoadImageFromFile → Resize(1024, keep_ratio) → Pad(1024, 114) → PackDetInputs`，
  也就是 `tailing.py` 里 `val_pipeline` 去掉标注后的版本，保证和算 mAP 时输入一致。
  输出框已按 `scale_factor` 还原到**原图像素坐标**，所以大图（>1024 的切片）也不会框错位。
- **角度**：模型输出 `(cx, cy, w, h, rad)`，`rad ∈ [0, π)`，宽度为长边。可视化时用
  `cv2.boxPoints` 转四个角点，表格里的角度已转成度。
- **中文标签**：用 PIL + 系统 CJK 字体（Noto/AR PL），找不到字体会自动退回英文类别名。
- **显存**：模型约 1.5 GB，1024 输入推理峰值 < 4 GB，4090 上单图 20~60 ms。
- **报错 `CPU dispatcher tracer already initlized` / `libnvJitLink.so.12: failed to map segment`**：
  这是 shell 的进程数/虚拟内存 `ulimit` 限制导致的（不是代码问题），在正常终端里跑即可；
  必要时 `ulimit -v unlimited` 或 `OPENBLAS_NUM_THREADS=4` 再运行。
