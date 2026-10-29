#!/usr/bin/env python
"""尾矿库(OBB)检测 —— 单图/批量推理与结果可视化。

用法示例
--------
# 单张图片
python deploy/infer.py data/tailing/test/images/2016_104.png

# 指定输出目录 + 阈值
python deploy/infer.py path/to/a.jpg -o deploy/outputs --score-thr 0.25

# 整个文件夹
python deploy/infer.py path/to/imgs/ --score-thr 0.3

# 打印 DOTA 格式的检测框 (x1 y1 x2 y2 x3 y3 x4 y4 class score)
python deploy/infer.py a.jpg --save-txt
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import sys
import time
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

# 屏蔽第三方库里对使用毫无影响的噪音警告
def _quiet_warnings() -> None:
    for msg in (r'.*TorchScript.*', r'.*torch\.meshgrid.*',
                r'.*weights_only.*', r'.*CPU dispatcher tracer.*'):
        warnings.filterwarnings('ignore', message=msg)


_quiet_warnings()

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_CONFIG = REPO_ROOT / 'projects/rotated_rtdetr/configs/o2_rtdetr_r50vd_2xb4_72e_tailing.py'
DEFAULT_CKPT = REPO_ROOT / 'my_chp/5strong.pth'
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / 'outputs'

# 类别中文名(仅用于展示, 找不到就用原始英文名)
CLASS_NAMES_ZH = {'tailing_pond': '尾矿库'}

# 每个类别一个颜色 (BGR)
PALETTE = [
    (0, 200, 255), (0, 165, 255), (60, 220, 60), (255, 128, 0),
    (255, 0, 200), (0, 255, 200), (200, 100, 255), (0, 100, 255),
]

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}

CJK_FONT_CANDIDATES = [
    '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
    '/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf',
    '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc',
    '/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc',
    '/usr/share/fonts/truetype/arphic/uming.ttc',
    '/usr/share/fonts/truetype/arphic/ukai.ttc',
    '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
    '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
]


# --------------------------------------------------------------------------- #
# 数据结构
# --------------------------------------------------------------------------- #
@dataclass
class Detection:
    """一个旋转框检测结果, 坐标为原图像素坐标, angle 为弧度。"""
    label: int
    class_name: str
    score: float
    cx: float
    cy: float
    w: float
    h: float
    angle: float  # rad, [0, pi)

    @property
    def angle_deg(self) -> float:
        return math.degrees(self.angle)

    @property
    def poly(self) -> np.ndarray:
        """四个角点 (4, 2) float32, 顺时针/逆时针顺序不影响绘制。"""
        return corners(self.cx, self.cy, self.w, self.h, self.angle)

    def to_dict(self, ndigits: int = 2) -> dict:
        d = asdict(self)
        for k in ('score', 'cx', 'cy', 'w', 'h', 'angle'):
            d[k] = round(d[k], ndigits if k != 'score' else 4)
        d['angle_deg'] = round(self.angle_deg, ndigits)
        d['poly'] = [[round(float(x), ndigits), round(float(y), ndigits)]
                     for x, y in self.poly]
        return d

    def to_dota_line(self) -> str:
        pts = ' '.join(f'{x:.1f} {y:.1f}' for x, y in self.poly)
        return f'{pts} {self.class_name} {self.score:.4f}'


def corners(cx: float, cy: float, w: float, h: float, angle: float) -> np.ndarray:
    """(cx, cy, w, h, rad) -> 四角点, 与 cv2.boxPoints 同一定义。"""
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    dx, dy = w / 2.0, h / 2.0
    pts = np.array([[-dx, -dy], [dx, -dy], [dx, dy], [-dx, dy]], dtype=np.float32)
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
    return pts @ rot.T + np.array([cx, cy], dtype=np.float32)


# --------------------------------------------------------------------------- #
# 检测器
# --------------------------------------------------------------------------- #
class TailingDetector:
    """加载配置文件 + 权重, 对外只暴露 detect()。"""

    def __init__(self,
                 config: str | Path = DEFAULT_CONFIG,
                 checkpoint: str | Path = DEFAULT_CKPT,
                 device: str = 'auto',
                 weights: str = 'auto',
                 score_thr: float = 0.5,
                 max_per_img: Optional[int] = None,
                 verbose: bool = True):
        # 注意: 导入 torch/mmengine 的过程中 numpy 会重置 warnings.filters,
        # 所以第三方库的导入噪音要在 catch_warnings 里压掉, 导完再重新装一次过滤器。
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            import torch
            # torch 在导入 distributed.optim 时会先插入一个 'once' 过滤器,
            # 再打印一条 TorchScript 弃用警告, 与本次推理无关, 直接静音掉
            with contextlib.redirect_stderr(io.StringIO()):
                import torch.distributed.optim  # noqa: F401
            from mmengine.config import Config
            from mmengine.registry import MODELS, init_default_scope
            from mmengine.runner import load_checkpoint
        _quiet_warnings()

        self.torch = torch
        self.config_path = Path(config).expanduser().resolve()
        self.checkpoint_path = Path(checkpoint).expanduser().resolve()
        if not self.config_path.is_file():
            raise FileNotFoundError(f'找不到配置文件: {self.config_path}')
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f'找不到权重文件: {self.checkpoint_path}')

        # device
        if device == 'auto':
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)

        # ---------------- 配置 / 模型 ----------------
        cfg = Config.fromfile(str(self.config_path))
        init_default_scope(cfg.get('default_scope', 'mmrotate'))
        # 推理阶段不需要初始化 backbone 的预训练权重(否则会联网下载)
        cfg.model.pop('init_cfg', None)
        self.classes: Tuple[str, ...] = tuple(cfg.get('classes', ('tailing_pond', )))
        self.model = MODELS.build(cfg.model)
        if max_per_img is not None:
            self.model.bbox_head.test_cfg['max_per_img'] = int(max_per_img)

        # ---------------- 权重 (支持 EMA) ----------------
        try:
            ckpt = torch.load(str(self.checkpoint_path), map_location='cpu',
                              weights_only=False)
        except TypeError:  # torch < 2.0 没有 weights_only
            ckpt = torch.load(str(self.checkpoint_path), map_location='cpu')
        if isinstance(ckpt, dict) and 'state_dict' in ckpt:
            raw = ckpt['state_dict']
            ema = ckpt.get('ema_state_dict')
        else:
            raw, ema = ckpt, None
        if weights == 'auto':
            # 实测该 checkpoint: raw state_dict mAP=0.8337, ema_state_dict mAP=0.7715
            # (EMAHook momentum=1e-4 相当于 1w 步窗口, 训练末端还没跟上, 故默认用 raw)
            weights = 'raw' if raw is not None else 'ema'
        if weights == 'ema' and not ema:
            weights = 'raw'
        if weights == 'raw' and raw is None:
            weights = 'ema'
        if weights == 'ema':
            state = {k.replace('module.', '', 1): v
                     for k, v in ema.items() if k != 'steps'}
        else:
            state = raw
        self.weights = weights
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if verbose:
            print(f'[load] config = {self.config_path}')
            print(f'[load] ckpt   = {self.checkpoint_path} (weights={weights})')
            print(f'[load] missing={len(missing)} unexpected={len(unexpected)}')

        self.model.to(self.device).eval()
        self.model.dataset_meta = dict(classes=list(self.classes))

        # ---------------- 推理 pipeline (无标注版 val_pipeline) ----------------
        from mmengine.dataset import Compose
        self.pipeline = Compose([
            dict(type='mmdet.LoadImageFromFile'),
            dict(type='mmdet.Resize', scale=(1024, 1024), keep_ratio=True),
            dict(type='mmdet.Pad', size=(1024, 1024),
                 pad_val=dict(img=(114, 114, 114))),
            dict(type='mmdet.PackDetInputs',
                 meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                            'scale_factor')),
        ])
        self.score_thr = float(score_thr)
        self.font_path = _find_font()

    # ------------------------------------------------------------------ #
    def preprocess(self, img_path: str | Path) -> dict:
        data = self.pipeline(dict(img_id=0, img_path=str(img_path)))
        return data

    def detect(self, img_path: str | Path,
               score_thr: Optional[float] = None) -> Tuple[List[Detection], float]:
        """返回 (检测框列表, 纯推理耗时ms)。"""
        from mmengine.dataset import pseudo_collate

        torch = self.torch
        score_thr = self.score_thr if score_thr is None else float(score_thr)

        data = self.preprocess(img_path)
        meta = data['data_samples'].metainfo
        batch = pseudo_collate([data])

        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            results = self.model.test_step(batch)
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        infer_ms = (time.perf_counter() - t0) * 1000.0

        pred = results[0].pred_instances
        bboxes = pred.bboxes
        if hasattr(bboxes, 'tensor'):          # RotatedBoxes
            bboxes = bboxes.tensor
        bboxes = bboxes.detach().cpu().numpy().reshape(-1, 5)
        scores = pred.scores.detach().cpu().numpy().reshape(-1)
        labels = pred.labels.detach().cpu().numpy().reshape(-1)

        keep = scores >= score_thr
        bboxes, scores, labels = bboxes[keep], scores[keep], labels[keep]
        order = np.argsort(-scores)
        bboxes, scores, labels = bboxes[order], scores[order], labels[order]

        dets = [
            Detection(label=int(l), class_name=self.class_name(int(l)),
                      score=float(s), cx=float(b[0]), cy=float(b[1]),
                      w=float(b[2]), h=float(b[3]), angle=float(b[4]))
            for b, s, l in zip(bboxes, scores, labels)
        ]
        self.last_meta = meta
        return dets, infer_ms

    def class_name(self, idx: int) -> str:
        if 0 <= idx < len(self.classes):
            return self.classes[idx]
        return str(idx)

    def display_name(self, idx: int) -> str:
        name = self.class_name(idx)
        if self.font_path:
            return CLASS_NAMES_ZH.get(name, name)
        return name

    # ------------------------------------------------------------------ #
    def render(self, img_path: str | Path, dets: Sequence[Detection],
               save_path: Optional[str | Path] = None,
               line_width: Optional[int] = None,
               font_scale: Optional[float] = None) -> np.ndarray:
        """在原图上画框 + 类别 + 置信度, 返回 BGR 图像。"""
        import cv2
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            arr = np.fromfile(str(img_path), dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise IOError(f'图片读取失败: {img_path}')

        vis = draw_detections(img, dets, self,
                              line_width=line_width, font_scale=font_scale)
        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            ok, buf = cv2.imencode(save_path.suffix or '.jpg', vis)
            if not ok:
                raise IOError(f'结果写入失败: {save_path}')
            with open(save_path, 'wb') as f:
                f.write(buf.tobytes())
        return vis


# --------------------------------------------------------------------------- #
# 绘图
# --------------------------------------------------------------------------- #
def _find_font() -> Optional[str]:
    for p in CJK_FONT_CANDIDATES:
        if os.path.isfile(p):
            return p
    return None


def _text_color(bgr: Tuple[int, int, int]) -> Tuple[int, int, int]:
    b, g, r = bgr
    return (0, 0, 0) if (0.299 * r + 0.587 * g + 0.114 * b) > 150 else (255, 255, 255)


def _color(label: int) -> Tuple[int, int, int]:
    return PALETTE[label % len(PALETTE)]


def draw_detections(img: np.ndarray,
                    dets: Sequence[Detection],
                    detector: Optional[TailingDetector] = None,
                    line_width: Optional[int] = None,
                    font_scale: Optional[float] = None) -> np.ndarray:
    """在图像上绘制旋转框和标签, 输入/输出均为 BGR ndarray。"""
    import cv2
    from PIL import Image, ImageDraw, ImageFont

    vis = img.copy()
    h_img, w_img = vis.shape[:2]
    base = max(h_img, w_img)
    if line_width is None:
        line_width = max(1, int(round(base * 0.0018)))
    if font_scale is None:
        font_scale = base * 0.021

    # 1) 旋转框用 cv2 画 (带抗锯齿)
    for det in dets:
        color = _color(det.label)
        pts = det.poly.reshape(-1, 1, 2).astype(np.int32)
        cv2.polylines(vis, [pts], isClosed=True, color=color,
                      thickness=line_width, lineType=cv2.LINE_AA)

    if not dets:
        return vis

    # 2) 文字用 PIL 画 (支持中文, 排版更整齐)
    font_path = getattr(detector, 'font_path', None) if detector else _find_font()
    pil = Image.fromarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil, 'RGBA')
    try:
        font = ImageFont.truetype(font_path, int(font_scale)) if font_path \
            else ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()

    for det in dets:
        name = detector.display_name(det.label) if detector \
            else CLASS_NAMES_ZH.get(det.class_name, det.class_name)
        text = f'{name} {det.score:.2f}'
        color = _color(det.label)
        color_rgb = (color[2], color[1], color[0])
        x, y = det.poly[:, 0].min(), det.poly[:, 1].min()
        try:
            l, t, r, b = draw.textbbox((0, 0), text, font=font)
            tw, th = r - l, b - t
        except Exception:
            l = t = 0
            tw, th = 8 * len(text), int(font_scale)
        pad = max(2, int(font_scale * 0.22))
        x = float(np.clip(x, 0, max(0, w_img - tw - 2 * pad)))
        y = max(0.0, y - th - 2 * pad)
        draw.rectangle([x, y, x + tw + 2 * pad, y + th + 2 * pad],
                       fill=color_rgb + (230, ))
        draw.text((x + pad - l, y + pad - t), text,
                  font=font, fill=_text_color(color))

    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _resolve_inputs(inp: Iterable[str]) -> List[Path]:
    files: List[Path] = []
    for item in inp:
        p = Path(item).expanduser()
        if p.is_dir():
            files.extend(sorted(x for x in p.iterdir()
                                if x.suffix.lower() in IMG_EXTS))
        elif p.is_file():
            files.append(p)
        else:
            raise FileNotFoundError(f'输入不存在: {item}')
    return files


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:  # 输出重定向到文件时也能实时看到日志
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    parser = argparse.ArgumentParser(
        description='尾矿库旋转框检测 —— 输入图片, 输出带框/类别/置信度的结果图')
    parser.add_argument('image', nargs='+', help='图片路径, 或文件夹')
    parser.add_argument('-o', '--out', default=str(DEFAULT_OUT_DIR),
                        help=f'结果保存目录 (默认 {DEFAULT_OUT_DIR})')
    parser.add_argument('--config', default=str(DEFAULT_CONFIG), help='配置文件')
    parser.add_argument('--checkpoint', default=str(DEFAULT_CKPT), help='权重文件')
    parser.add_argument('--device', default='auto', help='auto / cuda:0 / cpu')
    parser.add_argument('--weights', default='auto', choices=['auto', 'ema', 'raw'],
                        help='auto: 默认用 raw (精确权重, 实测 mAP 更高); ema: 用 EMA 权重')
    parser.add_argument('--score-thr', type=float, default=0.5,
                        help='置信度阈值 (test 集上 0.5 时 F1 最优, 0.3 召回更高)')
    parser.add_argument('--max-per-img', type=int, default=None,
                        help='最多保留的候选框数量 (默认沿用配置里的 300)')
    parser.add_argument('--save-txt', action='store_true',
                        help='同时保存 DOTA 格式的 txt 结果')
    parser.add_argument('--no-save', action='store_true', help='不保存图片, 只看终端输出')
    parser.add_argument('--json', default=None, help='把全部结果写成 json 文件')
    args = parser.parse_args(argv)

    try:
        files = _resolve_inputs(args.image)
    except FileNotFoundError as e:
        print(f'[错误] {e}', file=sys.stderr)
        return 2
    if not files:
        print('没有找到可处理的图片', file=sys.stderr)
        return 1

    detector = TailingDetector(config=args.config, checkpoint=args.checkpoint,
                               device=args.device, weights=args.weights,
                               score_thr=args.score_thr,
                               max_per_img=args.max_per_img)

    out_dir = Path(args.out).expanduser().resolve()
    all_results = []
    for path in files:
        dets, infer_ms = detector.detect(path, args.score_thr)
        vis = None
        save_path = None
        if not args.no_save:
            save_path = out_dir / f'{path.stem}_pred.jpg'
            vis = detector.render(path, dets, save_path)

        print(f'\n== {path}  ({len(dets)} 个目标, 推理 {infer_ms:.1f} ms) ==')
        if dets:
            print(f'{"#":>2}  {"类别":<12}{"置信度":>7}   '
                  f'{"cx":>7}{"cy":>7}{"w":>7}{"h":>7}{"角度":>8}')
            for i, d in enumerate(dets):
                print(f'{i + 1:>2}  {d.class_name:<12}{d.score:>7.3f}   '
                      f'{d.cx:>7.1f}{d.cy:>7.1f}{d.w:>7.1f}{d.h:>7.1f}'
                      f'{d.angle_deg:>7.1f}°')
        else:
            print('(没有超过阈值的检测结果)')
        if save_path is not None:
            print(f'   -> 已保存: {save_path}')
        if args.save_txt and save_path is not None:
            txt = out_dir / f'{path.stem}_pred.txt'
            txt.write_text('\n'.join(d.to_dota_line() for d in dets),
                           encoding='utf-8')
            print(f'   -> 框坐标: {txt}')

        all_results.append(dict(
            image=str(path), num=len(dets), infer_ms=round(infer_ms, 1),
            save_path=str(save_path) if save_path else None,
            size=[int(vis.shape[1]), int(vis.shape[0])] if vis is not None else None,
            detections=[d.to_dict() for d in dets]))

    if args.json:
        Path(args.json).write_text(
            json.dumps(all_results, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'\n[json] {args.json}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
