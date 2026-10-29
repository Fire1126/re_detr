#!/usr/bin/env python
"""尾矿库检测 Web 演示 —— 输入图片路径/上传图片, 返回带框、类别、置信度的结果。

启动:
    python deploy/app.py                 # http://127.0.0.1:7860
    python deploy/app.py --port 8000 --host 0.0.0.0

依赖: 仅使用 Python 标准库 + 项目已有依赖(mmengine/mmdet/mmrotate/opencv), 无需额外安装。
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import random
import sys
import tempfile
import threading
import time
import traceback
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from infer import DEFAULT_CKPT, DEFAULT_CONFIG, IMG_EXTS  # noqa: E402
from infer import TailingDetector  # noqa: E402

STATE: dict = {}


# --------------------------------------------------------------------------- #
# 图片读取: 支持 本地路径 / http(s) / base64
# --------------------------------------------------------------------------- #
def _read_local(path: str) -> bytes:
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = (REPO_ROOT / p)
    if not p.is_file():
        raise FileNotFoundError(f'文件不存在: {p}')
    return p.read_bytes()


def _download(url: str) -> bytes:
    req = urllib.request.Request(url, headers={'User-Agent': 'tailing-demo/1.0'})
    with urllib.request.urlopen(req, timeout=20) as r:
        return r.read()


def _decode_b64(data_url: str) -> bytes:
    if ',' in data_url:
        data_url = data_url.split(',', 1)[1]
    return base64.b64decode(data_url)


def _probe_ext(data: bytes) -> str:
    if data[:8] == b'\x89PNG\r\n\x1a\n':
        return '.png'
    if data[:2] == b'\xff\xd8':
        return '.jpg'
    if data[:2] == b'BM':
        return '.bmp'
    if data[:4] in (b'II*\x00', b'MM\x00*'):
        return '.tif'
    return '.png'


def run_detection(source: dict, score_thr: float) -> dict:
    """source: {'path': str} | {'image': dataURL} ；返回可直接 json 化的结果。"""
    tmp_path: Optional[Path] = None
    t_start = time.perf_counter()

    if source.get('image'):
        raw = _decode_b64(source['image'])
        fd, name = tempfile.mkstemp(suffix=_probe_ext(raw), dir=str(HERE / '.cache'))
        os.close(fd)
        tmp_path = Path(name)
        tmp_path.write_bytes(raw)
        img_path = tmp_path
        shown = '上传图片'
    else:
        given = str(source.get('path', '')).strip()
        if not given:
            raise ValueError('请上传图片或填写图片路径')
        shown = given
        if given.startswith(('http://', 'https://')):
            raw = _download(given)
            fd, name = tempfile.mkstemp(suffix=_probe_ext(raw),
                                        dir=str(HERE / '.cache'))
            os.close(fd)
            tmp_path = Path(name)
            tmp_path.write_bytes(raw)
            img_path = tmp_path
        else:
            img_path = Path(given).expanduser()
            if not img_path.is_absolute():
                img_path = REPO_ROOT / img_path
            if not img_path.is_file():
                raise FileNotFoundError(f'文件不存在: {img_path}')

    try:
        with STATE['lock']:
            dets, infer_ms = STATE['detector'].detect(img_path, score_thr)
            vis = STATE['detector'].render(img_path, dets)
        import cv2
        ok, buf = cv2.imencode('.jpg', vis, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        if not ok:
            raise IOError('结果图编码失败')
        b64 = base64.b64encode(buf.tobytes()).decode('ascii')
        h_img, w_img = vis.shape[:2]
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)

    return dict(
        ok=True,
        source=shown,
        image=f'data:image/jpeg;base64,{b64}',
        width=int(w_img),
        height=int(h_img),
        num=len(dets),
        infer_ms=round(infer_ms, 1),
        total_ms=round((time.perf_counter() - t_start) * 1000, 1),
        score_thr=score_thr,
        detections=[d.to_dict() for d in dets],
    )


# --------------------------------------------------------------------------- #
# HTTP 服务
# --------------------------------------------------------------------------- #
class Handler(BaseHTTPRequestHandler):
    server_version = 'TailingDemo/1.0'

    def log_message(self, fmt, *args):  # 精简日志
        sys.stderr.write('[%s] %s\n' % (time.strftime('%H:%M:%S'), fmt % args))

    # ---------------- helpers ----------------
    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False).encode('utf-8'),
                   'application/json; charset=utf-8')

    # ---------------- routes ----------------
    def do_GET(self):
        path = self.path.split('?')[0]
        try:
            if path in ('/', '/index.html'):
                self._send(200, PAGE.encode('utf-8'), 'text/html; charset=utf-8')
            elif path == '/api/info':
                self._json(info_payload())
            elif path == '/api/samples':
                self._json(dict(ok=True, items=sample_images()))
            elif path == '/health':
                self._json(dict(ok=True))
            elif path == '/favicon.ico':
                self._send(204, b'', 'image/x-icon')
            else:
                self._send(404, b'not found', 'text/plain; charset=utf-8')
        except BrokenPipeError:
            pass
        except Exception:
            traceback.print_exc()
            self._json(dict(ok=False, error='内部错误'), 500)

    def do_POST(self):
        if self.path.split('?')[0] != '/api/detect':
            self._json(dict(ok=False, error='未知接口'), 404)
            return
        try:
            length = int(self.headers.get('Content-Length', 0))
            payload = json.loads(self.rfile.read(length) or b'{}')
            score_thr = float(payload.get('score_thr',
                                          STATE['detector'].score_thr))
            score_thr = min(max(score_thr, 0.0), 0.99)
            source = dict(path=payload.get('path'), image=payload.get('image'))
            self._json(run_detection(source, score_thr))
        except FileNotFoundError as e:
            self._json(dict(ok=False, error=str(e)), 400)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            self._json(dict(ok=False, error=f'{type(e).__name__}: {e}'), 500)


def info_payload() -> dict:
    det = STATE['detector']
    return dict(
        ok=True,
        device=str(det.device),
        weights=det.weights,
        classes=list(det.classes),
        config=str(det.config_path),
        checkpoint=str(det.checkpoint_path),
        score_thr=det.score_thr,
        gpu=STATE.get('gpu_name'),
    )


def sample_images(limit: int = 6) -> list:
    root = REPO_ROOT / 'data' / 'tailing' / 'test' / 'images'
    if not root.is_dir():
        return []
    files = [p for p in root.iterdir() if p.suffix.lower() in IMG_EXTS]
    if not files:
        return []
    picked = random.sample(files, min(limit, len(files)))
    return [str(p.relative_to(REPO_ROOT)) for p in picked]


# --------------------------------------------------------------------------- #
# 前端页面
# --------------------------------------------------------------------------- #
PAGE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>尾矿库检测 · Rotated RT-DETR</title>
<style>
  :root{
    --bg:#f5f6f8; --card:#ffffff; --line:#e6e8ec; --text:#1b1f24; --muted:#6b7280;
    --accent:#1a6dff; --accent-soft:#eaf1ff; --ok:#0f9960; --warn:#d97706;
    --shadow:0 1px 2px rgba(16,24,40,.04), 0 8px 24px rgba(16,24,40,.06);
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--text);
    font:14px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif;}
  header{display:flex;align-items:center;gap:14px;padding:16px 24px;background:var(--card);
    border-bottom:1px solid var(--line);position:sticky;top:0;z-index:10}
  .logo{width:34px;height:34px;border-radius:9px;background:linear-gradient(135deg,#1a6dff,#38bdf8);
    display:grid;place-items:center;color:#fff;font-weight:700;font-size:15px}
  header h1{font-size:16px;margin:0;font-weight:650;letter-spacing:.2px}
  header .sub{font-size:12px;color:var(--muted);margin-top:1px}
  .badges{margin-left:auto;display:flex;gap:8px;flex-wrap:wrap}
  .badge{font-size:12px;color:var(--muted);background:#f2f4f7;border:1px solid var(--line);
    padding:3px 9px;border-radius:999px;white-space:nowrap}
  .badge b{color:var(--text);font-weight:600}

  main{display:grid;grid-template-columns:390px minmax(0,1fr);gap:18px;padding:18px 24px 40px;
    max-width:1680px;margin:0 auto}
  @media(max-width:1000px){main{grid-template-columns:1fr}}

  .card{background:var(--card);border:1px solid var(--line);border-radius:14px;box-shadow:var(--shadow)}
  .card .hd{display:flex;align-items:center;gap:8px;padding:13px 16px;border-bottom:1px solid var(--line);
    font-weight:600;font-size:13.5px}
  .card .hd .dot{width:7px;height:7px;border-radius:50%;background:var(--accent)}
  .card .bd{padding:16px}

  .drop{border:1.5px dashed #c9d3e0;border-radius:12px;padding:22px 16px;text-align:center;
    background:#fafbfd;cursor:pointer;transition:.16s}
  .drop:hover,.drop.over{border-color:var(--accent);background:var(--accent-soft)}
  .drop .icon{font-size:22px;line-height:1}
  .drop .t1{font-weight:600;margin-top:8px;font-size:13.5px}
  .drop .t2{color:var(--muted);font-size:12px;margin-top:2px}

  .sep{display:flex;align-items:center;gap:10px;color:var(--muted);font-size:12px;margin:16px 0 10px}
  .sep::before,.sep::after{content:"";height:1px;background:var(--line);flex:1}

  label.f{display:block;font-size:12.5px;color:var(--muted);margin:0 0 6px}
  input[type=text]{width:100%;padding:9px 11px;border:1px solid var(--line);border-radius:9px;
    font:inherit;font-size:13px;outline:none;background:#fff}
  input[type=text]:focus{border-color:var(--accent);box-shadow:0 0 0 3px rgba(26,109,255,.12)}
  .row{display:flex;gap:8px}
  button{font:inherit;font-weight:600;font-size:13px;border-radius:9px;border:1px solid var(--line);
    background:#fff;color:var(--text);padding:9px 14px;cursor:pointer;transition:.15s;white-space:nowrap}
  button:hover{background:#f7f8fa}
  button.primary{background:var(--accent);border-color:var(--accent);color:#fff;
    box-shadow:0 1px 2px rgba(26,109,255,.35)}
  button.primary:hover{background:#0f5fe0}
  button:disabled{opacity:.55;cursor:not-allowed}
  button.ghost{padding:6px 10px;font-size:12px;font-weight:500;color:var(--muted)}

  .thr{display:flex;align-items:center;justify-content:space-between;margin:18px 0 4px}
  .thr .v{font-variant-numeric:tabular-nums;font-weight:650;color:var(--accent)}
  input[type=range]{width:100%;-webkit-appearance:none;height:4px;border-radius:99px;
    background:#dfe4ec;outline:none}
  input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:16px;height:16px;border-radius:50%;
    background:#fff;border:2px solid var(--accent);box-shadow:0 1px 3px rgba(0,0,0,.2);cursor:pointer}
  .hint{color:var(--muted);font-size:11.5px;margin-top:8px;line-height:1.5}

  .viewer{position:relative;background:#0f1216;border-radius:12px;overflow:hidden;min-height:320px;
    display:grid;place-items:center}
  .viewer img{max-width:100%;max-height:calc(100vh - 290px);display:block}
  .viewer .empty{color:#8b95a5;font-size:13px;text-align:center;padding:60px 20px;line-height:1.9}
  .loading{position:absolute;inset:0;display:grid;place-items:center;background:rgba(15,18,22,.45);
    backdrop-filter:blur(1px)}
  .viewer .spinner{width:26px;height:26px;border:3px solid rgba(255,255,255,.25);
    border-top-color:#fff;border-radius:50%;animation:spin .8s linear infinite}
  @keyframes spin{to{transform:rotate(360deg)}}

  .stats{display:flex;gap:18px;flex-wrap:wrap;padding:12px 16px;border-top:1px solid var(--line);
    font-size:12.5px;color:var(--muted)}
  .stats b{color:var(--text);font-size:13.5px;font-variant-numeric:tabular-nums}
  .stats .ok{color:var(--ok)}

  table{width:100%;border-collapse:collapse;font-size:12.8px}
  th,td{padding:8px 10px;text-align:right;border-bottom:1px solid #f0f2f5;white-space:nowrap}
  th{font-weight:600;color:var(--muted);font-size:11.5px;background:#fbfcfd;position:sticky;top:0}
  th:first-child,td:first-child{text-align:left}
  tbody tr:hover{background:#f8fbff}
  .pill{display:inline-flex;align-items:center;gap:6px}
  .pill i{width:9px;height:9px;border-radius:2px;display:inline-block}
  .score{font-variant-numeric:tabular-nums;font-weight:600}
  .empty-row{color:var(--muted);text-align:center!important;padding:22px}
  .err{color:#b42318;background:#fef3f2;border:1px solid #fecdca;border-radius:9px;
    padding:9px 12px;font-size:12.5px;margin-top:12px;display:none}
  .tbl-wrap{max-height:340px;overflow:auto}
  .kbd{font:11px ui-monospace,monospace;background:#f2f4f7;border:1px solid var(--line);
    border-bottom-width:2px;border-radius:5px;padding:1px 5px}
</style>
</head>
<body>
<header>
  <div class="logo">RR</div>
  <div>
    <h1>尾矿库检测 · Rotated RT-DETR</h1>
    <div class="sub">上传图片或填入图片路径，输出旋转框 + 类别 + 置信度</div>
  </div>
  <div class="badges" id="badges"></div>
</header>

<main>
  <!-- 左栏 -->
  <section class="card">
    <div class="hd"><span class="dot"></span>输入</div>
    <div class="bd">
      <div class="drop" id="drop">
        <div class="icon">🖼️</div>
        <div class="t1">拖拽图片到这里 / 点击选择</div>
        <div class="t2">支持 jpg · png · tif · bmp · webp</div>
      </div>
      <input type="file" id="file" accept="image/*" hidden>

      <div class="sep">或者输入图片地址</div>
      <label class="f" for="path">本地绝对路径 / 相对路径 / http(s) URL</label>
      <div class="row">
        <input type="text" id="path" placeholder="/home/l/xxx/a.png"
               autocomplete="off" spellcheck="false">
        <button class="primary" id="go">检测</button>
      </div>
      <div class="hint">相对路径以仓库根目录为基准，例如
        <code>data/tailing/test/images/2016_104.png</code></div>

      <div class="thr">
        <label class="f" style="margin:0">置信度阈值</label>
        <span class="v" id="thrVal">0.50</span>
      </div>
      <input type="range" id="thr" min="0.05" max="0.9" step="0.05" value="0.5">
      <div class="hint">阈值越低召回越多、误检越多；调低后会自动重新检测。</div>

      <div class="row" style="margin-top:18px">
        <button class="ghost" id="sample" style="flex:1">🎲 取一张测试图</button>
        <button class="ghost" id="clear" style="flex:1">清空结果</button>
      </div>
      <div class="err" id="err"></div>
    </div>
  </section>

  <!-- 右栏 -->
  <section class="card" style="display:flex;flex-direction:column;min-width:0">
    <div class="hd"><span class="dot" style="background:#0f9960"></span>检测结果
      <span class="badge" id="srcTag" style="margin-left:auto;display:none"></span>
    </div>
    <div class="viewer" id="viewer">
      <div class="empty" id="empty">还没有结果<br><span style="font-size:12px">在左侧选择图片开始检测</span></div>
      <img id="out" style="display:none" alt="检测结果">
      <div class="loading" id="loading" style="display:none"><div class="spinner"></div></div>
    </div>
    <div class="stats" id="stats" style="display:none"></div>
    <div class="tbl-wrap">
      <table>
        <thead><tr>
          <th>#</th><th>类别</th><th>置信度</th><th>中心 x</th><th>中心 y</th>
          <th>宽 w</th><th>高 h</th><th>角度</th>
        </tr></thead>
        <tbody id="tbody">
          <tr><td colspan="8" class="empty-row">—</td></tr>
        </tbody>
      </table>
    </div>
  </section>
</main>

<script>
const $ = (s) => document.querySelector(s);
const COLORS = ['#ffc700','#ff9f2e','#3ddc84','#4f8cff','#ff5fa2','#2ee6d6'];
let lastSource = null;         // {path} 或 {image}
let busy = false;

function badge(text, isCode){
  const d = document.createElement('span');
  d.className = 'badge';
  d.innerHTML = text;
  return d;
}

async function loadInfo(){
  try{
    const r = await fetch('/api/info'); const j = await r.json();
    const b = $('#badges'); b.innerHTML = '';
    if(j.gpu || j.device) b.appendChild(badge('设备 <b>' + (j.gpu ? j.gpu : j.device) + '</b>'));
    b.appendChild(badge('权重 <b>' + (j.weights === 'ema' ? 'EMA' : 'RAW') + '</b>'));
    b.appendChild(badge('类别 <b>' + j.classes.join(', ') + '</b>'));
  }catch(e){}
}

function showError(msg){
  const e = $('#err');
  if(!msg){ e.style.display='none'; return; }
  e.textContent = msg; e.style.display='block';
}

function setBusy(on){
  busy = on;
  $('#go').disabled = on;
  $('#loading').style.display = on ? 'grid' : 'none';
}

function renderStats(j){
  const el = $('#stats');
  el.style.display = 'flex';
  el.innerHTML =
    '<div>目标数 <b class="ok">' + j.num + '</b></div>' +
    '<div>推理 <b>' + j.infer_ms + ' ms</b></div>' +
    '<div>总耗时 <b>' + j.total_ms + ' ms</b></div>' +
    '<div>图像尺寸 <b>' + j.width + ' × ' + j.height + '</b></div>' +
    '<div>阈值 <b>' + j.score_thr.toFixed(2) + '</b></div>';
}

function renderTable(dets){
  const tb = $('#tbody');
  if(!dets.length){
    tb.innerHTML = '<tr><td colspan="8" class="empty-row">没有超过阈值的检测结果</td></tr>';
    return;
  }
  tb.innerHTML = dets.map((d,i) => {
    const c = COLORS[d.label % COLORS.length];
    return '<tr>' +
      '<td>' + (i+1) + '</td>' +
      '<td><span class="pill"><i style="background:'+c+'"></i>' + d.class_name + '</span></td>' +
      '<td class="score">' + d.score.toFixed(3) + '</td>' +
      '<td>' + d.cx.toFixed(1) + '</td>' +
      '<td>' + d.cy.toFixed(1) + '</td>' +
      '<td>' + d.w.toFixed(1) + '</td>' +
      '<td>' + d.h.toFixed(1) + '</td>' +
      '<td>' + d.angle_deg.toFixed(1) + '°</td>' +
    '</tr>';
  }).join('');
}

async function detect(){
  if(busy) return;
  const src = lastSource || (($('#path').value.trim()) ? {path: $('#path').value.trim()} : null);
  if(!src){ showError('请先选择图片或填写路径'); return; }
  showError('');
  setBusy(true);
  const thr = parseFloat($('#thr').value);
  try{
    const r = await fetch('/api/detect', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify(Object.assign({score_thr: thr}, src))
    });
    const j = await r.json();
    if(!j.ok) throw new Error(j.error || '检测失败');

    const img = $('#out');
    img.src = j.image;
    img.style.display = 'block';
    $('#empty').style.display = 'none';

    const tag = $('#srcTag'); tag.style.display = 'inline-block';
    tag.textContent = j.source.length > 46 ? '…' + j.source.slice(-44) : j.source;
    tag.title = j.source;

    renderStats(j); renderTable(j.detections);
  }catch(e){
    showError(e.message);
  }finally{
    setBusy(false);
  }
}

/* ---- 交互 ---- */
const drop = $('#drop');
drop.addEventListener('click', () => $('#file').click());
['dragenter','dragover'].forEach(ev => drop.addEventListener(ev, e => {
  e.preventDefault(); drop.classList.add('over');
}));
['dragleave','drop'].forEach(ev => drop.addEventListener(ev, e => {
  e.preventDefault(); drop.classList.remove('over');
}));
drop.addEventListener('drop', e => {
  const f = e.dataTransfer.files && e.dataTransfer.files[0];
  if(f) sendFile(f);
});
$('#file').addEventListener('change', e => {
  const f = e.target.files[0]; if(f) sendFile(f);
});

function sendFile(file){
  const reader = new FileReader();
  reader.onload = () => {
    lastSource = {image: reader.result};
    $('#path').value = '';
    detect();
  };
  reader.readAsDataURL(file);
}

$('#go').addEventListener('click', () => {
  const p = $('#path').value.trim();
  if(p){ lastSource = {path: p}; detect(); }
  else detect();
});
$('#path').addEventListener('keydown', e => { if(e.key === 'Enter') $('#go').click(); });

$('#thr').addEventListener('input', e => { $('#thrVal').textContent = parseFloat(e.target.value).toFixed(2); });
$('#thr').addEventListener('change', () => { if(lastSource) detect(); });

$('#sample').addEventListener('click', async () => {
  const r = await fetch('/api/samples'); const j = await r.json();
  if(!j.items || !j.items.length){ showError('没找到测试图片目录 data/tailing/test/images'); return; }
  const p = j.items[Math.floor(Math.random()*j.items.length)];
  $('#path').value = p; lastSource = {path: p}; detect();
});

$('#clear').addEventListener('click', () => {
  lastSource = null;
  $('#out').style.display = 'none';
  $('#empty').style.display = 'block';
  $('#stats').style.display = 'none';
  $('#srcTag').style.display = 'none';
  $('#tbody').innerHTML = '<tr><td colspan="8" class="empty-row">—</td></tr>';
  showError('');
});

loadInfo();
</script>
</body>
</html>
"""


def _bind(host: str, port: int, max_tries: int = 20):
    """端口被占用时自动往后顺延, 而不是直接抛 OSError。"""
    import errno
    last_err = None
    for p in range(port, port + max_tries):
        try:
            return ThreadingHTTPServer((host, p), Handler), p
        except OSError as e:
            if e.errno in (errno.EADDRINUSE, getattr(errno, 'WSAEADDRINUSE', -1)):
                last_err = e
                print(f'[!] 端口 {p} 已被占用, 换 {p + 1} 试试 ...')
                continue
            raise
    raise SystemExit(
        f'[错误] {port}~{port + max_tries - 1} 都被占用了, 请手动换端口: '
        f'python deploy/app.py --port 8000  ({last_err})')


def main() -> int:
    try:  # 输出重定向到文件时也能实时看到日志
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    parser = argparse.ArgumentParser(description='尾矿库检测 Web 演示')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=7860)
    parser.add_argument('--config', default=str(DEFAULT_CONFIG))
    parser.add_argument('--checkpoint', default=str(DEFAULT_CKPT))
    parser.add_argument('--device', default='auto')
    parser.add_argument('--weights', default='auto', choices=['auto', 'ema', 'raw'])
    parser.add_argument('--score-thr', type=float, default=0.5)
    args = parser.parse_args()

    (HERE / '.cache').mkdir(exist_ok=True)

    print('[*] 正在加载模型 ...')
    detector = TailingDetector(config=args.config, checkpoint=args.checkpoint,
                               device=args.device, weights=args.weights,
                               score_thr=args.score_thr)
    gpu_name = None
    if detector.device.type == 'cuda':
        gpu_name = detector.torch.cuda.get_device_name(detector.device)
        print(f'[*] 设备: {detector.device} ({gpu_name})')
    else:
        print(f'[*] 设备: {detector.device}')

    STATE['detector'] = detector
    STATE['lock'] = threading.Lock()
    STATE['gpu_name'] = gpu_name

    # 预热: 第一次推理会有 cuDNN 选核开销
    samples = sample_images(limit=1)
    if samples:
        try:
            t0 = time.perf_counter()
            with STATE['lock']:
                detector.detect(REPO_ROOT / samples[0])
            print(f'[*] 预热完成 ({(time.perf_counter() - t0) * 1000:.0f} ms)')
        except Exception:
            traceback.print_exc()

    srv, port = _bind(args.host, args.port)
    print(f'[+] 服务已启动: http://{args.host}:{port}  (Ctrl+C 退出)')
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print('\n[*] 正在关闭 ...')
    finally:
        srv.server_close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
