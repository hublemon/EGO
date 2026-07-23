#!/usr/bin/env python3
"""Dependency-free live dashboard for a GoalStep feature/train run."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUN = REPO / "outputs/goalstep/runs/z1_jihun2"
CACHE = REPO.parent / "datasets/Ego4D/goalstep_feature_cache_jihun2"
TOTAL = {"train": 30374, "val": 7214}
TITLE = "GoalStep Z=1 · Full Training"
EPOCHS = 15


def tail(path: Path, lines: int = 30) -> list[str]:
    if not path.is_file():
        return []
    return path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]


def count_cache(split: str) -> int:
    folder = CACHE / split
    return sum(1 for _ in folder.glob("*.pt")) if folder.is_dir() else 0


def history() -> list[dict]:
    path = RUN / "training_history.csv"
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def gpu_stats() -> list[dict]:
    try:
        output = subprocess.check_output([
            "nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu",
            "--format=csv,noheader,nounits",
        ], text=True, timeout=3)
        keys = ("index", "name", "util", "memory_used", "memory_total", "temperature")
        return [dict(zip(keys, (x.strip() for x in row.split(",")))) for row in output.splitlines()]
    except Exception as exc:
        return [{"error": str(exc)}]


def status() -> dict:
    counts = {split: count_cache(split) for split in TOTAL}
    hist = history()
    queue_lines = tail(RUN / "logs/queue.log", 20)
    run_status_path = RUN / "run_status.json"
    run_status = (
        json.loads(run_status_path.read_text(encoding="utf-8"))
        if run_status_path.is_file()
        else None
    )
    final_path = RUN / "final_metrics.json"
    final = json.loads(final_path.read_text()) if final_path.is_file() else None
    if final:
        phase = "completed"
    elif run_status and run_status.get("state") == "paused":
        phase = "paused"
    elif (RUN / "logs/train.log").is_file() and tail(RUN / "logs/train.log"):
        phase = "training"
    elif any(counts.values()):
        phase = "feature_extraction"
    elif queue_lines and any("queued:" in line for line in queue_lines):
        phase = "queued"
    else:
        phase = "starting"
    return {
        "title": TITLE,
        "epochs": EPOCHS,
        "phase": phase,
        "run_status": run_status,
        "cache": {s: {"done": counts[s], "total": TOTAL[s], "percent": round(100 * counts[s] / TOTAL[s], 2)} for s in TOTAL},
        "history": hist,
        "latest": hist[-1] if hist else None,
        "final": final,
        "gpus": gpu_stats(),
        "queue_log": queue_lines,
        "pipeline_log": tail(RUN / "logs/pipeline.log", 20),
        "train_log": tail(RUN / "logs/train.log", 24),
        "extract_train_log": tail(RUN / "logs/extract_train.log", 8),
        "extract_val_log": tail(RUN / "logs/extract_val.log", 8),
    }


HTML = r'''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>EGO · GoalStep Z=1 Live</title><style>
:root{color-scheme:dark;--bg:#071018;--card:#101c27;--line:#243443;--mint:#5eead4;--blue:#60a5fa;--muted:#91a3b5}*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 10% 0,#123047 0,transparent 35%),var(--bg);font:15px system-ui;color:#edf6ff}.wrap{max-width:1180px;margin:auto;padding:34px 20px}h1{font-size:28px;margin:0}.sub{color:var(--muted);margin:7px 0 28px}.grid{display:grid;grid-template-columns:repeat(12,1fr);gap:14px}.card{grid-column:span 4;background:#101c27dd;border:1px solid var(--line);border-radius:16px;padding:18px;box-shadow:0 12px 35px #0004}.wide{grid-column:span 8}.full{grid-column:1/-1}.label{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.12em}.value{font-size:27px;font-weight:750;margin-top:5px}.bar{height:8px;background:#253440;border-radius:99px;overflow:hidden;margin-top:12px}.fill{height:100%;background:linear-gradient(90deg,var(--blue),var(--mint));transition:width .5s}.pill{display:inline-flex;gap:7px;align-items:center;border:1px solid #2b4654;border-radius:99px;padding:7px 12px;color:var(--mint)}.dot{width:8px;height:8px;border-radius:50%;background:var(--mint);box-shadow:0 0 12px var(--mint)}canvas{width:100%;height:270px}pre{margin:10px 0 0;white-space:pre-wrap;max-height:280px;overflow:auto;color:#bad0df;font:12px ui-monospace;background:#09131c;padding:13px;border-radius:10px}.metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-top:14px}.metric{background:#0a151e;border-radius:11px;padding:12px}.metric b{display:block;font-size:19px;margin-top:4px}@media(max-width:800px){.card,.wide{grid-column:1/-1}.metrics{grid-template-columns:1fr}}</style></head><body><div class="wrap">
<div style="display:flex;justify-content:space-between;align-items:start;gap:20px"><div><h1 id="title">GoalStep Z=1 · Full Training</h1><div class="sub">실제 피처 캐시와 학습 로그를 5초마다 갱신합니다.</div></div><div class="pill"><span class="dot"></span><span id="phase">loading</span></div></div>
<div class="grid"><section class="card"><div class="label">Train features</div><div class="value" id="trv">—</div><div class="bar"><div class="fill" id="trb"></div></div></section><section class="card"><div class="label">Val features</div><div class="value" id="vav">—</div><div class="bar"><div class="fill" id="vab"></div></div></section><section class="card"><div class="label">Epoch</div><div class="value" id="epoch">0 / 15</div><div class="sub" id="loss">학습 대기 중</div></section>
<section class="card wide"><div class="label">Validation metrics by epoch</div><canvas id="chart" width="760" height="270"></canvas></section><section class="card"><div class="label">Latest measured metrics</div><div class="metrics" id="metrics"></div></section>
<section class="card full"><div class="label">GPU telemetry</div><div class="metrics" id="gpus"></div></section><section class="card full"><div class="label">Live process log</div><pre id="logs">Waiting for logs…</pre></section></div></div>
<script>
const $=id=>document.getElementById(id), num=v=>v==null?'—':Number(v).toFixed(2);function chart(rows){const c=$('chart'),x=c.getContext('2d'),W=c.width,H=c.height;x.clearRect(0,0,W,H);x.strokeStyle='#243443';x.fillStyle='#91a3b5';x.font='12px system-ui';for(let y=0;y<=100;y+=20){let py=H-28-y*(H-48)/100;x.beginPath();x.moveTo(42,py);x.lineTo(W-12,py);x.stroke();x.fillText(y,8,py+4)}const series=[['action_cmr@5','#fbbf24'],['action_top1','#5eead4'],['action_top5','#60a5fa'],['action_top10','#c084fc'],['action_top15','#fb7185']];series.forEach(([k,col],si)=>{x.strokeStyle=col;x.lineWidth=3;x.beginPath();rows.forEach((r,i)=>{let px=42+i*(W-65)/Math.max(1,(window.EPOCHS||15)-1),py=H-28-Number(r[k])*(H-48)/100;i?x.lineTo(px,py):x.moveTo(px,py)});x.stroke();x.fillStyle=col;x.fillText(k.replace('action_',''),W-350+si*68,16)});}
async function refresh(){try{const d=await fetch('/api/status',{cache:'no-store'}).then(r=>r.json());window.EPOCHS=d.epochs;$('title').textContent=d.title;$('phase').textContent=d.phase.replaceAll('_',' ');for(const [s,p] of [['train','tr'],['val','va']]){let q=d.cache[s];$(p+'v').textContent=`${q.done.toLocaleString()} / ${q.total.toLocaleString()} (${q.percent}%)`;$(p+'b').style.width=q.percent+'%'}let l=d.latest;$('epoch').textContent=`${l?l.epoch:0} / ${d.epochs}`;$('loss').textContent=l?`train loss ${l.train_loss} · ${l.seconds}s`:d.phase==='queued'?'선행 16초 실험 완료 대기 중':'전체 피처 추출 중';let heads=['verb','noun','action'].filter(h=>l&&l[h+'_top5']!==undefined);$('metrics').innerHTML=heads.length?heads.map(h=>`<div class="metric"><span class="label">${h} CMR@5</span><b>${num(l[h+'_cmr@5'])}</b><small>top1 ${num(l[h+'_top1'])} · top5 ${num(l[h+'_top5'])} · top10 ${num(l[h+'_top10'])} · top15 ${num(l[h+'_top15'])}</small></div>`).join(''):'<span class="sub">첫 epoch 평가 후 표시됩니다.</span>';chart(d.history);$('gpus').innerHTML=d.gpus.map(g=>g.error?g.error:`<div class="metric"><span class="label">GPU ${g.index} · ${g.name}</span><b>${g.util}%</b><small>${g.memory_used} / ${g.memory_total} MiB · ${g.temperature}°C</small></div>`).join('');let logs=[...d.queue_log,...d.pipeline_log,...d.train_log,...d.extract_train_log,...d.extract_val_log];$('logs').textContent=logs.join('\n')||'Waiting for logs…'}catch(e){$('phase').textContent='reconnecting'}}refresh();setInterval(refresh,5000);
</script></body></html>'''


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/api/status"):
            body = json.dumps(status(), ensure_ascii=False).encode()
            ctype = "application/json; charset=utf-8"
        elif self.path == "/" or self.path.startswith("/?"):
            body = HTML.encode()
            ctype = "text/html; charset=utf-8"
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--run-dir", default=str(RUN))
    parser.add_argument("--cache-dir", default=str(CACHE))
    parser.add_argument("--train-total", type=int, default=TOTAL["train"])
    parser.add_argument("--val-total", type=int, default=TOTAL["val"])
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--title", default=TITLE)
    args = parser.parse_args()
    RUN = Path(args.run_dir).resolve()
    CACHE = Path(args.cache_dir).resolve()
    TOTAL = {"train": args.train_total, "val": args.val_total}
    TITLE = args.title
    EPOCHS = args.epochs
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()
