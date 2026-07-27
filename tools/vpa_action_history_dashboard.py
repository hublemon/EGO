#!/usr/bin/env python3
"""Public-safe live dashboard for the VPA action-history ablation queue."""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from zoneinfo import ZoneInfo


REPO = Path(__file__).resolve().parents[1]
RUN = REPO / "runs/vpa_v2"
OUT = RUN / "action_history_ablation"
STATE_PATH = OUT / "pipeline_state.json"
METHODS = (
    "ours_wm1st", "ours_full", "qwen_backbone", "frontier",
    "wm_top1_repeat", "wm_topk_rank",
)
LOCAL = {"ours_wm1st", "ours_full", "qwen_backbone"}
BASELINES = {"wm_top1_repeat", "wm_topk_rank"}
PHASES = (
    {"key": "T3_nohist", "horizon": 3, "history": "none", "total": 915},
    {"key": "T4_full", "horizon": 4, "history": "full", "total": 504},
    {"key": "T4_nohist", "horizon": 4, "history": "none", "total": 504},
)
KST = ZoneInfo("Asia/Seoul")

# Conservative handoff measurement: about 2 h / 915 samples / local arm.
FALLBACK_LOCAL_SEC_PER_SAMPLE = 7200 / 915
FALLBACK_FRONTIER_SEC_PER_SAMPLE = 2.5
EVAL_SEC_PER_PHASE = 120


def load_json(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default


def parse_time(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def seconds_between(a, b):
    aa, bb = parse_time(a), parse_time(b)
    return max(0.0, (bb - aa).total_seconds()) if aa and bb else None


def prefix(phase: dict, arm: str) -> Path:
    return OUT / phase["key"] / "preds" / f"{arm}_T{phase['horizon']}"


def record_stats(path_prefix: Path) -> dict:
    path = path_prefix.with_suffix(".records.jsonl")
    ok, fail, rows = set(), 0, 0
    if path.is_file():
        try:
            with path.open(encoding="utf-8") as fh:
                for line in fh:
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    rows += 1
                    if row.get("ok") and row.get("sample_id"):
                        ok.add(row["sample_id"])
                    elif not row.get("ok"):
                        fail += 1
        except OSError:
            pass
    return {
        "done": len(ok),
        "fail_rows": fail,
        "rows": rows,
        "mtime": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()
        if path.is_file() else None,
    }


def pid_alive(pid) -> bool:
    try:
        return bool(pid) and Path(f"/proc/{int(pid)}").is_dir()
    except (TypeError, ValueError):
        return False


def gpu_status() -> list[dict]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        out = subprocess.check_output(cmd, text=True, timeout=3)
        rows = []
        for line in out.splitlines():
            vals = [x.strip() for x in line.split(",")]
            rows.append({
                "index": vals[0], "name": vals[1], "memory_used_mb": int(vals[2]),
                "memory_total_mb": int(vals[3]), "utilization": int(vals[4]),
                "temperature_c": int(vals[5]),
            })
        return rows
    except Exception as exc:
        return [{"error": str(exc)[:160]}]


def completed_rates(state: dict) -> tuple[float, float]:
    local_rates, frontier_rates = [], []
    for phase in PHASES:
        ps = state.get("phases", {}).get(phase["key"], {})
        for arm in (*LOCAL, "frontier"):
            row = ps.get("arms", {}).get(arm, {})
            if row.get("state") == "completed":
                duration = seconds_between(row.get("started_at"), row.get("finished_at"))
                initial_done = int(row.get("initial_done") or 0)
                # A resume-only verification may set initial_done=total even
                # though started/finished still describe the original full run.
                processed = (
                    phase["total"]
                    if initial_done >= phase["total"]
                    else max(1, phase["total"] - initial_done)
                )
            elif row.get("state") in {"running", "smoke"}:
                started = parse_time(row.get("started_at"))
                stats = record_stats(prefix(phase, arm))
                processed = max(
                    0, stats["done"] - int(row.get("initial_done") or 0)
                )
                threshold = 20 if arm in LOCAL else 3
                duration = (
                    (datetime.now(timezone.utc) - started).total_seconds()
                    if started and processed >= threshold else None
                )
            else:
                duration, processed = None, 0
            if (
                row.get("state") == "deferred"
                and row.get("observed_sec_per_sample")
            ):
                (local_rates if arm in LOCAL else frontier_rates).append(
                    float(row["observed_sec_per_sample"])
                )
                continue
            if duration and processed:
                rate = duration / processed
                # Keep a transient API stall or model load from exploding all
                # future-phase ETAs beyond a useful display range.
                cap = (
                    FALLBACK_LOCAL_SEC_PER_SAMPLE * 4
                    if arm in LOCAL else 120.0
                )
                rate = min(max(rate, 0.15), cap)
                (local_rates if arm in LOCAL else frontier_rates).append(rate)
    local = statistics.median(local_rates) if local_rates else FALLBACK_LOCAL_SEC_PER_SAMPLE
    frontier = (
        statistics.median(frontier_rates)
        if frontier_rates else FALLBACK_FRONTIER_SEC_PER_SAMPLE
    )
    return local, frontier


def arm_eta(row: dict, done: int, total: int, fallback_sec: float) -> float:
    if row.get("state") == "completed" or done >= total:
        return 0.0
    remaining = max(0, total - done)
    started = parse_time(row.get("started_at"))
    initial = int(row.get("initial_done") or 0)
    delta = max(0, done - initial)
    if started and row.get("state") in {"running", "smoke"}:
        elapsed = max(0.0, (datetime.now(timezone.utc) - started).total_seconds())
        if delta:
            live = elapsed / delta
            # Avoid a transient one-sample outlier producing an absurd public ETA.
            sec_per = min(max(live, 0.15), fallback_sec * 4)
            return remaining * sec_per
        model_load_remaining = max(0.0, min(180.0, 133.0 - elapsed))
        return model_load_remaining + remaining * fallback_sec
    return remaining * fallback_sec


def metric_for(phase: dict, arm: str):
    summary = load_json(OUT / phase["key"] / "summary.json", {}) or {}
    metric = (summary.get("metrics") or {}).get(arm)
    if metric:
        return {
            "SR": metric.get("SR"),
            "mAcc": metric.get("mAcc"),
            "mIoU": metric.get("mIoU"),
            "reportable": metric.get("reportable"),
        }
    metrics = load_json(
        OUT / phase["key"] / "metrics"
        / f"metrics_T{phase['horizon']}_frames_subset_T{phase['horizon']}.json",
        {},
    ) or {}
    metric = metrics.get(arm)
    if not metric:
        return None
    return {
        "SR": metric.get("SR"),
        "mAcc": metric.get("mAcc"),
        "mIoU": metric.get("mIoU"),
        "reportable": metric.get("reportable"),
    }


def recent_logs() -> tuple[str | None, list[str]]:
    logs = list((OUT / "logs").glob("*.log")) if (OUT / "logs").is_dir() else []
    if not logs:
        return None, []
    latest = max(logs, key=lambda p: p.stat().st_mtime)
    try:
        lines = latest.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return latest.name, []
    return latest.name, lines[-45:]


def status() -> dict:
    state = load_json(STATE_PATH, {}) or {}
    local_sec, frontier_sec = completed_rates(state)
    phase_rows = []
    total_work = done_work = 0
    eta_total = 0.0
    gpu_eta_total = 0.0

    for phase in PHASES:
        ps = state.get("phases", {}).get(phase["key"], {})
        arms, local_etas, frontier_eta = [], [], 0.0
        for arm in METHODS:
            row = (ps.get("arms") or {}).get(arm, {})
            if arm in BASELINES:
                done = phase["total"] if row.get("state") == "completed" else 0
                fail_rows = 0
                eta = 0.0 if done else 8.0
            else:
                stats = record_stats(prefix(phase, arm))
                done, fail_rows = stats["done"], stats["fail_rows"]
                fallback = local_sec if arm in LOCAL else frontier_sec
                eta = arm_eta(row, done, phase["total"], fallback)
            arm_state = row.get("state", "queued")
            if done >= phase["total"]:
                arm_state = "completed"
            elif arm_state in {"running", "smoke"} and not pid_alive(row.get("pid")):
                # The child can briefly be between retry attempts; the orchestrator
                # PID distinguishes that from a genuinely disconnected queue.
                if not pid_alive(state.get("pid")):
                    arm_state = "stale"
            total_work += phase["total"]
            done_work += min(done, phase["total"])
            arms.append({
                "name": arm,
                "state": arm_state,
                "done": done,
                "total": phase["total"],
                "pct": round(100 * done / phase["total"], 2),
                "fail_rows": fail_rows,
                "eta_sec": round(eta),
                "pid": row.get("pid"),
                "attempt": row.get("attempt"),
                "metric": metric_for(phase, arm),
            })
            if arm in LOCAL:
                local_etas.append(eta)
            elif arm == "frontier":
                frontier_eta = eta

        phase_state = ps.get("state", "queued")
        if (OUT / phase["key"] / "summary.json").is_file():
            phase_state = "completed"
        post = 0 if phase_state == "completed" else EVAL_SEC_PER_PHASE
        schedule = state.get("schedule")
        if schedule == "gpu_parallel_first_then_frontier":
            gpu_phase_eta = max(local_etas, default=0.0)
            phase_eta = gpu_phase_eta + frontier_eta + post
        elif schedule == "gpu_first_then_frontier":
            gpu_phase_eta = sum(local_etas)
            phase_eta = gpu_phase_eta + frontier_eta + post
        else:
            gpu_phase_eta = sum(local_etas)
            phase_eta = max(gpu_phase_eta, frontier_eta) + post
        gpu_eta_total += gpu_phase_eta
        eta_total += phase_eta
        phase_rows.append({
            **phase,
            "state": phase_state,
            "started_at": ps.get("started_at"),
            "finished_at": ps.get("finished_at"),
            "eta_sec": round(phase_eta),
            "arms": arms,
        })

    expected = datetime.now(KST) + timedelta(seconds=eta_total)
    gpu_expected = datetime.now(KST) + timedelta(seconds=gpu_eta_total)
    log_name, logs = recent_logs()
    return {
        "title": state.get("title", "VPA action-history ablation"),
        "pipeline_state": state.get("pipeline_state", "preparing"),
        "schedule": state.get("schedule", "phase_parallel"),
        "queue_stage": state.get("queue_stage"),
        "pipeline_pid": state.get("pid"),
        "pipeline_alive": pid_alive(state.get("pid")),
        "error": state.get("error"),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "state_updated_at": state.get("updated_at"),
        "progress": round(100 * done_work / max(1, total_work), 2),
        "done_work": done_work,
        "total_work": total_work,
        "eta_total_sec": round(eta_total),
        "gpu_eta_total_sec": round(gpu_eta_total),
        "expected_finish_kst": expected.strftime("%Y-%m-%d %H:%M KST"),
        "expected_gpu_finish_kst": gpu_expected.strftime("%Y-%m-%d %H:%M KST"),
        "rate_basis": {
            "local_sec_per_sample": round(local_sec, 3),
            "frontier_sec_per_sample": round(frontier_sec, 3),
            "local_source": "measured" if local_sec != FALLBACK_LOCAL_SEC_PER_SAMPLE else "handoff fallback",
            "frontier_source": "measured" if frontier_sec != FALLBACK_FRONTIER_SEC_PER_SAMPLE else "conservative fallback",
        },
        "frame_contract": state.get("frame_contract") or {
            "with_frames": True, "blind_arms": False, "window_sec": 4,
            "safety_gap_sec": 1, "n_frames": 8, "short_side": 336,
        },
        "methods": list(METHODS),
        "phases": phase_rows,
        "gpus": gpu_status(),
        "events": (state.get("events") or [])[-12:],
        "log_name": log_name,
        "logs": logs,
    }


PAGE = r"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>VPA action-history ablation</title>
<style>
:root{color-scheme:dark;--bg:#071018;--card:#101c27;--card2:#09151f;--line:#253848;
--text:#edf6ff;--muted:#90a5b8;--mint:#5eead4;--blue:#60a5fa;--amber:#fbbf24;
--rose:#fb7185;--violet:#c084fc}*{box-sizing:border-box}body{margin:0;background:
radial-gradient(circle at 12% -8%,#143b53 0,transparent 32%),var(--bg);
color:var(--text);font:14px system-ui,-apple-system,sans-serif}.wrap{max-width:1220px;margin:auto;
padding:28px 18px 80px}h1{font-size:28px;margin:0}.sub{color:var(--muted);margin-top:7px}
.top{display:flex;justify-content:space-between;gap:18px;align-items:flex-start}.live{display:flex;
align-items:center;gap:8px;color:var(--mint);font-weight:700}.dot{width:9px;height:9px;border-radius:50%;
background:var(--mint);box-shadow:0 0 14px var(--mint)}.grid{display:grid;
grid-template-columns:repeat(5,1fr);gap:10px;margin:20px 0}.kpi,.phase,.contract,.logs{
background:#101c27dd;border:1px solid var(--line);border-radius:16px;padding:16px}.kpi span,
.label{display:block;color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.1em}
.kpi b{display:block;font-size:22px;margin-top:7px}.bar{height:8px;border-radius:99px;background:#263644;
overflow:hidden;margin-top:12px}.fill{height:100%;background:linear-gradient(90deg,var(--blue),var(--mint))}
.phase{margin-top:14px}.phasehead{display:flex;justify-content:space-between;gap:10px;align-items:center}
.phase h2{font-size:20px;margin:0}.badge{border:1px solid var(--line);border-radius:99px;padding:5px 9px;
font-size:11px;text-transform:uppercase}.completed{color:var(--mint);border-color:#2c6b62}
.running,.smoke,.evaluating,.gpu_running,.frontier_running{color:var(--blue);border-color:#365f82}
.queued,.preparing,.waiting_frontier,.deferred{color:var(--muted)}
.failed,.stale,.interrupted{color:var(--rose);border-color:#713947}.table{margin-top:14px;border-top:1px solid var(--line)}
.row{display:grid;grid-template-columns:1.2fr .72fr 1.25fr .65fr 1.2fr;gap:10px;align-items:center;
padding:11px 4px;border-bottom:1px solid #20313f}.head{color:var(--muted);font-size:11px;
text-transform:uppercase}.mono{font:12px ui-monospace,SFMono-Regular,monospace}.mini{height:5px;background:#293b49;
border-radius:9px;overflow:hidden;margin-top:5px}.mini i{display:block;height:100%;background:var(--blue)}
.metric{font-size:12px;color:#bed0dd}.contract{margin-top:18px}.checks{display:flex;flex-wrap:wrap;gap:8px;
margin-top:12px}.check{border:1px solid #2c6b62;color:var(--mint);border-radius:10px;padding:8px 10px}
pre{white-space:pre-wrap;word-break:break-word;background:#061019;border-radius:11px;padding:13px;
max-height:330px;overflow:auto;color:#bad0df;font:12px ui-monospace}.err{color:var(--rose);margin-top:12px}
@media(max-width:850px){.grid{grid-template-columns:repeat(2,1fr)}.row{grid-template-columns:1.2fr .8fr 1.2fr}
.row>*:nth-child(4),.row>*:nth-child(5){display:none}.top{display:block}.live{margin-top:12px}}
</style></head><body><main class="wrap">
<div class="top"><div><h1>VPA action-history ablation</h1>
<div class="sub">GPU-first 3-way parallel: T3/T4 local arms → deferred frontier API · 5초 자동 갱신</div></div>
<div class="live"><i class="dot"></i><span id="live">connecting</span></div></div>
<section class="grid">
<div class="kpi"><span>예측 커버리지 (arm×sample)</span><b id="progress">—</b><div class="bar"><div class="fill" id="fill"></div></div></div>
<div class="kpi"><span>Frontier 시작까지</span><b id="gpueta">—</b></div>
<div class="kpi"><span>GPU 작업 종료 (KST)</span><b id="gpufinish">—</b></div>
<div class="kpi"><span>전체 종료 · Frontier 포함</span><b id="finish">—</b></div>
<div class="kpi"><span>GPU</span><b id="gpu">—</b><div class="sub" id="gpud"></div></div>
</section>
<section class="contract"><span class="label">고정 실험 계약</span><div class="checks">
<span class="check">✓ 모든 모델 arm: video 8 frames</span><span class="check">✓ blind/no-video arm 없음</span>
<span class="check">✓ ablation: completed action text만</span><span class="check">✓ 요청한 6개 방법만</span>
<span class="check">✓ 전수 표본 · 100% 완료 후 채점</span></div><div class="err" id="error"></div></section>
<div id="phases"></div>
<section class="logs"><span class="label" id="logname">최근 로그</span><pre id="logs">waiting…</pre></section>
</main><script>
const $=x=>document.getElementById(x), esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
function dur(s){if(s==null)return '—';s=Math.max(0,Math.round(s));let h=Math.floor(s/3600),m=Math.floor(s%3600/60);return h?`${h}시간 ${m}분`:`${m}분`}
function metric(m){return m?`SR ${m.SR.toFixed(2)} · mAcc ${m.mAcc.toFixed(2)} · mIoU ${m.mIoU.toFixed(2)}`:'—'}
function phaseHTML(p){return `<section class="phase"><div class="phasehead"><div><h2>${esc(p.key)}</h2>
<div class="sub">T=${p.horizon} · action history ${p.history==='none'?'없음':'있음'} · ${p.total} samples</div></div>
<span class="badge ${esc(p.state)}">${esc(p.state)}</span></div><div class="table">
<div class="row head"><div>method</div><div>state</div><div>progress</div><div>ETA</div><div>metrics</div></div>
${p.arms.map(a=>`<div class="row"><div><b>${esc(a.name)}</b>${a.fail_rows?`<div class="err">${a.fail_rows} failed rows (retry log)</div>`:''}</div>
<div><span class="badge ${esc(a.state)}">${esc(a.state)}</span></div><div class="mono">${a.done}/${a.total} · ${a.pct.toFixed(1)}%
<div class="mini"><i style="width:${a.pct}%"></i></div></div><div class="mono">${dur(a.eta_sec)}</div>
<div class="metric">${metric(a.metric)}</div></div>`).join('')}</div></section>`}
async function refresh(){try{let d=await fetch('/api/status',{cache:'no-store'}).then(r=>r.json());
$('live').textContent=`${d.pipeline_state}${d.queue_stage?' · '+d.queue_stage:''} · UTC ${d.updated_at.slice(11,19)}`;
$('progress').textContent=`${d.progress.toFixed(1)}%`;$('fill').style.width=d.progress+'%';
$('gpueta').textContent=dur(d.gpu_eta_total_sec);$('gpufinish').textContent=d.expected_gpu_finish_kst.replace(' KST','');
$('finish').textContent=d.expected_finish_kst.replace(' KST','');
let g=d.gpus[0]||{};$('gpu').textContent=g.error?'unavailable':`${g.utilization}%`;
$('gpud').textContent=g.error?g.error:`${g.name} · ${(g.memory_used_mb/1024).toFixed(1)} / ${(g.memory_total_mb/1024).toFixed(1)} GiB · ${g.temperature_c}°C`;
$('error').textContent=d.error||'';$('phases').innerHTML=d.phases.map(phaseHTML).join('');
$('logname').textContent=d.log_name?`최근 로그 · ${d.log_name}`:'최근 로그';$('logs').textContent=(d.logs||[]).join('\n')||'waiting…';
}catch(e){$('live').textContent='reconnecting'}}refresh();setInterval(refresh,5000);
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def send_body(self, body: bytes, content_type: str, code: int = 200) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/api/status"):
            self.send_body(
                json.dumps(status(), ensure_ascii=False).encode(),
                "application/json; charset=utf-8",
            )
        elif self.path.startswith("/healthz"):
            self.send_body(b"ok\n", "text/plain; charset=utf-8")
        elif self.path == "/" or self.path.startswith("/?"):
            self.send_body(PAGE.encode(), "text/html; charset=utf-8")
        else:
            self.send_body(b"not found\n", "text/plain; charset=utf-8", 404)

    def log_message(self, fmt, *args):
        return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7868)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(
        f"[dashboard] http://{args.host}:{args.port} "
        f"(state={STATE_PATH})",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
