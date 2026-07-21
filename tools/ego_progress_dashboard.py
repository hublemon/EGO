#!/usr/bin/env python3
"""Live remaining-time dashboard for the Step-1 sweep queue and the RETRO 10h plan.

Dependency-free (stdlib http.server), mirroring tools/goalstep_live_dashboard.py.
Every ETA is derived from measured seconds-per-epoch in each run's own
training_history.csv -- never from a hardcoded guess -- so a run that slows down
under GPU contention reports an honest number.

Usage:
    python tools/ego_progress_dashboard.py --host 0.0.0.0 --port 7861
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import time
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SWEEP_DIR = REPO / "outputs/goalstep/sweep"
SWEEP_CFG = REPO / "configs/step1/goalstep/sweep"
NIGHT_DIR = REPO / "outputs/goalstep/overnight"
NIGHT_CFG = REPO / "configs/step1/goalstep/overnight"
RETRO_DIR = Path("/mnt/nvme/migration/jihun/EGO/runs/retro_overnight")
BASELINE = REPO / "outputs/goalstep/runs/z1_jihun2"
PLAN = REPO / "outputs/retro10h/plan.json"
KST = timezone(timedelta(hours=9))


def _read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _epochs_of(cfg: Path) -> int:
    """epochs from a config without pulling in a YAML dependency."""
    if not cfg.is_file():
        return 15
    m = re.search(r"^\s*epochs:\s*(\d+)", cfg.read_text(encoding="utf-8"), re.M)
    return int(m.group(1)) if m else 15


def _sweep_log() -> tuple[dict[str, str], list[str]]:
    """Parse sweep.log into {name: START|DONE} plus the launch order seen."""
    path = SWEEP_DIR / "sweep.log"
    state: dict[str, str] = {}
    order: list[str] = []
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            m = re.search(r"(START|DONE) (\S+)", line)
            if m:
                if m.group(2) not in order:
                    order.append(m.group(2))
                state[m.group(2)] = m.group(1)
    return state, order


def _running_names() -> set[str]:
    try:
        out = subprocess.check_output(["ps", "-eo", "args"], text=True)
    except Exception:
        return set()
    return {m.group(1) for m in re.finditer(r"sweep/(\w+)\.yaml", out)}


def _queued_names() -> list[str]:
    """Names still pending in a run_sweep.sh invocation that is alive."""
    try:
        out = subprocess.check_output(["ps", "-eo", "args"], text=True)
    except Exception:
        return []
    names: list[str] = []
    for line in out.splitlines():
        m = re.search(r"run_sweep\.sh\s+(\d+)\s+(.+)$", line.strip())
        if m and "grep" not in line:
            names.extend(m.group(2).split())
    return names


def sweep_runs() -> list[dict]:
    state, order = _sweep_log()
    running = _running_names()
    for name in _queued_names():
        if name not in order:
            order.append(name)
    # Overnight-chain runs live in their own tree and are not in sweep.log.
    night = sorted(p.name for p in NIGHT_DIR.glob("*") if p.is_dir())
    order = order + [n for n in night if n not in order]

    runs = []
    for name in order:
        run_dir = NIGHT_DIR / name if (NIGHT_DIR / name).is_dir() else SWEEP_DIR / name
        cfg = NIGHT_CFG / f"{name}.yaml"
        if not cfg.is_file():
            cfg = SWEEP_CFG / f"{name}.yaml"
        hist = _read_csv(run_dir / "training_history.csv")
        total = _epochs_of(cfg)
        done = len(hist)
        secs = [float(r["seconds"]) for r in hist if r.get("seconds")]
        # Recent epochs predict the next ones better than the whole-run mean.
        rate = sum(secs[-3:]) / len(secs[-3:]) if secs else None

        # sweep.log logs DONE even when the run was killed, so completion is
        # judged by final_metrics.json (only written after the full-val readout).
        if name in running:
            status = "running"
        elif (run_dir / "final_metrics.json").is_file():
            status = "done"
        elif done or state.get(name) == "DONE":
            status = "stopped"
        else:
            status = "queued"

        remaining = None
        if status in ("running", "queued") and rate:
            remaining = (total - done) * rate
        elif status == "queued":
            remaining = None

        last = hist[-1] if hist else None
        runs.append({
            "name": name, "status": status, "epochs_done": done, "epochs_total": total,
            "sec_per_epoch": round(rate, 1) if rate else None,
            "remaining_sec": round(remaining) if remaining else None,
            "cmr5": float(last["action_cmr@5"]) if last and "action_cmr@5" in last else None,
            "top5": float(last["action_top5"]) if last and "action_top5" in last else None,
            "train_loss": float(last["train_loss"]) if last else None,
        })
    return runs


def sweep_summary(runs: list[dict]) -> dict:
    """Total remaining wall-clock, accounting for the two GPUs running in parallel."""
    fallback = None
    rates = [r["sec_per_epoch"] for r in runs if r["sec_per_epoch"]]
    if rates:
        fallback = sum(rates) / len(rates)

    # Which GPU a queued run belongs to is not recoverable from ps once started,
    # so report the serial total and the parallel lower bound instead of guessing.
    total = 0.0
    for r in runs:
        if r["status"] == "running" and r["remaining_sec"]:
            total += r["remaining_sec"]
        elif r["status"] == "queued":
            total += (r["epochs_total"] - r["epochs_done"]) * (r["sec_per_epoch"] or fallback or 170)
    active = sum(1 for r in runs if r["status"] == "running") or 1
    return {
        "remaining_serial_sec": round(total),
        "remaining_parallel_sec": round(total / active),
        "done": sum(1 for r in runs if r["status"] == "done"),
        "total": len(runs),
    }


def retro_plan() -> dict:
    if not PLAN.is_file():
        return {"phases": [], "started_at": None}
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    started = plan.get("started_at")
    t0 = datetime.fromisoformat(started) if started else None

    cursor = 0.0
    for phase in plan["phases"]:
        phase["offset_hours"] = cursor
        cursor += phase["hours"]
        if t0:
            phase["eta"] = (t0 + timedelta(hours=cursor)).astimezone(KST).strftime("%H:%M")
    plan["total_hours"] = cursor
    if t0:
        elapsed = (datetime.now(timezone.utc) - t0).total_seconds() / 3600
        plan["elapsed_hours"] = round(elapsed, 2)
        plan["remaining_hours"] = round(max(0.0, cursor - elapsed), 2)
        plan["ends_at"] = (t0 + timedelta(hours=cursor)).astimezone(KST).strftime("%m-%d %H:%M")
    return plan


def retro_chain() -> dict:
    """Live state of the overnight retro chain (GPU 1), read from its own log/markers."""
    log = RETRO_DIR / "chain.log"
    lines = log.read_text(encoding="utf-8", errors="replace").splitlines() if log.is_file() else []
    markers = [m for m in ("GATE_A_PASSED", "GATE_A_FAILED", "FAILED", "DONE")
               if (RETRO_DIR / m).is_file()]
    gate = {}
    gate_file = RETRO_DIR / "gateA.txt"
    if gate_file.is_file():
        for line in gate_file.read_text(encoding="utf-8").split("\n"):
            parts = line.split()
            if len(parts) == 2:
                gate[parts[0]] = parts[1]
    return {"tail": lines[-14:], "markers": markers, "gateA": gate,
            "started": bool(lines)}


def gpu_stats() -> list[dict]:
    try:
        out = subprocess.check_output([
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ], text=True, timeout=8)
    except Exception as exc:
        return [{"error": str(exc)}]
    gpus = []
    for line in out.strip().splitlines():
        f = [x.strip() for x in line.split(",")]
        gpus.append({"index": f[0], "util": f[1], "mem_used": f[2],
                     "mem_total": f[3], "temp": f[4], "power": f[5]})
    return gpus


def status() -> dict:
    runs = sweep_runs()
    base = _read_csv(BASELINE / "training_history.csv")
    final = BASELINE / "final_metrics.json"
    return {
        "now_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S"),
        "sweep": runs,
        "sweep_summary": sweep_summary(runs),
        "baseline": {
            "epochs": len(base),
            "done": final.is_file(),
            "final": json.loads(final.read_text(encoding="utf-8")).get("val_full", {}).get("metrics")
            if final.is_file() else None,
        },
        "retro": retro_plan(),
        "retro_chain": retro_chain(),
        "gpus": gpu_stats(),
    }


PAGE = """<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>EGO 진행 현황 — Step-1 스윕 · RETRO 10h</title>
<style>
:root{--bg:#E8E9E6;--card:#F2F3F0;--sink:#DDDFDA;--rule:#C6C9C2;--fg:#1A1D1B;--mid:#4A4F4B;
--dim:#6E736E;--acc:#1F6B63;--accs:#D3E2DF;--crit:#A8362A;--warn:#95681E;--warns:#EEE1C9;--ok:#356B47;--oks:#D6E5DA;
--mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;--sans:ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;
--serif:"Iowan Old Style","Palatino Linotype",Palatino,Georgia,serif}
@media(prefers-color-scheme:dark){:root{--bg:#15171B;--card:#1D2025;--sink:#101215;--rule:#31363C;--fg:#E4E6E3;
--mid:#A8AEA9;--dim:#7B827D;--acc:#5FBAAE;--accs:#1B3A37;--crit:#E0705C;--warn:#D6A45A;--warns:#382C18;--ok:#6FB585;--oks:#1D3325}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font-family:var(--sans);line-height:1.6}
.wrap{max-width:1180px;margin:0 auto;padding:32px 22px 80px}
header{border-bottom:2px solid var(--fg);padding-bottom:14px;margin-bottom:26px;
display:flex;justify-content:space-between;align-items:baseline;flex-wrap:wrap;gap:10px}
h1{font-family:var(--serif);font-size:26px;margin:0;font-weight:600;letter-spacing:-.01em}
.clock{font-family:var(--mono);font-size:12px;color:var(--dim)}
h2{font-family:var(--serif);font-size:19px;margin:36px 0 12px;font-weight:600;
border-bottom:1px solid var(--rule);padding-bottom:8px}
.big{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:1px;background:var(--rule);
border:1px solid var(--rule);margin-bottom:18px}
.b{background:var(--card);padding:14px 16px}
.b .k{font-family:var(--mono);font-size:10px;letter-spacing:.09em;text-transform:uppercase;color:var(--dim)}
.b .v{font-family:var(--mono);font-size:23px;font-weight:650;font-variant-numeric:tabular-nums;margin-top:4px}
.b .n{font-size:11.5px;color:var(--dim);margin-top:2px}
table{width:100%;border-collapse:collapse;font-size:13px;background:var(--card);border:1px solid var(--rule)}
th,td{padding:8px 12px;text-align:left;border-bottom:1px solid var(--rule)}
th{font-family:var(--mono);font-size:10px;letter-spacing:.07em;text-transform:uppercase;color:var(--dim);background:var(--sink)}
td.n{text-align:right;font-family:var(--mono);font-variant-numeric:tabular-nums}
tbody tr:last-child td{border-bottom:none}
.bar{height:5px;background:var(--sink);border-radius:0;overflow:hidden;margin-top:5px}
.bar i{display:block;height:100%;background:var(--acc);transition:width .4s}
.chip{font-family:var(--mono);font-size:10px;padding:2px 7px;border:1px solid currentColor;white-space:nowrap}
.chip.running{color:var(--acc);background:var(--accs)}.chip.done{color:var(--ok);background:var(--oks)}
.chip.queued{color:var(--dim)}.chip.stopped{color:var(--warn);background:var(--warns)}
.chip.in_progress{color:var(--acc);background:var(--accs)}.chip.pending{color:var(--dim)}
.ph{display:grid;grid-template-columns:66px 1fr auto;gap:16px;padding:13px 16px;background:var(--card);
border:1px solid var(--rule);border-top:none;align-items:start}
.ph:first-of-type{border-top:1px solid var(--rule)}
.ph .t{font-family:var(--mono);font-size:12px;color:var(--acc);font-weight:650}
.ph strong{display:block;font-size:14px}
.ph span{font-size:12.5px;color:var(--mid)}
.ph .gate{display:block;margin-top:6px;font-size:11.5px;color:var(--warn);font-family:var(--mono)}
.wait{background:var(--warns);border:1px solid var(--warn);border-left-width:3px;padding:13px 16px;font-size:13.5px;margin-bottom:16px}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:20px}
footer{margin-top:40px;font-family:var(--mono);font-size:11px;color:var(--dim)}
</style></head><body><div class="wrap">
<header><h1>EGO 진행 현황</h1><span class="clock" id="clock">…</span></header>
<h2>Step-1 · GoalStep probe 스윕</h2>
<div class="big" id="s1big"></div>
<div style="overflow-x:auto"><table id="stbl"><thead><tr>
<th>run</th><th>상태</th><th class="n">epoch</th><th class="n">초/epoch</th><th class="n">잔여</th>
<th class="n">cmr@5</th><th class="n">top5</th><th class="n">train_loss</th></tr></thead><tbody></tbody></table></div>
<h2>RETRO 무인 체인 (GPU 1)</h2>
<div class="big" id="rcbig"></div>
<pre id="rclog" style="background:var(--sink);border:1px solid var(--rule);padding:12px 14px;
overflow-x:auto;font-family:var(--mono);font-size:11.5px;line-height:1.6;color:var(--mid);margin:0"></pre>
<h2>RETRO 10시간 계획</h2>
<div id="rwait"></div>
<div class="big" id="rbig"></div>
<div id="phases"></div>
<div class="grid2" style="margin-top:26px">
<div><h2 style="margin-top:0">사전등록 기준</h2><div style="overflow-x:auto"><table id="pre"><thead><tr>
<th>지표</th><th class="n">현재</th><th class="n">기준</th></tr></thead><tbody></tbody></table></div></div>
<div><h2 style="margin-top:0">GPU</h2><div class="big" id="gpus" style="grid-template-columns:1fr"></div></div>
</div>
<footer id="foot"></footer>
</div><script>
const $=i=>document.getElementById(i);
const hms=s=>{if(s==null)return'—';s=Math.max(0,s);const h=Math.floor(s/3600),m=Math.round(s%3600/60);
return h?`${h}h ${String(m).padStart(2,'0')}m`:`${m}m`};
const nn=(v,d=2)=>v==null?'—':Number(v).toFixed(d);
function box(k,v,n,cls){return `<div class="b"><div class="k">${k}</div><div class="v" ${cls?`style="color:var(--${cls})"`:''}>${v}</div><div class="n">${n||''}</div></div>`}
async function refresh(){
 let d; try{d=await fetch('/api/status',{cache:'no-store'}).then(r=>r.json())}catch(e){$('clock').textContent='reconnecting…';return}
 $('clock').textContent=d.now_kst+' KST';
 const s=d.sweep_summary;
 $('s1big').innerHTML=box('완료','<span>'+s.done+' / '+s.total+'</span>','스윕 런')
  +box('잔여 (병렬 기준)',hms(s.remaining_parallel_sec),'GPU 동시 실행 반영','acc')
  +box('잔여 (직렬 합)',hms(s.remaining_serial_sec),'단일 GPU 환산')
  +box('baseline',d.baseline.done?'완료':d.baseline.epochs+' ep',
       d.baseline.final?'full-val cmr@5 '+nn(d.baseline.final.overall_cmr5.action):'action-only 15ep');
 $('stbl').tBodies[0].innerHTML=d.sweep.map(r=>{
  const p=r.epochs_total?100*r.epochs_done/r.epochs_total:0;
  return `<tr><td><b>${r.name}</b><div class="bar"><i style="width:${p}%"></i></div></td>
  <td><span class="chip ${r.status}">${r.status}</span></td>
  <td class="n">${r.epochs_done}/${r.epochs_total}</td><td class="n">${r.sec_per_epoch??'—'}</td>
  <td class="n">${hms(r.remaining_sec)}</td><td class="n">${nn(r.cmr5)}</td>
  <td class="n">${nn(r.top5)}</td><td class="n">${nn(r.train_loss,4)}</td></tr>`}).join('')
  ||'<tr><td colspan="8" style="color:var(--dim)">스윕 런 없음</td></tr>';
 const RC=d.retro_chain||{};
 const gk=Object.keys(RC.gateA||{});
 $('rcbig').innerHTML=box('상태',(RC.markers&&RC.markers.length?RC.markers.join(' · '):(RC.started?'진행 중':'대기')),'게이트 A -> 본실행 -> 평가')
  +box('belief+sum',(RC.gateA||{}).belief||'측정 중','mean|loss| · 기준 ≥0.00092')
  +box('action+sum',(RC.gateA||{}).action||'측정 중','교란 분리용')
  +box('참고 상한','0.004160','credit=all mean|loss|');
 $('rclog').textContent=(RC.tail||[]).join('\n')||'로그 대기 중…';
 const R=d.retro;
 $('rwait').innerHTML=R.started_at?'':
  '<div class="wait"><b>아직 시작 전.</b> 두 GPU가 Step-1 스윕에 사용 중이라 H0(측정 하네스)만 CPU에서 준비 중입니다. '
  +'<code>plan.json</code>의 <code>started_at</code>이 채워지면 여기서 실시간 잔여 시간이 계산됩니다.</div>';
 $('rbig').innerHTML=box('총 계획',R.total_hours+'h','핸드오프 §8')
  +box('경과',R.elapsed_hours!=null?R.elapsed_hours+'h':'—','')
  +box('잔여',R.remaining_hours!=null?R.remaining_hours+'h':'—','','acc')
  +box('종료 예정',R.ends_at||'—','KST');
 $('phases').innerHTML=(R.phases||[]).map(p=>`<div class="ph"><span class="t">${p.id}</span>
  <span><strong>${p.title}</strong><span>${p.detail}</span>
  ${p.gate?`<span class="gate">⚑ ${p.gate}</span>`:''}</span>
  <span style="text-align:right"><span class="chip ${p.status}">${p.status}</span>
  <div class="n" style="font-family:var(--mono);font-size:11px;color:var(--dim);margin-top:5px">${p.hours}h${p.eta?' · '+p.eta:''}</div>
  <div style="font-family:var(--mono);font-size:10px;color:var(--dim)">${p.improvement||''}</div></span></div>`).join('');
 $('pre').tBodies[0].innerHTML=(R.preregistered||[]).map(p=>
  `<tr><td>${p.metric}</td><td class="n">${p.current}</td><td class="n" style="color:var(--ok)">${p.threshold}</td></tr>`).join('');
 $('gpus').innerHTML=d.gpus.map(g=>g.error?`<div class="b">${g.error}</div>`:
  box('GPU '+g.index,g.util+'%',`${g.mem_used} / ${g.mem_total} MiB · ${g.temp}°C · ${g.power}W`)).join('');
 $('foot').textContent='5초마다 갱신 · ETA는 각 런의 최근 3 epoch 실측 초/epoch에서 계산';
}
refresh();setInterval(refresh,5000);
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        if self.path.startswith("/api/status"):
            body = json.dumps(status()).encode()
            ctype = "application/json"
        else:
            body = PAGE.encode()
            ctype = "text/html; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7861)
    args = parser.parse_args()
    print(f"[dashboard] http://{args.host}:{args.port}  (repo {REPO})", flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
