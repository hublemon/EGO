#!/usr/bin/env python3
"""EGO Step-2 "candidate-CE ↔ projected-SFT combination" 아티팩트 베이커.

실험 run 디렉터리 하나를 읽어 **자기완결(self-contained) HTML 한 장**을 굽는다.
- 외부 네트워크 0 · CSS/JS 인라인 · 데이터는 <script> 안 JSON blob으로 임베드
- claude.ai Artifact로 게시되므로 완전 정적 · GPU 불필요(순수 stdlib + 파일 읽기)
- 반복 호출 안전(idempotent overwrite)

차트 렌더러(VZ.lineChart)와 팔레트는 tools/retro3_dashboard.py 와 공유해 스타일 일관.

사용:
  python3 tools/cesft_v2_artifact.py --run runs/cesft_v2 --out out.html [--now "<iso>"]
프로그램 API:
  render_html(run_dir, now_str) -> str
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from pathlib import Path

# 공유 차트 렌더러 — 같은 tools/ 디렉터리 (retro3 대시보드와 곡선/CSS 공유)
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from retro3_dashboard import CHART_CSS, CHART_JS
except Exception:  # 임포트 실패해도 페이지는 뜨게 — 곡선만 빈다
    CHART_CSS = ""
    CHART_JS = ("const VZ={lineChart:function(h,c){var d=document.createElement('div');"
                "d.textContent='(chart renderer unavailable)';h.appendChild(d);},"
                "renderAll:function(){},qualLog:function(){}};")


# ---------------------------------------------------------------- data helpers
def _read_json(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _read_jsonl(p: Path):
    rows = []
    try:
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
    except Exception:
        return []
    return rows


# 학습 곡선 필드 슬롯 (reasoning/task_belief/action, +sel_ce 있으면)
SFT_KEYS = ("reasoning", "task_belief", "action")


def _collect_train_curves(run_dir: Path, max_pts: int = 160):
    """arm별 train_log.jsonl → loss + field CE(EMA·다운샘플 ≤max_pts) 곡선.

    train 로그 위치(관례):
      outputs/step2_retrospection/<run_name>/<arm>/train_log.jsonl  (중첩)
      outputs/step2_retrospection/<arm>/train_log.jsonl             (평면)
    두 곳을 모두 훑는다. selection-CE arm은 sel_ce 필드가 추가로 있을 수 있다."""
    out = {}
    run_name = run_dir.name
    base = Path("outputs/step2_retrospection")
    candidates: list[Path] = []
    nested = base / run_name
    if nested.is_dir():
        # run 전용 중첩 디렉터리가 있으면 그 arm만 (타 실험 오염 방지)
        candidates += sorted(nested.iterdir())
    elif base.is_dir():
        candidates += sorted(base.iterdir())
    seen = set()
    for d in candidates:
        if not d.is_dir():
            continue
        f = d / "train_log.jsonl"
        if not f.is_file():
            continue
        arm = d.name
        if arm in seen:
            continue
        rows = [r for r in _read_jsonl(f) if "loss" in r and "seen" in r]
        if len(rows) < 2:
            continue
        seen.add(arm)
        # 존재하는 CE 필드만 (sel_ce 포함)
        keys = [k for k in SFT_KEYS if k in rows[0]]
        if "sel_ce" in rows[0]:
            keys.append("sel_ce")
        series_keys = ["loss"] + keys
        ema, pts = {}, []
        for r in rows:
            for k in series_keys:
                if k not in r:
                    continue
                ema[k] = r[k] if k not in ema else 0.9 * ema[k] + 0.1 * r[k]
            pt = {"x": r.get("seen"), "step": r.get("step")}
            for k in series_keys:
                if k in ema:
                    pt[k] = round(ema[k], 4)
            pts.append(pt)
        if len(pts) > max_pts:  # 균등 다운샘플(끝점 보존)
            step = (len(pts) - 1) / (max_pts - 1)
            pts = [pts[round(i * step)] for i in range(max_pts)]
        final = {}
        last = rows[-1]
        for k in series_keys:
            if k in last:
                final[k] = round(float(last[k]), 4)
        out[arm] = {"keys": series_keys, "points": pts, "final": final,
                    "n_rows": len(rows), "last_step": last.get("step")}
    return out


def _collect_probes(run_dir: Path):
    """arm별 probe/*.jsonl → 고정 8샘플 정성 타임라인 전문 + acc 곡선.

    각 행 = 한 checkpoint step: {run,step,ts,probe_acc,samples:[{sample_id,bucket,
    gt,action,correct,task_belief,reasoning_head,malformed}]}."""
    out = {}
    pdir = run_dir / "probe"
    if not pdir.is_dir():
        return out
    for p in sorted(pdir.glob("*.jsonl")):
        entries = _read_jsonl(p)
        entries = [e for e in entries if "samples" in e and "step" in e]
        if not entries:
            continue
        entries.sort(key=lambda e: e.get("step", 0))
        out[p.stem] = {
            "curve": [{"step": e["step"], "acc": e.get("probe_acc")} for e in entries],
            "entries": entries,
            "last_step": entries[-1].get("step"),
        }
    return out


def _collect_interventions(run_dir: Path):
    """arm별 harden_s3 per-sample 개입 덤프(있으면). own vs belief-swap 병치용.

    지원 소스:
      eval/<arm>.harden_s3.json 의 `records` 배열, 또는
      eval/<arm>.harden_s3.records.jsonl
    각 record는 sample_id + own/swap task_belief·action·p_gt (필드명 유연 처리)."""
    out = {}
    edir = run_dir / "eval"
    if not edir.is_dir():
        return out
    for p in sorted(edir.glob("*.harden_s3.json")):
        arm = p.stem[: -len(".harden_s3")]
        data = _read_json(p) or {}
        recs = data.get("records")
        if not recs:
            rf = edir / f"{arm}.harden_s3.records.jsonl"
            if rf.is_file():
                recs = _read_jsonl(rf)
        if not recs:
            continue
        idx = {}
        for r in recs:
            sid = r.get("sample_id") or r.get("id")
            if sid:
                idx[sid] = r
        if idx:
            out[arm] = idx
    return out


def _collect_evals(run_dir: Path):
    """eval/*.json 중 배터리 요약(arm eval)만 — acc 필드 보유 & records 아님."""
    out = {}
    edir = run_dir / "eval"
    if not edir.is_dir():
        return out
    for p in sorted(edir.glob("*.json")):
        st = p.stem
        if st.endswith(".harden_s3") or st.endswith("records"):
            continue
        if st.startswith(("precheck_", "fair_", "gadr_", "paired_", "wise_ft")):
            continue
        v = _read_json(p)
        if isinstance(v, dict) and "acc" in v:
            out[v.get("arm", st)] = v
    return out


def _collect_harden(run_dir: Path):
    """eval/*.harden_s3.json → arm별 belief-개입 요약."""
    out = {}
    edir = run_dir / "eval"
    if not edir.is_dir():
        return out
    for p in sorted(edir.glob("*.harden_s3.json")):
        arm = p.stem[: -len(".harden_s3")]
        v = _read_json(p)
        if isinstance(v, dict):
            out[arm] = v
    return out


def _collect_paired(run_dir: Path):
    """eval/paired_*.json → gate 결과(리스트)."""
    out = []
    edir = run_dir / "eval"
    if not edir.is_dir():
        return out
    for p in sorted(edir.glob("paired_*.json")):
        v = _read_json(p)
        if isinstance(v, dict):
            v = dict(v)
            v.setdefault("_file", p.stem)
            out.append(v)
    return out


def _collect_stages(run_dir: Path):
    """chain.json + markers/ + status/*.json → 스테이지 진행."""
    chain = _read_json(run_dir / "chain.json") or {"stages": []}
    mdir = run_dir / "markers"
    markers = {p.name for p in mdir.glob("*")} if mdir.is_dir() else set()
    statuses = {}
    sdir = run_dir / "status"
    if sdir.is_dir():
        for p in sdir.glob("*.json"):
            s = _read_json(p)
            if isinstance(s, dict) and "stage" in s:
                statuses[s["stage"]] = s
    stages, gpu_line, cur = [], None, None
    for st in chain.get("stages", []):
        live = None
        for key, sv in statuses.items():
            if key.startswith(st["id"]) or st["id"].startswith(key):
                live = sv
                break
        state = "pending"
        rec = {"id": st["id"], "title": st.get("title", st["id"])}
        if st.get("marker") in markers:
            state = "done"
        elif live and live.get("state") == "running":
            state = "running"
        elif live and live.get("state") == "failed":
            state = "failed"
        rec["state"] = state
        if live:
            for k in ("done", "total", "pct", "rate_per_s", "eta_sec", "elapsed_sec"):
                if live.get(k) is not None:
                    rec[k] = live[k]
            rec["metrics"] = {k: v for k, v in (live.get("metrics") or {}).items()
                              if not isinstance(v, dict)}
            if live.get("gpu"):  # 일부 status가 gpu 라인을 실을 수 있음
                gpu_line = live["gpu"]
        if state == "running":
            cur = rec
        stages.append(rec)
    done_n = sum(1 for s in stages if s["state"] == "done")
    return {"stages": stages, "done_n": done_n, "total": len(stages),
            "current": cur, "gpu": gpu_line,
            "chain_done": any(m.endswith("CHAIN_DONE") for m in markers)}


def collect(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    stg = _collect_stages(run_dir)
    return {
        "run_name": run_dir.name,
        "run_path": str(run_dir),
        "stages": stg["stages"],
        "done_n": stg["done_n"],
        "total_stages": stg["total"],
        "current": stg["current"],
        "gpu": stg["gpu"],
        "chain_done": stg["chain_done"],
        "curves": _collect_train_curves(run_dir),
        "probes": _collect_probes(run_dir),
        "interventions": _collect_interventions(run_dir),
        "evals": _collect_evals(run_dir),
        "harden": _collect_harden(run_dir),
        "paired": _collect_paired(run_dir),
        "frontier": _read_json(run_dir / "eval" / "wise_ft_frontier.json"),
    }


def _newest_mtime(run_dir: Path) -> float:
    newest = 0.0
    for root, _dirs, files in os.walk(run_dir):
        for f in files:
            try:
                m = os.path.getmtime(os.path.join(root, f))
                if m > newest:
                    newest = m
            except OSError:
                continue
    return newest


# ---------------------------------------------------------------- page assembly
_VZ_STYLE = """
.vz{--vz-s1:#2a78d6;--vz-s2:#eb6834;--vz-s3:#1baf7a;--vz-s4:#eda100;
 --vz-grid:#e1e0d9;--vz-axis:#c3c2b7;--vz-ink2:#52514e;--vz-mut:#898781;
 --vz-surface:#fcfcfb;--vz-good:#006300;--vz-bad:#d03b3b}
@media (prefers-color-scheme: dark){:root:where(:not([data-theme="light"])) .vz{
 --vz-s1:#3987e5;--vz-s2:#d95926;--vz-s3:#199e70;--vz-s4:#c98500;
 --vz-grid:#2c2c2a;--vz-axis:#383835;--vz-ink2:#c3c2b7;--vz-surface:#1a1a19;
 --vz-good:#0ca30c;--vz-bad:#e66767}}
:root[data-theme="dark"] .vz{
 --vz-s1:#3987e5;--vz-s2:#d95926;--vz-s3:#199e70;--vz-s4:#c98500;
 --vz-grid:#2c2c2a;--vz-axis:#383835;--vz-ink2:#c3c2b7;--vz-surface:#1a1a19;
 --vz-good:#0ca30c;--vz-bad:#e66767}
"""

_BASE_CSS = """
.cesft{--bg:#f9f9f7;--card:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--mut:#898781;
 --line:#e1e0d9;--blue:#2a78d6;--good:#006300;--bad:#d03b3b;--warn:#8a5a00;
 --pend:#898781;--chipbg:#f2f1ec;
 max-width:1080px;margin:0 auto;padding:24px 16px;
 font:14px/1.6 system-ui,-apple-system,"Segoe UI",sans-serif;
 color:var(--ink);background:var(--bg);box-sizing:border-box}
@media (prefers-color-scheme: dark){:root:where(:not([data-theme="light"])) .cesft{
 --bg:#0d0d0d;--card:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--mut:#898781;--line:#2c2c2a;
 --blue:#3987e5;--good:#0ca30c;--bad:#e66767;--warn:#c98500;--pend:#898781;--chipbg:#232320}}
:root[data-theme="dark"] .cesft{--bg:#0d0d0d;--card:#1a1a19;--ink:#fff;--ink2:#c3c2b7;
 --mut:#898781;--line:#2c2c2a;--blue:#3987e5;--good:#0ca30c;--bad:#e66767;--warn:#c98500;
 --pend:#898781;--chipbg:#232320}
:root[data-theme="light"] .cesft{--bg:#f9f9f7;--card:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;
 --mut:#898781;--line:#e1e0d9;--blue:#2a78d6;--good:#006300;--bad:#d03b3b;--warn:#8a5a00;
 --pend:#898781;--chipbg:#f2f1ec}
.cesft *{box-sizing:border-box}
.cesft h1{font-size:20px;margin:0 0 3px;letter-spacing:-.01em}
.cesft h2{font-size:15px;margin:26px 0 9px;padding-bottom:5px;border-bottom:1px solid var(--line)}
.cesft h3{font-size:12.5px;margin:14px 0 5px;color:var(--ink2)}
.cesft .sub{color:var(--mut);font-size:12.5px;margin-bottom:6px}
.cesft .card{background:var(--card);border:1px solid var(--line);border-radius:11px;padding:14px 16px}
.cesft .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(155px,1fr));gap:10px;margin:12px 0}
.cesft .kpi .l{font-size:11px;color:var(--mut);font-weight:650}
.cesft .kpi .v{font-size:21px;font-weight:800;font-variant-numeric:tabular-nums}
.cesft .kpi .s{font-size:11.5px;color:var(--ink2)}
.cesft .scroll{overflow-x:auto}
.cesft table{border-collapse:collapse;width:100%;font-size:12.5px}
.cesft th,.cesft td{padding:6px 10px;border-bottom:1px solid var(--line);text-align:right;
 font-variant-numeric:tabular-nums;white-space:nowrap}
.cesft th{font-size:10.5px;color:var(--mut);text-transform:uppercase;letter-spacing:.04em}
.cesft td:first-child,.cesft th:first-child{text-align:left}
.cesft tr:last-child td{border-bottom:none}
.cesft .ok{color:var(--good);font-weight:700}.cesft .bad{color:var(--bad);font-weight:700}
.cesft .mut{color:var(--mut)}.cesft .warn{color:var(--warn)}
.cesft .num{font-variant-numeric:tabular-nums;font-family:ui-monospace,Menlo,monospace}
.cesft .ci{color:var(--mut);font-size:11px}
/* gate chips */
.cesft .chips{display:flex;flex-wrap:wrap;gap:10px}
.cesft .chip{flex:1 1 190px;min-width:180px;border:1px solid var(--line);border-radius:10px;
 padding:10px 12px;background:var(--card)}
.cesft .chip .ver{font-size:11px;font-weight:800;letter-spacing:.05em;display:inline-flex;
 align-items:center;gap:5px}
.cesft .chip.pass{border-color:var(--good)}.cesft .chip.pass .ver{color:var(--good)}
.cesft .chip.fail{border-color:var(--bad)}.cesft .chip.fail .ver{color:var(--bad)}
.cesft .chip.pend .ver{color:var(--pend)}
.cesft .chip .nm{font-size:12.5px;font-weight:700;margin:3px 0 1px}
.cesft .chip .dsc{font-size:11px;color:var(--mut);line-height:1.35}
.cesft .chip .val{font-size:11.5px;margin-top:4px;font-variant-numeric:tabular-nums}
/* stage bars */
.cesft .stage{display:grid;grid-template-columns:14px 1fr 90px 120px;gap:9px;align-items:center;
 padding:6px 0;border-bottom:1px solid var(--line);font-size:12.5px}
.cesft .stage:last-child{border-bottom:none}
.cesft .dot{width:9px;height:9px;border-radius:50%;background:var(--line)}
.cesft .st-done .dot{background:var(--good)}.cesft .st-running .dot{background:var(--blue)}
.cesft .st-failed .dot{background:var(--bad)}
.cesft .bar{height:6px;background:var(--line);border-radius:4px;overflow:hidden}
.cesft .bar i{display:block;height:100%;background:var(--blue)}
.cesft .st-d{font-size:11px;color:var(--ink2);text-align:right}
.cesft .pending-note{color:var(--mut);font-size:12px;font-style:italic;padding:6px 2px}
/* probe explorer */
.cesft .pbar{display:flex;gap:6px;flex-wrap:wrap;align-items:center;margin:6px 0}
.cesft .pbtn{font:11.5px system-ui,sans-serif;padding:3px 10px;border-radius:999px;cursor:pointer;
 background:none;border:1px solid var(--line);color:var(--ink2)}
.cesft .pbtn.on{border-color:var(--blue);color:var(--blue);font-weight:700}
.cesft .pgt{font-size:12.5px;font-weight:650;margin:4px 0 8px}
.cesft .ptbl{width:100%;border-collapse:collapse;font-size:12px}
.cesft .ptbl th,.cesft .ptbl td{text-align:left;padding:5px 9px;border-bottom:1px solid var(--line);
 vertical-align:top;white-space:normal}
.cesft .ptbl td.n{white-space:nowrap;font-variant-numeric:tabular-nums}
.cesft .good{color:var(--good);font-weight:700;white-space:nowrap}
.cesft .bad2{color:var(--bad);font-weight:700;white-space:nowrap}
.cesft .chg{color:var(--warn);font-weight:700}
.cesft .belief{color:var(--ink2);min-width:150px;max-width:260px}
.cesft .reason{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:11px;color:var(--ink2);
 min-width:220px;max-width:440px;line-height:1.45}
.cesft .foot{margin-top:22px;color:var(--mut);font-size:11.5px}
@media(max-width:640px){.cesft .stage{grid-template-columns:14px 1fr 70px}.cesft .st-d{display:none}}
"""


def render_html(run_dir, now_str: str | None = None) -> str:
    run_dir = Path(run_dir)
    data = collect(run_dir)
    if now_str is None:
        mt = _newest_mtime(run_dir)
        now_str = (datetime.datetime.fromtimestamp(mt).strftime("%Y-%m-%d %H:%M:%S")
                   if mt else "(unknown)")
    data["now"] = now_str

    blob = json.dumps(data, ensure_ascii=False)
    # <script> 조기 종료 방지 (JSON 안 임의 텍스트에 "</script>"가 있어도 안전)
    blob = blob.replace("</", "<\\/")

    css = _BASE_CSS + _VZ_STYLE + CHART_CSS
    page = _PAGE_TMPL
    page = page.replace("__CSS__", css)
    page = page.replace("__BLOB__", blob)
    page = page.replace("__CHART_JS__", CHART_JS)
    page = page.replace("__APP_JS__", _APP_JS)
    return page


# APP_JS: 인라인 DATA(blob)로 헤더/게이트/조합/곡선/평가/프론티어/정성탐색기 렌더
_APP_JS = r"""
(function(){
const D = window.__CESFT_DATA__;
const $ = id => document.getElementById(id);
const esc = s => (s==null?"":String(s)).replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
const f = (x,d=3) => (x==null||isNaN(x))?"—":(+x).toFixed(d);
const pct = x => x==null?"—":(x*100).toFixed(1)+"%";
function ci(o){ if(!o) return ""; const p=o.point??o[0], lo=o.lo??o[1], hi=o.hi??o[2];
  if(p==null) return ""; return "Δ "+f(p)+" · CI["+f(lo)+", "+f(hi)+"]"; }

// ---- header --------------------------------------------------------------
$("cesft-now").textContent = "run: "+D.run_path+" · 생성 "+D.now;
(function(){
  const done=D.done_n, tot=D.total_stages||0;
  const cur=D.current;
  const kp=[
    ["스테이지 진행", (tot?done+"/"+tot:"—"), D.chain_done?"체인 완료":"marker 기준"],
    ["현재 스테이지", cur?(cur.title||cur.id):(D.chain_done?"완료":"대기/중단"),
      cur&&cur.pct!=null?(cur.done??"?")+"/"+(cur.total??"?")+" · "+(cur.pct).toFixed(0)+"%":""],
    ["학습 arm", String(Object.keys(D.curves||{}).length), "train_log 수집"],
    ["평가 arm", String(Object.keys(D.evals||{}).length), "battery 요약"],
  ];
  if(D.gpu){ const g=Array.isArray(D.gpu)?D.gpu[0]:D.gpu;
    kp.push(["GPU", (g&&g.util!=null?g.util+"%":"—"),
      g&&g.name?g.name:(g&&g.mem_used_mb!=null?(g.mem_used_mb/1024|0)+"GB":"status")]); }
  $("cesft-kpis").innerHTML = kp.map(([l,v,s])=>
    '<div class="card kpi"><div class="l">'+esc(l)+'</div><div class="v" style="font-size:'+
    (String(v).length>8?"15px":"21px")+'">'+esc(v)+'</div><div class="s">'+esc(s)+'</div></div>').join("");
})();
(function(){
  const st=D.stages||[];
  if(!st.length){$("cesft-stages").innerHTML='<div class="pending-note">chain.json 없음 — 스테이지 정보 PENDING</div>';return;}
  const lab={done:"완료",running:"진행 중",failed:"실패",pending:"대기"};
  $("cesft-stages").innerHTML = st.map(s=>{
    const p = s.state==="done"?100:(s.pct||0);
    const r = s.state==="running"&&s.eta_sec!=null?Math.round(s.eta_sec/60)+"m 남음":
              s.state==="done"&&s.elapsed_sec?Math.round(s.elapsed_sec/60)+"m":"";
    return '<div class="stage st-'+s.state+'"><span class="dot"></span><div>'+esc(s.title||s.id)+
      '</div><div class="bar"><i style="width:'+p+'%"></i></div><div class="st-d">'+
      (lab[s.state]||s.state)+(r?" · "+r:"")+'</div></div>';
  }).join("");
})();

// ---- gate scoreboard -----------------------------------------------------
(function(){
  const paired=D.paired||[], harden=D.harden||{};
  function byGate(g){ return paired.find(p=>p.gate===g); }
  // 성립부등식: WM-candidate-CE > candidate-free-CE (G-DELTA 계열)
  function inequality(){
    return paired.find(p=>p.gate==="G-DELTA" &&
      /cand/i.test(p.arm_a||"") && /free/i.test(p.arm_b||"")) ||
      paired.find(p=>p.gate==="G-DELTA");
  }
  // harden 대표 arm (base 아닌 첫 arm, 없으면 아무거나)
  const hkeys=Object.keys(harden);
  const hmain = hkeys.find(k=>k!=="base") || hkeys[0];
  const H = hmain?harden[hmain]:null;
  function chipPaired(name,dsc,p){
    if(!p) return {name,dsc,ver:"PENDING",cls:"pend",val:"소스 파일 없음"};
    if(p.error) return {name,dsc,ver:"PENDING",cls:"pend",val:esc(p.error)};
    const cls=p.pass?"pass":"fail", ver=p.pass?"PASS":"FAIL";
    const mark=p.pass?"✓":"✗";
    return {name,dsc,ver:mark+" "+ver,cls,val:ci(p.delta)+
      (p.point_a!=null?" · "+f(p.point_a)+" vs "+f(p.point_b):"")};
  }
  function chipHarden(name,dsc,gateflag,ciobj){
    if(!H) return {name,dsc,ver:"PENDING",cls:"pend",val:"harden_s3 없음"};
    const pass = !!gateflag;
    return {name,dsc,ver:(pass?"✓ PASS":"✗ FAIL"),cls:(pass?"pass":"fail"),
      val:(ciobj?ci(ciobj):"")+" · arm="+esc(hmain)};
  }
  const chips=[
    chipPaired("G-ACC1","SelAcc(θ_CE) > WM-top1",byGate("G-ACC1")),
    chipPaired("성립부등식","WM·candidate-CE > candidate-free-CE",inequality()),
    chipHarden("G-CC1","belief causal_sensitivity ↑ vs base",
      H&&H.gate_S3a_causal_real, H&&H.causal_sensitivity_ci&&H.causal_sensitivity_ci.belief),
    chipHarden("G-CC3","belief-only utility U_g · CI-low > 0",
      H&&(H.gate_S3b_utility_real||(H.utility_belief_only_ci&&H.utility_belief_only_ci.lo>0)),
      H&&(H.utility_belief_only_ci||H.utility_own_minus_swapboth_ci)),
    chipPaired("G-NH","non-harm: SFT가 SelAcc/GADR 훼손 안 함",byGate("G-NH")),
  ];
  $("cesft-gates").innerHTML = chips.map(c=>
    '<div class="chip '+c.cls+'"><div class="ver">'+esc(c.ver)+'</div>'+
    '<div class="nm">'+esc(c.name)+'</div><div class="dsc">'+esc(c.dsc)+'</div>'+
    '<div class="val">'+c.val+'</div></div>').join("");
})();

// ---- combination methods -------------------------------------------------
(function(){
  const host=$("cesft-combo"); host.innerHTML="";
  const evals=D.evals||{}, harden=D.harden||{}, paired=D.paired||[];
  const cs=a=>{const h=harden[a]; if(!h||!h.causal_sensitivity_ci)return null;
    return (h.causal_sensitivity_ci.both||{}).point ?? (h.causal_sensitivity_ci.belief||{}).point;};
  const ug=a=>{const h=harden[a]; if(!h)return null;
    return (h.utility_belief_only_ci||{}).point ?? (h.utility_own_minus_swapboth_ci||{}).point;};
  // r-sweep arms: sft_r<number>
  const rArms=Object.keys(evals).filter(a=>/(^|_)sft_r\d+/i.test(a)||/^r\d+_/i.test(a)&&/sft/i.test(a));
  const nh=paired.find(p=>p.gate==="G-NH"&&p.pass);
  const selArm=nh?nh.arm_a:null;
  let html="";
  html+='<h3>① CE-replay r-sweep</h3>';
  if(rArms.length){
    html+='<div class="scroll"><table><tr><th>arm</th><th>SelAcc</th><th>GADR</th>'+
      '<th>causal_sens</th><th>U_g</th><th>G-NH 선택</th></tr>';
    rArms.forEach(a=>{const v=evals[a];
      html+='<tr><td>'+esc(a)+'</td><td>'+f(v.acc)+'</td><td>'+f(v.G2_correction)+
        '</td><td>'+f(cs(a))+'</td><td>'+f(ug(a))+'</td><td>'+
        (a===selArm?'<span class="ok">✓ 선택</span>':'<span class="mut">—</span>')+'</td></tr>';});
    html+='</table></div>';
  } else html+='<div class="pending-note">sft_r* arm 평가 없음 — r-sweep PENDING</div>';

  html+='<h3>② WiSE-FT frontier</h3>';
  const fr=D.frontier;
  if(Array.isArray(fr)&&fr.length){
    html+='<div class="scroll"><table><tr><th>α</th><th>SelAcc</th><th>GADR</th><th>causal_sens</th></tr>';
    fr.forEach(p=>html+='<tr><td>'+f(p.alpha,2)+'</td><td>'+f(p.SelAcc)+'</td><td>'+
      f(p.GADR)+'</td><td>'+f(p.causal_sensitivity)+'</td></tr>');
    html+='</table></div>';
  } else html+='<div class="pending-note">eval/wise_ft_frontier.json 없음 — WiSE-FT PENDING</div>';

  html+='<h3>③ Appendix-A 3-stage (T-ACC)</h3>';
  const tacc=paired.find(p=>p.gate==="T-ACC"||/T-?ACC/i.test(p._file||""));
  const appArms=["B0","C-stack","C-ctrl"].filter(a=>evals[a]);
  if(appArms.length||tacc){
    if(appArms.length){
      html+='<div class="scroll"><table><tr><th>arm</th><th>SelAcc</th><th>GADR</th></tr>';
      appArms.forEach(a=>{const v=evals[a];
        html+='<tr><td>'+esc(a)+'</td><td>'+f(v.acc)+'</td><td>'+f(v.G2_correction)+'</td></tr>';});
      html+='</table></div>';
    }
    if(tacc) html+='<div class="val" style="margin-top:6px">T-ACC: '+
      (tacc.pass?'<span class="ok">✓ PASS</span>':'<span class="bad">✗ FAIL</span>')+' · '+ci(tacc.delta)+'</div>';
  } else html+='<div class="pending-note">B0 / C-stack / C-ctrl arm 없음 — Appendix-A PENDING</div>';
  host.innerHTML=html;
})();

// ---- per-step training curves (VZ.lineChart) -----------------------------
(function(){
  const host=$("cesft-curves"); host.innerHTML="";
  const curves=D.curves||{};
  const arms=Object.keys(curves);
  if(!arms.length){host.innerHTML='<div class="pending-note">train_log.jsonl 없음 — 학습 곡선 PENDING</div>';return;}
  const LAB={loss:"loss",reasoning:"reasoning",task_belief:"belief",action:"action",sel_ce:"sel_ce"};
  const COL={loss:"var(--vz-mut)",reasoning:"var(--vz-s1)",task_belief:"var(--vz-s2)",
             action:"var(--vz-s3)",sel_ce:"var(--vz-s4)"};
  arms.forEach(arm=>{
    const c=curves[arm];
    const wrap=document.createElement("div");wrap.className="vz";host.appendChild(wrap);
    const series=(c.keys||[]).map(k=>({name:LAB[k]||k,color:COL[k]||"var(--vz-s1)",
      points:c.points.filter(p=>p[k]!=null&&p.x!=null).map(p=>[p.x,p[k]])}))
      .filter(s=>s.points.length>=2);
    if(!series.length){return;}
    VZ.lineChart(wrap,{title:arm+" — loss + field CE (EMA)",
      sub:"↓ 낮을수록 projected trace 재현 · seen 기준 · ≤160pt 다운샘플",
      xLabel:"seen",series:series});
    // final metrics
    const fm=c.final||{};
    const fkeys=Object.keys(fm);
    if(fkeys.length){
      const t=document.createElement("div");t.className="scroll";
      t.innerHTML='<table><tr><th>arm final</th>'+fkeys.map(k=>'<th>'+esc(LAB[k]||k)+'</th>').join("")+
        '<th>step</th></tr><tr><td>'+esc(arm)+'</td>'+fkeys.map(k=>'<td>'+f(fm[k])+'</td>').join("")+
        '<td>'+(c.last_step??"—")+'</td></tr></table>';
      wrap.appendChild(t);
    }
  });
})();

// ---- eval metric table ---------------------------------------------------
(function(){
  const host=$("cesft-eval"); const evals=D.evals||{}, harden=D.harden||{}, paired=D.paired||[];
  const arms=Object.keys(evals);
  if(!arms.length){host.innerHTML='<div class="pending-note">eval/*.json (battery 요약) 없음 — PENDING</div>';return;}
  const cs=a=>{const h=harden[a]; if(!h||!h.causal_sensitivity_ci)return null;
    return (h.causal_sensitivity_ci.both||{}).point;};
  const ug=a=>{const h=harden[a]; if(!h)return null;
    return (h.utility_belief_only_ci||{}).point ?? (h.utility_own_minus_swapboth_ci||{}).point;};
  const paccByArm=a=>paired.find(p=>p.arm_a===a&&p.gate==="G-ACC1");
  const cols=["arm","SelAcc(acc)","GADR(G2)","G1-ret","L0/WMtop1","coverage","malformed","causal(both)","U_g"];
  let html='<div class="scroll"><table><tr>'+cols.map(c=>'<th>'+esc(c)+'</th>').join("")+'</tr>';
  arms.forEach(a=>{const v=evals[a];
    const cov=v.pool_coverage??v.coverage_at_k;
    const pa=paccByArm(a);
    const accCell=f(v.acc)+(pa?'<div class="ci">'+ci(pa.delta)+'</div>':'');
    html+='<tr><td>'+esc(a)+'</td>'+
      '<td class="'+(v.beats_L0?'ok':'')+'">'+accCell+'</td>'+
      '<td>'+f(v.G2_correction)+'</td><td>'+f(v.G1_retention)+'</td>'+
      '<td>'+f(v.L0_wm_top1)+'</td><td>'+f(cov)+'</td><td>'+f(v.malformed_rate)+'</td>'+
      '<td>'+f(cs(a))+'</td><td>'+f(ug(a))+'</td></tr>';
  });
  html+='</table></div>';
  host.innerHTML=html;
})();

// ---- WiSE-FT frontier scatter (custom SVG) -------------------------------
(function(){
  const host=$("cesft-frontier"); const fr=D.frontier;
  if(!Array.isArray(fr)||!fr.length){
    host.innerHTML='<div class="pending-note">eval/wise_ft_frontier.json 없음 — 프론티어 PENDING</div>';return;}
  const wrap=document.createElement("div");wrap.className="vz";host.appendChild(wrap);
  const pts=fr.filter(p=>p.SelAcc!=null&&p.causal_sensitivity!=null)
    .map(p=>({a:p.alpha,x:+p.SelAcc,y:+p.causal_sensitivity})).sort((u,v)=>u.a-v.a);
  if(pts.length<1){host.innerHTML='<div class="pending-note">프론티어 좌표 부족</div>';return;}
  const NS="http://www.w3.org/2000/svg", W=640,H=300,M={l:52,t:14,r:20,b:36};
  const xs=pts.map(p=>p.x),ys=pts.map(p=>p.y);
  let x0=Math.min(...xs),x1=Math.max(...xs),y0=Math.min(...ys),y1=Math.max(...ys);
  const px=(x1-x0||1)*.1,py=(y1-y0||1)*.1;x0-=px;x1+=px;y0-=py;y1+=py;
  const X=x=>M.l+(x-x0)/((x1-x0)||1)*(W-M.l-M.r);
  const Y=y=>H-M.b-(y-y0)/((y1-y0)||1)*(H-M.t-M.b);
  const el=(n,at)=>{const e=document.createElementNS(NS,n);for(const k in at)e.setAttribute(k,at[k]);return e;};
  const svg=el("svg",{viewBox:"0 0 "+W+" "+H,width:"100%",height:H,role:"img",
    "aria-label":"WiSE-FT SelAcc vs causal_sensitivity 프론티어"});
  function ticks(a,b,n){const s=(b-a)/n,m=Math.pow(10,Math.floor(Math.log10(s||1)));
    const st=[1,2,2.5,5,10].map(z=>z*m).find(z=>(b-a)/z<=n+.5)||s;const t=[];
    for(let v=Math.ceil(a/st)*st;v<=b+1e-9;v+=st)t.push(+v.toFixed(6));return t;}
  ticks(y0,y1,4).forEach(t=>{svg.appendChild(el("line",{x1:M.l,x2:W-M.r,y1:Y(t),y2:Y(t),
    stroke:"var(--vz-grid)","stroke-width":1}));
    const tx=el("text",{x:M.l-6,y:Y(t)+3.5,"text-anchor":"end",class:"vz-tick"});tx.textContent=t.toFixed(2);svg.appendChild(tx);});
  ticks(x0,x1,4).forEach(t=>{const tx=el("text",{x:X(t),y:H-M.b+14,"text-anchor":"middle",class:"vz-tick"});
    tx.textContent=t.toFixed(2);svg.appendChild(tx);});
  svg.appendChild(el("line",{x1:M.l,x2:W-M.r,y1:H-M.b,y2:H-M.b,stroke:"var(--vz-axis)","stroke-width":1}));
  const d=pts.map((p,i)=>(i?"L":"M")+X(p.x).toFixed(1)+" "+Y(p.y).toFixed(1)).join("");
  svg.appendChild(el("path",{d,fill:"none",stroke:"var(--vz-s1)","stroke-width":2,"stroke-linejoin":"round"}));
  pts.forEach(p=>{svg.appendChild(el("circle",{cx:X(p.x),cy:Y(p.y),r:6,fill:"var(--vz-surface)"}));
    svg.appendChild(el("circle",{cx:X(p.x),cy:Y(p.y),r:4,fill:"var(--vz-s1)"}));
    const tl=el("text",{x:X(p.x),y:Y(p.y)-9,"text-anchor":"middle",class:"vz-endlab"});
    tl.textContent="α="+(p.a!=null?(+p.a).toFixed(2):"?");svg.appendChild(tl);});
  const xl=el("text",{x:(M.l+W-M.r)/2,y:H-4,"text-anchor":"middle",class:"vz-tick"});xl.textContent="SelAcc →";svg.appendChild(xl);
  const yl=el("text",{x:14,y:(M.t+H-M.b)/2,"text-anchor":"middle",class:"vz-tick",
    transform:"rotate(-90 14 "+((M.t+H-M.b)/2)+")"});yl.textContent="causal_sensitivity ↑";svg.appendChild(yl);
  const plot=document.createElement("div");plot.className="scroll";plot.appendChild(svg);wrap.appendChild(plot);
})();

// ---- qualitative r/g/a explorer (interactive) ----------------------------
(function(){
  const host=$("cesft-probe");
  const probes=D.probes||{}, ivs=D.interventions||{};
  const arms=Object.keys(probes);
  if(!arms.length){host.innerHTML='<div class="pending-note">probe/*.jsonl 없음 — 정성 탐색기 PENDING</div>';return;}
  // 고정 8샘플 = 첫 arm 최신 entry의 samples
  const arm0=arms[0];
  const ent0=probes[arm0].entries;
  const base=ent0[ent0.length-1].samples||[];
  const st={si:0, arm:arm0};
  host.innerHTML='<div class="pbar" id="pb-sample"></div><div class="pgt" id="pb-gt"></div>'+
    '<h3>(i) 학습 step에 따른 변화 — action ✓/✗, ↺=직전 step 대비 action 변경</h3>'+
    '<div class="pbar" id="pb-arm"></div><div class="scroll"><table class="ptbl" id="pb-step"></table></div>'+
    '<h3>(ii) belief 개입 대비 — own vs belief-swap</h3><div id="pb-iv"></div>';
  function sampleBtns(){
    $("pb-sample").innerHTML='<span class="sub">샘플</span>';
    base.forEach((s,i)=>{const b=document.createElement("button");
      b.className="pbtn"+(i===st.si?" on":"");b.textContent=(s.bucket||"?")+"·"+(s.gt||"?");
      b.title=s.sample_id||"";b.onclick=()=>{st.si=i;render();};$("pb-sample").appendChild(b);});
  }
  function armBtns(){
    $("pb-arm").innerHTML='<span class="sub">타임라인 arm</span>';
    arms.forEach(a=>{const b=document.createElement("button");
      b.className="pbtn"+(a===st.arm?" on":"");b.textContent=a;
      b.onclick=()=>{st.arm=a;render();};$("pb-arm").appendChild(b);});
  }
  function render(){
    sampleBtns();armBtns();
    const s0=base[st.si]||{};
    $("pb-gt").innerHTML="GT: <b>"+esc(s0.gt||"?")+"</b> · bucket "+esc(s0.bucket||"?")+
      " · sample_id <span class='num'>"+esc((s0.sample_id||"").slice(0,32))+"</span>"+
      "<div class='sub'>G1=WM top1이 정답 · G2=정답이 후보에 있으나 top1 아님 · other=support 밖</div>";
    // (i) step timeline for selected arm
    const ents=probes[st.arm].entries;
    let rows='<tr><th>step</th><th>acc</th><th>action</th><th>task_belief</th><th>reasoning_head</th></tr>';
    let prev=null;
    ents.forEach(e=>{const s=(e.samples||[])[st.si]; if(!s)return;
      const chg=(prev!==null&&s.action!==prev);
      rows+='<tr><td class="n">'+e.step+'</td><td class="n">'+f(e.probe_acc,2)+'</td>'+
        '<td class="'+(s.correct?"good":"bad2")+'">'+esc(s.action||"∅")+(s.correct?" ✓":" ✗")+
        (chg?' <span class="chg">↺</span>':"")+'</td>'+
        '<td class="belief">'+esc(s.task_belief||"—")+'</td>'+
        '<td class="reason">'+esc(s.reasoning_head||"—")+'</td></tr>';
      prev=s.action;});
    $("pb-step").innerHTML=rows;
    // (ii) intervention own vs swap
    const sid=s0.sample_id;
    let ivHtml="";
    Object.keys(ivs).forEach(a=>{const rec=ivs[a][sid]; if(!rec)return;
      const own=rec.own||{}, sw=rec.swap||rec.swap_both||rec.belief_swap||{};
      const ob=rec.own_task_belief??own.task_belief, sb=rec.swap_task_belief??sw.task_belief;
      const oa=rec.own_action??own.action, sa=rec.swap_action??sw.action;
      const op=rec.own_p_gt??own.p_gt, sp=rec.swap_p_gt??sw.p_gt;
      ivHtml+='<div class="scroll" style="margin-bottom:8px"><b>'+esc(a)+'</b>'+
        '<table class="ptbl"><tr><th></th><th>task_belief</th><th>action</th><th>p_gt</th></tr>'+
        '<tr><td class="n">own</td><td class="belief">'+esc(ob||"—")+'</td><td>'+esc(oa||"—")+'</td><td class="n">'+f(op)+'</td></tr>'+
        '<tr><td class="n">belief-swap</td><td class="belief">'+esc(sb||"—")+'</td><td>'+esc(sa||"—")+'</td><td class="n">'+f(sp)+'</td></tr>'+
        '</table></div>';});
    $("pb-iv").innerHTML=ivHtml||'<div class="pending-note">개입 per-sample 덤프(harden_s3.records) 없음 — (ii) PENDING</div>';
  }
  render();
})();
})();
"""

_PAGE_TMPL = r"""<style>
__CSS__
</style>
<div class="cesft">
  <h1>EGO Step-2 — candidate-CE ↔ projected-SFT 조합</h1>
  <div class="sub" id="cesft-now">loading…</div>
  <div class="grid" id="cesft-kpis"></div>
  <div class="card" id="cesft-stages"></div>

  <h2>게이트 스코어보드</h2>
  <div class="chips" id="cesft-gates"></div>

  <h2>조합 방법 패널</h2>
  <div class="card" id="cesft-combo"></div>

  <h2>정량 — arm별 학습 곡선 (per-step)</h2>
  <div id="cesft-curves"></div>

  <h2>평가 지표 — 전 arm</h2>
  <div class="card" id="cesft-eval"></div>

  <h2>WiSE-FT — SelAcc ↔ causal_sensitivity 프론티어</h2>
  <div class="card" id="cesft-frontier"></div>

  <h2>정성 r/g/a 탐색기</h2>
  <div class="card" id="cesft-probe"></div>

  <div class="foot">자기완결 정적 아티팩트 · 외부 네트워크 0 · 데이터 인라인 ·
  성립부등식 = WM·candidate-CE &gt; candidate-free-CE (Appendix-A T-ACC) ·
  성공 판정은 belief 개입(G-CC1/G-CC3)과 non-harm(G-NH)을 함께 본다.</div>
</div>
<script>window.__CESFT_DATA__ = __BLOB__;</script>
<script>
__CHART_JS__
</script>
<script>
__APP_JS__
</script>
"""


def main():
    ap = argparse.ArgumentParser(description="EGO Step-2 candidate-CE↔projected-SFT 조합 아티팩트 베이커")
    ap.add_argument("--run", default=os.environ.get("RETRO3_RUNS", "runs/cesft_v2"),
                    help="실험 run 디렉터리 (default runs/cesft_v2 · env RETRO3_RUNS)")
    ap.add_argument("--out", required=True, help="출력 HTML 경로")
    ap.add_argument("--now", default=None, help="생성 타임스탬프 표시 문자열 (생략 시 최신 파일 mtime)")
    args = ap.parse_args()
    html = render_html(args.run, args.now)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"baked: {out}  ({len(html.encode('utf-8'))} bytes, run={args.run})")


if __name__ == "__main__":
    main()
