#!/usr/bin/env python3
"""Retrospection 실시간 대시보드 — stdlib http.server (외부 의존 0), port 7867.

보여주는 것: 체인 스테이지 진행/ETA(측정 기반), 실측 지표(coverage·gate·pair·
train margin·eval·개입③), GPU 상태, supervisor 상태, 로그 tail, 전체 남은 시간.

사용:  python3 tools/retro3_dashboard.py [--port 7867] [--runs runs/retro3]
접속:  http://<host>:7867   (ssh 터널: ssh -L 7867:localhost:7867 <host>)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

RUNS = Path("runs/retro3")


def _read_json(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def gpu_info():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout
        gpus = []
        for line in out.strip().splitlines():
            i, name, mu, mt, util, temp = [x.strip() for x in line.split(",")]
            gpus.append({"index": int(i), "name": name, "mem_used_mb": int(mu),
                         "mem_total_mb": int(mt), "util": int(util), "temp": int(temp)})
        return gpus
    except Exception:
        return []


OUT_DIR = Path("outputs/step2_retrospection")
# 학습 곡선 필드 (색 슬롯 고정: 1=reasoning 2=task_belief/belief 3=action)
SFT_KEYS = ("reasoning", "task_belief", "action")
DPO_KEYS = ("m_reason", "m_belief", "m_action")


def _collect_train_curves(max_pts: int = 160):
    """run별 train_log.jsonl → EMA 평활 field 곡선 (SFT: CE↓, DPO: margin↑).

    '추론 능력' 관측 소재: SFT는 projected reasoning CE 하락, DPO는 m_reason
    (chosen−rejected reasoning logp margin) 상승. 다운샘플 ≤max_pts."""
    out = {}
    if not OUT_DIR.is_dir():
        return out
    for d in sorted(OUT_DIR.iterdir()):
        f = d / "train_log.jsonl"
        if not f.is_file():
            continue
        rows = []
        try:
            with open(f, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        try:
                            r = json.loads(line)
                        except Exception:
                            continue
                        if "loss" in r and "seen" in r:
                            rows.append(r)
        except Exception:
            continue
        if len(rows) < 2:
            continue
        keys = DPO_KEYS if "m_belief" in rows[0] else SFT_KEYS
        ema, pts = {}, []
        for r in rows:
            if any(k not in r for k in keys):
                continue
            for k in keys:
                ema[k] = r[k] if k not in ema else 0.9 * ema[k] + 0.1 * r[k]
            pts.append({"x": r["seen"], **{k: round(ema[k], 4) for k in keys}})
        if len(pts) < 2:
            continue
        if len(pts) > max_pts:  # 균등 다운샘플 (끝점 보존)
            step = (len(pts) - 1) / (max_pts - 1)
            pts = [pts[round(i * step)] for i in range(max_pts)]
        out[d.name] = {"kind": "dpo" if keys is DPO_KEYS else "sft",
                       "keys": list(keys), "points": pts}
    return out


def _collect_probes():
    """run별 probe 곡선(acc) + 첫/최신 probe 전문 — '트레이스 진화' 섹션 원천."""
    out = {}
    pdir = RUNS / "probe"
    if not pdir.is_dir():
        return out
    for p in sorted(pdir.glob("*.jsonl")):
        entries = []
        try:
            with open(p, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entries.append(json.loads(line))
        except Exception:
            continue
        if not entries:
            continue
        # 정성 로그용 전체 히스토리 — 40개 초과 시 균등 샘플 (첫/마지막 보존)
        hist = entries
        if len(hist) > 40:
            step = (len(hist) - 1) / 39
            hist = [hist[round(i * step)] for i in range(40)]
        out[p.stem] = {
            "curve": [{"step": e["step"], "acc": e["probe_acc"]} for e in entries],
            "first": entries[0], "latest": entries[-1],
            "entries": hist,
        }
    return out


def collect():
    now = time.time()
    chain = _read_json(RUNS / "chain.json") or {"stages": []}
    markers = {p.name: _read_json(p) for p in (RUNS / "markers").glob("*")} if (RUNS / "markers").is_dir() else {}
    statuses = {}
    for p in (RUNS / "status").glob("*.json") if (RUNS / "status").is_dir() else []:
        s = _read_json(p)
        if s:
            statuses[s["stage"]] = s

    # 스테이지 상태 결정: done(marker) > running/failed(status) > pending
    stages, eta_total, reached_current = [], 0.0, False
    for st in chain["stages"]:
        s = {"id": st["id"], "title": st["title"], "est_sec": st["est_sec"]}
        live = None
        for key, sv in statuses.items():
            if key.startswith(st["id"]) or st["id"].startswith(key):
                live = sv
                break
        if st["marker"] in markers:
            s["state"] = "done"
            if live:
                s["elapsed"] = live.get("elapsed_sec")
                s["metrics"] = live.get("metrics", {})
        elif live and live["state"] == "running" and now - live["updated_at"] < 180:
            s["state"] = "running"
            s.update({k: live.get(k) for k in ("done", "total", "pct", "rate_per_s", "eta_sec", "elapsed_sec")})
            s["metrics"] = live.get("metrics", {})
            eta_total += live.get("eta_sec") or 0
            reached_current = True
        elif live and live["state"] == "failed":
            s["state"] = "failed"
            s["metrics"] = live.get("metrics", {})
            s["message"] = live.get("message")
            reached_current = True
        else:
            s["state"] = "stale" if (live and live["state"] == "running") else "pending"
            if s["state"] == "pending":
                eta_total += st["est_sec"]
            reached_current = reached_current or s["state"] == "stale"
        stages.append(s)

    data = {
        "ts": now,
        "stages": stages,
        "eta_total_sec": eta_total,
        "chain_done": "RETRO3_CHAIN_DONE" in markers,
        "chain_failed": markers.get("CHAIN_FAILED"),
        "chain_stuck": markers.get("CHAIN_STUCK"),
        "g3_abort": markers.get("G3_STYLE_ABORT"),
        "semantic_skipped": markers.get("S4_SEMANTIC_SKIPPED"),
        "gpu": gpu_info(),
        "coverage": _read_json(RUNS / "data" / "coverage.json"),
        "pairs": _read_json(RUNS / "data" / "pairs_report.json"),
        "context": _read_json(RUNS / "data" / "context_stats.json"),
        "eval": {p.stem: _read_json(p) for p in sorted((RUNS / "eval").glob("*.json"))
                 if not p.stem.endswith("records")} if (RUNS / "eval").is_dir() else {},
        "probes": _collect_probes(),
        "train_curves": _collect_train_curves(),
    }
    # supervisor 살아있나
    pid_f = RUNS / "chain.pid"
    data["supervisor_alive"] = False
    if pid_f.is_file():
        try:
            import os
            os.kill(int(pid_f.read_text().strip()), 0)
            data["supervisor_alive"] = True
        except Exception:
            pass
    # 로그 tail
    log = RUNS / "logs" / "chain.log"
    if log.is_file():
        with open(log, "rb") as f:
            f.seek(max(0, log.stat().st_size - 6000))
            data["log_tail"] = f.read().decode("utf-8", "replace")[-5500:]
    return data


# 공유 차트 렌더러 — 대시보드(PAGE)와 아티팩트 스냅샷(retro3_artifact_snapshot)이 함께 쓴다.
# 색은 CSS 변수(--vz-*)로만 참조 → light/dark는 페이지 쪽 CSS가 결정.
# 시리즈 슬롯 고정(validate_palette 통과 순서): 1 blue · 2 orange · 3 aqua · 4 yellow.
CHART_JS = r"""
const VZ=(()=>{
const NS="http://www.w3.org/2000/svg";
const SLOT=["var(--vz-s1)","var(--vz-s2)","var(--vz-s3)","var(--vz-s4)"];
const FIELD_LABEL={reasoning:"reasoning",task_belief:"belief",action:"action",
 m_reason:"reasoning",m_belief:"belief",m_action:"action"};
const RUN_SLOT={r1_sft:0,dpo_d1:1,dpo_d1_fix:1,dpo_d1_g3abort:1,dpo_all:2,dpo_sem:3};
const fmt=v=>Math.abs(v)>=100?v.toFixed(0):Math.abs(v)>=10?v.toFixed(1):v.toFixed(2);
function svgEl(n,at,p){const e=document.createElementNS(NS,n);for(const k in at||{})e.setAttribute(k,at[k]);if(p)p.appendChild(e);return e}
function htmlEl(n,cls,p,txt){const e=document.createElement(n);if(cls)e.className=cls;if(txt!=null)e.textContent=txt;if(p)p.appendChild(e);return e}
function niceTicks(a,b,n){const span=b-a,raw=span/n,mag=Math.pow(10,Math.floor(Math.log10(raw)));
 const step=[1,2,2.5,5,10].map(m=>m*mag).find(s=>span/s<=n+0.5)||raw;
 const t=[];for(let v=Math.ceil(a/step)*step;v<=b+1e-9;v+=step)t.push(+v.toFixed(6));return t}
function lineChart(host,cfg){
 const card=htmlEl("div","vz-chart",host);
 htmlEl("div","vz-title",card,cfg.title);
 if(cfg.sub)htmlEl("div","vz-sub",card,cfg.sub);
 const leg=htmlEl("div","vz-legend",card);
 cfg.series.forEach(s=>{const k=htmlEl("span","vz-key",leg);
  const sw=htmlEl("i","",k);sw.style.background=s.color;htmlEl("span","",k,s.name)});
 const plot=htmlEl("div","vz-plot",card);
 const W=Math.max(300,plot.clientWidth||host.clientWidth||460),H=cfg.height||185;
 const M={l:44,t:8,r:12,b:22};
 const pts=cfg.series.flatMap(s=>s.points);
 if(pts.length<2){htmlEl("div","vz-sub",plot,"데이터 수집 전");return}
 const xs=pts.map(p=>p[0]),ys=pts.map(p=>p[1]);
 const x0=Math.min(...xs),x1=Math.max(...xs);
 let y0=Math.min(...ys),y1=Math.max(...ys);
 if(cfg.yDomain){y0=cfg.yDomain[0];y1=cfg.yDomain[1]}
 else{if(cfg.includeZero){y0=Math.min(0,y0);y1=Math.max(0,y1)}
  if(y0===y1){y0-=1;y1+=1}const pad=(y1-y0)*.08;y0-=pad;y1+=pad}
 const X=x=>M.l+(x-x0)/((x1-x0)||1)*(W-M.l-M.r);
 const Y=y=>H-M.b-(y-y0)/((y1-y0)||1)*(H-M.t-M.b);
 const svg=svgEl("svg",{viewBox:`0 0 ${W} ${H}`,width:"100%",height:H,role:"img","aria-label":cfg.title},plot);
 for(const t of niceTicks(y0,y1,4)){
  svgEl("line",{x1:M.l,x2:W-M.r,y1:Y(t),y2:Y(t),stroke:"var(--vz-grid)","stroke-width":1},svg);
  svgEl("text",{x:M.l-6,y:Y(t)+3.5,"text-anchor":"end",class:"vz-tick"},svg).textContent=fmt(t)}
 if(cfg.zeroLine&&y0<0&&y1>0)
  svgEl("line",{x1:M.l,x2:W-M.r,y1:Y(0),y2:Y(0),stroke:"var(--vz-axis)","stroke-width":1},svg);
 svgEl("line",{x1:M.l,x2:W-M.r,y1:H-M.b,y2:H-M.b,stroke:"var(--vz-axis)","stroke-width":1},svg);
 [[x0,"start"],[(x0+x1)/2,"middle"],[x1,"end"]].forEach(([v,a])=>
  svgEl("text",{x:X(v),y:H-M.b+14,"text-anchor":a,class:"vz-tick"},svg).textContent=Math.round(v));
 if(cfg.xLabel)svgEl("text",{x:W-M.r,y:H-M.b+14,"text-anchor":"start",class:"vz-tick"},svg).textContent="";
 cfg.series.forEach(s=>{
  const d=s.points.map((p,i)=>(i?"L":"M")+X(p[0]).toFixed(1)+" "+Y(p[1]).toFixed(1)).join("");
  svgEl("path",{d,fill:"none",stroke:s.color,"stroke-width":2,"stroke-linejoin":"round","stroke-linecap":"round"},svg);
  if(cfg.markers)s.points.forEach(p=>{ // 2px 서피스 링 + ≥8px 마커
   svgEl("circle",{cx:X(p[0]),cy:Y(p[1]),r:6,fill:"var(--vz-surface)"},svg);
   svgEl("circle",{cx:X(p[0]),cy:Y(p[1]),r:4,fill:s.color},svg)})});
 // 끝값 직접 라벨 — 12px 이상 떨어질 때만 (겹치면 legend+tooltip으로 폴백)
 const ends=cfg.series.map(s=>({s,y:Y(s.points[s.points.length-1][1]),v:s.points[s.points.length-1][1]}))
  .sort((a,b)=>a.y-b.y);
 let ok=true;for(let i=1;i<ends.length;i++)if(ends[i].y-ends[i-1].y<12)ok=false;
 if(ok)ends.forEach(e=>svgEl("text",{x:W-M.r-2,y:e.y-5,"text-anchor":"end",class:"vz-endlab"},svg)
  .textContent=e.s.name+" "+fmt(e.v));
 // hover: 세로 크로스헤어 + 전 시리즈 툴팁 (X만 조준하면 됨)
 plot.style.position="relative";
 const tip=htmlEl("div","vz-tip",plot);tip.style.display="none";
 const cross=svgEl("line",{y1:M.t,y2:H-M.b,stroke:"var(--vz-axis)","stroke-width":1,visibility:"hidden"},svg);
 svg.addEventListener("pointerleave",()=>{tip.style.display="none";cross.setAttribute("visibility","hidden")});
 svg.addEventListener("pointermove",ev=>{
  const r=svg.getBoundingClientRect(),px=(ev.clientX-r.left)*W/r.width;
  const xv=x0+(Math.min(Math.max(px,M.l),W-M.r)-M.l)/(W-M.l-M.r)*(x1-x0);
  tip.textContent="";cross.setAttribute("visibility","visible");
  let cxs=[];
  cfg.series.forEach(s=>{let best=s.points[0];
   for(const p of s.points)if(Math.abs(p[0]-xv)<Math.abs(best[0]-xv))best=p;
   cxs.push(best[0]);
   const row=htmlEl("div","vz-tip-row",tip);
   const k=htmlEl("i","",row);k.style.background=s.color;
   htmlEl("b","",row,fmt(best[1]));htmlEl("span","",row,s.name)});
  const cx=X(cxs[0]);cross.setAttribute("x1",cx);cross.setAttribute("x2",cx);
  htmlEl("div","vz-tip-x",tip,(cfg.xLabel||"x")+" "+Math.round(cxs[0]));
  tip.style.display="block";
  const left=cx/W*r.width;
  tip.style.left=Math.min(left+10,r.width-tip.offsetWidth-4)+"px";tip.style.top="6px"});
 // 표 뷰 (tooltip은 게이트 아님 — 값은 표로도 항상 도달 가능)
 const det=htmlEl("details","vz-table",card);htmlEl("summary","",det,"표로 보기");
 const tb=htmlEl("table","",det),hr=htmlEl("tr","",tb);
 htmlEl("th","",hr,cfg.xLabel||"x");cfg.series.forEach(s=>htmlEl("th","",hr,s.name));
 const n=cfg.series[0].points.length,step=Math.max(1,Math.floor(n/12));
 for(let i=0;i<n;i+=step){const tr=htmlEl("tr","",tb);
  htmlEl("td","",tr,String(cfg.series[0].points[i][0]));
  cfg.series.forEach(s=>htmlEl("td","",tr,s.points[i]?fmt(s.points[i][1]):"—"))}
}
function renderAll(host,data){
 host.textContent="";
 const runs=Object.keys(data.curves||{});
 const order=["r1_sft","dpo_d1","dpo_all","dpo_sem"];
 const oi=x=>{const i=order.indexOf(x);return i<0?99:i};
 runs.sort((a,b)=>(oi(a)-oi(b))||a.localeCompare(b));
 for(const run of runs){
  const c=data.curves[run];
  const sft=c.kind==="sft";
  lineChart(host,{
   title:run+(sft?" — field CE (EMA)":" — margin chosen−rejected (EMA)"),
   sub:sft?"↓ 낮을수록 projected trace 적합 — reasoning 하락 = 추론 재현 학습":
        "↑ 높을수록 회고 trace 선호 — reasoning margin 상승 = 추론 개선 신호 · 0선 위=chosen 우세",
   xLabel:"seen",zeroLine:!sft,includeZero:!sft,
   series:c.keys.map((k,i)=>({name:FIELD_LABEL[k]||k,color:SLOT[i],
    points:c.points.map(p=>[p.x,p[k]])}))});
 }
 const probes=Object.entries(data.probes||{}).filter(([,v])=>v&&v.length>=2);
 if(probes.length){
  probes.sort((a,b)=>((RUN_SLOT[a[0]]??9)-(RUN_SLOT[b[0]]??9))||a[0].localeCompare(b[0]));
  lineChart(host,{
   title:"probe acc — 고정 8샘플 정답률 (행동 지표)",
   sub:"n=8 (G1 2·G2 4·기타 2, 전 run 공용) — 1칸=±0.125 · SFT 100step/DPO 50step 간격",
   xLabel:"step",markers:true,yDomain:[0,1],
   series:probes.map(([run,curve])=>({name:run,color:SLOT[RUN_SLOT[run]??0],
    points:curve.map(c=>[c.step,c.acc])}))});
 }
 if(!runs.length&&!probes.length)
  htmlEl("div","vz-sub",host,"S6 학습 시작 후 표시됩니다 — SFT: field CE · DPO: field margin · probe acc");
}
// 정성 로그 — probe 전문으로 reasoning/belief/action의 변화를 텍스트 그대로 본다.
// qual: {run: entries[]}, entries[i] = {step, probe_acc, samples:[{bucket,gt,action,correct,task_belief,reasoning_head,malformed}]}
function qualLog(host,qual){
 host.textContent="";
 const order=["r1_sft","dpo_d1","dpo_all","dpo_sem"];
 const oi=x=>{const i=order.indexOf(x);return i<0?99:i};
 const runs=Object.keys(qual||{}).filter(r=>(qual[r]||[]).length)
  .sort((a,b)=>(oi(a)-oi(b))||a.localeCompare(b));
 if(!runs.length){htmlEl("div","vz-sub",host,
  "probe 수집 전 — S6 학습이 진행되면 여기에 base→SFT→DPO 응답 전문이 쌓입니다");return}
 const st=(qualLog._sel=qualLog._sel||{});
 if(!runs.includes(st.run))st.run=runs[0];
 const sam0=qual[runs[0]][0].samples;
 if(st.si==null||st.si>=sam0.length)st.si=0;
 // 컨트롤 한 줄: 샘플 선택 (전 run 공용 고정 8샘플) + 타임라인 run 선택
 const bar=htmlEl("div","vz-qbar",host);
 htmlEl("span","vz-sub",bar,"샘플");
 sam0.forEach((s,i)=>{const b=htmlEl("button","vz-qbtn"+(i===st.si?" on":""),bar,
  s.bucket+"·"+(s.gt||"?"));b.title=s.gt||"";
  b.onclick=()=>{st.si=i;qualLog(host,qual)}});
 const s0=sam0[st.si]||{};
 htmlEl("div","vz-qgt",host,"GT: "+(s0.gt||"?")+" · bucket "+(s0.bucket||"?")+
  " (G1=WM top1이 정답 · G2=정답이 후보에 있으나 top1 아님 · other=support 밖)");
 // ① 단계 간 비교 — base(step0) + 각 run의 최신/완료 스냅샷
 htmlEl("div","vz-qh",host,"① 단계 완료 비교 — 같은 관측에 대한 응답 변화");
 const wrap1=htmlEl("div","vz-qwrap",host);
 const tbl=htmlEl("table","vz-qtbl",wrap1);
 const hr=htmlEl("tr","",tbl);
 ["체크포인트","action","task_belief","reasoning (앞 260자)"].forEach(h=>htmlEl("th","",hr,h));
 const rows=[["base @step 0",qual[runs[0]][0].samples[st.si]]];
 runs.forEach(r=>{const es=qual[r],last=es[es.length-1];
  if(!(r===runs[0]&&es.length===1))rows.push([r+" @step "+last.step,last.samples[st.si]])});
 let prev=null;
 rows.forEach(([name,s])=>{if(!s)return;const tr=htmlEl("tr","",tbl);
  htmlEl("td","vz-qname",tr,name);
  htmlEl("td",s.correct?"vz-qgood":"vz-qbad",tr,
   (s.action||"∅")+(s.correct?" ✓":" ✗")+(prev!==null&&s.action!==prev?" ↺":""));
  prev=s.action;
  htmlEl("td","vz-qbelief",tr,s.task_belief||"—");
  htmlEl("td","vz-qreason",tr,s.reasoning_head||"—")});
 // ② step 타임라인 — 선택 run에서 시간에 따른 변화 (action 바뀐 지점 ↺)
 const bar2=htmlEl("div","vz-qbar",host);
 htmlEl("span","vz-sub",bar2,"② 타임라인 run");
 runs.forEach(r=>{const b=htmlEl("button","vz-qbtn"+(r===st.run?" on":""),bar2,r);
  b.onclick=()=>{st.run=r;qualLog(host,qual)}});
 const wrap2=htmlEl("div","vz-qwrap",host);
 const t2=htmlEl("table","vz-qtbl",wrap2);
 const h2=htmlEl("tr","",t2);
 ["step","acc","action","task_belief","reasoning (앞 260자)"].forEach(h=>htmlEl("th","",h2,h));
 let pa=null;
 qual[st.run].forEach(e=>{const s=e.samples[st.si];if(!s)return;
  const tr=htmlEl("tr","",t2);
  htmlEl("td","vz-qname",tr,String(e.step));
  htmlEl("td","",tr,(e.probe_acc!=null?e.probe_acc.toFixed(2):"—"));
  htmlEl("td",s.correct?"vz-qgood":"vz-qbad",tr,
   (s.action||"∅")+(s.correct?" ✓":" ✗")+(pa!==null&&s.action!==pa?" ↺":""));
  pa=s.action;
  htmlEl("td","vz-qbelief",tr,s.task_belief||"—");
  htmlEl("td","vz-qreason",tr,s.reasoning_head||"—")});
}
return {lineChart,renderAll,qualLog}})();
"""

CHART_CSS = r"""
.vz .vz-chart{margin:0 0 18px}.vz .vz-chart:last-child{margin-bottom:0}
.vz .vz-title{font-size:12.5px;font-weight:700}
.vz .vz-sub{font-size:11px;color:var(--vz-mut);margin:1px 0 4px}
.vz .vz-legend{display:flex;gap:12px;flex-wrap:wrap;margin:2px 0 4px}
.vz .vz-key{display:inline-flex;align-items:center;gap:5px;font-size:11px;color:var(--vz-ink2)}
.vz .vz-key i{display:inline-block;width:12px;height:2px;border-radius:1px}
.vz .vz-tick{font-size:10.5px;fill:var(--vz-mut)}
.vz .vz-endlab{font-size:10.5px;fill:var(--vz-ink2)}
.vz .vz-tip{position:absolute;pointer-events:none;background:var(--vz-surface);
 border:1px solid var(--vz-grid);border-radius:7px;padding:6px 9px;font-size:11px;
 box-shadow:0 2px 8px rgba(0,0,0,.15);z-index:3}
.vz .vz-tip-row{display:flex;align-items:center;gap:6px;white-space:nowrap}
.vz .vz-tip-row i{width:10px;height:2px;border-radius:1px}
.vz .vz-tip-row b{font-variant-numeric:tabular-nums}
.vz .vz-tip-row span{color:var(--vz-mut)}
.vz .vz-tip-x{color:var(--vz-mut);margin-top:2px}
.vz .vz-table{margin-top:4px;font-size:11px}
.vz .vz-table summary{cursor:pointer;color:var(--vz-mut)}
.vz .vz-table table{margin-top:4px}
.vz .vz-qbar{display:flex;gap:6px;flex-wrap:wrap;align-items:center;margin:8px 0 6px}
.vz .vz-qbtn{font:11.5px system-ui,sans-serif;padding:3px 9px;border-radius:999px;cursor:pointer;
 background:none;border:1px solid var(--vz-grid);color:var(--vz-ink2)}
.vz .vz-qbtn.on{border-color:var(--vz-s1);color:var(--vz-s1);font-weight:700}
.vz .vz-qgt{font-size:12px;font-weight:600;margin:2px 0 6px}
.vz .vz-qh{font-size:12px;font-weight:700;margin:10px 0 4px}
.vz .vz-qwrap{overflow-x:auto}
.vz .vz-qtbl{width:100%;border-collapse:collapse;font-size:11.5px}
.vz .vz-qtbl th{font-size:10.5px;color:var(--vz-mut);text-transform:none;letter-spacing:0;
 text-align:left;padding:4px 8px;border-bottom:1px solid var(--vz-grid)}
.vz .vz-qtbl td{text-align:left;padding:5px 8px;border-bottom:1px solid var(--vz-grid);
 vertical-align:top;white-space:normal}
.vz .vz-qtbl tr:last-child td{border-bottom:none}
.vz .vz-qname{white-space:nowrap;font-weight:600}
.vz .vz-qgood{color:var(--vz-good);font-weight:700;white-space:nowrap}
.vz .vz-qbad{color:var(--vz-bad);white-space:nowrap}
.vz .vz-qbelief{min-width:150px;max-width:240px;color:var(--vz-ink2)}
.vz .vz-qreason{min-width:220px;max-width:420px;font-size:11px;color:var(--vz-ink2)}
"""

PAGE = r"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>retro3 · Retrospection 체인</title>
<style>
:root { color-scheme: dark; --bg:#0d0d0d; --card:#1a1a19; --ink:#fff; --ink2:#c3c2b7;
 --mut:#898781; --line:#2c2c2a; --blue:#3987e5; --orange:#d95926; --good:#0ca30c;
 --bad:#e66767; --warn:#c98500; }
*{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--ink);
 font:14px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif;padding:20px}
.wrap{max-width:1060px;margin:0 auto}
h1{font-size:19px;margin:0 0 2px} .sub{color:var(--mut);font-size:12px;margin-bottom:16px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:14px 0}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.kpi .v{font-size:22px;font-weight:800;font-variant-numeric:tabular-nums}
.kpi .l{font-size:11px;color:var(--mut);font-weight:600}
.kpi .s{font-size:11px;color:var(--ink2)}
.stage{display:grid;grid-template-columns:22px 1fr 90px 170px;gap:10px;align-items:center;
 padding:7px 0;border-bottom:1px solid var(--line);font-size:13px}
.stage:last-child{border-bottom:none}
.dot{width:10px;height:10px;border-radius:50%;margin:auto}
.done .dot{background:var(--good)} .running .dot{background:var(--blue);animation:p 1.2s infinite}
.failed .dot{background:var(--bad)} .pending .dot{background:var(--line)}
.stale .dot{background:var(--warn)}
@keyframes p{50%{opacity:.35}}
.bar{height:7px;background:var(--line);border-radius:4px;overflow:hidden}
.bar i{display:block;height:100%;background:var(--blue)}
.num{font-variant-numeric:tabular-nums;font-family:ui-monospace,Menlo,monospace;font-size:12px;color:var(--ink2)}
.mut{color:var(--mut)} .ok{color:var(--good)} .bad{color:var(--bad)} .warn{color:var(--warn)}
table{border-collapse:collapse;width:100%;font-size:12.5px}
th,td{padding:5px 10px;text-align:right;border-bottom:1px solid var(--line);white-space:nowrap;
 font-variant-numeric:tabular-nums}
th{color:var(--mut);font-size:11px;text-transform:uppercase;letter-spacing:.05em}
td:first-child,th:first-child{text-align:left}
pre{background:#111;border:1px solid var(--line);border-radius:8px;padding:10px;font-size:11px;
 overflow-x:auto;max-height:260px;overflow-y:auto;color:var(--ink2)}
.banner{border-radius:10px;padding:10px 14px;margin:10px 0;font-weight:600;font-size:13px}
.b-bad{background:#2a1414;border:1px solid var(--bad);color:var(--bad)}
.b-warn{background:#251c0d;border:1px solid var(--warn);color:var(--warn)}
.b-ok{background:#0f2010;border:1px solid var(--good);color:var(--good)}
h2{font-size:14px;margin:22px 0 8px}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:14px}
@media(max-width:800px){.cols{grid-template-columns:1fr}.stage{grid-template-columns:22px 1fr 80px}}
/* 차트 색 슬롯 — 대시보드는 dark 고정 (validate_palette dark 통과 세트) */
.vz{--vz-s1:#3987e5;--vz-s2:#d95926;--vz-s3:#199e70;--vz-s4:#c98500;
 --vz-grid:#2c2c2a;--vz-axis:#383835;--vz-ink2:#c3c2b7;--vz-mut:#898781;--vz-surface:#1a1a19;
 --vz-good:#0ca30c;--vz-bad:#e66767}
__CHART_CSS__
</style></head><body><div class="wrap">
<h1>Retrospection 체인 — EGO_jihun3</h1>
<div class="sub" id="updated">loading…</div>
<div id="banners"></div>
<div class="grid" id="kpis"></div>
<div class="card"><div id="stages"></div></div>
<div class="cols">
 <div><h2>실측값</h2><div class="card" id="facts"></div></div>
 <div><h2>평가 (heldout · L0 병기)</h2><div class="card" id="evals"></div></div>
</div>
<h2>추론 능력 추이 — SFT·DPO 학습 곡선</h2>
<div class="card vz" id="curves"></div>
<h2>정성 로그 — reasoning·action 변화 (probe 전문)</h2>
<div class="card vz" id="qual"></div>
<h2>트레이스 진화 — 고정 probe 8샘플 (SFT 100step · DPO 50step 간격)</h2>
<div class="card" id="probes"></div>
<h2>로그 tail</h2><pre id="log"></pre>
<script>
__CHART_JS__
</script>
<script>
const f=(x,d=3)=>x==null?"—":(+x).toFixed(d);
const dur=s=>{if(s==null)return"—";s=Math.max(0,s|0);const h=(s/3600)|0,m=((s%3600)/60)|0;
 return h?`${h}h ${m}m`:`${m}m ${s%60}s`};
async function tick(){
 let d; try{d=await(await fetch("/api/status")).json()}catch(e){return}
 document.getElementById("updated").textContent=
  `갱신 ${new Date(d.ts*1000).toLocaleTimeString("ko-KR")} · 5초마다 자동 새로고침`;
 const B=[];
 if(d.chain_done)B.push(['b-ok','✔ 체인 전체 완료']);
 if(d.chain_stuck)B.push(['b-bad','■ CHAIN_STUCK — 자동 재시도 한도 초과 또는 GPU 다운: '+JSON.stringify(d.chain_stuck)]);
 else if(d.chain_failed)B.push(['b-warn','⟳ 스테이지 실패 — supervisor가 재시작 예정: '+JSON.stringify(d.chain_failed)]);
 if(d.g3_abort)B.push(['b-bad','■ G3 문체-학습 가드 발동: '+JSON.stringify(d.g3_abort)]);
 if(d.semantic_skipped)B.push(['b-warn','△ semantic gate SKIP (LETSUR_API_KEY 없음) — pair는 규칙 판정만']);
 if(!d.supervisor_alive&&!d.chain_done)B.push(['b-warn','△ supervisor 프로세스 없음 — start_chain.sh로 재기동 필요']);
 document.getElementById("banners").innerHTML=B.map(([c,t])=>`<div class="banner ${c}">${t}</div>`).join("");
 const run=d.stages.find(s=>s.state==="running");
 const doneN=d.stages.filter(s=>s.state==="done").length;
 const g=d.gpu[0]||{};
 document.getElementById("kpis").innerHTML=[
  ["남은 시간(전체)",dur(d.eta_total_sec),"현재 스테이지 실측 + 대기 추정"],
  ["진행",`${doneN}/${d.stages.length}`,"완료 스테이지"],
  ["현재 스테이지",run?run.title:(d.chain_done?"완료":"대기/중단"),
   run?`${run.done??"?"}/${run.total??"?"} · ${dur(run.eta_sec)} 남음`:""],
  ["GPU",g.name?`${g.util}%`:"—",g.name?`${(g.mem_used_mb/1024).toFixed(0)}/${(g.mem_total_mb/1024).toFixed(0)}GB · ${g.temp}°C`:"접근 불가"],
 ].map(([l,v,s])=>`<div class="card kpi"><div class="l">${l}</div><div class="v">${v}</div><div class="s">${s}</div></div>`).join("");
 document.getElementById("stages").innerHTML=d.stages.map(s=>{
  const pct=s.state==="done"?100:(s.pct||0);
  const right=s.state==="running"?`<span class="num">${dur(s.eta_sec)} 남음 · ${f(s.rate_per_s,2)}/s</span>`
   :s.state==="done"?`<span class="num ok">완료${s.elapsed?` · ${dur(s.elapsed)}`:""}</span>`
   :s.state==="failed"?`<span class="num bad">실패</span>`
   :s.state==="stale"?`<span class="num warn">응답 없음</span>`
   :`<span class="num mut">~${dur(s.est_sec)}</span>`;
  const m=s.metrics||{};
  const chips=Object.entries(m).filter(([k,v])=>typeof v!=="object").slice(0,4)
   .map(([k,v])=>`${k}=${typeof v==="number"?f(v,3):v}`).join("  ");
  return `<div class="stage ${s.state}"><div class="dot"></div>
   <div>${s.title}<div class="num mut">${chips}</div></div>
   <div class="bar"><i style="width:${pct}%"></i></div>${right}</div>`}).join("");
 // 실측값 카드
 const F=[];
 if(d.coverage&&d.coverage.val)F.push(`<tr><td>coverage@10 (val)</td><td>${f(d.coverage.val["cov@10"])}</td></tr>
  <tr><td>coverage@10 (train)</td><td>${d.coverage.train?f(d.coverage.train["cov@10"]):"—"}</td></tr>`);
 if(d.pairs)F.push(`<tr><td>pair 채택</td><td>${d.pairs.accepted} (${f(d.pairs.acceptance_rate)})</td></tr>
  <tr><td>pair BA/B/A</td><td>${d.pairs.types.BA}/${d.pairs.types.B}/${d.pairs.types.A}</td></tr>`);
 const s3=d.stages.find(x=>x.id==="S3_hindsight");
 if(s3&&s3.metrics&&s3.metrics.pass_rate!=null)F.push(`<tr><td>projection gate 통과율</td><td>${f(s3.metrics.pass_rate)}</td></tr>`);
 const tr=d.stages.filter(x=>x.id.startsWith("S6")&&x.metrics&&x.metrics.loss_ema!=null);
 tr.forEach(x=>F.push(`<tr><td>${x.id} loss</td><td>${f(x.metrics.loss_ema)}</td></tr>`+
  (x.metrics.belief_margin_ema!=null?`<tr><td>· belief/action margin</td><td>${f(x.metrics.belief_margin_ema)} / ${f(x.metrics.action_margin_ema)}</td></tr>`:"")));
 document.getElementById("facts").innerHTML=F.length?`<table>${F.join("")}</table>`:'<span class="mut">아직 없음</span>';
 // eval 표
 const evs=Object.entries(d.eval||{});
 const arms=evs.filter(([k,v])=>v&&v.acc!=null), ivs=evs.filter(([k,v])=>v&&v.p_gt);
 let H="";
 if(arms.length){
  const cond=arms.some(([k,v])=>v.covered_only);
  if(cond)H+=`<div class="num warn" style="margin-bottom:6px">△ 조건부 지표 — GT∈Top-10(covered) 샘플만 평가`
   +` (heldout pool cov@10=${f(arms.find(([k,v])=>v.covered_only)[1].pool_coverage)})`
   +` · acc(full)=acc×pool_cov 환산 병기</div>`;
  H+=`<table><tr><th>arm</th><th>acc${cond?' |cov':''}</th>${cond?'<th>acc(full)</th>':''}<th>L0</th><th>G1 ret</th><th>G2 corr</th><th>mal</th></tr>`+
  arms.map(([k,v])=>`<tr><td>${v.arm}${v.covered_only?' <span class="warn">△</span>':''}</td><td class="${v.beats_L0?'ok':''}">${f(v.acc)}</td>${cond?`<td>${f(v.acc_full_equiv)}</td>`:''}<td>${f(v.L0_wm_top1)}</td><td>${f(v.G1_retention)}</td><td>${f(v.G2_correction)}</td><td>${f(v.malformed_rate)}</td></tr>`).join("")+`</table>`}
 if(ivs.length){H+=`<table style="margin-top:8px"><tr><th>③ arm</th><th>p(gen)</th><th>p(∅)</th><th>p(incompat)</th><th>swap−para</th></tr>`+
  ivs.map(([k,v])=>`<tr><td>${v.arm}</td><td>${f(v.p_gt.gen)}</td><td>${f(v.p_gt.empty)}</td><td>${f(v.p_gt.incompat)}</td><td class="${v.causal_sensitivity>0?'ok':''}">${f(v.causal_sensitivity)}</td></tr>`).join("")+`</table>`}
 document.getElementById("evals").innerHTML=H||'<span class="mut">아직 없음</span>';
 // 트레이스 진화: run별 acc 곡선 + 샘플별 step0→최신 대비
 const probes=Object.entries(d.probes||{});
 let P="";
 for(const [run,pv] of probes){
  const curve=pv.curve.map(c=>`${c.step}:${f(c.acc,2)}`).join(" → ");
  const rows=pv.latest.samples.map((s,i)=>{
   const s0=(pv.first.samples[i]||{});
   const chg=s0.action!==s.action;
   return `<tr><td>${s.bucket}</td><td class="mut">${(s.gt||"").slice(0,26)}</td>
    <td class="${s0.correct?'ok':''}">${s0.action||'∅'}</td>
    <td class="${s.correct?'ok':(chg?'warn':'')}">${s.action||'∅'}${chg?' ↺':''}</td>
    <td style="text-align:left;max-width:340px;white-space:normal;font-size:11.5px">${(s.task_belief||'').slice(0,150)}</td></tr>`}).join("");
  P+=`<div style="margin-bottom:14px"><b>${run}</b>
   <span class="num mut"> probe acc: ${curve} (step ${pv.latest.step})</span>
   <div style="overflow-x:auto"><table><tr><th>구간</th><th>GT</th><th>action@step${pv.first.step}</th>
   <th>action@step${pv.latest.step}</th><th>task_belief (최신)</th></tr>${rows}</table></div></div>`;
 }
 document.getElementById("probes").innerHTML=P||'<span class="mut">S6 학습 시작 후 표시됩니다 (probe: dev 고정 8샘플 — G1 2·G2 4·기타 2)</span>';
 // 추론 능력 곡선 — 데이터 길이가 변했을 때만 다시 그림 (hover 상태 보존)
 const vzData={curves:d.train_curves||{},probes:Object.fromEntries(
  Object.entries(d.probes||{}).map(([k,v])=>[k,v.curve]))};
 const vzQual=Object.fromEntries(Object.entries(d.probes||{}).map(([k,v])=>[k,v.entries||[]]));
 const vzSig=JSON.stringify([Object.entries(vzData.curves).map(([k,v])=>[k,v.points.length]),
  Object.entries(vzData.probes).map(([k,v])=>[k,v.length]),
  Object.entries(vzQual).map(([k,v])=>[k,v.length])]);
 if(vzSig!==window.__vzSig){window.__vzSig=vzSig;
  VZ.renderAll(document.getElementById("curves"),vzData);
  VZ.qualLog(document.getElementById("qual"),vzQual)}
 document.getElementById("log").textContent=d.log_tail||"";
}
tick(); setInterval(tick,5000);
</script></div></body></html>"""

PAGE = PAGE.replace("__CHART_CSS__", CHART_CSS).replace("__CHART_JS__", CHART_JS)


class H(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/api/status"):
            body = json.dumps(collect(), ensure_ascii=False).encode()
            ctype = "application/json; charset=utf-8"
        else:
            body = PAGE.encode()
            ctype = "text/html; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):  # 콘솔 소음 억제
        pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7867)
    ap.add_argument("--runs", default="runs/retro3")
    args = ap.parse_args()
    RUNS = Path(args.runs)
    globals()["RUNS"] = RUNS
    print(f"retro3 dashboard: http://0.0.0.0:{args.port}  (runs={RUNS})")
    ThreadingHTTPServer(("0.0.0.0", args.port), H).serve_forever()
