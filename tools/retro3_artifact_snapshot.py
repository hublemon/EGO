#!/usr/bin/env python3
"""retro3 체인 현황 → 공개 아티팩트용 정적 스냅샷 HTML 생성.

로컬 대시보드 API(localhost:7867)를 읽어 자기완결 HTML을 쓴다.
사용: python3 tools/retro3_artifact_snapshot.py --out <path.html>
"""
from __future__ import annotations

import argparse
import datetime
import json
import urllib.request

from retro3_dashboard import CHART_CSS, CHART_JS  # 같은 tools/ 디렉터리 — 차트 렌더러 공유

KST = datetime.timezone(datetime.timedelta(hours=9))


def dur(s):
    if s is None:
        return "—"
    s = max(0, int(s))
    h, m = s // 3600, (s % 3600) // 60
    return f"{h}h {m}m" if h else f"{m}m"


def fetch(url="http://localhost:7867/api/status"):
    with urllib.request.urlopen(url, timeout=10) as r:
        return json.loads(r.read())


STATE_LABEL = {"done": "완료", "running": "진행 중", "failed": "실패",
               "pending": "대기", "stale": "응답 없음"}


def _notices() -> list:
    """runs/retro3/notices.json → 스냅샷 상단 고정 배너 (재생성해도 유지).
    형식: [{"level": "ok|warn|bad", "text": "..."}]"""
    try:
        return json.loads(open("runs/retro3/notices.json", encoding="utf-8").read())
    except Exception:
        return []


def render(d: dict) -> str:
    now = datetime.datetime.fromtimestamp(d["ts"], KST)
    eta_done = now + datetime.timedelta(seconds=d["eta_total_sec"] or 0)
    run = next((s for s in d["stages"] if s["state"] == "running"), None)
    done_n = sum(1 for s in d["stages"] if s["state"] == "done")
    g = (d.get("gpu") or [{}])[0]

    banners = [(n.get("level", "warn"), n.get("text", "")) for n in _notices()]
    if d.get("chain_done"):
        banners.append(("ok", "체인 전체 완료"))
    if d.get("chain_stuck"):
        banners.append(("bad", "CHAIN_STUCK — 개입 필요"))
    elif d.get("chain_failed"):
        banners.append(("warn", "스테이지 실패 — supervisor 재시작 예정"))
    if d.get("g3_abort"):
        banners.append(("bad", "G3 문체-학습 가드 발동"))
    if d.get("semantic_skipped"):
        banners.append(("warn", "semantic gate SKIP (API 키 없음) — pair는 규칙 판정만"))

    rows = []
    for s in d["stages"]:
        pct = 100 if s["state"] == "done" else (s.get("pct") or 0)
        detail = ""
        if s["state"] == "running":
            detail = f'{s.get("done", "?")}/{s.get("total", "?")} · 남음 {dur(s.get("eta_sec"))}'
        elif s["state"] == "done" and s.get("elapsed"):
            detail = dur(s.get("elapsed"))
        elif s["state"] == "pending":
            detail = f'~{dur(s.get("est_sec"))}'
        m = s.get("metrics") or {}
        chips = " · ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                           for k, v in list(m.items())[:3] if not isinstance(v, dict))
        rows.append(f"""<div class="stage st-{s['state']}"><span class="dot"></span>
<div class="st-body"><div class="st-t">{s['title']}</div><div class="st-c">{chips}</div></div>
<div class="bar"><i style="width:{pct:.0f}%"></i></div>
<div class="st-d">{STATE_LABEL[s['state']]}{' · ' + detail if detail else ''}</div></div>""")

    facts = []
    cov = d.get("coverage") or {}
    if cov.get("val"):
        facts.append(("coverage@10 (heldout val)", f"{cov['val']['cov@10']:.4f}",
                      f"probe 참조 0.3984 · train {cov.get('train', {}).get('cov@10', 0):.3f}"))
    pr = d.get("pairs")
    if pr:
        facts.append(("DPO pair", f"{pr['accepted']}",
                      f"BA {pr['types']['BA']} / B {pr['types']['B']} / A {pr['types']['A']} · 채택률 {pr['acceptance_rate']:.2f}"))
    if run and run.get("metrics"):
        rm = {k: v for k, v in run["metrics"].items() if isinstance(v, (int, float))}
        if rm:
            k0 = list(rm)[:2]
            facts.append((f"현재: {run['title']}",
                          " · ".join(f"{k} {rm[k]:.3f}" if isinstance(rm[k], float) else f"{k} {rm[k]}" for k in k0),
                          f"{run.get('done')}/{run.get('total')}"))

    evals = {k: v for k, v in (d.get("eval") or {}).items() if v}

    def arm_row(v: dict) -> str:
        full = f"{v['acc_full_equiv']:.4f}" if v.get("acc_full_equiv") is not None else "—"
        l0_full = f"{v['L0_full_equiv']:.4f}" if v.get("L0_full_equiv") is not None else "—"
        return (f"<tr><td>{v['arm']}</td>"
                f"<td class='{'ok' if v.get('beats_L0') else ''}'>{v['acc']:.4f}</td>"
                f"<td>{full}</td>"
                f"<td>{v['L0_wm_top1']:.4f}</td><td>{l0_full}</td>"
                f"<td>{v['G1_retention']:.3f}</td><td>{v['G2_correction']:.3f}</td></tr>")

    arm_rows = "".join(arm_row(v) for v in evals.values() if v.get("acc") is not None)
    iv_rows = "".join(
        f"<tr><td>{v['arm']}</td><td>{v['p_gt']['gen']:.4f}</td><td>{v['p_gt']['empty']:.4f}</td>"
        f"<td>{v['p_gt']['incompat']:.4f}</td><td class='{'ok' if v['causal_sensitivity']>0 else ''}'>{v['causal_sensitivity']:.4f}</td></tr>"
        for v in evals.values() if v.get("p_gt"))

    kpi = f"""
<div class="card kpi"><div class="l">전체 남은 시간</div><div class="v">{dur(d['eta_total_sec'])}</div>
<div class="s">완주 예상 {eta_done.strftime('%m/%d %H:%M')} KST</div></div>
<div class="card kpi"><div class="l">진행</div><div class="v">{done_n}/{len(d['stages'])}</div><div class="s">완료 스테이지</div></div>
<div class="card kpi"><div class="l">현재 스테이지</div><div class="v" style="font-size:16px">{run['title'] if run else ('완료' if d.get('chain_done') else '대기')}</div>
<div class="s">{f"{run.get('done')}/{run.get('total')} · 남음 {dur(run.get('eta_sec'))}" if run else ''}</div></div>
<div class="card kpi"><div class="l">GPU (H200)</div><div class="v">{g.get('util', '—')}%</div>
<div class="s">{f"{g.get('mem_used_mb',0)//1024}/{g.get('mem_total_mb',1)//1024}GB · {g.get('temp','?')}°C" if g else '측정 없음'}</div></div>"""

    facts_html = "".join(f"<tr><td>{a}</td><td><b>{b}</b></td><td class='mut'>{c}</td></tr>" for a, b, c in facts)
    banners_html = "".join(f'<div class="banner b-{c}">{t}</div>' for c, t in banners)

    # 트레이스 진화 (probe): run별 acc 곡선 + step0 vs 최신 action/belief 대비
    # 주의: 스테이지 목록 `rows`(list)와 이름 충돌 금지 — 덮어쓰면 스테이지 카드가 probe 행으로 대체된다.
    probe_html = ""
    for rname, pv in (d.get("probes") or {}).items():
        curve = " → ".join(f"{c['step']}:{c['acc']:.2f}" for c in pv["curve"])
        prows = ""
        for i, s in enumerate(pv["latest"]["samples"]):
            s0 = pv["first"]["samples"][i] if i < len(pv["first"]["samples"]) else {}
            chg = " ↺" if s0.get("action") != s.get("action") else ""
            prows += (f"<tr><td>{s['bucket']}</td><td class='mut'>{(s['gt'] or '')[:24]}</td>"
                      f"<td class='{'ok' if s0.get('correct') else ''}'>{s0.get('action') or '∅'}</td>"
                      f"<td class='{'ok' if s.get('correct') else ''}'>{(s.get('action') or '∅') + chg}</td>"
                      f"<td class='belief'>{(s.get('task_belief') or '')[:140]}</td></tr>")
        probe_html += (f"<div style='margin-bottom:12px'><b>{rname}</b> "
                       f"<span class='mut' style='font-size:11.5px'>probe acc: {curve}</span>"
                       f"<div class='tw'><table><tr><th>구간</th><th>GT</th>"
                       f"<th>action@{pv['first']['step']}</th><th>action@{pv['latest']['step']}</th>"
                       f"<th>task_belief (최신)</th></tr>{prows}</table></div></div>")
    probe_section = (f"<h2>트레이스 진화 — 고정 probe 8샘플 (G1 2 · G2 4 · 기타 2)</h2>"
                     f"<div class='card'>{probe_html}</div>") if probe_html else ""

    # 추론 능력 추이 — 대시보드와 같은 렌더러(CHART_JS)에 스냅샷 데이터를 인라인 주입
    vz_data = {"curves": d.get("train_curves") or {},
               "probes": {k: v["curve"] for k, v in (d.get("probes") or {}).items()}}
    vz_qual = {k: v.get("entries") or [] for k, v in (d.get("probes") or {}).items()}
    curves_section = ""
    if vz_data["curves"] or vz_data["probes"]:
        # 차트 색 슬롯 — light/dark 각각 validate_palette 통과 세트, 테마 토글이 양방향으로 이김
        vz_style = ("<style>"
                    ".vz{--vz-s1:#2a78d6;--vz-s2:#eb6834;--vz-s3:#1baf7a;--vz-s4:#eda100;"
                    "--vz-grid:#e1e0d9;--vz-axis:#c3c2b7;--vz-ink2:#52514e;--vz-mut:#898781;"
                    "--vz-surface:#fcfcfb;--vz-good:#006300;--vz-bad:#d03b3b}"
                    '@media (prefers-color-scheme: dark){:root:where(:not([data-theme="light"])) .vz{'
                    "--vz-s1:#3987e5;--vz-s2:#d95926;--vz-s3:#199e70;--vz-s4:#c98500;"
                    "--vz-grid:#2c2c2a;--vz-axis:#383835;--vz-ink2:#c3c2b7;--vz-surface:#1a1a19;"
                    "--vz-good:#0ca30c;--vz-bad:#e66767}}"
                    ':root[data-theme="dark"] .vz{'
                    "--vz-s1:#3987e5;--vz-s2:#d95926;--vz-s3:#199e70;--vz-s4:#c98500;"
                    "--vz-grid:#2c2c2a;--vz-axis:#383835;--vz-ink2:#c3c2b7;--vz-surface:#1a1a19;"
                    "--vz-good:#0ca30c;--vz-bad:#e66767}"
                    f"{CHART_CSS}</style>")
        curves_section = (
            f"{vz_style}"
            "<h2>추론 능력 추이 — SFT·DPO 학습 곡선</h2>"
            '<div class="card vz" id="vz"></div>'
            "<h2>정성 로그 — reasoning·action 변화 (probe 전문)</h2>"
            '<div class="card vz" id="vzq"></div>'
            f"<script>const VZDATA={json.dumps(vz_data, ensure_ascii=False)};\n"
            f"const VZQUAL={json.dumps(vz_qual, ensure_ascii=False)};\n"
            f"{CHART_JS}\n"
            'VZ.renderAll(document.getElementById("vz"),VZDATA);\n'
            'VZ.qualLog(document.getElementById("vzq"),VZQUAL);</script>')

    return f"""<title>Retrospection 체인 현황 — EGO_jihun3</title>
<style>
:root {{ --bg:#f9f9f7; --card:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --mut:#898781;
 --line:#e1e0d9; --blue:#2a78d6; --good:#006300; --bad:#d03b3b; --warn:#8a5a00; }}
@media (prefers-color-scheme: dark) {{ :root:where(:not([data-theme="light"])) {{
 --bg:#0d0d0d; --card:#1a1a19; --ink:#fff; --ink2:#c3c2b7; --line:#2c2c2a;
 --blue:#3987e5; --good:#0ca30c; --bad:#e66767; --warn:#c98500; }} }}
:root[data-theme="dark"] {{ --bg:#0d0d0d; --card:#1a1a19; --ink:#fff; --ink2:#c3c2b7;
 --line:#2c2c2a; --blue:#3987e5; --good:#0ca30c; --bad:#e66767; --warn:#c98500; }}
:root[data-theme="light"] {{ --bg:#f9f9f7; --card:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e;
 --line:#e1e0d9; --blue:#2a78d6; --good:#006300; --bad:#d03b3b; --warn:#8a5a00; }}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);
 font:14px/1.6 system-ui,-apple-system,"Segoe UI",sans-serif;padding:24px 16px}}
.wrap{{max-width:900px;margin:0 auto}} h1{{font-size:20px;margin:0 0 2px;letter-spacing:-.01em}}
.sub{{color:var(--mut);font-size:12.5px;margin-bottom:14px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px;margin:14px 0}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:13px 15px}}
.kpi .l{{font-size:11px;color:var(--mut);font-weight:650}} .kpi .v{{font-size:22px;font-weight:800;font-variant-numeric:tabular-nums}}
.kpi .s{{font-size:11.5px;color:var(--ink2)}}
.banner{{border-radius:9px;padding:9px 13px;margin:8px 0;font-weight:600;font-size:13px;border:1px solid}}
.b-ok{{color:var(--good);border-color:var(--good)}} .b-bad{{color:var(--bad);border-color:var(--bad)}}
.b-warn{{color:var(--warn);border-color:var(--warn)}}
.stage{{display:grid;grid-template-columns:16px 1fr 110px 150px;gap:10px;align-items:center;
 padding:7px 0;border-bottom:1px solid var(--line);font-size:13px}}
.stage:last-child{{border-bottom:none}}
.dot{{width:9px;height:9px;border-radius:50%;background:var(--line)}}
.st-done .dot{{background:var(--good)}} .st-running .dot{{background:var(--blue)}}
.st-failed .dot{{background:var(--bad)}} .st-stale .dot{{background:var(--warn)}}
.st-t{{font-weight:600}} .st-c{{font-size:11px;color:var(--mut);font-variant-numeric:tabular-nums}}
.bar{{height:6px;background:var(--line);border-radius:4px;overflow:hidden}}
.bar i{{display:block;height:100%;background:var(--blue)}}
.st-d{{font-size:11.5px;color:var(--ink2);text-align:right;font-variant-numeric:tabular-nums}}
table{{border-collapse:collapse;width:100%;font-size:13px}}
th,td{{padding:6px 10px;border-bottom:1px solid var(--line);text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}}
th{{font-size:11px;color:var(--mut);text-transform:uppercase;letter-spacing:.05em}}
td:first-child,th:first-child{{text-align:left}} tr:last-child td{{border-bottom:none}}
.ok{{color:var(--good);font-weight:700}} .mut{{color:var(--mut)}}
td.belief{{text-align:left;max-width:320px;white-space:normal;font-size:11.5px;color:var(--ink2)}}
h2{{font-size:14.5px;margin:22px 0 8px}}
.tw{{overflow-x:auto}}
@media(max-width:640px){{.stage{{grid-template-columns:16px 1fr 90px}}.st-d{{display:none}}}}
</style>
<div class="wrap">
<h1>Retrospection 체인 현황</h1>
<div class="sub">EGO_jihun3 · GoalStep strict start−1s · Top-10 support ·
스냅샷 {now.strftime('%m/%d %H:%M')} KST (30분 주기 + 스테이지 전환 시 자동 갱신)</div>
{banners_html}
<div class="grid">{kpi}</div>
<div class="card">{''.join(rows)}</div>
<h2>실측값</h2>
<div class="card tw"><table>{facts_html or '<tr><td class="mut">아직 없음</td></tr>'}</table></div>
<h2>평가 (heldout n=1,000 · L0 병기)</h2>
<div class="card tw">{f'<div class="mut" style="font-size:11.5px;margin-bottom:6px">acc|cov = GT∈Top-10 조건부 정답률 (covered-only 평가) · acc(full) = acc|cov × pool_coverage — uncovered(정답이 후보에 없음, 구조적 0점)까지 포함한 전체 환산</div><table><tr><th>arm</th><th>acc|cov</th><th>acc(full)</th><th>L0|cov</th><th>L0(full)</th><th>G1 ret</th><th>G2 corr</th></tr>{arm_rows}</table>' if arm_rows else '<span class="mut">평가 단계 도달 전</span>'}
{f'<table style="margin-top:10px"><tr><th>③ arm</th><th>p(gen)</th><th>p(∅)</th><th>p(incompat)</th><th>swap−para</th></tr>{iv_rows}</table>' if iv_rows else ''}</div>
{curves_section}
{probe_section}
<div class="sub" style="margin-top:18px">체인: S0 support → context → base trace(y−) → hindsight Ψ→Φ→gate(y+) →
pairs → R1 SFT · FB-DPO → 배터리 + 개입③ → DPO ablation 3-arm(all/rule/sem).
성공 판정은 belief 개입 테스트(③)만 사용.</div>
</div>"""


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    html = render(fetch())
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"snapshot written: {args.out}")
