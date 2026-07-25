#!/usr/bin/env python3
"""cesft_v2 최적화 체인 진행 렌더러 — 관측 전용(파일만 읽음, 아무것도 안 죽임).

핸드오프 §4 최적화 직렬 체인(~14h)을 SSOT 로 삼아,
markers/ + status/*.json + eval/ 를 읽어:
  - 각 스테이지 상태(done/running/pending/skipped/conditional)
  - 현재 단계 잔여(초, 라이브 status eta 우선)
  - 총 잔여(러닝 eta + pending 예산 합; 조건부는 별도)
  - 지금까지 절약된 시간 합계
를 계산해 progress.json 과 self-refresh 로컬 HTML 로 출력한다.

사용: PYTHON tools/oom_opt/render_progress.py [RUN_DIR]
표준 출력물: <RUN>/optim_progress.json, <RUN>/optim_progress.html
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

RUN = Path(sys.argv[1] if len(sys.argv) > 1 else
           "/mnt/nvme/migration/jihun/EGO_jihun3/runs/cesft_v2")
MK = RUN / "markers"
ST = RUN / "status"
EV = RUN / "eval"

# ── 최적화 직렬 체인 (핸드오프 §4) — budget_sec 은 pending 추정용 ──
# status: StatusWriter 파일명(소문자). marker: 완료 판정.
PLAN = [
    dict(key="theta_ce",       title="θ_CE (WM-cand selection-CE)",           phase="A", budget=16200,
         marker="S_CE_THETA_CE_DONE",     status="S_CE_theta_ce"),
    dict(key="eval_theta_ce",  title="θ_CE 배터리 + G-ACC1",                  phase="A", budget=1800,
         marker="S7_EVAL_THETA_CE_DONE",  status="S7_eval_theta_ce"),
    dict(key="sft_r15",        title="core sft_r15 (CE-replay ρ=0.15)",       phase="A", budget=9000,
         marker="S6_SFT_R15_DONE",        status="S6_sft_r15"),
    dict(key="eval_sft_r15",   title="sft_r15 배터리",                        phase="A", budget=1800,
         marker="S7_EVAL_SFT_R15_DONE",   status="S7_eval_sft_r15"),
    dict(key="harden_sft_r15", title="sft_r15 harden(U_g) + G-NH · headline", phase="A", budget=2400,
         marker="S3H_SFT_R15_DONE",       status="S3H_sft_r15", note="IV_N=800"),
    dict(key="strip_eval",     title="θ_CE strip-eval (history 인과, paired)", phase="B*", budget=1800,
         marker="S_STRIP_THETA_CE_DONE",  status="S_STRIP_theta_ce", note="NEW · B_nohist 대체"),
    dict(key="sft_r0",         title="sft_r0 ablation (replay 없음)",         phase="C", budget=9000,
         marker="S6_SFT_R0_DONE",         status="S6_sft_r0"),
    dict(key="eval_sft_r0",    title="sft_r0 배터리",                         phase="C", budget=1800,
         marker="S7_EVAL_SFT_R0_DONE",    status="S7_eval_sft_r0"),
    dict(key="harden_sft_r0",  title="sft_r0 harden + G-NH",                  phase="C", budget=2400,
         marker="S3H_SFT_R0_DONE",        status="S3H_sft_r0"),
    dict(key="wise_a050",      title="WiSE-FT α=0.5 (merge+eval+harden)",     phase="C", budget=2160,
         marker="S3H_WISE_A050_DONE",     status="S3H_wise_a050", note="1점만"),
    dict(key="report",         title="리포트 아티팩트 확정",                  phase="—", budget=60,
         marker="CESFT_V2_CHAIN_DONE",    status=None),
]

# 조건부(게이트 통과 시에만) — 총 잔여에 별도 표기
CONDITIONAL = [
    dict(key="sft_r30", title="sft_r30 fallback (+eval+harden)", budget=12600,
         cond="r15 G-NH FAIL 시에만", marker="S6_SFT_R30_DONE"),
    dict(key="appendix_a", title="부록A C-stack/C-ctrl (T-ACC)", budget=10800,
         cond="P-UTIL PASS 시에만", marker="S_CE_C_STACK_DONE"),
]

# touch 로 스킵되는 스테이지 (핸드오프 §2) — 절약 시간 집계
SKIPPED = [
    dict(key="cand_free",     title="B candidate-free CE + 배터리",  saved=7920,
         marker="S_CE_CAND_FREE_DONE", why="EGO_jihun 성립부등식 확정치 재인용 (§2)"),
    dict(key="no_history",    title="B no-history CE 학습",          saved=12960,
         marker="S_CE_NO_HISTORY_DONE", why="strip-eval(0.5h)로 대체 — 같은 θ_CE paired (§2)"),
    dict(key="wise_a025_075", title="WiSE α∈{.25,.75} 2점",         saved=3960,
         marker="S3H_WISE_A025_DONE", why="frontier 곡선은 논문 필요 시 (§2)"),
]


def read_json(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def marker_done(name: str) -> bool:
    return (MK / name).is_file()


def stage_state(s: dict) -> dict:
    """스테이지 상태·잔여 계산."""
    done = marker_done(s["marker"])
    stj = read_json(ST / f"{s['status']}.json") if s.get("status") else None
    running = (not done) and stj is not None and stj.get("state") == "running"
    remaining = 0.0
    pct = None
    metrics = {}
    if done:
        remaining = 0.0
        pct = 100.0
    elif running:
        eta = stj.get("eta_sec")
        pct = stj.get("pct")
        metrics = stj.get("metrics", {}) or {}
        if eta is not None:
            remaining = float(eta)
        elif pct:
            remaining = s["budget"] * max(0.0, 1 - pct / 100.0)
        else:
            remaining = float(s["budget"])
    else:
        remaining = float(s["budget"])
    return dict(key=s["key"], title=s["title"], phase=s.get("phase", ""),
                note=s.get("note", ""), budget=s["budget"],
                state="done" if done else "running" if running else "pending",
                remaining_sec=round(remaining), pct=round(pct, 1) if pct is not None else None,
                metrics=metrics)


def gate_verdict() -> dict:
    """G-NH(r15) / P-UTIL 판정 파일이 있으면 읽는다."""
    out = {}
    gnh = read_json(EV / "paired_G-NH_sft_r15_vs_theta_ce.json")
    if gnh is not None:
        out["g_nh_r15"] = "PASS" if gnh.get("pass") else "FAIL"
    h = read_json(EV / "sft_r15.harden_s3.json")
    if h is not None:
        out["r15_verdict"] = h.get("verdict")
        ug = (h.get("utility_belief_only_ci") or {})
        out["r15_Ug_lo"] = ug.get("lo")
    strip = read_json(EV / "strip_verdict.json")
    if strip is not None:
        out["strip_delta_acc_pp"] = strip.get("delta_acc_all", {}).get("delta")
        out["strip_ci"] = strip.get("delta_acc_all", {}).get("ci")
    return out


def build() -> dict:
    stages = [stage_state(s) for s in PLAN]
    total_remaining = sum(s["remaining_sec"] for s in stages)
    cur = next((s for s in stages if s["state"] == "running"), None)
    saved = 0
    skipped_rows = []
    for sk in SKIPPED:
        applied = marker_done(sk["marker"])
        if applied:
            saved += sk["saved"]
        skipped_rows.append(dict(**{k: sk[k] for k in ("key", "title", "why")},
                                 saved_sec=sk["saved"], applied=applied))
    cond_rows = []
    for c in CONDITIONAL:
        resolved = marker_done(c["marker"])
        cond_rows.append(dict(key=c["key"], title=c["title"], budget_sec=c["budget"],
                              cond=c["cond"], engaged=resolved))
    # 체인 시작 시각 = theta_ce status started_at (있으면)
    tj = read_json(ST / "S_CE_theta_ce.json")
    started_at = tj.get("started_at") if tj else None
    return dict(
        generated_at=time.time(),
        run=str(RUN),
        stages=stages,
        current=cur,
        total_remaining_sec=total_remaining,
        conditional=cond_rows,
        skipped=skipped_rows,
        saved_sec_applied=saved,
        saved_sec_max=sum(s["saved"] for s in SKIPPED),
        gates=gate_verdict(),
        chain_started_at=started_at,
        chain_done=marker_done("CESFT_V2_CHAIN_DONE"),
        stuck=marker_done("CHAIN_STUCK"),
    )


def hms(sec: float) -> str:
    sec = max(0, int(sec))
    h, r = divmod(sec, 3600)
    m, s = divmod(r, 60)
    return f"{h}h {m:02d}m" if h else f"{m}m {s:02d}s"


def render_html(d: dict) -> str:
    def esc(x):
        return (str(x).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
    badge = {"done": "#1a7f37", "running": "#0969da", "pending": "#8b949e"}
    rows = []
    for s in d["stages"]:
        col = badge[s["state"]]
        pct = f'{s["pct"]}%' if s["pct"] is not None else ""
        bar = ""
        if s["state"] == "running" and s["pct"] is not None:
            bar = (f'<div class="bar"><div class="fill" style="width:{s["pct"]}%"></div></div>')
        rem = "—" if s["state"] == "done" else hms(s["remaining_sec"])
        met = ""
        if s["metrics"]:
            keys = [k for k in ("loss_ema", "step", "probe_acc", "verdict") if k in s["metrics"]]
            met = " · ".join(f'{k}={s["metrics"][k]}' for k in keys)
        note = f' <span class="note">{esc(s["note"])}</span>' if s["note"] else ""
        rows.append(f'''<tr class="{s['state']}">
          <td class="ph">{esc(s['phase'])}</td>
          <td><b>{esc(s['title'])}</b>{note}{bar}<span class="met">{esc(met)}</span></td>
          <td><span class="pill" style="background:{col}">{s['state']}</span></td>
          <td class="num">{pct}</td>
          <td class="num">{rem}</td></tr>''')
    cond = "".join(
        f'''<tr><td>{esc(c['title'])}</td><td class="num">+{hms(c['budget_sec'])}</td>
        <td>{esc(c['cond'])}</td><td>{"진입됨" if c['engaged'] else "대기"}</td></tr>'''
        for c in d["conditional"])
    skip = "".join(
        f'''<tr class="{'on' if s['applied'] else 'off'}"><td>{esc(s['title'])}</td>
        <td class="num">−{hms(s['saved_sec'])}</td><td>{esc(s['why'])}</td>
        <td>{"적용됨" if s['applied'] else "theta_ce 완료 후"}</td></tr>'''
        for s in d["skipped"])
    gates = d["gates"]
    gate_html = ""
    if gates:
        parts = []
        if "g_nh_r15" in gates:
            c = "#1a7f37" if gates["g_nh_r15"] == "PASS" else "#cf222e"
            parts.append(f'<span class="pill" style="background:{c}">G-NH r15: {gates["g_nh_r15"]}</span>')
        if gates.get("r15_verdict"):
            parts.append(f'<span class="gv">{esc(gates["r15_verdict"])}</span>')
        if gates.get("strip_delta_acc_pp") is not None:
            parts.append(f'<span class="gv">strip Δacc {gates["strip_delta_acc_pp"]}pp CI{gates.get("strip_ci")}</span>')
        gate_html = "<div class='gates'>" + " ".join(parts) + "</div>"
    cur = d["current"]
    cur_line = "체인 대기/완료" if not cur else (
        f'<b>{esc(cur["title"])}</b> — {hms(cur["remaining_sec"])} 남음'
        + (f' ({cur["pct"]}%)' if cur["pct"] is not None else ""))
    status_banner = ""
    if d["chain_done"]:
        status_banner = '<div class="banner ok">🎉 전체 체인 완료</div>'
    elif d["stuck"]:
        status_banner = '<div class="banner bad">⚠ CHAIN_STUCK — 개입 필요</div>'
    gen = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(d["generated_at"]))
    saved_now = hms(d["saved_sec_applied"])
    saved_max = hms(d["saved_sec_max"])
    return f'''<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta http-equiv="refresh" content="30">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>cesft_v2 최적화 체인</title>
<style>
:root{{color-scheme:light dark}}
body{{font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;margin:0;padding:22px;
  background:#0d1117;color:#e6edf3}}
h1{{font-size:19px;margin:0 0 2px}} .sub{{color:#8b949e;font-size:12px;margin-bottom:16px}}
.cards{{display:flex;gap:14px;flex-wrap:wrap;margin-bottom:18px}}
.card{{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:14px 18px;min-width:150px}}
.card .k{{color:#8b949e;font-size:11px;text-transform:uppercase;letter-spacing:.5px}}
.card .v{{font-size:26px;font-weight:700;margin-top:4px}}
.card .v.blue{{color:#58a6ff}} .card .v.green{{color:#3fb950}} .card .v.amber{{color:#d29922}}
table{{border-collapse:collapse;width:100%;margin:10px 0 22px;font-size:13px}}
th,td{{text-align:left;padding:8px 10px;border-bottom:1px solid #21262d;vertical-align:top}}
th{{color:#8b949e;font-weight:600;font-size:11px;text-transform:uppercase}}
td.num{{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}}
td.ph{{color:#8b949e;font-weight:700;width:34px}}
tr.done{{opacity:.62}} tr.running{{background:#132132}}
.pill{{color:#fff;padding:2px 9px;border-radius:20px;font-size:11px;font-weight:600}}
.note{{color:#d29922;font-size:11px}} .met{{color:#8b949e;font-size:11px;display:block}}
.bar{{height:5px;background:#21262d;border-radius:4px;margin:6px 0 2px;overflow:hidden}}
.fill{{height:100%;background:#58a6ff}}
tr.off{{opacity:.5}} .gates{{margin:6px 0 18px}} .gv{{color:#8b949e;font-size:12px;margin-left:8px}}
.banner{{padding:10px 14px;border-radius:8px;margin-bottom:14px;font-weight:600}}
.banner.ok{{background:#132e1c;color:#3fb950}} .banner.bad{{background:#3d1418;color:#f85149}}
h2{{font-size:13px;color:#8b949e;text-transform:uppercase;letter-spacing:.5px;margin:20px 0 4px}}
</style></head><body>
<h1>cesft_v2 최적화 체인 — 진행 현황</h1>
<div class="sub">30초 자동 새로고침 · 생성 {gen} · {esc(d["run"])}</div>
{status_banner}
<div class="cards">
  <div class="card"><div class="k">총 잔여(예상)</div><div class="v blue">{hms(d["total_remaining_sec"])}</div></div>
  <div class="card"><div class="k">현재 단계</div><div class="v amber" style="font-size:15px;line-height:1.4">{cur_line}</div></div>
  <div class="card"><div class="k">절약(적용/최대)</div><div class="v green">−{saved_now}</div>
     <div class="k" style="margin-top:2px">최대 −{saved_max}</div></div>
</div>
{gate_html}
<h2>최적화 직렬 체인 (핸드오프 §4)</h2>
<table><thead><tr><th></th><th>스테이지</th><th>상태</th><th>진행</th><th>잔여</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
<h2>조건부 (게이트 통과 시에만 · 총 잔여 별도)</h2>
<table><thead><tr><th>스테이지</th><th>추가</th><th>조건</th><th>상태</th></tr></thead><tbody>{cond}</tbody></table>
<h2>스킵/대체로 절약</h2>
<table><thead><tr><th>스테이지</th><th>절약</th><th>대체 근거</th><th>상태</th></tr></thead><tbody>{skip}</tbody></table>
</body></html>'''


def main():
    d = build()
    (RUN / "optim_progress.json").write_text(json.dumps(d, ensure_ascii=False, indent=1))
    (RUN / "optim_progress.html").write_text(render_html(d))
    print(f"[render] total_remaining={hms(d['total_remaining_sec'])} "
          f"saved={hms(d['saved_sec_applied'])} current={d['current']['key'] if d['current'] else None}")


if __name__ == "__main__":
    main()
