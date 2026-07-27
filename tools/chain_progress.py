#!/usr/bin/env python3
"""체인 v3 진행/잔여시간 — **실측 기반**. 무인 갱신용.

추정을 쓰지 않는다: 같은 종류의 완료된 셀 실측 소요시간으로 대기 셀을 재보정한다.
표본이 아직 없는 종류만 초기 상수를 쓰고, 그 사실을 UI 에 표시한다.

출력: runs/chain_progress.json + runs/chain_progress.html (자체 완결 · 클라이언트 카운트다운)
"""
from __future__ import annotations

import html
import json
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RC = ROOT / "runs" / "cesft_v2_fp_curve"
FG = ROOT / "runs" / "cesft_v2_fp_fg2"
LN = ROOT / "runs" / "cesft_v2_fp_lenient"
LOGS = [ROOT / "runs" / "step_curve_boot.log", ROOT / "runs" / "chain_v3.log"]

# 초기 상수 (분) — 실측이 하나라도 생기면 그 종류는 실측 평균으로 대체된다.
PRIOR = {"battery": 9.5, "freegen": 7.0, "belief": 19.0, "plan": 1.5}

CL = ROOT / "runs" / "chain_v3_claims"


# (그룹, 라벨, 종류, 산출물 경로, 워커 셀 이름|None)
def cells() -> list[tuple]:
    out = []
    for s in (100, 200, 300, 400, 500):
        out.append(("정확도 곡선 · Prospection", f"pro_s{s}", "battery", RC / "eval" / f"pro_s{s}.json", None))
    out.append(("정확도 곡선 · Prospection", "pro_final", "battery", RC / "eval" / "pro_final.json", None))
    for s in (100, 200, 300, 400, 500):
        out.append(("정확도 곡선 · Answer-Only", f"ans_s{s}", "battery", RC / "eval" / f"ans_s{s}.json", None))
    out.append(("정확도 곡선 · Answer-Only", "ans_final", "battery", RC / "eval" / "ans_final.json", None))
    for s in (100, 200, 300):
        out.append(("정확도 곡선 · Retrospection", f"retro_s{s}", "battery", RC / "eval" / f"retro_s{s}.json", None))
    out.append(("정확도 곡선 · Retrospection", "retro_final", "battery", RC / "eval" / "retro_final.json", None))
    for a in ("base", "cand_free", "theta_ce", "sft_r15_c"):
        out.append(("A · freegen 파싱 수정 재실행", a, "freegen",
                    FG / "eval" / f"freegen_{a}_cand_free.json", f"A_fg_{a}"))
    for t in ("r00_s100", "r00_s200", "r00_final"):
        out.append(("B · ρ=0 정확도 3점", t, "battery", RC / "eval" / f"{t}.json", f"B_{t}"))
    out.append(("C · belief 인과 곡선", "plan", "plan", RC / "eval" / "harden_paired_plan.json", "C_plan"))
    for t in ("base", "pro_final", "retro_final", "retro_s100", "pro_s200", "retro_s200", "r00_final"):
        out.append(("C · belief 인과 곡선", t, "belief", RC / "eval" / f"{t}.harden_paired.json", f"C_b_{t}"))
    for a in ("base", "cand_free", "theta_ce", "sft_r15", "sft_r15_c"):
        out.append(("D · 파싱 robustness (선택)", a, "battery", LN / "eval" / f"{a}.json", f"D_batt_{a}"))
    return out


def starts() -> list[float]:
    """로그의 타임스탬프들 — 완료 셀의 소요시간을 로그 간격으로 잡기 위해."""
    ts = []
    for lg in LOGS:
        if not lg.exists():
            continue
        for line in lg.read_text(errors="replace").splitlines():
            m = re.match(r"\[(?:CURVE|V3) (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\]", line)
            if m:
                ts.append(time.mktime(time.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")))
    return sorted(ts)


def claims() -> dict:
    """워커가 남긴 claim — 어느 서버가 어떤 셀을 잡았는지. 완료 후에도 남아 귀속을 보존한다."""
    out = {}
    if not CL.exists():
        return out
    for d in CL.glob("*.claim"):
        h = (d / "host").read_text().strip() if (d / "host").exists() else "?"
        st = float((d / "started").read_text().strip()) if (d / "started").exists() else None
        out[d.name[:-6]] = {"host": h, "started": st}
    return out


def heartbeats() -> dict:
    """서버별 생존 — 워커가 30초마다 갱신한다."""
    out = {}
    if not CL.exists():
        return out
    for f in CL.glob("hb.*"):
        try:
            out[f.name[3:]] = float(f.read_text().strip())
        except (ValueError, OSError):
            continue
    return out


def short(h: str) -> str:
    """호스트명을 UI 에 넣을 짧은 라벨로."""
    return (h or "?").split(".")[0][-12:]


def main() -> None:
    now = time.time()
    ts = starts()
    CLAIM, HB = claims(), heartbeats()
    live = {h: now - t for h, t in HB.items() if now - t < 300}      # 5분 내 하트비트 = 살아있음
    rows, done_min = [], {k: [] for k in PRIOR}
    prev_mtime = None
    for group, label, kind, path, cell in cells():
        ok = path.exists() and path.stat().st_size > 0
        mt = path.stat().st_mtime if ok else None
        cl = CLAIM.get(cell) if cell else None
        dur = None
        if ok and cl and cl["started"] and 0 < mt - cl["started"] < 7200:
            dur = (mt - cl["started"]) / 60           # claim 기준 — 두 서버가 섞여도 정확하다
        elif ok and prev_mtime is not None and 0 < mt - prev_mtime < 7200:
            dur = (mt - prev_mtime) / 60
        elif ok and prev_mtime is None:
            before = [t for t in ts if t < mt]
            if before and 0 < mt - before[-1] < 7200:
                dur = (mt - before[-1]) / 60
        if ok:
            prev_mtime = mt
            if dur:
                done_min[kind].append(dur)
        rows.append({"group": group, "label": label, "kind": kind, "done": ok, "cell": cell,
                     "min": round(dur, 1) if dur else None,
                     "host": short(cl["host"]) if cl else None,
                     "at": time.strftime("%H:%M", time.localtime(mt)) if ok else None})

    est = {k: (round(sum(v) / len(v), 1) if v else PRIOR[k]) for k, v in done_min.items()}
    measured = {k: bool(v) for k, v in done_min.items()}
    pending = [r for r in rows if not r["done"]]
    # 진행 중 = claim 은 있는데 산출물이 아직 없는 셀
    for r in pending:
        r["running"] = bool(r["cell"] and r["cell"] in CLAIM)
    running = [r["label"] for r in pending if r["running"]]

    n_srv = max(1, len(live))
    work = sum(est[r["kind"]] for r in pending)
    work -= sum(est[r["kind"]] / 2 for r in pending if r["running"])   # 진행 중은 절반 지났다고 본다
    remain = work / n_srv
    core_pending = [r for r in pending if not r["group"].startswith("D")]
    core_work = (sum(est[r["kind"]] for r in core_pending)
                 - sum(est[r["kind"]] / 2 for r in core_pending if r["running"]))

    out = {
        "generated_at": now,
        "generated_str": time.strftime("%m-%d %H:%M:%S", time.localtime(now)),
        "rows": rows, "est_min": est, "measured": measured,
        "n_done": sum(r["done"] for r in rows), "n_total": len(rows),
        "servers": [{"host": short(h), "age_sec": round(a)} for h, a in sorted(live.items())],
        "n_servers": len(live),
        "remain_min": round(max(0.0, remain), 1),
        "core_remain_min": round(max(0.0, core_work / n_srv), 1),
        "eta": time.strftime("%m-%d %H:%M", time.localtime(now + max(0.0, remain) * 60)),
        "core_eta": time.strftime("%m-%d %H:%M", time.localtime(now + max(0.0, core_work / n_srv) * 60)),
        "running": running,
        "chain_alive": bool(live),
    }
    (ROOT / "runs" / "chain_progress.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    (ROOT / "runs" / "chain_progress.html").write_text(render(out), encoding="utf-8")
    print(f"[progress] {out['n_done']}/{out['n_total']} · 서버 {out['n_servers']}대 "
          f"· 잔여 {out['remain_min']:.0f}분 · ETA {out['eta']}")


def render(d: dict) -> str:
    # 서버별 색 — 등장 순서대로 1,2,3… (호스트명을 UI 에 하드코딩하지 않는다)
    seen = []
    for r in d["rows"]:
        if r.get("host") and r["host"] not in seen:
            seen.append(r["host"])
    global HUE
    HUE = {h: i % 3 + 1 for i, h in enumerate(seen)}
    groups: dict[str, list] = {}
    for r in d["rows"]:
        groups.setdefault(r["group"], []).append(r)
    body = []
    for g, rs in groups.items():
        nd = sum(r["done"] for r in rs)
        body.append(f'<section><h2>{html.escape(g)}<span class="frac">{nd}/{len(rs)}</span></h2><ul>')
        for r in rs:
            run = (not r["done"]) and r.get("running")
            cls = "done" if r["done"] else ("run" if run else "wait")
            mark = "✓" if r["done"] else ("▶" if run else "·")
            meta = (f'{r["at"]} · {r["min"]:.0f}분' if r["done"] and r["min"]
                    else (r["at"] if r["done"] else
                          (f'진행 중 (~{d["est_min"][r["kind"]]:.0f}분)' if run
                           else f'대기 ~{d["est_min"][r["kind"]]:.0f}분')))
            hs = (f'<span class="h h{HUE.get(r["host"], 0)}">{html.escape(r["host"])}</span>'
                  if r.get("host") else '<span class="h hx">—</span>')
            body.append(f'<li class="{cls}"><span class="m">{mark}</span>'
                        f'<span class="l">{html.escape(r["label"])}</span>{hs}'
                        f'<span class="t">{html.escape(str(meta))}</span></li>')
        body.append("</ul></section>")
    src = ", ".join(f'{k} {v:.0f}분{"(실측)" if d["measured"][k] else "(초기값)"}'
                    for k, v in d["est_min"].items())
    if d["servers"]:
        srv = "".join(f'<span class="chip h{HUE.get(s["host"], 0)}">{html.escape(s["host"])}'
                      f'<i>{s["age_sec"]}초 전</i></span>' for s in d["servers"])
    else:
        srv = '<span class="chip hx">가동 중인 워커 없음</span>'
    # 서버별 담당 셀 수 (완료 기준)
    tally = {}
    for r in d["rows"]:
        if r["done"] and r.get("host"):
            tally[r["host"]] = tally.get(r["host"], 0) + 1
    tal = " · ".join(f'{h} {n}셀' for h, n in sorted(tally.items())) or "귀속 기록 없음"
    return f"""<title>EGO 체인 v3 진행</title>
<style>
:root{{--bg:#faf9f7;--fg:#1c1b19;--dim:#6f6a62;--line:#e3dfd8;--ok:#2f6f4f;--run:#a8621b;--card:#fff;
      --s1:#2a5d8f;--s2:#8a4a86;--s3:#7a6420}}
@media(prefers-color-scheme:dark){{:root{{--bg:#16151a;--fg:#eceaf2;--dim:#918c9e;--line:#2c2a33;--ok:#63c295;--run:#e2a35c;--card:#1e1d24;--s1:#6aa8e0;--s2:#c98ac4;--s3:#c9ae5a}}}}
:root[data-theme=dark]{{--bg:#16151a;--fg:#eceaf2;--dim:#918c9e;--line:#2c2a33;--ok:#63c295;--run:#e2a35c;--card:#1e1d24;--s1:#6aa8e0;--s2:#c98ac4;--s3:#c9ae5a}}
:root[data-theme=light]{{--bg:#faf9f7;--fg:#1c1b19;--dim:#6f6a62;--line:#e3dfd8;--ok:#2f6f4f;--run:#a8621b;--card:#fff;--s1:#2a5d8f;--s2:#8a4a86;--s3:#7a6420}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 ui-sans-serif,-apple-system,"Segoe UI",sans-serif}}
.wrap{{max-width:760px;margin:0 auto;padding:32px 20px 64px}}
h1{{font-size:20px;margin:0 0 4px;letter-spacing:-.01em}}
.sub{{color:var(--dim);font-size:13px;margin-bottom:24px}}
.hero{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:20px;margin-bottom:8px;display:flex;gap:28px;flex-wrap:wrap}}
.hero div{{flex:1;min-width:120px}}
.big{{font-size:30px;font-weight:600;font-variant-numeric:tabular-nums;letter-spacing:-.02em}}
.cap{{font-size:11px;text-transform:uppercase;letter-spacing:.08em;color:var(--dim)}}
.bar{{height:6px;background:var(--line);border-radius:3px;overflow:hidden;margin:14px 0 26px}}
.bar i{{display:block;height:100%;background:var(--ok);width:{100*d['n_done']/max(1,d['n_total']):.1f}%}}
section{{margin:0 0 22px}}
h2{{font-size:13px;text-transform:uppercase;letter-spacing:.07em;color:var(--dim);
   border-bottom:1px solid var(--line);padding-bottom:6px;margin:0 0 8px;display:flex}}
.frac{{margin-left:auto;font-variant-numeric:tabular-nums}}
ul{{list-style:none;margin:0;padding:0}}
li{{display:flex;gap:10px;padding:5px 2px;font-size:14px;align-items:baseline}}
.m{{width:14px;color:var(--dim)}}
li.done .m{{color:var(--ok)}} li.run .m{{color:var(--run)}}
li.wait{{color:var(--dim)}}
li.run{{color:var(--run);font-weight:600}}
.l{{flex:1;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:13px}}
.h{{font-size:10px;letter-spacing:.04em;padding:1px 6px;border-radius:9px;font-weight:600;
   border:1px solid currentColor;opacity:.85;white-space:nowrap}}
.h1{{color:var(--s1)}} .h2{{color:var(--s2)}} .h3{{color:var(--s3)}}
.hx{{color:var(--line);border-color:transparent}}
.chip{{display:inline-flex;gap:6px;align-items:baseline;font-size:12px;font-weight:600;
      border:1px solid currentColor;border-radius:11px;padding:2px 9px;margin-right:8px}}
.chip i{{font-style:normal;font-weight:400;opacity:.7;font-variant-numeric:tabular-nums}}
.t{{color:var(--dim);font-size:12px;font-variant-numeric:tabular-nums;font-weight:400}}
footer{{margin-top:32px;padding-top:14px;border-top:1px solid var(--line);color:var(--dim);font-size:12px}}
</style>
<div class="wrap">
<h1>EGO 체인 v3 — 진행</h1>
<div class="sub">{d['generated_str']} 기준 · 셀 완료 시각(mtime) 실측 · 서버 {d['n_servers']}대 병렬</div>
<div class="hero">
 <div><div class="cap">완료</div><div class="big">{d['n_done']}<span style="font-size:17px;color:var(--dim)">/{d['n_total']}</span></div></div>
 <div><div class="cap">잔여</div><div class="big" id="rem">{d['remain_min']:.0f}분</div></div>
 <div><div class="cap">종료 예상</div><div class="big">{d['eta']}</div>
     <div style="font-size:11px;color:var(--dim);margin-top:2px">핵심(D 제외) {d['core_eta']}</div></div>
</div>
<div style="margin:12px 0 0">{srv}</div>
<div style="font-size:12px;color:var(--dim);margin:8px 0 0">서버별 완료: {html.escape(tal)}</div>
<div class="bar"><i></i></div>
{''.join(body)}
<footer>셀당 소요 추정: {html.escape(src)}<br>
D 단계는 파싱 규칙 robustness 확인용이며 헤드라인 표를 대체하지 않는다. 중단해도 논문 주장에 영향 없음.<br>
잔여 시간은 페이지 열어둔 채로 자동 감소한다. 실제 갱신은 5분마다 파일이 다시 쓰인다.</footer>
</div>
<script>
let rem={d['remain_min']}*60, t0=Date.now();
setInterval(()=>{{let s=Math.max(0,rem-(Date.now()-t0)/1000);
 document.getElementById('rem').textContent=s>=3600
  ?Math.floor(s/3600)+'시간 '+Math.floor(s%3600/60)+'분':Math.ceil(s/60)+'분';}},1000);
</script>"""


if __name__ == "__main__":
    main()
