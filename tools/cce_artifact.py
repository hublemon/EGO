#!/usr/bin/env python3
"""cce_artifact.py — 2단계 candidate CE 현황을 공유용 정적 HTML 로 굽는다.

로컬 대시보드(tools/cce_dashboard.py, 포트 7865)는 사설 IP라 밖에서 못 본다.
Artifact 는 CSP 로 외부 요청이 막혀 있어 라이브가 아니라 **스냅샷**이다 — 굽는 시각을
페이지에 박고, 갱신은 이 스크립트를 다시 돌려 같은 URL 로 재배포한다.

    python tools/cce_artifact.py --out /path/to/cce.html
"""
from __future__ import annotations

import argparse
import html
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

RUN = Path("/mnt/nvme/migration/jihun/EGO/runs/cce")
TRAIN_LOG = Path("/mnt/nvme/migration/jihun/EGO_jihun/outputs/step2/cce_1f/gx_log.jsonl")
KST = timezone(timedelta(hours=9))
E = html.escape

# 전부 실측 고정값. 결과를 보고 바꾸지 않는다.
L0 = 0.3994             # WM top-1 무학습, heldout n=1,417
L0_COND = 0.6381        # 그 조건부 환산 = 0.3994 / 0.626
BASE_SCORING = 0.3876   # 학습 전 모델의 후보 스코어링 조건부 정확도 (n=1,370)
PREV_BEST = 0.3380      # 역대 최고 arm (f0gr_final, n=500)
CEILING = 0.6260        # R5 — GT 가 후보 안에 있을 확률
TOTAL = 5000
# 07-19 에 800샘플에서 버려진 동일 objective. 이번 run 의 비교 궤적이다.
REF = [(200, 6.2162, 0.395), (400, 4.949, 0.385), (600, 1.7853, 0.545), (800, 1.4777, 0.450)]

CSS = """
:root{--bg:#eef0f3;--fg:#10151d;--dim:#5a6472;--card:#fff;--line:#d5dae1;--rule:#c3cad3;
 --accent:#3d7f9d;--wash:rgba(61,127,157,.14);--good:#2f8f5b;--warn:#b8801f;--crit:#b8453c;
 --mono:ui-monospace,SFMono-Regular,'SF Mono',Menlo,Consolas,monospace;
 --sans:system-ui,-apple-system,'Segoe UI',Roboto,'Helvetica Neue',sans-serif;
 --serif:'Iowan Old Style','Palatino Linotype',Palatino,'Source Serif 4',Georgia,serif}
@media(prefers-color-scheme:dark){:root{--bg:#10151d;--fg:#e4e8ee;--dim:#96a0ad;--card:#171e28;
 --line:#2a3340;--rule:#333d4b;--accent:#63a8c6;--wash:rgba(99,168,198,.16);--good:#4fb277;
 --warn:#d3a03f;--crit:#d4695f}}
:root[data-theme="dark"]{--bg:#10151d;--fg:#e4e8ee;--dim:#96a0ad;--card:#171e28;--line:#2a3340;
 --rule:#333d4b;--accent:#63a8c6;--wash:rgba(99,168,198,.16);--good:#4fb277;--warn:#d3a03f;
 --crit:#d4695f}
:root[data-theme="light"]{--bg:#eef0f3;--fg:#10151d;--dim:#5a6472;--card:#fff;--line:#d5dae1;
 --rule:#c3cad3;--accent:#3d7f9d;--wash:rgba(61,127,157,.14);--good:#2f8f5b;--warn:#b8801f;
 --crit:#b8453c}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font-family:var(--sans);font-size:16px;
 line-height:1.6;-webkit-font-smoothing:antialiased}
.page{max-width:900px;margin:0 auto;padding:40px 22px 72px;display:flex;flex-direction:column;gap:34px}
.stack{display:flex;flex-direction:column;gap:13px}
.eyebrow{font-family:var(--mono);font-size:11.5px;letter-spacing:.14em;text-transform:uppercase;
 color:var(--dim)}
h1{font-family:var(--serif);font-weight:600;font-size:clamp(26px,4.9vw,38px);line-height:1.15;
 margin:0;text-wrap:balance;letter-spacing:-.01em}
h2{font-family:var(--serif);font-weight:600;font-size:21px;margin:0;text-wrap:balance}
p{margin:0;max-width:66ch}.lede{font-size:17px;color:var(--dim);max-width:64ch}.dim{color:var(--dim)}
.snap{font-family:var(--mono);font-size:12.5px;color:var(--dim);border-left:2px solid var(--accent);
 padding-left:11px}
.card{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:20px 22px;
 display:flex;flex-direction:column;gap:14px}
.gauge{position:relative;height:34px;background:var(--wash);border-radius:4px;margin-top:22px}
.gauge .f{position:absolute;inset:0 auto 0 0;background:var(--accent);border-radius:4px}
.gauge .mk{position:absolute;top:-4px;bottom:-4px;width:2px;background:var(--crit)}
.gauge .lb{position:absolute;top:-22px;font-family:var(--mono);font-size:10.5px;color:var(--crit);
 transform:translateX(-50%);white-space:nowrap}
.scroll{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:14px}
th,td{text-align:left;padding:8px 14px 8px 0;border-bottom:1px solid var(--line);white-space:nowrap}
thead th{font-family:var(--mono);font-size:11px;letter-spacing:.09em;text-transform:uppercase;
 color:var(--dim);font-weight:600;border-bottom-color:var(--rule)}
td.n,th.n{text-align:right;font-family:var(--mono);font-variant-numeric:tabular-nums}
tbody tr:last-child td{border-bottom:0}
td.note{white-space:normal;color:var(--dim);font-size:13px;min-width:13rem}
.big{font-family:var(--mono);font-size:clamp(32px,5.6vw,44px);font-variant-numeric:tabular-nums;
 line-height:1;letter-spacing:-.02em}
.pill{display:inline-block;font-family:var(--mono);font-size:11px;padding:2px 9px;border-radius:10px;
 border:1px solid currentColor;font-weight:600}
.p-run{color:var(--warn)}.p-done{color:var(--good)}.p-wait{color:var(--dim)}.p-bad{color:var(--crit)}
.good{color:var(--good)}.warn{color:var(--warn)}.crit{color:var(--crit)}
.verdict{font-family:var(--mono);font-size:13.5px;padding:12px 14px;border-radius:5px;
 border:1px solid currentColor;line-height:1.5}
svg.chart{width:100%;height:190px;display:block;overflow:visible}
code{font-family:var(--mono);font-size:.9em;background:var(--wash);padding:1px 5px;border-radius:3px}
footer{font-size:13px;color:var(--dim);max-width:66ch}
:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
@media(prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
"""


def read_jsonl(p: Path):
    if not p.is_file():
        return []
    out = []
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def read_json(p: Path):
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def loss_chart(cur: list[tuple[int, float]], ref: list[tuple[int, float]]) -> str:
    """이번 run 과 07-19 참조 run 의 loss 궤적을 같은 축에 겹친다.

    참조 run 은 seen 400~600 에서 급락했다. '학습이 되고 있는가'는 절대값이 아니라
    이 급락 구간을 통과했는지로 읽어야 한다 — 1차 시도는 그 전(300)에서 끊겼다.
    """
    pts = cur + ref
    if len(pts) < 2:
        return ""
    W, H, P = 640, 190, 34
    xs = [x for x, _ in pts]
    ys = [y for _, y in pts]
    x0, x1 = 0, max(max(xs), 800)
    y0, y1 = 0, max(ys) * 1.08
    def sx(v): return P + (v - x0) / max(x1 - x0, 1) * (W - P - 10)
    def sy(v): return H - P - (v - y0) / max(y1 - y0, 1e-9) * (H - P - 12)
    g = [f'<svg class="chart" viewBox="0 0 {W} {H}" role="img" '
         f'aria-label="loss 궤적 비교">']
    # 균등분포 기준선 ln(5)
    import math
    u = math.log(5)
    g.append(f'<line x1="{P}" y1="{sy(u):.1f}" x2="{W-10}" y2="{sy(u):.1f}" '
             f'stroke="var(--rule)" stroke-dasharray="3 3"/>')
    g.append(f'<text x="{W-12}" y="{sy(u)-5:.1f}" text-anchor="end" font-size="10" '
             f'fill="var(--dim)" font-family="var(--mono)">무작위 추측 {u:.2f}</text>')
    for name, series, color, dash in (("07-19 참조 (버려진 run)", ref, "var(--dim)", "5 4"),
                                      ("이번 run", cur, "var(--accent)", "")):
        if len(series) < 2:
            continue
        d = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in series)
        g.append(f'<polyline points="{d}" fill="none" stroke="{color}" stroke-width="2" '
                 f'stroke-linejoin="round" stroke-dasharray="{dash}"/>')
        for x, y in series:
            g.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="2.6" fill="{color}"/>')
    # 급락 구간 표시
    g.append(f'<rect x="{sx(400):.1f}" y="12" width="{sx(600)-sx(400):.1f}" height="{H-P-12}" '
             f'fill="var(--wash)" opacity=".55"/>')
    g.append(f'<text x="{(sx(400)+sx(600))/2:.1f}" y="10" text-anchor="middle" font-size="10" '
             f'fill="var(--dim)" font-family="var(--mono)">참조 run 급락 구간</text>')
    for v in (0, 200, 400, 600, 800):
        g.append(f'<text x="{sx(v):.1f}" y="{H-12}" text-anchor="middle" font-size="10" '
                 f'fill="var(--dim)" font-family="var(--mono)">{v}</text>')
    g.append(f'<text x="4" y="{sy(y1*.5):.1f}" font-size="10" fill="var(--dim)" '
             f'font-family="var(--mono)">loss</text>')
    g.append("</svg>")
    return "".join(g)


def build() -> str:
    now = datetime.now(KST)
    chain = (RUN / "chain.log").read_text(encoding="utf-8", errors="replace") \
        if (RUN / "chain.log").is_file() else ""
    train = read_jsonl(TRAIN_LOG)
    smoke = read_jsonl(RUN / "smoke" / "gx_log.jsonl")
    ev = read_json(RUN / "eval_scored.json")
    ev_b = read_json(RUN / "eval_scored_base.json")

    seen = train[-1].get("seen", 0) if train else 0
    if (RUN / "CHAIN_DONE").is_file():
        phase, pc = "완료", "p-done"
    elif ev:
        phase, pc = "평가 중", "p-run"
    elif train:
        phase, pc = f"학습 중 {seen}/{TOTAL}", "p-run"
    else:
        phase, pc = "시작 대기", "p-wait"

    h = [f"<title>후보 선택을 직접 학습하면 기준선을 넘는가</title><style>{CSS}</style>",
         '<div class="page">']
    h.append('<header class="stack">')
    h.append('<div class="eyebrow">EGO Step-2 · 2단계 candidate CE</div>')
    h.append("<h1>후보 선택을 직접 학습하면 무학습 기준선을 넘는가</h1>")
    h.append('<p class="lede">세계 모델이 제시한 다섯 개 후보 중 정답을 고르는 것 — 그 자체를 '
             '학습 목표로 삼는다. 지금까지 완주한 학습은 모두 <strong>글쓰기</strong>를 '
             '최적화했고 행동 선택은 부산물이었다. 평가 기준을 직접 겨냥한 목표는 이 '
             '프로젝트에서 한 번도 끝까지 학습된 적이 없다.</p>')
    h.append(f'<div class="snap">{now:%Y-%m-%d %H:%M} KST 스냅샷 · '
             f'<span class="pill {pc}">{E(phase)}</span><br>'
             '이 페이지는 실시간이 아니라 굽는 시점의 정지 화면이다.</div>')
    h.append("</header>")

    # ── 결과 게이지 ──
    h.append('<section class="card"><div class="eyebrow">heldout 정확도 — 넘어야 하는 선은 '
             f'{L0}</div>')
    if ev:
        cur = ev.get("acc", 0)
        cls = "good" if cur > L0 else "crit"
        h.append(f'<div class="big {cls}">{cur:.4f}</div>')
        w = min(100.0, cur / CEILING * 100)
        h.append('<div class="gauge">'
                 f'<div class="f" style="width:{w:.1f}%"></div>'
                 f'<div class="mk" style="left:{L0/CEILING*100:.1f}%"></div>'
                 f'<div class="lb" style="left:{L0/CEILING*100:.1f}%">기준선 {L0}</div></div>')
        v = ("기준선 초과 — 이 프로젝트에서 처음이다."
             if cur > L0 else "아직 기준선 미달. 학습의 순효과가 여전히 음수다.")
        h.append(f'<div class="verdict {"v" if False else ("p-done" if cur > L0 else "p-bad")}" '
                 f'style="color:var(--{cls})">{E(v)}</div>')
    else:
        h.append('<p class="dim">학습이 끝난 뒤 측정한다. 평가는 heldout 전량 1,417건.</p>')
        h.append('<div class="gauge">'
                 f'<div class="mk" style="left:{L0/CEILING*100:.1f}%"></div>'
                 f'<div class="lb" style="left:{L0/CEILING*100:.1f}%">기준선 {L0}</div></div>')
    h.append(f'<p class="dim" style="font-size:12.5px">가로축은 0부터 구조적 상한 {CEILING}까지. '
             '상한은 정답이 후보 다섯 개 안에 들어 있을 확률이며, 후보 안에서 고르는 방식으로는 '
             '그 위로 갈 수 없다.</p>')
    h.append("</section>")

    # ── 학습 궤적 ──
    if train or smoke:
        cur_pts = [(r["seen"], r["loss"]) for r in train if "loss" in r]
        h.append('<section class="stack"><h2>학습이 실제로 되고 있는가</h2>')
        h.append('<p>절대값이 아니라 <strong>급락 구간을 통과했는지</strong>로 읽어야 한다. '
                 '07-19에 같은 목표로 시작했다가 버려진 실험이 있는데, 그 실험의 loss는 '
                 '600번째 샘플 부근에서 갑자기 떨어졌다. 이번 실험의 1차 시도는 그 지점 '
                 '이전인 300에서 중단 판정을 받아 끊겼다 — 판정 자체가 너무 일렀다.</p>')
        h.append('<div class="card">')
        h.append(loss_chart(cur_pts, [(s, l) for s, l, _ in REF]))
        h.append('<div class="scroll"><table><thead><tr><th class="n">샘플</th>'
                 '<th class="n">이번 loss</th><th class="n">이번 정확도</th>'
                 '<th class="n">참조 loss</th><th class="n">참조 정확도</th>'
                 '</tr></thead><tbody>')
        ref_d = {s: (l, a) for s, l, a in REF}
        keys = sorted({r["seen"] for r in train} | set(ref_d))[:14]
        cur_d = {r["seen"]: (r.get("loss"), r.get("train_acc")) for r in train}
        for k in keys:
            cl, ca = cur_d.get(k, (None, None))
            rl, ra = ref_d.get(k, (None, None))
            f = lambda v, n=4: (f"{v:.{n}f}" if isinstance(v, (int, float)) else "—")
            h.append(f'<tr><td class="n">{k}</td><td class="n">{f(cl,3)}</td>'
                     f'<td class="n">{f(ca,3)}</td><td class="n dim">{f(rl,3)}</td>'
                     f'<td class="n dim">{f(ra,3)}</td></tr>')
        h.append("</tbody></table></div></div></section>")

    # ── 기준선 ──
    h.append('<section class="stack"><h2>기준선 — 전부 실측</h2>')
    h.append('<p>이 표의 첫 줄이 핵심이다. 학습 없이 세계 모델의 1등을 그대로 따르기만 해도 '
             '0.3994인데, <strong>이 프로젝트에서 학습된 어떤 모델도 아직 이걸 넘지 못했다.</strong> '
             '이 줄을 결과표에 넣지 않아서 그 사실이 오래 드러나지 않았다.</p>')
    h.append('<div class="card"><div class="scroll"><table><thead><tr><th>기준</th>'
             '<th class="n">전체 정확도</th><th>무엇인가</th></tr></thead><tbody>')
    rows = [("세계 모델 1등 그대로 (무학습)", L0, "넘어야 하는 선"),
            ("역대 최고 학습 모델", PREV_BEST, "f0gr_final · 기준선 미달"),
            ("구조적 상한", CEILING, "정답이 후보 안에 있을 확률")]
    for nm, v, why in rows:
        h.append(f'<tr><td>{E(nm)}</td><td class="n">{v:.4f}</td>'
                 f'<td class="note">{E(why)}</td></tr>')
    if ev_b:
        h.append(f'<tr><td>학습 전 모델 (같은 평가 경로)</td>'
                 f'<td class="n">{ev_b.get("acc","—")}</td>'
                 f'<td class="note">대조군 — 학습의 순효과를 재기 위한 것</td></tr>')
    if ev:
        cls = "good" if ev.get("acc", 0) > L0 else "crit"
        h.append(f'<tr><td><strong>이번 학습 결과</strong></td>'
                 f'<td class="n {cls}">{ev.get("acc","—")}</td>'
                 f'<td class="note">후보 안에 정답이 있을 때 맞히는 비율 '
                 f'{ev.get("conditional_acc","—")}</td></tr>')
    h.append("</tbody></table></div></div></section>")

    # ── G1/G2 ──
    if ev and ev_b:
        h.append('<section class="stack"><h2>어디가 움직였나</h2>')
        h.append('<p>전체 정확도 하나만 보면 두 가지 상반된 변화가 상쇄돼 보이지 않는다. '
                 '그래서 항상 나눠서 본다.</p>')
        h.append('<div class="card"><div class="scroll"><table><thead><tr><th>구간</th>'
                 '<th class="n">학습 전</th><th class="n">학습 후</th><th>의미</th>'
                 '</tr></thead><tbody>')
        for k, nm, why in (("g1_retention", "이미 맞던 것 지키기",
                            "세계 모델 1등이 정답인 경우 — 망치지 않아야 한다"),
                           ("g2_correction", "틀린 것 고치기",
                            "세계 모델 1등이 오답인 경우 — 정답으로 바꿔야 한다"),
                           ("conditional_acc", "후보 안에서의 정확도",
                            "정답이 후보에 있을 때 골라내는 비율")):
            h.append(f'<tr><td>{E(nm)}</td><td class="n">{ev_b.get(k,"—")}</td>'
                     f'<td class="n">{ev.get(k,"—")}</td><td class="note">{E(why)}</td></tr>')
        h.append("</tbody></table></div></div></section>")

    if chain:
        last = [x for x in chain.splitlines() if "]" in x][-6:]
        h.append('<section class="stack"><h2>실행 기록</h2><div class="card">'
                 '<div class="scroll"><table><tbody>')
        for x in last:
            h.append(f'<tr><td class="note" style="min-width:0">{E(x)}</td></tr>')
        h.append("</tbody></table></div></div></section>")

    h.append('<hr style="border:0;border-top:1px solid var(--rule)">')
    h.append('<footer>EGO Step-2 · 런 디렉터리 <code>EGO/runs/cce</code> · H200 1대(cuda:1) · '
             'heldout 전량 1,417건 · 학습 5,000샘플.</footer>')
    h.append("</div>")
    return "\n".join(h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(build(), encoding="utf-8")
    print(f"[done] {p} ({p.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
