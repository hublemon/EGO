#!/usr/bin/env python3
"""headroom_artifact.py — teacher headroom 실험을 공유용 정적 HTML 스냅샷으로 굽는다.

로컬 대시보드(tools/headroom_dashboard.py, 포트 7864)는 사설 IP라 밖에서 못 본다.
이 스크립트는 같은 결과 파일을 읽어 **의존성 없는 단일 HTML**을 만든다 — claude.ai
Artifact 로 올리면 계정이 있는 사람이 볼 수 있다.

Artifact 는 CSP 로 외부 요청이 막혀 있어 **라이브가 아니라 스냅샷**이다. 굽는 시각을
페이지에 박고, 갱신하려면 이 스크립트를 다시 돌려 같은 URL 로 재배포한다.

    python tools/headroom_artifact.py --out /path/to/headroom.html
"""
from __future__ import annotations

import argparse
import html
import json
import math
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

RUN = Path("/mnt/nvme/migration/jihun/EGO/runs/headroom")
KST = timezone(timedelta(hours=9))
ARMS = ("present", "future", "shuffled")
LABEL = {"present": "현재 정보만", "future": "실제 미래를 봄", "shuffled": "남의 미래를 봄"}
# ★ 무학습 기준선. 같은 1,370샘플에서 world model 1등을 그대로 따르면 0.7657 이다.
#   이 줄을 처음 판에서 빠뜨려 "가설 지지"만 보이고 teacher 가 기준선보다 33pp 낮다는
#   사실이 가려졌다. 모든 표에 강제로 넣는다.
L0_SAME_SAMPLES = 0.7657
E = html.escape

# 판정에 필요한 최소 표본. n=1,370 에서 이항 SE≈0.013 이고, paired 라 더 작다.
# 그 아래에서는 순위가 노이즈로 뒤집히므로 숫자를 보여주되 판정하지 않는다.
MIN_N = 200


def load():
    meta, recs = {}, []
    p = RUN / "headroom.jsonl"
    if not p.is_file():
        return meta, recs
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "meta" in d:
            meta = d["meta"]
        elif all(a in d for a in ARMS):
            recs.append(d)
    return meta, recs


def boot_paired(pairs, n_boot=5000, seed=0):
    n = len(pairs)
    if n < 2:
        return None
    diffs = [b - a for a, b in pairs]
    rnd = random.Random(seed)
    means = sorted(sum(diffs[rnd.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    pt = sum(diffs) / n
    mu = sum(means) / n_boot
    return {"point": pt, "lo": means[int(0.025 * n_boot)],
            "hi": means[min(n_boot - 1, int(0.975 * n_boot))],
            "se": math.sqrt(sum((x - mu) ** 2 for x in means) / max(1, n_boot - 1)), "n": n}


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
.page{max-width:900px;margin:0 auto;padding:40px 22px 72px;display:flex;flex-direction:column;gap:36px}
.stack{display:flex;flex-direction:column;gap:14px}
.eyebrow{font-family:var(--mono);font-size:11.5px;letter-spacing:.14em;text-transform:uppercase;
 color:var(--dim)}
h1{font-family:var(--serif);font-weight:600;font-size:clamp(27px,5vw,39px);line-height:1.15;
 margin:0;text-wrap:balance;letter-spacing:-.01em}
h2{font-family:var(--serif);font-weight:600;font-size:21px;margin:0;text-wrap:balance}
p{margin:0;max-width:66ch}
.lede{font-size:17px;color:var(--dim);max-width:64ch}
.snap{font-family:var(--mono);font-size:12.5px;color:var(--dim);border-left:2px solid var(--accent);
 padding-left:11px}
.card{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:20px 22px;
 display:flex;flex-direction:column;gap:14px}
.bars{display:flex;flex-direction:column;gap:11px}
.bar{display:grid;grid-template-columns:8.5rem 1fr 4.5rem;gap:13px;align-items:center}
.bar .t{font-size:14px}
.track{background:var(--wash);border-radius:3px;height:24px;position:relative;overflow:hidden}
.fill{position:absolute;inset:0 auto 0 0;background:var(--accent);border-radius:3px}
.bar .v{font-family:var(--mono);font-variant-numeric:tabular-nums;text-align:right;font-size:15px}
.scroll{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:14px}
th,td{text-align:left;padding:8px 14px 8px 0;border-bottom:1px solid var(--line);white-space:nowrap}
thead th{font-family:var(--mono);font-size:11px;letter-spacing:.09em;text-transform:uppercase;
 color:var(--dim);font-weight:600;border-bottom-color:var(--rule)}
td.n,th.n{text-align:right;font-family:var(--mono);font-variant-numeric:tabular-nums}
tbody tr:last-child td{border-bottom:0}
td.note{white-space:normal;color:var(--dim);font-size:13px;min-width:13rem}
.verdict{font-family:var(--mono);font-size:13.5px;padding:12px 14px;border-radius:5px;
 border:1px solid currentColor;line-height:1.5}
.v-good{color:var(--good)}.v-warn{color:var(--warn)}.v-bad{color:var(--crit)}.v-wait{color:var(--dim)}
.big{font-family:var(--mono);font-size:clamp(34px,6vw,46px);font-variant-numeric:tabular-nums;
 line-height:1;letter-spacing:-.02em}
.dim{color:var(--dim)}
.stripe{display:grid;grid-template-columns:3px 1fr;gap:0 14px}
.stripe>.bar2{background:var(--accent);border-radius:2px}
.stripe>.body{display:flex;flex-direction:column;gap:10px}
code{font-family:var(--mono);font-size:.9em;background:var(--wash);padding:1px 5px;border-radius:3px}
footer{font-size:13px;color:var(--dim);max-width:66ch}
:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
@media(prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
"""


def build() -> str:
    now = datetime.now(KST)
    meta, recs = load()
    n = len(recs)
    target = meta.get("n_usable", 0)
    acc = {a: (sum(r[a]["correct"] for r in recs) / n) if n else None for a in ARMS}

    h = [f"<title>미래를 본 teacher는 더 잘 고르는가</title><style>{CSS}</style>",
         '<div class="page">']

    h.append('<header class="stack">')
    h.append('<div class="eyebrow">EGO Step-2 · Retrospection · 1단계 진단</div>')
    h.append("<h1>미래를 본 teacher는 더 잘 고르는가</h1>")
    h.append('<p class="lede">완료된 미래 행동을 알려준 상태에서, 모델이 다섯 개 후보 중 '
             '정답 행동을 더 자주 고르는지 측정한다. 이 차이가 곧 <strong>Retrospection이 '
             '학생 모델에 전달할 수 있는 정보량의 상한</strong>이다 — 차이가 없다면 아무리 '
             '정교하게 가르쳐도 전달할 것이 없다.</p>')
    done = "완료" if (target and n >= target) else f"진행 중 {n}/{target or '—'}"
    h.append(f'<div class="snap">{now:%Y-%m-%d %H:%M} KST 스냅샷 · {E(done)}<br>'
             '이 페이지는 실시간이 아니라 굽는 시점의 정지 화면이다.</div>')
    h.append("</header>")

    if not n:
        h.append('<div class="card"><span class="verdict v-wait">아직 결과 없음</span></div>')
        h.append("</div>")
        return "\n".join(h)

    d_fp = boot_paired([(float(r["present"]["correct"]), float(r["future"]["correct"]))
                        for r in recs])
    d_fs = boot_paired([(float(r["shuffled"]["correct"]), float(r["future"]["correct"]))
                        for r in recs])

    # ── 판정 ──
    if n < MIN_N:
        verdict, vcls = (f"표본 부족 ({n}건) — 판정 보류. 아래 숫자는 참고용이며 이 구간에서는 "
                         f"순위가 노이즈로 쉽게 뒤집힌다. 최소 {MIN_N}건 이후에만 해석한다."), "v-wait"
    elif d_fp["lo"] > 0 and d_fs["lo"] > 0:
        verdict, vcls = ("가설 지지 — 미래 정보가 후보 판별에 실제로 기여한다. "
                         "맥락 길이를 통제한 대조군 대비로도 유의하다. 전달할 정보가 있다."), "v-good"
    elif d_fp["lo"] > 0:
        verdict, vcls = ("주의 — 이득이 미래 정보가 아니라 맥락이 길어진 효과로 설명된다. "
                         "남의 미래를 줘도 비슷하게 오르므로 Retrospection의 근거가 되지 못한다."), "v-warn"
    else:
        verdict, vcls = ("가설 위험 — 미래를 알려줘도 후보 선택이 나아지지 않는다. "
                         "가르칠 정보 자체가 없다는 뜻이므로 설계를 재고해야 한다."), "v-bad"

    h.append('<section class="card"><div class="stripe"><div class="bar2"></div><div class="body">')
    h.append('<div class="eyebrow">핵심 지표 — 미래가 주는 이득</div>')
    h.append(f'<div class="big">{d_fp["point"]:+.3f}</div>')
    h.append('<p class="dim" style="font-size:13.5px">미래를 봤을 때의 정확도 − 못 봤을 때의 '
             '정확도 (같은 샘플 짝지어 비교)</p>')
    h.append(f'<div class="verdict {vcls}">{E(verdict)}</div>')
    h.append("</div></div></section>")

    # ── 세 조건 ──
    h.append('<section class="stack"><h2>세 조건</h2>')
    h.append('<p>같은 샘플에 세 가지 조건을 모두 적용했다. <strong>남의 미래</strong> 조건이 '
             '핵심 대조군이다 — 이게 없으면 "미래를 알아서 좋아진 것"과 "설명이 길어져서 '
             '좋아진 것"을 구분할 수 없다.</p>')
    h.append('<div class="card"><div class="bars">')
    mx = max(list(acc.values()) + [L0_SAME_SAMPLES])
    w0 = L0_SAME_SAMPLES / max(mx, 1e-9) * 100
    h.append(f'<div class="bar"><div class="t">세계 모델 1등 그대로<br>'
             f'<span class="dim" style="font-size:11.5px">무학습 기준선</span></div>'
             f'<div class="track"><div class="fill" style="width:{w0:.1f}%;'
             f'background:var(--crit)"></div></div>'
             f'<div class="v">{L0_SAME_SAMPLES:.3f}</div></div>')
    for a in ARMS:
        w = acc[a] / max(mx, 1e-9) * 100
        h.append(f'<div class="bar"><div class="t">{LABEL[a]}</div>'
                 f'<div class="track"><div class="fill" style="width:{w:.1f}%"></div></div>'
                 f'<div class="v">{acc[a]:.3f}</div></div>')
    h.append("</div>")
    h.append('<div class="scroll"><table><thead><tr><th>비교</th><th class="n">차이</th>'
             '<th class="n">95% 신뢰구간</th><th>무엇을 통제하나</th></tr></thead><tbody>')
    for nm, d, why in (("미래 − 현재", d_fp, "통제 없음 (미래 정보 + 설명 길이)"),
                       ("미래 − 남의 미래", d_fs, "설명 길이를 통제 — 미래 정보만의 효과")):
        h.append(f'<tr><td>{E(nm)}</td><td class="n">{d["point"]:+.4f}</td>'
                 f'<td class="n">[{d["lo"]:+.4f}, {d["hi"]:+.4f}]</td>'
                 f'<td class="note">{E(why)}</td></tr>')
    h.append("</tbody></table></div>")
    h.append('<p class="dim" style="font-size:13.5px"><strong>중요한 단서.</strong> '
             '세 조건 모두 무학습 기준선(0.766)보다 한참 아래다. 미래를 봐서 얻은 +0.044는 '
             '이미 무너진 지점 위에서의 개선이며, 이 상태의 모델을 그대로 학생에게 가르치면 '
             '세계 모델을 그냥 따르는 것보다 못한 것을 가르치게 된다. 이 이득이 잘 학습된 '
             '모델에서도 남는지는 아직 측정되지 않았다.</p>')
    h.append("</div></section>")

    # ── 전이 ──
    w2r = sum(1 for r in recs if not r["present"]["correct"] and r["future"]["correct"])
    r2w = sum(1 for r in recs if r["present"]["correct"] and not r["future"]["correct"])
    h.append('<section class="stack"><h2>미래를 보고 답이 어떻게 바뀌었나</h2>')
    h.append('<p>순이득만 보면 안 된다. 고친 만큼 망친다면 미래는 정보를 준 것이 아니라 '
             '답을 흔든 것이다.</p>')
    h.append('<div class="card"><div class="scroll"><table><thead><tr><th>전이</th>'
             '<th class="n">건수</th><th class="n">비율</th><th>의미</th></tr></thead><tbody>')
    for nm, c, why in (("오답 → 정답", w2r, "미래가 실제로 교정한 사례"),
                       ("정답 → 오답", r2w, "미래가 오히려 흔든 사례"),
                       ("변화 없음", n - w2r - r2w, "")):
        h.append(f'<tr><td>{E(nm)}</td><td class="n">{c}</td><td class="n">{c/n:.3f}</td>'
                 f'<td class="note">{E(why)}</td></tr>')
    h.append(f'</tbody></table></div><p class="dim" style="font-size:13.5px">'
             f'순이득 = {w2r} − {r2w} = <strong>{w2r-r2w:+d}</strong></p></div></section>')

    # ── 방법 ──
    h.append('<section class="stack"><h2>어떻게 재는가</h2>')
    h.append('<p>모델에게 글을 쓰게 하지 않고, 다섯 후보 각각을 끝까지 읽혔을 때의 확률을 '
             '직접 재서 가장 높은 것을 고른 것으로 본다. 세 조건이 완전히 같은 방식이라 '
             '절대값의 치우침은 공통이고 <strong>조건 간 차이만</strong> 해석한다.</p>')
    h.append('<p class="dim" style="font-size:13.5px"><strong>누설 차단이 이 실험의 생명이다.</strong> '
             '미래 목록에 정답 행동이 그대로 들어 있으면 모델이 베끼기만 하면 되므로 측정이 '
             '무의미해진다. 그래서 미래에서 정답과 같은 행동은 <strong>전부</strong> 지웠다. '
             '실제로 첫 샘플의 미래 두 번째 항목이 정답과 동일했다.</p>')
    if meta:
        h.append(f'<p class="dim" style="font-size:13.5px">전체 1,500건 중 '
                 f'{meta.get("n_usable")}건 사용 — 정답이 후보 밖이라 제외 '
                 f'{meta.get("drop_gt_outside")}건, 지운 뒤 미래가 남지 않아 제외 '
                 f'{meta.get("drop_no_future")}건.</p>')
    h.append("</section>")

    h.append('<hr style="border:0;border-top:1px solid var(--rule)">')
    h.append('<footer>EGO Step-2 Retrospection · 런 디렉터리 <code>EGO/runs/headroom</code> · '
             'H200 1대(cuda:1) · 신뢰구간은 같은 샘플을 짝지어 5,000회 부트스트랩한 값이다.</footer>')
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
