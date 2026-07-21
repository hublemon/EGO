#!/usr/bin/env python3
"""p3_cons_artifact.py — P3 실행 상태를 공유용 정적 HTML 스냅샷으로 굽는다.

로컬 대시보드(tools/p3_cons_dashboard.py, 포트 7863)는 사설 IP라 밖에서 못 본다.
이 스크립트는 같은 런 디렉터리를 읽어 **의존성 없는 단일 HTML**을 만든다 — claude.ai
Artifact 로 올리면 계정이 있는 사람이 볼 수 있다.

Artifact 는 외부 요청이 CSP 로 차단되므로 이 페이지는 **라이브가 아니라 스냅샷**이다.
그래서 굽는 시각을 페이지 안에 크게 박고, 갱신하려면 이 스크립트를 다시 돌려 재배포한다.

    python tools/p3_cons_artifact.py --out /path/to/p3_status.html
"""
from __future__ import annotations

import argparse
import html
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

RUN = Path("/mnt/nvme/migration/jihun/EGO/runs/p3_cons")
PREV = Path("/mnt/nvme/migration/jihun/EGO/runs/retro_overnight")
KST = timezone(timedelta(hours=9))

# 어제(2026-07-21) 실측 — 이 페이지의 비교 기준. 고정값이다.
BASE = {
    "gt": {"acc": 0.2371, "c3": 0.0135, "c3x": 0.0137, "restate": 0.0191,
           "ci": (0.0080, 0.0202)},
    "wm": {"acc": 0.2484, "c3": 0.0255, "c3x": 0.0260, "restate": 0.0722,
           "ci": (0.0168, 0.0359)},
    "wm_top1": 0.3994,
}
E = html.escape


def read_jsonl(p: Path) -> list[dict]:
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


def read_json(p: Path) -> dict:
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def sparkline(vals: list[float], w: int = 300, h: int = 44) -> str:
    """면 채움 + 끝점 강조. 값이 1개 이하면 그리지 않는다(가짜 추세를 만들지 않기 위해)."""
    if len(vals) < 2:
        return ""
    lo, hi = min(vals), max(vals)
    rng = (hi - lo) or 1.0
    pad = 3
    pts = [(pad + i * (w - 2 * pad) / (len(vals) - 1),
            h - pad - (v - lo) / rng * (h - 2 * pad)) for i, v in enumerate(vals)]
    line = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    area = f"{pts[0][0]:.1f},{h} " + line + f" {pts[-1][0]:.1f},{h}"
    ex, ey = pts[-1]
    return (f'<svg class="spark" viewBox="0 0 {w} {h}" role="img" '
            f'aria-label="cons_loss 추이 {lo:.2f}에서 {hi:.2f} 사이">'
            f'<polygon points="{area}" fill="var(--accent-wash)"></polygon>'
            f'<polyline points="{line}" fill="none" stroke="var(--accent)" '
            f'stroke-width="1.6" stroke-linejoin="round"></polyline>'
            f'<circle cx="{ex:.1f}" cy="{ey:.1f}" r="2.8" fill="var(--accent)"></circle></svg>')


def ols_slope(y: list[float]) -> float | None:
    """관측점당 평균 변화량. 마지막 한 점에 좌우되지 않는 추세 추정."""
    n = len(y)
    if n < 3:
        return None
    xm, ym = (n - 1) / 2, sum(y) / n
    den = sum((i - xm) ** 2 for i in range(n))
    return (sum((i - xm) * (v - ym) for i, v in enumerate(y)) / den) if den else None


def smoke_arms() -> list[dict]:
    arms = []
    for d in sorted(x for x in RUN.glob("smoke_cw*") if x.is_dir()):
        rows = read_jsonl(d / "gr_log.jsonl")
        cons = [r["cons_loss"] for r in rows if r.get("cons_loss") is not None]
        rew = [r["reward_ma"] for r in rows if r.get("reward_ma") is not None]
        half = len(cons) // 2
        arms.append({
            "cw": d.name.replace("smoke_cw", ""),
            "seen": rows[-1].get("seen") if rows else 0,
            "cons": cons, "rew": rew,
            "drop": (cons[0] - cons[-1]) if len(cons) >= 2 else None,
            "slope": ols_slope(cons),
            "halfdiff": ((sum(cons[half:]) / len(cons[half:]))
                         - (sum(cons[:half]) / len(cons[:half]))) if half else None,
            "amp": (max(cons) - min(cons)) if cons else None,
            "done": (d / "checkpoint-final").is_dir(),
        })
    return arms


def fmt(v, nd=4, dash="—"):
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) else dash


CSS = """
:root{
  --ink:#10151d; --paper:#eef0f3; --slate:#5a6472;
  --accent:#3d7f9d; --accent-wash:rgba(61,127,157,.16);
  --good:#2f8f5b; --warn:#b8801f; --crit:#b8453c;
  --bg:var(--paper); --fg:#10151d; --dim:#5a6472;
  --card:#ffffff; --line:#d5dae1; --rule:#c3cad3;
  --serif:'Iowan Old Style','Palatino Linotype',Palatino,'Source Serif 4',Georgia,serif;
  --sans:system-ui,-apple-system,'Segoe UI',Roboto,'Helvetica Neue',sans-serif;
  --mono:ui-monospace,SFMono-Regular,'SF Mono',Menlo,Consolas,monospace;
}
@media (prefers-color-scheme:dark){
  :root{--bg:#10151d;--fg:#e4e8ee;--dim:#96a0ad;--card:#171e28;--line:#2a3340;--rule:#333d4b;
        --accent:#63a8c6;--accent-wash:rgba(99,168,198,.16);
        --good:#4fb277;--warn:#d3a03f;--crit:#d4695f;}
}
:root[data-theme="dark"]{--bg:#10151d;--fg:#e4e8ee;--dim:#96a0ad;--card:#171e28;--line:#2a3340;
  --rule:#333d4b;--accent:#63a8c6;--accent-wash:rgba(99,168,198,.16);
  --good:#4fb277;--warn:#d3a03f;--crit:#d4695f;}
:root[data-theme="light"]{--bg:#eef0f3;--fg:#10151d;--dim:#5a6472;--card:#fff;--line:#d5dae1;
  --rule:#c3cad3;--accent:#3d7f9d;--accent-wash:rgba(61,127,157,.16);
  --good:#2f8f5b;--warn:#b8801f;--crit:#b8453c;}

*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font-family:var(--sans);
  font-size:16px;line-height:1.6;-webkit-font-smoothing:antialiased}
.page{max-width:920px;margin:0 auto;padding:40px 22px 72px;display:flex;flex-direction:column;gap:38px}
.stack{display:flex;flex-direction:column;gap:14px}
.eyebrow{font-family:var(--mono);font-size:11.5px;letter-spacing:.14em;text-transform:uppercase;
  color:var(--dim)}
h1{font-family:var(--serif);font-weight:600;font-size:clamp(28px,5vw,40px);line-height:1.15;
  margin:0;text-wrap:balance;letter-spacing:-.01em}
h2{font-family:var(--serif);font-weight:600;font-size:22px;margin:0;text-wrap:balance}
p{margin:0;max-width:66ch}
.lede{font-size:17.5px;color:var(--dim);max-width:64ch}
.snapshot{font-family:var(--mono);font-size:12.5px;color:var(--dim);
  border-left:2px solid var(--accent);padding-left:11px}
hr{border:0;border-top:1px solid var(--rule);margin:0}

.card{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:20px 22px;
  display:flex;flex-direction:column;gap:14px}
.headline{display:flex;flex-wrap:wrap;align-items:baseline;gap:10px 20px}
.figure{font-family:var(--mono);font-size:clamp(38px,7vw,54px);line-height:1;
  font-variant-numeric:tabular-nums;letter-spacing:-.02em}
.figure .unit{font-size:16px;color:var(--dim);letter-spacing:0}
.spark{width:100%;max-width:320px;height:44px;display:block}

table{border-collapse:collapse;width:100%;font-size:14px}
.scroll{overflow-x:auto;-webkit-overflow-scrolling:touch}
th,td{text-align:left;padding:8px 14px 8px 0;border-bottom:1px solid var(--line);
  white-space:nowrap;vertical-align:baseline}
thead th{font-family:var(--mono);font-size:11px;letter-spacing:.09em;text-transform:uppercase;
  color:var(--dim);font-weight:600;border-bottom-color:var(--rule)}
td.n,th.n{text-align:right;font-family:var(--mono);font-variant-numeric:tabular-nums}
tbody tr:last-child td{border-bottom:0}
td.note{white-space:normal;color:var(--dim);font-size:13px;min-width:15rem}

.pill{display:inline-block;font-family:var(--mono);font-size:11px;letter-spacing:.05em;
  padding:2px 9px;border-radius:10px;border:1px solid currentColor;font-weight:600}
.p-run{color:var(--warn)} .p-done{color:var(--good)} .p-wait{color:var(--dim)}
.good{color:var(--good)} .warn{color:var(--warn)} .crit{color:var(--crit)}
.dim{color:var(--dim)}

.stripe{display:grid;grid-template-columns:3px 1fr;gap:0 14px;align-items:start}
.stripe>.bar{background:var(--accent);border-radius:2px;align-self:stretch}
.stripe>.body{display:flex;flex-direction:column;gap:8px}

.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:14px}
.crit-list{display:flex;flex-direction:column;gap:12px;margin:0;padding:0;list-style:none}
.crit-list li{display:grid;grid-template-columns:auto 1fr;gap:12px;align-items:baseline;
  font-size:14.5px}
.crit-list .mark{font-family:var(--mono);color:var(--accent);font-size:12px}
code{font-family:var(--mono);font-size:.9em;background:var(--accent-wash);padding:1px 5px;
  border-radius:3px}
footer{font-size:13px;color:var(--dim);max-width:66ch}
:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
@media (prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
"""


def build() -> str:
    now = datetime.now(KST)
    arms = smoke_arms()
    ev = read_json(RUN / "eval.json")
    rc = read_json(RUN / "recount.json")
    chosen = ((RUN / "chosen_cw.txt").read_text(encoding="utf-8").strip()
              if (RUN / "chosen_cw.txt").is_file() else "")
    full_rows = read_jsonl(RUN / "full" / "gr_log.jsonl")
    crash = read_jsonl(RUN / "crashcheck" / "gr_log.jsonl")

    # 지금 살아 있는 단계. 학습 로그는 200샘플마다 찍혀 최대 10분 늦으므로
    # 로그 유무가 아니라 **체인이 선언한 단계**를 기준으로 판정한다.
    chain = ((RUN / "chain.log").read_text(encoding="utf-8", errors="replace")
             if (RUN / "chain.log").is_file() else "")
    if (RUN / "CHAIN_DONE").is_file():
        phase, pcls = "완료", "p-done"
    elif (RUN / "CHAIN_FAILED").is_file():
        phase, pcls = "중단됨", "p-run"
    elif "재집계" in chain:
        phase, pcls = "③ 복창 제외 재집계", "p-run"
    elif "개입 평가" in chain:
        phase, pcls = "③ 개입 평가", "p-run"
    elif "생성 평가" in chain:
        phase, pcls = "생성 평가 (n=1,417)", "p-run"
    elif "본실행" in chain:
        n = full_rows[-1].get("seen") if full_rows else 0
        phase, pcls = f"본실행 진행 중 · {n or 0}/5,000", "p-run"
    elif arms:
        phase, pcls = "cons_weight 스윕 진행 중", "p-run"
    else:
        phase, pcls = "시작 대기", "p-wait"

    # 대표 숫자: 본실행이 있으면 그 최신 cons_loss, 없으면 마지막 스모크
    live_cons = [r["cons_loss"] for r in full_rows if r.get("cons_loss") is not None]
    src = "본실행"
    if not live_cons and arms:
        live_cons = arms[-1]["cons"]
        src = f"스모크 cw={arms[-1]['cw']}"
    if not live_cons and crash:
        live_cons = [r["cons_loss"] for r in crash if r.get("cons_loss") is not None]
        src = "크래시체크"
    cur = live_cons[-1] if live_cons else None

    h: list[str] = []
    h.append("<title>P3 — belief가 행동을 조종하는가</title>")
    h.append(f"<style>{CSS}</style>")
    h.append('<div class="page">')

    # ── 표제 ──
    h.append('<header class="stack">')
    h.append('<div class="eyebrow">EGO Step-2 · Retrospection · 개선 2 / P3</div>')
    h.append("<h1>모델이 말한 belief가 실제로 행동을 조종하는가</h1>")
    h.append('<p class="lede">belief를 남의 것으로 바꿔치기했을 때 행동이 따라 바뀌지 않으면 '
             '벌점을 주는 학습항(belief-swap consistency loss)을 넣고 5,000샘플을 돌리는 중이다. '
             '지금까지는 belief가 행동을 거의 조종하지 못했다.</p>')
    h.append(f'<div class="snapshot">{now:%Y-%m-%d %H:%M} KST 스냅샷 · '
             f'<span class="pill {pcls}">{E(phase)}</span><br>'
             '이 페이지는 실시간이 아니라 굽는 시점의 정지 화면이다. '
             '숫자를 갱신하려면 스냅샷을 다시 만들어 재배포한다.</div>')
    h.append("</header>")

    # ── 핵심 지표 ──
    h.append('<section class="card">')
    h.append('<div class="stripe"><div class="bar"></div><div class="body">')
    h.append('<div class="eyebrow">핵심 지표 — cons_loss</div>')
    h.append('<div class="headline">')
    h.append(f'<div class="figure">{fmt(cur, 2)} <span class="unit">nats</span></div>')
    if live_cons:
        h.append(sparkline(live_cons[-80:]))
    h.append("</div>")
    h.append('<p>belief를 바꾼 뒤에도 <em>원래 행동</em>이 여전히 최선인 정도를 로그 배율로 잰 값이다. '
             f'0이면 belief가 행동 선택을 좌우한다는 뜻이고, 지금 값은 원래 행동이 대안보다 '
             f'약 <strong>e<sup>{fmt(cur,0,"?")}</sup>배</strong> 선호된다는 뜻이다. '
             '<strong>내려가야 성공이다.</strong></p>')
    h.append(f'<p class="dim" style="font-size:13.5px">출처: {E(src)} · '
             f'관측 {len(live_cons)}점</p>')
    h.append("</div></div></section>")

    # ── 어제 남긴 것 ──
    h.append('<section class="stack">')
    h.append("<h2>어제 끝난 두 런이 남긴 것</h2>")
    h.append('<p>보상만 gt→wm으로 바꾼 두 런을 heldout 전량(n=1,417)으로 쟀다. '
             '핵심은 <strong>복창 가설이 기각됐다</strong>는 것이다. belief가 행동을 그대로 '
             '베껴 쓴 샘플을 빼면 인과 민감도가 무너질 것으로 예상했는데, 오히려 그대로였다. '
             '복창 때문에 부풀려진 게 아니라 belief가 정말로 조종을 못 하는 것이다.</p>')
    h.append('<div class="card"><div class="scroll"><table>')
    h.append('<thead><tr><th>지표</th><th class="n">보상 gt</th><th class="n">보상 wm</th>'
             '<th>읽는 법</th></tr></thead><tbody>')
    rows = [
        ("정확도", BASE["gt"]["acc"], BASE["wm"]["acc"],
         f'둘 다 world model의 top-1 추측({BASE["wm_top1"]})보다 낮다'),
        ("인과 민감도 ③", BASE["gt"]["c3"], BASE["wm"]["c3"], "belief를 바꿨을 때 행동이 바뀐 비율의 순증"),
        ("③ (복창 샘플 제외)", BASE["gt"]["c3x"], BASE["wm"]["c3x"],
         "제외해도 그대로 — 복창 가설 기각"),
        ("복창률", BASE["gt"]["restate"], BASE["wm"]["restate"], "belief가 행동을 그대로 베낀 비율"),
    ]
    for name, a, b, note in rows:
        h.append(f'<tr><td>{E(name)}</td><td class="n">{fmt(a)}</td>'
                 f'<td class="n">{fmt(b)}</td><td class="note">{E(note)}</td></tr>')
    h.append(f'<tr><td>③ 95% 신뢰구간</td>'
             f'<td class="n">[{fmt(BASE["gt"]["ci"][0],4)}, {fmt(BASE["gt"]["ci"][1],4)}]</td>'
             f'<td class="n">[{fmt(BASE["wm"]["ci"][0],4)}, {fmt(BASE["wm"]["ci"][1],4)}]</td>'
             f'<td class="note">둘 다 0을 넘지만 목표 0.05의 절반에 못 미친다</td></tr>')
    h.append("</tbody></table></div></div></section>")

    # ── 스윕 ──
    if arms:
        h.append('<section class="stack">')
        h.append("<h2>학습항 세기 고르기</h2>")
        h.append('<p>consistency 항의 크기를 추측하지 않고 300샘플 스모크로 고른다. '
                 '강화학습 쪽 loss가 0.6~1.0인데 cons_loss는 10을 넘어서, 세기를 잘못 잡으면 '
                 '한쪽이 다른 쪽을 완전히 눌러버린다. 45분을 써서 6시간짜리 본실행을 '
                 '잘못된 값으로 태우는 걸 막는 거래다.</p>')
        h.append('<div class="card"><div class="scroll"><table>')
        h.append('<thead><tr><th>세기</th><th class="n">진행</th><th class="n">첫 → 끝</th>'
                 '<th class="n">첫−끝</th><th class="n">기울기</th><th class="n">후반−전반</th>'
                 '<th class="n">진폭</th><th class="n">보상</th><th>상태</th></tr></thead><tbody>')
        for a in arms:
            drop, sl, hd = a["drop"], a["slope"], a["halfdiff"]
            # 세 통계 모두 '하락'을 가리킬 때만 추세로 인정한다
            votes = [(drop or 0) > 0, (sl or 0) < 0, (hd or 0) < 0]
            dcls = "good" if all(votes) else ("crit" if not any(votes) else "warn")
            rew = a["rew"][-1] if a["rew"] else None
            rcls = "crit" if (rew is not None and rew < 0.15) else ""
            state = ("선택됨" if a["cw"] == chosen else
                     ("완료" if a["done"] else ("진행 중" if a["cons"] else "대기")))
            scls = ("p-done" if (a["cw"] == chosen or a["done"]) else
                    ("p-run" if a["cons"] else "p-wait"))
            span = (f'{fmt(a["cons"][0],2)} → {fmt(a["cons"][-1],2)}' if a["cons"] else "—")
            h.append(f'<tr><td><code>{E(a["cw"])}</code></td>'
                     f'<td class="n">{a["seen"] or 0}/300</td>'
                     f'<td class="n">{span}</td>'
                     f'<td class="n {dcls}">{("%+.2f" % drop) if drop is not None else "—"}</td>'
                     f'<td class="n {dcls}">{("%+.3f" % sl) if sl is not None else "—"}</td>'
                     f'<td class="n {dcls}">{("%+.2f" % hd) if hd is not None else "—"}</td>'
                     f'<td class="n dim">{fmt(a["amp"], 2)}</td>'
                     f'<td class="n {rcls}">{fmt(rew, 2)}</td>'
                     f'<td><span class="pill {scls}">{E(state)}</span></td></tr>')
        h.append("</tbody></table></div>")
        h.append('<p class="dim" style="font-size:13.5px">'
                 '<strong>읽는 법</strong> — 「첫−끝」은 +가 하락(좋음), 「기울기」와 「후반−전반」은 '
                 '−가 하락(좋음). 셋의 방향이 일치할 때만 추세로 인정하고, 「진폭」이 그보다 크면 '
                 '노이즈로 본다. 선택 규칙(사전 등록)은 보상이 살아 있는(≥0.15) 후보 중 '
                 '「첫−끝」이 가장 큰 것인데, 관측점이 6개뿐이라 <em>마지막 한 점이 순위를 뒤집을 '
                 '수 있다</em> — 실제로 한 번 뒤집혔다. 그래서 나머지 두 통계를 함께 싣는다.</p>')
        h.append("</div></section>")

    # ── 1차 시도 실패 기록 (있으면) ──
    failed = read_jsonl(RUN / "full_FAILED_runaway_oom" / "gr_log.jsonl")
    if failed:
        h.append('<section class="stack">')
        h.append("<h2>1차 시도는 실패했다</h2>")
        h.append('<p>같은 설정으로 5,000샘플을 돌렸다가 2시간 만에 죽었다. 원인이 두 개였고, '
                 '메모리 부족보다 <strong>학습이 목표를 잘못된 방식으로 달성한 쪽</strong>이 '
                 '더 중요하다.</p>')
        h.append('<div class="card"><div class="scroll"><table>')
        h.append('<thead><tr><th class="n">샘플</th><th class="n">cons_loss</th>'
                 '<th class="n">보상</th><th>무슨 일이 일어났나</th></tr></thead><tbody>')
        notes = {200: "시작 — belief를 바꿔도 원 행동이 압도적으로 선호됨",
                 600: "목표 지점 통과 (0 부근)",
                 1000: "부호를 넘어 계속 밀림 — 멈출 장치가 없었다",
                 1400: "보상 붕괴 — 행동 선호 자체가 망가짐",
                 1800: "직후 메모리 부족으로 종료"}
        for r in failed:
            s = r.get("seen")
            cl, rw = r.get("cons_loss"), r.get("reward_ma")
            ccls = "crit" if (cl is not None and cl < 0) else ""
            rcls = "crit" if (rw is not None and rw < 0.2) else ""
            h.append(f'<tr><td class="n">{s}</td><td class="n {ccls}">{fmt(cl, 2)}</td>'
                     f'<td class="n {rcls}">{fmt(rw, 3)}</td>'
                     f'<td class="note">{E(notes.get(s, ""))}</td></tr>')
        h.append("</tbody></table></div>")
        h.append('<p><strong>고친 것 1 — 벌점에 멈출 지점을 줬다.</strong> 원래 식은 '
                 '"원래 행동보다 대안이 선호될수록 좋다"를 끝없이 밀어붙인다. 그래서 모델은 '
                 'belief를 참고하는 대신 <em>어떤 행동도 확신하지 않는</em> 쪽으로 도망쳤다. '
                 '이제 대안이 0.5나트만큼 앞서면 벌점이 0이 되어 학습이 멈춘다.</p>')
        h.append('<p><strong>고친 것 2 — 무너지면 스스로 멈춘다.</strong> 붕괴는 1,200샘플에서 '
                 '나타났는데 세기를 고를 때 쓴 스모크는 300샘플짜리였다. 짧은 예비 실험으로는 '
                 '원리적으로 볼 수 없는 실패다. 이제 본실행이 보상을 감시하다 0.20 아래로 '
                 '두 번 연속 떨어지면 체크포인트를 남기고 중단한다 — 6시간을 태우지 않는다.</p>')
        h.append('<p class="dim" style="font-size:13.5px">메모리 쪽은 부수적이다. 확률 계산 전에 '
                 '전체 어휘(15만)에 대한 32비트 사본을 만들고 있었는데, 실제로 필요한 구간은 '
                 '행동 토큰 21개뿐이라 그 부분만 잘라내도록 고쳤다(약 70배 절감).</p>')
        h.append("</section>")

    # ── 결과(있으면) ──
    if ev or rc:
        sub = (rc.get("subsets") or {}).get("excl_restatement") or {}
        ci = sub.get("causal_sensitivity_ci95") or {}
        acc = (ev.get("full") or {}).get("acc")
        h.append('<section class="stack"><h2>이번 런 결과</h2>')
        h.append('<div class="card"><div class="scroll"><table>')
        h.append('<thead><tr><th>지표</th><th class="n">어제 (gt)</th><th class="n">이번</th>'
                 '<th>판정</th></tr></thead><tbody>')
        okacc = acc is not None and acc >= BASE["gt"]["acc"] - 0.045
        h.append(f'<tr><td>정확도 (n=1,417)</td><td class="n">{fmt(BASE["gt"]["acc"])}</td>'
                 f'<td class="n">{fmt(acc)}</td>'
                 f'<td class="{"good" if okacc else "crit"}">'
                 f'{"유지" if okacc else "붕괴"}</td></tr>')
        lo = ci.get("lo")
        okc3 = lo is not None and lo > BASE["gt"]["c3x"]
        h.append(f'<tr><td>③ (복창 제외)</td><td class="n">{fmt(BASE["gt"]["c3x"])}</td>'
                 f'<td class="n">{fmt(sub.get("causal_sensitivity"))}</td>'
                 f'<td class="{"good" if okc3 else "warn"}">'
                 f'{"유의하게 상승" if okc3 else "기준 미달"}</td></tr>')
        if lo is not None:
            h.append(f'<tr><td>③ 95% 신뢰구간</td>'
                     f'<td class="n">[{fmt(BASE["gt"]["ci"][0])}, {fmt(BASE["gt"]["ci"][1])}]</td>'
                     f'<td class="n">[{fmt(lo)}, {fmt(ci.get("hi"))}]</td>'
                     f'<td class="dim">paired bootstrap</td></tr>')
        h.append("</tbody></table></div></div></section>")

    # ── 판정 기준 ──
    h.append('<section class="stack">')
    h.append("<h2>미리 정해둔 판정 기준</h2>")
    h.append('<p>결과를 보고 기준을 고치지 않기 위해 실행 전에 체인 스크립트 안에 박아뒀다.</p>')
    h.append('<div class="card"><ul class="crit-list">')
    for mark, txt in [
        ("성공", "복창 샘플을 뺀 인과 민감도의 신뢰구간 하한이 어제 값 0.0137을 넘고, "
                 "동시에 정확도가 0.2371에서 최소검출효과(0.045) 이상 떨어지지 않을 것"),
        ("실패", "정확도가 그보다 더 떨어지거나, 인과 민감도가 움직이지 않을 것"),
        ("주의", "reasoning은 고정한 채 belief만 바꾸므로, 학습이 belief를 쓰게 되는 대신 "
                 "reasoning의 영향력을 깎는 방향으로 갈 수 있다. 그러면 인과 민감도는 오르는데 "
                 "정확도가 무너진다 — 스모크의 보상 하한 0.15가 그 조기 경보다"),
    ]:
        h.append(f'<li><span class="mark">{E(mark)}</span><span>{E(txt)}</span></li>')
    h.append("</ul></div></section>")

    # ── 방법 ──
    h.append('<section class="stack">')
    h.append("<h2>어떻게 재는가</h2>")
    h.append('<p>모델이 스스로 만든 추론과 belief 중 <strong>belief만</strong> 다른 샘플의 것으로 '
             '바꿔치기한다. 그 상태에서 world model이 준 후보 5개의 점수를 매기고, 원래 행동이 '
             '여전히 1등이면 벌점을 준다. 비교 대상이 될 "바뀐 행동"은 새로 생성하지 않고 후보 '
             '안에서 고르기 때문에, 학습 스텝당 생성이 늘지 않는다 — 본실행이 9시간에서 '
             '6시간으로 줄어든 이유다.</p>')
    h.append('<p class="dim" style="font-size:13.5px">벌점 식에는 함정이 하나 있었다. "바뀐 행동"을 '
             '단순히 1등 후보로 정의하면, belief를 바꿔도 1등이 그대로일 때 두 항이 상쇄돼 벌점이 '
             '정확히 0이 된다 — 가장 벌해야 할 경우에 학습 신호가 사라진다. 그래서 원래 행동을 '
             '제외한 최선 후보로 정의했다.</p>')
    h.append("</section>")

    h.append('<hr><footer>EGO Step-2 Retrospection · 런 디렉터리 <code>EGO/runs/p3_cons</code> · '
             'H200 1대(cuda:1) 단독 사용 · 모든 수치는 heldout 전량 n=1,417 기준이며 '
             '정확도 신뢰구간은 10,000회 부트스트랩이다.</footer>')
    h.append("</div>")
    return "\n".join(h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(build(), encoding="utf-8")
    print(f"[done] {p}  ({p.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
