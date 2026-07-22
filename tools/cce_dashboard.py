#!/usr/bin/env python3
"""cce_dashboard.py — 2단계 candidate CE 실시간 현황 (포트 7865).

의존성 없는 stdlib http.server. 매 요청마다 로그를 다시 읽으므로 실행 중에도 최신이다.

이 단계의 질문은 하나다:
    후보 선택을 **직접** 최적화하면 무학습 베이스라인(WM top-1)을 넘는가?

지금까지 완주한 학습은 전부 텍스트 생성을 최적화했고 action 은 부산물이었다.
평가 지표를 직접 겨냥한 objective 는 한 번도 완주된 적이 없다 (f0_gx 는 800샘플에서 버려짐).

    python tools/cce_dashboard.py --host 0.0.0.0 --port 7865
"""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

RUN = Path("/mnt/nvme/migration/jihun/EGO/runs/cce")
KST = timezone(timedelta(hours=9))

# 고정 기준값 — 전부 실측이다. 결과를 보고 바꾸지 않는다.
L0 = 0.3994            # WM top-1 무학습 (heldout n=1,417)
BASE_SCORING = 0.3876  # base 모델 후보 스코어링 조건부 정확도 (teacher_headroom n=1,370)
GX_ABORTED = [(200, 0.395), (400, 0.385), (600, 0.545), (800, 0.450)]  # 버려진 run
PREV_BEST = 0.3380     # 역대 최고 arm (f0gr_final, n=500)
CEILING = 0.6260       # R5 — candidate-constrained 구조적 상한


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


def tail(p: Path, n=14) -> list[str]:
    if not p.is_file():
        return []
    return [x for x in p.read_text(encoding="utf-8", errors="replace").splitlines()
            if not x.startswith("Loading weights")][-n:]


def gpu_rows():
    try:
        r = subprocess.run(["nvidia-smi",
                            "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=8)
        return [[c.strip() for c in ln.split(",")] for ln in r.stdout.strip().splitlines() if ln]
    except Exception:
        return []


def spark(vals, w=70):
    if len(vals) < 2:
        return ""
    v = vals[-w:]
    lo, hi = min(v), max(v)
    if hi - lo < 1e-12:
        return "▄" * len(v)
    b = "▁▂▃▄▅▆▇█"
    return "".join(b[min(7, int((x - lo) / (hi - lo) * 7.999))] for x in v)


CSS = """
:root{--bg:#eef0f3;--fg:#10151d;--dim:#5a6472;--card:#fff;--line:#d5dae1;--rule:#c3cad3;
 --accent:#3d7f9d;--wash:rgba(61,127,157,.14);--good:#2f8f5b;--warn:#b8801f;--crit:#b8453c;
 --mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
 --sans:system-ui,-apple-system,'Segoe UI',Roboto,sans-serif;
 --serif:'Iowan Old Style','Palatino Linotype',Palatino,Georgia,serif}
@media(prefers-color-scheme:dark){:root{--bg:#10151d;--fg:#e4e8ee;--dim:#96a0ad;--card:#171e28;
 --line:#2a3340;--rule:#333d4b;--accent:#63a8c6;--wash:rgba(99,168,198,.16);--good:#4fb277;
 --warn:#d3a03f;--crit:#d4695f}}
:root[data-theme="dark"]{--bg:#10151d;--fg:#e4e8ee;--dim:#96a0ad;--card:#171e28;--line:#2a3340;
 --rule:#333d4b;--accent:#63a8c6;--wash:rgba(99,168,198,.16);--good:#4fb277;--warn:#d3a03f;
 --crit:#d4695f}
:root[data-theme="light"]{--bg:#eef0f3;--fg:#10151d;--dim:#5a6472;--card:#fff;--line:#d5dae1;
 --rule:#c3cad3;--accent:#3d7f9d;--wash:rgba(61,127,157,.14);--good:#2f8f5b;--warn:#b8801f;
 --crit:#b8453c}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font-family:var(--sans);
 font-size:15px;line-height:1.55}
.wrap{max-width:980px;margin:0 auto;padding:32px 20px 64px;display:flex;flex-direction:column;gap:28px}
.eyebrow{font-family:var(--mono);font-size:11px;letter-spacing:.14em;text-transform:uppercase;
 color:var(--dim)}
h1{font-family:var(--serif);font-size:clamp(23px,4.3vw,32px);margin:0;line-height:1.16;
 text-wrap:balance}
h2{font-family:var(--serif);font-size:19px;margin:0 0 9px}
p{margin:0;max-width:66ch}.dim{color:var(--dim)}
.sub{font-family:var(--mono);font-size:12px;color:var(--dim);border-left:2px solid var(--accent);
 padding-left:11px}
.card{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:18px 20px;
 display:flex;flex-direction:column;gap:12px}
.scroll{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:13.5px}
th,td{text-align:left;padding:7px 12px 7px 0;border-bottom:1px solid var(--line);white-space:nowrap}
thead th{font-family:var(--mono);font-size:10.5px;letter-spacing:.08em;text-transform:uppercase;
 color:var(--dim);border-bottom-color:var(--rule)}
td.n,th.n{text-align:right;font-family:var(--mono);font-variant-numeric:tabular-nums}
tbody tr:last-child td{border-bottom:0}
.pill{display:inline-block;font-family:var(--mono);font-size:10.5px;padding:2px 8px;
 border-radius:10px;border:1px solid currentColor;font-weight:600}
.p-run{color:var(--warn)}.p-done{color:var(--good)}.p-wait{color:var(--dim)}.p-bad{color:var(--crit)}
.big{font-family:var(--mono);font-size:30px;font-variant-numeric:tabular-nums;line-height:1}
.good{color:var(--good)}.warn{color:var(--warn)}.crit{color:var(--crit)}
.gauge{position:relative;height:30px;background:var(--wash);border-radius:4px;overflow:hidden}
.gauge .f{position:absolute;inset:0 auto 0 0;background:var(--accent);border-radius:4px}
.gauge .mk{position:absolute;top:0;bottom:0;width:2px;background:var(--crit)}
.gauge .lb{position:absolute;top:-18px;font-family:var(--mono);font-size:10px;color:var(--crit);
 transform:translateX(-50%);white-space:nowrap}
.spark{font-size:15px;letter-spacing:-1px;color:var(--accent)}
pre{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:10px 12px;
 overflow-x:auto;font-size:12px;color:var(--dim);margin:0}
:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
"""


def render():
    now = datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S KST")
    chain = (RUN / "chain.log").read_text(encoding="utf-8", errors="replace") \
        if (RUN / "chain.log").is_file() else ""
    smoke = read_jsonl(RUN / "smoke" / "gx_log.jsonl")
    train = read_jsonl(Path("/mnt/nvme/migration/jihun/EGO_jihun/outputs/step2/cce_1f/gx_log.jsonl"))
    ev = read_json(RUN / "eval_scored.json")
    ev_b = read_json(RUN / "eval_scored_base.json")
    gen = read_json(RUN / "eval_gen.json")

    if (RUN / "CHAIN_DONE").is_file():
        phase, pc = "완료", "p-done"
    elif (RUN / "CHAIN_FAILED").is_file():
        phase, pc = "중단됨", "p-bad"
    elif "분해" in chain or ev:
        phase, pc = "평가·분해", "p-run"
    elif "본실행" in chain:
        s = train[-1].get("seen", 0) if train else 0
        phase, pc = f"본실행 {s}/5,000", "p-run"
    elif "스모크" in chain:
        s = smoke[-1].get("seen", 0) if smoke else 0
        phase, pc = f"스모크 {s}/300", "p-run"
    else:
        phase, pc = "시작 대기", "p-wait"

    h = [f"<title>2단계 · 후보 선택 직접 학습</title><style>{CSS}</style><div class='wrap'>"]
    h.append("<header style='display:flex;flex-direction:column;gap:9px'>"
             "<div class='eyebrow'>EGO Step-2 · 2단계 candidate CE</div>"
             "<h1>후보 선택을 직접 학습하면 무학습 기준선을 넘는가</h1>"
             "<p class='dim'>지금까지 완주한 학습은 모두 <strong>글쓰기</strong>를 최적화했고 "
             "행동 선택은 부산물이었다. 평가 지표를 직접 겨냥한 목적함수는 이 프로젝트에서 "
             "한 번도 완주된 적이 없다.</p>"
             f"<div class='sub'>{now} · 10초 갱신 · <span class='pill {pc}'>{phase}</span></div>"
             "</header>")

    # ── 게이지: L0 대비 어디에 있나 ──
    cur = ev.get("acc") if ev else None
    h.append("<section class='card'><div class='eyebrow'>heldout 전체 정확도 — "
             "넘어야 하는 선은 0.3994</div>")
    if cur is None:
        h.append("<p class='dim'>학습 중 — 평가는 학습 종료 후 시작된다.</p>")
    else:
        w = min(100.0, cur / CEILING * 100)
        h.append(f"<div class='big {'good' if cur > L0 else 'crit'}'>{cur:.4f}</div>")
        h.append("<div style='padding-top:20px'><div class='gauge'>"
                 f"<div class='f' style='width:{w:.1f}%'></div>"
                 f"<div class='mk' style='left:{L0/CEILING*100:.1f}%'></div>"
                 f"<div class='lb' style='left:{L0/CEILING*100:.1f}%'>L0 {L0}</div>"
                 "</div></div>")
        h.append(f"<p class='dim' style='font-size:12.5px'>0 ─ 구조적 상한 {CEILING} "
                 f"(GT가 후보 안에 있을 확률). 붉은 선이 무학습 기준선.</p>")
    h.append("</section>")

    # ── 학습 곡선 ──
    for nm, rows, tot in (("스모크", smoke, 300), ("본실행", train, 5000)):
        if not rows:
            continue
        acc = [r["train_acc"] for r in rows if r.get("train_acc") is not None]
        ls = [r["loss"] for r in rows if r.get("loss") is not None]
        last = rows[-1]
        h.append(f"<section class='card'><div class='eyebrow'>{nm}</div><div class='scroll'><table>"
                 "<thead><tr><th class='n'>seen</th><th class='n'>train_acc</th>"
                 "<th class='n'>loss</th></tr></thead><tbody>"
                 f"<tr><td class='n'>{last.get('seen','—')}/{tot}</td>"
                 f"<td class='n'>{last.get('train_acc','—')}</td>"
                 f"<td class='n'>{last.get('loss','—')}</td></tr></tbody></table></div>")
        if acc:
            h.append(f"<div class='dim' style='font-size:12px'>train_acc 추이 "
                     f"({min(acc):.3f}~{max(acc):.3f})</div><div class='spark'>{spark(acc)}</div>")
        if ls:
            h.append(f"<div class='dim' style='font-size:12px'>loss 추이 "
                     f"({ls[0]:.2f} → {ls[-1]:.2f})</div><div class='spark'>{spark(ls)}</div>")
        h.append("</section>")

    # ── 기준선 대조 ──
    h.append("<section class='card'><div class='eyebrow'>기준선 — 전부 실측</div>"
             "<div class='scroll'><table><thead><tr><th>기준</th><th class='n'>값</th>"
             "<th>무엇인가</th></tr></thead><tbody>")
    rows_ref = [
        ("L0 · WM top-1 무학습", L0, "학습 없이 world model 1등을 그대로 따르기 — 아직 아무도 못 넘음"),
        ("역대 최고 arm", PREV_BEST, "f0gr_final (n=500) — L0 미달"),
        ("base 후보 스코어링", BASE_SCORING, "학습 전 모델의 조건부 선택 정확도"),
        ("구조적 상한 R5", CEILING, "GT가 후보 5개 안에 있을 확률"),
    ]
    for nm, v, why in rows_ref:
        h.append(f"<tr><td>{nm}</td><td class='n'>{v:.4f}</td><td class='dim'>{why}</td></tr>")
    if ev_b:
        h.append(f"<tr><td>base 스코어링 (이번 heldout)</td><td class='n'>{ev_b.get('acc','—')}</td>"
                 f"<td class='dim'>동일 경로 대조군</td></tr>")
    if ev:
        cls = "good" if ev.get("acc", 0) > L0 else "crit"
        h.append(f"<tr><td><strong>학습 후 (이번)</strong></td>"
                 f"<td class='n {cls}'>{ev.get('acc','—')}</td>"
                 f"<td class='dim'>후보 스코어링 · 조건부 {ev.get('conditional_acc','—')}</td></tr>")
    if gen:
        h.append(f"<tr><td>학습 후 (생성 경로)</td>"
                 f"<td class='n'>{(gen.get('full') or {}).get('acc','—')}</td>"
                 f"<td class='dim'>기존 arm 들과 비교 가능한 경로</td></tr>")
    h.append("</tbody></table></div></section>")

    # ── G1/G2 ──
    if ev:
        h.append("<section class='card'><div class='eyebrow'>G1 / G2 분해</div><div class='scroll'>"
                 "<table><thead><tr><th>구간</th><th class='n'>base</th><th class='n'>학습 후</th>"
                 "<th>의미</th></tr></thead><tbody>")
        for k, nm, why in (("g1_retention", "G1 보존", "world model이 이미 맞힌 것을 지키는 비율"),
                           ("g2_correction", "G2 교정", "world model 1등이 틀렸을 때 고치는 비율"),
                           ("conditional_acc", "조건부 정확도", "GT가 후보 안에 있을 때 맞히는 비율")):
            h.append(f"<tr><td>{nm}</td><td class='n'>{ev_b.get(k,'—')}</td>"
                     f"<td class='n'>{ev.get(k,'—')}</td><td class='dim'>{why}</td></tr>")
        h.append("</tbody></table></div></section>")

    # ── 버려진 run 참고 ──
    h.append("<section class='card'><div class='eyebrow'>참고 — 07-19에 버려진 같은 objective</div>"
             "<div class='scroll'><table><thead><tr><th class='n'>seen</th>"
             "<th class='n'>train_acc</th></tr></thead><tbody>")
    for s, a in GX_ABORTED:
        h.append(f"<tr><td class='n'>{s}</td><td class='n'>{a}</td></tr>")
    h.append("</tbody></table></div><p class='dim' style='font-size:12.5px'>"
             "체크포인트 없이 800샘플에서 중단됐다. 시작점 0.395가 base 스코어링 능력과 "
             "일치하고 37 step 만에 0.545로 올랐다 — 이번에 완주시키는 이유다.</p></section>")

    # ── GPU ──
    h.append("<section class='card'><div class='eyebrow'>GPU</div><div class='scroll'><table>"
             "<thead><tr><th>#</th><th class='n'>사용률</th><th class='n'>메모리</th></tr></thead>"
             "<tbody>")
    for g in gpu_rows():
        if len(g) >= 4:
            h.append(f"<tr><td>{g[0]}</td><td class='n'>{g[1]}%</td>"
                     f"<td class='n'>{g[2]}/{g[3]} MiB</td></tr>")
    h.append("</tbody></table></div></section>")

    if chain:
        h.append("<section><h2>체인 로그</h2><pre>" +
                 "\n".join(x.replace("&", "&amp;").replace("<", "&lt;")
                           for x in chain.splitlines()[-16:]) + "</pre></section>")
    for nm, f in (("스모크", RUN / "smoke.log"), ("본실행", RUN / "train.log"),
                  ("평가", RUN / "eval_scored.log")):
        lines = tail(f)
        if lines:
            h.append(f"<section><h2>{nm} 로그</h2><pre>" +
                     "\n".join(x.replace("&", "&amp;").replace("<", "&lt;")[:200]
                               for x in lines) + "</pre></section>")
    h.append("</div>")
    return ("<!doctype html><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            "<meta http-equiv='refresh' content='10'>" + "".join(h))


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        try:
            body = render().encode("utf-8")
        except Exception as e:
            body = f"<pre>dashboard error: {e}</pre>".encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=7865)
    args = ap.parse_args()
    print(f"[dashboard] http://{args.host}:{args.port}/", flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
