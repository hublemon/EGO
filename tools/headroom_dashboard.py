#!/usr/bin/env python3
"""headroom_dashboard.py — teacher headroom 실험 실시간 현황 (포트 7864).

의존성 없는 stdlib http.server. 결과 jsonl 을 매 요청마다 다시 읽어 집계하므로
실행 중에도 지금까지의 값이 그대로 보인다.

읽는 법:
  present  = 현재 정보만 본 teacher 의 후보 선택 정확도
  future   = **실제 미래**를 본 teacher
  shuffled = 다른 샘플의 미래를 본 teacher (맥락 길이 대조군)

  Retrospection 가설이 지지되려면  future > present  이면서  future > shuffled  여야 한다.
  future ≈ shuffled 면 이득의 정체는 '미래 정보'가 아니라 '맥락이 길어진 것'이다.

  이 실험은 학습이 아니라 **정보량 상한 측정**이다. future − present 가 0 에 가까우면
  아무리 정교하게 증류해도 student 에게 전달할 것이 없다.

    python tools/headroom_dashboard.py --host 0.0.0.0 --port 7864
"""
from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

RUN = Path("/mnt/nvme/migration/jihun/EGO/runs/headroom")
KST = timezone(timedelta(hours=9))
ARMS = ("present", "future", "shuffled")
LABEL = {"present": "현재 정보만", "future": "실제 미래 (hindsight)", "shuffled": "남의 미래 (대조군)"}


def load(path: Path):
    meta, recs = {}, []
    if not path.is_file():
        return meta, recs
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue                      # 마지막 줄이 쓰이는 중일 수 있다
        if "meta" in d:
            meta = d["meta"]
        elif all(a in d for a in ARMS):
            recs.append(d)
    return meta, recs


def boot_paired(pairs, n_boot=5000, seed=0):
    """(a_i, b_i) 쌍 재표집으로 mean(b)-mean(a) 의 95% CI. arm 간 비교는 반드시 paired."""
    n = len(pairs)
    if n < 2:
        return None
    diffs = [b - a for a, b in pairs]
    rnd = random.Random(seed)
    means = []
    for _ in range(n_boot):
        means.append(sum(diffs[rnd.randrange(n)] for _ in range(n)) / n)
    means.sort()
    pt = sum(diffs) / n
    mu = sum(means) / n_boot
    se = math.sqrt(sum((x - mu) ** 2 for x in means) / max(1, n_boot - 1))
    return {"point": pt, "lo": means[int(0.025 * n_boot)],
            "hi": means[min(n_boot - 1, int(0.975 * n_boot))], "se": se, "n": n}


def gpu_rows():
    try:
        r = subprocess.run(["nvidia-smi",
                            "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=8)
        return [[c.strip() for c in ln.split(",")] for ln in r.stdout.strip().splitlines() if ln]
    except Exception:
        return []


CSS = """
:root{--bg:#eef0f3;--fg:#10151d;--dim:#5a6472;--card:#fff;--line:#d5dae1;--rule:#c3cad3;
 --accent:#3d7f9d;--accent-wash:rgba(61,127,157,.14);--good:#2f8f5b;--warn:#b8801f;--crit:#b8453c;
 --mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
 --sans:system-ui,-apple-system,'Segoe UI',Roboto,sans-serif;
 --serif:'Iowan Old Style','Palatino Linotype',Palatino,Georgia,serif}
@media(prefers-color-scheme:dark){:root{--bg:#10151d;--fg:#e4e8ee;--dim:#96a0ad;--card:#171e28;
 --line:#2a3340;--rule:#333d4b;--accent:#63a8c6;--accent-wash:rgba(99,168,198,.16);
 --good:#4fb277;--warn:#d3a03f;--crit:#d4695f}}
:root[data-theme="dark"]{--bg:#10151d;--fg:#e4e8ee;--dim:#96a0ad;--card:#171e28;--line:#2a3340;
 --rule:#333d4b;--accent:#63a8c6;--accent-wash:rgba(99,168,198,.16);--good:#4fb277;
 --warn:#d3a03f;--crit:#d4695f}
:root[data-theme="light"]{--bg:#eef0f3;--fg:#10151d;--dim:#5a6472;--card:#fff;--line:#d5dae1;
 --rule:#c3cad3;--accent:#3d7f9d;--accent-wash:rgba(61,127,157,.14);--good:#2f8f5b;
 --warn:#b8801f;--crit:#b8453c}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font-family:var(--sans);
 font-size:15px;line-height:1.55}
.wrap{max-width:960px;margin:0 auto;padding:32px 20px 64px;display:flex;flex-direction:column;gap:30px}
h1{font-family:var(--serif);font-size:clamp(24px,4.4vw,33px);margin:0;line-height:1.16;
 text-wrap:balance}
h2{font-family:var(--serif);font-size:20px;margin:0 0 10px}
.eyebrow{font-family:var(--mono);font-size:11px;letter-spacing:.14em;text-transform:uppercase;
 color:var(--dim)}
p{margin:0;max-width:66ch}.dim{color:var(--dim)}
.sub{font-family:var(--mono);font-size:12px;color:var(--dim);border-left:2px solid var(--accent);
 padding-left:11px}
.card{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:18px 20px;
 display:flex;flex-direction:column;gap:12px}
.bars{display:flex;flex-direction:column;gap:12px}
.bar{display:grid;grid-template-columns:9rem 1fr 5rem;gap:12px;align-items:center}
.bar .t{font-size:13.5px}
.track{background:var(--accent-wash);border-radius:3px;height:22px;position:relative;overflow:hidden}
.fill{position:absolute;left:0;top:0;bottom:0;background:var(--accent);border-radius:3px}
.bar .v{font-family:var(--mono);font-variant-numeric:tabular-nums;text-align:right;font-size:15px}
table{border-collapse:collapse;width:100%;font-size:13.5px}
.scroll{overflow-x:auto}
th,td{text-align:left;padding:7px 12px 7px 0;border-bottom:1px solid var(--line);white-space:nowrap}
thead th{font-family:var(--mono);font-size:10.5px;letter-spacing:.08em;text-transform:uppercase;
 color:var(--dim);border-bottom-color:var(--rule)}
td.n,th.n{text-align:right;font-family:var(--mono);font-variant-numeric:tabular-nums}
.verdict{font-family:var(--mono);font-size:13px;padding:10px 12px;border-radius:5px;
 border:1px solid currentColor}
.v-good{color:var(--good)}.v-warn{color:var(--warn)}.v-bad{color:var(--crit)}.v-wait{color:var(--dim)}
.big{font-size:30px;font-family:var(--mono);font-variant-numeric:tabular-nums;line-height:1}
pre{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:10px 12px;
 overflow-x:auto;font-size:12px;color:var(--dim);margin:0}
:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
"""


def render():
    now = datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S KST")
    full = RUN / "headroom.jsonl"
    smoke = RUN / "smoke.jsonl"
    path = full if full.is_file() else smoke
    meta, recs = load(path)
    n_done = len(recs)
    n_target = meta.get("n_usable", 0)

    ok = {a: sum(1 for r in recs if r[a]["correct"]) for a in ARMS} if recs else {}
    acc = {a: (ok[a] / n_done) for a in ARMS} if n_done else {}

    h = [f"<title>Retrospection 정보량 상한 측정</title><style>{CSS}</style>",
         "<div class='wrap'>",
         "<header style='display:flex;flex-direction:column;gap:10px'>",
         "<div class='eyebrow'>EGO Step-2 · Retrospection · 1단계 진단</div>",
         "<h1>미래를 본 teacher는 더 잘 고르는가</h1>",
         "<p class='dim'>완료된 미래 궤적을 본 teacher가 후보 5개 중 정답을 더 자주 고르는지 "
         "측정한다. 이 차이가 Retrospection이 학생 모델에 전달할 수 있는 정보량의 "
         "<strong>상한</strong>이다. 차이가 없으면 아무리 정교하게 증류해도 전달할 것이 없다.</p>",
         f"<div class='sub'>{now} · 10초 자동 갱신 · "
         f"{'본실행' if path == full else '스모크'} {n_done}/{n_target or '—'} 샘플</div>",
         "</header>"]

    if not n_done:
        h.append("<div class='card'><span class='verdict v-wait'>아직 결과 없음 — "
                 "모델 로딩 중이거나 시작 대기</span></div>")
    else:
        # ── 막대 ──
        h.append("<section class='card'><div class='eyebrow'>후보 선택 정확도 "
                 "(GT가 후보 안에 있는 샘플만)</div><div class='bars'>")
        mx = max(acc.values()) or 1
        for a in ARMS:
            w = acc[a] / max(mx, 1e-9) * 100
            h.append(f"<div class='bar'><div class='t'>{LABEL[a]}</div>"
                     f"<div class='track'><div class='fill' style='width:{w:.1f}%'></div></div>"
                     f"<div class='v'>{acc[a]:.3f}</div></div>")
        h.append("</div></section>")

        # ── 핵심 차이 ──
        d_fp = boot_paired([(float(r["present"]["correct"]), float(r["future"]["correct"]))
                            for r in recs])
        d_fs = boot_paired([(float(r["shuffled"]["correct"]), float(r["future"]["correct"]))
                            for r in recs])
        h.append("<section class='card'><div class='eyebrow'>핵심 지표 — 미래가 주는 이득</div>")
        if d_fp:
            cls = "v-good" if d_fp["lo"] > 0 else ("v-bad" if d_fp["hi"] < 0 else "v-warn")
            h.append(f"<div class='big {cls.replace('v-','')}' style='color:inherit'>"
                     f"{d_fp['point']:+.3f}</div>")
            h.append("<div class='scroll'><table><thead><tr><th>비교</th><th class='n'>차이</th>"
                     "<th class='n'>95% CI</th><th>판정</th></tr></thead><tbody>")
            for nm, d, why in (("미래 − 현재", d_fp, "미래 정보 + 맥락 길이"),
                               ("미래 − 남의 미래", d_fs, "미래 정보만 (맥락 길이 통제)")):
                if not d:
                    continue
                v = ("유의하게 +" if d["lo"] > 0 else
                     ("유의하게 −" if d["hi"] < 0 else "0과 구분 안 됨"))
                c = "good" if d["lo"] > 0 else ("crit" if d["hi"] < 0 else "warn")
                h.append(f"<tr><td>{nm}</td><td class='n'>{d['point']:+.4f}</td>"
                         f"<td class='n'>[{d['lo']:+.4f}, {d['hi']:+.4f}]</td>"
                         f"<td style='color:var(--{c})'>{v} <span class='dim'>· {why}</span></td></tr>")
            h.append("</tbody></table></div>")
            # ── 사전 등록 판정 ──
            if d_fs and d_fp:
                if d_fs["lo"] > 0 and d_fp["lo"] > 0:
                    v, c = ("가설 지지 — 미래 정보가 후보 판별에 기여한다. "
                            "증류할 정보가 존재하므로 다음 단계로 간다."), "v-good"
                elif d_fp["lo"] > 0 and d_fs["hi"] <= 0:
                    v, c = ("주의 — 이득이 미래 정보가 아니라 맥락 길이로 설명된다. "
                            "Retrospection 주장의 근거가 되지 못한다."), "v-warn"
                elif n_done < 200:
                    v, c = f"표본 부족 ({n_done}) — 판정 보류", "v-wait"
                else:
                    v, c = ("★ 가설 위험 — 미래를 봐도 후보 선택이 나아지지 않는다. "
                            "증류 설계 전에 방향을 재고해야 한다."), "v-bad"
                h.append(f"<div class='verdict {c}'>{v}</div>")
        h.append("</section>")

        # ── 전이 행렬 ──
        h.append("<section class='card'><div class='eyebrow'>미래를 보고 답이 어떻게 바뀌었나</div>"
                 "<div class='scroll'><table><thead><tr><th>전이</th><th class='n'>건수</th>"
                 "<th class='n'>비율</th><th>의미</th></tr></thead><tbody>")
        w2r = sum(1 for r in recs if not r["present"]["correct"] and r["future"]["correct"])
        r2w = sum(1 for r in recs if r["present"]["correct"] and not r["future"]["correct"])
        same = n_done - w2r - r2w
        for nm, c, why in (("오답 → 정답", w2r, "미래가 실제로 교정한 사례"),
                           ("정답 → 오답", r2w, "미래가 오히려 흔든 사례"),
                           ("변화 없음", same, "")):
            h.append(f"<tr><td>{nm}</td><td class='n'>{c}</td><td class='n'>{c/n_done:.3f}</td>"
                     f"<td class='dim'>{why}</td></tr>")
        h.append("</tbody></table></div><p class='dim' style='font-size:12.5px'>"
                 "순이득 = (오답→정답) − (정답→오답). 두 값이 비슷하면 미래는 정보를 준 게 "
                 "아니라 답을 흔든 것이다.</p></section>")

    # ── 실행 환경 ──
    h.append("<section class='card'><div class='eyebrow'>실행</div><div class='scroll'><table>"
             "<thead><tr><th>GPU</th><th class='n'>사용률</th><th class='n'>메모리</th>"
             "</tr></thead><tbody>")
    for g in gpu_rows():
        if len(g) >= 4:
            h.append(f"<tr><td>{g[0]}</td><td class='n'>{g[1]}%</td>"
                     f"<td class='n'>{g[2]}/{g[3]} MiB</td></tr>")
    h.append("</tbody></table></div>")
    if meta:
        h.append(f"<p class='dim' style='font-size:12.5px'>사용 샘플 {meta.get('n_usable')} · "
                 f"GT가 후보 밖이라 제외 {meta.get('drop_gt_outside')} · "
                 f"미래 없어서 제외 {meta.get('drop_no_future')} "
                 f"(미래 suffix에서 정답과 같은 행동은 전부 제거 — 누설 차단)</p>")
    h.append("</section>")

    log = RUN / ("headroom.log" if path == full else "smoke.log")
    if log.is_file():
        lines = [x for x in log.read_text(encoding="utf-8", errors="replace").splitlines()
                 if not x.startswith("Loading weights")][-12:]
        h.append("<section><h2>로그</h2><pre>" +
                 "\n".join(x.replace("&", "&amp;").replace("<", "&lt;") for x in lines) +
                 "</pre></section>")
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
    ap.add_argument("--port", type=int, default=7864)
    args = ap.parse_args()
    print(f"[dashboard] http://{args.host}:{args.port}/", flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
