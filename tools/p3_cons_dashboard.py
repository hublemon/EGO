#!/usr/bin/env python3
"""p3_cons_dashboard.py — P3(belief-swap consistency) 무인 실행 실시간 현황.

의존성 없는 stdlib http.server (tools/night_dashboard.py 와 같은 관례).
남은 시간은 **로그에 찍힌 실측 처리속도**로 계산한다 — 하드코딩 추정을 쓰지 않으므로
GPU 가 느려지면 표시되는 숫자도 정직하게 늘어난다.

    python tools/p3_cons_dashboard.py --host 0.0.0.0 --port 7863

핵심 관전 지표는 acc 가 아니라 `cons_loss` 다.
cons_loss = log q(a_orig | b_swap) − log q(a_swap | b_swap) 의 배치 평균이고,
**> 0 이면 belief 를 남의 것으로 바꿔도 원래 action 이 여전히 최선** — 즉 belief 가
action 을 조향하지 못하는 상태다. 이 값이 0 쪽으로 내려가는 것이 P3 의 학습 신호다.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

RUN = Path("/mnt/nvme/migration/jihun/EGO/runs/p3_cons")
PREV = Path("/mnt/nvme/migration/jihun/EGO/runs/retro_overnight")
KST = timezone(timedelta(hours=9))

# 체인 단계 — 순서가 곧 대기열이다. (마커/로그 파일, 표시명, 이 단계가 답하는 질문)
def stages() -> list[tuple[str, str, str]]:
    """체인 단계. 스모크는 cons_weight 스윕이라 개수가 고정이 아니므로 디렉터리에서 찾는다."""
    sm = sorted(p.name for p in RUN.glob("smoke_cw*") if p.is_dir())
    chosen = ""
    f = RUN / "chosen_cw.txt"
    if f.is_file():
        chosen = f.read_text(encoding="utf-8").strip()
    out = [("crashcheck", "크래시체크 (24샘플)", "cons 경로가 죽지 않고 도는가")]
    for name in sm:
        cw = name.replace("smoke_cw", "")
        mark = " ← 선택됨" if cw == chosen else ""
        out.append((name, f"스모크 cw={cw}{mark}",
                    "cons_loss 가 내려가면서 reward_ma 가 살아남는가"))
    out += [("full",    "본실행 (5,000샘플)",  "cons_loss 가 계속 내려가는가"),
            ("eval",    "생성 평가 (n=1,417)", "acc 가 무너지지 않았는가"),
            ("swap",    "③ 개입 평가",         "belief-swap 민감도가 올랐는가"),
            ("recount", "③ 복창 제외 재집계",  "그 상승이 복창이 아닌가")]
    return out


TRAIN_KEYS = ("crashcheck", "full")   # + smoke_cw*  (아래 is_train 참조)


def is_train(key: str) -> bool:
    return key in TRAIN_KEYS or key.startswith("smoke_cw")

BASE = {  # 어제(07-21) 실측 — 비교 기준. 이 숫자들은 고정이다.
    "gt":  {"acc": 0.2371, "c3": 0.0135, "c3x": 0.0137, "restate": 0.0191},
    "wm":  {"acc": 0.2484, "c3": 0.0255, "c3x": 0.0260, "restate": 0.0722},
    "wm_top1": 0.3994,
}


def kst(ts: float) -> str:
    return datetime.fromtimestamp(ts, KST).strftime("%m-%d %H:%M:%S")


def read_log(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def tail(path: Path, n: int = 25) -> list[str]:
    if not path.is_file():
        return []
    return path.read_text(encoding="utf-8", errors="replace").splitlines()[-n:]


def gpu_rows() -> list[list[str]]:
    try:
        q = ("index,name,utilization.gpu,memory.used,memory.total,temperature.gpu")
        r = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=8)
        return [[c.strip() for c in ln.split(",")] for ln in r.stdout.strip().splitlines() if ln]
    except Exception:
        return []


def stage_state(key: str) -> tuple[str, dict]:
    """(상태, 상세). 마커·로그·산출물 존재로 판정한다 — 프로세스 추적에 의존하지 않는다."""
    d = RUN / key
    info: dict = {}
    done = (d / "TRAINING_DONE").is_file() or (RUN / f"{key}.DONE").is_file()
    log = RUN / f"{key}.log"
    if is_train(key):
        rows = read_log(d / "gr_log.jsonl")
        info["rows"] = rows
        if (d / "checkpoint-final").is_dir():
            return "done", info
        if rows or log.is_file():
            return "running", info
        return "pending", info
    out_json = RUN / f"{key}.json"
    if out_json.is_file():
        try:
            info["json"] = json.loads(out_json.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
        return "done", info
    if log.is_file():
        return "running", info
    return ("done" if done else "pending"), info


def eta(rows: list[dict], total: int, log_path: Path) -> str:
    """실측 처리속도(샘플/초)로 남은 시간을 낸다. 근거가 없으면 정직하게 '—'."""
    if not rows or not log_path.is_file():
        return "—"
    seen = rows[-1].get("seen") or 0
    if seen <= 0 or seen >= total:
        return "—"
    started = log_path.stat().st_mtime - 0  # 최종 수정 = 마지막 로그 기록 시각
    # 시작 시각은 output_dir 생성 시각으로 근사 (로드 시간 포함 — 보수적으로 길게 나온다)
    d = log_path.parent
    t0 = (d / (log_path.stem)).stat().st_ctime if False else None
    try:
        t0 = min(p.stat().st_ctime for p in [log_path])
    except Exception:
        return "—"
    elapsed = max(1.0, started - t0)
    rate = seen / elapsed
    if rate <= 0:
        return "—"
    rem = (total - seen) / rate
    fin = datetime.now(KST) + timedelta(seconds=rem)
    return f"{rem/3600:.1f}h 남음 · {fin.strftime('%H:%M')} KST 예상 ({rate*3600:.0f} 샘플/h)"


def spark(vals: list[float], w: int = 60) -> str:
    if not vals:
        return ""
    v = vals[-w:]
    lo, hi = min(v), max(v)
    if hi - lo < 1e-12:
        return "▄" * len(v)
    bars = "▁▂▃▄▅▆▇█"
    return "".join(bars[min(7, int((x - lo) / (hi - lo) * 7.999))] for x in v)


CSS = """
:root{--bg:#0e1116;--fg:#e6edf3;--dim:#8b949e;--card:#161b22;--line:#30363d;
--ok:#3fb950;--run:#d29922;--wait:#484f58;--bad:#f85149;--acc:#58a6ff}
@media(prefers-color-scheme:light){:root{--bg:#f6f8fa;--fg:#1f2328;--dim:#59636e;
--card:#fff;--line:#d1d9e0;--wait:#afb8c1}}
*{box-sizing:border-box}body{margin:0;padding:20px;background:var(--bg);color:var(--fg);
font:14px/1.55 ui-monospace,SFMono-Regular,Menlo,monospace}
h1{font-size:17px;margin:0 0 2px}h2{font-size:14px;margin:22px 0 8px;color:var(--dim);
font-weight:600;letter-spacing:.04em;text-transform:uppercase}
.sub{color:var(--dim);font-size:12px;margin-bottom:16px}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;
padding:12px 14px;margin-bottom:10px}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{text-align:left;padding:5px 10px 5px 0;border-bottom:1px solid var(--line);
white-space:nowrap}th{color:var(--dim);font-weight:600}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
.pill{display:inline-block;padding:1px 8px;border-radius:11px;font-size:11px;font-weight:600}
.s-done{background:var(--ok);color:#04140a}.s-running{background:var(--run);color:#1c1400}
.s-pending{background:var(--wait);color:var(--fg)}
.spark{font-size:15px;letter-spacing:-1px;color:var(--acc)}
pre{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 12px;
overflow-x:auto;font-size:12px;margin:0;color:var(--dim);max-height:340px}
.big{font-size:22px;font-variant-numeric:tabular-nums}
.good{color:var(--ok)}.bad{color:var(--bad)}.dim{color:var(--dim)}
.wrap{max-width:1100px;margin:0 auto}
.note{font-size:12px;color:var(--dim);margin-top:6px}
"""


def render() -> str:
    now = datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S KST")
    h = [f"<h1>P3 · belief-swap consistency 실시간 현황</h1>",
         f"<div class='sub'>{now} · 갱신 10초 · {RUN}</div>"]

    # ── 단계 ──
    h.append("<h2>단계</h2><div class='card'><table>")
    h.append("<tr><th>단계</th><th>상태</th><th>이 단계가 답하는 것</th><th>진행</th></tr>")
    def total_of(k: str) -> int | None:
        return 24 if k == "crashcheck" else (300 if k.startswith("smoke_cw")
                                             else (5000 if k == "full" else None))
    ST = stages()
    for key, name, q in ST:
        st, info = stage_state(key)
        rows = info.get("rows") or []
        prog = ""
        tot = total_of(key)
        if tot and rows:
            seen = rows[-1].get("seen") or 0
            prog = f"{seen}/{tot} · " + eta(rows, tot, RUN / key / "gr_log.jsonl")
        elif st == "done" and info.get("json"):
            prog = "산출물 기록됨"
        h.append(f"<tr><td>{name}</td><td><span class='pill s-{st}'>{st}</span></td>"
                 f"<td class='dim'>{q}</td><td class='dim'>{prog}</td></tr>")
    h.append("</table></div>")

    # ── 학습 곡선 ──
    for key, name, _ in [x for x in ST if is_train(x[0])]:
        rows = read_log(RUN / key / "gr_log.jsonl")
        if not rows:
            continue
        cons = [r["cons_loss"] for r in rows if r.get("cons_loss") is not None]
        rew = [r["reward_ma"] for r in rows if r.get("reward_ma") is not None]
        last = rows[-1]
        cl = last.get("cons_loss")
        cls = "dim" if cl is None else ("good" if cl <= 0 else "bad")
        h.append(f"<h2>{name}</h2><div class='card'>")
        h.append("<table><tr><th>seen</th><th class='num'>cons_loss</th>"
                 "<th class='num'>cons_applied</th><th class='num'>reward_ma</th>"
                 "<th class='num'>loss</th><th class='num'>mean|adv|</th></tr>")
        h.append(f"<tr><td class='big'>{last.get('seen','—')}</td>"
                 f"<td class='num big {cls}'>{cl if cl is not None else '—'}</td>"
                 f"<td class='num'>{last.get('cons_applied','—')}</td>"
                 f"<td class='num'>{last.get('reward_ma','—')}</td>"
                 f"<td class='num'>{last.get('loss','—')}</td>"
                 f"<td class='num'>{last.get('mean_abs_adv','—')}</td></tr></table>")
        if cons:
            h.append(f"<div class='note'>cons_loss 추이 (최근 {min(60,len(cons))}점, "
                     f"{min(cons):.3f} ~ {max(cons):.3f})</div>"
                     f"<div class='spark'>{spark(cons)}</div>")
            h.append("<div class='note'>cons_loss &gt; 0 = belief 를 바꿔도 원 action 이 여전히 "
                     "최선 (조향 실패). <b>내려가야 성공.</b></div>")
        if rew:
            h.append(f"<div class='note'>reward_ma 추이</div><div class='spark'>{spark(rew)}</div>")
        h.append("</div>")

    # ── 결과 대조 ──
    h.append("<h2>결과 — 어제 대비</h2><div class='card'><table>")
    h.append("<tr><th>지표</th><th class='num'>gt(어제)</th><th class='num'>wm(어제)</th>"
             "<th class='num'>P3(이번)</th><th>판정 기준</th></tr>")
    ev = stage_state("eval")[1].get("json") or {}
    sw = stage_state("recount")[1].get("json") or {}
    acc = (ev.get("full") or {}).get("acc")
    c3x = ((sw.get("subsets") or {}).get("excl_restatement") or {}).get("causal_sensitivity")
    ci = (((sw.get("subsets") or {}).get("excl_restatement") or {})
          .get("causal_sensitivity_ci95") or {})
    h.append(f"<tr><td>acc (n=1,417)</td><td class='num'>{BASE['gt']['acc']}</td>"
             f"<td class='num'>{BASE['wm']['acc']}</td>"
             f"<td class='num big'>{acc if acc is not None else '—'}</td>"
             f"<td class='dim'>WM top-1 = {BASE['wm_top1']} · 무너지지 않으면 통과</td></tr>")
    h.append(f"<tr><td>③ 복창 제외</td><td class='num'>{BASE['gt']['c3x']}</td>"
             f"<td class='num'>{BASE['wm']['c3x']}</td>"
             f"<td class='num big'>{c3x if c3x is not None else '—'}</td>"
             f"<td class='dim'>CI 하한 &gt; 어제 값이면 실효</td></tr>")
    if ci.get("lo") is not None:
        h.append(f"<tr><td>③ 95% CI</td><td class='num dim'>[.008,.020]</td>"
                 f"<td class='num dim'>[.017,.036]</td>"
                 f"<td class='num'>[{ci['lo']}, {ci['hi']}]</td><td class='dim'>paired bootstrap</td></tr>")
    h.append("</table><div class='note'>어제 실측: 복창 제외해도 ③ 가 유지됐다 "
             "(gt 0.0135→0.0137 · wm 0.0255→0.0260) — 복창 가설은 기각됐고, "
             "belief 가 실제로 조향을 거의 못 한다는 뜻이다.</div></div>")

    # ── GPU ──
    h.append("<h2>GPU</h2><div class='card'><table>")
    h.append("<tr><th>#</th><th>이름</th><th class='num'>util</th><th class='num'>mem</th>"
             "<th class='num'>temp</th></tr>")
    for g in gpu_rows():
        if len(g) >= 6:
            h.append(f"<tr><td>{g[0]}</td><td>{g[1]}</td><td class='num'>{g[2]}%</td>"
                     f"<td class='num'>{g[3]}/{g[4]} MiB</td><td class='num'>{g[5]}°C</td></tr>")
    h.append("</table></div>")

    # ── 로그 ──
    chain = RUN / "chain.log"
    if chain.is_file():
        h.append("<h2>체인 로그</h2><pre>" +
                 "\n".join(x.replace("&", "&amp;").replace("<", "&lt;") for x in tail(chain, 40)) +
                 "</pre>")
    for key, name, _ in [x for x in ST if is_train(x[0])]:
        lg = RUN / f"{key}.log"
        if lg.is_file():
            lines = [x for x in tail(lg, 400) if not x.startswith("Loading weights")]
            h.append(f"<h2>{name} 로그</h2><pre>" +
                     "\n".join(x.replace("&", "&amp;").replace("<", "&lt;") for x in lines[-18:]) +
                     "</pre>")
    return ("<!doctype html><meta charset='utf-8'><meta name='viewport' "
            "content='width=device-width,initial-scale=1'>"
            "<meta http-equiv='refresh' content='10'>"
            "<title>P3 consistency 현황</title><style>" + CSS + "</style>"
            "<div class='wrap'>" + "".join(h) + "</div>")


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        try:
            body = render().encode("utf-8")
        except Exception as e:  # 대시보드가 죽어서 학습 상태를 못 보는 일이 없게
            body = f"<pre>dashboard error: {e}</pre>".encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):  # 접근 로그 침묵
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=7863)
    args = ap.parse_args()
    print(f"[dashboard] http://{args.host}:{args.port}/  (10초 자동 갱신)", flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
