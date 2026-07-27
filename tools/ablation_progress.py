#!/usr/bin/env python3
"""ablation_progress — 실행 큐의 잔여 시간을 **실측 기반**으로 산출하고 UI 를 갱신한다.

근거:
  · 완료 항목  : timeline.jsonl 의 start/done 타임스탬프 차이 = 실측 소요
  · 진행 항목  : start 이후 경과
  · 대기 항목  : 같은 부류에서 이미 측정된 평균으로 **재보정**. 측정치가 없으면 사전 추정치.
사전 추정치는 오늘 실측한 단가에서 나왔다 (battery 1000 = 10분, harden 400 = 10분,
SFT 마이크로스텝 0.393s / CE 3.305s, accum=8).

출력:
  runs/ablation_v2/progress.json      기계 판독용
  runs/ablation_v2/progress.html      아티팩트로 게시하는 자립 HTML (클라이언트 측 실시간 카운트다운)

사용:
  python tools/ablation_progress.py                # 갱신 + 요약 출력
  python tools/ablation_progress.py --coverage_only  # K 커버리지만 JSON 으로
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QD = ROOT / "runs" / "ablation_v2"
TL = QD / "timeline.jsonl"
FP_C = ROOT / "runs" / "cesft_v2_fp_c" / "eval"

# (id, 표시명, 부류, 사전추정 분) — run_ablations_v2.sh 의 실행 순서와 일치해야 한다
QUEUE = [
    ("A_train",              "ρ=0 학습 (294 스텝)",          "train",     17),
    ("A_battery",            "ρ=0 battery (n=1000)",         "battery",   10),
    ("A_harden",             "ρ=0 harden (n=400)",           "harden",    10),
    ("B_theta_ce_k5",        "K=5 · θ_CE",                   "kabl",       9),
    ("B_sft_r15_c_k5",       "K=5 · sft_r15_c",              "kabl",       9),
    ("B_theta_ce_k3",        "K=3 · θ_CE",                   "kabl",       8),
    ("B_sft_r15_c_k3",       "K=3 · sft_r15_c",              "kabl",       8),
    ("C_base_noimage",       "no-image · base",              "noimage",    7),
    ("C_cand_free_noimage",  "no-image · GT-only",           "noimage",    7),
    ("C_theta_ce_noimage",   "no-image · θ_CE",              "noimage",    7),
    ("C_sft_r15_c_noimage",  "no-image · sft_r15_c",         "noimage",    7),
    ("C_base_nohist_noimage",      "no-image ∧ no-hist · base",      "noimage2", 7),
    ("C_cand_free_nohist_noimage", "no-image ∧ no-hist · GT-only",   "noimage2", 7),
    ("C_theta_ce_nohist_noimage",  "no-image ∧ no-hist · θ_CE",      "noimage2", 7),
    ("C_sft_r15_c_nohist_noimage", "no-image ∧ no-hist · sft_r15_c", "noimage2", 7),
    ("C_base_othervideo",       "other-video hist · base",      "othervid", 10),
    ("C_cand_free_othervideo",  "other-video hist · GT-only",   "othervid", 10),
    ("C_theta_ce_othervideo",   "other-video hist · θ_CE",      "othervid", 10),
    ("C_sft_r15_c_othervideo",  "other-video hist · sft_r15_c", "othervid", 10),
]
GROUP = {"P": "Plan B · harden_paired 4-arm (belief 교란 통제 + G_CC2)",
         "A": "A · ρ=0 대조군 (main.tex L205 약속)",
         "B": "B · K ablation (main.tex L289 tab:kablation)",
         "C": "C · 축소 Tier 1 — image/history 식별"}
PLANB_ARMS = ["base", "theta_ce", "sft_r15", "sft_r15_c"]
PLANB_LOG = ROOT / "runs" / "cesft_v2_fp_c" / "logs" / "run_planB.log"
PLANB_STAGE_RE = __import__("re").compile(
    r"^\[PB (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\] ==== (plan|run:([A-Za-z0-9_]+)|agg)")


def read_timeline() -> dict:
    ev: dict[str, dict] = {}
    if TL.is_file():
        for line in TL.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            ev.setdefault(r["item"], {})[r["event"]] = r["ts"]
    return ev


def _planb_starts() -> dict:
    """run_planB.log 의 스테이지 시작 시각. strptime+mktime 이라 서버 TZ 와 자동으로 일치한다."""
    out = {}
    if not PLANB_LOG.is_file():
        return out
    for line in PLANB_LOG.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = PLANB_STAGE_RE.match(line)
        if m:
            ts = time.mktime(time.strptime(m.group(1), "%Y-%m-%d %H:%M:%S"))
            out.setdefault("plan" if m.group(2) == "plan" else m.group(2), ts)
    return out


def planb_items() -> list[dict]:
    """Plan B 를 큐 항목으로 전개 — 산출 파일 mtime 과 로그 시작 시각이 곧 실측이다."""
    starts = _planb_starts()
    plan_f = FP_C / "harden_paired_plan.json"
    summ_f = FP_C / "harden_paired_summary.json"
    items = [{"id": "P_plan", "name": "plan · 공통셋 300 + 의역 1,200건", "cls": "planb_plan",
              "start": starts.get("plan"),
              "done": plan_f.stat().st_mtime if plan_f.is_file() else None, "est": 1}]
    prev = plan_f.stat().st_mtime if plan_f.is_file() else starts.get("plan")
    for a in PLANB_ARMS:
        f = FP_C / f"{a}.harden_paired.json"
        d = f.stat().st_mtime if f.is_file() else None
        items.append({"id": f"P_{a}", "name": f"run · {a} (7 variant × 300)", "cls": "planb_arm",
                      "start": starts.get(f"run:{a}", prev), "done": d, "est": 55})
        if d:
            prev = d
    items.append({"id": "P_agg", "name": "agg · arm간 paired + G_CC2 판정", "cls": "planb_agg",
                  "start": starts.get("agg"),
                  "done": summ_f.stat().st_mtime if summ_f.is_file() else None, "est": 1})
    return items


def build() -> dict:
    ev = read_timeline()
    now = time.time()

    # Plan B + ablation 큐를 하나의 목록으로 합친다 — 실제 실행 순서 그대로.
    raw = [{"id": p["id"], "name": p["name"], "cls": p["cls"], "est": p["est"],
            "start": p["start"], "done": p["done"], "group": "P"} for p in planb_items()]
    for iid, name, cls, est in QUEUE:
        e = ev.get(iid, {})
        raw.append({"id": iid, "name": name, "cls": cls, "est": est,
                    "start": e.get("start"), "done": e.get("done"), "group": iid[0]})

    # 부류별 실측 평균으로 재보정
    actual: dict[str, list[float]] = {}
    for r in raw:
        if r["start"] and r["done"]:
            actual.setdefault(r["cls"], []).append((r["done"] - r["start"]) / 60)
    cal = {c: sum(v) / len(v) for c, v in actual.items()}

    items, remaining, done_n = [], 0.0, 0
    running_seen = False
    for r in raw:
        est = cal.get(r["cls"], r["est"])
        if r["done"]:
            st = "done"
            mins = ((r["done"] - r["start"]) / 60) if r["start"] else est
            src = "실측" if r["start"] else "추정"
            done_n += 1
        elif r["start"] and not running_seen:
            st, mins, src = "running", (now - r["start"]) / 60, "경과"
            running_seen = True
            remaining += max(0.0, est - mins)
        else:
            st, mins = "pending", est
            src = "실측 보정" if r["cls"] in cal else "사전 추정"
            remaining += mins
        items.append({"id": r["id"], "name": r["name"], "cls": r["cls"], "state": st,
                      "minutes": round(mins, 1), "src": src, "group": r["group"]})

    pb_done = [i for i in items if i["group"] == "P" and i["state"] == "done"]
    pb_arms_done = [i for i in pb_done if i["cls"] == "planb_arm"]
    return {
        "generated_at": now,
        "planb": {"arms_done": [i["id"][2:] for i in pb_arms_done],
                  "arms_total": len(PLANB_ARMS),
                  "per_arm_min": round(cal["planb_arm"], 1) if "planb_arm" in cal else None,
                  "per_arm_is_measured": "planb_arm" in cal,
                  "summary_ready": (FP_C / "harden_paired_summary.json").is_file()},
        "queue_started": ev.get("queue", {}).get("start"),
        "queue_done": ev.get("queue", {}).get("done") is not None,
        "items": items,
        "n_done": done_n,
        "n_total": len(items),
        "remaining_min_queue": round(sum(
            i["minutes"] for i in items if i["group"] != "P" and i["state"] == "pending"), 1),
        "remaining_min_total": round(remaining, 1),
        "eta_epoch": now + remaining * 60,
        "calibrated": {k: round(v, 1) for k, v in cal.items()},
    }


HTML = """<title>EGO ablation queue — 잔여 시간</title>
<style>
:root{
  --bg:#f6f4ef; --panel:#fffdf9; --line:#e2ddd2; --ink:#22201c; --dim:#6f6a5f;
  --accent:#b4552d; --ok:#3f7a52; --run:#c08a1e; --wait:#8d8779;
  --mono:ui-monospace,SFMono-Regular,"SF Mono",Menlo,monospace;
  --sans:system-ui,-apple-system,"Segoe UI",sans-serif;
}
@media (prefers-color-scheme:dark){:root{
  --bg:#14161a; --panel:#1b1e24; --line:#2c313a; --ink:#e6e4df; --dim:#8f9299;
  --accent:#e2814f; --ok:#5fa87a; --run:#e0b040; --wait:#6d7280;}}
:root[data-theme="dark"]{--bg:#14161a;--panel:#1b1e24;--line:#2c313a;--ink:#e6e4df;--dim:#8f9299;
  --accent:#e2814f;--ok:#5fa87a;--run:#e0b040;--wait:#6d7280;}
:root[data-theme="light"]{--bg:#f6f4ef;--panel:#fffdf9;--line:#e2ddd2;--ink:#22201c;--dim:#6f6a5f;
  --accent:#b4552d;--ok:#3f7a52;--run:#c08a1e;--wait:#8d8779;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);
     font-size:15px;line-height:1.55;-webkit-font-smoothing:antialiased}
.wrap{max-width:940px;margin:0 auto;padding:28px 20px 64px;display:flex;flex-direction:column;gap:22px}
header{display:flex;flex-direction:column;gap:4px}
h1{margin:0;font-size:1.15rem;font-weight:620;letter-spacing:-.01em}
.sub{color:var(--dim);font-size:.85rem}
.hero{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:22px 24px;
      display:grid;grid-template-columns:1fr auto;gap:18px;align-items:end}
.big{font-family:var(--mono);font-size:2.9rem;font-weight:600;letter-spacing:-.03em;
     font-variant-numeric:tabular-nums;line-height:1;color:var(--accent)}
.lab{font-size:.72rem;text-transform:uppercase;letter-spacing:.09em;color:var(--dim);margin-bottom:6px}
.eta{font-family:var(--mono);font-size:1.35rem;font-variant-numeric:tabular-nums;text-align:right}
.bar{height:7px;background:var(--line);border-radius:4px;overflow:hidden;grid-column:1/-1}
.bar i{display:block;height:100%;background:var(--accent);border-radius:4px;transition:width .6s}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:13px 15px}
.card .v{font-family:var(--mono);font-size:1.3rem;font-variant-numeric:tabular-nums}
h2{margin:6px 0 0;font-size:.78rem;text-transform:uppercase;letter-spacing:.09em;color:var(--dim);font-weight:600}
.scroll{overflow-x:auto}
table{width:100%;border-collapse:collapse;font-size:.87rem;min-width:520px}
th{text-align:left;font-weight:600;font-size:.7rem;text-transform:uppercase;letter-spacing:.07em;
   color:var(--dim);padding:7px 10px;border-bottom:1px solid var(--line)}
td{padding:8px 10px;border-bottom:1px solid var(--line)}
tr:last-child td{border-bottom:none}
td.n{font-family:var(--mono);font-variant-numeric:tabular-nums;text-align:right;white-space:nowrap}
.pill{display:inline-block;font-size:.7rem;padding:2px 8px;border-radius:99px;font-weight:600;
      border:1px solid currentColor}
.s-done{color:var(--ok)} .s-running{color:var(--run)} .s-pending{color:var(--wait)}
tr.running td{background:color-mix(in srgb,var(--run) 9%,transparent)}
.grp td{padding-top:16px;font-size:.72rem;text-transform:uppercase;letter-spacing:.08em;
        color:var(--dim);font-weight:600;border-bottom:none}
.src{color:var(--dim);font-size:.75rem}
pre{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px 16px;
    overflow-x:auto;font-family:var(--mono);font-size:.8rem;margin:0}
.note{color:var(--dim);font-size:.82rem}
code{font-family:var(--mono);font-size:.85em}
</style>
<div class="wrap">
<header>
  <h1>EGO ablation queue</h1>
  <div class="sub">defense plan v2 순서 1·2·8 — 잔여 시간은 완료 항목의 <strong>실측 소요</strong>로 재보정됩니다.</div>
</header>

<div class="hero">
  <div><div class="lab">남은 시간 (전체)</div><div class="big" id="rem">—</div></div>
  <div><div class="lab">완료 예상 (KST)</div><div class="eta" id="eta">—</div></div>
  <div class="bar"><i id="bar" style="width:0%"></i></div>
</div>

<div class="grid">
  <div class="card"><div class="lab">Plan B</div><div class="v" id="pb">—</div><div class="src" id="pbs"></div></div>
  <div class="card"><div class="lab">Ablation 큐</div><div class="v" id="qn">—</div><div class="src" id="qs"></div></div>
  <div class="card"><div class="lab">현재 작업</div><div class="v" id="cur" style="font-size:.95rem">—</div><div class="src" id="curs"></div></div>
  <div class="card"><div class="lab">데이터 기준</div><div class="v" id="gen" style="font-size:.95rem">—</div><div class="src">KST</div></div>
</div>

<h2>큐</h2>
<div class="scroll"><table><thead><tr>
  <th>항목</th><th>상태</th><th style="text-align:right">소요 / 예상</th><th>근거</th>
</tr></thead><tbody id="rows"></tbody></table></div>

<h2>실측 확인 (서버에서)</h2>
<pre>tail -f runs/ablation_v2/timeline.jsonl        # 단계 전환 실측 타임라인
python tools/ablation_progress.py              # 이 페이지의 원본 수치 재계산
cat runs/cesft_v2_fp_c/eval/harden_paired_summary.json   # Plan B 최종 판정</pre>

<p class="note" id="foot"></p>
</div>
<script>
const D = __DATA__;
const KST = t => { const d = new Date((t + 32400) * 1000);
  return d.getUTCFullYear() + "-" + String(d.getUTCMonth()+1).padStart(2,"0") + "-" +
    String(d.getUTCDate()).padStart(2,"0") + " " + String(d.getUTCHours()).padStart(2,"0") + ":" +
    String(d.getUTCMinutes()).padStart(2,"0"); };
const HM = m => { if (m <= 0) return "완료"; const h = Math.floor(m/60), mm = Math.round(m%60);
  return h ? h + "시간 " + String(mm).padStart(2,"0") + "분" : mm + "분"; };
const GRP = __GROUPS__;

function rows(){
  const tb = document.getElementById("rows"); tb.innerHTML = "";
  let last = null;
  for (const it of D.items){
    if (it.group !== last){ last = it.group;
      const tr = document.createElement("tr"); tr.className = "grp";
      tr.innerHTML = '<td colspan="4">' + GRP[it.group] + '</td>'; tb.appendChild(tr); }
    const tr = document.createElement("tr");
    if (it.state === "running") tr.className = "running";
    const lab = {done:"완료", running:"진행 중", pending:"대기"}[it.state];
    tr.innerHTML = '<td>' + it.name + '</td>' +
      '<td><span class="pill s-' + it.state + '">' + lab + '</span></td>' +
      '<td class="n">' + it.minutes.toFixed(1) + '분</td>' +
      '<td class="src">' + it.src + '</td>';
    tb.appendChild(tr);
  }
}
const SPENT = D.items.filter(i=>i.state!=="pending").reduce((a,b)=>a+b.minutes,0);
function tick(){
  const drift = (Date.now()/1000) - D.generated_at;          // 페이지가 열려 있는 동안 계속 감산
  const rem = Math.max(0, D.remaining_min_total - drift/60);
  document.getElementById("rem").textContent = HM(rem);
  document.getElementById("eta").textContent = rem > 0 ? KST(D.generated_at + drift + rem*60) : "완료";
  const spent = SPENT + drift/60;
  document.getElementById("bar").style.width =
    Math.min(100, Math.max(0, 100*spent/Math.max(1e-9, spent+rem))).toFixed(1) + "%";
}
const pb = D.planb;
document.getElementById("pb").textContent = pb.summary_ready ? "완료"
  : pb.arms_done.length + " / " + pb.arms_total + " arm";
document.getElementById("pbs").textContent = pb.summary_ready ? "harden_paired_summary.json 생성됨"
  : (pb.per_arm_is_measured ? "arm당 실측 " + pb.per_arm_min + "분" : "arm당 추정 55분");
document.getElementById("qn").textContent = D.n_done + " / " + D.n_total;
document.getElementById("qs").textContent = D.queue_done ? "전체 완료"
  : (D.queue_started ? "실행 중" : "Plan B 종료 대기");
const cur = D.items.find(i => i.state === "running");
document.getElementById("cur").textContent = cur ? cur.name : (D.queue_done ? "없음" : "대기 중");
document.getElementById("curs").textContent = cur ? cur.minutes.toFixed(1) + "분 경과" : "";
document.getElementById("gen").textContent = KST(D.generated_at);
document.getElementById("foot").textContent =
  "대기 항목의 예상치는 같은 부류에서 이미 완료된 항목의 실측 평균으로 대체됩니다"
  + (Object.keys(D.calibrated).length ? " (현재 보정된 부류: " + Object.keys(D.calibrated).join(", ") + ")." : ".")
  + " 페이지는 열려 있는 동안 스스로 카운트다운하며, 서버의 실제 진행은 위 명령으로 확인할 수 있습니다.";
rows(); tick(); setInterval(tick, 1000);
</script>
"""


def write_html(data: dict) -> Path:
    html = (HTML.replace("__DATA__", json.dumps(data, ensure_ascii=False))
                .replace("__GROUPS__", json.dumps(GROUP, ensure_ascii=False)))
    p = QD / "progress.html"
    p.write_text(html, encoding="utf-8")
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coverage_only", action="store_true")
    args = ap.parse_args()
    if args.coverage_only:
        ctx = ROOT / "runs" / "cesft_v2_fp" / "data" / "context_val.jsonl"
        h = [json.loads(l) for l in open(ctx, encoding="utf-8") if l.strip()]
        h = [r for r in h if r["split"] == "heldout"]
        out = {f"coverage_at_{K}": {"rate": round(sum(1 for r in h if r["gt_rank"] <= K) / len(h), 4),
                                    "n_covered": sum(1 for r in h if r["gt_rank"] <= K)}
               for K in (3, 5, 10)}
        out["n_heldout"] = len(h)
        print(json.dumps(out, indent=1))
        return
    QD.mkdir(parents=True, exist_ok=True)
    d = build()
    (QD / "progress.json").write_text(json.dumps(d, indent=1, ensure_ascii=False))
    write_html(d)
    pb = d["planb"]
    cur = next((i for i in d["items"] if i["state"] == "running"), None)
    print(f"Plan B: {len(pb['arms_done'])}/{pb['arms_total']} arm "
          f"(arm당 {'실측 ' + str(pb['per_arm_min']) if pb['per_arm_is_measured'] else '추정 55'}분)")
    print(f"전체: {d['n_done']}/{d['n_total']} 완료"
          + (f", 현재 «{cur['name']}» {cur['minutes']:.0f}분 경과" if cur else ""))
    print(f"잔여 {d['remaining_min_total']:.0f}분 → 완료 예상 "
          f"{time.strftime('%m-%d %H:%M', time.gmtime(d['eta_epoch'] + 32400))} KST")


if __name__ == "__main__":
    main()
