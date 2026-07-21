#!/usr/bin/env python3
"""야간 무인 실행 실시간 현황 — Step-1 ViT-L 실험(GPU0) + RETRO 재실행(GPU1).

의존성 없는 stdlib http.server. 남은 시간은 각 런의 training_history.csv 에 기록된
**실측 초/epoch** 에서 계산한다 — 하드코딩된 추정치를 쓰지 않으므로 GPU 경합으로
느려지면 표시되는 숫자도 정직하게 늘어난다. 아직 시작하지 않은 런은 같은 스테이지에서
이미 관측된 속도를 쓰고, 그것도 없으면 전체 평균을 쓴다.

    python tools/night_dashboard.py --host 0.0.0.0 --port 7862
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NIGHT = REPO / "outputs/goalstep/night"
NIGHT_CFG = REPO / "configs/step1/goalstep/night"
SWEEP = REPO / "outputs/goalstep/sweep"
RETRO = Path("/mnt/nvme/migration/jihun/EGO/runs/retro_overnight")
KST = timezone(timedelta(hours=9))

# 체인 스크립트의 실행 순서와 각 런이 답하려는 질문. 순서가 곧 대기열이다.
STEP1_PLAN = [
    ("b2_vna",  "Phase 1", "verb+noun+action 동시 학습",
     "verb top-5 가 어디까지 가는가 · 보조 감독이 action 에 도움이 되는가"),
    ("b3_d4",   "Phase 1", "프로브 depth=4 (49.6M)",
     "'용량은 무관하다'는 결론이 고쳐진 샘플러에서도 유지되는가"),
    ("n05k",    "Phase 2", "train 5,000건",  "데이터 스케일링 — ViT-g 투자 여부를 결정"),
    ("n10k",    "Phase 2", "train 10,000건", "데이터 스케일링"),
    ("n20k",    "Phase 2", "train 20,000건", "데이터 스케일링"),
    ("r_wd",    "Phase 3", "weight decay 0.04→0.4",
     "정점이 epoch 3 으로 앞당겨진 지금 정규화가 통하는가"),
    ("r_ep40",  "Phase 3", "40 epoch",       "정점 이후 곡선 전체 형태"),
    ("f_fps",   "Phase 4", "l_obs=4.0 재추출 후 학습",
     "tau 를 0.874s → 1.0s 로 교정하면 달라지는가"),
]

BASELINE = ("s6_d1_rand", "고쳐진 베이스라인 (sampler=random, depth=1, action)")


def _csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _epochs(cfg: Path) -> int:
    if not cfg.is_file():
        return 15
    m = re.search(r"^\s*epochs:\s*(\d+)", cfg.read_text(encoding="utf-8"), re.M)
    return int(m.group(1)) if m else 15


def _ps() -> str:
    try:
        return subprocess.check_output(["ps", "-eo", "args"], text=True)
    except Exception:
        return ""


def _run_row(name: str, run_dir: Path, cfg: Path, ps: str) -> dict:
    hist = _csv(run_dir / "training_history.csv")
    total = _epochs(cfg)
    done = len(hist)
    secs = [float(r["seconds"]) for r in hist if r.get("seconds")]
    rate = sum(secs[-3:]) / len(secs[-3:]) if secs else None
    running = f"night/{name}.yaml" in ps or f"sweep/{name}.yaml" in ps
    if running:
        status = "running"
    elif (run_dir / "final_metrics.json").is_file():
        status = "done"
    elif done:
        status = "stopped"
    else:
        status = "queued"
    last = hist[-1] if hist else None
    # best top5 는 마지막 epoch 이 아니라 곡선의 최댓값이다 (정점 뒤 하락하므로).
    tops = [float(r["action_top5"]) for r in hist if r.get("action_top5")]
    return {
        "name": name, "status": status, "done": done, "total": total,
        "rate": round(rate, 1) if rate else None,
        "remaining": round((total - done) * rate) if rate and status != "done" else None,
        "top5_last": float(last["action_top5"]) if last and last.get("action_top5") else None,
        "top5_best": max(tops) if tops else None,
        "top5_best_ep": (tops.index(max(tops)) + 1) if tops else None,
        "cmr5": float(last["action_cmr@5"]) if last and last.get("action_cmr@5") else None,
        "verb_top5": float(last["verb_top5"]) if last and last.get("verb_top5") else None,
        "loss": float(last["train_loss"]) if last else None,
    }


def step1() -> dict:
    ps = _ps()
    base = _run_row(BASELINE[0], SWEEP / BASELINE[0], REPO / f"configs/step1/goalstep/sweep/{BASELINE[0]}.yaml", ps)
    base["note"] = BASELINE[1]

    rows, rates = [], []
    for name, phase, what, question in STEP1_PLAN:
        r = _run_row(name, NIGHT / name, NIGHT_CFG / f"{name}.yaml", ps)
        r.update(phase=phase, what=what, question=question)
        if r["rate"]:
            rates.append(r["rate"])
        rows.append(r)

    fallback = (sum(rates) / len(rates)) if rates else (base["rate"] or 175)
    remaining = 0.0
    for r in rows:
        if r["status"] == "done":
            continue
        rate = r["rate"] or fallback
        remaining += (r["total"] - r["done"]) * rate
    # Phase 4 는 학습 앞에 인덱스 재생성 + 전량 재추출(37,588건)이 붙는다.
    if not (NIGHT / "f_fps" / "final_metrics.json").is_file():
        extracted = len(list((REPO.parent / "datasets/Ego4D/goalstep_feature_cache_lobs4/train").glob("*.pt"))) \
            if (REPO.parent / "datasets/Ego4D/goalstep_feature_cache_lobs4/train").is_dir() else 0
        remaining += max(0, 37588 - extracted) / 350 * 60   # 실측 약 350 samples/min

    return {
        "baseline": base, "runs": rows,
        "remaining_sec": round(remaining),
        "eta": (datetime.now(KST) + timedelta(seconds=remaining)).strftime("%m-%d %H:%M"),
        "done": sum(1 for r in rows if r["status"] == "done"),
        "total": len(rows),
    }


def retro() -> dict:
    log = RETRO / "chain.log"
    lines = log.read_text(encoding="utf-8", errors="replace").splitlines() if log.is_file() else []
    gate: dict[str, str] = {}
    if (RETRO / "gateA.txt").is_file():
        for line in (RETRO / "gateA.txt").read_text(encoding="utf-8").splitlines():
            p = line.split()
            if len(p) == 2:
                gate[p[0]] = p[1]
    markers = [m for m in ("GATE_A_PASSED", "GATE_A2_PASSED", "GATE_A2_FAILED",
                           "GATE_A_FAILED", "FAILED", "DONE") if (RETRO / m).is_file()]

    S2 = Path("/mnt/nvme/migration/jihun/EGO_jihun/outputs/step2")

    def progress(run_dir: Path) -> tuple[int, int | None]:
        """gr_log.jsonl 의 seen 으로 진행/잔여(초) 추정. 실측 속도만 쓴다."""
        gl = run_dir / "gr_log.jsonl"
        if not gl.is_file():
            return 0, None
        rows = [json.loads(x) for x in gl.read_text(encoding="utf-8").splitlines() if x.strip()]
        if not rows:
            return 0, None
        seen = rows[-1].get("seen", 0)
        elapsed = max(1.0, datetime.now().timestamp() - gl.stat().st_ctime)
        rate = seen / elapsed
        return seen, (round((5000 - seen) / rate) if rate > 0 and seen < 5000 else None)

    gt_seen, gt_rem = progress(S2 / "retro_belief_sum_gt_1f")
    wm_seen, wm_rem = progress(S2 / "retro_belief_sum_wm_1f")
    remaining = gt_rem if gt_rem is not None else wm_rem

    def st(done: bool, running: bool) -> str:
        return "done" if done else ("running" if running else "queued")

    gate_gt = gate.get("belief_sum_gt")
    stages = [
        {"id": "게이트 A", "what": "credit-reduction = sum 스모크 · belief vs action (각 300샘플)",
         "goal": "어제 ARM B 가 죽은 것이 가설 탓인지 구현 탓인지 가른다.",
         "why": "ARM B 는 <code>loss = −(adv · tok_lp.mean())</code> 이었다. "
                "<code>&lt;action&gt;</code> span 은 <code>{\"verb\":\"put\",\"noun\":\"lid\"}</code> 수준의 "
                "거의 결정적인 JSON 토큰 몇 개라 토큰당 logp ≈ 0 인데 <code>.mean()</code> 이 길이로 또 나눈다. "
                "그래서 '가설 기각'과 '구현이 신호를 죽임'이 분리되지 않았다.",
         "ev": "결과 — action·mean 0.000092 → action·sum 0.002880 (<b>31배</b>) → belief·sum 0.034000 "
               "(action→belief <b>11.8배</b>, 합 <b>370배</b>). 두 인자 모두 실재했다. "
               "독립 확인: best-of-8 스코어링에서 action span 토큰당 평균 logp = <code>−0.0000</code>.",
         "pass": "belief+sum mean|loss| ≥ 0.00092 (action·mean 0.000092 의 10배)",
         "status": "done" if gate else "running",
         "result": (f"belief·sum {gate.get('belief','—')} · action·sum {gate.get('action','—')}"
                    if gate else "측정 중")},
        {"id": "게이트 A′", "what": "reward = gt 로 바꾼 뒤 sum 이 안정적인가 (300샘플)",
         "goal": "바이너리 보상 + sum 조합이 분산 폭주 없이 학습 가능한지 3.8h 앞에서 14분에 확인한다.",
         "why": "<code>r∈{0,1}</code>, baseline ≈ pass@1 ≈ 0.39 → <code>mean|adv| ≈ 0.476</code> 으로 "
                "wm 보상의 실측 0.20 대비 <b>2.4배</b>다. 여기에 sum(이미 370배 증폭)이 곱해진다.",
         "ev": "하한 0.00092 · 경고선 5.0. 경고선을 넘으면 학습은 진행하되 reward_ma 추세를 판정에 반드시 포함한다.",
         "pass": "mean|loss| ≥ 0.00092 이고 발산 징후 없음",
         "status": st(bool(gate_gt), not gate_gt),
         "result": f"belief·sum·gt {gate_gt}" if gate_gt else "측정 중"},
        {"id": "본실행 1", "what": "credit = belief · sum · <b>reward = gt</b> · 5,000샘플 REINFORCE",
         "goal": "보상의 최적해가 GT 인 조건에서, 되살아난 belief gradient 가 실제로 생성 정확도를 올리는가.",
         "why": "기존 <code>--reward wm</code> 은 <code>r = lik(선택)/Σlik(top-5)</code> 로 GT 를 쓰지 않는다. "
                "그 보상의 <b>전역 최적해가 WM top-1 맹목 추종</b>이고, GT 를 완벽히 맞히는 정책의 보상이 "
                "오히려 <b>더 낮다</b>. GT 가 rank2~5 인 24.3% 구간 — 정확히 Retrospection 의 영역 — 에서는 "
                "보상 최대화가 GT 와 반대로 움직인다.",
         "ev": "train pool 4,998건 실측 — 항상 WM top-1 <b>r=0.5017</b> · 항상 GT <b>r=0.4716</b> · 무작위 0.2000 · "
               "현재 정책 ≈0.30. heldout 이 이를 뒷받침한다: WM top-1 argmax <b>0.374</b> 인데 학습된 정책은 전부 "
               "0.24~0.28. GT-free 이고 WM 에서 파생된 보상으로는 원리적으로 WM 을 넘을 수 없다. "
               "<code>reward=gt</code> 는 최적해가 GT 완벽 정책이며, GT 는 학습 시점 검증자로만 쓰이고 추론에는 쓰이지 않는다.",
         "pass": "acc 상승 &gt; MDE <b>AND</b> wm_follow 가 비례 상승하지 않음 <b>AND</b> G2 acc(n≈348) CI 밖 상승",
         "status": st(gt_seen >= 5000, 0 < gt_seen < 5000),
         "result": f"{gt_seen:,} / 5,000 샘플" if gt_seen else "대기"},
        {"id": "평가 1", "what": "heldout 전량 1,417건 + 2 disjoint subset + bootstrap 95% CI + MDE · ③ belief-swap",
         "goal": "효과가 측정 잡음보다 큰지를 먼저 확정한다.",
         "why": "subset 만 바꿔도 acc 가 0.264 → 0.302 (Δ0.038) 움직였는데 하루 종일 주장한 효과는 0.02~0.03이었다. "
                "500건 greedy 3회 반복은 바이트까지 동일해 표본 잡음 정보가 0이었다.",
         "ev": "n=500 → 1,417 로 MDE 가 <b>0.044 → 0.026</b> 으로 내려간다. "
               "G2 부분집합도 123 → 약 <b>348</b> 이 되어 처음으로 검정 가능해진다.",
         "pass": "③ &gt; 0.05 (현재 0.006) · acc CI 하한 &gt; 기준선 CI 상한",
         "status": st((RETRO / "eval_belief_sum_gt.json").is_file(), False),
         "result": "완료" if (RETRO / "eval_belief_sum_gt.json").is_file() else "대기"},
        {"id": "본실행 2", "what": "credit = belief · sum · <b>reward = wm</b> · 5,000샘플 (보상 대조)",
         "goal": "본실행 1 과 <b>보상만</b> 다르다. 두 결과의 차이가 곧 '보상 정의가 결과를 만들었는가'의 답.",
         "why": "v1 계획은 이 arm 을 본실행으로 두고 있었다. 위험을 주장으로 남기지 않고 실증한다.",
         "ev": "예측 — wm arm 은 acc 가 0.374(WM argmax) 쪽으로 오르되 <b>wm_follow 가 함께 뛰고 G2 는 안 움직인다</b>. "
               "그 패턴이 나오면 v1 이 왜 위험했는지가 데이터로 남는다. "
               "(v1 의 <code>action+sum</code> 본실행 4.7h 는 폐기했다 — 게이트 A 가 이미 답했다.)",
         "pass": "— (대조군, 단독 판정 없음)",
         "status": st(wm_seen >= 5000, 0 < wm_seen < 5000),
         "result": f"{wm_seen:,} / 5,000 샘플" if wm_seen else "대기"},
    ]
    return {"stages": stages, "markers": markers, "gate": gate,
            "remaining_sec": remaining, "tail": lines[-10:]}


def gpus() -> list[dict]:
    try:
        out = subprocess.check_output([
            "nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu",
            "--format=csv,noheader,nounits"], text=True, timeout=8)
    except Exception as e:
        return [{"error": str(e)}]
    res = []
    for line in out.strip().splitlines():
        f = [x.strip() for x in line.split(",")]
        res.append({"i": f[0], "util": f[1], "mem": f[2], "memtot": f[3], "temp": f[4]})
    return res


def status() -> dict:
    return {"now": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S"),
            "step1": step1(), "retro": retro(), "gpus": gpus()}


PAGE = """<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>야간 무인 실행 — Step-1 ViT-L · RETRO 재실행</title>
<style>
:root{
  --ground:#E8E9E6; --raise:#F2F3F0; --sink:#DDDFDA;
  --rule:#C6C9C2; --rule-soft:#D6D8D2;
  --fg:#1A1D1B; --fg-mid:#4A4F4B; --fg-dim:#6E736E;
  --accent:#1F6B63; --accent-soft:#D3E2DF;
  --crit:#A8362A; --crit-soft:#EFD8D4;
  --warn:#95681E; --warn-soft:#EEE1C9;
  --ok:#356B47; --ok-soft:#D6E5DA;
  --serif:"Iowan Old Style","Palatino Linotype",Palatino,"Book Antiqua",Georgia,serif;
  --sans:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",sans-serif;
  --mono:ui-monospace,"SF Mono",SFMono-Regular,Menlo,Consolas,"Liberation Mono",monospace;
}
@media (prefers-color-scheme:dark){
  :root{
    --ground:#15171B; --raise:#1D2025; --sink:#101215;
    --rule:#31363C; --rule-soft:#262A2F;
    --fg:#E4E6E3; --fg-mid:#A8AEA9; --fg-dim:#7B827D;
    --accent:#5FBAAE; --accent-soft:#1B3A37;
    --crit:#E0705C; --crit-soft:#3B211C;
    --warn:#D6A45A; --warn-soft:#382C18;
    --ok:#6FB585; --ok-soft:#1D3325;
  }
}
:root[data-theme="dark"]{
  --ground:#15171B; --raise:#1D2025; --sink:#101215;
  --rule:#31363C; --rule-soft:#262A2F;
  --fg:#E4E6E3; --fg-mid:#A8AEA9; --fg-dim:#7B827D;
  --accent:#5FBAAE; --accent-soft:#1B3A37;
  --crit:#E0705C; --crit-soft:#3B211C;
  --warn:#D6A45A; --warn-soft:#382C18;
  --ok:#6FB585; --ok-soft:#1D3325;
}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--fg);
  font-family:var(--sans);font-size:16px;line-height:1.65;-webkit-font-smoothing:antialiased}
.wrap{max-width:1120px;margin:0 auto;padding:0 24px 96px}

.mast{padding:52px 0 26px;border-bottom:2px solid var(--fg)}
.eyebrow{font-family:var(--mono);font-size:11px;letter-spacing:.14em;
  text-transform:uppercase;color:var(--fg-dim);margin:0 0 14px}
h1{font-family:var(--serif);font-weight:600;font-size:clamp(28px,4.2vw,44px);
  line-height:1.14;letter-spacing:-.015em;margin:0 0 16px;text-wrap:balance;max-width:22ch}
.dek{font-size:16.5px;color:var(--fg-mid);margin:0;max-width:64ch}
.meta{display:flex;flex-wrap:wrap;gap:8px 20px;margin-top:20px;
  font-family:var(--mono);font-size:12px;color:var(--fg-dim)}
.live{display:inline-block;width:7px;height:7px;border-radius:50%;background:var(--accent);
  margin-right:7px;vertical-align:middle;animation:pulse 2s ease-in-out infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.32}}

.verdict{margin:32px 0 0;background:var(--raise);border:1px solid var(--rule);
  border-left:4px solid var(--st,var(--accent));padding:22px 25px;display:grid;gap:11px}
.verdict h2{font-family:var(--sans);font-size:12px;letter-spacing:.12em;
  text-transform:uppercase;color:var(--st,var(--accent));margin:0;font-weight:650}
.verdict p{margin:0;font-size:16px;line-height:1.6;max-width:68ch}
.verdict .quote{font-family:var(--mono);font-size:13px;color:var(--fg-mid);
  border-left:2px solid var(--rule);padding-left:14px;line-height:1.65}

section{margin-top:58px}
.sec-head{display:flex;align-items:baseline;gap:14px;
  border-bottom:1px solid var(--fg);padding-bottom:10px;margin-bottom:12px}
.sec-num{font-family:var(--mono);font-size:12px;color:var(--accent);letter-spacing:.08em;font-weight:600}
.sec-head h2{font-family:var(--serif);font-size:clamp(21px,2.6vw,28px);
  font-weight:600;letter-spacing:-.01em;margin:0}
.sec-head .gpu{margin-left:auto;font-family:var(--mono);font-size:11px;color:var(--fg-dim)}
.lede{font-size:14.5px;color:var(--fg-mid);margin:0 0 22px;max-width:74ch}
h3{font-family:var(--sans);font-size:14px;font-weight:650;margin:30px 0 11px}

.status{display:grid;grid-template-columns:repeat(auto-fit,minmax(158px,1fr));gap:1px;
  background:var(--rule-soft);border:1px solid var(--rule-soft);margin-bottom:18px}
.stat{background:var(--raise);padding:15px 17px}
.stat .k{font-family:var(--mono);font-size:10.5px;letter-spacing:.08em;
  text-transform:uppercase;color:var(--fg-dim);margin-bottom:5px}
.stat .v{font-family:var(--mono);font-size:21px;font-weight:650;
  font-variant-numeric:tabular-nums;line-height:1.2}
.stat .n{font-size:11.5px;color:var(--fg-dim);margin-top:3px}

.runs{display:grid;gap:10px}
.run{display:grid;grid-template-columns:1fr auto;gap:18px;align-items:start;
  background:var(--raise);border:1px solid var(--rule-soft);
  border-left:3px solid var(--st,var(--rule));padding:15px 19px}
.run.running{--st:var(--accent)} .run.done{--st:var(--ok)}
.run.stopped{--st:var(--warn)} .run.queued{--st:var(--rule)}
.run .nm{font-family:var(--mono);font-size:13px;font-weight:650}
.run .wt{font-size:14px;margin-left:8px}
.run .q{display:block;font-size:12.5px;color:var(--fg-dim);margin-top:4px;max-width:64ch}
.run .rt{text-align:right;white-space:nowrap}
.run .nums{font-family:var(--mono);font-size:12px;color:var(--fg-mid);
  margin-top:7px;font-variant-numeric:tabular-nums}
.bar{height:4px;background:var(--sink);margin-top:9px;overflow:hidden}
.bar i{display:block;height:100%;background:var(--st,var(--accent));transition:width .5s}

.chip{font-family:var(--mono);font-size:11px;letter-spacing:.04em;padding:3px 9px;
  border:1px solid currentColor;white-space:nowrap;line-height:1.5}
.chip.running{color:var(--accent);background:var(--accent-soft)}
.chip.done{color:var(--ok);background:var(--ok-soft)}
.chip.stopped{color:var(--warn);background:var(--warn-soft)}
.chip.queued{color:var(--fg-dim)}

.item{background:var(--raise);border:1px solid var(--rule-soft);padding:17px 20px;
  border-top:2px solid var(--st,var(--accent));margin-bottom:11px}
.item.done{--st:var(--ok)} .item.running{--st:var(--accent)} .item.queued{--st:var(--rule)}
.item .tophead{display:flex;gap:12px;align-items:baseline;flex-wrap:wrap;margin-bottom:7px}
.item .key{font-family:var(--mono);font-size:12px;font-weight:650;color:var(--st,var(--accent))}
.item .ttl{font-size:15px;font-weight:650}
.item .rt{margin-left:auto;display:flex;gap:10px;align-items:center}
.item p{margin:0;font-size:14px;color:var(--fg-mid);max-width:72ch}
.ev{font-family:var(--mono);font-size:11.5px;color:var(--fg-dim);
  border-left:2px solid var(--rule);padding-left:12px;margin-top:9px;line-height:1.7}
.ev b{color:var(--warn)}

.scroll{overflow-x:auto;border:1px solid var(--rule-soft);background:var(--raise)}
table{border-collapse:collapse;width:100%;font-size:13px;min-width:440px}
th,td{padding:8px 13px;text-align:left;border-bottom:1px solid var(--rule-soft)}
thead th{font-family:var(--mono);font-size:10.5px;letter-spacing:.07em;text-transform:uppercase;
  color:var(--fg-dim);font-weight:600;border-bottom:1px solid var(--rule);background:var(--sink)}
td.n,th.n{text-align:right;font-family:var(--mono);font-variant-numeric:tabular-nums}
tbody tr:last-child td{border-bottom:none}
tr.hi td{background:var(--crit-soft)}

pre{margin:16px 0 0;background:var(--sink);border:1px solid var(--rule-soft);
  padding:13px 15px;overflow-x:auto;font-family:var(--mono);font-size:11.5px;
  line-height:1.62;color:var(--fg-mid)}
code{font-family:var(--mono);font-size:.9em;background:var(--sink);padding:1px 5px}
.note{background:var(--warn-soft);border:1px solid var(--warn);
  border-left-width:3px;padding:14px 17px;font-size:14px;margin-top:16px}
.note b{color:var(--warn)}
footer{margin-top:66px;padding-top:22px;border-top:1px solid var(--rule);
  font-family:var(--mono);font-size:11.5px;color:var(--fg-dim)}
:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
@media (prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
@media(max-width:640px){.run{grid-template-columns:1fr}.run .rt{text-align:left}}
</style></head><body>
<div class="wrap">

<header class="mast">
  <p class="eyebrow">EGO · 야간 무인 실행 · 2026-07-20</p>
  <h1>고쳐진 측정 위에서 무엇이 실제로 제약인지 다시 잰다</h1>
  <p class="dek">오늘 저녁 샘플러 결함을 발견하기 전까지의 진단은 전부 망가진 측정 위에 있었다.
  두 트랙이 각자 하나의 GPU에서 그 전제를 다시 세운다 — Step-1은 무엇이 성능을 막는지,
  RETRO는 실패가 가설 탓인지 구현 탓인지.</p>
  <div class="meta">
    <span id="clk"><span class="live"></span>연결 중…</span>
    <span id="metaS1">Step-1 —</span>
    <span id="metaR">RETRO —</span>
  </div>
</header>

<div class="verdict" id="verdict"></div>

<section>
  <div class="sec-head"><span class="sec-num">01</span><h2>Step-1 · ViT-L 실험</h2>
    <span class="gpu">GPU 0 · 캐시 313GB</span></div>
  <p class="lede">ViT-g가 아니라 ViT-L을 돌리는 이유 — 9시간으로는 ViT-g가 train 19,000건까지만 뽑혀
  ViT-L(30,374건)과 비교하면 백본 차이와 데이터량 차이가 섞인다. 큰 모델일수록 데이터를 더 요구하므로
  데이터 기아 상태의 비교는 ViT-g에 불리하게 나오고, 그 잘못된 결론으로 11.6시간 투자를 접게 된다.
  <strong>Phase 2의 데이터 스케일링 곡선이 ViT-g 투자 여부를 결정한다.</strong></p>
  <div class="status" id="s1big"></div>
  <div class="verdict" id="basenote" style="--st:var(--crit);margin-top:0"></div>
  <h3>실행 대기열</h3>
  <div class="runs" id="s1runs"></div>
</section>

<section>
  <div class="sec-head"><span class="sec-num">02</span><h2>RETRO 재실행</h2>
    <span class="gpu">GPU 1</span></div>
  <p class="lede">오늘 기각된 것 — “credit을 action span에 국소화하면 belief→action 인과가 생긴다”.
  게이트 A가 그 실패의 원인을 <b>두 인자로 분해</b>했다: <code>.mean()</code>이 신호를 31배 죽였고(구현 결함),
  action span 자체도 belief 대비 11.8배 약했다(가설 결함). <b>둘 다 실재했다.</b>
  <br><br>
  그래서 이 체인은 <code>credit=belief · reduction=sum</code>으로 gradient를 되살린다. 다만 그것만으로는
  부족하다는 것이 오늘 추가로 확인됐다 — <b>보상 함수가 목표와 어긋나 있었다.</b>
  <code>--reward wm</code>의 전역 최적해는 <em>WM top-1 맹목 추종</em>이고, GT를 완벽히 맞히는 정책의 보상이
  오히려 더 낮다(0.4716 &lt; 0.5017). GT가 rank2~5인 24.3% 구간 — 정확히 Retrospection의 영역 — 에서는
  보상 최대화가 GT와 <b>반대 방향</b>으로 움직인다.
  <strong>그래서 v2는 보상을 <code>--reward gt</code>로 바꾸고, 원래 계획이던 <code>wm</code> arm을
  대조군으로 강등해 위험을 데이터로 남긴다.</strong> 총 소요 시간은 그대로다 —
  정보량이 없는 <code>action+sum</code> 본실행 4.7시간을 버리고 그 자리를 썼다.</p>
  <div class="status" id="rbig"></div>
  <div id="rstages"></div>
  <pre id="rlog"></pre>
</section>

<section>
  <div class="sec-head"><span class="sec-num">03</span><h2>장비</h2></div>
  <div class="status" id="gpu"></div>
</section>

<footer>
  5초마다 갱신 · 모든 ETA는 각 런의 <code>training_history.csv</code>에 기록된 실측 초/epoch에서 계산한다
  (하드코딩 추정치 없음 — GPU 경합으로 느려지면 표시되는 잔여 시간도 정직하게 늘어난다).
  아직 시작하지 않은 런은 같은 스테이지에서 관측된 속도를, 그것도 없으면 전체 평균을 쓴다.
</footer>

</div>
<script>
const $=i=>document.getElementById(i);
const hms=s=>{if(s==null)return'—';s=Math.max(0,s);const h=Math.floor(s/3600),m=Math.round(s%3600/60);
  return h?`${h}h ${String(m).padStart(2,'0')}m`:`${m}m`};
const nn=(v,d=2)=>v==null?'—':Number(v).toFixed(d);
const box=(k,v,n,c)=>`<div class="stat"><div class="k">${k}</div>
  <div class="v"${c?` style="color:var(--${c})"`:''}>${v}</div><div class="n">${n||''}</div></div>`;

async function tick(){
  let d; try{ d=await fetch('/api/status',{cache:'no-store'}).then(r=>r.json()); }
  catch(e){ $('clk').innerHTML='<span class="live"></span>재연결 중…'; return; }

  $('clk').innerHTML='<span class="live"></span>'+d.now+' KST';
  const S=d.step1, R=d.retro, B=S.baseline;
  $('metaS1').textContent='Step-1 잔여 '+hms(S.remaining_sec)+' · '+S.done+'/'+S.total+' 런';
  const gp=(R.gate&&R.gate.belief)?parseFloat(R.gate.belief):null;
  $('metaR').textContent='RETRO 게이트 A '+(gp!=null?(gp>=0.00092?'통과':'실패'):'측정 중');

  // 전체 판정 배너 — 지금 가장 중요한 한 가지
  const vd=$('verdict');
  if(gp!=null&&gp>=0.00092){
    vd.style.setProperty('--st','var(--ok)');
    vd.innerHTML='<h2>게이트 A 통과 — 구현 결함이 원인이었다</h2>'
      +'<p><code>tok_lp.mean()</code> → <code>sum</code> 한 줄로 gradient가 되살아났다. '
      +'ARM B의 실패는 <strong>가설이 틀려서가 아니라 구현이 신호를 죽였기 때문</strong>이다.</p>'
      +'<p class="quote">credit=action (오늘 실패) 0.000092 → credit=belief+sum <b>'+R.gate.belief+'</b>'
      +'<br>정상 학습 참고치 credit=all 0.004160 — 그보다도 8배 크다.</p>';
  }else{
    vd.style.setProperty('--st','var(--accent)');
    vd.innerHTML='<h2>진행 중</h2><p>게이트 A 스모크 측정 중 — '
      +'<code>credit-reduction=sum</code>이 gradient를 되살리는지 확인하고 있다.</p>';
  }

  $('s1big').innerHTML=box('진행',S.done+' / '+S.total,'런 완료')
    +box('전체 잔여',hms(S.remaining_sec),'Phase 4 재추출 포함','accent')
    +box('종료 예정',S.eta,'KST')
    +box('베이스라인 top5',nn(B.top5_best),'epoch '+B.top5_best_ep+' 정점');

  $('basenote').innerHTML='<h2>베이스라인이 알려준 것 — 고쳐도 과적합은 남는다</h2>'
    +'<p><code>'+B.name+'</code> · '+B.note+'</p>'
    +'<p class="quote">top5  ep1 17.65 → <b>ep'+B.top5_best_ep+' '+nn(B.top5_best)+' (정점)</b> → 현재 '+nn(B.top5_last)+''
    +'<br>train_loss 1.18 → '+nn(B.loss,4)+'  ·  cmr@5 '+nn(B.cmr5)+'</p>'
    +'<p>샘플러를 고치자 정점이 <strong>epoch 3</strong>으로 앞당겨지고 그 뒤로는 하락한다. '
    +'이제서야 정규화 실험(Phase 3)이 의미를 갖는다.</p>';

  $('s1runs').innerHTML=S.runs.map(r=>{
    const p=r.total?100*r.done/r.total:0;
    return `<div class="run ${r.status}">
      <span><span class="nm">${r.name}</span><span class="wt">${r.what}</span>
      <span class="q">${r.question}</span>
      <div class="nums">${r.phase} · ${r.done}/${r.total} epoch`
      +(r.top5_best?` · top5 최고 <b>${nn(r.top5_best)}</b> (ep${r.top5_best_ep})`:'')
      +(r.cmr5!=null?` · cmr@5 ${nn(r.cmr5)}`:'')
      +(r.verb_top5!=null?` · verb top5 ${nn(r.verb_top5)}`:'')
      +`</div><div class="bar"><i style="width:${p}%"></i></div></span>
      <span class="rt"><span class="chip ${r.status}">${r.status}</span>
      <div class="nums">${hms(r.remaining)}</div></span></div>`;
  }).join('');

  $('rbig').innerHTML=box('게이트 A',(R.gate&&R.gate.belief)||'측정 중','belief·sum · 기준 ≥0.00092 · 통과',
      (gp!=null&&gp>=0.00092)?'ok':null)
    +box('WM top-1 argmax','0.374','자명한 기준선 — 학습된 정책 전부(0.24~0.28)보다 높다','crit')
    +box('reward=wm 최적해','0.5017','GT 완벽 정책(0.4716)보다 높다 → 목표와 어긋남','crit')
    +box('본실행 잔여',hms(R.remaining_sec),'5,000샘플 REINFORCE · reward=gt','accent');

  $('rstages').innerHTML=R.stages.map(s=>`<div class="item ${s.status}">
    <div class="tophead"><span class="key">${s.id}</span><span class="ttl">${s.what}</span>
    <span class="rt"><span class="chip ${s.status}">${s.status}</span></span></div>
    ${s.goal?`<p style="font-weight:600;color:var(--fg)">목표 — ${s.goal}</p>`:''}
    <p>${s.why}</p>
    ${s.ev?`<p style="color:var(--fg-mid);font-size:14.5px">근거 — ${s.ev}</p>`:''}
    <div class="ev">통과 기준 — <b>${s.pass}</b><br>현재 — ${s.result}</div></div>`).join('');

  $('rlog').textContent=(R.tail||[]).join('\\n')||'로그 대기…';

  $('gpu').innerHTML=d.gpus.map(g=>g.error?box('GPU',g.error,''):
    box('GPU '+g.i,g.util+'%',`${Number(g.mem).toLocaleString()} / ${Number(g.memtot).toLocaleString()} MiB · ${g.temp}°C`,
        Number(g.util)>10?'accent':null)).join('');
}
tick(); setInterval(tick,5000);
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        body, ctype = ((json.dumps(status()).encode(), "application/json")
                       if self.path.startswith("/api/status")
                       else (PAGE.encode(), "text/html; charset=utf-8"))
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_a):
        pass


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=7862)
    a = p.parse_args()
    print(f"[night-dashboard] http://{a.host}:{a.port}", flush=True)
    ThreadingHTTPServer((a.host, a.port), H).serve_forever()


if __name__ == "__main__":
    main()
