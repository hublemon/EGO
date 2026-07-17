# NOTE: F0 (WM-only GRPO) 트랙의 **실제로 검증된 코드**를 그대로 옮긴 것이다.
# 2026-07-17 F0 final 결과(docs/experiments/2026-07-17_f0_final.md)를 만든 코드가 이 파일이다.
# 패키지 구조에 맞춘 추측성 리팩터는 하지 않았다 — 재검증 없이 쪼개면 결과 재현이 깨진다.
# 실행은 configs/step2/f0_final_wm_only.yaml 과 scripts/step2/train_f0_final.sh 참조.
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""judge_reasoning_curve.py — 외부 judge 로 리즈닝 품질의 **학습 step 곡선**을 만든다.

왜 필요한가
-----------
F0 내부 지표는 리즈닝 '품질'을 못 잰다 (docs/F0_FINAL_HANDOFF.md §3.6):
  - lift  : 전 run 24 근처 포화 → 판별 불가
  - mask  : lift 와 사실상 동일 (24.52 vs 24.52)
  - shuffle: 샘플 결합도. "의미 있는 추론"과 "도구적 인코딩"을 구분 못 함
  - 그리고 belief 는 어떤 리워드도 채점하지 않으므로(계약) 내부 지표가 아예 없다
→ 외부 모델(Gemini/Claude/GPT)에게 **history + WM 후보 → 행동 선택의 일관성**을 수치로 받는다.

설계 원칙
---------
1. **학습에 절대 미사용.** 순수 관측. 리워드는 WM likelihood 단독 유지 (§4.3).
2. **judge 에게 GT 를 주지 않는다.** 주면 '품질'이 아니라 '정답률'을 재게 되어 acc 와 중복된다.
   우리가 알고 싶은 건 "맞았나"가 아니라 "history·후보로부터 일관되게 골랐나"다.
3. **학습 루프 밖에서 실행.** train 은 completion_samples.jsonl 만 쓰고(롤아웃 재사용, GPU 비용 0),
   이 스크립트가 그 파일을 읽는다. 네트워크 지연이 학습을 막지 않는다.
   --follow 로 학습 중 tailing 도 가능 → 학습 끝나면 곡선도 끝나 있다.

사용
----
  export LETSUR_API_KEY="..."          # 절대 하드코딩 금지 (CLAUDE.md)
  python judge_reasoning_curve.py --run_dir runs/f0_final                 # 1회
  python judge_reasoning_curve.py --run_dir runs/f0_final --follow        # 학습과 동시에 tailing
출력: <run_dir>/judge_curve.jsonl (step 별 원점수) + judge_curve_summary.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

EGO_ROOT = Path(os.path.expanduser("~/work/jihun/EGO"))
sys.path.insert(0, str(EGO_ROOT))

# ── 루브릭 ─────────────────────────────────────────────────────────────────
# 구 루브릭(eval_reasoning_trace.py) 대비 변경:
#   - verb/noun 분리 리스트 → **joint action 후보 5개** (F0 v2 포맷)
#   - goal_awareness 제거   → task_goal 은 가짜(=video ID, GT 예측력 1.1%)라 프롬프트에서 뺐다
#   - **history_grounding 신설** → 이 측정의 핵심. WM 은 history 를 못 보므로,
#     history 사용 여부가 곧 "VLM 만의 정보 우위를 쓰고 있는가"다.
JUDGE_RUBRIC = """You are evaluating the reasoning trace of an embodied action-selection model.

The model saw one egocentric kitchen frame, the actor's recent action history, and FIVE
candidate next-actions proposed by a world model (unordered, scores hidden). It had to pick
exactly one candidate.

Action history given to the model:
---
{history}
---

The five candidates it could choose from:
{candidates}

Its reasoning:
---
{think}
---

The overall goal it stated: {belief}

Its final choice: {action}

Score each criterion 0 (absent), 1 (partial), 2 (clear). Judge the REASONING, not whether the
choice is objectively correct — you are not told the right answer and should not guess it.

1. history_grounding  — the reasoning actually uses the action history (what was already done,
                        what that implies about what comes next). 0 = ignores it entirely.
2. candidate_review   — explicitly weighs named candidates against each other, not generic talk.
3. visual_grounding   — cites observable evidence from the frame for its judgments.
4. conclusion_follows — the final choice follows from the reasoning it just gave, rather than
                        appearing out of nowhere or contradicting it.
5. no_confabulation   — invents no actions/objects outside the five candidates (2 = none).
6. belief_globality   — the stated goal is a GLOBAL objective spanning several actions
                        (2 = e.g. "wash up after cooking"), not a restatement of the single
                        chosen action (0 = e.g. goal "open the cupboard" while choosing
                        "open cupboard"), and not vacuous (0 = "do a kitchen task").

Output JSON only:
{{"history_grounding": N, "candidate_review": N, "visual_grounding": N,
"conclusion_follows": N, "no_confabulation": N, "belief_globality": N,
"note": "<one short sentence>"}}"""

KEYS = ["history_grounding", "candidate_review", "visual_grounding",
        "conclusion_follows", "no_confabulation", "belief_globality"]


def extract_think(text: str) -> str:
    """<reasoning> 또는 <think> 블록. F0 v2 는 <reasoning> 을 쓴다 —
    Qwen3-VL 토크나이저에서 <think>/</think> 는 예약 단일토큰(151667/151668)이라
    Instruct 변형이 절대 생성하지 않기 때문. <think> 은 Qwen2.5 기반 구 run 하위호환용."""
    m = re.search(r"<(?:reasoning|think)>(.*?)</(?:reasoning|think)>", text or "", re.DOTALL)
    return m.group(1).strip() if m else ""


def extract_belief(text: str) -> str:
    m = re.search(r"<task_belief>(.*?)</task_belief>", text or "", re.DOTALL)
    return m.group(1).strip() if m else "(none stated)"


def extract_action(text: str) -> str:
    m = re.search(r"<action>(.*?)</action>", text or "", re.DOTALL)
    if not m:
        return "(parse fail)"
    jm = re.search(r"\{.*\}", m.group(1), re.DOTALL)
    if not jm:
        return "(parse fail)"
    try:
        o = json.loads(jm.group(0))
        return f'{o.get("verb")} {o.get("noun")}'
    except Exception:
        return "(parse fail)"


def judge_one(client, model, hist, cands, think, belief, action):
    prompt = JUDGE_RUBRIC.format(history=hist or "(none)", candidates=cands,
                                 think=think[:2000], belief=belief[:300], action=action)
    r = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}],
        # Gemini 계열은 응답 전 reasoning 토큰을 소모(실측: 한 단어 응답에 133토큰).
        # 예산이 작으면 content 가 None 으로 온다 → 루브릭 JSON 여유분 포함 1500.
        max_completion_tokens=1500, temperature=0)
    txt = r.choices[0].message.content or ""
    # gw.letsur.ai 는 estimated_cost 를 dict 로 준다: {'amount': '0.0024', 'currency': 'unit', ...}
    cost = None
    try:
        raw = getattr(r, "estimated_cost", None) or (getattr(r, "model_extra", None) or {}).get("estimated_cost")
        if isinstance(raw, dict):
            raw = raw.get("amount")
        cost = float(raw) if raw is not None else None
    except Exception:
        pass
    m = re.search(r"\{.*\}", txt, re.DOTALL)
    if not m:
        return None, cost
    try:
        return json.loads(m.group(0)), cost
    except Exception:
        return None, cost


def load_samples(path: Path, done_steps: set):
    """completion_samples.jsonl → step 별 레코드. 이미 처리한 step 은 건너뛴다."""
    out = []
    if not path.exists():
        return out
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        if d.get("step") in done_steps:
            continue
        out.append(d)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--judge_model", default="gemini-2.5-pro",
                    help="gw.letsur.ai 카탈로그. 실측 호출당 비용: gemini-2.5-pro $0.0024 "
                         "(reasoning 토큰 소모) vs claude-sonnet-4-6 $0.0001 (25배 저렴). "
                         "교차검증 시 두 모델로 각각 돌려 일치 여부를 볼 것.")
    ap.add_argument("--base_url", default="https://gw.letsur.ai/v1")
    ap.add_argument("--per_step", type=int, default=4, help="step 당 채점할 생성 개수")
    ap.add_argument("--follow", action="store_true", help="학습 중 tailing (30초 주기)")
    ap.add_argument("--poll_sec", type=int, default=30)
    args = ap.parse_args()

    if "LETSUR_API_KEY" not in os.environ:
        sys.exit("LETSUR_API_KEY 환경변수가 없습니다. `export LETSUR_API_KEY=\"...\"` 후 재실행.\n"
                 "(CLAUDE.md 규칙: 키는 절대 파일에 하드코딩하지 않는다)")
    import openai
    client = openai.OpenAI(base_url=args.base_url, api_key=os.environ["LETSUR_API_KEY"])

    run = Path(args.run_dir)
    src = run / "completion_samples.jsonl"
    out = run / "judge_curve.jsonl"
    done = set()
    if out.exists():
        for l in out.read_text().splitlines():
            try:
                done.add(json.loads(l)["step"])
            except Exception:
                pass
        print(f"[resume] 이미 채점된 step {len(done)}개 건너뜀")

    total_cost = 0.0
    while True:
        rows = load_samples(src, done)
        for d in sorted(rows, key=lambda x: x.get("step", 0)):
            step = d.get("step")
            comps = d.get("completions") or []
            hist = d.get("memory_context") or d.get("prompt_tail") or ""
            cands = d.get("topk_actions_display") or "(not logged)"
            scored = []
            for c in comps[:args.per_step]:
                txt = c if isinstance(c, str) else (c.get("text") or c.get("content") or "")
                think = extract_think(txt)
                if not think:
                    continue
                s, cost = judge_one(client, args.judge_model, hist, cands, think,
                                    extract_belief(txt), extract_action(txt))
                if cost:
                    total_cost += cost
                if s and all(k in s for k in KEYS):
                    scored.append(s)
            if not scored:
                continue
            rec = {"step": step, "n": len(scored),
                   **{k: round(sum(x[k] for x in scored) / len(scored), 3) for k in KEYS}}
            rec["total"] = round(sum(rec[k] for k in KEYS), 3)   # 0~12 (6항목 × 2점)
            rec["notes"] = [x.get("note", "")[:80] for x in scored[:2]]
            with out.open("a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            done.add(step)
            print(f"  step {step:>4}  총점 {rec['total']:>5.2f}/12  " +
                  "  ".join(f"{k[:4]}={rec[k]:.2f}" for k in KEYS))

        if not args.follow:
            break
        time.sleep(args.poll_sec)

    # ── 요약: 학습 step 에 따라 실제로 개선되는가 ──
    rows = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    if len(rows) >= 4:
        rows.sort(key=lambda r: r["step"])
        h = len(rows) // 2
        first, last = rows[:h], rows[h:]
        def mean(rs, k): return sum(r[k] for r in rs) / len(rs)
        summary = {"n_steps": len(rows), "judge_model": args.judge_model,
                   "step_range": [rows[0]["step"], rows[-1]["step"]],
                   "estimated_cost_usd": round(total_cost, 4)}
        print("\n" + "=" * 70)
        print(f"리즈닝 품질 곡선 — 전반부 vs 후반부 (judge={args.judge_model})")
        print("=" * 70)
        print(f"{'항목':<22}{'전반부':>9}{'후반부':>9}{'변화':>9}")
        for k in KEYS + ["total"]:
            a, b = mean(first, k), mean(last, k)
            summary[k] = {"first_half": round(a, 3), "second_half": round(b, 3),
                          "delta": round(b - a, 3)}
            print(f"{k:<22}{a:>9.2f}{b:>9.2f}{b-a:>+9.2f}")
        (run / "judge_curve_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False))
        print(f"\n누적 비용 ≈ ${total_cost:.4f}")
        print(f"저장 → {out}  ·  {run/'judge_curve_summary.json'}")
        print("\n주의: judge 에게 GT 를 주지 않았다 → 이 점수는 '정답률'이 아니라 '추론 일관성'이다.")
        print("      학습에는 일절 미사용 (리워드는 WM likelihood 단독).")


if __name__ == "__main__":
    main()
