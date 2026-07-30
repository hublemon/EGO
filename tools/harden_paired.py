#!/usr/bin/env python3
"""harden_paired.py — belief 개입 지표의 **arm-paired** 재측정 (Base·θ_CE 열 확보 + G-CC2 정식 판정).

기존 `eval/harden_s3.py` 는 arm 마다 독립 실행이라 arm 간 비교에 교란이 3개 있다:

  (1) 모집단 불일치 : arm 별 seed 42 shuffle 후 n 개 → 서로 다른 샘플 집합 (paired 아님)
  (2) swap partner  : `recs[(i+7)%n]` 가 **arm 내부**에서 나옴. belief 문체가 arm 마다 달라
                      (base 8.3단어 명령형 / θ_CE 5.8단어 / r15 15.2단어 100% 서술형)
                      주입되는 prefix 정보량 자체가 다르다 → flip 율이 감수성 아닌 문체를 잼
  (3) paraphrase    : 대조군 의역을 **평가 대상 모델 자신이** 생성 → arm 마다 다른 잣대.
                      sensitivity = flip(swap_b) − flip(para) 라 기준선 차이가 그대로 실림

셋을 각각 고정한다:

  (1) 공통 sample_id 셋 1개를 전 arm 이 **같은 순서로** 사용 → arm 간 paired CI 가능
  (2) swap 을 2종으로 분리
        swap_b        : arm 내부 partner belief   (기존 정의 — 기존 수치와의 연속성 유지)
        swap_b_shared : **전 arm 동일 문자열**(donor arm 의 partner belief) 주입
                        → flip 차이가 텍스트가 아니라 모델에서 왔음이 보장된다
      두 값의 차이가 곧 "문체 교란의 크기"이며 그 자체로 보고 가치가 있다.
  (3) paraphrase 를 **base 모델 1개로만** 생성해 전 arm 이 공유 (plan 단계 1회)

3단계 — arm 마다 프로세스를 분리해 GPU 메모리를 반납한다:

  plan : 공통 셋·partner 매핑·shared paraphrase 생성  → eval/harden_paired_plan.json
  run  : arm 1개 강제 추론 (7 variant)                → eval/{arm}.harden_paired.{json,records.json,arrays.json}
  agg  : arm 간 paired 차이 + G-CC2 판정              → eval/harden_paired_summary.json

사용:
  export RETRO3_RUNS=runs/cesft_v2 RETRO_NEXT_GAP_TEXT="after the current action ends" PYTHONPATH=src
  PY tools/harden_paired.py --stage plan --arms base,theta_ce,sft_r15 --n 800
  PY tools/harden_paired.py --stage run  --arm base
  PY tools/harden_paired.py --stage run  --arm theta_ce --adapter outputs/.../theta_ce/adapter
  PY tools/harden_paired.py --stage run  --arm sft_r15  --adapter outputs/.../sft_r15/adapter
  PY tools/harden_paired.py --stage agg  --arms base,theta_ce,sft_r15 --ref sft_r15
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import yaml

from ego.step2_retrospection import vlm
from ego.step2_retrospection.eval.battery import load_arm
from ego.step2_retrospection.runtime import StatusWriter, read_jsonl, runs_root, write_marker

PLAN_NAME = "harden_paired_plan.json"
VARIANTS = ("own", "empty", "swap_b", "swap_b_shared", "swap_r", "swap_both", "para")
FLIPPABLE = ("swap_b", "swap_b_shared", "swap_r", "swap_both", "para")

# --curve: 스텝별 belief 인과 곡선용 축소 집합. 곡선이 그리는 값은 swap_b_shared 민감도
# (= flip(swap_b_shared) − flip(para)) 하나뿐이라 own/swap_b_shared/para 3개면 충분하다.
# 7 → 3 변형이므로 채점 비용이 arm 당 ~54분에서 ~23분으로 준다. 전체 요약 대신 축소 요약을 낸다.
CURVE_VARIANTS = ("own", "swap_b_shared", "para")


# ── 후보 채점 — vision-grounded (프레임 포함) ─────────────────────────────────
# 기존 eval/intervention.py:candidate_probs 와 두 군데가 다르다.
#   ① 프레임: build_messages(rec, []) → **실제 8프레임**. 학습(select_ce)과 같은 조건.
#   ② 프롬프트: 기존 구현은 apply_chat_template(add_generation_prompt=True) 가 이미 붙인
#      "<|im_start|>assistant\n" 뒤에 prefix 로 **같은 헤더를 한 번 더** 붙여
#      assistant 헤더가 2개인 off-distribution 프롬프트가 됐다(검증: 템플릿 끝이
#      '<|im_start|>assistant\n' 로 끝남 → intervention.py:36 이 중복 추가).
#      여기서는 select_ce._candidate_logps 규약대로 헤더를 1개만 둔다.
# 이 두 변경 때문에 산출값은 기존 harden_s3 수치와 **절대값 비교 불가**다(내부 비교는 유효).
def _assistant_body(reasoning: str, belief: str | None) -> str:
    s = "<reasoning>\n" + reasoning.strip() + "\n</reasoning>\n\n"
    if belief is not None:
        s += "<task_belief>\n" + belief.strip() + "\n</task_belief>\n\n"
    return s + "<action>\n"


@torch.no_grad()
def candidate_probs_v(model, processor, rec, reasoning, belief, imgs, device="cuda", chunk=10):
    """후보별 length-norm logp → softmax. imgs 가 비면 텍스트 전용(기존 조건)."""
    tok = processor.tokenizer
    text = processor.apply_chat_template(vlm.build_messages(rec, imgs), tokenize=False,
                                         add_generation_prompt=True)
    pre_ids = tok(_assistant_body(reasoning, belief), add_special_tokens=False)["input_ids"]
    cand_tok = [tok(c, add_special_tokens=False)["input_ids"] for c in rec["candidates"]]
    scores = []
    for i0 in range(0, len(cand_tok), chunk):
        sub = cand_tok[i0:i0 + chunk]
        K = len(sub)
        proc = processor(text=[text] * K, images=[imgs] * K if imgs else None,
                         return_tensors="pt").to(device)
        P = proc["input_ids"].shape[1]
        comp = [pre_ids + ct for ct in sub]
        maxc = max(len(x) for x in comp)
        cid = torch.full((K, maxc), tok.eos_token_id, dtype=torch.long, device=device)
        cmk = torch.zeros((K, maxc), dtype=torch.long, device=device)
        for j, x in enumerate(comp):
            cid[j, :len(x)] = torch.tensor(x, device=device)
            cmk[j, :len(x)] = 1
        new = {}
        for k, v in proc.items():
            if k == "input_ids":
                new[k] = torch.cat([v, cid], dim=1)
            elif k == "attention_mask":
                new[k] = torch.cat([v, cmk], dim=1)
            elif isinstance(v, torch.Tensor) and v.dim() >= 2 and v.shape[1] == P:
                pad = torch.zeros((v.shape[0], maxc, *v.shape[2:]), dtype=v.dtype, device=device)
                new[k] = torch.cat([v, pad], dim=1)
            else:
                new[k] = v
        logits = model(**new).logits
        lp = torch.log_softmax(logits[:, :-1].float(), dim=-1)
        tok_lp = lp.gather(-1, new["input_ids"][:, 1:].unsqueeze(-1)).squeeze(-1)
        s0 = P + len(pre_ids) - 1
        scores.extend(tok_lp[j, s0:s0 + len(ct)].mean().item() for j, ct in enumerate(sub))
        del proc, new, logits, lp, tok_lp, cid, cmk
    return torch.softmax(torch.tensor(scores), dim=0)


def score_all_variants(model, processor, rec, variants, imgs, chunk):
    """한 샘플의 7 variant 를 **같은 조건**(같은 imgs·chunk)에서 채점. OOM 은 호출부가 처리."""
    return {name: candidate_probs_v(model, processor, rec, rr, bb, imgs, "cuda", chunk)
            for name, (rr, bb) in variants.items()}


def free_gb() -> float:
    try:
        return torch.cuda.mem_get_info()[0] / 1024 ** 3
    except Exception:                                                # noqa: BLE001
        return 999.0


def sample_probs(model, processor, rec, variants, imgs0, chunk0, log, min_free_gb=20.0):
    """OOM 안전 실행.
    정책: 메모리가 빠듯하면 **먼저 그 샘플의 프레임을 버린다**(사전 가드 + OOM 즉시 강등).
    variant 마다 조건이 달라지면 개입 비교가 깨지므로 **샘플 전체를 같은 조건으로** 재시도한다.
    반환: (probs, used_frames, used_chunk) — 완전 실패 시 (None, False, chunk)."""
    imgs, chunk = imgs0, chunk0
    if imgs and free_gb() < min_free_gb:      # 사전 가드 — 터지기 전에 피한다
        log(f"free {free_gb():.0f}GB < {min_free_gb:.0f}GB → 프레임 skip ({rec['sample_id']})")
        imgs = []
    while True:
        try:
            return score_all_variants(model, processor, rec, variants, imgs, chunk), bool(imgs), chunk
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if imgs:                          # 1순위: 프레임 즉시 포기
                imgs = []
                log(f"OOM → 프레임 즉시 skip, 텍스트 전용 ({rec['sample_id']})")
            elif chunk > 1:                   # 텍스트 전용인데도 터지면 그때 chunk 축소
                chunk = max(1, chunk // 2)
                log(f"OOM(텍스트) → chunk={chunk} ({rec['sample_id']})")
            else:
                log(f"OOM → 샘플 포기 ({rec['sample_id']})")
                return None, False, chunk


# ── 통계 (harden_s3.py 와 동일 규약: sample bootstrap 2,000회) ────────────────
def bootstrap_ci(xs, n_boot=2000, seed=0):
    rng = random.Random(seed)
    n = len(xs)
    if n == 0:
        return {"point": 0.0, "lo": 0.0, "hi": 0.0}
    point = sum(xs) / n
    boots = sorted(sum(xs[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return {"point": round(point, 4), "lo": round(boots[int(.025 * n_boot)], 4),
            "hi": round(boots[int(.975 * n_boot)], 4)}


def diff_ci(a, b, n_boot=2000, seed=0):
    """paired 차이 mean(a)−mean(b) — 같은 인덱스 resample."""
    rng = random.Random(seed)
    n = len(a)
    if n == 0:
        return {"point": 0.0, "lo": 0.0, "hi": 0.0}
    point = (sum(a) - sum(b)) / n
    boots = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        boots.append((sum(a[i] for i in idx) - sum(b[i] for i in idx)) / n)
    boots.sort()
    return {"point": round(point, 4), "lo": round(boots[int(.025 * n_boot)], 4),
            "hi": round(boots[int(.975 * n_boot)], 4)}


def eligible(arm: str) -> dict:
    """harden 자격 records — harden_s3.py 와 동일 필터."""
    return {r["sample_id"]: r for r in read_jsonl(runs_root() / "eval" / f"{arm}.records.jsonl")
            if not r.get("malformed") and r.get("task_belief") and r.get("reasoning")
            and r.get("gt_in_support")}


# ── stage: plan ──────────────────────────────────────────────────────────────
def stage_plan(args):
    arms = [a.strip() for a in args.arms.split(",")]
    recs = {a: eligible(a) for a in arms}
    for a in arms:
        print(f"[plan] {a:10s} eligible={len(recs[a])}")
    common = set.intersection(*[set(recs[a]) for a in arms])
    ids = sorted(common)                       # 결정적 정렬 후 고정 seed shuffle
    random.Random(args.seed).shuffle(ids)
    ids = ids[: args.n]
    n = len(ids)
    print(f"[plan] common={len(common)} → n={n}")
    if n < 100:
        raise SystemExit(f"[plan] 공통 셋이 너무 작다 (n={n}) — arms 를 줄이거나 battery 를 먼저 확장할 것")

    # partner: harden_s3 와 같은 (i+7)%n 규약이되 **인덱스 매핑을 전 arm 이 공유**
    partner = [ids[(i + 7) % n] for i in range(n)]
    donor = args.donor if args.donor in arms else arms[0]

    print(f"[plan] paraphraser = base 모델(어댑터 없음) — 전 arm 공유 · donor={donor}")
    model, processor = load_arm(None)
    paras = {}
    for a in arms:
        msgs = [[{"role": "user", "content": [{"type": "text", "text":
                 f'Paraphrase this sentence, keeping the exact same meaning: '
                 f'"{recs[a][s]["task_belief"]}"\nReply with the paraphrase only.'}]}]
                for s in ids]
        out = []
        for i0 in range(0, len(msgs), 16):
            out.extend(vlm.generate_batch(model, processor, msgs[i0:i0 + 16], max_new_tokens=60))
            print(f"\r[plan] para {a}: {min(i0 + 16, len(msgs))}/{len(msgs)}", end="", flush=True)
        print()
        paras[a] = [o.strip().strip('"') for o in out]
    vlm.close_readers()

    plan = {
        "seed": args.seed, "n": n, "arms": arms, "donor_arm": donor,
        "ids": ids, "partner": partner,
        # 전 arm 공통 donor belief — 문자열이 arm 간 완전히 동일하다
        "shared_swap_belief": [recs[donor][p]["task_belief"] for p in partner],
        "para": paras,
        "note": "paraphrase 는 base 모델 1회 생성분을 전 arm 공유. partner 는 인덱스 (i+7)%n 고정.",
    }
    dst = runs_root() / "eval" / PLAN_NAME
    dst.write_text(json.dumps(plan, ensure_ascii=False))
    write_marker("S3HP_PLAN_DONE", {"n": n, "arms": arms, "donor": donor})
    print(f"[plan] -> {dst}  (n={n}, arms={arms}, donor={donor})")


# ── stage: run ───────────────────────────────────────────────────────────────
def stage_run(args):
    plan = json.loads((runs_root() / "eval" / PLAN_NAME).read_text())
    arm = args.arm
    if arm not in plan["arms"]:
        raise SystemExit(f"[run] {arm} 이 plan 에 없다: {plan['arms']}")
    ctx = {r["sample_id"]: r for r in read_jsonl(runs_root() / "data" / "context_val.jsonl")}
    recs = eligible(arm)
    ids, partner, n = plan["ids"], plan["partner"], plan["n"]
    missing = [s for s in ids if s not in recs]
    if missing:
        raise SystemExit(f"[run] {arm} records 결손 {len(missing)}건 — plan 을 다시 만들 것")

    video_root = Path(yaml.safe_load(open(args.config, encoding="utf-8"))
                      ["shared_assets"]["video_root"])
    if args.bench:
        ids, partner, n = ids[:args.bench], partner[:args.bench], args.bench
        print(f"[run] BENCH 모드 — {n} 샘플만")

    active = CURVE_VARIANTS if args.curve else VARIANTS
    flippable = tuple(k for k in FLIPPABLE if k in active)
    model, processor = load_arm(args.adapter)
    sw = StatusWriter(f"S3HP_{arm}", total=n)
    P = {k: [] for k in active}
    F = {k: [] for k in flippable}
    ACC = {k: [] for k in active}        # variant 별 top1==gt (추가 연산 0 — top1 재사용)
    correct, out_rows, kept_ids = [], [], []
    n_noframe = n_skip = 0
    log = lambda m: print(f"[run:{arm}] {m}", flush=True)  # noqa: E731

    for i, sid in enumerate(ids):
        r, rec, pr = recs[sid], ctx[sid], recs[partner[i]]
        gt_idx = rec["candidates"].index(r["gt"])
        variants = {
            "own":           (r["reasoning"], r["task_belief"]),
            "empty":         (r["reasoning"], None),
            "swap_b":        (r["reasoning"], pr["task_belief"]),
            "swap_b_shared": (r["reasoning"], plan["shared_swap_belief"][i]),
            "swap_r":        (pr["reasoning"], r["task_belief"]),
            "swap_both":     (pr["reasoning"], pr["task_belief"]),
            "para":          (r["reasoning"], plan["para"][arm][i]),
        }
        variants = {k: v for k, v in variants.items() if k in active}
        imgs = []
        if not args.no_frames:
            try:   # 디코딩 실패·캐시 결손은 그 샘플만 프레임 없이 진행 (전체 중단 금지)
                imgs = vlm.extract_frames(video_root, rec["video_uid"],
                                          rec["obs_start_sec"], rec["obs_end_sec"], rec=rec)
            except Exception as e:                                   # noqa: BLE001
                log(f"프레임 추출 실패 → skip ({sid}): {type(e).__name__}")
                imgs = []
        probs, used_frames, _ = sample_probs(model, processor, rec, variants,
                                             imgs, args.cand_chunk, log)
        if probs is None:
            n_skip += 1
            continue
        if not used_frames:
            n_noframe += 1
        top1 = {name: int(p.argmax()) for name, p in probs.items()}
        for name in active:
            P[name].append(float(probs[name][gt_idx]))
            ACC[name].append(int(top1[name] == gt_idx))
        own_t = top1["own"]
        correct.append(int(own_t == gt_idx))
        for k in flippable:
            F[k].append(int(top1[k] != own_t))
        kept_ids.append(sid)
        c = rec["candidates"]
        if args.curve:
            out_rows.append({
                "frames_used": used_frames, "sample_id": sid, "gt": r["gt"],
                "own_belief": r["task_belief"],
                "shared_swap_belief": plan["shared_swap_belief"][i],
                "own_action": c[own_t], "swap_b_shared_action": c[top1["swap_b_shared"]],
                "p_gt_own": round(P["own"][-1], 4),
                "p_gt_swap_b_shared": round(P["swap_b_shared"][-1], 4),
                "flip_swap_b_shared": F["swap_b_shared"][-1], "flip_para": F["para"][-1],
                "own_correct": correct[-1]})
            sw.update(done=i + 1, metrics={
                "flip_bs": round(sum(F["swap_b_shared"]) / max(1, len(kept_ids)), 3),
                "noframe": n_noframe, "skipped": n_skip, "free_gb": round(free_gb(), 1)})
            continue
        out_rows.append({
            "frames_used": used_frames,
            "sample_id": sid, "gt": r["gt"],
            "own_belief": r["task_belief"], "swap_belief": pr["task_belief"],
            "shared_swap_belief": plan["shared_swap_belief"][i], "para_belief": plan["para"][arm][i],
            "own_action": c[own_t], "swap_b_action": c[top1["swap_b"]],
            "swap_b_shared_action": c[top1["swap_b_shared"]], "para_action": c[top1["para"]],
            "p_gt_own": round(P["own"][-1], 4), "p_gt_swap_b": round(P["swap_b"][-1], 4),
            "p_gt_swap_b_shared": round(P["swap_b_shared"][-1], 4),
            "flip_swap_b": F["swap_b"][-1], "flip_swap_b_shared": F["swap_b_shared"][-1],
            "flip_para": F["para"][-1], "own_correct": correct[-1]})
        sw.update(done=i + 1, metrics={"flip_b": round(sum(F["swap_b"]) / max(1, len(kept_ids)), 3),
                                       "noframe": n_noframe, "skipped": n_skip,
                                       "free_gb": round(free_gb(), 1)})

    n = len(kept_ids)                      # 프레임/OOM 으로 빠진 샘플 반영
    if n == 0:
        raise SystemExit(f"[run] {arm}: 살아남은 샘플이 0")
    if args.curve:
        # 곡선 셀 — 축소 요약. 절대값은 같은 plan 안에서만 비교한다.
        sens_bs = diff_ci(F["swap_b_shared"], F["para"])
        summary = {
            "arm": arm, "n": n, "paired_plan": PLAN_NAME, "donor_arm": plan["donor_arm"],
            "curve": True, "variants": list(active),
            "scoring": {"frames": not args.no_frames, "cand_chunk": args.cand_chunk,
                        "n_frames_used": n - n_noframe, "n_frame_skipped": n_noframe,
                        "n_sample_skipped": n_skip},
            "flip": {k: bootstrap_ci(F[k]) for k in flippable},
            "p_gt": {k: bootstrap_ci(P[k]) for k in active},
            "acc_by_variant": {k: bootstrap_ci(ACC[k]) for k in active},
            "sensitivity": {"swap_b_shared": sens_bs},
            "utility": {"U_b_shared": diff_ci(P["own"], P["swap_b_shared"])},
            "acc_own": round(sum(correct) / n, 4),
        }
        ev = runs_root() / "eval"
        (ev / f"{arm}.harden_paired.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
        (ev / f"{arm}.harden_paired.records.json").write_text(json.dumps(out_rows, ensure_ascii=False))
        (ev / f"{arm}.harden_paired.arrays.json").write_text(json.dumps(
            {"ids": kept_ids, "P": P, "F": F, "ACC": ACC, "correct": correct}, ensure_ascii=False))
        sw.finish(metrics={"acc_own": summary["acc_own"], "sens_bs": sens_bs["point"]})
        write_marker(f"S3HP_{arm.upper()}_DONE", {"n": n, "sens_bs": sens_bs["point"]})
        print(json.dumps(summary, indent=1, ensure_ascii=False))
        vlm.close_readers()
        return

    sens = {k: diff_ci(F[k], F["para"]) for k in ("swap_b", "swap_b_shared", "swap_r", "swap_both")}
    util = {"U_b": diff_ci(P["own"], P["swap_b"]),
            "U_b_shared": diff_ci(P["own"], P["swap_b_shared"]),
            "U_g": diff_ci(P["own"], P["swap_both"]),
            "U_empty": diff_ci(P["own"], P["empty"])}
    fl = [i for i in range(n) if F["swap_b"][i]]
    c_idx = [i for i in range(n) if correct[i]]
    w_idx = [i for i in range(n) if not correct[i]]
    summary = {
        "arm": arm, "n": n, "paired_plan": PLAN_NAME, "donor_arm": plan["donor_arm"],
        "scoring": {"frames": not args.no_frames, "cand_chunk": args.cand_chunk,
                    "n_frames_used": n - n_noframe, "n_frame_skipped": n_noframe,
                    "n_sample_skipped": n_skip,
                    "note": "프레임 포함 vision-grounded 채점 · assistant 헤더 중복 수정본 — "
                            "기존 harden_s3 수치와 절대값 비교 불가(내부·arm간 비교는 유효)"},
        "flip": {k: bootstrap_ci(F[k]) for k in FLIPPABLE},
        "p_gt": {k: bootstrap_ci(P[k]) for k in VARIANTS},
        "sensitivity": sens,
        "utility": util,
        "directional_dg": bootstrap_ci([int(P["own"][i] > P["swap_b"][i]) for i in range(n)]),
        "acc_by_variant": {k: bootstrap_ci(ACC[k]) for k in VARIANTS},
        "acc_drop_vs_own": {k: diff_ci(ACC["own"], ACC[k]) for k in VARIANTS if k != "own"},
        "correct_switch": {
            "flip_rate_swap_b": round(sum(F["swap_b"]) / n, 4),
            "mean_pgt_drop_on_flip": round(
                sum(P["own"][i] - P["swap_b"][i] for i in fl) / max(1, len(fl)), 4),
            "n_flip": len(fl)},
        "acc_orthogonality": {
            "acc_own": round(sum(correct) / n, 4),
            "flip_both_on_correct": round(sum(F["swap_both"][i] for i in c_idx) / max(1, len(c_idx)), 4),
            "flip_both_on_wrong": round(sum(F["swap_both"][i] for i in w_idx) / max(1, len(w_idx)), 4)},
        "style_confound": {
            "note": "swap_b(arm 내부 문체) − swap_b_shared(전 arm 동일 문자열). 0에서 멀수록 "
                    "기존 harden_s3 의 arm 간 비교가 문체에 오염돼 있었다는 뜻.",
            "delta_sensitivity": round(sens["swap_b"]["point"] - sens["swap_b_shared"]["point"], 4),
            "delta_U_b": round(util["U_b"]["point"] - util["U_b_shared"]["point"], 4)},
        "gates_self": {"CC1_belief_sensitivity": sens["swap_b"]["lo"] > 0,
                       "CC1_shared": sens["swap_b_shared"]["lo"] > 0,
                       "CC3_belief_only_utility": util["U_b"]["lo"] > 0},
    }
    ev = runs_root() / "eval"
    (ev / f"{arm}.harden_paired.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    (ev / f"{arm}.harden_paired.records.json").write_text(json.dumps(out_rows, ensure_ascii=False))
    (ev / f"{arm}.harden_paired.arrays.json").write_text(json.dumps(
        {"ids": kept_ids, "P": P, "F": F, "ACC": ACC, "correct": correct}, ensure_ascii=False))
    sw.finish(metrics={"acc_own": summary["acc_orthogonality"]["acc_own"],
                       "sens_b": sens["swap_b"]["point"]})
    write_marker(f"S3HP_{arm.upper()}_DONE", {"n": n, "sens_b": sens["swap_b"]["point"]})
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    vlm.close_readers()


# ── stage: agg ───────────────────────────────────────────────────────────────
def stage_agg(args):
    arms = [a.strip() for a in args.arms.split(",")]
    ev = runs_root() / "eval"
    A = {}
    for a in arms:
        p = ev / f"{a}.harden_paired.arrays.json"
        if p.is_file():
            A[a] = json.loads(p.read_text())
        else:
            print(f"[agg] SKIP {a} — {p.name} 없음")
    if len(A) < 2:
        raise SystemExit("[agg] arm 이 2개 미만")
    # arm 마다 OOM/프레임 결손으로 빠진 샘플이 다를 수 있어 **교집합**으로 정렬한다
    common = set.intersection(*[set(d["ids"]) for d in A.values()])
    ids0 = [s_ for s_ in next(iter(A.values()))["ids"] if s_ in common]
    for a, d in A.items():
        pos = {s_: k for k, s_ in enumerate(d["ids"])}
        sel = [pos[s_] for s_ in ids0]
        d["P"] = {k: [v[j] for j in sel] for k, v in d["P"].items()}
        d["F"] = {k: [v[j] for j in sel] for k, v in d["F"].items()}
        if "ACC" in d:
            d["ACC"] = {k: [v[j] for j in sel] for k, v in d["ACC"].items()}
        d["correct"] = [d["correct"][j] for j in sel]
        if len(d["ids"]) != len(ids0):
            print(f"[agg] {a}: {len(d['ids'])} → 공통 {len(ids0)} 으로 정렬")
    n = len(ids0)

    def sens(d, k):
        return [d["F"][k][i] - d["F"]["para"][i] for i in range(n)]

    def ub(d, k="swap_b"):
        return [d["P"]["own"][i] - d["P"][k][i] for i in range(n)]

    ref = args.ref if args.ref in A else list(A)[-1]
    pairs = {}
    for a in A:
        if a == ref:
            continue
        pairs[f"{ref} − {a}"] = {
            "belief_sensitivity": diff_ci(sens(A[ref], "swap_b"), sens(A[a], "swap_b")),
            "belief_sensitivity_shared": diff_ci(sens(A[ref], "swap_b_shared"),
                                                 sens(A[a], "swap_b_shared")),
            "reasoning_sensitivity": diff_ci(sens(A[ref], "swap_r"), sens(A[a], "swap_r")),
            "both_sensitivity": diff_ci(sens(A[ref], "swap_both"), sens(A[a], "swap_both")),
            "U_b": diff_ci(ub(A[ref]), ub(A[a])),
            "U_b_shared": diff_ci(ub(A[ref], "swap_b_shared"), ub(A[a], "swap_b_shared")),
            "acc_own": diff_ci(A[ref]["correct"], A[a]["correct"]),
        }
    bk = f"{ref} − base"
    g_cc2 = ({"definition": "학습 arm 의 belief sensitivity 가 동일 셋 base 보다 유의하게 높은가",
              "delta": pairs[bk]["belief_sensitivity"],
              "delta_shared": pairs[bk]["belief_sensitivity_shared"],
              "verdict": "PASS" if pairs[bk]["belief_sensitivity"]["lo"] > 0 else "FAIL",
              "verdict_shared": "PASS" if pairs[bk]["belief_sensitivity_shared"]["lo"] > 0 else "FAIL"}
             if bk in pairs else {"verdict": "N/A — base arm 미포함"})
    out = {"n": n, "arms": list(A), "ref": ref,
           "per_arm": {a: json.loads((ev / f"{a}.harden_paired.json").read_text()) for a in A},
           "paired_diff_vs_ref": pairs, "G_CC2": g_cc2}
    dst = ev / "harden_paired_summary.json"
    dst.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    write_marker("S3HP_AGG_DONE", {"n": n, "arms": list(A), "G_CC2": g_cc2.get("verdict")})
    print(json.dumps({k: out[k] for k in ("n", "arms", "ref", "paired_diff_vs_ref", "G_CC2")},
                     indent=1, ensure_ascii=False))
    print(f"\n[agg] -> {dst}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["plan", "run", "agg"])
    ap.add_argument("--arms", default="base,theta_ce,sft_r15")
    ap.add_argument("--arm", default=None)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--donor", default="base", help="shared swap belief 를 제공할 arm")
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ref", default="sft_r15")
    ap.add_argument("--config", default="configs/step2_retrospection/cesft_v2.yaml")
    ap.add_argument("--cand_chunk", type=int, default=10, help="후보 채점 배치 행 수")
    ap.add_argument("--no_frames", action="store_true", help="프레임 없이(기존 harden_s3 조건)")
    ap.add_argument("--bench", type=int, default=0, help=">0 이면 그 개수만 돌려 속도 측정")
    ap.add_argument("--curve", action="store_true",
                    help="스텝별 belief 곡선용 — CURVE_VARIANTS 3개만 채점하고 축소 요약을 낸다")
    args = ap.parse_args()
    {"plan": stage_plan, "agg": stage_agg}.get(args.stage, stage_run)(args)


if __name__ == "__main__":
    main()
