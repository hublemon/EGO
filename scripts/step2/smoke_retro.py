#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""smoke_retro.py — B0 full-trace DPO 결정론적 로직 스모크 (GPU/torch/model 불필요).

검증 대상 (핸드오프):
  §8·§10 routing table + SAME/SAME drop + UNCERTAIN/SAME audit
  §13 candidate support (gt ∈ D_t)
  §15 leakage assertions (GT/future/trace/equivalence 가 policy prompt 에 없음)
  §15 pair invariants (chosen/rejected 완결 full-trace, action 일치, no-splicing)
  §26 build_pairs 오케스트레이션 (mock teacher 주입)
  §18 accuracy split / recovery-regression, §20 coherence proxies

순수 로직만 — b0.teacher/build 의 GPU 경로(HF/torch)는 지역 import 라 로드 자체는 안전.
exit 0 = 전부 통과.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

PASS, FAIL = 0, 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [OK  ] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}" + (f" — {detail}" if detail else ""))


from ego.step2_vlm_alignment.retro import route_pairs as R          # noqa: E402
from ego.step2_vlm_alignment.retro import trace_utils as U          # noqa: E402
from ego.step2_vlm_alignment.retro import validate_dpo_dataset as V  # noqa: E402
from ego.step2_vlm_alignment.retro import build_dpo_dataset as B     # noqa: E402
from ego.step2_vlm_alignment.retro import evaluate_b0 as E          # noqa: E402


def test_routing() -> None:
    print("[1] routing table + SAME/SAME drop (§8·§10)")
    check("DIFFERENT/DIFFERENT → KEEP (train)", R.route("DIFFERENT", "DIFFERENT").training)
    check("DIFFERENT/SAME → KEEP (train)", R.route("DIFFERENT", "SAME").training)
    check("SAME/DIFFERENT → KEEP (train)", R.route("SAME", "DIFFERENT").training)
    r = R.route("SAME", "SAME")
    check("SAME/SAME → DROP (not train, audit)", (not r.training) and r.audit
          and r.decision == "DROP_SAME_SAME")
    check("SAME/SAME drop_reason=semantic_tie", r.drop_reason == "semantic_tie")
    ru = R.route("UNCERTAIN", "SAME")
    check("UNCERTAIN/SAME → DROP/audit", (not ru.training) and ru.audit)
    check("UNCERTAIN/DIFFERENT → KEEP+tag (train)", R.route("UNCERTAIN", "DIFFERENT").training
          and R.route("UNCERTAIN", "DIFFERENT").decision == "KEEP_TAG")


def test_action_relation_and_candidates() -> None:
    print("[2] action relation + candidate support (§8·§13)")
    check("action SAME on canonical match", R.action_relation("cut", "tomato", "cut", "tomato") == "SAME")
    check("action DIFFERENT otherwise", R.action_relation("take", "plate", "cut", "tomato") == "DIFFERENT")
    check("action SAME case-insensitive", R.action_relation("Cut", "Tomato", "cut", "tomato") == "SAME")
    cands = [{"verb": "cut", "noun": "tomato"}, {"verb": "take", "noun": "plate"}]
    check("gt in candidates", R.gt_in_candidates("cut", "tomato", cands))
    check("gt NOT in candidates", not R.gt_in_candidates("wash", "hand", cands))


def test_trace_utils() -> None:
    print("[3] full-trace parse / build (§14)")
    t = U.build_full_trace("I took the knife earlier.", "prepare a salad", "cut", "tomato")
    tr = U.parse_full_trace(t)
    check("round-trip reasoning", "took the knife" in tr.reasoning)
    check("round-trip belief", tr.belief == "prepare a salad")
    check("round-trip action", tr.verb == "cut" and tr.noun == "tomato")
    check("complete trace", tr.is_complete())
    check("incomplete detected", not U.parse_full_trace("<reasoning>x</reasoning>").is_complete())
    check("future-leak language flagged",
          U.has_future_leak_language("The task is spaghetti because future actions show pasta"))
    check("clean reasoning not flagged",
          not U.has_future_leak_language("A bowl is visible so taking it is plausible"))


def test_validators() -> None:
    print("[4] leakage + pair invariants (§15)")
    chosen = U.build_full_trace("A bowl is visible; food prep is underway.", "prepare a meal", "take", "bowl")
    rejected = U.build_full_trace("The pan looks hot.", "cook something", "stir", "pan")
    rec = {
        "record_id": "s1:0", "prompt": [{"role": "user", "content": "history: open drawer"}],
        "chosen": chosen, "rejected": rejected,
        "metadata": {"belief_relation": "DIFFERENT", "action_relation": "DIFFERENT"},
        "_leak_check": {
            "gt_action": {"verb": "take", "noun": "bowl"},
            "faa_action": {"verb": "stir", "noun": "pan"},
            "gt_action_str": "take bowl", "raw_task": "make soup",
            "projected_belief": "prepare a meal", "faa_belief": "cook something",
            "belief_relation": "DIFFERENT",
            "future_gt_actions": [{"verb": "pour", "noun": "soup"}],
            "projected_full_trace": chosen, "faa_full_trace": rejected,
        },
    }
    check("clean record: no leakage", V.check_prompt_leakage(rec) == [])
    check("clean record: pair invariants ok", V.check_pair_invariants(rec) == [])

    # GT action leaked into prompt
    leak = dict(rec); leak["prompt"] = [{"role": "user", "content": "next is take bowl"}]
    check("GT-in-prompt leak detected", len(V.check_prompt_leakage(leak)) > 0)

    # future action leaked
    leak2 = dict(rec); leak2["prompt"] = [{"role": "user", "content": "then pour soup"}]
    check("future-in-prompt leak detected", len(V.check_prompt_leakage(leak2)) > 0)

    # splicing: chosen action swapped to non-GT
    spliced = dict(rec)
    spliced["chosen"] = U.build_full_trace("A bowl is visible.", "prepare a meal", "stir", "pan")
    errs = V.check_pair_invariants(spliced)
    check("splicing (chosen.action != GT) detected", any("chosen.action" in e for e in errs))

    # SAME/SAME present in training (must be flagged)
    ss = dict(rec)
    ss["metadata"] = {"belief_relation": "SAME", "action_relation": "SAME"}
    check("SAME/SAME in train flagged", any("SAME/SAME" in e for e in V.check_pair_invariants(ss)))

    # future-knowledge language in chosen reasoning
    fl = dict(rec)
    fl["chosen"] = U.build_full_trace("because that is what actually happens next", "prep", "take", "bowl")
    fl["_leak_check"] = dict(rec["_leak_check"])
    fl["_leak_check"]["projected_full_trace"] = fl["chosen"]
    check("future-language in chosen flagged", any("future-knowledge" in e for e in V.check_pair_invariants(fl)))


class MockTeacher:
    """결정론 mock — projection 은 past-grounded, equivalence 는 belief 문자열로 판정."""
    def infer_raw_trace(self, gt_trajectory):
        return "<task_belief>make a meal</task_belief><reasoning>sequence builds a dish</reasoning>"

    def project_full_trace(self, raw_trace, memory_context, candidates, gt_verb, gt_noun,
                           image_path=None):
        return U.parse_full_trace(U.build_full_trace(
            "A relevant object is visible and prior actions support this step.",
            "prepare a meal", gt_verb, gt_noun))

    def equivalence(self, faa_belief, projected_belief):
        if faa_belief.strip().lower() == projected_belief.strip().lower():
            return "SAME"
        if "meal" in faa_belief.lower() and "meal" in projected_belief.lower():
            return "SAME"
        return "DIFFERENT"


def test_build_pairs() -> None:
    print("[5] build_pairs 오케스트레이션 (§26) + SAME/SAME 물리 분리")
    gt = {"verb": "take", "noun": "bowl"}
    cands = [{"verb": "take", "noun": "bowl"}, {"verb": "stir", "noun": "pan"}]
    faa_diff = U.build_full_trace("Pan is hot.", "cook pasta", "stir", "pan")     # belief DIFF, action DIFF
    faa_same = U.build_full_trace("Bowl visible.", "prepare a meal", "take", "bowl")  # belief SAME, action SAME
    samples = [{
        "sample_id": "s1", "prompt": [{"role": "user", "content": "history: open drawer"}],
        "image_path": "", "memory_context": "open drawer",
        "candidates": cands, "gt_action": gt,
        "future_gt_actions": [{"verb": "pour", "noun": "water"}],
        "faa_traces": [faa_diff, faa_same], "trigger_time": 10.0, "policy_history": [],
    }]
    train, audit, stats = B.build_pairs(samples, MockTeacher(), strict_validate=True)
    check("KEEP pair emitted (belief/action DIFFERENT)", len(train) == 1)
    check("SAME/SAME routed to audit", any(
        a["metadata"].get("training_status") == "DROPPED_SAME_SAME" for a in audit))
    check("stats: 1 same_same dropped", stats.dropped_same_same == 1)
    # 물리 분리: 저장 레코드에 _leak_check(원본 GT/future) 없음
    check("train record has no _leak_check (GT/future 물리 제거)",
          all("_leak_check" not in r for r in train))
    check("audit record has no _leak_check", all("_leak_check" not in r for r in audit))
    # 학습 레코드가 prompt 누설 없음 (validate 통과분만 emit)
    check("emitted train record passes leakage", all(
        V.check_prompt_leakage({**r, "_leak_check": {}}) == [] for r in train))

    # candidate support: gt 가 후보 밖이면 drop
    s2 = dict(samples[0]); s2["gt_action"] = {"verb": "wash", "noun": "hand"}
    _, _, st2 = B.build_pairs([s2], MockTeacher())
    check("gt outside candidates dropped", st2.gt_outside_candidates == 1)


def test_eval_math() -> None:
    print("[6] eval 순수 계산 (§18·§20)")
    ms = E.compute_margin_stats([1.0, -0.5, 2.0])
    check("preference_accuracy 2/3", abs(ms["preference_accuracy"] - 0.6667) < 0.01)
    preds = [
        {"pred_verb": "cut", "pred_noun": "tomato", "gt_verb": "cut", "gt_noun": "tomato", "gt_in_cand": True},
        {"pred_verb": "take", "pred_noun": "plate", "gt_verb": "cut", "gt_noun": "tomato", "gt_in_cand": True},
        {"pred_verb": "x", "pred_noun": "y", "gt_verb": "wash", "gt_noun": "hand", "gt_in_cand": False},
    ]
    acc = E.accuracy_split(preds)
    check("candidate_recall 2/3", abs(acc["candidate_recall"] - 0.6667) < 0.01)
    check("conditional 1/2", abs(acc["conditional_accuracy"] - 0.5) < 0.01)
    check("end_to_end 1/3", abs(acc["end_to_end_accuracy"] - 0.3333) < 0.01)
    faa_p = [{"pred_verb": "a", "pred_noun": "b", "gt_verb": "c", "gt_noun": "d", "gt_in_cand": True}]
    b0_p = [{"pred_verb": "c", "pred_noun": "d", "gt_verb": "c", "gt_noun": "d", "gt_in_cand": True}]
    rr = E.recovery_regression(faa_p, b0_p)
    check("recovery counted", rr["recovery"] == 1 and rr["regression"] == 0)
    coh = E.coherence_proxies(U.build_full_trace(
        "The task is spaghetti because future actions show pasta", "cut the tomato", "cut", "tomato"))
    check("coherence: future_leak flagged", coh["future_leak"])
    check("coherence: belief_restatement flagged", coh["belief_restatement"])


def main() -> None:
    print("=" * 68)
    print("B0 full-trace DPO — smoke test (pure logic, no GPU)")
    print("=" * 68)
    for fn in (test_routing, test_action_relation_and_candidates, test_trace_utils,
               test_validators, test_build_pairs, test_eval_math):
        try:
            fn()
        except Exception as e:
            global FAIL
            FAIL += 1
            print(f"  [FAIL] {fn.__name__} raised {type(e).__name__}: {e}")
    print("-" * 68)
    print(f"passed={PASS}  failed={FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
