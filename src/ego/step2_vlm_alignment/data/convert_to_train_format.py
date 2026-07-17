"""convert_to_train_format.py — grpo_dataset.jsonl → TRL GRPO 학습 포맷.

GRPO_TRAIN_SPEC.md 매핑 (실험 4/5a/5b/6 공통, superset 출력):
  wm_output.top5_action[i] {verb_class,noun_class,likelihood} → topk_actions[i] {verb,noun,score}
                                                              → topk_actions_with_score[i] {rank,verb,noun,likelihood}
  wm_output.top5_verb[i]   {verb,likelihood}                  → topk_verbs[i] (이름, rank순)
  wm_output.top5_noun[i]   {noun,likelihood}                  → topk_nouns[i]   {noun,score}
                                                              → topk_nouns_with_score[i] {rank,noun,likelihood}
  gt_label.verb/noun                                          → gt_verb / gt_noun
  memory_context.task_history (list)                          → memory_context (str)
  frame_path (상대)                                           → image_path (절대)

--mode {all, think_format, wm_ranking, noun_ranking} 는 모두 동일한 superset 을 출력한다.
(reward_mode 별 사용 필드만 train 스크립트가 골라 쓰므로 단일 파일이 모든 실험에 호환된다.
 think 계열의 "score 제거 + 셔플"은 prompt 생성 시점(train 스크립트)에 처리한다.)

출력 각 라인 필수 필드:
  image_path, topk_nouns, topk_actions, topk_verbs,
  topk_actions_with_score, topk_nouns_with_score, gt_verb, gt_noun
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

EGO_ROOT = Path(os.path.expanduser("~/work/jihun/EGO"))
ANN = EGO_ROOT / "src/epic-kitchens-100-annotations"
VERB_ID2KEY = pd.read_csv(ANN / "EPIC_100_verb_classes.csv").set_index("id")["key"].to_dict()
NOUN_ID2KEY = pd.read_csv(ANN / "EPIC_100_noun_classes.csv").set_index("id")["key"].to_dict()

MODES = ["all", "think_format", "wm_ranking", "noun_ranking"]


def serialize_memory(task_history: list[str]) -> str:
    """task_history list → 짧은 자연어 문자열. 비어있으면 빈 문자열."""
    if not task_history:
        return ""
    # 최근 행동이 마지막. 가독성 위해 화살표 직렬화.
    return "Previously completed actions: " + " -> ".join(task_history) + "."


def convert(rec: dict) -> dict | None:
    wm = rec.get("wm_output", {})
    top5_action = wm.get("top5_action", [])
    top5_noun = wm.get("top5_noun", [])
    top5_verb = wm.get("top5_verb", [])
    if not top5_action:
        return None

    # top5_action 은 verb_class/noun_class 만 분리 보관 → CSV 로 key 복원.
    topk_actions = [
        {
            "verb": VERB_ID2KEY[int(a["verb_class"])],
            "noun": NOUN_ID2KEY[int(a["noun_class"])],
            "score": a.get("likelihood"),
        }
        for a in top5_action
    ]
    # rank 정보를 보존한 버전 (wm_ranking reward 용). top5_action 은 likelihood 내림차순(rank=i+1).
    topk_actions_with_score = [
        {
            "rank": a.get("rank", i + 1),
            "verb": VERB_ID2KEY[int(a["verb_class"])],
            "noun": NOUN_ID2KEY[int(a["noun_class"])],
            "likelihood": a.get("likelihood"),
        }
        for i, a in enumerate(top5_action)
    ]
    topk_nouns = [{"noun": n["noun"], "score": n.get("likelihood")} for n in top5_noun]
    topk_nouns_with_score = [
        {"rank": n.get("rank", i + 1), "noun": n["noun"], "likelihood": n.get("likelihood")}
        for i, n in enumerate(top5_noun)
    ]
    # think 계열 입력용: verb/noun 후보 이름만 (rank순; 셔플·score제거는 train 프롬프트에서)
    topk_verbs = [v["verb"] for v in top5_verb] if top5_verb else \
        list(dict.fromkeys(a["verb"] for a in topk_actions))

    frame_path = rec["frame_path"]
    image_path = frame_path if os.path.isabs(frame_path) else str(EGO_ROOT / frame_path)

    gt = rec["gt_label"]
    mem = rec.get("memory_context", {}) or {}
    return {
        "image_path": image_path,
        "episode_id": rec.get("video_id", ""),
        "frame_id": rec.get("sample_id", ""),
        "task_goal": rec.get("task_goal", ""),
        "topk_nouns": topk_nouns,
        "topk_actions": topk_actions,
        "topk_verbs": topk_verbs,
        "topk_actions_with_score": topk_actions_with_score,
        "topk_nouns_with_score": topk_nouns_with_score,
        "memory_context": serialize_memory(mem.get("task_history", [])),
        "gt_verb": gt["verb"],
        "gt_noun": gt["noun"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(EGO_ROOT / "data/grpo_dataset/grpo_dataset.jsonl"))
    ap.add_argument("--output", default=str(EGO_ROOT / "data/grpo_dataset/grpo_train.jsonl"))
    ap.add_argument("--mode", default="all", choices=MODES,
                    help="모든 mode 가 동일 superset 출력 (spec 명령 호환용). 필드 검증 메시지만 다름.")
    ap.add_argument("--n", type=int, default=None, help="앞에서 n개만 (None=전체)")
    args = ap.parse_args()

    inp = Path(args.input)
    out = Path(args.output)
    rows = [json.loads(l) for l in inp.read_text().splitlines() if l.strip()]
    print(f"[load] {len(rows)} records from {inp}  (mode={args.mode})")

    converted = []
    n_drop_img = 0
    n_drop_fmt = 0
    for r in rows:
        c = convert(r)
        if c is None:
            n_drop_fmt += 1
            continue
        if not os.path.exists(c["image_path"]):
            n_drop_img += 1
            continue
        converted.append(c)
        if args.n and len(converted) >= args.n:
            break

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for c in converted:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"[done] wrote {len(converted)} → {out}")
    print(f"  dropped (no top5_action): {n_drop_fmt}, (image missing): {n_drop_img}")

    # 필드 무결성 검증
    c0 = converted[0]
    need = ["topk_verbs", "topk_actions_with_score", "topk_nouns_with_score"]
    for k in need:
        assert k in c0 and c0[k], f"missing/empty field: {k}"
    print(f"[check] topk_verbs={len(c0['topk_verbs'])} "
          f"topk_actions_with_score={len(c0['topk_actions_with_score'])} "
          f"topk_nouns_with_score={len(c0['topk_nouns_with_score'])}")

    # 육안 확인용 샘플
    print("\n=== sample[0] ===")
    print(json.dumps(c0, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
