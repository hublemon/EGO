"""로컬 스모크용 support_val.jsonl 생성 — WM prior 대체(파생 store 부재).

정식 경로(build_support_hk8)는 goalstep_history_context_store([17,1024] summaries +
frozen visual logits)와 retro4 index parquet을 요구하는데 로컬에 둘 다 없다.
여기서는 retro4 시간 계약(A2.end−1s 관측 / strict-next A3 타깃)만 정확히 재현하고,
Top-10 후보는 WM prior 대신 동영상/전역 빈도 기반 distractor로 합성한다.
→ 후보 분포가 학습 분포와 다르므로 SelAcc는 sft_r15.json과 직접 비교 불가.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path("/home/hogun/Project/EGO")
TAX = REPO / "src/ego/step1_action_anticipation/goalstep/taxonomy/goalstep_verbnoun_taxonomy.json"
LABELS = REPO / "src/ego/step1_action_anticipation/goalstep/taxonomy/goalstep_step_labels.csv"
VIDEO_DIR = REPO / "data/Ego4D/v2/goalstep_videos"

TOP_K = 10
L_OBS = 8.0      # 최대 관측창
TAU_A = 1.0      # 최소 anticipation gap
EPS = 1e-3


def load_rows():
    tax = json.loads(TAX.read_text())
    verbs, nouns = tax["verbs"], tax["nouns"]
    per_video = defaultdict(set)
    split_of = {}
    with open(LABELS, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            per_video[r["video_uid"]].add(
                (float(r["start_time"]), float(r["end_time"]),
                 verbs[int(r["verb_label"])], nouns[int(r["noun_label"])]))
            split_of[r["video_uid"]] = r["split"]
    timelines = {v: sorted(s) for v, s in per_video.items()}
    return timelines, split_of


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", default=str(REPO / "runs/cesft_v2/data/support_val.jsonl"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    local = {p.stem for p in VIDEO_DIR.glob("*.mp4")}
    timelines, split_of = load_rows()
    val_vids = sorted(v for v in local if split_of.get(v) == "val")
    print(f"local videos={len(local)} val={len(val_vids)}: {val_vids}")

    global_actions = Counter()
    for v, tl in timelines.items():
        for _, _, vb, nn in tl:
            global_actions[f"{vb} {nn}"] += 1

    samples = []
    for vid in val_vids:
        tl = timelines[vid]
        per_vid = Counter(f"{vb} {nn}" for _, _, vb, nn in tl)
        for i, (a2_start, a2_end, _, _) in enumerate(tl):
            # strict-next A3: A2가 끝난 뒤(또는 동시에) 시작하는 첫 action
            nxt = next((tl[j] for j in range(i + 1, len(tl))
                        if tl[j][0] >= a2_end - EPS), None)
            if nxt is None:
                continue
            a3_start, a3_end, gt_v, gt_n = nxt
            obs_end = a2_end - 1.0
            obs_start = max(a2_start, a2_end - (L_OBS + 1.0))
            if obs_end - obs_start < 1.0:          # 관측창 최소 1s
                continue
            if a3_start - obs_end < TAU_A - EPS:   # 최소 gap
                continue
            gt = f"{gt_v} {gt_n}"

            sid = f"{vid}_s{i:04d}"
            rng = random.Random(f"cand:{sid}")
            # distractor: 같은 영상 빈도 우선(절차적으로 그럴듯) → 전역 빈도로 보충
            pool_v = [a for a in per_vid if a != gt]
            pool_g = [a for a in global_actions if a != gt]
            picked: list[str] = []
            for pool, want in ((pool_v, 6), (pool_g, TOP_K - 1)):
                w = [per_vid[a] if pool is pool_v else global_actions[a] for a in pool]
                avail = [(a, wi) for a, wi in zip(pool, w) if a not in picked]
                while avail and len(picked) < want:
                    tot = sum(wi for _, wi in avail)
                    x = rng.random() * tot
                    for k, (a, wi) in enumerate(avail):
                        x -= wi
                        if x <= 0:
                            picked.append(a)
                            avail.pop(k)
                            break
            if len(picked) < TOP_K - 1:
                continue
            cands = picked[: TOP_K - 1] + [gt]
            rng2 = random.Random(f"shuffle:{sid}")
            rng2.shuffle(cands)
            # WM prior 없음 → 결정적 의사난수 점수 (L0는 무작위 기준선, 실제 WM 아님)
            rng3 = random.Random(f"score:{sid}")
            scores = sorted((rng3.random() for _ in cands), reverse=True)
            rng3.shuffle(scores)

            samples.append({
                "sample_id": sid, "video_uid": vid, "clip_uid": vid,
                "obs_start_sec": round(obs_start, 3), "obs_end_sec": round(obs_end, 3),
                "target_start_sec": round(a3_start, 3),
                "gt_verb": gt_v, "gt_noun": gt_n,
                "candidates": cands, "wm_scores": [round(s, 6) for s in scores],
                "boundary_flag": False,
                "gt_rank": cands.index(gt) + 1,
                "scenario": "COOKING:GOALSTEP",
                "split": "heldout",
                "target_horizon_sec": round(a3_start - obs_end, 3),
                "annotation_level": "step_labels",
            })

    print(f"eligible samples={len(samples)}")
    rng = random.Random(args.seed)
    rng.shuffle(samples)
    sel = sorted(samples[: args.n], key=lambda r: r["sample_id"])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for s in sel:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    hz = [s["target_horizon_sec"] for s in sel]
    ow = [s["obs_end_sec"] - s["obs_start_sec"] for s in sel]
    print(f"wrote {len(sel)} -> {out}")
    print(f"horizon(gap) mean={sum(hz)/len(hz):.2f}s min={min(hz):.2f} max={max(hz):.2f}")
    print(f"obs window mean={sum(ow)/len(ow):.2f}s min={min(ow):.2f} max={max(ow):.2f}")
    print(f"videos={Counter(s['video_uid'] for s in sel)}")


if __name__ == "__main__":
    main()
