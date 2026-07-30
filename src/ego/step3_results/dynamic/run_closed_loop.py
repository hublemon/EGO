"""Closed-Loop Dynamic Planning 러너 — WM 예측 → VLM 선택을 영상 끝까지 반복.

한 에피소드(=한 영상, 한 goal) 안에서 결정지점 k = 0,1,2,... 를 시간순으로 밟는다:

    입력  8프레임([t−5s, t−1s]) + GOAL + **모델 자신이 앞서 고른 action 열** +
          **모델 자신의 이전 task_belief** + WM top-10 후보
    출력  <reasoning> / <task_belief> / <action∈top10>
    갱신  고른 action 과 belief 를 다음 스텝 입력에 누적

GT 는 채점에만 쓰고 프롬프트에 넣지 않는다. 유일한 예외가 `oracle_gt_hist` arm 이며,
이 arm 은 "자기 히스토리 오염의 비용"을 재기 위한 의도적 대조군이다.

**스텝 동기 배치**: 에피소드끼리는 독립이므로 같은 step_idx 를 여러 에피소드에서 모아
한 번에 생성한다(에피소드 내부는 순차 유지). 배치 크기만큼 처리량이 오른다.

사용:
  PYTHONPATH=src python -m ego.step3_results.dynamic.run_closed_loop \
      --arm ego_closed --adapter outputs/step2_retrospection/cesft_v2/sft_r15/adapter
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

from ego.step3_results.dynamic import common as C
from ego.step3_results.vpa.v2 import frames as F

DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_ADAPTER = "outputs/step2_retrospection/cesft_v2/sft_r15/adapter"


def load_model(model_path: str, adapter: str | None):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, dtype=torch.bfloat16,
        device_map="cuda:0" if torch.cuda.is_available() else "auto")
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    # ⚠ 패치는 로딩 **후**. 먼저 걸면 로딩 경로의 prod 호출이 CPU 왕복으로 누적돼 사실상 멈춘다.
    from ego.step3_results.vpa.v2 import gb10_compat
    if gb10_compat.apply():
        print("[info] GB10(sm_121) nvrtc JIT 우회 패치 적용")
    return model, processor


def generate_batch(model, processor, batch: list[tuple[str, str, list]], max_new_tokens: int) -> list[str]:
    """[(system, user, images)] → [생성 텍스트]. 좌측 패딩 배치 (step2 generate_batch 와 동일 방식)."""
    import torch

    messages_list = []
    for system, user, images in batch:
        content = [{"type": "image", "image": im} for im in images]
        content.append({"type": "text", "text": user})
        messages_list.append([{"role": "system", "content": [{"type": "text", "text": system}]},
                              {"role": "user", "content": content}])
    texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
             for m in messages_list]
    images = [[c["image"] for msg in m for c in msg["content"]
               if isinstance(c, dict) and c.get("type") == "image"] for m in messages_list]
    tok = processor.tokenizer
    old_side = tok.padding_side
    tok.padding_side = "left"
    try:
        inputs = processor(text=texts, images=images, return_tensors="pt", padding=True).to(model.device)
    finally:
        tok.padding_side = old_side
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    new = gen[:, inputs["input_ids"].shape[1]:]
    return tok.batch_decode(new, skip_special_tokens=True)


class EpisodeState:
    """에피소드 하나의 루프 상태 — 모델 자신의 선택/belief 만 담는다."""

    def __init__(self, ep: dict, arm: str):
        self.ep, self.arm = ep, arm
        self.chosen: list[str] = []          # 모델이 고른 action (arm=pred 히스토리의 원천)
        self.beliefs: list[tuple[int, str]] = []
        self.k = 0                            # 다음에 처리할 step_idx

    def history(self) -> list[str]:
        if C.ARMS[self.arm]["history"] == "gt":
            return [s["gt_action"] for s in self.ep["steps"][: self.k]]
        return list(self.chosen)

    def prompt(self) -> tuple[str, str]:
        step = self.ep["steps"][self.k]
        return (C.system_prompt(self.arm),
                C.user_prompt(step, self.ep["goal_text"], self.history(), self.beliefs, self.arm))

    def apply(self, action: str, belief: str) -> None:
        self.chosen.append(action)
        self.beliefs.append((self.k, belief))
        self.k += 1


def restore(states: dict[str, EpisodeState], rec_path: Path) -> int:
    """중단 재개 — 기록된 (video_uid, step_idx) 를 순서대로 되먹여 상태를 복원한다.
    greedy 디코딩이라 같은 접두사에서 같은 결과가 나오므로 이어 붙여도 궤적이 일관된다."""
    done = 0
    per_vid: dict[str, dict[int, dict]] = defaultdict(dict)
    for r in C.read_jsonl(rec_path):
        per_vid[r["video_uid"]][r["step_idx"]] = r
    for vid, st in states.items():
        recs = per_vid.get(vid, {})
        k = 0
        while k in recs:
            st.apply(recs[k]["pred_action"], recs[k].get("belief", ""))
            k += 1
            done += 1
    return done


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", choices=sorted(C.ARMS), required=True)
    p.add_argument("--episodes", default="runs/dynamic_v1/episodes.json")
    p.add_argument("--out-dir", default="runs/dynamic_v1/preds")
    p.add_argument("--model-path", default=DEFAULT_MODEL)
    p.add_argument("--adapter", default=DEFAULT_ADAPTER, help="'none' 이면 백본 그대로")
    p.add_argument("--cache-root", default=f"runs/vpa_v2/{F.cache_dirname()}")
    p.add_argument("--batch-size", type=int, default=4, help="에피소드 간 스텝 동기 배치 크기")
    p.add_argument("--max-new-tokens", type=int, default=C.MAX_NEW_TOKENS)
    p.add_argument("--limit-episodes", type=int, default=None)
    p.add_argument("--limit-steps", type=int, default=None, help="에피소드당 처리 스텝 상한(스모크용)")
    args = p.parse_args()

    data = C.load_json(args.episodes)
    episodes = data["episodes"][: args.limit_episodes] if args.limit_episodes else data["episodes"]
    cache_root = Path(args.cache_root)

    # 프레임이 없는 스텝이 있으면 그 에피소드는 사슬이 끊기므로 통째로 제외한다
    # (중간을 건너뛰면 "연속된 관찰" 전제가 깨진다 — 조용히 이어붙이지 않는다).
    usable, skipped = [], []
    for e in episodes:
        miss = [s["sample_id"] for s in e["steps"]
                if not all(q.is_file() for q in F.frame_paths(
                    cache_root, {"video_uid": e["video_uid"], "sample_id": s["sample_id"]}))]
        if miss:
            skipped.append((e["video_uid"], len(miss)))
        else:
            usable.append(e)
    if skipped:
        print(f"[warn] 프레임 결손으로 제외된 에피소드 {len(skipped)}개: {skipped[:5]}")
    episodes = usable
    if args.limit_steps:
        episodes = [{**e, "steps": e["steps"][: args.limit_steps],
                     "n_steps": min(e["n_steps"], args.limit_steps)} for e in episodes]

    n_steps = sum(e["n_steps"] for e in episodes)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rec_path = out_dir / f"{args.arm}.records.jsonl"
    states = {e["video_uid"]: EpisodeState(e, args.arm) for e in episodes}
    resumed = restore(states, rec_path)

    print(f"[info] arm={args.arm} ({C.ARMS[args.arm]['why']})")
    print(f"[info] episodes={len(episodes)} steps={n_steps} resumed={resumed} todo={n_steps - resumed}")
    if resumed >= n_steps:
        print("[info] 남은 스텝 없음.")
        return

    from PIL import Image
    adapter = None if args.adapter in (None, "none", "") else args.adapter
    t_load = time.time()
    model, processor = load_model(args.model_path, adapter)
    print(f"[info] model loaded in {time.time() - t_load:.0f}s (adapter={adapter})", flush=True)

    max_k = max(e["n_steps"] for e in episodes)
    t0, done = time.time(), 0
    with open(rec_path, "a") as fh:
        for k in range(max_k):
            pending = [st for st in states.values() if st.k == k and k < st.ep["n_steps"]]
            for i in range(0, len(pending), args.batch_size):
                chunk = pending[i: i + args.batch_size]
                batch = []
                for st in chunk:
                    system, user = st.prompt()
                    step = st.ep["steps"][st.k]
                    imgs = [Image.open(q).convert("RGB") for q in F.frame_paths(
                        cache_root, {"video_uid": st.ep["video_uid"], "sample_id": step["sample_id"]})]
                    batch.append((system, user, imgs))
                try:
                    outs = generate_batch(model, processor, batch, args.max_new_tokens)
                    err = None
                except Exception as exc:  # noqa: BLE001 — 배치 하나가 죽어도 루프는 계속
                    outs, err = [""] * len(chunk), str(exc)[:200]
                    print(f"[error] step {k} batch: {err}", flush=True)

                for st, raw in zip(chunk, outs):
                    step = st.ep["steps"][st.k]
                    trace = C.parse_trace(raw)
                    malformed = trace is None
                    raw_action = (trace or {}).get("action", "")
                    belief = (trace or {}).get("task_belief", "")
                    reasoning = (trace or {}).get("reasoning", "")
                    action, forced = C.force_into_candidates(raw_action, step["candidates"])
                    rec = {
                        "arm": args.arm, "video_uid": st.ep["video_uid"], "step_idx": st.k,
                        "sample_id": step["sample_id"], "target_start_sec": step["target_start_sec"],
                        "pred_action": action, "raw_action": raw_action, "forced": forced,
                        "malformed": malformed, "belief": belief, "reasoning": reasoning,
                        "gt_action": step["gt_action"], "correct": action == step["gt_action"],
                        "gt_in_candidates": step["gt_in_candidates"], "gt_rank": step["gt_rank"],
                        "wm_top1": C.wm_top1(step), "candidates": step["candidates"],
                        "history_used": st.history(),
                        "beliefs_used": [b for _, b in st.beliefs[-C.BELIEF_CARRY:]],
                        "raw": raw[:1200], "error": err,
                    }
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    st.apply(action, belief)
                    done += 1
                fh.flush()
                el = time.time() - t0
                left = n_steps - resumed - done
                print(f"  step_idx={k} [{done}/{n_steps - resumed}] {el / max(1, done):.1f}s/step "
                      f"· ETA {left * el / max(1, done) / 60:.0f}min", flush=True)

    # 최종 예측 요약 (에피소드별 궤적)
    traj = {}
    for r in C.read_jsonl(rec_path):
        traj.setdefault(r["video_uid"], {})[r["step_idx"]] = r["pred_action"]
    C.dump_json(out_dir / f"{args.arm}.trajectories.json",
                {v: [traj[v][i] for i in sorted(traj[v])] for v in traj})
    print(f"\nwrote {rec_path} · {out_dir}/{args.arm}.trajectories.json")


if __name__ == "__main__":
    main()
