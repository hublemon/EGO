"""
② vjepa_infer_train.py — V-JEPA2 forward + softmax likelihood 추출.

selected_train.jsonl 의 각 샘플마다:
  - trigger_frame 기준 4초 클립(@8fps=32frame) 추출
  - V-JEPA2 encoder+predictor → AttentiveClassifier (verb/noun/action 3 head)
  - 각 head 의 logits → softmax → Top-5 (verb / noun / action) + likelihood

기존 src/make_samples/vjepa_infer.py 의 model/classifier 빌더 재사용.
spec 명시: likelihood 추출 가능 → 실제 softmax 값 저장 (null 사용 안 함).
tqdm 진행률 표시.

출력: data/grpo_dataset/predictions_train.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from decord import VideoReader, cpu
from tqdm import tqdm

EGO_ROOT = Path(os.path.expanduser("~/work/jihun/EGO"))
VJEPA2_SRC = EGO_ROOT / "src/vjepa2"
sys.path.insert(0, str(VJEPA2_SRC))

from evals.action_anticipation_frozen.dataloader import make_transforms  # noqa: E402
from evals.action_anticipation_frozen.epickitchens import filter_annotations as ek_filter  # noqa: E402
from evals.action_anticipation_frozen.models import init_classifier, init_module  # noqa: E402

CONFIG_YAML = EGO_ROOT / "configs/vitg-384/ek100_inference.yaml"
CLF_CKPT = Path("/mnt/ddn/prod-shared/datasets/EK100/checkpoints/vjepa2-vitg384/ek100-vitg-384.pt")
SELECTED = EGO_ROOT / "data/grpo_dataset/selected_train.jsonl"
OUT = EGO_ROOT / "data/grpo_dataset/predictions_train.jsonl"
VERB_CSV = EGO_ROOT / "src/epic-kitchens-100-annotations/EPIC_100_verb_classes.csv"
NOUN_CSV = EGO_ROOT / "src/epic-kitchens-100-annotations/EPIC_100_noun_classes.csv"

ANTICIPATION_SEC = 1.0


def load_video_path(video_id: str, base_path: str, file_format: int = 0) -> str:
    pid = video_id.split("_")[0]
    if file_format == 0:
        return os.path.join(base_path, pid, "videos", f"{video_id}.MP4")
    return os.path.join(base_path, pid, f"{video_id}.MP4")


def clip_from_reader(vr: "VideoReader", trigger_frame: int, frames_per_clip: int, fps_target: int):
    """이미 열린 VideoReader 에서 trigger_frame 기준 4초 클립 추출.

    추출 프레임은 (trigger_frame, fps_target, frames_per_clip) 만으로 결정되므로
    비디오당 VideoReader 를 한 번만 열어도 결과(프레임)는 샘플별 open 과 비트 단위로 동일.
    """
    vfps = vr.get_avg_fps()
    fstp = int(vfps / fps_target)
    nframes = frames_per_clip * fstp
    indices = np.arange(trigger_frame - nframes, trigger_frame, fstp).astype(np.int64)
    indices[indices < 0] = 0
    return vr.get_batch(indices).asnumpy()


def strip_module_prefix(sd: dict) -> dict:
    return {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}


def build_id_to_key(csv: Path) -> dict[int, str]:
    df = pd.read_csv(csv)
    return {int(r["id"]): r["key"] for _, r in df.iterrows()}


def top5_with_prob(logits: torch.Tensor) -> tuple[list[int], list[float]]:
    """logits → softmax → top-5 indices + probabilities."""
    probs = F.softmax(logits.float(), dim=-1)
    vals, idx = probs.topk(5)
    return idx.tolist(), [round(float(v), 6) for v in vals.tolist()]


def decode_topk(out, b, inv_verb, inv_noun, inv_action, verb_id2key, noun_id2key):
    """배치 b 번째 샘플의 verb/noun/action top-5 레코드 생성."""
    v_idx, v_prob = top5_with_prob(out["verb"][b])
    n_idx, n_prob = top5_with_prob(out["noun"][b])
    a_idx, a_prob = top5_with_prob(out["action"][b])
    top5_verb = []
    for r, (vid_, p) in enumerate(zip(v_idx, v_prob), 1):
        ov = int(inv_verb[vid_])
        top5_verb.append({"rank": r, "verb": verb_id2key[ov], "verb_class": ov, "likelihood": p})
    top5_noun = []
    for r, (nid, p) in enumerate(zip(n_idx, n_prob), 1):
        on = int(inv_noun[nid])
        top5_noun.append({"rank": r, "noun": noun_id2key[on], "noun_class": on, "likelihood": p})
    top5_action = []
    for r, (aid, p) in enumerate(zip(a_idx, a_prob), 1):
        ov, on = inv_action[aid]
        ov, on = int(ov), int(on)
        top5_action.append({
            "rank": r, "action": f"{verb_id2key[ov]} {noun_id2key[on]}",
            "verb_class": ov, "noun_class": on, "action_class": int(aid), "likelihood": p,
        })
    return top5_verb, top5_noun, top5_action


def assign_shards(by_video: dict, num_shards: int) -> list[dict]:
    """비디오를 샘플 수 기준 greedy 로 num_shards 개에 균등 분배."""
    loads = [0] * num_shards
    shards = [dict() for _ in range(num_shards)]
    for vid, vs in sorted(by_video.items(), key=lambda kv: -len(kv[1])):
        i = min(range(num_shards), key=lambda j: loads[j])
        shards[i][vid] = vs
        loads[i] += len(vs)
    return shards


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--resume", action="store_true",
                    help="기존 predictions_train.jsonl 을 읽어 처리된 sample_id 건너뜀")
    ap.add_argument("--num_shards", type=int, default=1, help="GPU 분할 수 (병렬 프로세스 수)")
    ap.add_argument("--shard_id", type=int, default=0, help="이 프로세스가 맡을 shard index")
    ap.add_argument("--batch_size", type=int, default=1, help="비디오 내 클립 배치 forward 크기")
    args = ap.parse_args()

    cfg = yaml.safe_load(CONFIG_YAML.read_text())
    data_cfg = cfg["experiment"]["data"]
    clf_cfg = cfg["experiment"]["classifier"]
    pretrain = cfg["model_kwargs"]

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # class mapping
    anns = ek_filter(
        base_path=data_cfg["base_path"],
        train_annotations_path=data_cfg["dataset_train"],
        val_annotations_path=data_cfg["dataset_val"],
        file_format=data_cfg.get("file_format", 0),
    )
    verb_classes = anns["verbs"]      # orig_verb_class → unified_verb_id
    noun_classes = anns["nouns"]      # orig_noun_class → unified_noun_id
    action_classes = anns["actions"]  # (orig_v, orig_n) → unified_action_id
    inv_verb = {v: k for k, v in verb_classes.items()}
    inv_noun = {v: k for k, v in noun_classes.items()}
    inv_action = {v: k for k, v in action_classes.items()}

    verb_id2key = build_id_to_key(VERB_CSV)
    noun_id2key = build_id_to_key(NOUN_CSV)

    print(f"[init] building model on {device} ...")
    model = init_module(
        module_name=pretrain["module_name"],
        frames_per_clip=data_cfg["frames_per_clip"],
        frames_per_second=data_cfg["frames_per_second"],
        resolution=data_cfg["resolution"],
        checkpoint=pretrain["checkpoint"],
        model_kwargs=pretrain["pretrain_kwargs"],
        wrapper_kwargs=pretrain["wrapper_kwargs"],
        device=device,
    )
    classifiers = init_classifier(
        embed_dim=model.embed_dim,
        num_heads=clf_cfg["num_heads"],
        num_blocks=clf_cfg["num_probe_blocks"],
        device=device,
        num_classifiers=1,
        action_classes=action_classes,
        verb_classes=verb_classes,
        noun_classes=noun_classes,
    )
    clf = classifiers[0]
    print(f"[init] loading classifier ckpt {CLF_CKPT}")
    ckpt = torch.load(CLF_CKPT, map_location="cpu", weights_only=False)
    clf.load_state_dict(strip_module_prefix(ckpt["classifiers"][0]))
    clf.eval()
    for p in clf.parameters():
        p.requires_grad = False

    transform = make_transforms(
        training=False,
        random_horizontal_flip=False,
        random_resize_aspect_ratio=(3/4, 4/3),
        random_resize_scale=(1.0, 1.0),
        reprob=0, auto_augment=False, motion_shift=False,
        crop_size=data_cfg["resolution"],
    )

    samples = [json.loads(l) for l in SELECTED.read_text().splitlines() if l.strip()]
    if args.limit:
        samples = samples[: args.limit]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    # shard>1 이면 shard 별 part 파일에 기록 (동시 append 충돌 방지). 나중에 머지.
    out_path = OUT if args.num_shards == 1 else OUT.with_suffix(f".part{args.shard_id}.jsonl")

    seen = set()
    if args.resume:
        # 정식 파일 + 모든 shard part 파일을 skip 대상으로.
        # (두 shard 가 동일한 seen 을 봐야 assign_shards 분배가 일치 → 누락/중복 방지)
        seen_files = [OUT] + sorted(OUT.parent.glob("predictions_train.part*.jsonl"))
        for p in seen_files:
            if p.exists():
                for line in p.read_text().splitlines():
                    if line.strip():
                        seen.add(json.loads(line)["sample_id"])
    print(f"[resume] already processed (skip): {len(seen)}")

    use_bf16 = cfg["experiment"]["optimization"]["use_bfloat16"]

    # 속도 최적화: video_id 별로 묶어 VideoReader 를 비디오당 한 번만 open.
    from collections import defaultdict
    by_video: dict[str, list] = defaultdict(list)
    for s in samples:
        if s["sample_id"] in seen:
            continue
        by_video[s["video_id"]].append(s)

    # shard 분배: 이 프로세스가 맡을 비디오만 남김
    if args.num_shards > 1:
        by_video = assign_shards(by_video, args.num_shards)[args.shard_id]
    n_todo = sum(len(v) for v in by_video.values())
    print(f"[shard {args.shard_id}/{args.num_shards}] videos={len(by_video)} samples={n_todo} "
          f"batch_size={args.batch_size} -> {out_path}")

    mode = "a" if args.resume else "w"
    out_f = out_path.open(mode)
    n_done = 0
    n_err = 0
    B = max(1, args.batch_size)
    fpc = data_cfg["frames_per_clip"]
    fps_t = data_cfg["frames_per_second"]
    pbar = tqdm(total=n_todo, desc=f"infer[{args.shard_id}]", unit="sample")
    for vid, vsamples in by_video.items():
        try:
            vpath = load_video_path(vid, data_cfg["base_path"], data_cfg.get("file_format", 0))
            vr = VideoReader(vpath, num_threads=-1, ctx=cpu(0))
        except Exception as e:
            n_err += len(vsamples)
            pbar.write(f"[err] open {vid}: {type(e).__name__}: {e}")
            pbar.update(len(vsamples))
            continue

        # 배치 단위로 클립을 모아 한 번에 forward
        for i in range(0, len(vsamples), B):
            chunk = vsamples[i:i + B]
            clips, ok_samples = [], []
            for s in chunk:
                try:
                    buf = clip_from_reader(vr, int(s["trigger_frame"]), fpc, fps_t)
                    clips.append(transform(buf))
                    ok_samples.append(s)
                except Exception as e:
                    n_err += 1
                    pbar.write(f"[err] {s['sample_id']}: {type(e).__name__}: {e}")
                    pbar.update(1)
            if not clips:
                continue
            try:
                batch = torch.stack(clips, 0).to(device)
                ant = torch.full((batch.shape[0],), ANTICIPATION_SEC, device=device)
                with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bf16):
                    feat = model(batch, ant)
                    out = clf(feat)
                for b, s in enumerate(ok_samples):
                    t5v, t5n, t5a = decode_topk(out, b, inv_verb, inv_noun, inv_action,
                                                verb_id2key, noun_id2key)
                    out_f.write(json.dumps({
                        "sample_id": s["sample_id"],
                        "top5_verb": t5v, "top5_noun": t5n, "top5_action": t5a,
                    }, ensure_ascii=False) + "\n")
                    n_done += 1
                out_f.flush()
            except Exception as e:
                n_err += len(ok_samples)
                pbar.write(f"[err] batch@{vid}: {type(e).__name__}: {e}")
            pbar.update(len(ok_samples))
            pbar.set_postfix(ok=n_done, err=n_err)
        del vr

    out_f.close()
    print(f"[done] shard {args.shard_id}: processed={n_done}, errors={n_err} → {out_path}")


if __name__ == "__main__":
    main()
