#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""colab_train_f0_full.py — Step2 F0 v2 실데이터 학습 러너 (Colab A100 40GB, 단일 GPU).

smoke(colab_smoke_f0.py) 통과 후 **실제 grpo_train.jsonl** 로 validation/full 학습을 돌린다.
train_f0_final_v2.sh(2×H200) 레시피를 A100 40GB 단일 GPU 에 맞게 등가 변환한 것:

  원본:  2 GPU × per_device 8 → 스텝당 생성 16개 = 프롬프트 2개 × gen 8
  A100:  1 GPU × per_device 2 × grad_accum 4 → 스텝당 생성 8개 = 프롬프트 1개 × gen 8
  → num_generations=8 (gen 4 는 frac_reward_zero_std 붕괴 실측 — 절대 줄이지 않는다)
  → 같은 스텝 수면 프롬프트 방문 수가 원본의 절반. 나머지 레시피(동결값)는 전부 유지:
     dr_grpo · scale none · eps_high 0.28 · min_wm_spread 0.05 · hide_scores+shuffle ·
     beta 0 · T 1.0 · compl 384 · lr 1e-5 · LoRA r16/α32 · save 125

전제 (러너가 대신 못 해주는 것):
  - v2 데이터 빌드 산출물(grpo_train.jsonl + 4f grid JPEG 프레임)이 Drive 등에 있어야 한다.
    이 빌드는 학습 서버에서 build_f0_v2_data.sh 로 만든 것 — Colab 에서 EK100 원본으로
    재생성은 불가(영상 접근 게이트).
  - jsonl 의 image_path 가 서버 절대경로면 --path_map "서버프리픽스=콜랩프리픽스" 로 재작성.

세션 유의:
  - trainer 는 resume_from_checkpoint 미지원 → 세션이 끊기면 --adapter_path <마지막 checkpoint>
    로 재시작 (LoRA 가중치만 이어받음, 옵티마이저 상태는 초기화). save_steps 125 가 복구 지점.
  - output_dir 를 Drive 로 지정하면 끊겨도 체크포인트가 남는다 (권장).
  - full(1 epoch=5000 프롬프트=5000 스텝)은 단일 A100 에서 Colab 세션 한도를 넘길 가능성이 큼.
    Colab 에서는 validation(500 스텝) 권장, full 은 학습 서버(2×H200) 권장.

사용 예 (Colab 셀):
    !python scripts/step2/colab_train_f0_full.py \
        --train_jsonl /content/drive/MyDrive/ego/grpo_train.jsonl \
        --path_map "/home/server/EGO/data=/content/drive/MyDrive/ego/data" \
        --out_dir /content/drive/MyDrive/ego/runs/f0_v2_val_a100 \
        --run_mode validation --copy_frames_local
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def hr(title: str = "") -> None:
    print("\n" + "=" * 72)
    if title:
        print(title)
        print("=" * 72)


def sh(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    print(f"$ {' '.join(str(c) for c in cmd)}")
    return subprocess.run(cmd, **kw)


def install_deps(skip: bool) -> None:
    hr("[1] 의존성")
    if skip:
        print("--no_install → 건너뜀")
        return
    pkgs = ["transformers==5.9.0", "trl==1.5.1", "peft>=0.14", "accelerate>=1.0",
            "datasets>=3.0", "qwen-vl-utils", "pillow", "tensorboard"]
    r = sh([sys.executable, "-m", "pip", "install", "-q", *pkgs])
    if r.returncode != 0:
        print("⚠ pip install 실패")
        sys.exit(2)


def prepare_jsonl(a, work: Path) -> Path:
    """실 jsonl 로드 → path_map 적용 → 이미지 존재 검사 → (옵션) 로컬 복사 → 재작성 jsonl."""
    hr("[2] 데이터 준비 (path remap / 존재 검사 / 로컬 복사)")
    work.mkdir(parents=True, exist_ok=True)
    src = Path(a.train_jsonl)
    rows = [json.loads(l) for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"레코드 {len(rows)}개 로드: {src}")

    maps = []
    for m in a.path_map or []:
        if "=" not in m:
            print(f"✗ --path_map 형식 오류 (OLD=NEW): {m}")
            sys.exit(2)
        old, new = m.split("=", 1)
        maps.append((old, new))

    def remap(p: str) -> str:
        for old, new in maps:
            if p.startswith(old):
                return new + p[len(old):]
        return p

    # remap + 존재 검사
    missing, total = 0, len(rows)
    for r in rows:
        r["image_path"] = remap(str(r["image_path"]))
        if not Path(r["image_path"]).exists():
            missing += 1
    print(f"이미지 존재: {total - missing}/{total}  (missing={missing})")
    if missing == total:
        print("✗ 이미지가 하나도 없음 — --path_map 프리픽스를 확인하라.")
        print(f"  예시 image_path: {rows[0]['image_path']}")
        sys.exit(2)
    if missing:
        print(f"⚠ {missing}개는 학습 로더가 자동 skip 한다 (train_grpo_action 의 존재검사).")

    # Drive I/O 는 느리고 불안정 → 참조 이미지를 로컬 SSD 로 복사 (파일명 충돌 방지: md5)
    if a.copy_frames_local:
        local = work / "frames_local"
        local.mkdir(parents=True, exist_ok=True)
        n_copied = 0
        t0 = time.time()
        for r in rows:
            p = Path(r["image_path"])
            if not p.exists():
                continue
            dest = local / (hashlib.md5(str(p).encode()).hexdigest()[:16] + p.suffix)
            if not dest.exists():
                shutil.copy2(p, dest)
                n_copied += 1
            r["image_path"] = str(dest)
        print(f"로컬 복사: {n_copied}개 신규 ({time.time() - t0:.0f}s) → {local}")

    out_jsonl = work / "grpo_train_remapped.jsonl"
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"재작성 jsonl → {out_jsonl}")
    return out_jsonl


def main() -> None:
    p = argparse.ArgumentParser(description="F0 v2 real-data training on Colab A100 40GB")
    p.add_argument("--train_jsonl", required=True, help="실 grpo_train.jsonl (v2 빌드)")
    p.add_argument("--out_dir", required=True, help="출력/체크포인트 위치 — Drive 경로 권장")
    p.add_argument("--run_mode", choices=["validation", "full"], default="validation")
    p.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    p.add_argument("--path_map", action="append",
                   help="image_path 프리픽스 재작성 'OLD=NEW' (반복 가능)")
    p.add_argument("--copy_frames_local", action="store_true",
                   help="참조 이미지를 로컬 SSD 로 복사 (Drive 직접 읽기보다 빠르고 안정적)")
    p.add_argument("--adapter_path", default=None,
                   help="세션 재개: 마지막 checkpoint-N 경로 (가중치만 이어받음)")
    p.add_argument("--num_frames", type=int, default=4, choices=[1, 4],
                   help="데이터 빌드와 일치시킬 것 (v2=4f grid)")
    p.add_argument("--mask_frame_prob", type=float, default=0.0,
                   help="L2-a. validation=0.0, full 에서 프록시 재악화 시 0.15~0.2")
    p.add_argument("--max_steps", type=int, default=None,
                   help="validation 기본 500. save_steps(125) 배수여야 최종 체크포인트 생성")
    p.add_argument("--per_device_batch", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--max_pixels", type=int, default=768 * 28 * 28,
                   help="원본 레시피값. OOM 시 1차 폴백 401408(=512*28*28)")
    p.add_argument("--train_samples", type=int, default=5000)
    p.add_argument("--no_install", action="store_true")
    a = p.parse_args()

    # gen 8 고정 — global generation batch(per_device × accum)가 8의 배수여야 함
    gen_batch = a.per_device_batch * a.grad_accum
    if gen_batch % 8 != 0:
        print(f"✗ per_device_batch×grad_accum={gen_batch} 는 num_generations=8 의 배수여야 한다.")
        sys.exit(2)

    work = Path(os.environ.get("EGO_FULL_WORK", "/content/f0_work"))
    work.mkdir(parents=True, exist_ok=True)
    out_dir = Path(a.out_dir)

    hr("F0 v2 실데이터 학습 러너 (A100 40GB 단일 GPU)")
    print(f"run_mode={a.run_mode}  model={a.model}  num_frames={a.num_frames}")
    print(f"per_device={a.per_device_batch} accum={a.grad_accum} → 스텝당 프롬프트 {gen_batch // 8}개 × gen 8")
    print(f"out_dir={out_dir}")
    if a.run_mode == "full":
        print("⚠ full: 1 epoch ≈ 5000 프롬프트 = 이 구성에서 ~5000 옵티마이저 스텝.")
        print("  단일 A100 에서 Colab 세션 한도를 넘길 가능성이 크다 — 서버(2×H200) 실행 권장.")

    install_deps(a.no_install)
    train_jsonl = prepare_jsonl(a, work)

    if a.run_mode == "full":
        step_args = ["--num_train_epochs", "1.0", "--max_steps", "-1"]
    else:
        ms = a.max_steps if a.max_steps is not None else 500
        if ms % 125 != 0:
            print(f"⚠ max_steps={ms} 는 save_steps(125) 배수가 아님 → 최종 체크포인트 미생성 (handoff §11.6)")
        step_args = ["--max_steps", str(ms)]

    hr("[3] 학습 실행")
    cmd = [
        sys.executable, str(REPO / "src/ego/step2_vlm_alignment/train_grpo_action.py"),
        "--model_name", a.model,
        "--train_jsonl", str(train_jsonl),
        "--output_dir", str(out_dir),
        "--reward_mode", "wm_likelihood_joint",
        "--wm_likelihood_norm", "candidate",
        "--num_frames", str(a.num_frames),
        "--mask_frame_prob", str(a.mask_frame_prob),
        "--loss_type", "dr_grpo", "--scale_rewards", "none", "--epsilon_high", "0.28",
        "--min_wm_spread", "0.05", "--dynamic_sampling_std_threshold", "0",
        "--train_samples", str(a.train_samples), *step_args,
        "--num_generations", "8",
        "--per_device_train_batch_size", str(a.per_device_batch),
        "--gradient_accumulation_steps", str(a.grad_accum),
        "--hide_scores", "--shuffle_candidates", "--beta", "0.0", "--temperature", "1.0",
        "--max_completion_length", "384", "--learning_rate", "1e-5",
        "--lora_r", "16", "--lora_alpha", "32",
        "--save_steps", "125", "--logging_steps", "2", "--completion_log_every", "25",
        "--attn_impl", "sdpa", "--max_pixels", str(a.max_pixels),
    ]
    if a.adapter_path:
        cmd += ["--adapter_path", a.adapter_path]
        print(f"[resume] adapter 이어받기: {a.adapter_path} (옵티마이저 상태는 초기화)")

    env = dict(os.environ)
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    t0 = time.time()
    r = sh(cmd, cwd=str(REPO), env=env)
    dt = (time.time() - t0) / 3600

    hr("결과")
    print(f"종료 code={r.returncode}  경과={dt:.2f}h")
    ckpts = sorted(out_dir.glob("checkpoint-*"))
    print(f"체크포인트: {[c.name for c in ckpts] or '없음'}")
    print(f"로그: {out_dir}/reward_log.jsonl · completion_samples.jsonl · reasoning_proxy.jsonl")
    print("다음 단계: evaluate.py 로 held-out 3중 비교 (--reward_mode wm_likelihood_joint 일치 필수)")
    sys.exit(r.returncode)


if __name__ == "__main__":
    main()
