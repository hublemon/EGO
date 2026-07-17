#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""colab_smoke_f0.py — Step2 F0 (WM-only GRPO) GPU 스모크 러너 (Colab A100 40GB 대상).

한 파일로 다음을 자동 실행한다:
  1. GPU/환경 점검
  2. 의존성 설치 (transformers 5.9.0 / trl 1.5.1 / peft / accelerate / qwen-vl-utils)
  3. 학습 asset 준비
       - 기본: F0 v2 jsonl 스키마에 맞는 **합성 tiny 데이터**(회색 2x2 grid JPEG + train.jsonl)를 생성.
         reward(wm_likelihood_joint)는 jsonl 안의 precompute likelihood 만 읽으므로
         V-JEPA2 / EPIC-Kitchens 원본이 없어도 학습 루프 전체를 GPU 에서 검증할 수 있다.
       - 실제 데이터가 있으면 --train_jsonl 로 그 경로를 그대로 물린다 (합성 생략).
  4. 순수 로직 스모크(smoke_f0_v2.py) 실행 — 데이터/프롬프트 규칙 회귀 검사
  5. GRPO 학습 스모크 (축소 설정 2 step) — Qwen3-VL-8B + LoRA + wm_likelihood_joint reward
  6. 산출물 검증(checkpoint / reward_log.jsonl / completion_samples.jsonl) 후 PASS/FAIL 요약

이 러너가 검증하는 것: "F0 학습 코드가 이 GPU 에서 처음부터 끝까지 돈다".
검증하지 않는 것: 학습 품질/수렴/지표 — 그건 실제 데이터 + full run 의 몫.

Colab 사용법 (셀 하나):
    !cd /content/EGO && python scripts/step2/colab_smoke_f0.py

실데이터로:
    !cd /content/EGO && python scripts/step2/colab_smoke_f0.py \
        --train_jsonl /content/drive/MyDrive/ego/grpo_train.jsonl --num_frames 4

무거운 라이브러리(torch/transformers 등)는 설치 이후 함수 안에서 lazy import 한다
(같은 프로세스에서 pip install → import 가 되도록 top-level 임포트는 stdlib 만).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# F0 v2 고정 레시피 (configs/step2/f0_final_v2_wm_only.yaml 과 일치)
FRAME_OFFSETS_4 = [4.0, 2.67, 1.33, 0.0]
# EK100 스타일 소형 어휘 — 합성 후보 구성용 (품질과 무관, 스키마 충족이 목적)
VERBS = ["take", "put", "wash", "cut", "open", "close", "mix", "pour"]
NOUNS = ["knife", "tomato", "plate", "pan", "tap", "board", "onion", "bowl"]
# 후보 5개의 재정규화-전 likelihood. spread(std) > min_wm_spread(0.05) 를 만족하도록 고정.
# '0.4','0.25' 같은 라운드 값은 assert_no_score_leak 오탐 위험이 있어 일부러 비라운드로.
LIKS = [0.41, 0.26, 0.16, 0.11, 0.06]


# ─────────────────────────────────────────────────────────────────────────────
# 유틸
# ─────────────────────────────────────────────────────────────────────────────
def hr(title: str = "") -> None:
    print("\n" + "=" * 72)
    if title:
        print(title)
        print("=" * 72)


def sh(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    print(f"$ {' '.join(str(c) for c in cmd)}")
    return subprocess.run(cmd, **kw)


# ─────────────────────────────────────────────────────────────────────────────
# 1. GPU / 환경
# ─────────────────────────────────────────────────────────────────────────────
def gpu_report() -> None:
    hr("[1] GPU / 환경")
    try:
        import torch  # Colab 기본 제공
        print(f"torch={torch.__version__}  cuda_available={torch.cuda.is_available()}")
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            print(f"device={p.name}  vram={p.total_memory / 1e9:.1f} GB")
        else:
            print("⚠ CUDA 미탐지 — 학습 스모크는 GPU 런타임에서만 유효하다.")
    except Exception as e:  # noqa: BLE001
        print(f"torch import 실패(설치 전 정상): {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. 의존성
# ─────────────────────────────────────────────────────────────────────────────
def install_deps(skip: bool, wm: str) -> None:
    hr("[2] 의존성 설치")
    if skip:
        print("--no_install → 설치 건너뜀 (이미 설치된 환경으로 가정)")
        return
    # torch 는 Colab 기본 제공분을 재사용(건드리면 커널 재시작 필요) — 아래만 설치.
    pkgs = [
        "transformers==5.9.0",
        "trl==1.5.1",
        "peft>=0.14",
        "accelerate>=1.0",
        "datasets>=3.0",
        "qwen-vl-utils",
        "pillow",
        "tensorboard",
    ]
    if wm == "real":
        # V-JEPA2 백본/probe 로드·forward 에 필요한 공식 repo 의존
        # (webdataset: epickitchens.py 가 module-level import — 실측 누락 보고 반영)
        pkgs += ["einops", "timm", "decord", "pyyaml", "webdataset"]
    r = sh([sys.executable, "-m", "pip", "install", "-q", *pkgs])
    if r.returncode != 0:
        print("⚠ pip install 실패 — 버전 충돌 시 --no_install 로 수동 환경에서 재시도")
        sys.exit(2)
    if wm == "real":
        # vit_giant_xformers 인코더가 xformers 를 요구할 수 있다 (best-effort — 실패 시 sdpa 폴백 시도).
        rx = sh([sys.executable, "-m", "pip", "install", "-q", "xformers"])
        if rx.returncode != 0:
            print("⚠ xformers 설치 실패 — 인코더 import 시 xformers 필요하면 수동 설치 요망")
    print("설치 완료.")


# ─────────────────────────────────────────────────────────────────────────────
# 2b. V-JEPA2 asset (공식 repo + 체크포인트 + 어노테이션) 준비
# ─────────────────────────────────────────────────────────────────────────────
VJEPA_REPO_URL = "https://github.com/facebookresearch/vjepa2"
BACKBONE_URL = "https://dl.fbaipublicfiles.com/vjepa2/vitg-384.pt"
PROBE_URL = "https://dl.fbaipublicfiles.com/vjepa2/evals/ek100-vitg-384.pt"
ANN_BASE = "https://raw.githubusercontent.com/epic-kitchens/epic-kitchens-100-annotations/master"
ANN_FILES = ["EPIC_100_train.csv", "EPIC_100_validation.csv",
             "EPIC_100_verb_classes.csv", "EPIC_100_noun_classes.csv"]


def _valid_pt(path: Path) -> bool:
    """torch .pt(=zip) 무결성 검사. 잘린 다운로드는 central directory 가 없어 열기 실패."""
    if path.suffix != ".pt":
        return True
    import zipfile
    try:
        zipfile.ZipFile(path).close()
        return True
    except Exception:
        return False


def _download(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        if _valid_pt(dest):
            print(f"  cached: {dest.name} ({dest.stat().st_size / 1e6:.0f} MB)")
            return
        # 잘린 캐시 (Drive 쓰기 중단 등) — wget -c 로 이어받기 시도
        print(f"  ⚠ corrupt cache: {dest.name} ({dest.stat().st_size / 1e6:.0f} MB) — 이어받기/재다운로드")
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  download: {url}")
    # -c: 부분 파일 이어받기 (Drive 마운트로 수 GB 쓰다 끊긴 경우 처음부터 다시 안 받음)
    r = sh(["wget", "-c", "-q", "--show-progress", "-O", str(dest), url])
    if r.returncode != 0:
        sh(["curl", "-fL", "-C", "-", "-o", str(dest), url])
    if not (dest.exists() and dest.stat().st_size > 0 and _valid_pt(dest)):
        print(f"✗ 다운로드 무결성 실패: {dest}")
        print("  Drive 용량 부족 가능성 — 확인 후 재시도하거나 --assets_dir 를 /content 로컬로 지정")
        sys.exit(2)


def ensure_vjepa_assets(assets_dir: Path) -> dict:
    hr("[2b] V-JEPA2 asset 준비 (공식 repo + 체크포인트 + 어노테이션)")
    assets_dir.mkdir(parents=True, exist_ok=True)
    repo = assets_dir / "vjepa2"
    if not (repo / "evals").exists():
        sh(["git", "clone", "--depth", "1", VJEPA_REPO_URL, str(repo)])
    else:
        print(f"  cached repo: {repo}")
    backbone = assets_dir / "vitg-384.pt"
    probe = assets_dir / "ek100-vitg-384.pt"
    _download(BACKBONE_URL, backbone)
    _download(PROBE_URL, probe)
    ann_dir = assets_dir / "ek100_annotations"
    for name in ANN_FILES:
        _download(f"{ANN_BASE}/{name}", ann_dir / name)
    return {"repo": repo, "backbone": backbone, "probe": probe, "ann_dir": ann_dir}


def run_vjepa_infer(assets: dict, data_dir: Path, n: int, num_frames: int) -> Path:
    hr("[3] V-JEPA2 실추론 → 후보 jsonl (합성 클립)")
    helper = REPO / "scripts/step2/vjepa_ek100_smoke.py"
    out_jsonl = data_dir / "grpo_train_vjepa.jsonl"
    cmd = [
        sys.executable, str(helper),
        "--vjepa_repo", str(assets["repo"]),
        "--backbone_ckpt", str(assets["backbone"]),
        "--probe_ckpt", str(assets["probe"]),
        "--ann_dir", str(assets["ann_dir"]),
        "--out_jsonl", str(out_jsonl),
        "--frames_dir", str(data_dir / "frames"),
        "--n_samples", str(n), "--num_frames", str(num_frames),
    ]
    r = sh(cmd, cwd=str(REPO))
    if r.returncode != 0 or not out_jsonl.exists():
        print("✗ V-JEPA2 추론 실패 — 위 로그 확인")
        sys.exit(2)
    return out_jsonl


# ─────────────────────────────────────────────────────────────────────────────
# 3. asset (합성 데이터) 준비
# ─────────────────────────────────────────────────────────────────────────────
def _grid_jpeg(path: Path, num_frames: int, idx: int) -> None:
    """회색 톤 placeholder 이미지. num_frames==4 면 2x2 grid(사분면 명도 차)."""
    from PIL import Image, ImageDraw

    side = 448
    img = Image.new("RGB", (side, side), (90, 90, 90))
    d = ImageDraw.Draw(img)
    if num_frames == 4:
        shades = [(70, 70, 70), (100, 100, 100), (130, 130, 130), (160, 160, 160)]
        for q, (x0, y0) in enumerate([(0, 0), (side // 2, 0), (0, side // 2), (side // 2, side // 2)]):
            d.rectangle([x0, y0, x0 + side // 2, y0 + side // 2], fill=shades[(q + idx) % 4])
            d.text((x0 + 8, y0 + 8), f"f{q + 1}", fill=(230, 230, 230))
        d.line([side // 2, 0, side // 2, side], fill=(30, 30, 30), width=2)
        d.line([0, side // 2, side, side // 2], fill=(30, 30, 30), width=2)
    else:
        d.text((16, 16), f"frame {idx}", fill=(230, 230, 230))
    img.save(path, "JPEG", quality=85)


def _synth_record(i: int, img_path: Path, num_frames: int) -> dict:
    # 후보 5개 구성 (index 로 회전시켜 샘플마다 다르게)
    acts, acts_ws, nouns, nouns_ws = [], [], [], []
    for r in range(5):
        v = VERBS[(i + r) % len(VERBS)]
        n = NOUNS[(i + 2 * r) % len(NOUNS)]
        lik = LIKS[r]
        acts.append({"verb": v, "noun": n, "action": f"{v} {n}", "score": lik})
        acts_ws.append({"verb": v, "noun": n, "likelihood": lik, "rank": r + 1})
        nouns.append({"noun": n, "score": lik})
        nouns_ws.append({"noun": n, "likelihood": lik, "rank": r + 1})
    gt = acts[i % 5]  # GT 를 후보 안에 두어 로깅 지표가 의미를 갖게 함 (학습엔 미사용)

    if num_frames == 4:
        mem = ("Frame 1 (4.0s ago): take knife\nFrame 2 (2.67s ago): no completed action\n"
               "Frame 3 (1.33s ago): wash tomato\nFrame 4 (0.0s ago): no completed action")
        fmeta = {"n_frames": 4, "offsets_sec": FRAME_OFFSETS_4}
    else:
        mem = "Previously completed actions: take knife -> wash tomato."
        fmeta = {"n_frames": 1, "offsets_sec": [0.0]}

    return {
        "sample_id": f"smoke{i:03d}",
        "frame_id": f"smoke{i:03d}",
        "episode_id": f"V{i % 3:02d}",
        "image_path": str(img_path),
        "gt_verb": gt["verb"],
        "gt_noun": gt["noun"],
        "gt_label": {"verb": gt["verb"], "noun": gt["noun"]},
        "memory_context": mem,
        "frame_meta": fmeta,
        "topk_actions": acts,
        "topk_actions_with_score": acts_ws,
        "topk_nouns": nouns,
        "topk_nouns_with_score": nouns_ws,
    }


def synth_data(data_dir: Path, n: int, num_frames: int) -> Path:
    hr("[3] 합성 asset 생성")
    frames = data_dir / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    recs = []
    for i in range(n):
        p = frames / f"grid_{i:03d}.jpg"
        _grid_jpeg(p, num_frames, i)
        recs.append(_synth_record(i, p, num_frames))
    jsonl = data_dir / "grpo_train_smoke.jsonl"
    with open(jsonl, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"생성: {n} 레코드 + {n} grid JPEG → {jsonl}")
    print(f"       각 레코드 필드: {sorted(recs[0].keys())}")
    return jsonl


# ─────────────────────────────────────────────────────────────────────────────
# 4. 순수 로직 스모크
# ─────────────────────────────────────────────────────────────────────────────
def logic_smoke() -> bool:
    hr("[4] 순수 로직 스모크 (smoke_f0_v2.py)")
    script = REPO / "scripts/step2/smoke_f0_v2.py"
    if not script.exists():
        print(f"⚠ 없음: {script} — 건너뜀")
        return True
    r = sh([sys.executable, str(script)])
    ok = r.returncode == 0
    print("결과:", "PASS" if ok else "FAIL")
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# 5. GRPO 학습 스모크
# ─────────────────────────────────────────────────────────────────────────────
def train_smoke(a, train_jsonl: Path, out_dir: Path, min_wm_spread: str = "0.05") -> bool:
    hr("[5] GRPO 학습 스모크 (축소 설정)")
    train_py = REPO / "src/ego/step2_vlm_alignment/train_grpo_action.py"
    out_dir.mkdir(parents=True, exist_ok=True)
    # save_steps == max_steps 여야 최종 체크포인트가 생성된다 (handoff §11.6).
    cmd = [
        sys.executable, str(train_py),
        "--train_jsonl", str(train_jsonl),
        "--output_dir", str(out_dir),
        "--model_name", a.model,
        "--reward_mode", "wm_likelihood_joint",
        "--wm_likelihood_norm", "candidate",
        "--num_frames", str(a.num_frames),
        "--mask_frame_prob", "0.0",
        "--loss_type", "dr_grpo", "--scale_rewards", "none", "--epsilon_high", "0.28",
        "--min_wm_spread", min_wm_spread, "--dynamic_sampling_std_threshold", "0",
        "--train_samples", str(a.n_samples),
        "--max_steps", str(a.max_steps),
        "--num_generations", str(a.num_generations),
        "--per_device_train_batch_size", str(a.per_device_batch),
        "--gradient_accumulation_steps", "1",
        "--hide_scores", "--shuffle_candidates",
        "--beta", "0.0", "--temperature", "1.0",
        "--max_completion_length", str(a.max_completion_length),
        "--learning_rate", "1e-5",
        "--lora_r", "16", "--lora_alpha", "32",
        "--save_steps", str(a.max_steps), "--logging_steps", "1",
        "--completion_log_every", "1",
        "--attn_impl", "sdpa",
        "--max_pixels", str(512 * 28 * 28), "--min_pixels", str(128 * 28 * 28),
    ]
    env = dict(os.environ)
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    t0 = time.time()
    r = sh(cmd, cwd=str(REPO), env=env)
    dt = time.time() - t0
    print(f"학습 프로세스 종료 code={r.returncode}  경과={dt:.0f}s")
    return r.returncode == 0


# ─────────────────────────────────────────────────────────────────────────────
# 6. 산출물 검증
# ─────────────────────────────────────────────────────────────────────────────
def verify_outputs(out_dir: Path, max_steps: int) -> bool:
    hr("[6] 산출물 검증")
    checks = []
    ckpt = out_dir / f"checkpoint-{max_steps}"
    checks.append(("checkpoint dir", ckpt.exists()))
    rl = out_dir / "reward_log.jsonl"
    checks.append(("reward_log.jsonl 비어있지 않음", rl.exists() and rl.stat().st_size > 0))
    cs = out_dir / "completion_samples.jsonl"
    checks.append(("completion_samples.jsonl 존재", cs.exists()))
    ok = True
    for name, cond in checks:
        print(f"  [{'OK  ' if cond else 'FAIL'}] {name}")
        ok = ok and cond
    # reward_log 마지막 줄 요약
    if rl.exists() and rl.stat().st_size > 0:
        try:
            last = rl.read_text(encoding="utf-8").strip().splitlines()[-1]
            d = json.loads(last)
            keys = [k for k in ("step", "reward", "loss", "grad_norm") if k in d]
            print("  reward_log 마지막:", {k: d[k] for k in keys})
        except Exception as e:  # noqa: BLE001
            print(f"  (reward_log 파싱 스킵: {e})")
    return ok


# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    p = argparse.ArgumentParser(description="Step2 F0 GPU smoke runner (Colab A100)")
    p.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct",
                   help="A100 40GB 부족 시 Qwen/Qwen2.5-VL-7B-Instruct 로 대체 가능")
    p.add_argument("--wm", choices=["synth", "real"], default="synth",
                   help="synth=합성 jsonl(다운로드 없음) | real=공식 V-JEPA2 백본+EK100 probe 실추론")
    p.add_argument("--assets_dir", default=None,
                   help="V-JEPA2 체크포인트/repo 캐시 위치 (--wm real). 재다운로드 방지하려면 Drive 경로 권장")
    p.add_argument("--train_jsonl", default=None,
                   help="실데이터 jsonl 경로. 지정 시 합성/추론 생략하고 이 파일로 학습")
    p.add_argument("--num_frames", type=int, default=4, choices=[1, 4])
    p.add_argument("--n_samples", type=int, default=8)
    p.add_argument("--max_steps", type=int, default=2)
    p.add_argument("--num_generations", type=int, default=2)
    p.add_argument("--per_device_batch", type=int, default=2)
    p.add_argument("--max_completion_length", type=int, default=128)
    p.add_argument("--data_dir", default=None, help="합성 데이터 출력 위치 (기본: 스크래치)")
    p.add_argument("--out_dir", default=None, help="학습 출력 위치 (기본: 스크래치)")
    p.add_argument("--no_install", action="store_true", help="pip 설치 건너뜀")
    p.add_argument("--skip_train", action="store_true", help="로직 스모크까지만, GPU 학습 생략")
    a = p.parse_args()

    scratch = Path(os.environ.get("EGO_SMOKE_DIR", "/tmp/ego_f0_smoke"))
    data_dir = Path(a.data_dir) if a.data_dir else scratch / "data"
    out_dir = Path(a.out_dir) if a.out_dir else scratch / "out"
    assets_dir = Path(a.assets_dir) if a.assets_dir else scratch / "assets"

    hr("Step2 F0 GPU 스모크 러너")
    print(f"repo={REPO}\nmodel={a.model}  wm={a.wm}  num_frames={a.num_frames}")
    print(f"steps={a.max_steps} gen={a.num_generations} batch={a.per_device_batch} "
          f"compl={a.max_completion_length} n_samples={a.n_samples}")
    print(f"scratch={scratch}")

    gpu_report()
    install_deps(a.no_install, a.wm)

    if a.train_jsonl:
        train_jsonl = Path(a.train_jsonl)
        if not train_jsonl.exists():
            print(f"✗ --train_jsonl 없음: {train_jsonl}")
            sys.exit(2)
        print(f"\n[3] 실데이터 사용: {train_jsonl} (합성/추론 생략)")
    elif a.wm == "real":
        assets = ensure_vjepa_assets(assets_dir)
        train_jsonl = run_vjepa_infer(assets, data_dir, a.n_samples, a.num_frames)
    else:
        train_jsonl = synth_data(data_dir, a.n_samples, a.num_frames)

    results = {"logic": logic_smoke()}

    if a.skip_train:
        print("\n--skip_train → GPU 학습 스모크 생략")
    else:
        # 실추론 모드(합성 클립): 3천+ 클래스에 대한 probe softmax 가 거의 균등 → 후보 재정규화 후
        # 분포가 flat 이라 min_wm_spread=0.05 가 전 샘플을 걸러냄 (실측: 0/8 kept — 필터 동작 자체는
        # 이것으로 검증됨). 스모크의 목적은 학습 루프 관통이므로 이 경우만 필터를 끈다.
        spread = "0.0" if (a.wm == "real" and not a.train_jsonl) else "0.05"
        if spread == "0.0":
            print("[INFO] real 모드 합성 클립 → WM 분포 flat → min_wm_spread=0 (스모크 한정)")
        train_ok = train_smoke(a, train_jsonl, out_dir, min_wm_spread=spread)
        results["train"] = train_ok
        results["verify"] = verify_outputs(out_dir, a.max_steps) if train_ok else False

    hr("요약")
    for k, v in results.items():
        print(f"  {k:8s}: {'PASS' if v else 'FAIL'}")
    all_ok = all(results.values())
    print("\n최종:", "✅ ALL PASS" if all_ok else "❌ FAIL 있음")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
