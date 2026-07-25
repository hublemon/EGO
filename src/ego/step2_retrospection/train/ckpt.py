"""학습 중간 체크포인트 · resume (full 스케일 내구성).

왜 필요한가: full 학습은 θ_CE ≈ 19h · SFT ≈ 13h 다. 종전 트레이너는 끝날 때 한 번만
`save_pretrained` 해서 OOM-kill·스톨·컨테이너 재시작 한 번이면 전부 잃었다
(2026-07-24 사고: sft_r0 스톨로 7h 손실). opt.step 주기로 LoRA+optimizer+scheduler+
스트림 위치를 원자적으로 저장하고, 재기동 시 그 지점부터 이어 돌린다.

저장 레이아웃 (out_dir = outputs/step2_retrospection/<run>/<run_name>):
    ckpt/adapter/            PEFT LoRA (수십 MB)
    ckpt/train_state.pt      optimizer·scheduler·rng state
    ckpt/train_state.json    사람이 읽는 진행 요약 (step/n_seen/stream_pos/ema)

원자성: ckpt.tmp 에 완성 → ckpt → ckpt.old 로 물러난 뒤 rename → ckpt.old 삭제.
저장 중 죽어도 직전 ckpt 는 온전하다.

resume 계약: 데이터 순서는 seed 고정 + video-group 정렬로 **결정적**이므로,
저장된 스트림 위치(select_ce=stream_pos / sft_v2=n_sft)만으로 정확히 이어붙는다.
accum 버퍼는 이어받지 않는다 (최대 accum−1 샘플의 gradient 손실 = 무시 가능).
"""
from __future__ import annotations

import json
import random
import shutil
import time
from pathlib import Path

import torch


def _rm(p: Path) -> None:
    if p.is_dir():
        shutil.rmtree(p, ignore_errors=True)
    elif p.exists():
        p.unlink()


def save_ckpt(out_dir: Path, model, opt, sched, meta: dict) -> None:
    """원자적 체크포인트 교체. 실패해도 학습을 죽이지 않는다(경고만)."""
    out_dir = Path(out_dir)
    ck, tmp, old = out_dir / "ckpt", out_dir / "ckpt.tmp", out_dir / "ckpt.old"
    try:
        _rm(tmp)
        tmp.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(tmp / "adapter")
        torch.save({"opt": opt.state_dict(), "sched": sched.state_dict(), **meta},
                   tmp / "train_state.pt")
        (tmp / "train_state.json").write_text(
            json.dumps({**{k: v for k, v in meta.items() if k != "rng"},
                        "saved_at": time.time()}, indent=1, ensure_ascii=False))
        _rm(old)
        if ck.exists():
            ck.rename(old)
        tmp.rename(ck)
        _rm(old)
    except Exception as e:  # 디스크/권한 문제로 학습이 죽으면 안 된다
        print(f"[ckpt] save 실패 (계속 진행): {e}", flush=True)


def load_ckpt(out_dir: Path, model, opt, sched) -> dict | None:
    """LoRA·optimizer·scheduler 를 복원하고 진행 메타를 돌려준다. 없으면 None."""
    ck = Path(out_dir) / "ckpt"
    st_path = ck / "train_state.pt"
    if not st_path.is_file():
        return None
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    sd = load_file(str(ck / "adapter" / "adapter_model.safetensors"))
    set_peft_model_state_dict(model, sd)
    st = torch.load(st_path, map_location="cpu", weights_only=False)
    opt.load_state_dict(st["opt"])      # 텐서는 파라미터 device 로 자동 이동
    sched.load_state_dict(st["sched"])
    print(f"[ckpt] resume from step={st.get('step')} n_seen={st.get('n_seen')} "
          f"stream_pos={st.get('stream_pos')}", flush=True)
    return st


def restore_rng(st: dict | None, rng: random.Random) -> None:
    if not st or "rng" not in st or st["rng"] is None:
        return
    try:
        rng.setstate(st["rng"])
    except Exception as e:
        print(f"[ckpt] rng 복원 실패 (무시): {e}", flush=True)


def clear_ckpt(out_dir: Path) -> None:
    """정상 완료 후 정리 — 다음 런이 낡은 상태를 물지 않게."""
    for name in ("ckpt", "ckpt.tmp", "ckpt.old"):
        _rm(Path(out_dir) / name)
