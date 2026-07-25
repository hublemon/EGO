"""WiSE-FT / model-soup for LoRA adapters — θ(α) = (1−α)·θ_A + α·θ_B (조합 §2④).

학습 0. 두 LoRA 어댑터(같은 base·target_modules)의 **정확한 weight-space 보간**.
LoRA delta 는 scaling·B@A 라 텐서 선형보간은 α에 비선형 → rank-2r concat 로 정확 구현:
  lora_A_C = concat([A_A ; A_B], dim0)                 # (2r, in)
  lora_B_C = concat([(1−α)·s_A/s_C·B_A, α·s_B/s_C·B_B], dim1)   # (out, 2r)
  ⇒ s_C·(B_C@A_C) = (1−α)·s_A·B_A@A_A + α·s_B·B_B@A_B = (1−α)ΔA + αΔB
(s_C 는 새 어댑터 scaling = alpha_C/r_C. 여기선 alpha_C=r_C 로 s_C=1.)

두 어댑터에 공통인 LoRA 키만 보간하고, 한쪽에만 있는 키는 해당 계수로 스케일해 rank 유지.
비-LoRA 키(있으면)는 선형보간.

사용:
  python3 tools/merge_adapters.py --adapter_a <θ_CE/adapter> --adapter_b <θ_CE+SFT/adapter> \
      --alpha 0.5 --out <outdir/adapter>
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def _sd(adapter: Path) -> dict:
    f = adapter / "adapter_model.safetensors"
    if f.is_file():
        return load_file(str(f))
    import glob
    bins = glob.glob(str(adapter / "adapter_model.bin"))
    if bins:
        return torch.load(bins[0], map_location="cpu")
    raise FileNotFoundError(f"no adapter weights in {adapter}")


def _scaling(cfg: dict) -> float:
    r = cfg.get("r", 16)
    alpha = cfg.get("lora_alpha", 2 * r)
    if cfg.get("use_rslora"):
        return alpha / (r ** 0.5)
    return alpha / r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter_a", required=True, help="θ_A (예: θ_CE) — α=0 극점")
    ap.add_argument("--adapter_b", required=True, help="θ_B (예: θ_CE+SFT) — α=1 극점")
    ap.add_argument("--alpha", type=float, required=True, help="0=A, 1=B, 사이=보간")
    ap.add_argument("--out", required=True, help="출력 adapter 디렉토리")
    args = ap.parse_args()

    a, b = Path(args.adapter_a), Path(args.adapter_b)
    cfg_a = json.loads((a / "adapter_config.json").read_text())
    cfg_b = json.loads((b / "adapter_config.json").read_text())
    s_a, s_b = _scaling(cfg_a), _scaling(cfg_b)
    r_a, r_b = cfg_a.get("r", 16), cfg_b.get("r", 16)
    al = args.alpha

    sd_a, sd_b = _sd(a), _sd(b)
    out_sd: dict[str, torch.Tensor] = {}
    r_c = r_a + r_b
    s_c = 1.0  # alpha_C = r_C 로 설정 → scaling 1

    keys = set(sd_a) | set(sd_b)
    lora_a_keys = {k for k in keys if "lora_A" in k}
    for ka in sorted(lora_a_keys):
        kb = ka  # 동일 모듈 이름 (같은 target_modules)
        base = ka.replace("lora_A", "lora_B")
        A_a = sd_a.get(ka)
        A_b = sd_b.get(kb)
        B_a = sd_a.get(base)
        B_b = sd_b.get(base)
        rows_A, rows_B, cols_B = [], [], []
        if A_a is not None and B_a is not None:
            rows_A.append(A_a.float())
            cols_B.append(((1 - al) * s_a / s_c) * B_a.float())
        if A_b is not None and B_b is not None:
            rows_A.append(A_b.float())
            cols_B.append((al * s_b / s_c) * B_b.float())
        if not rows_A:
            continue
        out_sd[ka] = torch.cat(rows_A, dim=0).to(A_a.dtype if A_a is not None else A_b.dtype)
        out_sd[base] = torch.cat(cols_B, dim=1).to(B_a.dtype if B_a is not None else B_b.dtype)

    # 비-LoRA 키 (드묾: bias 등) — 선형보간
    for k in sorted(keys):
        if "lora_A" in k or "lora_B" in k:
            continue
        ta, tb = sd_a.get(k), sd_b.get(k)
        if ta is not None and tb is not None and ta.shape == tb.shape:
            out_sd[k] = ((1 - al) * ta.float() + al * tb.float()).to(ta.dtype)
        elif tb is not None:
            out_sd[k] = tb
        elif ta is not None:
            out_sd[k] = ta

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    save_file(out_sd, str(out / "adapter_model.safetensors"))
    cfg_c = dict(cfg_a)
    cfg_c["r"] = r_c
    cfg_c["lora_alpha"] = r_c  # scaling 1
    cfg_c.pop("use_rslora", None)
    # rank_pattern/alpha_pattern 이 있으면 rank 변경과 충돌 → 제거
    cfg_c["rank_pattern"] = {}
    cfg_c["alpha_pattern"] = {}
    (out / "adapter_config.json").write_text(json.dumps(cfg_c, indent=2))
    for extra in ("README.md",):
        if (a / extra).is_file():
            shutil.copy(a / extra, out / extra)
    print(f"[merge] α={al} → {out}  (rank {r_a}+{r_b}={r_c}, keys={len(out_sd)})")


if __name__ == "__main__":
    main()
