# Step-2 (cesft_v2) 학습 모델 사용 Handoff — EGO_jihun2 사용자용

> 작성: 2026-07-25 KST. 대상: EGO_jihun3 step2_retrospection(cesft_v2)에서 학습된 모델을
> EGO_jihun2 쪽에서 이어서 사용하려는 사람.
> 방법론 SSOT: `EGO_jihun3/docs/experiments/2026-07-25_cesft_v2_paper_methodology_final_handoff.md`

---

## 0. 요약 — 산출물이 무엇인가

Step-2의 학습 결과물은 **단일 .pt 체크포인트가 아니라 LoRA 어댑터**다. 추론 시스템은 2단 구조:

1. **WM prior** (Step-1, 읽기전용): action anticipation 체크포인트가 샘플별 **Top-10 후보**를
   오프라인으로 미리 생성 → jsonl 데이터에 저장돼 있음. **추론 시 이 .pt는 로드하지 않는다.**
2. **VLM selector** (Step-2, 이것이 학습된 모델): `Qwen/Qwen3-VL-8B-Instruct` +
   **LoRA 어댑터 `sft_r15`** (CE → SFT + CE-replay ρ=0.15 체인의 최종 산출).
   8초 관측창 8프레임 + 완료행동 이력 + Top-10 후보를 받아 `<reasoning>/<task_belief>/<action>`
   trace를 생성하고 후보 중 하나를 선택한다.

## 1. 파일 위치 (모두 절대경로)

| 구성요소 | 경로 |
|---|---|
| **LoRA 어댑터 (핵심 산출물)** | `/mnt/nvme/migration/jihun/EGO_jihun3/outputs/step2_retrospection/cesft_v2/sft_r15/adapter/` |
| 베이스 VLM | HuggingFace `Qwen/Qwen3-VL-8B-Instruct` (로컬 캐시 자동 사용) |
| WM prior 체크포인트 (원본, jihun2) | `/mnt/nvme/migration/jihun/EGO_jihun2/outputs/goalstep/runs/z1_history_context_k8_vna_ep10/best_action_top5.pt` |
| WM prior export본 (jihun3, sha256 동일) | `/mnt/nvme/migration/jihun/EGO_jihun3/outputs/goalstep/exports/RETRO4-goalstep-end-m1-history-k8-phase1/best_action_top5.pt` |
| Top-10 후보 포함 평가 데이터 | `/mnt/nvme/migration/jihun/EGO_jihun3/runs/cesft_v2/data/context_val.jsonl` (train: `context_train.jsonl`) |
| 프레임 캐시 (선택, 디코딩 스킵) | `/mnt/nvme/migration/jihun/EGO_jihun3/runs/cesft_v2/frame_cache/` |
| 학습 로그 | `.../cesft_v2/sft_r15/train_log.jsonl` (435 steps, final loss_ema 0.6001) |

- 어댑터 사양: LoRA r=16, α=32, target `q/k/v/o_proj`, peft 0.19.1 (`adapter_config.json` 참조).
- WM prior 무결성: sha256 `1d20c942e7ec71ef76326679c44d20dead88e084dc8ac4367b7e3ebebdaa61cd`
  (`EGO_jihun3/configs/step2_retrospection/cesft_v2.yaml:16`에 고정).

## 2. 결합 방법 — 최소 코드

```python
# 환경: /mnt/nvme/migration/jihun/envs/miniforge3/envs/eve-cu124/bin/python
#       (transformers 5.9.0, peft 0.19.1 확인됨)
# PYTHONPATH에 EGO_jihun3/src 추가 필요 (vlm 유틸 재사용 시)
from ego.step2_retrospection import vlm
from peft import PeftModel

ADAPTER = "/mnt/nvme/migration/jihun/EGO_jihun3/outputs/step2_retrospection/cesft_v2/sft_r15/adapter"

model, processor = vlm.load_model()            # Qwen3-VL-8B-Instruct, bf16, cuda
model = PeftModel.from_pretrained(model, ADAPTER)
model.eval()

# rec = context_val.jsonl 한 줄 (candidates, history, video_uid, obs 창 등 포함)
# imgs = 관측창 8프레임 (PIL, 336px short side) — vlm.prefetch_chunks 또는 frame_cache
text   = vlm.generate_batch(model, processor, [vlm.build_messages(rec, imgs)])[0]
parsed = vlm.parse_trace(text)                              # reasoning/task_belief/action
action = vlm.match_candidate(parsed["action"], rec["candidates"])  # 최종 선택
```

peft 의존성 없이 쓰고 싶으면 merge 후 standalone으로 저장 가능:

```python
merged = model.merge_and_unload()
merged.save_pretrained("/원하는/경로/qwen3vl_sft_r15_merged")   # ~16GB, processor도 save_pretrained
```

## 3. 프롬프트/시간 계약 — 반드시 학습 때와 동일하게

- 환경변수 **`RETRO_NEXT_GAP_TEXT="after the current action ends"`** 를 vlm import 전에 설정.
  (retro4/cesft_v2는 end−1s 가변 horizon 계약. 미설정 시 기본 "starting 1 second after the
  last frame"(retro3 계약)로 프롬프트가 달라져 성능이 왜곡됨. `EGO_jihun3/src/ego/step2_retrospection/vlm.py:51`)
- 관측창: 마지막 8초에서 1fps 8프레임, short side 336px.
- 출력 파싱: `<reasoning>…</reasoning><task_belief>…</task_belief><action>…</action>` 3태그.
  `vlm.parse_trace` 실패(None) = malformed 처리.
- 후보 매칭: exact → 소문자·공백 정규화 → 유일 부분일치 (`vlm.match_candidate`).

## 4. 기존 heldout에서 바로 돌려보기 (검증용)

```bash
cd /mnt/nvme/migration/jihun/EGO_jihun3
RETRO3_RUNS=runs/cesft_v2 \
RETRO_NEXT_GAP_TEXT="after the current action ends" \
PYTHONPATH=src /mnt/nvme/migration/jihun/envs/miniforge3/envs/eve-cu124/bin/python \
  -m ego.step2_retrospection.eval.battery \
  --config configs/step2_retrospection/cesft_v2.yaml \
  --arm my_test \
  --adapter outputs/step2_retrospection/cesft_v2/sft_r15/adapter \
  --eval_n 100
# 결과: runs/cesft_v2/eval/my_test.json + my_test.records.jsonl
```

- `RETRO3_RUNS=runs/cesft_v2` 필수 (기본값 runs/retro3라 데이터를 못 찾음).
- `--arm` 이름이 같으면 기존 records에서 이어받기(resume) — 새 실험은 새 arm 이름 사용.
- 기존 결과 비교 기준: `runs/cesft_v2/eval/sft_r15.json` (같은 프로토콜로 이미 평가돼 있음).

## 5. 새 비디오/새 구간에 쓰려면 — WM prior가 다시 필요

context jsonl에 없는 샘플은 후보가 없으므로, Step-1 WM prior로 Top-10을 먼저 생성해야 한다:

1. `EGO_jihun3/src/ego/step2_retrospection/data/build_support_hk8.py` 가
   `best_action_top5.pt` (§1의 export본)를 로드해 샘플별 Top-K support + `wm_scores`를 덤프.
   history-context K8 요약(`derived_store_dir`, cesft_v2.yaml:23)에 의존.
2. `data/build_context.py` 로 VLM 입력용 context jsonl로 변환 (후보 셔플 포함).
3. 이후 §2 코드 그대로.

EGO_jihun2 쪽 자체 파이프라인에 붙일 경우에도 **입력 계약(§3)과 후보 생성 방식(Top-10,
셔플, WM rank 비공개)** 을 유지해야 학습 분포와 일치한다.

## 6. 성능 특성 — 사용 전 알아야 할 것

(상세: `2026-07-25_cesft_v2_quantitative_evidence_handoff.md`)

- **sft_r15의 목적은 정확도 향상이 아니라 검증가능한 추론 trace**다. 선택 정확도(SelAcc)는
  θ_CE 대비 +1.1pp로 통계적 비유의(G-NH FAIL). 정확도만 필요하면 θ_CE 어댑터
  (`.../cesft_v2/theta_ce/adapter`)로 충분하고, reasoning/task_belief 품질이 필요할 때 sft_r15를 쓴다.
- 현재 수치는 N=2,500 파일럿 기준. full 학습으로 갱신 예정이었음.
- GT가 Top-10 밖인 샘플(cov@10 ≈ 43%)은 구조적으로 0점 — covered-only 평가가 표준
  (`runs/cesft_v2/overrides.json` 참조).

## 7. 체크리스트 (다음 사람용)

- [ ] eve-cu124 env 사용 (transformers 5.9.0 / peft 0.19.1)
- [ ] `RETRO_NEXT_GAP_TEXT` 설정 후 vlm import
- [ ] 어댑터 경로로 `PeftModel.from_pretrained` (또는 merge본 사용)
- [ ] §4 커맨드로 heldout 100개 스모크 → `sft_r15.json`과 SelAcc 대략 일치 확인
- [ ] 새 데이터면 §5로 Top-10 후보 먼저 생성
