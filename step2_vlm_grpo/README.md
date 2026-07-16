# EGO — Step 2: VLM GRPO (World-Model-only, GT-free)

> 월드 모델(V-JEPA2)이 준 후보만 보고, **정답 라벨(GT) 없이** 월드 모델의 예측 분포에 정합하도록
> VLM(Qwen2.5-VL-7B)을 GRPO 강화학습으로 파인튜닝하는 코드베이스.
> **목표: held-out에서 월드 모델 자신의 top-1을 능가하는 VLM.**

이 브랜치는 진행 중인 실험의 스냅샷입니다. 아래만 읽으면 *무엇을·왜·어떻게* 하는지, 그리고 *지금 어디까지 왔는지* 파악할 수 있습니다.

---

## 1. 한 문장 논지

기존 next-action 예측 연구의 흔한 함정은 **모델이 스스로 생성하고 스스로 채점하는 순환 구조**(self-certainty, 다수결 등)입니다. EGO는 이걸 깨기 위해 **외부 신호원**을 씁니다:

- **월드 모델(V-JEPA2 probe)** 이 egocentric 프레임을 보고 다음 행동 후보 top-5와 각 후보의 likelihood(softmax 확률)를 냅니다.
- **VLM**은 그 후보들 사이에서 하나를 고르고, 그 선택이 **월드 모델의 likelihood 분포에 얼마나 정합하는지**로만 보상을 받습니다.
- 학습 신호에 **GT(사람이 단 정답 라벨)는 전혀 들어가지 않습니다.** GT는 오직 *사후 held-out 평가*에만 등장합니다.

성공하면 "human label 없이, 검증 가능한(verifiable) 보상만으로 학습된다"는 논문 핵심 클레임이 성립합니다.

---

## 2. 왜 이게 어려운가 (그리고 이 코드가 푸는 것)

한 번 시도했다가 **실패한 이력**이 있습니다(Exp.10, `docs/GRPO_TRAINING_LOG.md`):

- GRPO는 **그룹 내 답안들의 점수 *차이*** 로 배웁니다. 한 프롬프트에 답안 8개를 생성해 서로 비교하죠.
- 월드 모델 likelihood 분포가 평평한(flat) 샘플에서는 어떤 후보를 골라도 보상이 비슷 → 점수 차이 0 → **gradient 0**. 이런 샘플이 배치를 채우면 학습 신호 자체가 생산되지 않습니다. 이게 Exp.10의 "reward≈0 수렴"입니다.

이 코드는 그 실패를 구조적으로 막는 처방들의 묶음입니다 — 자세한 건 §4 Reward/최적화 설계.

---

## 3. 저장소 구조

```
train_qwen25vl_grpo_ek100.py   # ★ 핵심: GRPO 학습 스크립트 (reward 함수·dynamic sampling·로깅 전부 여기)
run_grpo_run1_wmonly.sh        # Run 1 (WM-only 성립 검증) 실행
run_grpo_run2_branch{A,B,C}.sh # Run 2 분기별 실행 (Run 1 진단 결과에 따라 택1)
run_grpo_final.sh              # Exp.14 (GT-primary, 과거 최고 성적 — 이제 "GT-oracle 상한 참조선")

eval_heldout.py                # held-out 평가: GT 정확도 / G2 구간 / 후보 이탈률 / WM-follow
eval_checkpoints_run1.sh       # 전 체크포인트 일괄 평가 + G1/G2 곡선 요약
eval_reasoning_trace.py        # reasoning 인과성 검증 (lift / 반사실 테스트 / 외부 judge)
plot_run1_curves.py            # G1/G2 곡선 figure 생성

watch_run1.sh                  # 학습 감시자 (250 step마다 진단 → diagnostics.log, 종료 후 자동 평가)
after_run1_gtoracle.sh         # Run 1 종료 후 GT-oracle(Exp.14) 참조선 자동 평가

make_grpo_dataset/             # 학습/held-out 데이터셋 생성 파이프라인 (①~⑥, 아래 5장)
docs/
  GRPO_WMONLY_HANDOFF.md       # ★ 현재 실험의 마스터 플랜 (3-run 계획·분기·판정 기준)
  GRPO_TRAINING_LOG.md         # 실험 1~14 이력 (무엇이 왜 실패/성공했는가)
  GRPO_DATASET_SPEC.md         # 데이터 JSONL 필드 정의
  GRPO_TRAIN_SPEC.md           # 학습 스펙
  RESULTS.md / THINK_FORMAT_SPEC.md / MEMORY_CONTEXT_SPEC.md / VLM_CHECKLIST.md
configs/vitg-384/              # V-JEPA2 inference/eval 설정
src/                           # Step 1 인터페이스(schema)·Phase 1~2 데이터/프롬프트 (맥락용)
```

> **대용량 산출물(데이터셋 프레임, 체크포인트, 로그)은 `.gitignore`로 제외**되어 있습니다.
> 코드와 문서만으로 파이프라인 전체를 재구성할 수 있게 구성했습니다.

---

## 4. Reward / 최적화 설계 (핵심 로직)

학습 스크립트는 reward를 **컴포넌트별로 분리 로깅**합니다(`reward_log.jsonl`) — 붕괴가 나면 어느 항이 원인인지 로그로 역추적할 수 있게. 현재 Run 1 구성은 `reward_mode=wm_likelihood`:

| 항 | 함수 | 하는 일 | 왜 |
|---|---|---|---|
| **P1** (주신호) | `wm_likelihood_reward` | 선택한 (verb,noun)의 월드 모델 likelihood를 **후보셋 내 재정규화**해 연속 보상으로 | "reward는 오직 WM의 예측 분포에서 산출" — 논문 클레임의 실체. raw softmax는 스케일이 작아(median std 0.015) 다른 항에 묻혀서 재정규화(median std 0.147) 사용 |
| **P4** (보조) | `think_convergence_reward` | think 속 후보 언급이 최종 선택으로 **수렴**하면 가점, 표류/장식이면 감점 | reasoning이 장식이 되지 않게. GT 없이 후보 집합+텍스트만의 결정론적 함수 (배제 논리의 "옳음"은 판정 안 함 → GT 뒷문 차단) |
| gate | `candidate_gate_reward_think` | 후보 목록 밖 verb/noun을 지어내면 감점 | hallucination 차단 (최소 구조 검증만) |
| format | `format_reward_think` | `<think>…</think><action>…</action>` 구조 확인 | 파싱 가능성 보장 |

**최적화 장치 (reward가 아니라 학습 역학 보정):**

- **Dynamic sampling** (`DynamicSamplingGRPOTrainer` + `--min_wm_spread`): flat한 프롬프트를 사전 제거(정적) + 무신호 그룹의 advantage 마스킹(런타임). **Exp.10 실패의 직접 처방.**
- **Clip-higher** (`--epsilon_high 0.28`): 새로운 시도가 성공했을 때 확률을 크게 올릴 수 있게 상한만 완화 → "4생성 전부 동일" 다양성 붕괴 방지.
- **Dr. GRPO** (`--loss_type dr_grpo --scale_rewards none`): "길게 쓰면 벌점이 희석되는" 길이 편향을 회계 수준에서 제거.
- **생성 수 8** (고정, 상한도 8): 그룹 내 점수 차이가 존재할 확률 확보. 하향은 Exp.10 조건 복원이라 금지.

**Run 2 대비로 구현되어 있으나 Run 1에는 미포함:**

- **P3** (`think_support_reward`, `reward_mode=wm_likelihood_p3`): 동결 base 모델로 `p(선택 | 결론-마스킹된 think)`를 측정해 reasoning이 결론을 지지할수록 보너스. **신경망 판정이 개입하는 유일한 항**이라 논문에선 "coherence regularizer"로 분리 서술하고, 안전장치(결론 마스킹 = 답안 예고편 hacking 차단, 가중치 ≤ P1·P4)를 강제합니다. **Run 1이 건강하면(분기 A) Run 2에서 추가.**

> 설계 원칙: **형식을 강제하는 규칙을 추가하지 않고, 잘못된 유인 구조를 제거한다.**
> reward에 관여하는 것은 WM의 출력물뿐 (P3는 예외적 regularizer로 별도 관리).

---

## 5. 데이터 파이프라인 (`make_grpo_dataset/`)

EPIC-Kitchens-100 프레임 → 월드 모델 추론 → 학습용 JSONL. `--split {train,validation}`으로 학습셋/held-out셋 모두 생성:

```
① select_train.py         # CSV에서 샘플 선정 (길이·trigger frame 필터, 디스크 보유 비디오만)
② vjepa_infer_train.py    # V-JEPA2 forward → verb/noun/action top-5 + likelihood
③ extract_frame_train.py  # trigger frame(=stop_frame - 1s) JPEG 추출
④ extract_memory_train.py # task_history + temporal_proximity (메모리 컨텍스트)
⑤ assemble_train.py       # 위를 sample_id로 조인 → grpo_dataset.jsonl
⑥ convert_to_train_format.py # 학습 스크립트가 먹는 최종 포맷 → grpo_train.jsonl
```

- 학습셋: `EPIC_100_train.csv` 기반, 현재 4,998 샘플 (P01~P06).
- held-out셋: `EPIC_100_validation.csv` 기반, 1,417 샘플 (**train과 비디오 단위 완전 분리**).
- 핵심 필드: `topk_actions_with_score`(각 후보의 `likelihood` = P1 reward의 입력), `gt_verb`/`gt_noun`(평가 전용).

---

## 6. 3-Run 계획 + 현재 진행

시간 제약으로 **총 3번의 학습 run 안에** 모든 개선 적용 + 검증 완료가 목표입니다. 항별 ablation 대신 **컴포넌트별 로깅 + 학습-외 검증**으로 원인을 추적합니다. 판정 지표는 셋 다 **held-out에서**:

| 코드 | 목표 | 판정 기준 |
|---|---|---|
| **G1** | GT-free 학습 성립 | WM-likelihood reward만으로 advantage 소실 없이 held-out 곡선 상승 |
| **G2** | WM 능가 | WM top-1이 틀렸지만 top-5엔 정답이 있는 구간(held-out의 22.7%)에서 VLM 정답률 > chance(0.20) |
| **G3** | Reasoning 인과성 | think 조건부 선택 likelihood(lift)가 학습에 따라 상승 |

```
Run 1 (wm_likelihood)          → G1 성립 검증. 종료 후 로그 진단으로 Run 2 분기 결정.
  ├─ 분기 A (건강)             → Run 2 = Run 1 + P3 (run_grpo_run2_branchA.sh)
  ├─ 분기 B (신호 부족)        → Run 2 = 필터 완화 + 온도↑ (branchB.sh)
  └─ 분기 C/C' (hacking/붕괴)  → Run 2 = P4 가중치↓ + gate↑ (branchC.sh)
Run 2 (분기별 1회 수정)        → 설정 확정. 분기 A면 P3 유지/제외를 마스킹 테스트로 판정.
Run 3 (설정 고정)              → 논문용 수치 생산: G2·G3 곡선 + GT-oracle 참조선 + judge 채점.
```

### 현재 상태 (2026-07-16)

- **사전 준비 완료**: held-out 1,417샘플 생성, wm_likelihood/P3 reward 구현, dynamic sampling 구현, 평가 도구 3종, 6-step 스모크로 전 경로 무오류 검증.
- **Run 1 학습 진행 중** (1,250 step, ~4시간). **초반 진단이 전부 분기 A(건강) 방향:**
  - P1(WM likelihood) 4분위 궤적 **단조 상승** (0.079 → 0.153), advantage std 유지 (Exp.10 극복)
  - total reward −0.023 → +0.45, gate 감점 → 0 (hallucination 소멸)
  - 생성 다양성 0.2 → 0.625 **증가** (붕괴의 정반대), think 단어수 안정
- **다음 관문**: Run 1 종료 후 **held-out 곡선**(G1 확정) — train reward 상승이 일반화되는지. 이게 되면 논문 핵심 클레임이 살고, 안 되면 주장 범위를 좁히는 분기점.

> 마스터 플랜 전문(분기 판정 기준·리스크·고정 설정)은 **`docs/GRPO_WMONLY_HANDOFF.md`**.

---

## 7. 실행 방법

```bash
# 환경 (conda: eve-cu124, trl 1.5.1, transformers 5.x)
source activate.sh

# 데이터셋 생성 (예: held-out)
python make_grpo_dataset/select_train.py --split validation
python make_grpo_dataset/vjepa_infer_train.py --selected data/grpo_dataset/selected_heldout.jsonl \
       --out data/grpo_dataset/predictions_heldout.jsonl
python make_grpo_dataset/extract_frame_train.py --selected data/grpo_dataset/selected_heldout.jsonl \
       --manifest data/grpo_dataset/frames_manifest_heldout.jsonl
python make_grpo_dataset/extract_memory_train.py --split validation
python make_grpo_dataset/assemble_train.py --split validation
python make_grpo_dataset/convert_to_train_format.py \
       --input data/grpo_dataset/grpo_dataset_heldout.jsonl --output data/grpo_dataset/grpo_heldout.jsonl

# 학습 (2×H200)
bash run_grpo_run1_wmonly.sh

# 세션과 무관하게 감시 + 250 step 진단 + 종료 후 자동 held-out 평가
setsid nohup bash watch_run1.sh > /dev/null 2>&1 < /dev/null &
cat runs/grpo_run1_wmonly/diagnostics.log   # 진단 누적 확인

# 종료 후 곡선
bash eval_checkpoints_run1.sh runs/grpo_run1_wmonly 500
/opt/conda/bin/python3 plot_run1_curves.py
```

**하드웨어**: 2×H200 (policy LoRA + rollout). LoRA r=16, `--num_generations 8`, `--max_completion_length 256`.
**의존성 주의**: V-JEPA2 추론용 `cv2`가 시스템 라이브러리 `libxcb1 libgl1 libglib2.0-0`를 요구 (학습 스크립트는 불필요).

---

## 8. 검증 도구 (학습과 독립, 보상 미사용)

- **`eval_heldout.py`** — 체크포인트를 held-out에서 평가. **G2**(WM-disagreement 구간 정답률, chance 0.20), GT 정확도, 후보 이탈률, WM-follow rate(rank-1 복사 collapse 감시), WM top-1 참조선.
- **`eval_reasoning_trace.py`** — reasoning 인과성(G3):
  - `--mode lift`: `log p(선택|이미지,후보,think) − log p(선택|…,think 없음)`. think가 장식이면 ≈0.
  - `--mode mask`: 결론 문장 마스킹 후 lift 붕괴 여부 → **P3 hacking 검출기** (Run 2에서 P3 유지/제외 판정).
  - `--mode shuffle`: 다른 샘플의 think와 짝지어 확률 붕괴 여부 → 범용 템플릿 검출.
  - `--mode judge`: 다른 모델 계열(Gemini)로 루브릭 5항목 채점 (self-preference bias 방지).

---

## 9. 지표 표기 주의 (혼용 금지)

- 자체 CSV 실측은 **sample-level top-5 hit rate** (verb 90.1 / noun 76.1 / action 상한 69.9%, n=704).
- V-JEPA2 논문 공식은 **mean-class recall@5** (verb 63.6 / noun 57.1 / action 39.7).
- 정의가 달라 **직접 비교 불가**. 자체 수치 인용 시 반드시 "sample-level top-5 hit rate" 명기.

---

## 관련 문서 빠른 링크

- **마스터 플랜** → [`docs/GRPO_WMONLY_HANDOFF.md`](docs/GRPO_WMONLY_HANDOFF.md)
- **실험 이력(왜 실패/성공)** → [`docs/GRPO_TRAINING_LOG.md`](docs/GRPO_TRAINING_LOG.md)
- **데이터 스펙** → [`docs/GRPO_DATASET_SPEC.md`](docs/GRPO_DATASET_SPEC.md)
- **think 포맷 설계** → [`docs/THINK_FORMAT_SPEC.md`](docs/THINK_FORMAT_SPEC.md)
