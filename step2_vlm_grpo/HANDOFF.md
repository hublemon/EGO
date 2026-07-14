# Step 2 (VLM GRPO) — 개발 현황 핸드오프

> 작성: 2026-07-14. `dummy` 브랜치. 이 저장소(`hublemon/EGO`)의 `main`은 아직 스캐폴딩 단계라,
> 로컬에서 실제로 진행된 **Step 2 = VLM GRPO 강화학습** 개발 코드와 실험 결과만 골라 이 브랜치에 정리했다.
> Step 1(V-JEPA2), Step 3(메모리/트리거), 프론티어 VLM 베이스라인(Phase 1~4 전체) 등 Step 2와 직접 관련 없는 내용은 모두 제외했다.

## 한 줄 요약

WM(V-JEPA2)이 뽑은 Top-5 verb/noun 후보 중에서 Qwen2.5-VL-7B-Instruct가 GT next-action을 고르도록
GRPO로 파인튜닝하는 실험을 총 14차(+파생 실험 포함 19 run) 진행했다. **가장 최근 실험(`grpo_final`, 실험 14)의
GT reward가 이전 최선 대비 +40.6% 향상(0.578, 0~1.5 스케일)**했고, WM 순위 보조 신호의 발산 문제도 해소했다.
단, **지금까지의 모든 수치는 train 배치 위 측정값이며, held-out(validation) 평가는 아직 수행 전**이다 — 다음 세션의 최우선 작업.

---

## 1. 디렉토리 구조 (이 핸드오프 패키지)

```
step2_vlm_grpo/
├── HANDOFF.md                       # 이 문서
├── docs/
│   ├── GRPO_DATASET_SPEC.md         # GRPO 학습용 데이터셋 설계 명세
│   ├── GRPO_TRAIN_SPEC.md           # 학습 코드/reward 함수 설계 명세 (실험 4~6)
│   ├── GRPO_TRAINING_LOG.md         # 실험 1~14 전체 결과 로그 (Living Document, 가장 중요)
│   └── STEP2_PRE_BASELINE.md        # Step 2 착수 직전, 무학습 Qwen vs Gemini 비교
├── code/
│   ├── train_qwen25vl_grpo_ek100.py # GRPO 학습 메인 스크립트 (reward 함수 전부 포함)
│   ├── run_grpo_final.sh            # 현재 최선 실험(실험 14)의 실행 커맨드
│   └── make_grpo_dataset/           # GRPO 학습 데이터셋 생성 6단계 파이프라인
│       ├── select_train.py          # ① EK100 train에서 샘플 선정
│       ├── vjepa_infer_train.py     # ② V-JEPA2로 Top-5 verb/noun/action 예측 추출
│       ├── extract_frame_train.py   # ③ trigger frame 이미지 추출
│       ├── extract_memory_train.py  # ④ task_history/temporal_proximity 메모리 컨텍스트 추출
│       ├── assemble_train.py        # ⑤ 위 산출물을 합쳐 grpo_dataset.jsonl 생성
│       ├── analyze_train.py         # ⑥ GT hit rate 등 통계 산출
│       ├── convert_to_train_format.py  # grpo_dataset.jsonl → 학습용 superset JSONL 변환
│       └── HANDOFF.md               # 데이터셋 파이프라인 자체의 세션 핸드오프 (2026-05-28 시점)
└── results/
    ├── dataset_stats/hit_rate.json  # GRPO 학습 데이터셋(4,998 샘플)의 GT hit-rate 통계
    └── experiments/{experiment_name}/      # 19개 실험 run의 경량 메타데이터
        ├── meta.json                # reward 구성, 모델, LoRA 설정, 시작 시각
        ├── summary.json             # 종료 시 총 스텝/소요 시간/최종 reward
        ├── training_metadata.json   # 학습 커맨드 파라미터
        ├── reward_log.jsonl         # step별 reward 구성요소 평균 (5 step 주기)
        ├── completion_samples.jsonl # step별 실제 생성 샘플 원문 (100 step 주기)
        └── think_analysis.jsonl     # think 블록 단어수/다양성/후보언급률 (100 step 주기, think 계열만)
```

**의도적으로 제외한 것** (용량·목적상 부적합):
- 모델 체크포인트/LoRA 가중치(`adapter_model.safetensors`, `checkpoint-*/`, optimizer state) — 총 12GB, 로컬 `~/work/jihun/EGO/runs/`에 원본 보존
- `runs/*/launch.log` (run당 최대 230MB, 원시 stdout), `tokenizer.json`, `training_args.bin` — 재생성 가능한 프레임워크 산출물
- `data/grpo_dataset/frames/`(833MB 원본 프레임 이미지), `grpo_dataset.jsonl`/`grpo_train*.jsonl`(원본 학습 데이터 7~10MB×3) — 코드가 아닌 데이터 자체, 필요 시 로컬 `~/work/jihun/EGO/data/grpo_dataset/`에서 파이프라인 재실행으로 재생성 가능
- `make_grpo_dataset/_bg/` 하위 자동화 스크립트·원시 로그 — 운영용 부산물, 핵심 로직은 위 6단계 스크립트에 모두 포함됨
- Step 1(V-JEPA2 원본 코드 `src/vjepa2/`), Step 3(`src/eve_memory_context/`), 프론티어 VLM 베이스라인(Phase 1~4, `docs/RESULTS.md` 전체, `src/vlm_prompter.py`, `src/evaluate.py`, `data/results/` 등) — Step 2 범위 밖

---

## 2. 파이프라인 개요

```
EK100 train CSV (~67,000 narrations)
      │
      ▼
① select_train.py          필터링(action ≥1.5s) 후 랜덤 샘플링 → selected_train.jsonl
      │
      ▼
② vjepa_infer_train.py     V-JEPA2 ViT-g/384 + EK100 classifier → verb/noun/action Top-5 + likelihood
      │
      ▼
③ extract_frame_train.py   trigger_frame(= stop_frame − 1초) JPEG 추출
      │
      ▼
④ extract_memory_train.py  task_history / temporal_proximity 메모리 컨텍스트 추출
      │
      ▼
⑤ assemble_train.py        위 4개 산출물 병합 → grpo_dataset.jsonl (최종 4,998 샘플)
      │
      ▼
⑥ analyze_train.py         GT-in-Top5 hit rate 등 통계
      │
      ▼
convert_to_train_format.py grpo_dataset.jsonl → 학습용 superset (grpo_train.jsonl / grpo_train_think.jsonl)
      │
      ▼
train_qwen25vl_grpo_ek100.py   TRL GRPOTrainer + 커스텀 reward 함수 → LoRA 어댑터 (runs/{실험명}/)
```

자세한 설계 배경은 `docs/GRPO_DATASET_SPEC.md`(데이터셋), `docs/GRPO_TRAIN_SPEC.md`(학습 코드/reward), `code/make_grpo_dataset/HANDOFF.md`(데이터셋 파이프라인 진행 상태, 2026-05-28 기준)에 있다.

### 데이터셋 통계 (`results/dataset_stats/hit_rate.json`, n=4,998)

| 지표 | verb | noun | **action(joint)** |
|---|---|---|---|
| GT ∈ Top-5 (VLM 이론 상한) | 96.0% | 95.2% | **92.4%** |
| WM rank-1 == GT (WM 베이스라인) | 72.1% | 76.2% | **70.0%** |

→ VLM이 후보 안에서 항상 올바르게 고르기만 해도 WM 베이스라인(70%) → 이론 상한(92%)까지 **+22pp** 여지가 있다.

---

## 3. Step 2 착수 전 pre-baseline

무학습 Qwen2.5-VL-7B-Instruct가 Gemini-2.0-flash 대비 raw action 정확도 +20pp(60% vs 40%)를 보였으나
포맷 준수율(80%)이 약점 — GRPO reward에 포맷/후보 준수를 강하게 넣으면 즉시 개선 가능하다는 가설이 GRPO 착수의 출발점이었다.
자세한 내용: `docs/STEP2_PRE_BASELINE.md`.

---

## 4. 실험 전체 요약 (14차, 19 run)

| # | 디렉토리 | 보상 설계 요약 | steps | GT reward(초/중/후) | 판정 |
|---|---|---|---|---|---|
| 1 | `grpo_stage1_noun` | WM 명사 rank-1 그대로 복사하면 최고점 (단서 노출) | 2,499 | — | ❌ 즉시 포화·collapse |
| 2 | `grpo_stage2_action` | WM 행동 rank-1 그대로 복사하면 최고점 (단서 노출) | 2,499 | — | ❌ 즉시 포화·collapse |
| 3 | `grpo_gt_improved` | GT verb+noun 맞추면 점수 (단순 GT 채점, 추론 없음) | 2,499 | ~0.7 평탄 | △ 형식·후보만 학습 |
| 4 | `grpo_think` | 추론 태그 + 후보 준수(+0.5) + GT 약한 점수 → 형식 보상이 GT 압도 | 1,500 | 0.26/0.47/0.25 | ❌ 형식 collapse |
| 5a | `grpo_ranking` | WM 후보 순위 점수 단독, 단서 노출 | 1,500 | — | ❌ rank1 복사 collapse |
| 5b | `grpo_think_ranking` | 추론 태그 + 후보 준수 + WM 순위 점수 | 1,500 | — | ◐ collapse 없음 |
| 6-S1 | `grpo_stage_noun` | WM 명사 순위 점수 단독, 단서 노출 | 1,500 | — | ❌ rank1 collapse |
| 6-S2 | `grpo_stage_action` | WM 행동 순위 점수 단독 (S1 이어받기) | 1,500 | — | ❌ rank1 collapse |
| 7 | `grpo_think_gt` | 추론 태그+품질+후보 이탈 패널티+GT 강한 점수(최대 1.5) | 1,500 | 0.34/0.46/0.47 | ◐ 피크 step 1240 |
| 8 | `grpo_ranking_fix` | WM 순위 점수 + 단서 제거(셔플·점수 숨김) | 450(중단) | — | ◐ collapse 차단 확인 |
| 9 | `grpo_think_gt_fix` | 실험7 동일 + max_steps=750(overshoot 방지) | 750 | 0.26/0.35/0.40 | ◐ |
| 10 | `grpo_think_wm_rank_fix` | 추론 태그+품질+후보 이탈 패널티+**WM 순위만(GT 없음)** | 750 | — | ❌ reward≈0, 학습 실패 |
| 11 | `grpo_think_gt_combo` | 실험9 + WM 순위 점수 보조 추가 | 750 | 0.27/0.36/0.41 | ◐ GT 미세 우위 |
| 12 | `grpo_2stage_gt_s1/s2` | S1 명사 GT → S2 행동 GT (2단계 순차) | 375+375 | 0.33/0.34/0.34 | △ flat, 2-stage 효과 없음 |
| 13 | `grpo_2stage_combo_s1/s2` | S1 명사 순위+GT → S2 행동 GT+WM 순위 | 375+375 | 0.33/0.34/0.35 | △ flat |
| **14** | **`grpo_final`** | 실험11 설계 + 5,000샘플·num_gen=8·beta=0.01·GT v3 퍼지 매칭·1,250 steps | **1,250** | **0.38/0.52/0.58** | ✅ **현재 최선** |

GT reward는 `reward_gt_accuracy_reward_think_v2`(또는 실험14는 v3) 배치 평균, 0~1.5 스케일(÷1.5 ≈ joint 정확도 근사). 학습 시간 누계 ≈ 38.7h GPU (2×H200).
전체 판정 근거와 실험 9~14 상세 표, think 분석은 `docs/GRPO_TRAINING_LOG.md` §1·§3·§4 참조.

### 누적 교훈 (10개, 자세한 근거는 `GRPO_TRAINING_LOG.md` §2)

1. **collapse 원인은 보상이 아니라 "후보 단서 노출"** — 점수·rank 순으로 후보가 보이면 rank1 복사가 자명해짐. 단서 제거(셔플+점수 숨김)만으로 즉시 해소.
2. think-format(답 전 추론 강제) + verb·noun 분리 입력이 collapse에 가장 강함.
3. easy saturating reward(format·candidate 가점)가 정답 신호를 압도하면 think도 collapse — 후보 준수는 "당연한 것"으로 게이트화(0/−0.5)하고 GT 배점을 올려야 함.
4. 1 epoch(1,500 step)은 overshoot — best checkpoint는 피크 지점(step ~1,240) 근처.
5. WM rank reward 단독으로는 학습 신호 없음(reward≈0) — GT가 주신호로 반드시 필요.
6. GT 기반 2-stage(noun→action)는 단일 stage 대비 개선 없음.
7. GT+WM rank 복합은 GT 단독 대비 미세 우위 수준 — 통계적 유의성은 held-out 전까지 불명.
8. **모든 train-time 지표는 held-out 평가로만 확정** — 소표본 우연 일치 가능성 배제 불가.
9. 데이터 증량(3000→5000)+num_gen 증가(4→8)+GT-not-in-top5 필터 조합이 학습 효율을 크게 향상(실험14, +40.6%).
10. beta(KL)=0.01(기존 0.04)로 낮추면 더 긴 학습에서도 안정적으로 GT reward 상승 지속.

---

## 5. 현재 최선 모델과 다음 단계

**현재 최선**: 실험 14 `grpo_final` (`results/experiments/grpo_final/`, 학습 코드는 `code/run_grpo_final.sh`) — GT reward 후반 평균 0.578, WM rank 보조신호 말기 +0.259로 안정화, candidate gate 이탈 사실상 0.
LoRA 가중치 원본은 로컬 `~/work/jihun/EGO/runs/grpo_final/adapter_model.safetensors` (용량상 이 브랜치에는 미포함).

### 🔴 즉시 — held-out 평가 (다음 세션 최우선)

지금까지의 모든 수치는 **train 배치 위 측정값**이다. 학습된 모델이 WM rank-1 베이스라인(70%)을 실제로 넘는지
EK100 **validation** set(GRPO 학습에 쓰지 않은 split)으로 확인해야 다음 방향을 결정할 수 있다.

평가 우선순위: 무학습 Qwen2.5-VL(베이스라인) → WM rank-1 그대로 → `grpo_final`(실험14) → `grpo_think_gt_combo`(실험11) → `grpo_think_gt_fix`(실험9). 지표: action joint / verb / noun 정확도.

### 🟠 그 다음 (held-out 결과에 따라)
- 옵션 A: 실험14 방향 유지 + max_steps 증가(1,250→2,000~2,500, ~0.8 epoch)
- 옵션 B: 실험14 + learning rate 조정 또는 LoRA rank 증가
- 실험 8(`ranking_fix`) 완주 (450→1,500 step)

### 🟡 추후
- memory_context(temporal) on/off ablation (Step 2에 아직 미적용)
- reasoning 정성 평가 자동화 (영상 단서 인용·일관성)

---

## 6. 로컬 원본 위치 (참고용)

이 핸드오프는 `~/work/jihun/EGO/` (로컬, git 미관리)에서 Step 2 관련 부분만 추출한 것이다. 원본 전체(체크포인트 가중치, 원본 프레임 이미지, Step 1/3 코드 포함)는 로컬 환경에 그대로 남아 있다.
