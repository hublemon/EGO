# 현재 Step1 학습 데이터 규모 정리

- 작성일: 2026-07-25 UTC
- 실측 기준: `EGO_jihun2`가 참조하는 `../datasets/Ego4D`
- 범위: 최근 결과에 사용된 LTA-aux strict next-action 및 adaptive-transition
  Step1 파이프라인
- 용량 표기: `du -sb` 결과를 GiB(`2^30 bytes`)로 환산

## 1. 요약

| 범위 | 학습 row | Validation row | 입력 데이터 | V-JEPA2 포함 |
|---|---:|---:|---:|---:|
| LTA-aux strict next-action + history | 44,219 | 6,960 | 754.5 GiB | 759.3 GiB |
| Adaptive transition + history | 18,962 | 4,458 | 450.5 GiB | 455.3 GiB |
| 두 실험군을 현재처럼 함께 보존 | 중복 cohort 포함 | 중복 cohort 포함 | **949.8 GiB** | **954.6 GiB** |

두 실험군은 GoalStep 원본 영상 255.1 GiB를 공유한다. 따라서 각 행의 용량을
단순 합산하지 않고, 공유 데이터를 한 번만 센 현재 전체 실측값은
약 `949.8 GiB = 0.928 TiB`다.

학습 결과 checkpoint와 prediction 등 run 산출물은 입력 데이터가 아니므로 위
합계에서 제외했다. 최근 관련 run 4개의 산출물은 별도로 28.2 GiB다.

## 2. LTA-aux strict next-action 파이프라인

현재 보고용 대표 결과인 LTA-aux cross-fitted final blend를 만들기 위해 사용한
데이터다.

### 2.1 논리적 데이터 규모

| 원천 | Train row | Val row | Train video | Val video | 라벨 |
|---|---:|---:|---:|---:|---|
| GoalStep strict-next | 29,293 | 6,960 | 564 | 130 | Verb/Noun/Action 전체 |
| LTA auxiliary | 14,926 | — | 793 | — | Verb/Noun 14,926, Action mask 3,029 |
| 합계 | **44,219** | **6,960** | 아래 참고 | **130** | — |

GoalStep train과 LTA aux의 video 교집합은 82개다. 학습 video 합집합은
`564 + 793 − 82 = 1,275`개로, GoalStep만 사용할 때보다 711개 늘었다.

LTA 원천 annotation은 97,105 action, 2,431 clip, 1,315 video였으며 다음 필터 후
14,926 auxiliary row가 남았다.

- strict-future A3가 없는 action 제외
- GoalStep validation 130개 video와 겹치는 771 row 제외
- 로컬 media가 없는 5 row 제외
- GoalStep taxonomy에서 verb와 noun이 모두 매핑되는 row만 유지

Direct joint training의 batch는 GoalStep 22개와 LTA aux 10개로 구성된다.
따라서 14,926 aux row는 전체가 별도 validation으로 쓰이는 것이 아니라 학습 전용
partial-label supervision으로 사용된다.

### 2.2 디스크 실측

| 데이터 | 파일/row | 실측 용량 | 비고 |
|---|---:|---:|---|
| GoalStep 원본 영상 | 현재 저장된 MP4 717개 | 255.141 GiB | 두 실험군 공용 |
| 전체 LTA `clip_256ss` media | 2,430 MP4 | 61.888 GiB | 실제 aux 선택 clip은 1,287개 |
| GoalStep V-JEPA feature cache | 37,588 tensor | 312.068 GiB | strict-next가 실제 참조하는 row는 36,253개 |
| LTA aux V-JEPA feature cache | 14,926 tensor | 123.934 GiB | aux index와 1:1 |
| LTA-aux history derived store | 37,588 summary | 1.365 GiB | `[4352,1024] → [17,1024]` 요약 |
| Annotation와 index | 약 106 MiB | 합계 영향 미미 | JSON/Parquet/registry |
| **입력 데이터 합계** | — | **약 754.5 GiB** | metadata 포함 |

GoalStep cache에는 기존 endpoint index의 train 30,374개와 val 7,214개가 모두
남아 있다. Strict-next 학습은 그중 train 29,293개와 val 6,960개만 참조하므로,
실제 디스크에는 현재 run에서 직접 사용하지 않는 1,335개 cache tensor도 포함된다.

각 raw V-JEPA cache tensor는 약 8.9 MB이며 shape은 `[4352, 1024]`다.
이 feature cache가 전체 입력 용량의 가장 큰 부분인 436.0 GiB를 차지한다.

## 3. Adaptive-transition 파이프라인

### 3.1 논리적 데이터 규모

| Split | Row | Unique video | 평균 history 길이 |
|---|---:|---:|---:|
| Train | 18,962 | 556 | 6.3938 |
| Validation | 4,458 | 125 | 6.3811 |
| 합계 | **23,420** | **681** | — |

History K=8 학습은 row를 추가하거나 버리지 않고 동일한 18,962/4,458 cohort를
그대로 사용한다. 새 원본 영상 decode나 V-JEPA backbone 재추출도 하지 않고,
기존 adaptive cache에서 summary store만 만든다.

### 3.2 디스크 실측

| 데이터 | 파일/row | 실측 용량 | 비고 |
|---|---:|---:|---|
| GoalStep 원본 영상 | 공용 | 255.141 GiB | LTA-aux 계열과 공유 |
| Adaptive V-JEPA feature cache | 23,420 tensor | 194.507 GiB | train + val |
| Adaptive history derived store | 23,420 summary | 0.805 GiB | K=8 history 학습 입력 |
| Annotation와 index | 약 29 MiB | 합계 영향 미미 | JSON + base/history index |
| **입력 데이터 합계** | — | **약 450.5 GiB** | metadata 포함 |

## 4. 공용 모델 및 산출물

### V-JEPA2 backbone

- 경로: `../EGO_jihun/checkpoints/vjepa2/vitl.pt`
- 실측: 5,127,726,842 bytes = **4.776 GiB**
- raw video에서 feature를 새로 추출할 때 필요하다.
- feature cache가 이미 완성돼 있고 probe/history head만 다시 학습한다면 backbone
  checkpoint와 원본 영상 decode는 필요하지 않다.

### 최근 run 산출물

| Run | 용량 |
|---|---:|
| LTA-aux direct | 7.254 GiB |
| LTA-aux history | 7.001 GiB |
| Adaptive direct | 7.255 GiB |
| Adaptive history | 6.701 GiB |
| **합계** | **28.211 GiB** |

이는 epoch checkpoint, full-val prediction 및 평가 결과이므로 학습 입력 데이터
합계에는 포함하지 않았다.

## 5. 재학습 시 실제로 필요한 양

| 재실행 범위 | 필요한 데이터 |
|---|---|
| Probe만 재학습 | 해당 feature cache + index, 원본 영상 불필요 |
| History head만 재학습 | derived store + history index + frozen probe checkpoint |
| Endpoint/window를 바꿔 feature 재추출 | 원본 영상 + V-JEPA2 backbone + 새 index |
| LTA-aux 전체 cold start | 약 759.3 GiB + 새 run 출력 공간 |
| 두 최근 실험군 전체 cold start | 약 954.6 GiB + 새 run 출력 공간 |

따라서 현재 Step1에서 데이터 비용의 핵심은 annotation row 수가 아니라
샘플당 약 8.9 MB인 frozen V-JEPA feature cache다. History 단계는 기존 feature를
17개 temporal summary로 압축해 재사용하므로 추가 용량은 1 GiB 안팎에 그친다.

## 6. 근거 파일

- GoalStep strict-next 통계:
  `src/ego/step1_action_anticipation/goalstep/index_end_m1_lobs8_next_action/build_stats.json`
- LTA aux 통계:
  `src/ego/step1_action_anticipation/goalstep/index_lta_aux_end_m1_lobs8/build_stats.json`
- LTA-aux history store:
  `../datasets/Ego4D/goalstep_history_context_store_ltaaux/manifest.json`
- Adaptive 통계:
  `src/ego/step1_action_anticipation/goalstep/index_adaptive_transition_mr24x8/build_stats.json`
- Adaptive history store:
  `../datasets/Ego4D/goalstep_history_context_store_adaptive_transition_mr24x8/manifest.json`
