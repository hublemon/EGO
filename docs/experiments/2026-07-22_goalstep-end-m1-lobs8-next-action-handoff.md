# GoalStep `A2.end-1s / 8s -> next A3` 실험 보고서

- 날짜: 2026-07-22
- 실험 ID: `z1_end_m1_lobs8_next_action_vna_ep10`
- 상태: **사용자 판단으로 epoch 9 도중 중단 (epoch 8까지 완료·보존)**
- 목적: 관찰 구간에 포함된 action이 아니라 시간적으로 그 뒤에 오는 action을 예측

## 1. 핵심 계약

현재 action을 `A2`, 예측할 다음 action을 `A3`라고 한다.

```text
observation = [A2.end - 9s, A2.end - 1s]
target      = first same-level A3 with A3.start >= A2.end
```

따라서 classifier의 정답은 `A2`가 아니라 `A3`다. 관찰 구간에는 대부분 `A2`가
보이지만, 학습 목표는 그 다음 action이므로 action recognition label leakage는
제거했다.

GoalStep에는 step과 substep annotation이 서로 중첩된다. 기존 endpoint index에서
단순히 다음 행으로 label을 한 칸 shift하면 target이 관찰 종료보다 과거가 되는
표본이 train 4,429개, val 1,028개 발생한다. 이를 막기 위해 다음 조건을 모두
만족하는 첫 target만 사용했다.

1. `A2`와 동일한 annotation level (`step -> step`, `substep -> substep`)
2. `A3.start >= A2.end`

이 정의로 모든 retained sample에서 `A3.start - observation_end >= 1s`가 성립한다.

## 2. 우선순위 전환 및 실행 상태

기존 `action_start-1s / 16s` feature extraction과 이를 포함하던 serial queue를
중단했다. 중단 당시 생성된 cache 약 31GB/3,704개 파일은 삭제하지 않고 보존했다.
따라서 추후 resumable extraction으로 이어갈 수 있다.

새 실험은 다음 tmux 세션에서 실행한다.

```text
session: ego_goalstep_end_m1_lobs8_next_action
windows: pipeline, dashboard, tunnel
GPU: 0 (NVIDIA H200)
epochs: 10
```

실행 설정과 산출물 위치:

- config: `configs/step1/goalstep/z1_end_m1_lobs8_next_action_vna_ep10.yaml`
- index: `src/ego/step1_action_anticipation/goalstep/index_end_m1_lobs8_next_action/`
- run: `outputs/goalstep/runs/z1_end_m1_lobs8_next_action_vna_ep10/`
- source feature cache: `../datasets/Ego4D/goalstep_feature_cache_end_m1_lobs8_vna/`
- local dashboard: `http://127.0.0.1:17866`
- public dashboard: `https://metallica-bunny-vcr-dame.trycloudflare.com`

## 3. 피처 재사용 방식

새로운 영상 decode나 V-JEPA2 feature extraction은 수행하지 않는다. 기존
`action_end-1s / 8s` cache의 frozen visual token을 그대로 읽는다.

기존 cache `.pt`에는 원래 `A2` label도 들어 있으므로 index 파일만 바꾸는 것으로는
충분하지 않다. 이번 구현은 다음 방식으로 label leakage를 방지한다.

1. 새 index의 `cache_sample_id`가 기존 endpoint index의 원래 row/sample ID를 보존한다.
2. loader는 해당 ID로 기존 visual feature를 읽는다.
3. `verb_id`, `noun_id`, `action_id`만 새 index의 `A3` label로 메모리에서 override한다.
4. 기존 313GB cache 파일은 수정하거나 복제하지 않는다.

train/val 모든 새 sample ID에 대응하는 cache 파일이 존재하고, 실제 loader 출력에
override label이 적용되는 것을 학습 전에 검증했다.

## 4. 데이터 규모와 horizon

| 구분 | 기존 endpoint 표본 | retained | 다음 target 없음으로 제외 |
|---|---:|---:|---:|
| train | 30,374 | 29,293 | 1,081 |
| val | 7,214 | 6,960 | 254 |

Target horizon은 `A3.start - observation_end`다.

| 구분 | 최소 | 중앙값 | p90 | 최대 |
|---|---:|---:|---:|---:|
| train | 1.00s | 1.52s | 20.00s | 5,445.32s |
| val | 1.00s | 1.55s | 25.69s | 1,469.95s |

긴 horizon 표본 수:

| 구분 | >10s | >30s | >60s | >300s |
|---|---:|---:|---:|---:|
| train | 4,593 | 2,243 | 1,284 | 142 |
| val | 1,197 | 627 | 354 | 49 |

Annotation level 구성:

- train: step 11,924 / substep 17,369
- val: step 2,934 / substep 4,026

다음 segment의 action class가 관찰된 `A2`와 우연히 같은 경우도 있다.

- train: 동일 class 2,884 / 다른 class 26,409
- val: 동일 class 718 / 다른 class 6,242

동일 class 표본도 서로 다른 시간 segment의 다음 action이므로 제거하지 않았다.

## 5. 학습 설정

- frozen feature source: V-JEPA2 ViT-L endpoint cache
- heads: verb / noun / action
- epochs: 10
- batch size: 32
- optimizer: AdamW
- learning rate: 3e-4
- weight decay: 1e-4
- focal loss: gamma 2.0, alpha 0.25
- precision: bf16 train, fp32 evaluation
- fixed per-epoch validation subset: 2,000, seed 42
- checkpoint selection: action Top-5 accuracy
- metrics: verb/noun/action CMR@5, Top-1, Top-5, Top-10, Top-15

## 6. 결과

### 최초 정상성 확인: epoch 1, fixed validation subset `n=2,000`

| head | CMR@5 | Top-1 | Top-5 | Top-10 | Top-15 |
|---|---:|---:|---:|---:|---:|
| verb | 15.04 | 14.55 | 47.60 | 63.35 | 70.65 |
| noun | 8.92 | 28.20 | 51.25 | 61.70 | 71.25 |
| action | 6.10 | 4.85 | 19.55 | 30.15 | 37.75 |

최종 완료된 8개 epoch 중 best는 epoch 3이다.

| CMR@5 | Top-1 | Top-5 | Top-10 | Top-15 |
|---:|---:|---:|---:|---:|
| 11.53 | 7.10 | 25.70 | 37.70 | 46.55 |

epoch 9는 도중에 중단했으므로 평가 결과가 없다. 중단 시점까지의 checkpoint와
학습 이력은 보존했으며 full-validation은 수행하지 않았다.

## 7. 한계

### 7.1 classifier target horizon은 고정 1초가 아니다

V-JEPA2 predictor에 전달된 anticipation conditioning은 기존 cache와 같은 1초다.
하지만 classifier가 맞혀야 하는 `A3` 시작 시점은 annotation gap 때문에 가변이다.
중앙값은 약 1.5초지만 p90은 20~26초이고 극단치는 수천 초다. 따라서 이 실험은
엄밀한 fixed-horizon 1초 예측이 아니라 **A2 종료 직전 문맥으로 다음 annotated
action을 예측하는 variable-horizon task**다.

### 7.2 step/substep은 하나의 단일 행동열이 아니다

GoalStep step과 substep은 계층적으로 중첩된다. 과거 target을 방지하기 위해
동일 level 안에서만 다음 action을 찾았지만, 결과 데이터는 step-transition과
substep-transition 두 종류가 섞여 있다. 향후에는 level별 성능도 별도로 보고하거나
한 level만 택하는 ablation이 필요하다.

### 7.3 긴 annotation 공백과 outlier

라벨링되지 않은 공백이 길어도 “다음 annotated action”이라는 규칙을 유지했기
때문에 수 분~수십 분 뒤 target도 포함된다. 이것은 사용자가 요구한 “빈 시간이
있어도 다음을 예측”하는 의미에는 부합하지만, 짧은-horizon 표본과 난도가 크게
다르다. 결과 해석 시 horizon bucket별 평가가 필요하다.

### 7.4 마지막 action 제외에 따른 selection bias

같은 영상·같은 level에서 뒤따르는 action이 없는 표본 1,335개는 정답을 만들 수
없어 제외했다. 따라서 평가 결과는 후속 action이 존재하는 segment에만 적용된다.

### 7.5 관찰 구간은 A2 recognition 단서를 강하게 포함한다

정답 label은 A3이므로 직접적인 recognition leakage는 아니지만, A2를 거의 끝까지
보는 것은 A3 transition을 맞히는 강한 간접 단서다. 이 결과를 `action_start-1s`
실험과 비교할 때 관찰 정보량과 target horizon이 동시에 달라진다는 점을 고려해야
한다.

## 8. 재현 및 검증 파일

- index builder: `src/ego/step1_action_anticipation/goalstep/build_goalstep_next_action_index.py`
- relabel-aware loader: `src/ego/step1_action_anticipation/data/feature_cache.py`
- trainer integration: `src/ego/step1_action_anticipation/goalstep/train_goalstep_z1.py`
- run script: `scripts/step1/goalstep/run_end_m1_lobs8_next_action_vna_ep10.sh`
- tmux/UI launcher: `scripts/step1/goalstep/start_tmux_end_m1_lobs8_next_action.sh`
