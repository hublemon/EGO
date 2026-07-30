# GoalStep LTA-aux Phase 1 실행 Handoff

- 작성일: 2026-07-24 UTC
- 범위: LTA A1 `both-match` 보조 감독으로 direct visual foundation을 다시 학습하고,
  새 P0-a와 GoalStep-only visual-history Phase 1까지 재실행
- 실행 스크립트:
  `scripts/step1/goalstep/run_ltaaux_phase1.sh`
- 실행 시작: **2026-07-24 12:19:46 UTC**
- tmux: `ego_goalstep_ltaaux_phase1`
- 상태: **Stage 1 LTA feature extraction 진행 중 / 결과 pending**. 아래 경로의
  runtime artifact와 UI를 정본으로 확인한다.

## 1. 고정된 실험 계약

이번 실험의 관찰 계약은 다음 하나뿐이다.

```text
A2 observation: [max(clip_start, A2.end - 1s - 8s), A2.end - 1s]
sampling:        uniform 32 frames
target A3:       same LTA clip에서 action_idx가 더 크고
                 A3.start >= A2.end인 첫 action
```

- endpoint: **action_end−1s**
- 최대 observation 길이: **8초**
- sampling: **32-frame uniform**
- target: 관찰 중인 A2가 아니라 strict-future A3
- output 불변식:
  `A3.start >= A2.end` 및 `obs_end < A3.start`
- operational `*_sec` 좌표는 `clip_256ss/<clip_uid>.mp4` 디코딩을 위한
  clip-relative 좌표다.
- `*_video_sec` 감사 컬럼은
  `clip_parent_start_sec + action_clip_*_sec`으로 별도 보존한다.

**Adaptive transition window는 완전히 배제한다.** MR24+8, terminal mask, adaptive
endpoint, adaptive cache 및 temporal metadata를 이 파이프라인 어디에서도 사용하지
않는다. Direct probe의 `use_temporal_metadata`도 `false`다.

## 2. 실제 Stage 0 index 결과

정본 경로:

```text
src/ego/step1_action_anticipation/goalstep/index_lta_aux_end_m1_lobs8/
├── train.parquet
├── build_stats.json
└── action_registry.json
```

Aux는 학습 전용이므로 `val.parquet`은 만들지 않는다. 동봉된
`action_registry.json`은 GoalStep 81/140/293 registry의 byte-identical 복사본이다.

| 항목 | 실제 값 |
|---|---:|
| 원본 LTA action | 97,105 |
| 원본 LTA clip / video | 2,431 / 1,315 |
| strict target + verb/noun both-match 후보 | 15,702 |
| GoalStep 전체 val 130-video 누수 제외 | 771 |
| 로컬 `clip_256ss` media 부재 제외 | 5 |
| **최종 aux row** | **14,926** |
| **최종 aux video / clip** | **793 / 1,287** |
| LTA train / LTA val source row | 10,544 / 4,382 |
| action mask가 켜진 row | 3,029 |
| action mask가 커버하는 GoalStep action | 65 / 293 |

LTA annotation은 겹치는 8초 구간이 많다. 단순히 바로 다음 `action_idx`를 target으로
삼으면 action recognition이 되는 행이 다수 생기므로, 각 A2에서
`A3.start >= A2.end`를 만족하는 첫 later action을 선택했다. 이 과정에서 겹치는
중간 annotation을 누적 225,311회 건너뛰었고, strict target이 없는 A2 3,742개는
제외했다.

### 비디오 다양성 증가

| 학습 원천 | row | unique video |
|---|---:|---:|
| GoalStep strict-next train | 29,293 | 564 |
| LTA aux A1 | 14,926 | 793 |
| 두 집합의 video 교집합 | — | 82 |
| **합집합** | — | **1,275** |

따라서 visual foundation이 접하는 비디오 다양성은
**564 → 1,275, 즉 2.2606배**가 된다. 단순 합이 아니라 `video_uid` 합집합 기준이며,
순증가는 711개 비디오다.

## 3. 피처 정책

### 새로 추출해야 하는 것

LTA aux 14,926행은 기존 GoalStep 비디오와 별도의 `clip_256ss`를 사용하므로
**새 LTA V-JEPA feature 추출이 필요하다.**

- extraction config:
  `configs/step1/goalstep/z1_lta_aux_end_m1_lobs8.yaml`
- video:
  `../datasets/Ego4D/v2/clip_256ss/<clip_uid>.mp4`
- output cache:
  `../datasets/Ego4D/lta_aux_feature_cache_end_m1_lobs8`
- 계약: fixed end−1s / max 8s / uniform 32f / 256px

### 재사용하는 것

GoalStep 29,293행은 이미 동일한 fixed endpoint 계약으로 추출되어 있으므로 아래
cache를 그대로 재사용한다.

```text
../datasets/Ego4D/goalstep_feature_cache_end_m1_lobs8_vna
```

다만 joint direct probe가 새로 학습되므로, Phase 1에 들어가는 frozen visual logits의
fingerprint도 달라진다. 따라서 기존 GoalStep raw feature cache는 재사용하지만
아래 derived history store는 **새로 생성**해야 한다.

```text
../datasets/Ego4D/goalstep_history_context_store_ltaaux
```

## 4. Partial-label 및 action-mask 주의점

LTA target은 GoalStep taxonomy에 exact match된 head에만 loss를 건다.

- `verb_mask=true`: verb focal loss 허용
- `noun_mask=true`: noun focal loss 허용
- `action_mask=true`: `(verb,noun)` raw pair가 기존 293-action registry에 있을 때만
  action focal loss 허용
- registry에 없는 class 또는 pair는 `-1`이며 해당 loss를 차단
- GoalStep 81/140/293 class 수는 절대 늘리지 않는다.

초기 handoff는 action 완전 일치를 약 150행으로 근사했지만, canonical strict-later A3
builder에서 실제 `action_mask=true`는 **3,029행, 65 classes**다. 이는 구현 오류가
아니라 A3 target 재선택 뒤 실제 registry pair로 재집계한 결과다. 따라서 이 arm은
계획 당시 생각했던 “거의 V/N-only”보다 action-head aux gradient가 훨씬 크다.
결과 해석 시 반드시 이 차이를 명시하고, direct action Top-1/5/10과 GoalStep train
loss 추이를 함께 확인해야 한다.

taxonomy surface 매칭은 handoff §3.1의 보수적 규칙을 그대로 쓴다. `class_key`와
members를 comma/pipe로만 분리하며 실제 CSV 일부에 있는 semicolon compound alias는
확장하지 않는다.

## 5. 경로 계약

| 산출물 | 경로 |
|---|---|
| Stage 0 builder | `src/ego/step1_action_anticipation/goalstep/build_lta_aux_index.py` |
| LTA aux index | `src/ego/step1_action_anticipation/goalstep/index_lta_aux_end_m1_lobs8` |
| LTA extraction config | `configs/step1/goalstep/z1_lta_aux_end_m1_lobs8.yaml` |
| 새 LTA feature cache | `../datasets/Ego4D/lta_aux_feature_cache_end_m1_lobs8` |
| 기존 GoalStep feature cache | `../datasets/Ego4D/goalstep_feature_cache_end_m1_lobs8_vna` |
| joint direct config | `configs/step1/goalstep/z1_end_m1_lobs8_next_action_vna_ep10_ltaaux.yaml` |
| joint direct run | `outputs/goalstep/runs/z1_end_m1_lobs8_next_action_vna_ep10_ltaaux` |
| 새 P0-a run | `outputs/goalstep/runs/history_context_phase0_ltaaux` |
| 재사용 history index | `src/ego/step1_action_anticipation/goalstep/index_end_m1_lobs8_next_action_history_k8` |
| 새 derived store | `../datasets/Ego4D/goalstep_history_context_store_ltaaux` |
| Phase 1 config | `configs/step1/goalstep/z1_history_context_k8_vna_ep10_ltaaux.yaml` |
| Phase 1 run | `outputs/goalstep/runs/z1_history_context_k8_vna_ep10_ltaaux` |
| pipeline log | `outputs/goalstep/runs/z1_end_m1_lobs8_next_action_vna_ep10_ltaaux/logs/pipeline.log` |

기존 non-aux run과 canonical OOF artifact는 읽기 전용이며, 모든 신규 run/store는
`_ltaaux` suffix를 사용한다.

## 6. 실행 순서

`run_ltaaux_phase1.sh`가 다음 순서를 fail-closed로 직렬 실행한다.

1. strict LTA aux A1 index 재생성 및 row/leak/adaptive 계약 검사
2. LTA 14,926행 V-JEPA feature 추출
3. GoalStep+LTA joint direct probe 10 epochs
   - batch 32: GoalStep 22 + aux 10
   - aux loss weight `λ=0.3`
   - BF16, LR `3e-4`, WD `1e-4`
4. 새 direct epoch 1–8로 video-disjoint P0-a 재생성
5. 새 frozen visual foundation으로 GoalStep-only history store 재생성
6. GoalStep-only visual-history K=8 Phase 1을 10 epochs 학습
7. 새 P0-a 대비 paired OOF + 10,000 video-bootstrap 평가

SSH 종료 뒤에도 유지하려면 repo root에서 다음처럼 실행한다.

```bash
tmux new-session -d \
  -s ego_goalstep_ltaaux_phase1 \
  -n pipeline \
  "cd /root/nvme/migration/jihun/EGO_jihun2 && \
   exec bash scripts/step1/goalstep/run_ltaaux_phase1.sh"
```

진행 확인:

```bash
tmux attach -t ego_goalstep_ltaaux_phase1
tail -f outputs/goalstep/runs/z1_end_m1_lobs8_next_action_vna_ep10_ltaaux/logs/pipeline.log
```

Launcher v1은 부분 checkpoint resume 계약이 없다. Direct 또는 Phase 1 run에
부분 산출물만 남아 있으면 처음부터 덮어쓰지 않고 오류로 중단한다.

## 7. UI

기존 GoalStep Experiment Board의 첫 카드
`fixed action_end−1s · 8s · uniform 32f + LTA aux`가 이 pipeline을 표시한다.

- local: `http://127.0.0.1:17867`
- 현재 Cloudflare tunnel:
  `https://parts-sleeve-handbook-bidder.trycloudflare.com`
- dashboard source: `tools/goalstep_experiments_dashboard.py`

UI는 다음 7개 stage를 한 카드에서 순서대로 보여준다.

```text
LTA aux index → LTA feature → Direct 10ep → P0-a
→ Derived store → Phase 1 10ep → Paired OOF evaluation
```

Action 지표는 Top-1, Top-5, Top-10을 우선 표시하며 CMR@5와 Top-15도 함께 보존한다.
Cloudflare quick-tunnel URL은 재시작 시 달라질 수 있으므로 local port와 tmux dashboard
세션을 최종 정본으로 삼는다.

## 8. 이번 실행 범위

포함:

- A1 both-match LTA feature 추출
- joint direct probe
- 새 P0-a
- GoalStep-only history Phase 1
- Phase 1 vs P0-a paired OOF/bootstrap 판정

**Phase 2 history probe zoo와 Phase 2 champion 재선정은 이번 실행 범위 밖이다.**
Launcher도 Phase 1 paired 평가가 끝나면 명시적으로 종료한다. Phase 2는 Phase 1
결과와 direct/P0-a 변화가 확인된 뒤 별도 승인·별도 run suffix로 진행한다.

## 9. 한계

1. LTA action 구간이 심하게 겹쳐 A3는 단순 `action_idx+1`이 아니라 첫 strict-later
   action이다. 이 때문에 건너뛴 annotation 수와 target rank가 가변적이다.
2. LTA train과 LTA val annotation을 모두 aux train으로 사용한다. 대신 GoalStep 전체
   val 130개 비디오는 fail-closed로 제외했으며, GoalStep val 평가는 aux에 사용하지
   않는다.
3. exact taxonomy 매칭은 보수적이며 semicolon synonym을 확장하지 않는다. 커버리지가
   낮아질 수 있지만 수동 검수 없는 오매칭을 줄인다.
4. action aux mask가 예상보다 훨씬 많아 V/N diversity 효과와 action supervision 효과가
   한 arm에 함께 들어간다. 후속 해석에는 V/N-only ablation이 필요할 수 있다.
5. Phase 1 history는 GoalStep annotation의 oracle boundary/level을 사용하고 LTA history
   chain은 만들지 않는다.
6. 기존 실험과 마찬가지로 full validation adaptivity와 untouched test 부재 한계를
   상속한다.
7. UI와 tmux는 실행 지속성과 관측을 제공할 뿐, 실패한 partial run의 자동 resume를
   제공하지 않는다.

## 10. 결과 pending

문서 작성 시점에는 최종 성능이 확정되지 않았다. 다음 artifact가 생성되어야 완료다.

```text
outputs/goalstep/runs/z1_end_m1_lobs8_next_action_vna_ep10_ltaaux/final_metrics.json
outputs/goalstep/runs/history_context_phase0_ltaaux/p0a_primary_same_decision_results.json
outputs/goalstep/runs/z1_history_context_k8_vna_ep10_ltaaux/final_metrics.json
outputs/goalstep/runs/z1_history_context_k8_vna_ep10_ltaaux/history_context_vs_p0a_results.json
```

완료 보고에서 최소한 아래를 채운다.

| 단계 | Action Top-1 | Top-5 | Top-10 | 비고 |
|---|---:|---:|---:|---|
| joint direct best | pending | pending | pending | full GoalStep val |
| new P0-a | pending | pending | pending | video-disjoint OOF |
| Phase 1 fused/blend | pending | pending | pending | paired OOF |

최종 Phase 1 승격 규칙은 기존과 동일하다:
`Action Top-5 Δ > 0`이면서 video-bootstrap 95% CI 하한도 `> 0`.
