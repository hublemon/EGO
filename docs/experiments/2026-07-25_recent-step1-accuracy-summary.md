# 최근 Step1 학습 결과: Accuracy Top-1/5/10 요약

- 작성일: 2026-07-25 UTC
- 범위: 2026-07-24에 완료된 GoalStep Step1 학습과 직접 연결되는 visual baseline
- 주 지표: instance accuracy Top-1, Top-5, Top-10
- 단위: `%` (percentage point)

## 1. 결론

최근 결과 중 Action accuracy가 가장 높은 단일 full-validation 결과는
`adaptive transition + history K=8`의 epoch 5 fused 출력이다.

| 평가군 | 결과 | Epoch | Val n | Action Top-1 | Action Top-5 | Action Top-10 |
|---|---|---:|---:|---:|---:|---:|
| Adaptive transition | Visual baseline | 4 | 4,458 | 9.40 | 26.74 | 39.59 |
| Adaptive transition | **History K=8 fused** | **5** | **4,458** | **10.63** | **30.55** | **44.50** |
| LTA-aux strict next-action | Direct visual | 4 | 6,960 | 8.25 | 25.43 | 37.27 |
| LTA-aux strict next-action | History K=8 fused, exploratory | 6 | 6,960 | 10.03 | 29.78 | 43.16 |
| LTA-aux strict next-action | **Cross-fitted final blend** | fold별 선택 | 6,960 | **10.69** | **30.07** | **43.48** |

두 평가군은 대상 sample 수와 transition 구성 자체가 다르므로
`30.55 > 30.07`을 동일 조건의 모델 우열로 해석하면 안 된다.

## 2. Adaptive transition 계열

### 2.1 Visual baseline

- Run: `outputs/goalstep/runs/z1_adaptive_transition_mr24x8_vna_ep10/`
- Best checkpoint: `best_action_top5.pt` → epoch 4
- Full validation: 4,458 samples

| Head | Top-1 | Top-5 | Top-10 |
|---|---:|---:|---:|
| Verb | 22.88 | 52.94 | 69.49 |
| Noun | 28.22 | 55.07 | 67.05 |
| Action | 9.40 | 26.74 | 39.59 |

### 2.2 History K=8

- Run: `outputs/goalstep/runs/z1_adaptive_transition_history_context_k8_vna_ep10/`
- Best exploratory checkpoint: `best_action_top5.pt` → epoch 5
- Full validation: visual baseline과 동일한 4,458 samples

| Head | Top-1 | Top-5 | Top-10 |
|---|---:|---:|---:|
| Verb | 24.34 | 55.79 | 74.09 |
| Noun | 29.03 | 57.00 | 69.49 |
| Action | **10.63** | **30.55** | **44.50** |

Action 기준 증분:

| 비교 | Δ Top-1 | Δ Top-5 | Δ Top-10 |
|---|---:|---:|---:|
| Fused − visual baseline | +1.23 | +3.81 | +4.91 |
| Fused − same-epoch current-only | +0.45 | +1.23 | +1.41 |

이 결과는 full-validation에서 epoch를 선택한 exploratory ablation이다.
OOF 또는 독립 test 승격 결과로 해석하지 않는다.

## 3. LTA-aux strict next-action 계열

공통 계약은 `A2.end−1s → strict same-level A3`이며 full validation은
6,960 samples다.

### 3.1 Direct visual

- Run: `outputs/goalstep/runs/z1_end_m1_lobs8_next_action_vna_ep10_ltaaux/`
- Best checkpoint: `best_action_top5.pt` → epoch 4

| Head | Top-1 | Top-5 | Top-10 |
|---|---:|---:|---:|
| Verb | 21.06 | 53.41 | 68.76 |
| Noun | 28.36 | 53.02 | 65.39 |
| Action | 8.25 | 25.43 | 37.27 |

### 3.2 History K=8 exploratory checkpoint

- Run: `outputs/goalstep/runs/z1_history_context_k8_vna_ep10_ltaaux/`
- Best exploratory checkpoint: `best_action_top5.pt` → epoch 6

| Head | Top-1 | Top-5 | Top-10 |
|---|---:|---:|---:|
| Verb | 22.92 | 56.91 | 72.80 |
| Noun | 30.11 | 57.08 | 69.51 |
| Action | 10.03 | 29.78 | 43.16 |

Action 기준 same-epoch current-only 대비 fused 증분은
Top-1 `+0.65pp`, Top-5 `+2.37pp`, Top-10 `+3.91pp`다.

### 3.3 Cross-fitted final blend

체크포인트 epoch와 P0-a 혼합 비율을 video-disjoint 반대 fold에서 선택한
결과이므로, LTA-aux history 실험의 주 판정에는 아래 수치를 사용한다.

| Head | Top-1 | Top-5 | Top-10 |
|---|---:|---:|---:|
| Verb | 23.16 | 57.31 | 73.45 |
| Noun | 30.45 | 57.37 | 69.81 |
| Action | **10.69** | **30.07** | **43.48** |

Action Top-5는 cross-fitted P0-a `28.72%` 대비 `+1.35pp`이며,
video-bootstrap 95% CI lower bound도 0보다 커 provisional engineering
adoption 조건을 통과했다. 독립 test를 사용한 confirmatory 결과는 아니다.

## 4. 해석

1. 두 cohort 모두 history 결합 후 Action Top-1/5/10이 함께 상승했다.
2. 증분은 Top-1보다 Top-5와 Top-10에서 더 크다. History가 정답 한 개를
   즉시 1위로 올리는 효과보다 후보 순위를 상위권으로 끌어올리는 효과가 강하다.
3. Adaptive transition 결과의 절대 accuracy가 가장 높지만 4,458-sample
   cohort라서 6,960-sample LTA-aux 결과와 직접 순위를 매기지 않는다.
4. 현재 보고용 대표값은 검증 절차를 고려해 LTA-aux cross-fitted final
   blend의 Action `10.69 / 30.07 / 43.48`을 우선 사용한다.

## 5. 출처

- Direct LTA-aux:
  `outputs/goalstep/runs/z1_end_m1_lobs8_next_action_vna_ep10_ltaaux/final_metrics.json`
- LTA-aux history exploratory:
  `outputs/goalstep/runs/z1_history_context_k8_vna_ep10_ltaaux/final_metrics.json`
- LTA-aux cross-fitted evaluation:
  `outputs/goalstep/runs/z1_history_context_k8_vna_ep10_ltaaux/history_context_vs_p0a_results.json`
- Adaptive transition visual:
  `outputs/goalstep/runs/z1_adaptive_transition_mr24x8_vna_ep10/final_metrics.json`
- Adaptive transition history:
  `outputs/goalstep/runs/z1_adaptive_transition_history_context_k8_vna_ep10/final_metrics.json`
