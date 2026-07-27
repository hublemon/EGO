# VPA v2 Action-History Ablation 종합 결과 — T3/T4 × full/no-history

## ★ 주요 수치 한눈에

값은 모두 `%`다. **굵게**는 각 조건의 6개 비교 arm 중 최고 점추정이다.
`†`가 붙은 WM baseline은 action-history 텍스트를 입력으로 사용하지 않으므로 full/no-history에서
수치가 동일하다.

| Horizon | Action history | arm | SR | mAcc | mIoU |
|---|---|---|---:|---:|---:|
| T3 | full | `ours_wm1st` | 9.40 | **21.38** | **31.55** |
| T3 | full | `ours_full` | **12.46** | 20.62 | 30.82 |
| T3 | full | `qwen_backbone` | 11.58 | 17.49 | 25.99 |
| T3 | full | `frontier` | 3.06 | 11.55 | 28.09 |
| T3 | full | `wm_top1_repeat`† | 0.98 | 15.34 | 14.92 |
| T3 | full | `wm_topk_rank`† | 0.00 | 12.60 | 18.65 |
| T3 | none | `ours_wm1st` | 0.11 | 12.17 | **22.19** |
| T3 | none | `ours_full` | **4.70** | 11.04 | 18.12 |
| T3 | none | `qwen_backbone` | 4.15 | 9.33 | 17.63 |
| T3 | none | `frontier` | 0.22 | 8.05 | 17.32 |
| T3 | none | `wm_top1_repeat`† | 0.98 | **15.34** | 14.92 |
| T3 | none | `wm_topk_rank`† | 0.00 | 12.60 | 18.65 |
| T4 | full | `ours_wm1st` | 10.32 | 21.53 | **38.77** |
| T4 | full | `ours_full` | **16.07** | **24.06** | 37.48 |
| T4 | full | `qwen_backbone` | 15.08 | 21.28 | 31.00 |
| T4 | full | `frontier` | 3.77 | 13.19 | 35.38 |
| T4 | full | `wm_top1_repeat`† | 0.60 | 15.62 | 15.18 |
| T4 | full | `wm_topk_rank`† | 0.00 | 11.81 | 20.66 |
| T4 | none | `ours_wm1st` | 0.00 | 10.86 | **23.36** |
| T4 | none | `ours_full` | **9.92** | **16.02** | 23.22 |
| T4 | none | `qwen_backbone` | 2.78 | 12.25 | 20.10 |
| T4 | none | `frontier` | 0.00 | 9.08 | 20.80 |
| T4 | none | `wm_top1_repeat`† | 0.60 | 15.62 | 15.18 |
| T4 | none | `wm_topk_rank`† | 0.00 | 11.81 | 20.66 |

표본은 T3 **915개/71영상**, T4 **504개/54영상**이다. 모든 신규 arm은 100% prediction
coverage이고 `reportable=true`다. 같은 horizon의 full/no-history는 동일한 sample ID와 cached
frame을 사용하도록 설계·검증했으며, 생성형 arm에는 같은 4초 관측창의 8개 video frame이
들어간다. 두 WM baseline은 raw frame 대신 Step1 WM 출력만 사용한다. `none` 조건에서
생성형 arm으로부터 제거한 것은 **완료된 action-history 텍스트뿐**이며, 나머지 method별 입력은
유지했다. WM 제약은 `ours_wm1st`에만 해당한다. 단, 과거 T3 full raw artifact가 없어 그 조건의
sample/frame byte hash를 지금 다시 대조할 수는 없고, 동일성은 당시 공식 문서의 계약에
근거한다.

> **주 지표는 mAcc와 mIoU다.** SR은 완전한 순서 일치만 인정하며, 특히 T3 full-history에서는
> 전체 성공의 95%가 상위 3개 영상에 집중된다. SR 점추정만으로 방법의 우열을 해석하면 안 된다.

---

## 0. 결론

1. **Action history는 모든 생성형 arm에 매우 강한 신호다.** 동일한 504개 T4 표본을 직접
   짝비교하면 history 제공으로 mAcc가 **+4.12~+10.66pp**, mIoU가 **+10.90~+15.41pp**
   증가한다. 네 생성형 arm 모두 두 지표의 95% CI가 0을 제외한다.
2. **“History가 없으면 우리 모델의 상대적 강점이 더 커질 것”이라는 사전 가설은 지지되지
   않는다.** `ours_full − qwen_backbone`은 full-history에서 T3와 T4 모두 mAcc·mIoU가
   유의하지만, no-history에서는 두 horizon 모두 유의하지 않다.
3. **EGO 학습 이득은 full-history 조건에서 가장 견고하다.** T3 full에서 백본 대비
   mAcc **+3.13pp**, mIoU **+4.84pp**, T4 full에서 **+2.78pp**, **+6.48pp**로 모두
   유의하다. No-history에서도 점추정은 앞서지만 불확실성이 커서 견고한 우위로 볼 수 없다.
4. **WM 1-step 힌트는 history 부재를 안정적으로 보상하지 못한다.**
   `ours_wm1st − ours_full`은 네 조건의 모든 지표에서 유의한 개선이 없다. T4 no-history의
   mAcc 점추정은 오히려 −5.16pp다.
5. **`ours_full`이 frontier를 이긴다고 주장할 수 없다.** 네 조건 모두 `ours_full − frontier`
   짝비교 CI가 0을 포함한다. 점추정상 SR·mAcc는 대체로 높지만 통계적 우위는 아니다.
6. **T3와 T4 수치를 horizon 효과로 직접 비교하면 안 된다.** T4는 T3와 다른 504개/54영상
   부분집합이다. T4 점수가 더 높아 보이는 것은 “더 긴 계획이 쉽다”는 뜻이 아니다.

---

## 1. 실험 및 평가 계약

| 항목 | T3 | T4 |
|---|---:|---:|
| 표본 수 | 915 | 504 |
| video cluster 수 | 71 | 54 |
| split | heldout | heldout |
| 후보 어휘 | 293 | 293 |
| prediction coverage | 100% | 100% |

| 입력/평가 항목 | 계약 |
|---|---|
| 관측창 | `[target_start − 5s, target_start − 1s]`, 총 4초 |
| 생성형 arm의 프레임 | 8장, 2fps, short side 336px |
| 미래 오염 방지 | target action 시작 1초 전에 관측 종료 |
| full-history | 최근 완료 행동 최대 15개를 oldest→newest 텍스트로 제공 |
| no-history | 완료 행동 텍스트와 그에 대한 system clause만 제거 |
| 유지된 입력 | video frame, goal, 293개 후보, horizon, `ours_wm1st`의 WM 제약 |
| 모델 백본 | `Qwen/Qwen3-VL-8B-Instruct` |
| EGO arms | `ours_full`, `ours_wm1st`: 위 백본 + step2 `sft_r15` LoRA |
| frontier | `gemini-2.5-pro` |
| SR | T개 action의 순서까지 완전 일치한 표본 비율 |
| mAcc | 위치별 action 정확도의 평균 |
| mIoU | 순서를 무시한 예측/정답 action 집합 IoU의 평균 |
| arm별 CI | video-cluster bootstrap, 1,000회 |
| arm 간 차이 CI | 같은 표본의 video-cluster paired bootstrap, 2,000회 |

`wm_top1_repeat`와 `wm_topk_rank`는 Step1 WM 출력만 사용하는 baseline이다. 두 baseline은
VLM prompt의 action-history 텍스트를 읽지 않기 때문에 같은 horizon에서는 full/no-history
예측과 점수가 정확히 같다.

---

## 2. 조건별 상세 결과

### 2-1. T3 full-history

915개 표본, 71개 영상. 값은 점추정과 video-cluster bootstrap 95% CI다.

| arm | SR | mAcc | mIoU |
|---|---:|---:|---:|
| `ours_wm1st` | 9.40 [0.8, 23.5] | **21.38** [13.2, 32.7] | **31.55** [19.3, 48.2] |
| `ours_full` | **12.46** [1.1, 31.7] | 20.62 [9.8, 37.9] | 30.82 [17.8, 48.1] |
| `qwen_backbone` | 11.58 [0.4, 30.4] | 17.49 [6.5, 35.1] | 25.99 [13.2, 44.8] |
| `frontier` | 3.06 [0.4, 6.1] | 11.55 [8.4, 14.4] | 28.09 [15.0, 45.0] |
| `wm_top1_repeat` | 0.98 [0.2, 2.0] | 15.34 [10.3, 21.3] | 14.92 [10.0, 20.7] |
| `wm_topk_rank` | 0.00 [0.0, 0.0] | 12.60 [9.9, 15.4] | 18.65 [15.0, 22.4] |

같은 표본에서의 직접 짝비교:

| 비교 A − B | ΔSR [95% CI] | ΔmAcc [95% CI] | ΔmIoU [95% CI] | 판정 |
|---|---:|---:|---:|---|
| `ours_full − qwen_backbone` | **+0.87** [+0.23, +1.57] | **+3.13** [+1.33, +5.11] | **+4.84** [+2.29, +7.66] | 3개 모두 유의 |
| `ours_full − frontier` | +9.40 [−0.63, +25.00] | +9.07 [−1.44, +24.29] | +2.73 [−0.66, +6.23] | 모두 비유의 |
| `ours_wm1st − ours_full` | −3.06 [−8.19, +0.77] | +0.77 [−5.25, +5.93] | +0.72 [−1.52, +2.90] | 모두 비유의 |

**해석.** Full-history에서는 EGO step2 학습의 백본 대비 이득이 세 지표에서 모두 확인된다.
특히 주 지표인 mAcc와 mIoU의 CI가 0을 명확히 제외한다. 반면 WM 후보를 첫 스텝에 명시하는
추가 힌트는 유의한 이득이 없고, frontier와의 차이도 불확실하다.

### 2-2. T3 no-history

915개 표본, 71개 영상. Video frame은 T3 full-history와 동일하게 8장 들어갔다.

| arm | SR | mAcc | mIoU |
|---|---:|---:|---:|
| `ours_wm1st` | 0.11 [0.00, 0.35] | 12.17 [8.65, 15.64] | **22.19** [14.14, 33.40] |
| `ours_full` | **4.70** [0.00, 12.91] | 11.04 [4.13, 21.48] | 18.12 [7.76, 34.27] |
| `qwen_backbone` | 4.15 [0.00, 11.62] | 9.33 [3.57, 17.98] | 17.63 [6.91, 35.31] |
| `frontier` | 0.22 [0.00, 0.52] | 8.05 [5.41, 11.31] | 17.32 [8.93, 29.75] |
| `wm_top1_repeat` | 0.98 [0.20, 2.00] | **15.34** [10.28, 21.30] | 14.92 [10.00, 20.71] |
| `wm_topk_rank` | 0.00 [0.00, 0.00] | 12.60 [9.92, 15.37] | 18.65 [15.00, 22.43] |

같은 표본에서의 직접 짝비교:

| 비교 A − B | ΔSR [95% CI] | ΔmAcc [95% CI] | ΔmIoU [95% CI] | 판정 |
|---|---:|---:|---:|---|
| `ours_full − qwen_backbone` | +0.55 [+0.00, +1.31] | +1.71 [+0.00, +3.49] | +0.49 [−1.12, +2.31] | 모두 비유의 |
| `ours_full − frontier` | +4.48 [−0.26, +12.24] | +2.99 [−2.50, +10.34] | +0.81 [−2.60, +4.61] | 모두 비유의 |
| `ours_wm1st − ours_full` | −4.59 [−12.51, +0.00] | +1.13 [−6.25, +7.08] | +4.07 [−0.95, +8.43] | 모두 비유의 |

**해석.** Action history를 빼면 네 생성형 arm이 모두 크게 하락한다. `ours_full`은 백본과
frontier보다 높은 점추정을 유지하지만, 차이는 유의하지 않다. mAcc에서는 단순한
`wm_top1_repeat`가 15.34로 전체 최고다. 이는 no-history 조건에서 생성형 모델이 활용할 수 있는
절차적 단서가 크게 줄어, WM의 전이 prior가 상대적으로 경쟁적이 됐음을 보여준다. 다만 WM
baseline은 원래 action-history prompt를 사용하지 않으므로 이를 “history에 강건한 VLM”으로
해석해서는 안 된다.

### 2-3. T4 full-history

504개 표본, 54개 영상.

| arm | SR | mAcc | mIoU |
|---|---:|---:|---:|
| `ours_wm1st` | 10.32 [0.28, 24.59] | 21.53 [11.50, 32.50] | **38.77** [20.64, 58.27] |
| `ours_full` | **16.07** [0.00, 40.15] | **24.06** [8.24, 45.39] | 37.48 [19.70, 57.24] |
| `qwen_backbone` | 15.08 [0.00, 38.10] | 21.28 [6.31, 42.63] | 31.00 [13.41, 52.79] |
| `frontier` | 3.77 [0.00, 9.29] | 13.19 [8.24, 17.05] | 35.38 [16.22, 55.63] |
| `wm_top1_repeat` | 0.60 [0.00, 1.51] | 15.62 [8.99, 23.23] | 15.18 [8.77, 22.79] |
| `wm_topk_rank` | 0.00 [0.00, 0.00] | 11.81 [8.35, 15.37] | 20.66 [15.91, 25.15] |

같은 표본에서의 직접 짝비교:

| 비교 A − B | ΔSR [95% CI] | ΔmAcc [95% CI] | ΔmIoU [95% CI] | 판정 |
|---|---:|---:|---:|---|
| `ours_full − qwen_backbone` | +0.99 [−0.33, +2.30] | **+2.78** [+1.19, +4.03] | **+6.48** [+3.54, +10.10] | mAcc·mIoU 유의 |
| `ours_full − frontier` | +12.30 [+0.00, +30.88] | +10.86 [−2.96, +29.25] | +2.10 [−1.26, +5.65] | 모두 비유의 |
| `ours_wm1st − ours_full` | −5.75 [−15.56, +1.52] | −2.53 [−13.74, +7.32] | +1.30 [−2.38, +4.90] | 모두 비유의 |

**해석.** T4에서도 full-history의 EGO 어댑터 이득이 재현된다. `ours_full`은 백본보다
mAcc +2.78pp, mIoU +6.48pp 높고 두 차이 모두 유의하다. SR 차이는 유의하지 않으며,
WM 1-step 힌트와 frontier 대비 차이도 유의하지 않다.

### 2-4. T4 no-history

504개 표본, 54개 영상. T4 full-history와 sample ID가 완전히 같은 paired 조건이다.

| arm | SR | mAcc | mIoU |
|---|---:|---:|---:|
| `ours_wm1st` | 0.00 [0.00, 0.00] | 10.86 [7.95, 13.18] | **23.36** [14.81, 32.85] |
| `ours_full` | **9.92** [0.00, 25.22] | **16.02** [3.73, 33.37] | 23.22 [7.32, 45.17] |
| `qwen_backbone` | 2.78 [0.00, 7.21] | 12.25 [3.32, 24.43] | 20.10 [7.38, 37.59] |
| `frontier` | 0.00 [0.00, 0.00] | 9.08 [5.03, 12.96] | 20.80 [8.75, 35.41] |
| `wm_top1_repeat` | 0.60 [0.00, 1.51] | 15.62 [8.99, 23.23] | 15.18 [8.77, 22.79] |
| `wm_topk_rank` | 0.00 [0.00, 0.00] | 11.81 [8.35, 15.37] | 20.66 [15.91, 25.15] |

같은 표본에서의 직접 짝비교:

| 비교 A − B | ΔSR [95% CI] | ΔmAcc [95% CI] | ΔmIoU [95% CI] | 판정 |
|---|---:|---:|---:|---|
| `ours_full − qwen_backbone` | +7.14 [+0.00, +17.92] | +3.77 [+0.00, +8.77] | +3.11 [−0.45, +7.50] | 모두 비유의 |
| `ours_full − frontier` | +9.92 [+0.00, +25.04] | +6.94 [−2.85, +20.37] | +2.42 [−3.92, +9.82] | 모두 비유의 |
| `ours_wm1st − ours_full` | −9.92 [−25.04, +0.00] | −5.16 [−20.77, +6.15] | +0.15 [−12.62, +10.02] | 모두 비유의 |

**해석.** `ours_full`이 SR·mAcc에서 최고이고 `ours_wm1st`가 mIoU에서 근소하게 최고지만,
생성형 arm 사이의 모든 차이는 비유의다. `wm_top1_repeat`의 mAcc 15.62가 `ours_full`
16.02와 거의 같아질 만큼 action-history 제거의 영향이 크다. WM 힌트가 있는
`ours_wm1st`도 mAcc가 10.86으로 하락해, 첫 스텝 후보 힌트만으로 전체 T4 계획의 순서를
복원할 수 없음을 시사한다.

---

## 3. Action-history 효과

### 3-1. T4: 정식 paired 비교

아래 값은 동일한 504개 표본에서 **full-history − no-history**를 직접 계산하고, 같은 video
cluster를 양쪽에서 동시에 재표집한 결과다.

| arm | ΔSR [95% CI] | ΔmAcc [95% CI] | ΔmIoU [95% CI] | 유의한 지표 |
|---|---:|---:|---:|---|
| `ours_wm1st` | **+10.32** [+0.21, +24.59] | **+10.66** [+2.96, +19.70] | **+15.41** [+5.05, +25.50] | SR, mAcc, mIoU |
| `ours_full` | +6.15 [+0.00, +15.08] | **+8.04** [+3.20, +12.70] | **+14.26** [+9.63, +19.73] | mAcc, mIoU |
| `qwen_backbone` | +12.30 [+0.00, +30.68] | **+9.03** [+1.89, +18.47] | **+10.90** [+5.42, +15.73] | mAcc, mIoU |
| `frontier` | +3.77 [+0.00, +9.32] | **+4.12** [+2.02, +6.17] | **+14.58** [+6.90, +20.44] | mAcc, mIoU |

두 WM baseline의 차이는 세 지표 모두 정확히 0이다.

**판정.** Action history는 특정 모델 하나만의 편의 기능이 아니라, 네 생성형 arm 모두의
위치 정확도와 행동 집합 선택에 큰 영향을 준다. SR은 `ours_wm1st`에서만 유의하다. 나머지
arm의 SR CI 하한은 실제로 0.00이며, CI가 0을 포함하므로 비유의다.

### 3-2. T3: 기술통계 비교

T3 full-history의 현재 공식 근거는 최종 handoff의 집계표다. 당시 raw prediction과 metrics
JSON이 현재 서버에 남아 있지 않아, sample-level paired bootstrap을 새로 계산할 수 없다.
따라서 아래는 **집계 점추정의 차이일 뿐 유의성 검정이 아니다.**

| arm | ΔSR | ΔmAcc | ΔmIoU |
|---|---:|---:|---:|
| `ours_wm1st` | ≈+9.29 | ≈+9.21 | ≈+9.36 |
| `ours_full` | ≈+7.76 | ≈+9.58 | ≈+12.70 |
| `qwen_backbone` | ≈+7.43 | ≈+8.16 | ≈+8.36 |
| `frontier` | ≈+2.84 | ≈+3.50 | ≈+10.77 |
| `wm_top1_repeat` | 0.00 | 0.00 | 0.00 |
| `wm_topk_rank` | 0.00 | 0.00 | 0.00 |

방향은 T4와 일관되게 모두 양수다. 그러나 T3에 대해서 “history 효과가 유의하다”거나
“어떤 모델이 history에 더 의존한다”고 이 표만으로 주장할 수는 없다.

### 3-3. 모델별 history 의존도 가설

사전 가설은 “action history를 제거하면 EGO 학습 모델이 백본·frontier보다 상대적으로 덜
나빠질 것”이었다. 현재 결과는 이 가설을 단순 지지하지 않는다.

| Horizon | 조건 | `ours_full − qwen_backbone` ΔmAcc | ΔmIoU | 판정 |
|---|---|---:|---:|---|
| T3 | full | +3.13 [+1.33, +5.11] | +4.84 [+2.29, +7.66] | 둘 다 유의 |
| T3 | none | +1.71 [+0.00, +3.49] | +0.49 [−1.12, +2.31] | 둘 다 비유의 |
| T4 | full | +2.78 [+1.19, +4.03] | +6.48 [+3.54, +10.10] | 둘 다 유의 |
| T4 | none | +3.77 [+0.00, +8.77] | +3.11 [−0.45, +7.50] | 둘 다 비유의 |

- T3에서는 no-history에서 EGO의 백본 대비 점추정 이득 자체가 축소된다.
- T4 mAcc 점추정 격차는 커지지만 CI가 0을 포함한다. mIoU 격차는 축소되고 비유의가 된다.
- T4 full−none mAcc 하락은 `ours_full` +8.04pp, 백본 +9.03pp로 비슷하다.
- Frontier의 mAcc 하락은 +4.12pp로 작지만 mIoU 하락은 +14.58pp로 크다. 한 지표만 보고
  history 의존도를 서열화하면 안 된다.

엄밀한 “우리 모델이 history에 덜 의존한다” 주장은
`(full−none)_ours − (full−none)_comparison`의 **paired difference-in-differences** CI가
필요하다. 이번 산출물에는 DiD가 없으므로, 현재 가능한 결론은 **사전 가설을 지지하는 증거가
확보되지 않았다**까지다.

---

## 4. 종합 해석

### 4-1. Action history는 단순 부가 정보가 아니라 핵심 planning signal이다

No-history 조건에도 goal과 8개 video frame이 그대로 남아 있지만, 모든 생성형 arm의 mAcc와
mIoU가 크게 하락한다. 4초 영상은 현재 시각 상태를 보여주지만, 완료 행동의 누적 텍스트는
현재 절차 위치와 반복 패턴을 더 직접적으로 알려준다. VPA에서는 “무엇이 보이는가”뿐 아니라
“지금까지 무엇을 했는가”가 다음 여러 행동의 순서를 정하는 핵심 정보다.

### 4-2. EGO 어댑터의 이득은 history와 함께 있을 때 더 견고하다

Full-history에서는 T3와 T4 모두 `ours_full`이 동일 백본보다 mAcc·mIoU에서 유의하게 높다.
반면 no-history에서는 점추정 우위는 남지만 유의성이 사라진다. 이는 적어도 현재 모델과
프롬프트에서는 EGO 학습의 계획 이득이 **영상만으로 독립적으로 발현된다기보다, 영상과
history를 함께 해석할 때 가장 안정적으로 나타난다**는 패턴이다.
이는 서술적 패턴이며, history×adapter 상호작용에 대한 정식 결론에는 DiD가 필요하다.

단, no-history 자체가 EGO step2 학습 분포 밖 조건이라는 점이 중요하다. Step2 학습 prompt는
항상 completed-action history를 포함했다. 따라서 no-history 하락을 곧바로 “시각 이해 부족”으로
환원할 수 없다. 입력 modality 제거와 prompt distribution shift가 함께 작용했을 가능성이 있다.

### 4-3. WM 1-step prior는 다중 스텝 history를 대체하지 못한다

`ours_wm1st`는 첫 번째 예측을 WM top-10 중 하나로 제한하지만, 네 조건 모두
`ours_full`보다 유의하게 좋아지지 않는다. 특히 T4 no-history에서 SR −9.92pp, mAcc −5.16pp의
점추정은 첫 스텝 제약이 이후 계획을 잘못 앵커링할 가능성을 시사한다. 다만 CI가 넓어
“WM 힌트가 해롭다”고 단정할 수도 없다.

### 4-4. 이 평가셋의 점추정에서 frontier는 집합 겹침이 상대적으로 강하다

이 평가셋에서 frontier는 mAcc와 SR 점추정보다 mIoU 점추정이 상대적으로 강하다. 예를 들어 T4 full에서
`frontier`의 mIoU는 35.38로 `ours_full` 37.48에 가깝지만 mAcc는 13.19 대 24.06이다.
이는 적절한 행동 **집합**은 고르되, 이 데이터셋의 반복적 행동 순서와 정확한 위치를 덜
맞추는 기존 분석과 일치한다. 어느 조건에서도 `ours_full − frontier`가 유의하지 않으므로
능력의 일반적 우열로 확대하면 안 된다.

### 4-5. No-history에서는 단순 WM baseline이 강한 기준점이 된다

T3 no-history에서 `wm_top1_repeat`는 mAcc 15.34로 6개 arm 중 1위이고, T4 no-history에서도
15.62로 `ours_full` 16.02와 근접한다. 반면 mIoU에서는 생성형 arm이 대체로 WM baseline보다
높다. 즉 `wm_top1_repeat`의 높은 mAcc는 동일 action 반복 prior에서 오며, 높은 집합 겹침으로
그대로 이어지지는 않는다. 한편 `wm_topk_rank`는 mIoU에서 일부 생성형 arm과 근접하므로,
이를 WM 전체의 미래 action-set 구성 능력 부족으로 일반화하지 않는다.

### 4-6. Horizon 간 원시 수치 비교는 금지한다

T4의 504개 표본은 T3의 915개와 동일한 모집단 크기·video 구성이 아니다. 미래 action이 4개
이상 남아 있는 샘플만 T4에 들어가므로 선택 효과가 있다. 따라서 T4의 full-history 점수가
T3보다 높다는 관찰을 horizon 난이도나 성능 향상으로 해석할 수 없다. Horizon 효과를 보려면
T3와 T4의 공통 sample ID만 추려 별도의 paired 분석을 해야 한다.

---

## 5. 논문에서 가능한 주장과 피해야 할 주장

### 주장 가능

- “T4에서 action history 제공은 모든 생성형 arm의 mAcc와 mIoU를 유의하게 개선했다.”
- “Full-history에서 `ours_full`의 EGO step2 어댑터는 동일 Qwen 백본 대비 T3와 T4의 mAcc·mIoU를
  유의하게 개선했다.”
- “No-history에서도 `ours_full`의 점추정은 백본보다 높지만, 차이는 통계적으로 유의하지 않았다.”
- “WM 1-step 후보 힌트는 네 조건에서 일관된 유의 이득을 보이지 않았다.”
- “`ours_full`과 frontier의 차이는 네 조건 모두 유의하지 않았다.”

### 주장 불가

- “우리 모델은 백본이나 frontier보다 action history에 덜 의존한다.”
  — 모델 간 degradation 차이에 대한 DiD가 없다.
- “No-history 실험이 순수한 시각 능력만 측정한다.”
  — EGO 모델에는 학습 prompt 대비 OOD이기도 하다.
- “우리 모델이 frontier를 이겼다.”
  — 모든 직접 짝비교가 비유의다.
- “T4가 T3보다 쉽다/어렵다.”
  — 평가 표본과 cluster 구성이 다르다.
- “SR이 높으므로 전체 계획 능력이 우월하다.”
  — SR은 소수 video cluster에 크게 지배된다.

---

## 6. 재현성 및 산출물

### 공식 결과

| 내용 | 위치 |
|---|---|
| T3 full-history 공식 최종 기록 | [`2026-07-26_vpa_v2_results_handoff.md`](./2026-07-26_vpa_v2_results_handoff.md) |
| 전체 ablation 상태 | [`pipeline_state.json`](../../runs/vpa_v2/action_history_ablation/pipeline_state.json) |
| T3 no-history 요약 | [`T3_nohist/summary.json`](../../runs/vpa_v2/action_history_ablation/T3_nohist/summary.json) |
| T4 full-history 요약 | [`T4_full/summary.json`](../../runs/vpa_v2/action_history_ablation/T4_full/summary.json) |
| T4 no-history 요약 | [`T4_nohist/summary.json`](../../runs/vpa_v2/action_history_ablation/T4_nohist/summary.json) |
| T4 full−none paired 요약 | [`T4_cross_history/summary.json`](../../runs/vpa_v2/action_history_ablation/T4_cross_history/summary.json) |

### 데이터와 방법

| 내용 | 위치 |
|---|---|
| T3 GT | [`vpa_v2_T3.json`](../../runs/vpa_v2/vpa_v2_T3.json) |
| T4 GT | [`vpa_v2_T4.json`](../../runs/vpa_v2/vpa_v2_T4.json) |
| T3 subset | [`frames_subset_T3.json`](../../runs/vpa_v2/frames_subset_T3.json) |
| T4 subset | [`frames_subset_T4.json`](../../runs/vpa_v2/frames_subset_T4.json) |
| 데이터 manifest | [`manifest.json`](../../runs/vpa_v2/manifest.json) |
| 프레임 캐시 | [`frame_cache_w4_g1_n8_s336/`](../../runs/vpa_v2/frame_cache_w4_g1_n8_s336/) |
| 방법 정의 | [`METHODS.md`](../../src/ego/step3_results/vpa/v2/METHODS.md) |
| 프롬프트 정의 | [`PROMPTS.md`](../../src/ego/step3_results/vpa/v2/PROMPTS.md) |
| ablation orchestration | [`run_vpa_action_history_ablation.py`](../../tools/run_vpa_action_history_ablation.py) |

신규 ablation pipeline은 2026-07-26 19:33:45 UTC에 오류 없이 끝났다. 세 신규 phase의 여섯
arm 모두 100% coverage, `returncode=0`, `error=null`이다.

| provenance | 값 |
|---|---|
| pipeline 당시 git HEAD | `8aa03dfe5e50e05d0c4947a85daf5aea05f819c9` — dirty이므로 실행 코드 snapshot 자체는 아님 |
| git 상태 | dirty — history ablation 구현 및 orchestration 변경 포함 |
| base model | `Qwen/Qwen3-VL-8B-Instruct` |
| adapter config SHA-256 | `3249fb68a3cc9afa655b260196401466df242c2d38f1e31f83edc0e6e1de3734` |
| adapter weights SHA-256 | `d33d695d6432e61a849320dce9f98d35c5c615aa9dbd07ae62d428bcc0e1cc9d` |
| frontier alias | `gemini-2.5-pro` |

### T3 full-history provenance 제한

T3 full-history의 최종 aggregate와 당시 paired 결과는 commit된 공식 handoff에 남아 있지만,
그 실행의 raw prediction/metrics JSON은 현재 checkout과 마운트 어디에도 남아 있지 않다.
따라서 다음 제한이 있다.

1. T3 full−no-history의 sample-level paired CI와 DiD를 사후 계산할 수 없다.
2. 기존 T3 full 실행의 adapter 파일 SHA와 frontier backend snapshot을 현재 ablation 실행과
   byte-level로 대조할 수 없다.
3. 과거 T3 full runner는 non-empty prediction을 완료로 인정했고 evaluator가 짧은 출력을
   오답으로 padding했다. 신규 runner는 정확히 T개 label을 요구한다. 지표 정의는 같지만,
   raw record 부재로 과거 short-output 발생 여부를 감사할 수 없다.
4. 동일 T3 sample/frame 사용은 당시 문서의 GT·subset·cache 경로와 915/71 계약으로 확인된다.
   과거 sample ID 목록과 frame hash 자체는 보존되지 않아 byte-level 동일성은 재검증할 수 없다.

문서화된 데이터·프레임·모델·후보·평가 계약과 표본 수는 일치하며, 새 구현은
`history=full` 프롬프트가 기존 프롬프트와 byte-identical인지 fail-closed 검증했다. 그럼에도
T3 history 효과는 본 문서에서 의도적으로 **기술통계**로만 취급한다.

수치가 충돌하는 오래된 계획 문구보다 “전 arm 완주” 상태의
`2026-07-26_vpa_v2_results_handoff.md`를 최종 근거로 사용했다. 특히 T3 full의
`ours_full − frontier` mAcc 차이는 최종 완주값인 **+9.07pp [−1.44, +24.29]**다.
