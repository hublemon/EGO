# V-JEPA2 시간축 교차검증 & 대응 옵션 핸드오프

- 작성일: 2026-07-22
- 실시간 UI (상세 근거·다이어그램): https://claude.ai/code/artifact/42aed0e3-4cd7-4b23-9702-67ee50c76860
- 선행 문서: [2026-07-22_step1-step2-temporal-semantics-risk-report.md](2026-07-22_step1-step2-temporal-semantics-risk-report.md), [2026-07-21_goalstep-action-end-m1-lobs8-vna-results.md](2026-07-21_goalstep-action-end-m1-lobs8-vna-results.md)

## 핵심 결론 3줄

1. **공식 V-JEPA2 EK100 코드는 end−1s에서 평가한다** — val 기본 `ap=0.0` → anchor = `stop_frame − 1s` (action 내부). 공개 R@5 39.7은 이 세팅의 숫자. GitHub main 원본과 로컬 사본 대조로 확인.
2. **논문 부록의 ap 정의와 코드 구현이 정확히 반대** (부록: ap=0 → 첫 프레임 / 코드: ap=0 → 마지막 프레임). 공개 체크포인트는 ap=0.0에서만 벤치마크 수준을 재현(joint Top-5 84% vs ap=1.0에서 48%, 5월 n=50) → 배포된 모델 자체가 end-anchor 분포로 학습된 물건.
3. **Ego4D(GoalStep) end−1s의 성능 점프는 시간 설정이 지배적 원인** — val 샘플 61.4%에서 8초 관찰창 전체가 target 내부. 단 endpoint 단독 기여는 통제 ablation 필요 (8s 관찰·probe depth가 함께 바뀜).

## 외부 확증 (2026-07-22 확인)

- **[facebookresearch/vjepa2 issue #173](https://github.com/facebookresearch/vjepa2/issues/173)** (2026-07-20, KirillRed): 제3자가 독립적으로 동일 발견 — "the sampled clip ends 1 second before the action **end**, rather than … start", 해결책으로 `val_anticipation_point=[1.0,1.0]` 제시. **Meta 무응답 (comments: 0).**
- 커밋 이력 (GitHub API): `epickitchens.py`는 Initial commit(2025-06-09) + 오타 수정 1건이 전부 — 공식 수정 없음, 공개 후 13개월간 공론화 없음.
- 공개 저장소의 **어떤 EK100 config에도 `val_anticipation_point` 키가 없음** → 릴리즈 코드 실행 시 예외 없이 기본값 `[0.0, 0.0]`(end−1s) 적용.
- 참고: 5월에 이미 내부 발견·문서화됨 — `EGO/INTERFACE_FOR_WM.md §5` "ap=0.0은 엄밀히는 anticipation이 아니라 late-action recognition. 외부 보고 시 명시 필요" (경고가 후속 개발에서 유실).

## 핵심 코드 증거

```python
# evals/action_anticipation_frozen/epickitchens.py:127-134 (GitHub main과 일치)
at = random.uniform(*self.anticipation_time)   # val: 1.0s
ap = random.uniform(*self.anticipation_point)  # val 기본값: [0.0, 0.0]
af = int(sf * ap + (1 - ap) * ef - aframes)    # ap=0 → af = stop_frame − 1s ← 핵심
indices = np.arange(af - nframes, af, fstp)    # anchor 이전 4초(32f@8fps) 클립

# eval.py:96
val_anticipation_point = args_data.get("val_anticipation_point", [0.0, 0.0])
```

부록 정의대로면 `af = sf·(1−ap) + ef·ap − aframes` 여야 함 → **lerp 계수 스왑이 가장 유력한 원인** (한 줄짜리 보간 인자 순서 실수, 상세 가설 H1–H5는 UI 참조).

## Leakage 정량 (직접 재계산)

| | EK100 (릴리즈 프로토콜, val 9,668) | GoalStep end−1s (val 7,214) |
|---|---:|---:|
| 액션 길이 중앙값 | 1.96s | 12.6s |
| anchor가 action 내부 | 82.8% | 99.9% |
| 클립 내 target 노출 중앙값 | 0.96s / 4s | **8s / 8s (전부)** |
| 클립 전체가 target 내부 | 19.0% | **61.4%** |
| 진짜 anticipation 샘플 (길이 ≤1s) | 17.2% | ~0.1% |

→ EK100은 "이미 시작된 행동의 도입부를 보고 정체를 맞추는" partial-observation recognition + anticipation(17%) 혼합. 같은 규칙을 액션이 긴 GoalStep에 이식하면 거의 순수 recognition으로 넘어감 — Ego4D에서 효과가 극적으로 증폭된 이유.

## Figure 18 재해석 — "anticipation time이 길수록 성능 하락"의 실체

논문 부록 Figure 18(Left)은 anticipation time 1s→2s→4s→10s 스윕에서 "performance sharply decreases … forecasting the future is non-deterministic"이라고 해석한다. 그러나 **코드에서 두 해석(미래 지평 증가 vs end에서 멀어짐)은 같은 조작**이다 — val `ap=0`이므로 anchor = `end − τ`. 이 스윕은 세 가지를 동시에 바꾼다:

1. **진짜 예측 지평 증가** — predictor가 `anticipation_times`를 조건으로 τ초 뒤 표현 예측 (논문이 주장하는 유일한 요인)
2. **target leakage 제거** — τ가 커질수록 관찰창이 target 밖으로 이탈 (아래 표)
3. **시간 조건 OOD** — train `at∈[0.25,1.75]`뿐이라 τ=2/4/10s는 전부 학습 분포 밖

| τ | target 보임 | 노출 중앙값 | 진짜 anticipation (len≤τ) | 시간 조건 |
|---:|---:|---:|---:|---|
| 1.0s | 82.8% | 0.96s | 17.2% | in-dist |
| 1.5s | 62.5% | 0.46s | 37.5% | in-dist |
| 2.0s | 48.7% | 0.00s | 51.3% | **OOD** |
| 4.0s | 24.2% | 0.00s | 75.8% | **OOD** |
| 10.0s | 7.1% | 0.00s | 92.9% | **OOD** |

(EK100 val 9,668, annotation 재계산)

**결론**: 급락의 상당 부분이 ②+③으로 설명될 개연성이 높고, 순수 지평 효과(①)의 크기는 이 세팅에서 측정된 적이 없다. Figure 18은 그림뿐이라 τ별 수치가 본문에 없음 — 정량 분해는 우리 τ 스윕으로만 가능.

### val anchor는 조정 가능한가 — 가능, config 한 줄

```yaml
experiment:
  data:
    val_anticipation_point: [1.0, 1.0]   # anchor = start − at (strict). issue #173 제안과 동일
```

단 **eval만 바꾸면 반쪽**: 릴리즈 체크포인트는 end-anchor 분포(train ap∈[0,0.25])로 학습돼 val만 옮기면 OOD 평가(5월 실측 84%→48%). anchor 조정은 train 분포와 세트(=옵션 5). 우리 GoalStep 파이프라인은 자체 인덱스 빌더(`build_goalstep_endpoint_index.py --tau-a --l-obs`)로 **이미 anchor 완전 통제 중**.

## 성능 비교와 교란변수 분해 (GoalStep full-val Action Top-5)

| run | 설정 | Top-5 |
|---|---|---:|
| b2_vna | start−1s · 3.875s@8fps · depth1 · VNA | 20.68 |
| b3_d4 | start−1s · 3.875s@8fps · depth4 · A-only | 24.54 |
| z1_end_m1_lobs8_vna | end−1s · 8s@4fps · depth4 · VNA | 47.44 (ep6 시점 50.04) |

분해: depth 1→4 ≈ **+3.9pp** (start−1s 통제) / 나머지 **+22.9pp** = {endpoint, 8s 창, 4fps} 미분리 → **endpoint-only ablation 필요** (canonical ↔ end_m1 index만 교체, l_obs·fps·depth·seed 고정 2-run).

## 파이프라인 시간 계약 지도

| 구성요소 | anchor | GT | 실질 task | 비고 |
|---|---|---|---|---|
| Step 1 EK100 probe (`video_sampling.py`) | start−τ (클램프) | 동일 annotation(미래) | strict anticipation | 논문 텍스트·챌린지 표준과 정합 |
| Step 1 GoalStep canonical | start−1s | 〃 | strict anticipation | 〃 |
| Step 1 GoalStep end−1s (`build_goalstep_endpoint_index.py`) | end−1s, 8s | 동일(진행 중) | late-action recognition | 릴리즈 V-JEPA2 프로토콜 재현 |
| Step 2 Pro (`select_train.py` 등) | trigger=stop−1s | 동일 row | ongoing-action reranking | 5월에 ap=0.0 의도적 채택. 프롬프트 "NEXT"와 충돌 |
| Step 2 Retro (`build_dpo_dataset.py` 등) | 동일 trigger | chosen=현재 action | 〃 | Pro와 내부 정합 |

**불일치는 Step 1 ↔ Step 2 사이**에 있음 (Pro/Retro는 서로, 그리고 릴리즈 V-JEPA2와 정합).

## 논문 초안(EGO) 영향 요약

- **시간축과 무관하게 안전**: 모듈러 정렬 메커니즘("first to use WM distribution as alignment signal"), GT-free Prospection reward (Eq. 5–6), support/selection 분해(Coverage/SelAcc/GADR), Retrospection 구조
- **수정 필요**: Preliminary "next-action candidates" / Eq.3의 causal context (target이 x≤t 안에 있음) / Retrospection a^GT의 시간 의미 / Coverage@K 해석(인플레이트) / "NEXT" 프롬프트 / Step1↔2 probe 계약 통일
- **취약**: planning 확장 — LLM procedural prior와 기여 분리 불가, ablation 필수

## 대응 경로 & 실행 옵션

### 프레이밍 경로 (회의 3안과 매핑)

- **A. Strict 재정렬 (start−1s 전면 이행)** ← 회의 1안. 논문 무수정, planning 안전. 비용: coverage 급락·전 파이프라인 재실행
- **B. τ-occupancy 재정의** ← 회의 2안의 정식화. `q_WM(a_{t+τ}|x≤t)` = "t+τ에 수행 중일 action" → end−1s가 τ=1s occupancy의 정확한 구현. 의무: persistence baseline + boundary-crossing subset 분리 보고
- **C. Decision-time 스윕 (추천)**: start−1s → 내부 → end−1s 스윕으로 Coverage@K 곡선(WM boundary 정보량) + SelAcc/GADR(LM 활용도)을 함께 그림 → 제목 명제 "WM의 한계 = LM의 한계"가 **측정된 정리**가 됨. endpoint index 빌더가 이미 스윕용 인프라

### 실행 옵션 4–8

| # | 옵션 | 비용 | 핵심 |
|---|---|---|---|
| 4 | **후보 구성 재설계** (verb×noun cross-product, K↑, continue 후보) | 낮음 | 5월 실측: action head 24% → cross-product 84% (end−1s) / start−1s에서도 48%. **"coverage 20%" 전제가 깨짐. 모든 경로의 선행 조건** |
| 5 | **ap 커리큘럼 재학습** (train ap∈[0,1] → eval strict) | 중간 | 48%는 학습 분포 밖 페널티 → in-distribution화로 회복 여지. 8s 관찰도 strict에 이식 |
| 6 | **이중 헤드 WM** (state head + anticipation head) | 중간 | recognition 강점을 state estimation으로 정당 배치. c_t에 â_current 추가 |
| 7 | **Ego4D LTA 프로토콜** (이전 segment 끝에서 multi-step 예측) | 중간 | boundary-anchored라 leakage 구조적 불가. `configs/step1/ego4d_lta` 이미 존재. planning 섹션과 자연 결합 |
| 8 | **GT만 다음 annotation(j+1)으로** (WM 현행 유지) | 낮음 | 재학습 없이 진짜 next-action 출력. "recognition-grounded anticipation". 옵션 4와 결합 필수 |
| 9 | **τ-중간점 타협** (호건 제안): anchor = end−τ (τ>1s) + 관찰창 연장(8→16s) — target도 보되 이전 맥락 비율↑ | 낮음 | 인프라 즉시 가용(`--tau-a --l-obs`). 단 아래 정량표와 한계 3가지 확인 |

### 옵션 9 정량 근거 (GoalStep val 7,214, anchor = end−τ)

| 설정 | target 보임 | 노출 중앙값 | 이전 맥락 중앙값 | 맥락 비율(중앙값) | 진짜 anticipation |
|---|---:|---:|---:|---:|---:|
| τ=1s · 8s (현행) | 99.9% | 8.00s | 0.00s | 0% | 0.1% |
| τ=4s · 8s | 83.4% | 8.00s | 0.00s | 0% | 16.6% |
| τ=6s · 8s | 74.3% | 6.57s | 1.43s | 17.9% | 25.7% |
| τ=8s · 8s | 65.3% | 4.57s | 3.43s | 42.9% | 34.7% |
| τ=4s · 16s | 83.4% | 8.57s | 7.43s | 46.4% | 16.6% |
| τ=6s · 16s | 74.3% | 6.57s | 9.43s | 58.9% | 25.7% |
| τ=8s · 16s | 65.3% | 4.57s | 11.43s | 71.4% | 34.7% |

**옵션 9의 정직한 한계 3가지**:
1. GoalStep은 액션이 길어 **τ≤4s에서는 창 연장 없이 아무것도 안 변한다** (중앙값 창은 여전히 target 100%). 맥락을 만들려면 τ≥6s 또는 l_obs=16s 필요 — 16s는 32프레임 기준 2fps 해상도 비용.
2. target이 보이는 한(65–83%) GT=현재 annotation이라 recognition 성분이 남는다. **"action prediction" 명칭은 τ-occupancy 정의(경로 B) + leakage 표 공개 + persistence baseline 동반 시에만 방어 가능.**
3. 이 knob의 성능 급락 곡선은 V-JEPA2 Fig 18이 이미 보여줬다. 옵션 9는 **경로 C 스윕의 한 점을 고르는 일** — 스윕 전체를 돌린 뒤 곡선 위에서 고르는 것이 한 점만 찍는 것보다 방어력이 높다.

**조합 가이드**: 4 무조건 먼저 → planning 지킬 거면 5 or 7 → recognition 강점까지 살리면 6 → 시간 없으면 8 → 절충이 필요하면 9를 단독이 아니라 C의 스윕으로. 어떤 조합이든 C를 얹으면 제목 명제의 실증 곡선.

## 어떤 경로에서든 필수 조치

1. Method에 decision time 명시 + 데이터로더 assertion: `max(observed_time) < target_start` (anticipation) / target을 τ로 명시 (occupancy)
2. 프롬프트–supervision 정합화: end−1s 계열에서 "NEXT" 제거
3. Step 1·Step 2 probe 계약 통일 (논문의 "the world model"은 한 물건이어야 함)
4. Planning 귀속 ablation: LLM-only / +current-recognizer / +true-anticipation-WM
5. V-JEPA2 인용 시 disclosure: "released protocol evaluates at end−τ; we report both contracts"

## 즉시 실행 가능한 액션

- [ ] **endpoint-only ablation** (오늘 계획): canonical ↔ end_m1 index만 교체, l_obs 동일값 재생성, fps·depth·heads·seed·sampler 고정 2-run → +22.9pp 중 endpoint 몫 확정
- [ ] 옵션 4 선행 측정: start−1s probe의 verb×noun cross-product coverage@K (K=5/10/20) — 기존 체크포인트로 추론만 하면 됨
- [ ] **τ 스윕 인덱스 생성** (옵션 9 / 경로 C 공용): `build_goalstep_endpoint_index.py --tau-a {2,4,6,8} --l-obs {8,16}` — 빌더 파라미터만 바꾸면 됨. feature 재추출은 τ당 1회
- [ ] issue #173 팔로우 (Meta 답변 시 논문 인용 근거). 우리 정량 근거(ap=0.0 vs 1.0 실측) 코멘트 보탤지 회의에서 결정
- [ ] 회의: 경로(A/B/C) × 옵션(4–9) 조합 결정

## 출처

- 공식 코드: [epickitchens.py](https://github.com/facebookresearch/vjepa2/blob/main/evals/action_anticipation_frozen/epickitchens.py) · [eval.py](https://github.com/facebookresearch/vjepa2/blob/main/evals/action_anticipation_frozen/eval.py) (로컬 `EGO/src/vjepa2/evals/…`와 일치 확인)
- 논문: [V-JEPA 2, arXiv 2506.09985](https://arxiv.org/abs/2506.09985) §6 · 부록 D.1 · Table 19 / EGO 초안 `EGO_초안_영문.pdf`
- 데이터 재계산: `EGO/src/epic-kitchens-100-annotations/EPIC_100_{train,validation}.csv` · `EGO_jihun2/src/ego/step1_action_anticipation/goalstep/index_end_m1_lobs8/` parquet
- 5월 기록: `EGO/INTERFACE_FOR_WM.md §5` · `EGO/docs/RESULTS.md §2`
- 결과: [2026-07-21_step1_night_and_retro_belief_sum_handoff.md](2026-07-21_step1_night_and_retro_belief_sum_handoff.md) (b2_vna 20.68 · b3_d4 24.54) · [2026-07-21_goalstep-action-end-m1-lobs8-vna-results.md](2026-07-21_goalstep-action-end-m1-lobs8-vna-results.md) (47.44 / ep6 50.04)
