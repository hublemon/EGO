# History Context + 챌린지 우승 레시피 기반 V-JEPA2 Next-Action Anticipation 구현·학습 계획 Handoff

- 작성일: 2026-07-23 KST (2026-07-22 심야 세션 논의 종합)
- 수신: Step 1 (V-JEPA2 probe) 학습 담당자
- 선행 문서:
  - [2026-07-22_goalstep-end-m1-lobs8-next-action-handoff.md](2026-07-22_goalstep-end-m1-lobs8-next-action-handoff.md) (옵션 8 실측)
  - [2026-07-22_option8_vs_planA_wm_handoff.md](2026-07-22_option8_vs_planA_wm_handoff.md)
  - [2026-07-22_vjepa2-time-contract-audit-handoff.md](2026-07-22_vjepa2-time-contract-audit-handoff.md)
  - [2026-07-22_goalstep-adaptive-transition-window-proposal.md](2026-07-22_goalstep-adaptive-transition-window-proposal.md)

---

## 0. 세 줄 요약

1. **단일 관찰창 계열은 top5 ~26-28에서 수렴 확정** — start−1s/8s(27.52), start−1s/16s(28.40), 옵션 8 next-action(25.70, subset). 창 길이·anchor·GT 계약을 바꿔도 같은 자리다. 남은 상승 축은 **정보를 추가하는 축(history)**과 **분산을 줄이는 축(앙상블)**뿐이다.
2. **History의 정보 존재는 실측으로 증명됐다**: GT 직전 action 라벨 "하나"만으로 next-action top5 30.4 — 시각 probe(25.7)를 이긴다. 단 oracle 수치이므로, **무학습 soft-mixture 게이트(Phase 0)를 통과했을 때만** 학습 투자를 집행한다.
3. **EgoVis 2026 우승팀(JFAA/VISTA)의 레시피는 우리 골격과 동일**(frozen encoder+predictor concat, attentive probe, focal loss, strict start-anchor). 차이는 field-aware 앙상블·probe zoo·백본 스케일이며, 그중 앙상블은 사실상 보장 수준의 이득이라 기본 채택한다. **우승팀도 history 축은 건드리지 않았다** — 우리 차별점으로 성립.

---

## 1. 확정된 실측 근거 (2026-07-22~23 기준)

### 1.1 우리 실험 (GoalStep, V-JEPA2 ViT-L frozen probe)

| 런 | 계약 | val | top5 | top10 | top15 | 판정 |
|---|---|---|---:|---:|---:|---|
| `z1_end_m1_lobs8_vna` | end−1s → A2 (관찰=target 내부) | FULL 7214 | **47.44** | — | — | **recognition leakage** — 3중 검증으로 기각 |
| `z1_start_m1_lobs8_vna` | start−1s → A2 (strict, Plan A) | FULL 7214 | 27.52 | 39.84 | 47.93 | 유효 baseline (best ep4) |
| `z1_start_m1_lobs16_vna` | start−1s, 16s 창 | FULL 7214 | 28.40 | 39.53 | 47.12 | **창 연장 무효** (best ep4) |
| `z1_end_m1_lobs8_next_action_vna_ep10` | end−1s → A3 (옵션 8) | subset 2000 | 25.70 | 37.70 | 46.55 | **관찰 이점 무효** (best ep3, ep9 중단) |
| `z1_adaptive_transition_mr24x8_vna_ep10` | A1 경계 정렬 창 | — | — | — | — | 2026-07-22 16:23 UTC 학습 시작, 결과 대기 |

공통 패턴: **ep3~4 과적합 → best 갱신 없이 종료.** train loss는 0.01대까지 떨어지나 val은 하락.

### 1.2 History 정보량 실측 (annotation만, 시각 0)

next_action index(train 29,293 / val 6,960) 기준, train 전이행렬을 val에 적용:

| 방법 | top5 | top10 | top15 | top20 |
|---|---:|---:|---:|---:|
| 전역 빈도 prior | 12.2 | 21.7 | 30.3 | 37.7 |
| scenario prior | 20.8 | — | 41.4 | — |
| **Markov-1: T(A3\|A2), GT A2 (oracle)** | **30.4** | **41.5** | **48.9** | **53.5** |
| Markov-2 (A1,A2 backoff) | 30.0 | 40.9 | 48.2 | 53.1 |

- persistence(A3=A2) = 10.3%, A2당 후속 분기 중앙값 25개 → 반복 암기가 아닌 **진짜 절차 전이 구조**
- Markov-2가 Markov-1을 못 넘는 것은 raw n-gram 희소성 — 학습형 encoder가 필요한 이유
- **주의: 30.4는 GT A2 oracle 상한.** 실전 회수율은 Phase 0에서 측정

### 1.3 EgoVis 2026 우승팀 레시피 (외부 근거)

- **JFAA** (EK100 AA 1위, [arXiv:2605.20904](https://arxiv.org/abs/2605.20904)): frozen **V-JEPA 2.1 ViT-G/384** encoder+**predictor 롤아웃 concat**, 32f@8fps(4s 창), strict start-anchor(t_s−1s 이후 사용 금지), attentive probe(4블록·16헤드·V/N/A 별도 query), sigmoid focal loss, **LR/WD 다른 probe 20개 동시 학습**, epoch×head 격자에서 **field별(V/N/A) 별도 선발·앙상블**. val action MT5R 39.6 → test 27.95 (1위, 2위와 0.04 차)
- **VISTA** (Ego4D STA 1위, [arXiv:2605.20901](https://arxiv.org/abs/2605.20901)): frozen V-JEPA 2.1을 **글로벌 컨텍스트 토큰 1개**로 요약 → FiLM + ROI 잔차로 검출 경로에 주입
- **우리 골격과의 대조**: [vjepa2_backbone.py:90-127](../../src/ego/step1_action_anticipation/models/vjepa2_backbone.py)이 이미 encoder+predictor concat 구조 — 골격 일치 확인 완료. strict start-anchor가 우승 프로토콜이라는 외부 검증 획득
- **JFAA에 history/시퀀스 컨텍스트 없음** — 단일 창 극한 + 앙상블로 우승. 우리 history 축은 미개척 영역

---

## 2. 계약 (전 Phase 공통 — leakage 0 불변)

```text
target:   next-action A3 (같은 annotation level, A3.start >= A2.end)
관찰:     현재 창 + 과거 창들 — 전부 시각 feature만 (GT 라벨을 입력으로 쓰지 않음)
불변식:   max(관찰 시각) < A3.start   → assert로 강제
```

- history는 **캐시된 시각 feature**로만 공급한다. 라벨 시퀀스 입력은 oracle 문제가 생기므로 금지 (Markov 수치는 정보량 증명용이지 입력 설계가 아님)
- history 창들은 모두 현재 관찰창보다 과거 → 불변식 자동 성립. 추론 시에도 그대로 쓸 수 있는 완전 deployable 구성

---

## 3. 실행 계획 — 게이트 방식 (측정이 앞서고, 학습은 통과 시에만)

### Phase 0 — 무학습 판정 게이트 (GPU 큐와 무충돌, 각 ~1h)

**P0-a. Checkpoint field-aware 앙상블 (버전 A)**
- 기존 per-epoch checkpoint를 전부 활용: next_action 8개(`z1_end_m1_lobs8_next_action_vna_ep10/checkpoints/`), Plan A 10개, lobs16 10개
- 각 checkpoint의 val logits 추출(캐시 feature 추론, ~2-3분/개) → V/N/A **field별로** 후보 선발·가중평균
- val을 선발용/평가용 절반 분할해 선발 과적합 방어
- **기대: +1~3pp. 앙상블이 top-k recall을 해치는 경우는 드묾 — 사실상 보장 축**

**P0-b. Soft-mixture 전이 prior (history go/no-go 판정기)**
```
p(A3) ∝ Σ_A2  p_probe(A2 | obs) · T(A3 | A2)    (+ α·global prior smoothing)
```
- p_probe(A2|obs): 기존 **end−1s recognition** probe(`z1_end_m1_lobs8_vna`, top5 47.4)의 val logits
- T: train 전이행렬 (1.2절 코드 재사용)
- α, temperature는 선발용 절반에서 튜닝
- **판정 (사전 등록): cov_next@5가 25.7(현 probe) 대비 +2pp 이상 → Phase 1 집행. 미달 → history 학습 기각, P0-a 앙상블만 채택**

### Phase 1 — History-fused probe (P0-b 통과 시에만, GPU ~1.5h/런)

**아키텍처 (VISTA 주입 패턴 + 게이트 잔차):**
```
현재 창:  기존 attentive probe 경로 그대로 (비변경)
history:  이전 K=8개 segment의 캐시 feature → 각각 attentive pooling → K개 토큰
          + Δt 임베딩(현재 anchor와의 시간차) + level 임베딩(step/substep)
fusion:   2-layer transformer([현재 pooled; history 토큰들]) → history_logits
출력:     logits = visual_logits + g · history_logits    (g: 학습 스칼라, init 0)
```
- **g init 0 → 하방이 현 성능으로 고정.** history가 무익하면 g≈0 수렴으로 자동 기각
- history dropout p=0.3 (시각 경로 퇴화 방지)
- 영상 초반 segment는 padding+mask, history 길이별 성능 병기

**구현 재료 (전부 존재, feature 재추출 0):**
| 재료 | 위치 | 상태 |
|---|---|---|
| history 체인 구성 | `index_end_m1_lobs8_next_action/` parquet의 `video_uid`/`annotation_level`/`observed_action_start_sec` | groupby+shift로 K개 과거 `cache_sample_id` 컬럼 추가 — `build_goalstep_next_action_index.py` 확장 |
| 과거 창 feature | `goalstep_feature_cache_end_m1_lobs8_vna/` (313GB, 전 segment 캐시됨) | 완비 |
| loader | `feature_cache.py` (이미 `cache_sample_id` 임의 조회 + 라벨 override 지원) | 소폭 확장 |
| same-level 체인·경계 방어 | next_action index builder에 기구현 | 재사용 |

**Ablation (필수 3종): visual-only(=25.70, 기존) / history-only / fused** — history 지배 여부는 이 표와 g 값으로 정량 판별.

### Phase 2 — Probe zoo + field-aware 앙상블 학습판 (버전 B, ~2-3h)

- Phase 1 최종 구조(또는 P0-b 기각 시 기존 next_action 구조)에 대해 **LR ∈ {1e-4, 3e-4, 1e-3} × WD ∈ {1e-5, 1e-4, 1e-3, 1e-2}** 등 12~20개 head를 한 dataloader로 동시 학습
- 효과: (i) 정규화 다양성 → 오류 비상관 → top-k 앙상블 이득, (ii) 과적합 시점 분산 → field별 sweet spot, (iii) 공짜 HP 스윕. 기대 +1~3pp, 하방 리스크 구조적으로 없음
- epoch×head 격자에서 P0-a와 동일한 field-aware 선발·앙상블

### Phase 3 (선택, headline 필요 시) — 백본 V-JEPA 2.1 ViT-G/384 재추출

- 우승팀 사용 백본 ([V-JEPA 2.1, arXiv:2603.14482](https://arxiv.org/abs/2603.14482)). 스케일 이득은 반복 검증된 축이나 **캐시 재추출(수 시간) + 313GB급 저장** 비용
- Phase 0~2 완료 후 최종 숫자 확정 단계에서만 집행 권장

---

## 4. 판정 기준 요약 (사전 등록)

| 게이트 | 조건 | 통과 시 | 미달 시 |
|---|---|---|---|
| P0-a | 앙상블 top5 > 단일 best (평가용 절반 기준) | 상시 채택 | (사실상 발생 안 함) 단일 best 유지 |
| P0-b | soft-mixture cov@5 ≥ 27.7 (+2pp) | Phase 1 집행 | history 학습 기각 |
| Phase 1 | fused top5 > max(visual, P0-b) +1pp | Phase 2에 fused 구조 사용 | visual 구조로 Phase 2 |
| 전체 목표 | **top-k 50% coverage** | k=10~15에서 달성 시 Step-2 후보 공급 k 확정 | k=20까지 완화 검토 |

목표 감각: 현 top15 46.5~47.9 + 앙상블(+1~3) + history(P0-b 결과에 따라 +0~5) → **top10~15에서 50% 달성이 현실적 경로.** top5 단독 50%는 EK100 SOTA(39.6 val)를 크게 넘는 요구라 목표로 삼지 않는다.

---

## 5. 스케줄링 주의

- 현재 GPU: `z1_adaptive_transition_mr24x8_vna_ep10` 학습 중 (2026-07-22 16:23 UTC 시작, ~1h 예상). **Phase 0은 CPU/유휴 추론만이라 병행 가능**
- Phase 1/2는 adaptive transition 완료 후 큐에 추가 (`serial_start_first.log` 방식 재사용)
- adaptive transition 결과가 좋으면 Phase 1의 "현재 창"을 adaptive 창으로 교체하는 조합도 검토 (직교 축이므로 합산 가능)

## 6. 참조

- Markov 실측 스크립트: 본 문서 1.2절 로직 (세션 인라인 실행됨 — `scripts/step1/goalstep/`에 `measure_transition_prior.py`로 저장 권장)
- 우승팀 코드: [JFAA](https://github.com/CorrineQiu/JFAA) · [VISTA](https://github.com/CorrineQiu/VISTA)
- 외부 선례 (history→anticipation): MeMViT(CVPR'22), RULSTM, AVT, AntGPT(Ego4D LTA)
