# F0 최종 개선 계획 (확정 · v2)

2026-07-18 · 회의 「F0-B0 최종 확정 논의」(7/17) × 심야 회의 「미저닝/리저닝 프롬프트 전략」(7/17)
× B0 피드백 2건(「FAA_F0_REVIEW…」·「EGO_FAA_FINAL_DESIGN_FEEDBACK…」) 통합
기존 계획: `docs/experiments/2026-07-18_f0_handoff.md` (본 문서가 이를 개정한다)
계획 UI: https://claude.ai/code/artifact/3d2599de-b9db-45ed-9ec3-76ccf3dea79e

**모든 항목 확정.** 최종 피드백에서 수용한 핵심 4개(§0)까지 반영 완료.

## 0. 최종 피드백 수용 4개 (v2 개정 사항)

1. **Credit assignment 진단 — 문서로만 수용 (코드 개입 없음).** GRPO의 sequence-level reward는
   completion 전체 토큰에 credit을 배분하므로 reasoning·belief는 WM reward로 간접 학습되며,
   이것이 선택적 위축(belief −0.45 · history −0.17)의 기계적 원인이다. 이 설명을 공식 채택한다.
   단 처방은 기각: action-only(필드 삭제)는 심야 회의의 "리즈닝 통합 유지" 결정과 충돌하고
   train-inference 분포 불일치를 만들며, 토큰 마스킹은 하드코딩적 개입이라 지양(팀 확정).
   대응은 기존 확정 장치(L2-a/c/d + 프록시 로깅 + 사후 judge 선정)로 하고,
   full run에서 프록시가 심각히 재악화하면 재논의한다.
2. **학습량 재정의.** 500 step = configuration **검증 run** (프롬프트 방문 1,000회 = 데이터 20%×1회).
   **full-data ≥ 1 epoch run 이 freeze 전 필수** — 덜 학습된 정책을 freeze하면 B0의
   초기값·reference가 전부 거기 묶인다(불가역).
3. **교란 분리.** 검증 run의 큰 변수는 멀티프레임 하나: **r16 + 4프레임**으로 진행.
   3중 비교(1f-base / 4f-base / 4f-trained) + 모든 frame timestamp ≤ trigger 검증.
   **⚠ r64는 폐기가 아니라 연기 — 4프레임 효과 확정 후 분리 ablation으로 반드시 시도한다**
   (회의 결정이었던 항목이며, 학습 용량 부족이 진단 원인 ③이기도 하다. 잊지 말 것).
4. **cutoff 엄격화 = freeze 게이트.** `stop_time < trigger_time` 엄격 부등호 + 3개 제외 조건
   — ⑴ trigger를 가로지르는 segment ⑵ stop == trigger ⑶ timestamp 불완전·경계 판정 불가.
   전 split 재생성 + 자동 leakage 검사 통과를 freeze 수용 기준에 편입.

---

## 1. 기존 계획 대비 변경 사항

| 항목 | 기존 계획 (7/18) | 확정 방향 | 근거 |
|---|---|---|---|
| **모델** | Qwen2.5-VL 회귀 권고 | **Qwen3-VL 유지** (회귀 잠정 취소). Qwen2.5 통제 비교는 최종 성능 주장 시점(P1)으로 연기 | 회의: Qwen3 verbose 특성상 −1.8%p를 단순 저하로 단정 어려움 |
| **학습량 증가** | 최우선 (2,500~5,000 step) | **후순위** — 멀티프레임·LoRA·검증 후 최종 단계 | 회의: 시간 비용 + 2.8% 미약. 가설 자체는 유효 (§1.1) |
| **멀티프레임** | 배터리 ⑤ (사전 검증 후 결정) | **최우선 확정** — 4프레임, base 평가로 30% 초과 확인 | 회의 최우선 = 진단 원인②(관측 비대칭). 추론 시점 정보 확대라 연구 의의 무손상 |
| **LoRA** | 보류 (플래토 시) | **rank 64 우선 적용** | 회의 결정. B0 전환 대비 겸용 |
| **리즈닝 품질** | 관측 지표 | **전 구간 유지 = F0 확정 게이트.** 메커니즘 §3 확정 | 회의: "손실 발생한 채로 B0 진입 금지" |
| **history cutoff** | 개선 항목 ① | **F0 확정의 필수 선행 조건** + 자동 leakage 검사 + base·trained 재평가. 완료 전 기존 수치 잠정 표기 | B0 피드백 §3, 양측 합의. 경계 중첩 action 처리 규칙 명문화 포함 |
| **judge 정책** | gemini-2.5-pro 단독 강제 | **gemini-2.5-pro 단독 강제 유지** (변경 없음) | B0 피드백의 교차 calibration 제안은 **기각 (팀 확정)**. 루브릭·프롬프트·버전 고정, 학습·preference 미사용 원칙 유지 |
| **F0 belief의 지위** | B0 DPO rejected 재료 | **진단 baseline 전용** — B0는 별도 belief proposal로 online belief 생성. 단 F0의 3태그 출력·belief 품질 유지 요구는 존속 | B0 피드백 §1.3. 부수 효과: 문체·자기모순 교란(리뷰 C4·C5) 자동 해소 |
| **hindsight 소스** | 전체 GT trajectory | **next 3~5 GT actions** — 데이터 계약에 `future_gt_actions` 필드 추가 (policy prompt 노출 절대 금지, offline teacher·평가 전용) | B0 피드백 §6.3. F0 학습에는 영향 없음 — 확정 |
| **수치 표현** | "학습 성립(+0.028)" "0.374 도달 목표" | 단일 run 관측치로 완화 + 명칭 교체 (§4) | B0 피드백 §2 전면 수용. CI·유의성·시드 반복은 P1 |

### 1.1 회의 반론("학습량 부족만으로 납득 어렵다")에 대한 데이터 답변

이전 25조합 포맷의 +13%p(0.044→0.17)와 이번 +2.8%p는 같은 종류의 학습이 아니다.
25조합 상승분의 대부분은 후보 밖 출력(52.6%, 정확도 0.027)을 후보 안으로 끌어오는 **포맷 결함
보상 학습**이었고, joint 포맷은 시작부터 in_joint5 0.98이라 그 여지가 없다. 남은 학습 여지는
"WM이 맞는 구간의 일치율 0.49→0.85"라는 다른 종류의 과제다. 따라서 학습량 가설은 기각이 아니라
후순위 — 멀티프레임·LoRA로 스텝당 효율을 먼저 올린 뒤 마지막에 스텝을 늘린다.

## 2. B0 피드백 수용 정리

| 판정 | 항목 | 반영 내용 |
|---|---|---|
| 수용 | 명칭 교체 (§2.3) | 0.374 → **"WM top-1 reference accuracy"** (WM-following target) · 0.620 → **"candidate-recall oracle ceiling"**. "도달 장벽 없음" → "추가 학습에서 검증할 목표" |
| 수용 | GT-free 표현 (§2.4) | "GT label 없이 학습" 금지 → **"GT reward 없이, GT annotation 유래 history를 입력으로 사용해 WM likelihood에 정렬"** |
| 수용 | 통계 신중화 (§2.1) | +0.028 = "단일 Qwen3 run의 관측 향상 (held-out 500 중 ~14개 차이)". paired bootstrap CI · McNemar/permutation · 시드 반복은 P1 |
| 수용 | G2 서술 완화 (§2.2) | "joint 후보를 본 **base VLM**이 G2 구간에서 우연 초과 — formulation·base의 공. GRPO 개선 증거는 아직 없음" |
| 수용 | cutoff 필수 조건화 (§3) | 수정 + 전 split memory 재생성 + 자동 검사 + 재평가 = F0 확정 선행 조건 |
| 수용 | P0 산출물·데이터 계약 (§5~7) | sample-level prediction/candidate/split manifest JSONL · `f0_selected_action` 메타 분리(프롬프트 노출 금지) · `future_gt_actions`(next 3~5) · 재현성 패키지(commit·config·커맨드·버전·시드) |
| **기각** | judge 다중화 (§4) | **gemini-2.5-pro 단독 강제로 진행 (팀 확정).** 교차 calibration·agreement 보고 미채택. 루브릭·버전 고정과 학습 미사용 원칙은 유지 |
| 조정 | Qwen2.5 재학습 (P1-1) | 회의 결정에 따라 시점을 "최종 성능 주장 전"으로 이동. 취지(모델 효과 분리)는 동의 |
| 기록 | B0 belief proposal 분리 (§1.3) | F0 belief는 B0 재료 아님 → C4·C5 교란 자동 해소. 단 F0 belief 품질 유지 요구는 존속 (회의 결정 3이 상위 원칙) |
| 기록 | B0 2단계 분리와 "리즈닝 내 액션 자명성" 우려 | B0 측 설계 사안. F0 출력 포맷 영향 가능성(후보별 리즈닝 분리안)만 추적 |

## 3. 리즈닝 품질 유지 — 확정안

**요구사항 (회의 결정 3):** 히스토리·belief 활용이 학습 전 구간에서 최소 유지 (하락 금지).
마스킹 등 하드코딩 허용. 리즈닝 *개선*은 B0의 몫, *유지*는 F0의 책임.

**설계 제약:** "리워드에 WM-외 신호 금지" 원칙 유지. 금지 대상은 semantic reward 오염이며,
입력 조작·체크포인트 선정 기준은 허용 공간.

### 3.1 결정 기록

| 제안 | 결정 | 사유 |
|---|---|---|
| L1 Gemini 상시 모니터링 게이트 | **기각** | judge 상시 가동 부담 + 예방이 아닌 사후 차단. 대체: §3.3 |
| L3 KL 앵커 (β>0) | **기각** | WM 정렬에 직접 영향 (gradient 억제) |
| L2-b 히스토리-예측적 샘플 가중 | **기각** | 사전 offline 계산 필요 + 오버샘플의 학습 분포 외부 간섭. 핵심 근거(최근 4초 히스토리 ↔ WM top-1 상관)는 L2-c가 분포 간섭 없이 부분 흡수 |
| **L2-a · L2-c · L2-d** | **확정** | 전부 입력측 조치 — 리워드·데이터 분포 무간섭 |

### 3.2 확정 메커니즘 (전부 입력측)

- **L2-a 프레임 마스킹 커리큘럼**: 학습 프롬프트의 **15~20%에서 이미지 제거** — 해당 샘플에서
  히스토리가 WM 예측의 유일한 경로가 되어 히스토리 회로에 gradient 보장. 리워드 무오염.
  멀티프레임 도입 시 비율을 20% 쪽으로 (프레임 정보가 풍부할수록 위축 압력 증가).
- **L2-c 프레임-히스토리 시간 정렬** (멀티프레임 연계로 확장 확정): 멀티프레임 4f가 관측 4초
  구간에서 샘플되므로, **각 프레임에 시점 라벨을 붙이고 (Frame 1 = 4.0s ago … Frame 4 = now),
  temporal proximity 히스토리 항목을 같은 시점에 정렬 표기**한다:
  `"4.0s ago (Frame 1): stir pan · 2.7s ago (Frame 2): …"`.
  Type-2 temporal_proximity 조회 offset(현재 0.5/1/2s)을 프레임 샘플 시각과 일치시킨다.
  효과: ① 프레임과 히스토리가 하나의 시간축 서사가 되어 모델이 히스토리를 프레임으로 교차 검증
  (히스토리 사용이 시각 근거와 결합되어 강화) ② 이 4초 구간이 정확히 WM이 보는 클립이므로
  "프레임 + 정렬된 히스토리" = WM 관측의 텍스트-시각 통합 재구성 — 멀티프레임이 히스토리를
  밀어내는 대신 서로 참조하게 만드는 구조. 근거: cutoff 버그에서 직전·중첩 행동의 예측력이
  가장 높았다는 실측. 구현: 프레임 추출 타임스탬프 라벨 + memory 직렬화 offset을 프레임 시각으로.
  **누설 주의**: Type-2의 기존 안전장치(현재 진행 action 누설 방지용 start 기준 조회) 유지.
- **L2-d belief 지시문 강화**: 퇴화 형태가 "선택 행동의 재진술"로 특정돼 있으므로 프롬프트에
  **"방금 고른 행동의 재진술 금지 — 여러 행동을 아우르는 목표"** 명문화. belief 무채점 계약은
  그대로, base 분포의 출발점을 올려 침식 감속.

**상호작용 경고**: 멀티프레임은 acc를 올리지만 히스토리의 한계 효용을 줄여 위축 압력을 키운다 —
L2-a 마스킹과 반드시 병행.

### 3.3 검출 (L1 기각의 대체 — Gemini 상시 가동 없음)

1. **무료 구조 프록시 상시 로깅 (API 비용 0)**: ① reasoning 내 히스토리 토큰 참조율
   ② belief 재진술율 (belief가 선택 action 문자열 포함) — `sample_reasoning_traces.py`의
   휴리스틱을 학습 로거로 이식. 리워드 미사용, 순수 관측.
2. **Gemini judge는 저장 체크포인트(125/250/375/500)에 사후 1회만** (~$0.3) —
   최종 체크포인트 선정 기준: **"acc 상승 ∧ history_grounding ≥ 1.9 ∧ belief_globality ≥ 1.9"**
   (base 2.00/2.00 대비 −0.1 이내).

**정량 판정 기준**: 위 judge 기준 + `--no_memory` 평가의 acc 하락폭(히스토리 실제 기여량)
base 대비 비감소 + belief-swap 민감도 비감소 (회의 지정 이밸류에이션 2종과 동일).
측정 시점: 재학습 전 base baseline → 학습 중 프록시 → freeze 후보 재측정.

**한계 (정직 기록)**: belief에는 L2-b 같은 유인 경로가 없어 L2-d + 검출이 전부다.
근본 개선은 설계대로 B0의 몫 — F0의 현실적 목표는 "재진술율이 base 수준을 넘지 않는 것".

## 4. 표현·명칭 확정

| 기존 | 확정 표현 |
|---|---|
| 실질적 상한 0.374 | WM top-1 reference accuracy (WM-following target) 0.374 |
| 논리적 상한 0.620 | candidate-recall oracle ceiling 0.620 |
| "GT 없이 학습이 성립한다" | "GT reward 없이, GT annotation 유래 history를 입력으로 사용하여 WM likelihood에 정렬한다" |
| "학습 효과 +0.028 — 성립" | "단일 Qwen3-VL run에서 +2.8%p 관측 향상 (500샘플 중 ~14개). CI·유의성·시드 반복으로 안정성 확인 필요" |
| "G2가 우연을 넘었다" | "joint 후보를 본 base VLM이 WM top-1 오류 구간에서 우연 초과 선택 — formulation·base의 공. WM-only GRPO가 이를 개선한다는 증거는 아직 없음" |
| "Goodhart가 도움 안 되는 추론을 정확히 제거" | "제한된 judge 평가에서 후보 비교 증가·belief 전역성/히스토리 활용 감소 패턴 관측 — 선택적 위축 가설과 일치하나 trace 검토·표본 확대 필요" |

B0 피드백 §10의 논문용 요약 문단은 그대로 채택 가능 (잠정 요약으로 명시된 표현).

## 5. 확정 실행 계획

| # | 항목 | 내용 | 상태 |
|---|---|---|---|
| 1 | **멀티프레임 4f 도입** | base 평가로 30% 초과 확인 → 입력 계약 확정. rank1\|in5·wm_follow 상한 상승분 기록 | 확정 · 최우선 |
| 2 | **LoRA rank** | **검증 run은 r16 유지** (교란 분리 — §0-3). **r64는 4프레임 효과 확정 후 분리 ablation으로 필수 시도** | v2 개정 |
| 3 | **cutoff 수정 + 재검증** | `action_stop < trigger_time` · 전 split memory 재생성 · 자동 leakage 검사 · base/step500 재평가. 완료 전 기존 수치 잠정 표기 | 확정 · B0 인계 필수 |
| 4 | **리즈닝 유지 메커니즘** | L2-a 마스킹(15~20%) + L2-c 최근-4초 표기 + L2-d belief 지시문 + 프록시 상시 로깅 + 사후 judge 1회 (L1·L3·L2-b 기각) | 확정 |
| 5 | **500-step 검증 run** | Qwen3 유지 · 4프레임 · **r16** · gen 8 · T 1.0 + #4 포함. 목적 = configuration validation (loss/reward/parsing/checkpoint 확인) | v2 개정 |
| 6 | **기여도·인과성 이밸류에이션** | 히스토리 제거/교체 평가 · belief-swap · `belief_action_link` 지표 (GPU 서버 확보 후) | 확정 (회의 4번) |
| 7 | **P0 산출물 패키지** | B0 데이터 계약 JSONL (선택 action 메타 분리 · `future_gt_actions` next 3~5 · split manifest) + 재현성 패키지 + frozen checkpoint | 확정 |
| 8 | **full-data 최종 run (≥1 epoch)** | 검증 run 통과 후 수행. **freeze 전 필수 조건** (§0-2) — 이 run의 체크포인트만 freeze 후보 | v2 개정 |
| 9 | Qwen2.5 통제 비교 + 통계 패키지 (P1) | paired bootstrap CI · McNemar · 시드 반복 — 최종 성능 주장 전 | 연기 |

B0 착수 조건(피드백 §9 체크리스트)은 #3·#5·#7 완료로 충족. P1·P2는 B0 구현과 병렬 진행 가능.

### 재학습 스펙 요약 (#5)

```
모델      : Qwen3-VL-8B (회귀 취소)
입력      : 멀티프레임 4f (base 평가 통과 후, 프레임별 시점 라벨) + L2-a 마스킹 15~20%
            + L2-c 프레임-히스토리 시간 정렬 표기 + L2-d belief 지시문 강화
어댑터    : LoRA r16 (검증 run · 교란 분리) — ⚠ r64는 이후 분리 ablation으로 필수 시도
샘플링    : num_generations 8 · temperature 1.0 · max_completion 384
학습      : dr_grpo · scale_rewards none · epsilon_high 0.28 · beta 0
            · min_wm_spread 0.05 · save/eval 125
런 구성   : ① 500-step 검증 run (config validation)
            ② full-data ≥1 epoch 최종 run — freeze 전 필수, freeze 후보는 이 run에서만
선행 조건 : history cutoff 엄격 수정(stop < trigger + 3제외) + 전 split memory 재생성
            + 자동 leakage 검사 통과
검출      : 프록시(히스토리 참조율·belief 재진술율) 상시 로깅
            + 체크포인트에 gemini-2.5-pro 사후 judge 1회
선정 기준 : action 지표 1차 (acc·wm_follow·G2) → judge 확인 (history ≥ 1.9 ∧ belief ≥ 1.9)
```
