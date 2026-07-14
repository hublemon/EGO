# GRPO 학습 결과 로그 (Living Document)

> Step 2 = GRPO 강화학습. WM(V-JEPA2) Top-5 action 후보 중 VLM(Qwen2.5-VL-7B)이 올바른 next action을 고르도록 학습.
> 코드: [train_qwen25vl_grpo_ek100.py](../train_qwen25vl_grpo_ek100.py)

최종 갱신: **2026-06-04** (실험 14 `grpo_final` 완료 추가, **현재 최선 갱신**, 누적 교훈 L9·L10 추가)

---

## 0. 베이스라인

| 지표 | verb | noun | **action(joint)** |
|---|---|---|---|
| GT ∈ Top-5 (**VLM 이론 상한**) | 96.0% | 95.2% | **92.4%** |
| WM rank-1 == GT (**WM 베이스라인**) | 72.1% | 76.2% | **70.0%** |
| Qwen2.5-VL-7B 무학습 (`qwen_n10`, n=10) | — | — | **~50%** |

→ VLM이 후보 안에서 올바르게 고르면 **70% → 최대 92%** (+22pp 여지). 무학습 Qwen은 70%에도 못 미침.

---

## 1. 전체 실험 요약

| # | 디렉토리 | 보상 설계 (한글 요약) | steps | GT reward (초/중/후)† | 판정 |
|---|---|---|---|---|---|
| 1 | `grpo_stage1_noun` | WM 명사 rank-1 그대로 복사하면 최고점 (단서 노출) | 2,499 | — | ❌ 즉시 포화·collapse |
| 2 | `grpo_stage2_action` | WM 행동 rank-1 그대로 복사하면 최고점 (단서 노출) | 2,499 | — | ❌ 즉시 포화·collapse |
| 3 | `grpo_gt_improved` | GT verb+noun 맞추면 점수 (단순 GT 채점, 추론 없음) | 2,499 | ~0.7 평탄 | △ 형식·후보만 학습 |
| 4 | `grpo_think` | 추론 태그 형식 + 후보 준수 +0.5 + GT 약한 점수 → 형식 보상이 GT 압도 | 1,500 | 0.26/0.47/0.25 | ❌ 형식 collapse |
| 5a | `grpo_ranking` | WM 후보 순위 점수 단독 (rank1=1.0 → rank5=0.1, 밖=-0.2), 단서 노출 | 1,500 | — | ❌ rank1 복사 collapse |
| 5b | `grpo_think_ranking` | 추론 태그 + 후보 준수 +0.5 + WM 순위 점수 | 1,500 | — | ◐ collapse 없음, WM rank 미포화 |
| 6-S1 | `grpo_stage_noun` | WM 명사 순위 점수 단독, 단서 노출 | 1,500 | — | ❌ rank1 collapse |
| 6-S2 | `grpo_stage_action` | WM 행동 순위 점수 단독, 단서 노출 (S1 이어받기) | 1,500 | — | ❌ rank1 collapse |
| 7 | `grpo_think_gt` | **추론 태그(0.15) + 추론 품질(0.20) + 후보 이탈 패널티(−0.5) + GT 강한 점수(최대 1.5)** | 1,500 | 0.34/0.46/0.47 | ◐ **현재 최선** — 피크 step 1240 |
| 8 | `grpo_ranking_fix` | WM 순위 점수 + 단서 제거(셔플·점수 숨김) | 450 (중단) | — | ◐ collapse 차단 확인 |
| **9** | `grpo_think_gt_fix` | 실험 7 동일 + **max_steps=750** (overshoot 방지) | **750** | **0.26/0.35/0.40** | ◐ 처방 1+2+3 완료 |
| **10** | `grpo_think_wm_rank_fix` | 추론 태그 + 추론 품질 + 후보 이탈 패널티 + **WM 순위 점수만 (GT 없음)** | **750** | — | ❌ reward≈0, 학습 실패 |
| **11** | `grpo_think_gt_combo` | 실험 9 구성 + **WM 순위 점수 보조 추가** | **750** | **0.27/0.36/0.41** | ◐ 처방 4 완료, GT 미세 우위 |
| **12** | `grpo_2stage_gt_s1/s2` | S1: 명사 GT 채점 → S2: 행동 GT 채점 (2단계 순차 학습) | 375+375 | **0.33/0.34/0.34** | △ flat, 2-stage 효과 없음 |
| **13** | `grpo_2stage_combo_s1/s2` | S1: 명사 순위+GT → S2: 행동 GT+WM 순위 (2단계 복합) | 375+375 | **0.33/0.34/0.35** | △ flat, 2-stage 효과 없음 |
| **14** | `grpo_final` | 실험 11 설계 + **5000샘플·num_gen=8·beta=0.01·GT v3 퍼지 매칭·1250 steps** | **1250** | **0.38/0.52/0.58** | ✅ **현재 최선** — 전 실험 최고치 |

†`reward_gt_accuracy_reward_think_v2` 배치 평균 (0~1.5 스케일, 실제 joint 정확도 ≈ 값÷1.5). 실험 3·5a·5b·6·8 은 해당 키 없음. 실험 14는 v3 스케일(동일 max 1.5). 구간 = 전체 steps 의 1/3씩.

**학습 시간**: 실험 1~8 ≈ 26.6h GPU (2×H200). 실험 9~13 ≈ 8.5h (9·10·11 각 750 step×~7s ≈ 1.5h씩, 12·13 각 750 step×~7s ≈ 1.5h씩). 실험 14 ≈ 3.72h (1250 step, 2×H200). **누계 ≈ 38.7h**.

---

## 2. 누적 교훈

| 번호 | 교훈 | 근거 실험 |
|---|---|---|
| L1 | **collapse 원인은 보상이 아니라 "후보 단서 노출"** — 점수·rank 순 노출 상태에서는 보상 구성과 무관하게 rank1 복사가 자명해. 단서 제거만으로 즉시 해소. | 5a vs 8 대조 |
| L2 | **think-format(답 전 추론 강제) + verb·noun 분리 입력이 collapse에 가장 강함** | 5b·7 |
| L3 | **easy saturating reward(format·candidate +0.5)가 정답 신호를 압도하면 think도 collapse** → candidate 게이트화(유효=0, 무효=−0.5) + gt 비중 강화 + KL β·온도로 완화 | 4 vs 7 |
| L4 | **1 epoch(1,500 step)은 overshoot** — 실험 7 피크가 step ~1,240, 이후 하락. best checkpoint ≈ 중간 지점. | 7 |
| L5 | **WM rank reward 단독으로는 학습 신호 없음** — reward≈0 수렴. GT 신호가 필수 주신호. | 10 |
| L6 | **GT 기반 2-stage(noun→action)는 단일 stage 대비 개선 없음** — GT acc 거의 flat(0.33→0.35). | 12·13 |
| L7 | **GT+WM rank 복합(처방 4)은 GT 단독 대비 후반 평균 +0.01 수준 미세 우위** — 통계적 유의성은 held-out 평가 전까지 불명. WM rank 서브 신호 자체는 말기에 음수(-0.1)로 발산. | 9 vs 11 |
| L8 | **모든 train-time 지표는 held-out 평가로만 확정** — 4·5b·7 말기 joint 25%, 신규 9~13도 수치 유사 → 소표본(n=16 배치) 우연 일치 가능성 배제 불가. | 전체 |
| L9 | **데이터 증량(3000→5000) + num_gen 증가(4→8) + GT-not-in-top5 필터 조합이 학습 효율을 크게 향상** — GT reward 후반 0.578로 기존 최선(0.411, 실험 11) 대비 +40.6% 향상. WM rank 말기 발산 문제도 해소(+0.259 양수 유지). | 14 vs 11 |
| L10 | **beta=0.01(기존 0.04)로 낮추면 더 긴 학습(1250 step)에서도 안정** — KL 페널티를 줄여 탐색 공간을 넓혀도 collapse 없이 GT reward 상승 지속. | 14 |

---

## 3. 신규 실험 상세 (실험 9~13, 2026-06-01)

### 설계 배경

실험 7(think_gt)까지의 주요 미결 과제:
- **처방 1+2+3 동시 적용한 실험 없음** — 7은 처방 2·3만, max_steps=1,500으로 overshoot
- **처방 4(WM rank + GT 복합) 미검증**
- **GT 기반 2-stage(noun→action) 미검증**
- **WM rank 단독 신호 효과 미검증**

공통 설정 변경: `max_steps` 750 (처방 3), 후보 셔플+점수 숨김(처방 1) 적용.

---

### 실험 9 — `grpo_think_gt_fix` (처방 1+2+3 통합)

**목적**: 처방 1(단서 제거) + 2(gt 강화) + 3(max_steps=750)을 동시에 적용한 기준 실험.

| 항목 | 값 |
|---|---|
| reward_funcs | format_think(0.15) · think_quality(0.2) · candidate_gate(0/−0.5) · **gt_v2**(max 1.5) |
| max_steps | **750** (처방 3) |
| 입력 | 셔플+점수 숨김 (처방 1) |
| 시작 | 2026-06-01 06:13 |

**결과**:

| 지표 | 초반(0~250) | 중반(250~500) | 후반(500~750) |
|---|---|---|---|
| GT reward 구간 평균 | 0.264 | 0.350 | **0.398** |
| total reward | — | — | 0.335 |
| loss | — | — | −0.049 |

| think 분석 | step 1 | step 401 | step 701 |
|---|---|---|---|
| think_words | 71.0 | 133.0 | 108.0 |
| diversity | 1.00 | 1.00 | 1.00 |
| cand_mention_rate | 0.50 | 1.00 | 1.00 |

**판정 ◐**: collapse 없음. diversity 전 구간 유지. think_words 71→108 (증가 경향, 실험 7 패턴과 동일). GT reward 완만히 상승. **실험 7과 동일 reward 구성 + max_steps=750** — overshoot 없이 상승 구간에서 종료.

---

### 실험 10 — `grpo_think_wm_rank_fix` (WM rank 단독)

**목적**: WM rank reward만으로 학습 가능한지 확인 (GT 없음).

| 항목 | 값 |
|---|---|
| reward_funcs | format_think · think_quality · candidate_gate · **wm_ranking_reward** |
| max_steps | 750 |

**결과**: `reward_total` 마지막 0.023, loss 0.135 (높음). GT acc 키 없음 (reward 설계에 없음).

**판정 ❌**: WM rank reward가 GT 없이는 학습 신호 역할 불가. **교훈 L5 확정**. GT가 주신호로 반드시 필요.

---

### 실험 11 — `grpo_think_gt_combo` (처방 4: GT + WM rank 복합)

**목적**: GT accuracy(주신호) + WM rank(보조신호) 동시 적용.

| 항목 | 값 |
|---|---|
| reward_funcs | format_think · think_quality · candidate_gate · **gt_v2**(주) · **wm_ranking**(보조) |
| max_steps | 750 |
| 시작 | 2026-06-01 10:01 |

**결과**:

| 지표 | 초반 | 중반 | 후반 |
|---|---|---|---|
| GT reward 평균 | 0.268 | 0.358 | **0.411** |
| WM rank reward | 평균 0.108 | — | −0.100 (말기 발산) |
| total reward | — | — | 0.123 |

| think 분석 | step 1 | step 401 | step 701 |
|---|---|---|---|
| think_words | 71.0 | 118.2 | 104.0 |
| diversity | 1.00 | 1.00 | 0.75 |

**판정 ◐**: GT reward 후반 0.411로 실험 9(0.398) 대비 미세 우위. 그러나 **WM rank 서브 신호가 말기에 −0.100으로 발산** — WM rank reward가 GT 방향과 충돌하거나 모델이 이를 무시하는 방향으로 수렴한 것으로 보임. 처방 4의 효과는 held-out 평가 전까지 불확실 (교훈 L7).

---

### 실험 12 — `grpo_2stage_gt` (GT 기반 2-stage)

**목적**: Stage-1(noun GT) → Stage-2(action GT) 순차 학습. 한계 L6를 GT 방식으로 재시도.

| 단계 | 디렉토리 | reward_mode | steps |
|---|---|---|---|
| S1 | `grpo_2stage_gt_s1` | think_noun_gt (noun GT 전용) | 375 |
| S2 | `grpo_2stage_gt_s2` | think_gt (action GT, S1 이어받기) | 375 |

**S2 결과**:

| 지표 | 초반 | 중반 | 후반 |
|---|---|---|---|
| GT reward 평균 | 0.331 | 0.340 | **0.344** |

think_words: 108.0→98.5, diversity: 0.25(초반 낮음)→0.75

**판정 △**: GT reward 거의 flat(0.331→0.344). 단일 stage 실험 9(0.264→0.398)에 비해 상승폭 현저히 작음. **2-stage 구성이 학습 효율을 높이지 않음** (교훈 L6). S1에서 noun 방향으로 수렴된 정책이 S2에서 action GT로 전환하는 데 오히려 비효율적일 수 있음.

---

### 실험 13 — `grpo_2stage_combo` (GT+WM rank 2-stage)

**목적**: Stage-1(noun GT + noun ranking) → Stage-2(action GT + WM ranking).

| 단계 | 디렉토리 | reward_mode | steps |
|---|---|---|---|
| S1 | `grpo_2stage_combo_s1` | think_noun_combo | 375 |
| S2 | `grpo_2stage_combo_s2` | think_gt_combo, S1 이어받기 | 375 |

**S2 결과**:

| 지표 | 초반 | 중반 | 후반 |
|---|---|---|---|
| GT reward 평균 | 0.330 | 0.336 | **0.347** |
| WM rank (S2 말기) | — | — | 0.237 |

**판정 △**: 실험 12와 동일하게 flat. 실험 11(GT+WM rank, 단일 stage) 후반 0.411 대비 열세. **2-stage + 복합 reward 조합은 단일 stage보다 불리**. WM rank는 S2 말기에 0.237로 양수 유지 (실험 11의 −0.100보다 안정적)이나 overall 학습 개선은 없음.

---

### 신규 실험 종합 비교

| 실험 | GT후반avg | collapse | 특이사항 |
|---|---|---|---|
| 7 `think_gt` (1,500 steps) | 0.472 | 없음 | 피크 step 1,240. 단, 1,500 step은 overshoot 구간 포함 |
| **9 `think_gt_fix`** (750 steps) | **0.398** | 없음 | 처방 1+2+3 통합. 상승 구간에서 종료 |
| **11 `think_gt_combo`** (750 steps) | **0.411** | 없음 | 처방 4. WM rank 보조신호 말기 발산 |
| 12 `2stage_gt` (750 steps) | 0.344 | 없음 | 2-stage 효과 없음 |
| 13 `2stage_combo` (750 steps) | 0.347 | 없음 | 2-stage+combo 효과 없음 |
| 10 `wm_rank_fix` (750 steps) | — | ❌ | WM rank 단독 학습 불가 |

**현재 최선 후보**: 실험 11 (`grpo_think_gt_combo`, 후반 0.411) > 실험 9 (`grpo_think_gt_fix`, 0.398). 단, 차이가 미미해 **held-out 평가 없이는 확정 불가**.

---

## 4. 실험 14 상세 (`grpo_final`, 2026-06-02)

### 설계 배경

실험 11(`grpo_think_gt_combo`, 후반 0.411)이 이전 최선이었으나 다음 한계가 남아 있었음:
- 학습 데이터 3,000샘플 — 에폭 내 다양성 부족
- num_generations=4 — 그룹 내 분산 추정 불안정
- GT-not-in-top5 샘플이 unrewarded noise로 포함됨
- WM rank 보조신호 말기 발산(−0.100) 원인 미해소

실험 14는 이를 모두 동시에 개선하는 **최종 통합 실험**.

---

### 실험 14 — `grpo_final` (최종 통합 실험)

**목적**: 데이터·생성수·하이퍼파라미터를 모두 개선한 최종 설정.

| 항목 | 값 | 실험 11 대비 변경 |
|---|---|---|
| reward_funcs | format_think · think_quality · candidate_gate · **gt_v3**(퍼지 noun) · wm_ranking | gt_v2 → gt_v3 |
| train_samples | **4,947** | 3,000 → +65% |
| num_generations | **8** | 4 → 2× |
| max_steps | **1,250** | 750 → +67% (~0.5 epoch) |
| beta (KL) | **0.01** | 0.04 → 낮춤 |
| temperature | 0.8 | 동일 |
| drop_unrewardable_samples | **적용** (GT∉Top-5 51개 제거) | 미적용 → 적용 |
| 시작 | 2026-06-02 11:50 | — |
| 종료 | 2026-06-02 15:33 (3.72h) | — |

**gt_accuracy_reward_think_v3 변경 내용**: v2(exact match, max 1.5)에 퍼지 noun 매칭 추가.  
`'towel:kitchen' ↔ 'towel'` 처럼 EK100 계층 레이블에서 base 일치하면 부분 점수(noun +0.25, 보너스 +0.3, 최대 0.95). 데이터셋 내 영향 샘플 ~0.1%로 드물지만 오채점 구제 목적.

**결과**:

| 지표 | 초반(0~417) | 중반(417~833) | 후반(833~1250) |
|---|---|---|---|
| GT reward (v3, 0~1.5) | 0.378 | 0.523 | **0.578** |
| WM rank reward | 0.129 | 0.249 | **+0.259** (양수 안정) |
| total reward | 0.757 | 1.108 | 1.171 |
| candidate gate | −0.092 | −0.013 | −0.014 (이탈 거의 없음) |
| loss | 0.012 | 0.005 | 0.009 |

| think 분석 | step 1 | step 401 | step 801 | step 1201 |
|---|---|---|---|---|
| think_words (mean) | 84.8 | 78.4 | 75.9 | 90.6 |
| diversity | 0.60 | 0.25 | 0.25 | 0.38 |
| cand_mention_rate | 0.50 | 1.00 | 1.00 | 1.00 |

summary.json: `final_gt_accuracy = 0.628` (최종 step 1250의 배치 GT reward 값)

**판정 ✅ 현재 최선**:
- GT reward 후반 평균 **0.578** — 이전 최선 실험 11(0.411) 대비 **+40.6% 향상**.
- WM rank 보조신호 말기 **+0.259 양수 유지** — 실험 11의 말기 발산(−0.100) 완전 해소.
- candidate gate 후반 −0.014, 사실상 0 — 후보 이탈 없음.
- diversity 0.25(중반)~0.38(후반) — 단일 답으로의 완전 collapse 없음.
- think_words 75~91 범위 유지, cand_mention_rate 1.00 — 추론 형식 안정.

### 실험 14 종합 비교 (전체 실험 대비)

| 실험 | GT후반avg | WM rank 말기 | collapse | 특이사항 |
|---|---|---|---|---|
| 7 `think_gt` (1,500 steps) | 0.472 | — | 없음 | 1,500 step overshoot 포함 |
| 9 `think_gt_fix` (750 steps) | 0.398 | +0.033 | 없음 | 처방 1+2+3 통합 |
| 11 `think_gt_combo` (750 steps) | 0.411 | **−0.100** (발산) | 없음 | 처방 4, WM rank 불안정 |
| **14 `grpo_final`** (1,250 steps) | **0.578** | **+0.259** (안정) | 없음 | **현재 최선** |

---

## 4. 실험 8 상태 (중단)

`grpo_ranking_fix` — **step 450/1,500에서 중단 상태 (진행률 30%)**. collapse 차단(4생성 전부 상이, wm_rank 0.71 미포화)은 450 step에서 확인됐으나 최종 수렴 및 held-out 일반화는 미확인.

---

## 5. 다음 단계 (우선순위)

### 🔴 즉시 — held-out 평가

지금까지의 **모든 수치는 train 배치 위 측정값**. 학습된 모델이 WM rank-1 베이스라인(70%)을 실제로 넘는지 held-out 세트로 확인해야 이후 방향 결정 가능.

평가 대상 (우선순위순):

| 모델 | 체크포인트 |
|---|---|
| 무학습 Qwen2.5-VL (베이스라인) | base model |
| WM rank-1 그대로 | — |
| **실험 14** `grpo_final` | `runs/grpo_final/` (final, **현재 최선**) |
| 실험 11 `grpo_think_gt_combo` | `runs/grpo_think_gt_combo/` (final) |
| 실험 9 `grpo_think_gt_fix` | `runs/grpo_think_gt_fix/` (final) |

지표: action joint 정확도, verb 정확도, noun 정확도.

### 🟠 그 다음 — 재학습 옵션 (held-out 결과에 따라)

- **옵션 A**: 실험 14 방향 유지 + max_steps 증가 (1,250 → 2,000~2,500, ~0.8 epoch)
- **옵션 B**: 실험 14 + learning rate 조정 또는 LoRA rank 증가
- **실험 8 완주**: ranking_fix 재시작 (450 step → 1,500 step 완주)

### 🟡 추후

- memory_context(temporal) on/off ablation
- reasoning 정성 평가 자동화 (영상 단서 인용·일관성)

---

## 6. 보상 함수 설계 상세 (한글)

각 보상 함수가 실제로 무엇을 측정하고 왜 그렇게 설계했는지 기록.

### 6.1 형식 보상 — "답변 구조 지킴"

| 함수 | 값 | 조건 |
|---|---|---|
| `format_reward_think` | +0.15 | `<think>...</think>` + `<action>...</action>` 태그 **둘 다** 존재 |
| `format_reward` (구형) | +0.15 / 0.0 | JSON `{"verb":..,"noun":..}` 파싱 성공 여부 |

**설계 의도**: 모델이 매번 다른 형식으로 답변하면 파싱이 실패해 보상을 아예 못 받는 문제를 방지. 형식 자체에 최소 점수를 주어 "최소한 구조는 지키도록" 유도.

**한계**: 형식 보상(0.15)이 너무 쉽게 달성 가능 → 다른 보상과 합산 시 비중이 크면 모델이 형식만 맞추고 내용(GT 정답)을 무시하는 collapse 유발.

---

### 6.2 추론 품질 보상 — "think 블록이 실제로 생각하고 있는가"

| 조건 | 값 |
|---|---|
| think 블록 20단어 이상 **AND** 후보 verb/noun 중 1개 이상 언급 | +0.20 |
| think 블록 10단어 이상 (언급 무관) | +0.08 |
| 그 외 | 0.0 |

**설계 의도**: 단순히 `<think>I think</think>` 같은 형식만 채우는 것을 방지. 실제로 어떤 행동을 선택할지 추론하는 내용이 있어야 점수.

---

### 6.3 후보 준수 보상 vs 후보 이탈 패널티 — "WM이 제안한 후보 안에서 고를 것"

두 가지 버전이 있으며, 설계 방향이 반대:

**구형 (실험 4): 후보 준수 적극 보상**
| 조건 | 값 |
|---|---|
| verb ∈ Top-5 **AND** noun ∈ Top-5 | +0.50 |
| 둘 중 하나만 | +0.10 |
| 둘 다 후보 밖 | −0.20 |

→ **문제**: +0.50이 너무 커서 4개 generation 전부 "후보 안 아무거나"를 고르면 그룹 내 reward 차이가 없어짐 → GT 방향 gradient 사라짐.

**개선형 (실험 7~13): 게이트 패널티만**
| 조건 | 값 |
|---|---|
| verb ∈ Top-5 **AND** noun ∈ Top-5 | 0.0 (가점 없음) |
| 그 외 | −0.50 (패널티) |

→ 후보 안에서 고르는 건 "당연한 것"으로 취급, 별도 보상 없음. **GT를 맞추는 것만이 양의 보상**이 되어 그룹 내 변별력 확보.

---

### 6.4 GT 정확도 보상 — "실제로 정답을 골랐는가"

두 버전:

**v1 (실험 3·4·5b·6 일부)**: 최대 1.0
| 조건 | 값 |
|---|---|
| verb == GT verb | +0.25 |
| noun == GT noun | +0.35 |
| 둘 다 | +0.25 + 0.35 + 0.40 = **1.0** |

**v2 (실험 7~13)**: 최대 1.5 — GT 신호를 형식·후보 보상보다 강하게
| 조건 | 값 |
|---|---|
| verb == GT verb | +0.4 |
| noun == GT noun | +0.5 |
| 둘 다 | +0.4 + 0.5 + 0.6 = **1.5** |

**설계 의도**: v1에서는 형식(0.15) + 후보(0.50) = 0.65가 GT(1.0)에 근접 → 형식+후보만 맞춰도 충분 → v2에서 GT 배점을 1.5로 올려 형식·후보를 압도하게 설계.

> **주의**: reward_log의 `reward_gt_accuracy_reward_think_v2` 수치는 0~1.5 스케일의 raw 보상값. 직관적인 정확도로 환산하면 `값 ÷ 1.5 ≈ joint 정확도`(verb=noun 가정 시 근사값).

---

### 6.5 WM 순위 보상 — "WM이 높게 예측한 행동을 선택했는가"

| WM에서 예측한 rank | 보상 |
|---|---|
| rank 1 (최우선 후보) | +1.0 |
| rank 2 | +0.7 |
| rank 3 | +0.4 |
| rank 4 | +0.2 |
| rank 5 | +0.1 |
| Top-5 밖 | −0.2 |

**설계 의도**: GT가 항상 Top-5에 있는 건 아닐 수 있고, WM이 맞을 때(~70%) WM rank-1을 선택하도록 추가 유도. GT 보상과 함께 쓰면 "WM rank-1 = GT일 때" 보상이 두 배가 되어 시너지 기대.

**실험 결과 (L5, L7)**: WM 순위 보상 단독으로는 학습 신호 없음(실험 10). GT와 병용 시 후반에 WM 순위 보상이 음수(−0.1)로 발산(실험 11). GT가 주신호일 때 모델이 WM rank-1과 다른 방향으로 수렴하는 경향.

---

## 7. GT 정확도 vs WM rank-1 동조율 분석

### 7.1 측정 방법

- **GT reward**: `reward_log.jsonl`의 배치 평균 (전체 steps 포함, 신뢰도 높음)
- **WM1 동조율**: `completion_samples.jsonl` 파싱 → 모델 예측이 WM rank-1과 일치하는 비율. **소표본(n=8~20/구간)이므로 방향성만 참고.**

### 7.2 실험별 측정값

| 실험 | 구간 | GT reward | GT% 근사 | WM1_sel% | n | 해석 |
|---|---|---|---|---|---|---|
| **7** `think_gt` | 초반 | 0.340 | ~23% | 11% | 18 | 학습 초기, 모델 아직 형식 익힘 |
| | 중반 | 0.460 | ~31% | 50% | 20 | GT 상승 = WM rank-1 추종 |
| | 후반 | 0.472 | ~31% | 32% | 19 | overshoot 전 피크 구간 |
| **9** `think_gt_fix` | 초반 | 0.264 | ~18% | 0% | 10 | — |
| | 중반 | 0.350 | ~23% | 25% | 8 | — |
| | **후반** | **0.398** | **~27%** | **33%** | 12 | **GT(42%) > WM1(33%): WM 안 따르고 GT 맞춘 사례 존재** |
| **11** `think_gt_combo` | 초반 | 0.268 | ~18% | 0% | 8 | — |
| | 중반 | 0.358 | ~24% | 13% | 8 | — |
| | **후반** | **0.411** | **~27%** | **46%** | 11 | GT = WM1: 맞출 때는 항상 rank-1 경유 |
| 12 `2stage_gt_s2` | 전체 | ~0.344 평탄 | ~23% | 6% | 16 | flat, 2-stage 무효 |
| 13 `2stage_combo_s2` | 전체 | ~0.347 평탄 | ~23% | 13% | 16 | flat, 2-stage 무효 |

### 7.3 관찰된 패턴

**패턴 1 — GT 맞출 때는 대부분 WM rank-1을 경유**

GT%와 WM1_sel%가 거의 동일하거나 근사함(대부분 실험). 모델이 GT를 맞추는 경로가 "WM rank-1이 마침 GT인 경우(~70%)"에 집중되어 있다. WM rank-1이 아닌데 GT를 맞추는 경우(GT ∩ ¬WM1)는 소표본 내에서 1~2개 수준.

**패턴 2 — 모델이 WM rank-1을 "능동적으로 override"하는 행동은 아직 관측 미약**

WM 베이스라인(70%)보다 훨씬 낮은 GT% (~27%)는 모델이 WM rank-1을 무조건 따르지는 않지만, 그렇다고 WM이 틀렸을 때 올바른 GT를 찾아가지도 못하는 상태를 시사함.

**패턴 3 — WM rank 보조신호는 후반 발산 (실험 11)**

GT+WM 복합 보상(실험 11)에서 WM rank 보상이 중반(+0.108) → 후반(−0.100)으로 발산. GT 방향과 WM rank 방향이 충돌하는 훈련 샘플이 존재함을 의미. GT 신호가 강해질수록 모델이 WM rank-1 외의 선택을 하는 경우가 생기고, 그 때 WM rank 보상이 음수가 됨.

### 7.4 의의와 한계

**현재 알 수 있는 것**:
- GT 보상이 학습 신호로 작동함(실험 3·7·9·11에서 초반→후반 일관된 상승).
- WM rank 순위 단독 신호로는 학습 불가(실험 10), GT 주신호가 필수.
- 형식·후보 보상 비중을 낮추고 GT 배점을 1.5로 높인 설계(실험 7~11)가 collapse 없이 가장 안정적.
- 훈련 배치 기준 GT 정확도가 27~31%(GT reward ÷ 1.5)로 수렴. WM 베이스라인 70%에는 아직 미달이나, 이는 train-time 측정으로 held-out과 다를 수 있음.

**아직 모르는 것 (held-out 평가로만 확인 가능)**:
- held-out에서 WM rank-1 동조율 및 GT 정확도가 실제 70% 베이스라인을 초과하는지.
- WM이 틀렸을 때(30% 케이스) 모델이 GT로 재유도할 수 있는지.
- 훈련 중 상승한 GT reward가 일반화되는지, 아니면 훈련 분포에 과적합인지.

---

## 부록 A: 실험 1~8 핵심 압축

**실험 1·2** (`grpo_stage1_noun`, `grpo_stage2_action`): top-1 정답을 프롬프트에 노출 + score 순 정렬 → "index 1 복사" 자명해. reward 1.65(이론 최대)에 수십 step 만에 도달, loss→0. **이 설계 자체가 collapse를 유발함을 확립.**

**실험 3** (`grpo_gt_improved`): top-1 hint 제거 + 후보 셔플 + GT accuracy 보상으로 전환. 즉시 포화는 사라졌으나 gt_acc ~0.7 평탄, reasoning 15→8 단어로 퇴화. 형식·후보 준수만 학습. **think-format 필요성 확인.**

**실험 4** (`grpo_think`): think-format 도입. 초반 gt_acc 상승(0.47)했다가 말기 0.25 원위치. format(0.15)+candidate(0.5)+think_quality(0.2) = 형식만 갖추면 0.85 확보 → 그룹 내 정답 변별 압력 부족. think 단어수 159→82 감소. **형식 보상이 내용 보상을 압도하면 think도 collapse.**

**실험 5a** (`grpo_ranking`): wm_ranking reward. 4생성 전부 `(roll,bread) ×4` collapse. 점수 순 노출로 rank1 복사 자명해. **실험 8에서 단서 제거로 해소 확인.**

**실험 5b** (`grpo_think_ranking`): think + wm_ranking 조합. collapse 없음. wm_rank 0.13→0.35 (미포화 = 다양한 rank 선택). **think-format이 ranking collapse에 가장 강함.**

**실험 6-S1·S2** (`grpo_stage_noun`, `grpo_stage_action`): noun ranking → action ranking 2-stage. 둘 다 rank1 복사 collapse (wm_rank→1.0, loss→0). think 없는 JSON 출력 + 점수 노출 = 구조적 취약.

**실험 7** (`grpo_think_gt`): think_gt = candidate 게이트화 + gt_v2 강화(max 1.5) + KL β=0.04 + temp=1.0. gt_acc 0.32→0.57(피크, step ~1,240) → 0.36(하락). 다양성 0.50 유지, think_words 71→132 증가. **처방 2·3 의도대로 작동 확인. 단, 1,500 step 전체 학습 시 overshoot.**

**실험 8** (`grpo_ranking_fix`): 실험 5a 동일 보상 + 점수 숨김 + 셔플. 450 step에서 중단. 4생성 전부 상이, wm_rank 0.71 미포화. **단서 제거만으로 collapse 해소됨을 입증 (보상 문제 아님). 중단 상태 유지.**

---

## 부록 B: 산출물 경로

```
runs/grpo_stage1_noun/          # 실험 1
runs/grpo_stage2_action/        # 실험 2
runs/grpo_gt_improved/          # 실험 3
runs/grpo_think/                # 실험 4
runs/grpo_ranking/              # 실험 5a
runs/grpo_think_ranking/        # 실험 5b
runs/grpo_stage_noun/           # 실험 6-S1
runs/grpo_stage_action/         # 실험 6-S2
runs/grpo_think_gt/             # 실험 7 ← checkpoint-1000 평가 후보
runs/grpo_ranking_fix/          # 실험 8 (중단)
runs/grpo_think_gt_fix/         # 실험 9 ← held-out 평가 후보
runs/grpo_think_wm_rank_fix/    # 실험 10
runs/grpo_think_gt_combo/       # 실험 11 ← held-out 평가 후보 (현재 최선)
runs/grpo_2stage_gt_s1/         # 실험 12 S1
runs/grpo_2stage_gt_s2/         # 실험 12 S2
runs/grpo_2stage_combo_s1/      # 실험 13 S1
runs/grpo_2stage_combo_s2/      # 실험 13 S2
runs/grpo_final/                # 실험 14 ← 현재 최선, held-out 평가 1순위

# 각 runs/{exp}/ 안: reward_log.jsonl · completion_samples.jsonl · think_analysis.jsonl · meta.json · summary.json
```
