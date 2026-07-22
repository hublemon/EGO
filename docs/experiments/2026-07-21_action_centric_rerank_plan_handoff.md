# Action-Centric Rerank 전환 — handoff 평가 및 다음 실행 계획

> 작성일: 2026-07-21 KST
> 대상 문서: `belief_reward_action_centric_handoff.md` (이하 **원안**)
> 저장소: `/mnt/nvme/migration/jihun/EGO_jihun` (`retro` 브랜치, origin `hublemon/EGO`)
> 이 문서의 역할: 원안의 타당성을 **실측으로 검증**하고, 그 결과에 따라 수정된 실행 계획·ablation·소요시간을 제시한다.
> 실행 중이던 P3 런은 이 문서 작성 시점에 계속 진행 중이며 아무것도 변경하지 않았다.

---

## 0. 세 줄 요약

1. **원안의 방향 전환(belief 제거 → candidate-constrained action selection)은 옳다.** 다만 근거는 원안이 제시한 것보다 훨씬 강하다 — belief 유무가 아니라 *어떤 arm이든* 후보 선택 자체를 못 하고 있다.
2. **원안 §4의 수치 가정은 틀렸다.** `r=0.03`을 가정했으나 실측은 **0.503~0.597**이다. 40개 arm 전수 조사에서 `r < 0.37`인 arm이 하나도 없다. 따라서 원안이 부차적 안전장치로 둔 `L_keep`(G1 보존)이 **주 목표**이고, 주 목표로 둔 G2 margin이 부차적이다.
3. **가장 먼저 돌릴 실험은 이미 저장소에 구현되어 있다.** `pro_gx_train.py`가 원안 §5.1의 candidate CE를 그대로 수행한다. Tier 0(GPU 2h)로 방향을 판정할 수 있다.

---

## 1. 원안 평가

### 1.1 타당한 것

| 원안 주장 | 평가 | 근거 |
|---|---|---|
| 자연어 belief에 네 역할(상태표현·bottleneck·인과원인·scalar 최적화 대상)을 동시에 지우는 건 과하다 (§1.1) | **동의** | 07-21 야간 2런이 정확히 이 지점에서 실패 |
| 기존 consistency loss가 "올바른 belief"가 아니라 "과도한 action sensitivity"를 보상한다 (§1.2) | **동의 — 실측으로 확인됨** | margin 없는 원식에서 cons_loss 10.71 → −13.85로 부호를 넘겨 계속 밀렸고 reward 0.34 → 0.115 붕괴 |
| full-trace DPO는 차이의 출처를 분리하지 못한다 (§10.1) | **동의** | 기존 DPO trainer가 completion 전체를 단일 preference로 다룸 |
| candidate ID 분류가 문자열 길이·토크나이즈 편향을 제거한다 (§5.2) | **동의** | 오늘 후보 스코어링 구현 중 실제로 길이·경계 병합 버그를 겪음 |
| 평가를 G1/G2로 분해해 보고해야 한다 (§16) | **강하게 동의** | §2가 그 이유다 |

### 1.2 ★ 틀린 것 — §4의 수치 가정

원안은 `P(G1)=0.34, P(G2)=0.36, r=0.03`을 놓고 `c > 0.473`이면 50%가 가능하다고 결론짓는다. 세 값이 모두 실측과 다르다.

**실측 (heldout 전량 n=1,417):**

```
P(G1) = 0.399     P(G2) = 0.227     P(G3=OUT) = 0.374
상한 R5 = P(G1)+P(G2) = 0.626
```

`P(G2)`가 0.36이 아니라 **0.227**이다. G2는 원안이 생각한 것보다 1.6배 작은 표적이다.

그리고 결정적으로, `r`(G1 퇴행률)이:

| arm | acc | G1 보존 | **G1 퇴행 r** | G2 교정 c |
|---|---:|---:|---:|---:|
| `belief_sum_wm` (07-21) | 0.248 | 0.443 | **0.557** | 0.305 |
| `belief_sum_gt` (07-21) | 0.237 | 0.403 | **0.597** | 0.327 |
| `f0wema_final` | 0.280 | 0.497 | **0.503** | 0.382 |
| `f0gr_final` (belief 없음, 최고) | 0.338 | 0.626 | **0.374** | 0.415 |
| `base_1f_histonly` (최악) | 0.210 | 0.364 | **0.636** | 0.301 |

**40개 arm 전수 조사에서 `r`의 범위는 0.374 ~ 0.636이다. `r=0.03`에 근접한 arm은 존재하지 않는다.**

### 1.3 ★ 그래서 우선순위가 뒤집힌다

40개 arm에 대해 상관을 냈다.

```
corr(acc, G1 보존율) = +0.930      G1 보존율 범위 0.364 ~ 0.626 (폭 0.262)
corr(acc, G2 교정률) = +0.825      G2 교정률 범위 0.268 ~ 0.422 (폭 0.154)
```

이 프로젝트에서 지금까지 관측된 **모든 정확도 차이는 사실상 G1 보존율의 차이다.** G2 교정률은 변동 폭 자체가 절반이고 기여도 작다.

`Acc = 0.399 × (G1 보존) + 0.227 × (G2 교정)` 로 투영하면:

| G1 보존 | G2 교정 | 예상 Acc | 해석 |
|---:|---:|---:|---|
| 0.443 | 0.305 | 0.246 | 현재 |
| 0.497 | 0.382 | 0.285 | F0-W-EMA |
| 0.626 | 0.415 | 0.344 | 역대 최고 (`f0gr_final`) |
| **0.900** | 0.305 | **0.428** | **G1만 고치고 G2는 방치** |
| 0.443 | 0.600 | 0.313 | G2만 고치고 G1은 방치 |
| 0.900 | 0.500 | 0.473 | 둘 다 |

**G1만 고쳐도 0.246 → 0.428이다. G2를 0.305에서 0.60까지 두 배로 올려도 0.313에 그친다.**

가장 뼈아픈 사실: **WM top-1을 그냥 따르기만 하면 0.399다.** 역대 최고 arm(0.338)조차 이 무학습 베이스라인을 못 넘었다. 지금까지의 모든 학습은 순효과가 음수였다.

> **결론: 원안의 objective 세 항 중 `L_keep`(G1 보존)이 주 항이고 `L_G2-margin`이 보조다. 원안은 이 둘의 위상을 반대로 놓았다.**

### 1.4 원안이 놓친 진단 — G2 실패의 형태

G2에서 틀렸을 때 무엇으로 가는지 셌다. `belief_sum_wm` 기준 **44.5%가 WM top-1도 GT도 아닌 제3의 오답**이다.

원안 §3.2는 G2의 문제를 "WM prior를 그대로 복사한다"로 규정하고 hard negative로 WM top-1을 지목한다. 그런데 실측은 정책이 WM을 *복사*하는 게 아니라 **산개**하고 있음을 보여준다 (`wm_follow`도 0.339로 낮다). WM top-1만 hard negative로 눌러도 나머지 세 후보로 흩어질 수 있다.

→ pairwise margin보다 **listwise CE(원안 §5.1)가 더 적합**하다는 근거이며, 원안의 우선순위(candidate CE > G2 margin)와는 일치한다.

### 1.5 유보 — belief 완전 폐기에 대해

원안 §13의 "belief direct reward / swap consistency 제거"에는 동의한다. 다만 §18의 "belief-action causality 주장 폐기"는 **지금 결정할 필요가 없다.**

이유: `Δ_belief = Acc(x,C,b) − Acc(x,C)` (원안 §14)는 **학습 없이 추론만으로 측정 가능**하고, 이 값이 0인지 양수인지가 belief를 논문에서 어떻게 다룰지를 결정한다. 아직 아무도 이 값을 재지 않았다. 폐기 결정 전에 측정하는 것이 순서다 — 아래 **Exp-0**에 넣었다.

---

## 2. 수정된 실행 계획

원안 §15의 A~E를 유지하되 **우선순위를 G1 중심으로 재배열**하고, 무학습 진단 하나를 앞에 추가한다.

### Tier 0 — 학습 없는 진단 (GPU 2h)

| ID | 내용 | 답하는 질문 |
|---|---|---|
| **Exp-0a** | 후보 스코어링만으로 heldout 전량 평가 (생성 없이). base / F0-W-EMA / `f0gr_final` 3 arm | 생성이 병목인가 후보 판별력이 병목인가. **G1 퇴행이 디코딩 아티팩트인지 아닌지가 여기서 갈린다** |
| **Exp-0b** | 같은 prefix에서 belief 유/무만 바꿔 `Δ_belief` 측정 | belief를 폐기해도 되는지의 유일한 정량 근거 |

**Exp-0a가 이 계획 전체의 분기점이다.** G1 퇴행 0.55가 "모델이 WM top-1을 후보 중 최선으로 평가하지 못한다"면 학습이 필요하다. 반대로 스코어링에서는 G1을 잘 맞히는데 생성에서만 놓친다면, **문제는 objective가 아니라 디코딩**이고 원안의 재설계 전체가 과잉대응이 된다.

### Tier 1 — 핵심 3런 (GPU 8h)

원안 §15 A/B/C. 단 보고 지표를 G1 중심으로 바꾼다.

| ID | Objective | 1차 판정 지표 |
|---|---|---|
| **Exp-A** | `L_action` (candidate CE) | **G1 보존율** ≥ 0.85 |
| **Exp-B** | `+ λ_G2 · L_G2-margin` | G1 보존 유지 하에 G2 교정 상승 |
| **Exp-C** | `+ λ_keep · L_keep` (G1에서 F0 참조 KL) | G1 보존 ≥ 0.90 · G2 교정 비퇴행 |

**Exp-A는 이미 구현되어 있다.** `scripts/step2/pro_gx_train.py:104-116`이 정확히 `-log softmax(cand_lp)[gt_idx]`다. 새로 짤 것은 평가 글루뿐이다.

### Tier 2 — 구조 실험 (GPU 12h)

| ID | 내용 | 비고 |
|---|---|---|
| **Exp-D** | Residual reranker `z = z_F0 + δ_θ` | F0 후보 logit 캐시 필요 (1회 1.4h) |
| **Exp-E** | Reasoning ablation 5종 | 3종이 생성을 요구해 가장 비쌈 |

---

## 3. Ablation 설계

### 3.1 주 ablation (누적) — Tier 1

`L0` 무학습 WM top-1 (0.399) → `A` candidate CE → `A+B` +G2 margin → `A+B+C` +G1 KL

**모든 표에 `L0` 행을 강제한다.** 지금까지 이 베이스라인을 표에 넣지 않아서, 아무도 0.399를 못 넘었다는 사실이 3일간 드러나지 않았다.

### 3.2 스코어 정의 ablation (원안 §5.2, GPU 0 — 캐시 재사용)

1. candidate ID 분류 (`[A]~[E]`)
2. action span **길이 정규화** log-prob
3. action span **sum** log-prob (비정규화 — 길이 편향 확인용)
4. full-trace likelihood (원안이 피하라고 한 것 — 실제로 나쁜지 확인)

한 번 학습한 모델에서 스코어 정의만 바꿔 재평가하므로 **추가 학습 0**이다.

### 3.3 G1 보존 강도 (Exp-C 내부)

`λ_keep ∈ {0, 0.1, 0.5}` — G1 보존과 G2 교정의 trade-off 곡선. `λ_keep=0`이 Exp-B이므로 실제 추가는 2점.

### 3.4 Reasoning ablation (Tier 2, 원안 §15-E)

| 변형 | 생성 필요 | 비용 |
|---|---|---|
| direct candidate classification | ✗ | Exp-A 재사용 (0) |
| reasoning → action | ✓ | 2h |
| action-first → explanation | ✓ | 2h |
| belief + reasoning + action | ✓ | 2h |
| belief 제거 + reasoning + action | ✓ | 2h |

### 3.5 하지 않을 것

- belief reward λ 스윕 · belief swap consistency 확장 · full-trace DPO — 원안 §13에 동의
- 대규모 `λ`/`μ` 스윕 — Tier 1 결과 전에는 근거 없음

---

## 4. 소요 시간 — H200 1장

### 4.1 결론

| 범위 | GPU | 벽시계 | 산출 |
|---|---:|---:|---|
| **Tier 0** (진단) | 2h | **3h** | 방향 판정 |
| **+ Tier 1** (A/B/C) | 10h | **14h** | 핵심 결과 |
| **+ Tier 2** (D/E) | 22h | **28h** | 논문용 전체 |

### 4.2 왜 그만큼인가

**측정된 처리율을 근거로 한다:**

| 실측 | 값 | 출처 |
|---|---:|---|
| 생성 포함 RL | 2.74 s/샘플 | 5,000샘플 3.8h |
| 생성 + 후보 스코어링 | 4.0 s/샘플 | 오늘 P3 런 |
| → **스코어링 단독 (추정)** | **1.2 s/샘플** | 위 둘의 차 |
| 생성 평가 n=1,417 | 13분 | eval_harness_v2 |

**Tier 0 (2h):** 후보 스코어링 평가 1,417 × 1.2s = 28분/arm. 3 arm + belief 유무 2조건 = 약 2h. **학습 0.**

**Tier 1 (10h):**
- 학습: 5,000샘플 × 1.2s = 1.7h/에폭. 1에폭 기준 3런 = **5h** (A는 코드가 이미 있어 즉시 시작 가능)
- F0 참조 후보 분포 캐시 (Exp-C용): 4,998 × 1.0s = **1.4h**, 1회만
- 평가: 3 arm × 30분 = **1.5h**
- 여유 **1.1h**

**핵심 절감 이유:** 이 계획에는 **teacher 데이터 빌드가 없다.** 직전에 검토한 hindsight SFT 안은 teacher 빌드가 9.9h로 전체의 3분의 1을 먹었다 (실측 7.14 s/샘플 × 4,998, gate 통과 53%로 샘플당 2.63회 시도). candidate CE는 `(x, C_WM, a_GT)`를 데이터셋에서 그대로 쓰므로 그 항목이 통째로 사라진다. **이것이 원안 채택 시 가장 큰 실무적 이득이다.**

**Tier 2 (+12h):** Exp-D 학습 2h + 캐시 재사용, Exp-E 생성 4변형 × 2h = 8h, 평가 2h.

**코딩 (GPU 0, 약 10h):** Exp-A 평가 글루 1h · G2 margin 1.5h · 참조 KL + 캐시 2.5h · residual 3h · reasoning 변형 2h. 여기에 **오늘 만든 자산이 직접 기여한다** — `build_candidate_batch`/`score_candidates`(단위테스트 통과)가 §5.1의 `s_i` 계산 그대로이고, 방금 쓴 G1/G2 분해 스크립트가 §16 평가표의 8할이다. 코딩은 데이터 준비가 없어 GPU와 겹치기 어려우므로 벽시계에 대체로 드러난다.

### 4.3 불확실성

- **가장 큰 변수는 Tier 0의 결과다.** Exp-0a에서 "스코어링으로는 G1을 잘 맞힌다"가 나오면 Tier 1의 절반이 불필요해지고 문제는 디코딩으로 옮겨간다. 그래서 Tier 0을 먼저 돌린다.
- 스코어링 1.2 s/샘플은 차분으로 얻은 추정이다. Tier 0에서 실측되며, 2배 빗나가도 Tier 1은 10h → 13h로 늘 뿐이다.
- 에폭 수를 3으로 늘리면 Tier 1 학습이 5h → 15h가 된다. **1에폭으로 시작하고 곡선을 보고 결정한다.**

---

## 5. 사전 등록 판정 기준

실행 전에 고정하고 결과를 보고 바꾸지 않는다.

**Gate 0 (Tier 0):** 3개 arm에 대해 후보 스코어링 acc·G1 보존·G2 교정이 산출됨 · `Δ_belief`가 CI와 함께 보고됨

**Gate A (Exp-A):** `G1 보존율 ≥ 0.85` (현재 0.443, F0-W-EMA 0.497) · overall acc가 **무학습 베이스라인 0.399를 초과** · G2 교정률이 0.30 밑으로 떨어지지 않음

> 0.399 초과를 명시적 gate로 둔다. 이 프로젝트에서 아직 아무 모델도 달성하지 못했고, 달성 못 하면 학습의 순효과가 여전히 음수다.

**Gate B:** Gate A 유지 + G2 교정률 상승분의 paired CI 하한 > 0 · `G2 non-GT switch` 증가 없음

**Gate C:** G1 보존 ≥ 0.90 · Gate B 유지 · calibration(ECE) 악화 없음

**중단 조건 (원안 §17 채택):** G2 교정이 올라도 G1 퇴행이 더 크게 증가 · loss는 내려가는데 `GT−WM1` margin이 개선되지 않음 · 특정 candidate position 편향 발생

---

## 6. 보고 형식

모든 결과표에 다음을 강제한다.

- **`L0` = WM top-1 무학습 (acc 0.399) 행을 항상 포함**
- G1 보존율 / G1 퇴행률 / G2 교정률 / G2 non-GT switch를 **분리** 기재
- n 명시 (전량 1,417 · G1 566 · G2 321 · OUT 530)
- 구조적 상한 0.626 각주 (OUT 37.4%)
- paired delta + 95% CI. 단일 arm 절대값으로 우열 주장 금지 (MDE ≈ 0.045)
- `run_config.json` (git SHA · 데이터 sha256) 첨부

---

## 7. 지금 주장하면 안 되는 것

원안 §18에 다음을 **추가**한다.

- ~~belief intervention으로 causality를 학습했다~~ (원안)
- ~~자연어 belief가 faithful latent state로 작동한다~~ (원안)
- **G2 교정이 이 문제의 핵심 병목이다** ← 실측상 G1 퇴행이 3배 큰 항이다
- **belief 제거가 정확도를 올렸다** ← `f0gr_final`이 최고인 것은 사실이나 프롬프트 레짐이 달라 직접 비교 불가. Exp-0b의 `Δ_belief`로만 주장한다

---

## 8. 진행 중인 작업과의 관계

P3 belief-swap consistency 런은 이 문서 작성 시점에 진행 중이며 **중단하지 않는다.** 이유:

1. hinge(`margin=0.5`)가 실제로 폭주를 막는지가 이미 확인됐다 — `cons_loss`가 `seen=800`부터 0.29~0.37에서 정지 (margin=0 런은 같은 구간에서 −13.85). 이 공학적 사실은 재사용 가능한 지식이다.
2. 원안 §1.2의 진단("unbounded objective가 action preference destruction을 허용한다")에 대한 **직접 증거**를 남긴다. 원안을 채택하는 근거 자체가 된다.
3. 이 런의 ③ 상승은 **belief causality의 증거로 주장하지 않는다** (지표를 직접 최적화하는 구조이므로). 결과가 나오기 전에 사전 등록해 둔다.

다만 관측된 `reward_ma` 하락(0.425 → 0.245)은 원안 §1.2가 예측한 열화가 hinge 하에서도 약하게 진행됨을 시사한다. 최종 평가의 G1/G2 분해에서 확인한다.

---

## 9. 근거

**실측 (2026-07-21):**
- 그룹 분포·G1/G2 분해: `EGO/runs/f0_battery/*.records.jsonl` 40 arm + `heldout_1f_root/.../grpo_heldout_1f.jsonl` (n=1,417)
- 복창 제외 ③ 재집계: `EGO/runs/retro_overnight/causal_excl_restate_{gt,wm}.json`
- teacher 빌드 처리율: `EGO/runs/f0_battery/b0_r1/build_{0,1}.log` (7.14 s/샘플, gate 53.0%, G2 40.0%)
- consistency 폭주/hinge: `EGO/runs/p3_cons/full_FAILED_runaway_oom/gr_log.jsonl` vs `full/gr_log.jsonl`

**코드:** `scripts/step2/pro_gx_train.py:104-116` (candidate CE, Exp-A 기구현) · `pro_gr_train.py:build_candidate_batch/score_candidates` (스코어러, 단위테스트 `tests/step2/test_cons_mask.py`) · `eval_harness_v2.py` (전량 평가·CI·MDE) · `recount_causal_excl_restatement.py`

**문서:** `belief_reward_action_centric_handoff.md` (원안) · `2026-07-21_step1_night_and_retro_belief_sum_handoff.md`
