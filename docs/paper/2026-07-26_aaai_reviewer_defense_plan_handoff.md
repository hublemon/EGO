# AAAI reviewer defense — 예상 공격 대응 계획 handoff

> **[SUPERSEDED 2026-07-27]** 이 문서는
> `docs/paper/2026-07-27_aaai_reviewer_defense_plan_v2_handoff.md` 로 대체되었다.
> `docs/paper/2026-07-27_ablation_plan_audit_handoff.md` 감사에서 확인된 오류:
> ρ=0 arm 부재(§8-4 오인), `sft_r15_c` 오분류, gx 최우선 판단, `paired_boot.py` 플래그 부재,
> `last0`/`nohist` 중복, 학습시간 최대 3배 과대, G-NH 실패의 성격 오독.
> **공격 14건 ↔ 대응 매핑(§3·§4·§5)과 사전등록 결정규칙(§6)은 v2에서도 유효하므로 참조 가치가 있다.**

- 작성일: 2026-07-26
- 대응 대상: `docs/paper/2026-07-26_aaai_reviewer_attack_handoff.md` (가상 판정 **Weak Reject 4/10**, confidence 4/5)
- 방어 대상 원고:
  - `docs/paper/2026-07-26_embodied_reasoning_results.tex` (Results 초안)
  - `../EGO_paper/EGO_AAAI27_EN/main.tex` (본문)
- 실행 환경: H200 143GB × 1 (현재 유휴), frame cache 공유(`runs/cesft_v2/frame_cache`)

---

## 0. 한 줄 결론

리뷰어 공격 14건을 실행 자산과 대조한 결과, **새 GPU 실험이 필요한 것은 4건뿐**이다.
6건은 이미 산출된 수치의 **보고 누락**, 2건은 **코드가 있는데 안 돌린 스테이지**,
2건은 **문구 하향**으로 끝난다. 따라서 대응은 "실험 목록"이 아니라
**보고 → 실행 → 재집계 → 신규실험**의 비용 오름차순 사다리로 짠다.

---

## 1. 계획을 바꾼 코드베이스 확인 사실

계획 수립 전에 실행 자산을 직접 확인했다. 아래 둘이 우선순위를 뒤집는다.

### 1.1 final EGO(`sft_r15_gx`)는 개입 실험 2종을 "안 돌린 것"뿐이다

`runs/cesft_v2_fp_gx/markers/` 에 존재하는 마커는 셋뿐이다.

```
S6_SFT_R15_GX_DONE                    07-26 11:35
S_FREEGEN_SFT_R15_GX_CAND_FREE_DONE   07-26 11:56
S7_EVAL_SFT_R15_GX_DONE               07-26 12:05
```

`S_STRIP_*`(history 개입)와 `S3H_*`(belief-swap 개입)가 **없다**. 같은 스테이지가
`runs/cesft_v2_fp/` 의 다른 arm에서는 각각 약 9분에 완주했다(마커 실측:
`S7_EVAL_CAND_FREE 05:14 → S_STRIP_CAND_FREE 05:23 → S3H_CAND_FREE 05:32`).

→ 리뷰어 A3·A5 및 핵심질문 Q8·Q11의 근거인 "final EGO 미측정"은 **약 20 GPU-분**의 문제다.
   전체 계획에서 비용 대비 이득이 가장 크다.

### 1.2 belief utility \(U_g\)는 이미 측정되어 있다 — Results가 안 실었을 뿐

`runs/cesft_v2_fp/markers/S3H_SFT_R15_DONE` 원문 발췌:

```json
"causal_sensitivity_ci":  {"belief": {"point": 0.0975, "lo": 0.07,   "hi": 0.1275}},
"utility_belief_only_ci": {"point": 0.0314, "lo": 0.0191, "hi": 0.0441},
"directional_dg_ci":      {"point": 0.3425, "lo": 0.295,  "hi": 0.3875},
"correct_switch": {"flip_rate_swap_b": 0.11, "mean_pgt_drop_on_flip": 0.0256, "n_flip": 44},
"gate_CC3_belief_only_utility": true
```

리뷰어가 A5에서 "없다"고 지적한 **belief-only utility \(U_g\), directional \(D_g\),
correct-switch가 전부 `harden_s3`의 표준 출력**이다. Results 초안이 sensitivity 한 줄만
옮겨 적고 나머지를 버렸다. 전 arm(`base`/`cand_free`/`theta_ce`)의 `*.harden_s3.json`에
동일 필드가 있다.

→ A5 "방향맹" 공격의 대부분은 실험 문제가 아니라 **보고 문제**다. GPU 0.

### 1.3 그 밖의 기확보 자산

| 자산 | 경로 | 쓰임 |
|---|---|---|
| history-strip 평가 | `tools/oom_opt/strip_eval.py` | `user_prompt`만 치환하는 배터리 클론 — 모든 history 교란의 템플릿 |
| DiD (arm×history) | `tools/did_history.py` | 이미 `DiD_history_theta_ce_vs_cand_free.json` 산출됨 |
| paired cluster bootstrap | `tools/paired_boot.py`, `tools/harden_paired.py` | estimand 재계산 |
| 후보 미제시 레짐 | `ego.step2_retrospection.eval.freegen` | train-cand × test-cand 2×2의 test 축 |
| GT-only 대조군 | adapter `cand_free` | 학습·평가 완료 |
| 프레임 마스킹 선례 | `train_grpo_action.py`의 `MASK_FRAME_PROB` / `JOINT_SYSTEM_PROMPT_MASKED` | no-image 조건 구현 참고 |
| 미사용 adapter | `sft_r15_c`, `sft_r30_c` | ρ ablation(ρ=0 대조군) 승격 가능성 — §8 |

### 1.4 비용 앵커 (마커 실측)

| 스테이지 | 규모 | 실측 |
|---|---|---|
| battery | n=1,000 | 10\~16분/arm |
| strip_eval | n=1,000 covered | 약 9분/arm |
| harden_s3 | n=400 | 약 9분/arm |
| freegen | n=500 | 5\~20분/arm |
| SFT(retro) 368 steps | — | **44분** (gx 실측 10:51:49 → 11:35:28) |
| select-CE 523 steps | — | 약 4\~5시간 (resume 로그 기준 추정) |

---

## 2. 대응 유형 분류

| 코드 | 유형 | 비용 |
|---|---|---|
| **RPT** | 보고 — 산출물이 이미 있고 Results에 안 실었을 뿐 | GPU 0 |
| **RUN** | 실행 — 코드가 있는데 안 돌린 스테이지 | 분 단위 |
| **AGG** | 재집계 — 로그된 records로 CPU 재계산 (신규 CPU 스크립트 포함) | CPU 수 시간 |
| **EXP** | 신규 실험 — 스크립트 신설 + GPU | 시간\~일 |
| **TXT** | 문구 — 주장 범위 하향, 실험 불필요 | 0 |

한 공격이 여러 유형을 동시에 요구할 수 있다.

---

## 3. 치명 공격 5건 — 공격 ↔ 대응

### A1. "embodied reasoning이 아니라 textual script completion" (원문 §2)

**공격.** 정책이 이미지·긴 completed-action history·의미가 명시된 Top-10 후보를 동시에 받는데,
image 제거·history-only·후보텍스트-only·transition prior 통제가 없다. 따라서 `+7.70pp`가
픽셀에서 온 건지 행동 이름 텍스트의 script completion인지 분리되지 않는다.
예상 리뷰 문장: *"The experiments establish trajectory-conditioned candidate selection,
not embodied visual reasoning."*

**추가 논리 문제.** WM도 observation과 history를 받는다. 그런데 결론은 "WM=visual boundary,
LM=trajectory reasoner"로 모듈을 분해한다. 후보 경계 자체가 history-conditioned라면
`history→WM→후보집합`과 `history→LM→선택` 두 경로가 섞인다. 현행 history-strip은 두 번째
경로만 끊고 후보집합은 원래 history로 만든 것을 고정한다.

**대응** — `AGG` + `EXP` + `TXT`

1. `AGG` **텍스트-only baseline 3종**(last-action repeat / history transition-prior /
   candidate 문자열 유사도)을 로그된 records만으로 계산. **image ablation 없이도 A1의 절반을
   막는 가장 싼 반박**이다. transition-prior가 20%인데 Cand.-CE가 28.8%면 그 차이가 텍스트
   통계 너머의 몫이다.
2. `EXP` **image × history 2×2 factorial**(C1\~C4) + shuffled/other-video image(C8).
3. `TXT` WM이 history를 쓰는 설계를 유지하는 한 `visually grounded action boundary` →
   **`observation- and history-conditioned proposal boundary`**로 하향. 비용 0인데 안 하면
   Q3에서 반드시 잡힌다.
4. `EXP`(선택, 고비용) history-free WM으로 후보 재생성 → WM 경로까지 분리 (T3-6).

**결정규칙.** no-image에서 Cand.-CE−GT-only 우위의 CI가 0을 제외하고 유지되면 시각 근거
주장을 철회하고 제목을 *candidate-aligned, trajectory-conditioned selection*으로 내린다.

---

### A2. "28.8%는 전체 과제 성능이 아니다" (원문 §3)

**공격.** headline `28.8%`는 GT가 이미 Top-10에 있는 covered subset의 conditional accuracy다.
held-out Coverage@10 `43.4%`, full-set equivalent 약 `12.5%`, frozen WM Top-1 full-set
`10.5%` → 전체 분포에서의 절대 개선은 약 `+2.0pp`. 즉 held-out의 `56.6%`에서는 selector가
정답을 고를 기회조차 없다. *"Why is 28.8% the headline rather than 12.5%?"*

**estimand 불일치.** 점추정은 malformed=오답 처리한 common covered `n=1,000`인데, 공식 paired
CI는 arm별 non-malformed intersection(948 / 957 / 937)을 쓴다. `non-malformed`는 모델 출력이
결정하므로 이를 조건으로 표본을 빼면 **arm-dependent post-treatment selection**이 생긴다.

**대응** — `AGG` + `TXT`

1. `AGG` 점추정·CI를 **모두 common covered `n=1,000`, malformed=incorrect**로 재계산
   (`tools/paired_boot.py`). non-malformed intersection은 secondary diagnostic으로 강등.
2. `AGG` arm별 malformed rate·paired attrition 표를 부록에 추가.
3. `TXT` 주 표 첫 열에 full-set equivalent를 두고, headline은 항상
   **`28.8% conditional / 12.5% full-set equivalent`** 쌍으로 표기.
4. `EXP`(여유 시) sampled 1,000이 아니라 held-out 전체 5,326 / covered 전체 2,313 재평가.

**성격.** 이 공격은 실험이 아니라 **보고 규율** 문제다. covered-set 분석 자체는 selector 모듈
진단으로 정당하므로, 방어는 "전면에 두지 않는다"이지 "버린다"가 아니다.

---

### A3. "최종 EGO가 Cand.-CE보다 좋아진 것이 없다" (원문 §4)

**공격.** Cand.-CE와 final EGO 모두 strict SelAcc `28.8%`, official paired 차이는
SelAcc `−0.64pp` / \(G_2\) `−2.53pp`, 사전등록 `1pp` 비-열등성 **FAIL**, final EGO의
belief-swap **미실행**. *"If the final method neither improves over candidate CE nor passes
its own preservation criterion, why is candidate CE not the entire paper?"*

**Figure 1(b) 공격.** `28.8% → 28.8%` 평평한 선이 "holding"처럼 보이는데, 본문은 중간
Retrospection-with-Replay 체크포인트가 `26.4%`로 하락 후 회복했다고 밝힌다. 실측점을 빼고
endpoint를 직선으로 이으면 training path를 유리하게 시각화했다는 공격을 받는다.

**대응** — `RUN` + `AGG` + `TXT`

1. `RUN` **final EGO의 `harden_s3`·`strip_eval` 실행**(§1.1, 약 20분). 이것만으로 Q8·Q11의
   "미실행" 근거가 사라진다.
2. `AGG` Figure 1(b)에 replay `26.4%`를 **실측점으로 추가**, 미측정 구간은 곡선 금지·점·점선만.
3. `TXT` 최종 단계의 역할을 **"improvement"가 아니라 "recovery after retrospection
   degradation"**으로 정확히 기술. final EGO가 Cand.-CE보다 우월하다는 인상 제거.
4. `TXT` 비-열등성 FAIL을 숨기지 않되, contribution 서술과 분리한다(기여는 candidate alignment).
5. `EXP`(고비용) Retrospection을 핵심 기여로 유지하려면 **독립 run에서 multi-seed 개선 또는
   사전등록 비-열등성 통과**가 필요. 불가하면 exploratory extension으로 강등.

---

### A4. "history removal만으로 인과 기제를 확립할 수 없다" (원문 §5)

**공격.** history 제거는 학습 때 항상 있던 입력 블록을 없애는 **OOD intervention**이다.
Cand.-CE의 `−10.1pp` drop은 (1) 유용한 trajectory reasoning을 학습했다 (2) history omission /
prompt-format shift에 더 취약해졌다 — 둘 중 어느 쪽으로도 설명된다. 또한 필요한 검정은
arm별 개별 CI가 아니라 difference-in-differences
\((\Delta_{\text{Cand.-CE}}-\Delta_{\text{GT-only}})\), \((\Delta_{\text{Cand.-CE}}-\Delta_{\text{Base}})\)
의 interaction CI인데 제시되지 않았다.

**대응** — `AGG` + `EXP` + `TXT`

1. `AGG` **arm×history interaction DiD를 본문 지표로 승격**. `tools/did_history.py`가 이미 있고
   `DiD_history_theta_ce_vs_cand_free.json`도 산출되어 있다. base 대비만 추가 실행.
2. `EXP` **semantic corruption ladder**: 같은 길이 shuffled history(C5) / other-video
   history(C6) / reversed history(C7). "입력 블록 제거"가 아니라 "의미만 파괴"하므로
   OOD 반론과 의미 사용 주장을 분리한다.
3. `EXP` **dose-response**: 최근 0 / 1 / 3 / 7 / all (C9\~C12). 단조 반응이면 OOD 절벽이
   아니라 실제 사용의 증거다.
4. `EXP`(고비용) history 없이 따로 학습한 control arm — 양가해석을 끊는 유일한 실험.
5. `TXT` 소제목 `Causal Test` → **`Inference-time History Ablation`**.

**결정규칙.** 의미붕괴 history의 손실이 no-history 손실과 통계적으로 구분되지 않으면
인과 주장을 철회하고 "history dependence under inference-time ablation"으로 쓴다.

---

### A5. "belief 결과가 '학습된 유용한 belief'를 보이지 않는다" (원문 §6)

**공격 4갈래.**
- *방향맹*: sensitivity는 belief를 바꾸면 top-1이 얼마나 바뀌는지만 재고, **정답 방향인지**는
  안 잰다. 무작위로 흔들려도 올라간다. 필요한 지표는
  \(U_g = p(a_{\mathrm{GT}}\mid r_{\mathrm{own}},b_{\mathrm{own}}) - p(a_{\mathrm{GT}}\mid r_{\mathrm{own}},b_{\mathrm{swap}})\).
- *크기·검정*: Cand.-CE `0.093` vs Base `0.073` 차이는 `0.020`뿐이고 arm 간 차이의 CI가 없다.
- *OOD*: donor가 `(i+7) mod n`이고 reasoning은 원표본 것을 유지 → 의미적으로 모순된 hybrid
  prefix가 생성분포 밖일 수 있다. paraphrase flip 단순 차감이 semantic effect를 식별한다는
  보장도 없다.
- *echo 해석*: 낮은 echo는 "action 문자열을 복사하지 않았다"는 뜻뿐이며 decision-relevant를
  함의하지 않는다.

**대응** — `RPT` + `RUN` + `EXP` + `TXT`

1. `RPT` **\(U_g\)·\(D_g\)·correct-switch를 전 arm에 대해 보고**. §1.2대로 이미 측정되어 있다.
   `utility_belief_only_ci`, `directional_dg_ci`, `correct_switch` 필드를 Table 2로 승격.
2. `AGG` **Cand.-CE − Base의 paired CI**를 `*.harden_s3.records.json`에서 재계산.
3. `RUN` **final EGO의 belief-swap 실행**(약 9분). Table 2의 빈칸이 채워진다.
4. `EXP` donor를 **video-disjoint**로 제한 + donor seed 3개 반복.
5. `EXP` **hard-negative belief swap** — 동일 task·verb-class의 의미적으로 그럴듯한 donor.
   "모순된 hybrid prefix라 OOD" 반론을 정면으로 막는 유일한 통제.
6. `AGG` belief-only 조건의 GT 확률 변화를 **teacher-forcing forward로만** 산출(생성 불필요).
7. `TXT` \(U_g\)의 cluster CI가 0을 포함하면 `decision-relevant state` →
   **`action-sensitive generated prefix`**로 하향.

---

## 4. 추가 공격 9건 — 공격 ↔ 대응

| ID | 공격 (원문 §) | 대응 | 유형 | 비용 |
|---|---|---|---|---|
| **B1** | 단일 학습 run (§7.1). video-cluster bootstrap은 evaluation sampling만 다루고 seed·data order·adapter init·checkpoint selection 불확실성은 미반영. `+7.70pp`가 1회 run이면 방법 효과로 주장하기 어렵다 | `theta_ce`·`cand_free` × seed {1,2} 추가(base는 frozen이라 불필요), seed 평균·표준편차·seed별 paired 병기. 불가 시 single-run limitation 명시 | `EXP`/`TXT` | 20h |
| **B2** | GT-only가 충분한 통제 아님 (§7.2). Cand.-CE는 10후보 discriminative, GT-only는 answer-only — candidate exposure 외에 negative supervision·출력공간·optimization geometry가 다르다 | `rand_cand`(무작위 후보 CE) / `freq_cand`(전역 빈도 후보) / `gt_inbatch`(in-batch negative GT-only) 3 arm + **train-cand × test-cand 2×2**(test 축은 `freegen`으로 무료). 동일 token budget·target format·optimizer step | `EXP` | 14h |
| **B3** | capability axis가 독립 능력처럼 제시됨 (§7.3). \(G_1\)/\(G_2\)는 같은 SelAcc를 WM correctness로 조건부 분해한 것이고 continuation·evidence utility도 표본이 겹친다 | 각 조건부 지표의 **denominator와 CI 병기**, arm×subset interaction 검정, confirmatory/exploratory axis 분리, 분석 family 명시 | `AGG`/`TXT` | CPU |
| **B4** | continuation precision 해석 과장 (§7.4). `68–73%`가 비슷하다는 것만으로 "looser tendency to repeat"를 배제할 수 없다 | precision 차이의 CI, 반복 예측률, 직전 action=GT 비율, non-continuation subset의 false repetition rate, last-action baseline 보고. `stable precision rules out` → **`is consistent with`** | `AGG`/`TXT` | CPU |
| **B5** | evidence utility의 selection bias (§7.5). 언급 trace와 미언급 trace는 난이도·confidence·continuation 여부가 다르고 mention rate도 arm마다 크게 다르다 | 정규식 annotation의 agreement·precision/recall 보고, continuation·history length·WM rank 통제 regression, within-video/matched 분석. **mechanism 증거가 아니라 descriptive diagnostic으로만 유지** | `AGG`/`TXT` | CPU |
| **B6** | 정성 예시가 outcome-conditioned (§7.6). 선정 규칙이 "GT-only 오답 ∧ EGO 정답"을 조건으로 하므로 고정 규칙이어도 cherry-picking 반론이 남는다. 게다가 제시 예시의 EGO reasoning("just finished pressing the dough")이 실제 마지막 action(`cook flatbread`)과 불일치 | outcome-independent random + 성공/실패 stratified sample로 교체, 선정 기준의 **사전 고정 기록** 첨부, 예시별 frame crop 병기, **history factuality annotation** 추가 | `AGG`/`TXT` | CPU\~중간 |
| **B7** | 표준·강한 baseline 부재 (§7.7). 비교가 자체 Base/GT-only/WM Top-1뿐이라 full-set `12.5%`가 강한지 판단 불가 | 무료분(last-action, transition prior, candidate similarity)은 A1 대응과 공유. 유료분으로 frontier VLM selector와 oracle candidate ranker 추가 | `AGG`/`EXP` | CPU + 중간 |
| **B8** | train–heldout coverage shift (§7.8). WM train Coverage@10 `71.6%` vs held-out `43.4%`, 학습은 covered example만 사용 → WM이 이미 잘 맞히는 쉬운 사례에 supervision 편중 가능 | train/heldout GT rank 히스토그램, **rank-stratified SelAcc·\(G_2\)**, inverse-frequency 또는 heldout-like reweighting, coverage shift limitation 명시 | `AGG`/`TXT` | CPU |
| **B9** | 단일 GT의 과제 모호성 (§7.9). 평균 12.8s 뒤 행동을 exact label 하나로 채점하면 여러 행동이 합리적일 수 있다 | time-to-target별 성능, action-frequency·ambiguity별 분석은 records로 즉시 가능. human agreement / multiple-valid-action 평가는 비용이 커서 limitation 명시로 대체 | `AGG`/`TXT` | CPU |

---

## 5. 핵심 질문 13개 ↔ 대응 매핑

리뷰어가 rebuttal에서 한 문장씩 답을 요구할 항목이다. 각 질문이 어느 대응으로 닫히는지 고정한다.

| # | 질문 | 닫는 대응 |
|---|---|---|
| Q1 | 이미지가 없어도 같은 향상이 나는가? | A1-2 (C3 no-image) |
| Q2 | 다른 비디오 history를 넣으면 떨어지는가? | A4-2 (C6) |
| Q3 | WM도 history를 받는데 왜 boundary를 purely visual이라 부르는가? | A1-3 문구 하향 (필수, 비용 0) |
| Q4 | 왜 28.8%가 headline이고 12.5%는 각주인가? | A2-3 병기 |
| Q5 | malformed=오답 점추정에 왜 non-malformed subset CI를 붙였는가? | A2-1 estimand 통일 |
| Q6 | Cand.-CE 효과가 여러 seed에서 재현되는가? | B1 (또는 limitation 명시) |
| Q7 | 동일 negative supervision을 가진 GT-only control이 있는가? | B2 |
| Q8 | final EGO가 나아진 지표는 무엇이며 CI가 있는가? | A3-1 `RUN` + A3-3 서술 |
| Q9 | 비-열등성 실패인데 왜 최종 방법인가? | A3-3·A3-4 서술 재정의 |
| Q10 | sensitivity 0.073→0.093 증가가 유의한가? | A5-2 paired CI |
| Q11 | swapped belief가 GT 확률을 올바른 방향으로 바꾸는가? | A5-1 \(U_g\)·\(D_g\) 보고 + A5-3 `RUN` |
| Q12 | 정성 예시의 EGO trace가 마지막 action을 잘못 기술하는데 왜 state-bearing인가? | B6 예시 교체 + factuality annotation |
| Q13 | 개선이 pixel grounding이 아니라 action-history LM이라는 설명을 무엇이 배제하는가? | A1-1 텍스트 baseline + A1-2 factorial |

---

## 6. 사전등록 결정규칙 — 실험보다 먼저 고정한다

리뷰어를 이기는 것은 실험 개수가 아니라 **"이 실험이 실패하면 어떤 주장을 내리는가"를
실험 전에 써두는 것**이다. 아래 표를 논문 부록에 그대로 싣고, **Tier 1 착수 전에 커밋해
타임스탬프를 남긴다**(B6의 사후선택 반론에 대한 일반 방어이기도 하다).

| 유지하려는 주장 | 식별 실험 | 반증조건 → 강등 후 표현 |
|---|---|---|
| embodied **visual** reasoning | image × history 2×2 | no-image에서 Cand.-CE−GT-only 우위 CI가 0을 제외하고 유지 → *candidate-aligned, trajectory-conditioned selection* |
| history의 **의미적** 사용 | shuffled / other-video history | 의미붕괴 손실이 no-history 손실과 구분 불가 → *inference-time history ablation, OOD 취약성 배제 불가* |
| state-bearing belief | \(U_g\) + hard-negative donor | \(U_g\)의 cluster CI가 0을 포함 → *action-sensitive generated prefix* |
| candidate exposure가 원인 | `rand_cand` / `freq_cand` / `gt_inbatch` | random-candidate CE가 동등 이득 → *negative supervision 효과*로 재서술 |
| final EGO가 방법의 일부 | 독립 run 비-열등성 재검정 | 재실패 → Retrospection을 exploratory extension으로 강등, 기여를 candidate alignment로 이동 |
| WM 경계가 **visual** | history-free WM 후보 재생성 | 미실행 → *observation- and history-conditioned proposal boundary*로 하향 (비용 0, 필수) |

**정직한 사전 선언.** C3(no-image)에서 Cand.-CE의 우위가 상당 부분 살아남을 가능성이 낮지
않다고 본다. 그 경우 위 규칙대로 **경로 A(제목·abstract에서 `embodied` 하향)를 rebuttal이
아니라 본문에서 먼저 택한다.** 리뷰어가 예상한 결론을 우리가 먼저 쓰면 그것은 공격이 아니라
방법론적 엄밀성의 증거가 된다.

---

## 7. 실행 계획 — 4 tier

### Tier 0 — GPU 0, 기존 로그 재집계 (즉시)

| ID | 작업 | 닫는 공격 |
|---|---|---|
| T0-1 | estimand 통일: 점추정·CI 모두 common covered `n=1,000`, malformed=incorrect | A2, Q5 |
| T0-2 | arm별 malformed rate·paired attrition 표 | A2 |
| T0-3 | \(U_g\)·\(D_g\)·correct-switch 전 arm 표기 + Cand.-CE−Base paired CI | A5, Q10·Q11 |
| T0-4 | arm×history interaction DiD를 본문 지표로 승격 | A4, Q13 |
| T0-5 | **텍스트-only baseline 3종** (`tools/text_baselines.py` 신규, CPU) | **A1**, B4, B7, Q13 |
| T0-6 | full-set `12.5%` 병기 (headline 쌍 표기) | A2, Q4 |
| T0-7 | Figure 1(b)에 replay `26.4%` 실측점 추가, 미측정 구간 점선화 | A3 |
| T0-8 | rank-stratified SelAcc + train/heldout GT-rank 히스토그램 | B8 |
| T0-9 | 문구 하향 일괄 (§6 표의 강등 표현 사전 적용분) | A1·A4·A5·B4, Q3 |
| T0-10 | final EGO 실패를 contribution 서술과 분리 | A3, Q9 |
| T0-11 | 조건부 지표 denominator·CI 병기, confirmatory/exploratory 분리 | B3 |
| T0-12 | evidence utility를 descriptive diagnostic으로 고정 + 통제 regression | B5 |

**T0-5를 가장 먼저 돌린다.** 결과가 경로 A/B 선택의 1차 신호다. transition-prior가
Cand.-CE에 근접하면 Tier 1을 돌리기 전에 이미 경로 A를 택해야 한다.

### Tier 1 — 하룻밤 (~8 GPU-h), 핵심 식별 실험

4 arms(`base`, `cand_free`, `theta_ce`, `sft_r15_gx`) × 아래 조건.
**지표는 arm별 drop이 아니라 arm × condition interaction의 paired video-cluster bootstrap CI.**

| # | 조건 | 구현 | 닫는 공격 |
|---|---|---|---|
| C1 | full (기측정) | — | 기준 |
| C2 | no-history | `strip_eval.py` (gx만 미실행) | A4 |
| C3 | **no-image** | blank 프레임 + masked system prompt | **A1, Q1** |
| C4 | no-image ∧ no-history | C2+C3 | 후보-prior floor |
| C5 | shuffled history | `rec["history"]` 셔플 | A4 |
| C6 | **other-video history** (길이 매칭, video-disjoint) | 도너 치환 | A4, **Q2** |
| C7 | reversed history | `[::-1]` | A4 |
| C8 | other-video image (history 유지) | `video_uid`/`obs_start_sec` 치환 | A1 |
| C9\~C12 | dose-response: 최근 0/1/3/7/all | `[-k:]` | A4 |

**구현 방침.** `tools/oom_opt/strip_eval.py`가 이미 "배터리와 동일하되 `user_prompt`만 치환"
패턴이므로, 이를 일반화한 **`tools/oom_opt/perturb_eval.py`** 하나를 만든다.
`--mode {nohist,noimage,nohist_noimage,shuffle,othervideo,reverse,lastk,othervideo_image}`.
history 조작은 전부 `rec["history"]` 인메모리 변형(`vlm.fmt_history` 재사용),
**후보 집합 `rec["candidates"]`는 모든 조건에서 불변**(WM 경로 고정, 정책 경로만 개입).
출력 스키마는 `{arm}_{mode}.records.jsonl`로 `strip_metrics.py`/`paired_boot.py`가 그대로
읽게 유지.

**구현 리스크.** Qwen3-VL이 이미지 0장을 허용하지 않을 수 있다 → 옛 트랙의 `MASK_FRAME_PROB`
경로와 동일하게 **blank 이미지 + 마스킹 system prompt**로 구현한다. 정책 입력 형식이 유지되어
"프롬프트 포맷 shift" 반론까지 함께 막는 이점이 있다.

### Tier 2 — 반나절 (~2.5 GPU-h), belief 기제 완결

| ID | 작업 | 닫는 공격 |
|---|---|---|
| T2-1 | **final EGO `harden_s3` 실행** (약 9분, 최우선) | A5, Q11 |
| T2-2 | donor를 video-disjoint로 제한 + donor seed 3회 반복 | A5 OOD |
| T2-3 | **hard-negative belief swap** (동일 task·verb-class donor) | A5 OOD |
| T2-4 | belief-only GT 확률 변화를 teacher-forcing forward로 산출 | A5 방향맹 |
| T2-5 | arm 간 sensitivity 차이의 paired CI 명시 (유의하지 않으면 그렇게 쓴다) | A5, Q10 |

### Tier 3 — 2\~3일, 교란요인 제거 (embodied 제목을 지키려면 필수)

| ID | 작업 | 비용 | 닫는 공격 |
|---|---|---|---|
| T3-1 | matched negative-supervision 3 arm + train×test 2×2 | 14h + 2h | **B2, Q7** |
| T3-2 | 3 seeds (`theta_ce`·`cand_free` × seed 1,2) | 20h | B1, Q6 |
| T3-3 | 학습된 통제 arm (image-free / history-free selector) | 2 × 4\~5h | A1, A4 |
| T3-4 | **K ablation (K=3/5/10)** — `main.tex` Table 4가 아직 placeholder다. "boundary가 판단을 좌우한다"는 본문 핵심 주장의 유일한 직접 증거인데 비어 있다. Top-10에서 잘라내면 WM 재학습 없이 K=3/5 생성 가능 | 2 × 1h | **본문 핵심 주장** |
| T3-5 | held-out 전체 5,326 / covered 전체 2,313 재평가 | 5\~8h | A2, B7 |
| T3-6 | history-free WM 후보 재생성 (Step-1 재학습 필요. 기존 `RETRO-goalstep-start-m1-lobs8`는 시간 계약이 달라 교란) | 높음 | A1 추가 논리, Q3 |
| T3-7 | 정성 예시 stratified 교체 + history factuality annotation | 낮음 | B6, Q12 |
| T3-8 | frontier VLM selector / oracle candidate ranker baseline | 중간 | B7 |

---

## 8. 실행 우선순위

| 순서 | 묶음 | 비용 | 생략 시 |
|---|---|---|---|
| **1** | §1.1 미실행 스테이지: `sft_r15_gx`의 `strip_eval` + `harden_s3` | **20\~30 GPU-분** | Q8·Q11 무방비. 비용 대비 손해 최대 |
| **2** | Tier 0 전체 (특히 T0-5 → T0-1 → T0-3) | CPU 수 시간 | A2·A5·A4통계 그대로 실점 |
| **3** | §6 결정규칙 표 커밋 (타임스탬프 고정) | 0 | B6 및 사후선택 반론 |
| **4** | Tier 1의 C3 / C5 / C6 | 하룻밤 일부 | 제목의 `embodied` 포기 |
| **5** | Tier 2 전체 | 2.5h | belief claim → exploratory 강등 |
| **6** | Tier 1 잔여 + T3-4 (K ablation) | 하룻밤 | 본문 핵심 주장 직접증거 부재 |
| **7** | T3-1 → T3-2 | 2\~3일 | single-run limitation 명시로 부분 대체 |

**순서 1\~5(하룻밤 + CPU)만으로 Weak Reject 근거 5개 중 3개가 사라진다.**
순서 6까지 가면 제목 유지 여부를 데이터가 결정하게 만들 수 있다.

### 열린 판단 사항

1. **경로 A vs B 결정 시점** — T0-5 결과를 보고 Tier 1 착수 전 1차 판단.
   transition-prior가 Cand.-CE에 근접하면 Tier 1의 값어치가 떨어지고 경로 A로 직행이 합리적.
2. **T3-6(history-free WM) 마감 내 실행 여부** — 불가 판단이면 T0-9 문구 하향을 최종안으로
   확정하고 limitation으로 명시한다.
3. **T3-1 vs T3-2 GPU 배분** — 단일 GPU라 동시 불가. 리뷰어 가중치상 B2(Q7)가 B1(Q6)보다
   반박 난도가 높으므로 **T3-1 우선**. T3-2는 single-run limitation 명시로 부분 대체 가능하지만
   T3-1은 대체 불가.
4. **`sft_r15_c` / `sft_r30_c` adapter 용도 확인** — Results에 등장하지 않는다.
   `main.tex` L205가 약속한 **ρ=0 대조군 ablation**으로 승격 가능한지 확인. 가능하면
   Tier 3에서 하나가 무료로 해결된다.

---

## 9. 실행 커맨드 부록

공통 환경 (기존 체인과 동일):

```bash
REPO=/mnt/nvme/migration/jihun/EGO_jihun3
PY=/mnt/nvme/migration/jihun/envs/miniforge3/envs/eve-cu124/bin/python
export PYTHONPATH=$REPO/src HF_HOME=/mnt/nvme/cache
export FRAME_CACHE_DIR=$REPO/runs/cesft_v2/frame_cache      # 재추출 금지
export RETRO_NEXT_GAP_TEXT="after the current action ends"  # 시간 계약 불변
CFG=configs/step2_retrospection/cesft_v2_fp.yaml
ADAPT=outputs/step2_retrospection/cesft_v2_fp
```

### 순서 1 — final EGO 미실행 스테이지 (즉시)

```bash
export RETRO3_RUNS=runs/cesft_v2_fp_gx

# history 개입 (약 9분)
$PY tools/oom_opt/strip_eval.py --config $CFG --arm sft_r15_gx \
    --adapter $ADAPT/sft_r15_gx/adapter --eval_n 1000 --covered_only

# belief-swap 개입 (약 9분) — U_g·D_g·correct-switch 동시 산출
$PY -m ego.step2_retrospection.eval.harden_s3 --config $CFG --arm sft_r15_gx \
    --adapter $ADAPT/sft_r15_gx/adapter --n 400
```

**선행 확인**: `runs/cesft_v2_fp_gx/data/context_val.jsonl`이 `runs/cesft_v2_fp`와 동일
코호트인지 확인할 것(동일해야 paired 비교 성립). 다르면 `RETRO3_RUNS=runs/cesft_v2_fp`로
돌리고 adapter만 gx를 가리킨다.

### 순서 2 — Tier 0 재집계 (GPU 불필요)

```bash
# estimand 통일: common covered n=1000, malformed=incorrect
$PY tools/paired_boot.py --run runs/cesft_v2_fp --arm_a theta_ce --arm_b cand_free \
    --common_set --malformed_as_incorrect \
    --out runs/cesft_v2_fp/eval/paired_commonset_ce_vs_gtonly.json
# WM Top-1 대비, final EGO 대비도 동일 형식으로 반복

# arm×history DiD (기산출분 재사용 + base 대비 추가)
$PY tools/did_history.py --run runs/cesft_v2_fp --arm_a theta_ce --arm_b base

# 텍스트-only baseline (신규, CPU) — records만 사용
$PY tools/text_baselines.py --run runs/cesft_v2_fp \
    --baselines last_action,transition_prior,cand_similarity
```

### 순서 4·6 — Tier 1 교란 스윕

`tools/oom_opt/perturb_eval.py` 구현 후:

```bash
export RETRO3_RUNS=runs/cesft_v2_fp
for arm in base cand_free theta_ce sft_r15_gx; do
  case "$arm" in base) AD="";; *) AD="$ADAPT/$arm/adapter";; esac
  for mode in noimage nohist_noimage shuffle othervideo reverse othervideo_image \
              last0 last1 last3 last7; do
    $PY tools/oom_opt/perturb_eval.py --config $CFG --arm "$arm" ${AD:+--adapter "$AD"} \
        --mode "$mode" --eval_n 1000 --covered_only
  done
done

# 집계: arm×condition interaction, paired video-cluster bootstrap
$PY tools/strip_metrics.py --run runs/cesft_v2_fp --interaction --cluster video_uid
```

무인 실행 시 기존 `supervisor.sh` + 마커 멱등 패턴(`run_stage`)을 따를 것.
마커 이름은 `S_PERTURB_{ARM}_{MODE}_DONE` 규약을 쓴다.

---

## 10. 참조

- 공격 원문: `docs/paper/2026-07-26_aaai_reviewer_attack_handoff.md`
- Results 초안: `docs/paper/2026-07-26_embodied_reasoning_results.tex`
- 본문: `../EGO_paper/EGO_AAAI27_EN/main.tex`
- 실행 체인 선례: `scripts/step2_retrospection/cesft_fp_chain.sh` (마커 멱등·preflight 패턴)
- 산출물: `runs/cesft_v2_fp/eval/`, `runs/cesft_v2_fp_gx/eval/`
