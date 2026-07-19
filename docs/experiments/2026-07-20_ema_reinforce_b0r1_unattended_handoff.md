# F0 최적화 재편(EMA-REINFORCE) · B0-R1 리팩터 통합 핸드오프 — 리뷰용

작성: 2026-07-20 01:00 KST · 대상: 코드 리뷰 담당자 (현재 구현을 처음 보는 사람 기준)
목적: 지난 24시간의 실험 결과 → 문제 진단 → 설계 결정 → 현재 무인 실행까지의 전체 논리와,
**각 방법론이 어느 코드에 구현되어 있는지**를 한 문서로 전달한다. 연구 의의 훼손이나 구현
결함이 있는지 코드 수준에서 확인받기 위한 문서다. 본 커밋 기준 main 에 전부 반영되어 있다.

---

## 0. 3분 배경 (처음 읽는 리뷰어용)

- **과제**: EK100 1인칭 프레임 + 행동 히스토리 + V-JEPA2 월드모델(WM)의 joint top-5 후보를 받아
  VLM(Qwen3-VL-8B + LoRA r16)이 `<reasoning>→<task_belief>→<action>` full-trace 를 생성하며
  다음 행동을 고르게 한다.
- **참조 눈금** (heldout 500): 무작위 0.200 · WM top-1 복사 시 0.374 · GT∈top5 전부 정답 시(oracle) 0.620.
- **트랙**: **extro(F0)** = 온라인 RL(보상 기반), **intro(B0)** = teacher trace 증류(DPO).
  ①~④ 검증 배터리 중 이 문서에서 자주 쓰는 것: **③ belief-swap 인과 민감도**(belief 를 바꾸면
  action 이 바뀌는가 — intro 의 주 지표), **G2**(GT 는 top5 안, WM top-1 은 오답인 부분집합 —
  "VLM 이 WM 을 이길 수 있는가"의 직접 지표, chance 0.2).

---

## 1. F0 개선안 실행 결과 → 문제 진단 (7/19 낮~밤)

리워드 오염(P1-1: think_convergence 사후-언급 보상)을 제거한 재편 실행 결과:

| arm (전 조건 동일, 보상만 차이) | 보상 | acc | G2 | 판정 |
|---|---|---|---|---|
| F0-N base | — | 0.242 | 0.309 | 기준선 |
| F0-W | WM likelihood 단독 (GT-free) | 0.240 | **0.333** | acc 평탄, judge +0.45 유일 상승 |
| F0-G | **GT 바이너리** (oracle-subset) | 0.244 | 0.301 | **GT 를 줘도 평탄** |
| F0-WA | F0-W + action-span credit (P1-6) | 0.244 | 0.325 | credit 재배분도 평탄 |

**진단**: 완벽한 신호(GT)로 바꿔도, credit 을 action 토큰에 몰아줘도 안 오른다 → 병목은
보상 설계가 아니라 **학습 역학**이다. 유일하게 남은 용의자가 §2 의 advantage 산출 방식이었다.

## 2. 그룹-상대 advantage 의 구조적 무용 확인 (실측)

GRPO 의 advantage 는 `r − (같은 프롬프트의 형제 롤아웃 8개의 평균)`이다. 즉 **그룹 안에 차이가
있어야만** 신호가 생긴다. 실측:

- full-trace 학습에서도 그룹의 37~53%가 전원 동일 보상(zero-std) → 해당 스텝 gradient 0.
- 결정적 실험: 출력을 `<action>` 한 줄로 줄인 action-only GRPO 에서 **T1.0 은 물론 T1.3 에서도
  8개 롤아웃이 전부 동일한 action** → 100% zero-std → 스모크 2회 연속 완전 무학습.
  (부산물 발견: **GRPO 의 탐색 다양성은 사실상 reasoning 텍스트의 샘플링에서 나온다.**)
- 8개가 전부 정답인 그룹도, 전부 오답인 그룹도 advantage 는 똑같이 0 — 보상이 정확한데도
  비교 대상이 없어 버려진다.

중간에 시도했다 **설계 기각**한 것: 후보 5개 문자열을 teacher-forcing 스코어링해 softmax CE 를
거는 exact-gradient(`f0_gx_train.py`, 코드는 기록용으로 보존). 학습은 되지만 "생성으로 행동을
선택한다"는 에이전트 전제와 train-test 불일치(학습은 스코어링 모드, 배포는 생성 모드)라 채택 안 함.

## 3. 도입한 해법: EMA 기준선 REINFORCE (생성 유지)

기준선을 그룹 내부가 아니라 **보상의 지수이동평균**에서 가져온다:

```
프롬프트당 롤아웃 1개 생성(T 샘플링) → r 계산
advantage = r − EMA(r)          # EMA 모멘텀 0.99
loss = −advantage · mean_logp(생성 토큰)
```

- 형제 롤아웃이 필요 없으므로 **출력이 결정론적이어도 r ≠ EMA 면 gradient 가 흐른다.**
- 롤아웃·프롬프트·파싱·평가는 기존 스택 그대로 — 바뀐 것은 advantage 의 출처 하나.
- 스모크 실측: action-only 에서 mean|adv| ≈ 0.47~0.57 로 살아 있음 (그룹 방식은 0).
- 진단 run **F0-GR**(action-only + GT 바이너리 + EMA)이 현재 학습 중 — "EMA 로 바꾸면
  GT 학습이 0.39 벽(F0-G 의 conditional acc)을 넘는가"를 측정한다.

## 4. GT-EMA vs WM-EMA: 두 확정 후보의 목표와 최적점

같은 최적화(EMA-REINFORCE, full-trace 생성 유지)에서 보상만 갈라 **둘 다 실행·비교**한다:

| | **F0-W-EMA** | **F0-WE** |
|---|---|---|
| 보상 | WM likelihood (후보 재정규화, 연속) | GT 바이너리 (=outcome reward 해석) |
| GT 사용 | **없음 (GT-free — '방법' 자격)** | 있음 (outcome-reward 서사 채택 시 방법) |
| 신호 밀도 | 연속값 — 사실상 매 샘플 gradient | 이진 — EMA 기준선으로 보완 |
| **이론적 최적점** | WM top-1 복사기 → **acc ≤ 0.374, G2 → 0** | GT∈top5 전부 정답 → **acc ≤ 0.620, G2 ↑** |
| 사전 등록 예상 | wm_follow↑ 수렴, acc 상승하되 0.374 한계 | 성공 시 acc·G2 동반 상승 |

핵심 논점(리뷰 요청): WM likelihood 에는 "WM 이 틀리는 순간"이라는 정보가 정의상 없으므로,
G2 를 보상할 수 있는 것은 GT/outcome 뿐이다 — **보상의 정의가 도달 가능한 최적점을 결정한다.**
이 사전 등록 예상이 실측과 갈리는지가 이번 무인 실행의 F0 측 핵심 산출물이다. 채택 판단은
수치가 아니라 포지셔닝(GT-free 주장 vs "WM=분포 인터페이스 + outcome RL" 서사) 결정이므로
자동화하지 않고 사람 리뷰로 남긴다.

## 5. B0-R1 리팩터: 목표·설계·재실행 계획

**문제**: MVP teacher 는 GT 를 본 채 reasoning/belief 를 쓰고 코드가 action 을 GT 로 덮어썼다
→ chosen 의 우위가 전부 belief/reasoning 문체(span-margin belief +0.802 vs action +0.014)에
있고 ③ 인과는 0.006 으로 무개선 — "정답 옆에 어울리는 문체"만 가르친 것.

**리팩터(hard action gate)**: teacher 가 (a) 미래 suffix(타깃 action 의 모든 등장 제거)에서
상위 goal 만 추출(GT verb/noun·활용형 누출 검사, 위반 시 금지어 재시도→드랍) → (b) **GT 를
못 본 채** goal+과거 근거만으로 reasoning/belief/action 을 공동 생성 → (c)
`canonical(predicted)==canonical(GT)` 일 때만 chosen 채택. "좋은 belief 였다면 정답에
도달했어야 한다"를 생성 시점에 강제한다. FAA(rejected) 롤아웃은 MVP 산출 전량 재사용 —
변수는 teacher 구성 하나.

**성공 기준(사전 등록, 자동 판정 → `B0_VALIDATED` 마커)**: ③ > 0.03 · action-span margin
≥ +0.023(A1 이상) · acc ≥ 0.248 · G2/G1 gate-retention ≥ 0.5.

**재실행 계획**: R1 검증 통과 시에만 **풀 스케일**(신규 ~3.5k 프롬프트 FAA 롤아웃 → gated pair
재구축 → 전체 DPO → 동일 평가) 자동 진행. 미충족 시 풀 스케일 금지 유지 + NEEDS 마커.
현재 R1 재구축 진행 중 — gate 통과율 실측 51%(리스크 임계 50% 상회), pass 당 pair ~2개.

## 6. 무인 실행 계획 전체 (마커 게이트 DAG)

```
[학습 중] F0-GR 진단 (cuda:0)                                   → ~03:30 KST
[학습 중] B0-R1 검증: pair 재구축→DPO→평가→자동판정 (cuda:1+0)     → ~05:30 KST
[대기] F0-WE   ← F0_GA_DONE 이후 "GR−F0G ≥ +0.02" 자동판정 시 (cuda:0) → ~10:00 KST
[대기] F0-W-EMA ← B0_R1 종료 시 (cuda:1, 무조건 실행 — GT-free 후보)   → ~10:30 KST
[대기] B0 풀 스케일 ← B0_VALIDATED ∧ 두 F0 run 종료 (양 GPU)          → 7/21 새벽 04~06시
```

- 모든 체인: 멱등(산출물 있으면 skip) · 실패 마커 · smoke 게이트(grad/advantage 실재 assert) ·
  DPO 는 no-train weight-diff guard. 기준 미달 시 run 을 강행하지 않고 SKIPPED/NEEDS 마커로 정지.
- 총 소요: F0 확정 산출물까지 ~10:30 KST(시작 00:01 기준 ~10.5h), B0 풀 스케일 포함 ~28~30h.

## 7. 코드 맵 — 방법론 ↔ 구현 위치 (본 커밋 기준)

### extro (F0)

| 구현 내용 | 위치 |
|---|---|
| 프롬프트 빌더 (joint top-5 셔플·score 은닉·`ACTION_ONLY` 분기) | `src/ego/step2_vlm_alignment/train_grpo_action.py` — `build_joint_conversation`, `JOINT_SYSTEM_PROMPT_ACTION_ONLY` |
| 리워드 모드 정의 (`wm_clean`/`gt_only`/`gt_action_only` 등) | 같은 파일 — `build_reward_funcs`, `validity_floor_reward_joint`, `gt_binary_reward_joint`, `wm_likelihood_reward` |
| oracle-subset 필터 + manifest | 같은 파일 — `filter_rows_for_stage` |
| 그룹-상대 advantage(기존) + zero-std 마스킹 | 같은 파일 — `DynamicSamplingGRPOTrainer` |
| action-span credit (P1-6, F0-WA) | 같은 파일 — `SpanCreditGRPOTrainer` |
| **EMA-REINFORCE 트레이너 (신규 핵심)** — 단일 롤아웃 생성, `--reward {gt,wm}`, `--full_trace`, `--batch_gen`, EMA 기준선, wm 모드에서 GT-필터 차단 | `scripts/step2/f0_gr_train.py` |
| exact-CE (설계 기각, 기록 보존) | `scripts/step2/f0_gx_train.py` |
| 진단/확정 체인: F0-GR / F0-WE(자동판정 게이트) / F0-W-EMA | `scripts/step2/f0_ga_chain.sh` / `f0_we_chain.sh` / `f0_wema_chain.sh` |
| Phase 1/2b 체인 (완료된 재편 실행) | `scripts/step2/f0_clean_chain.sh` / `f0_span_chain.sh` |
| 평가 러너 (`--action_only` 포함, 3-way oracle 보고) | `scripts/step2/eval_battery.py` |
| ③ belief-swap 인과 개입 | `scripts/step2/eval_belief_swap.py` |

### intro (B0)

| 구현 내용 | 위치 |
|---|---|
| **R1 teacher 확장 (신규 핵심)** — `goal_prompt`(suffix 전용) · `goal_leaks`(활용형 포함) · `gated_trace_prompt`(GT-hidden) · `GatedTeacherMixin.generate_gated_trace`(greedy+T0.8 재시도, hard gate) | `src/ego/step2_vlm_alignment/b0/teacher.py` 하단 "B0-R1" 섹션 |
| **R1 pair 빌더 (신규)** — suffix 필터, gate 통계, G1/G2 retention, 전건 audit, emit 직전 validate | `src/ego/step2_vlm_alignment/b0/build_dpo_dataset_r1.py` |
| MVP 빌더/routing/무결성 검사 (재사용) | `b0/build_dpo_dataset.py`, `b0/route_pairs.py`, `b0/validate_dpo_dataset.py`, `b0/trace_utils.py` |
| DPO 학습 (FAA init + frozen FAA ref) | `b0/train_b0_dpo.py` (체인에서 `--max_length 4096` 강제) |
| span-margin 재측정 (길이 정규화 + 태그 span 귀속) | `scripts/step2/remeasure_b0_margin.py` |
| R1 검증 체인 (retention 게이트·B0_VALIDATED 자동판정) / 풀 스케일 체인 | `scripts/step2/b0_r1_chain.sh` / `b0_full_chain.sh` |

### 결과물 위치

`$EGO_ROOT/runs/f0_battery/` — `F0_CLEAN_RESULTS.md`·`F0_SPAN_RESULTS.md`(완료),
`F0_GA_RESULTS.md`·`F0_WE_RESULTS.md`·`F0_WEMA_RESULTS.md`·`B0_R1_RESULTS.md`·`B0_FULL_RESULTS.md`(무인 생성 예정),
run 로그 `b0_r1/`·`b0_full/`, 마커 파일 일체.

## 8. 리뷰 요청 포인트

1. **EMA-REINFORCE 의 통계적 타당성** (`f0_gr_train.py`): 단일 롤아웃 + 전역 EMA 기준선은
   프롬프트 난이도 분산을 흡수하지 못한다 (쉬운 프롬프트는 늘 +adv, 어려운 프롬프트는 늘 −adv).
   진단·비교 목적엔 충분하다고 판단했으나, 확정 방법으로 갈 경우 프롬프트-조건부 기준선
   (난이도 버킷 EMA 또는 leave-one-out 배치)이 필요한지 의견 요청.
2. **WM-EMA 최적점 논증** (§4): "WM likelihood 보상의 최적 정책 = top-1 복사기, 고로 G2→0"
   — 이 논증에 빈틈이 있는지 (예: 온도 샘플링 하에서 기대보상 최대 정책이 복사기가 아닐 여지).
3. **B0-R1 gate 의 선택 편향**: gate 통과 샘플이 쉬운 쪽으로 쏠리는 문제를 G2/G1 retention ≥ 0.5
   로 방어하는데, 이 임계와 지표 정의가 적절한가 (`build_dpo_dataset_r1.py`의 `sample_group`).
4. **goal 누출 방어의 잔여 구멍**: 단어-경계+활용형 검사(`teacher.py`의 `goal_leaks`)로 명시 누출은
   잡지만 의미적 힌트("샐러드 준비" → cut lettuce)는 남는다. entity provenance 기록으로 audit 은
   가능하나 자동 차단은 아님 — 허용 범위인지.
5. **무인 자동판정의 안전성**: GR≥+0.02 → WE 실행, 4기준 → 풀 스케일 같은 사전 등록 게이트가
   사람 확인 없이 run 을 이어가는 구조 — 게이트 조건이 충분히 보수적인지.
