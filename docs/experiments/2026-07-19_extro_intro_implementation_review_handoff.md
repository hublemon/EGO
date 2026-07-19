# extro / intro 구현 상세 핸드오프 — 리뷰 담당자용

작성: 2026-07-19 (KST) · 대상: 코드 리뷰 담당자
목적: **extro(F0 트랙)와 intro(B0 트랙)가 방법론·코드 수준에서 어떻게 구현되어 있는지** 세밀하게 전달하고, 문제 여부를 검토받기 위함.
범위: extro 는 **현재 실행 중인 구현 그대로**, intro 는 **현행 B0-MVP 구현 + 리팩터(B0-R1)는 구현 계획**으로 기술한다. 리팩터는 코드가 한 줄도 작성되지 않았다.

---

## 0. 용어와 파일 맵

| 용어 | 의미 | 학습 신호의 원천 |
|---|---|---|
| **extro** | F0 트랙: **외부(external) 세계 신호** 기반 GRPO. V-JEPA2 WM 의 joint top-5 likelihood 분포를 reward 로 사용 | WM (GT-free) |
| **intro** | B0 트랙: **내부(introspective) 신호** 증류. teacher 가 만든 past-grounded full-trace(reasoning+belief+action)를 preference 로 FAA 에 DPO | teacher trace (GT 는 오프라인 구성에만) |

핵심 파일:

```
extro (F0)
  src/ego/step2_vlm_alignment/train_grpo_action.py   # 프롬프트·파싱·리워드·트레이너 (2251줄)
  scripts/step2/eval_battery.py                      # 평가 배터리 (학습 빌더 재사용)
  scripts/step2/eval_belief_swap.py                  # ③ belief-swap 인과 개입
  scripts/step2/f0_clean_chain.sh / f0_span_chain.sh # 무인 실행 하네스 (Phase 1 / 2b)
  src/ego/step2_vlm_alignment/judge_reasoning.py     # gemini 외부 judge (25 step 간격)

intro (B0, 현행 MVP)
  src/ego/step2_vlm_alignment/b0/
    generate_faa_traces.py   # frozen FAA online rollout
    teacher.py               # frozen base VLM teacher (hindsight·projection·equivalence)
    route_pairs.py           # pair routing table (순수 로직)
    build_dpo_dataset.py     # 오케스트레이션 + emit 직전 검증
    validate_dpo_dataset.py  # leakage / pair-invariant 검사 (순수 로직)
    merge_b0_samples.py      # faa_traces + b0meta 병합
    train_b0_dpo.py          # sequence-level DPO (TRL)
    evaluate_b0.py           # margin / accuracy 분해 / coherence
  scripts/step2/remeasure_b0_margin.py               # 길이 정규화 + span 분해 재측정
  scripts/step2/b0_auto_chain.sh / b0_ablation_chain.sh

intro 리팩터 (B0-R1) — 계획 문서만
  docs/experiments/2026-07-19_b0_teacher_refactor_handoff.md
```

공통 출력 계약: 정책 모델은 `<reasoning>…</reasoning> <task_belief>…</task_belief> <action>{"verb","noun"}</action>` 3태그 full-trace 를 생성한다. extro 는 이 trace 를 **온라인으로 생성해 reward 로 채점**하고, intro 는 이 trace 를 **오프라인 preference pair 로 비교 학습**한다. 두 트랙이 같은 출력 계약을 공유하므로 B0 는 F0(FAA) 위에 그대로 얹힌다.

---

## 1. extro (F0) — 구현 상세

### 1.1 입력 계약: 프롬프트 빌더 (`build_joint_conversation`, train_grpo_action.py:424)

- **WM joint top-5 를 5지선다 후보로 제시**: `topk_actions_with_score[:5]` 에서 (verb,noun) 쌍을 추출, **score 는 제거**하고 `rng.sample` 로 **표시 순서를 셔플** (rank 가 자명한 답이 되는 지름길 차단, :440).
- **task_goal 은 의도적으로 미포함** (:442) — video ID 기반 지름길이 되는 것을 사전 실험으로 확인 (GT 예측력 1.1%).
- 컨텍스트 = 이미지 1프레임(또는 4f grid) + `Action history:`(memory_context) + 후보 목록. `--no_memory`, `--mask_frame_prob`(L2-a 히스토리-단독 경로) 스위치가 같은 빌더에서 분기.
- 빌더가 reward 함수용 컬럼을 함께 기록: `topk_actions_display`(화면에 실제로 보인 쌍 목록 — gate 의 기준), `memory_context`(judge 용, **reward 는 이 컬럼을 읽지 않음**), `topk_actions_with_score`(likelihood reward 용).
- 평가(eval_battery)와 FAA rollout(generate_faa_traces)이 **이 빌더를 그대로 import** 하므로 학습/평가/rollout 의 입력 분포가 강제로 일치한다. 셔플 seed 도 고정(42)이라 재현 가능.

### 1.2 파싱 계약 (:616–673)

- `parse_action_from_think_format`(:642): `<action>` 블록에서 ```json 펜스 제거 → `{…}` 추출 → json.loads. 실패 시 (None,None) — **fuzzy 복구를 하지 않는다** (관대한 파싱은 format 신호를 오염시키므로).
- `<reasoning>` 과 `<think>` 를 모두 수용(:631)하는 이유: Qwen3-VL 에서 `<think>`/`</think>` 는 **예약된 단일 토큰**(151667/151668)이고 Instruct 변형은 이를 생성하지 않도록 튜닝되어 있어(실측 0회) `<reasoning>` 태그로 우회했다. `<think>` 는 Qwen2.5 구 run 하위호환용.

### 1.3 리워드 체계 (`build_reward_funcs`, :1357)

현재 실행 중인 확정 모드 두 가지 (2026-07-19 클린 재편):

```python
"wm_clean": [validity_floor_reward_joint, wm_likelihood_reward]          # F0-W
"gt_only":  [validity_floor_reward_joint, gt_binary_reward_joint]        # F0-G (skyline)
```

- **`validity_floor_reward_joint`** (:930): parse 실패 **또는** 선택 쌍이 화면에 보인 joint top-5 밖이면 −0.5, 유효하면 0.0. **additive 보너스가 아니라 constraint(고정 floor)** — 유효하기만 하면 이 항은 상수 0 이 되어 advantage 에 기여하지 않고, task reward 가 유일한 varying 신호가 된다.
- **`wm_likelihood_reward`** (:1109) + `_likelihood_reward_value`(:1086): 선택 쌍에 매칭되는 후보의 WM likelihood 를 **후보셋 내 재정규화**(`--wm_likelihood_norm candidate`)한 연속값. 매칭 실패(후보 밖)는 0.0 — 감점은 floor 가 전담하므로 이중 처벌 없음. reward 는 **WM 예측 분포 그 자체의 함수**이고 GT 를 일절 참조하지 않는다.
- **`gt_binary_reward_joint`** (:946): (verb,noun)==GT → 1.0, else 0.0. 부가적으로 그룹 조성(all_wrong/mixed/all_correct)을 `group_stats.jsonl` 로 side-log — GRPO 는 그룹 내 분산이 있어야 학습하므로 mixed 비율이 곧 유효 신호량이다.

**레거시와의 차이 (P1-1 오염 제거).** 직전 run(F0-L)의 `wm_likelihood_joint` 모드는 `[format_reward_joint(+0.2), candidate_gate, wm_likelihood, think_convergence_joint]` 구성이었다. 실측 분해 결과 format/gate 는 사실상 상수(+0.198/−0.005)여서 advantage 를 만드는 varying 항은 wm_likelihood(0.315)와 **think_convergence(0.180) — varying 신호의 36%** 였다. `think_convergence_reward_joint`(:1162)는 "think 에서 마지막으로 언급한 후보 쌍 == 최종 선택이면 +0.15" 류의 **사후 언급 보상**이라, 결론을 먼저 정하고 think 에 그 이름을 쓰는 것만으로 만점이 된다 — ④ judge 포화(≈1.97/2.0)와 ③ 인과 0 의 괴리를 설명하는 메커니즘. `wm_clean` 은 이 항과 format 보너스를 **학습 reward 에서 제거**하고 (평가 지표로만 유지), 3태그 출현은 validity floor 의 parse 요건이 유지한다.

### 1.4 데이터 게이트 (`filter_rows_for_stage`, :1415)

- **GT-free 모드**(wm_clean 포함): GT 기반 필터를 일절 쓰지 않는다 (GT 가 학습 분포 선택에 새는 뒷문 차단). 대신 `--min_wm_spread 0.05` — 후보셋 재정규화 likelihood 의 표준편차(`_wm_spread`, :1399)가 임계 미만인 샘플을 제거. reward 가 WM likelihood 만의 함수이므로 **프롬프트별 달성 가능 reward spread 가 학습 전에 결정**된다는 점을 이용한 정적(dynamic-sampling 의 사전) 필터다.
- **gt_only 모드**: GT∉top-5 샘플은 전 롤아웃 reward 0(학습 불능)이므로 **oracle-subset 으로 drop** 하고, `oracle_manifest.json` 에 coverage(`num_gt_in_topk/num_total`)를 기록 — 결과 보고 시 coverage@5 / conditional acc / overall 3-way 로 분리 해석하기 위함 (skyline 이 전체 acc 로 오독되는 것 방지).
- `assert_no_score_leak`(:1476): 변환된 프롬프트 텍스트에 WM score 수치가 노출되지 않았는지 학습 시작 전에 assert.

### 1.5 트레이너: `DynamicSamplingGRPOTrainer` (:1803)

TRL 1.5.1 GRPOTrainer 서브클래스. `_generate_and_score_completions` 후처리로 **그룹 내 reward std ≤ 임계인 그룹의 advantage 를 0 으로 마스킹** (DAPO dynamic sampling 의 마스킹 변형 — TRL 생성 루프를 침습하지 않기 위해 재샘플 대신 마스킹). 전제 검증이 `__init__` 에 있다: `scale_rewards="none"`(Dr. GRPO)이어야 advantage std == reward std 가 보존되며, `"group"`(기본값)이면 std 정규화가 그룹 분산을 소거해 필터가 무의미 → ValueError.

주요 하이퍼(체인 COMMON_ARGS): `dr_grpo` + `scale_rewards none` + `epsilon_high 0.28`, num_generations 8, temperature 1.0, max_completion 384, LoRA r16/α32, lr 1e-5, beta 0.0(Phase 1) — Phase 1 두 arm 은 reward_mode 외 전 조건 동일하므로 차이는 신호 원천으로 귀속된다.

### 1.6 `SpanCreditGRPOTrainer` (:1843, Phase 2b = F0-WA)

P1-6(credit 배분) 대응. TRL 1.5.1 `_compute_loss` 가 **(B,T) 토큰별 advantage 를 공식 지원**하는 것을 이용해 손실 코드를 포크하지 않는다:

- `_action_span_weights`(:1869): completion 토큰열에서 `<action>`/`</action>` **태그 토큰열 서브시퀀스 탐색**으로 span 을 찾고, 실패 시 **누적 디코드 폴백**(태그가 BPE 경계에 걸린 경우). span 내부 가중치 1.0, 외부 `span_credit_lambda`(기본 0.0). 태그가 아예 없으면 전 토큰 λ — validity floor 감점 신호만 남는다(의도된 동작). 탐지율 카운터를 512 샘플마다 출력.
- `_compute_loss`(:1898): `advantages (B,) → (B,T) × weights` 후 super() 호출.
- **한계(리뷰 포인트)**: reasoning/belief 붕괴 방지는 `--beta 0.04`(ref-KL)가 담당하는데, TRL 구조상 **KL 은 전 토큰 적용**이라 비-action 토큰에만 한정할 수 없다. action 토큰에도 걸리는 KL 은 약한 정규화로 무해하다고 판단했다 — 이 판단의 타당성 검토 요청.

### 1.7 관측성

- `GRPOLogger`(:1620) → `reward_log.jsonl`: step 별 **리워드 함수별 평균**(floor/task 분리 관측 가능), loss, grad_norm, `frac_reward_zero_std`(그룹 내 reward 가 전부 동일한 그룹 비율).
- `--completion_log_every 25` → `completion_samples.jsonl` 에 실제 completion 저장 → `judge_reasoning.py`(gemini-2.5-pro, step 당 3개, 7항목 루브릭: history/candidate/visual grounding, conclusion, no-confabulation, belief globality, **belief_action_link**)가 체인의 judge follower 루프(10분 주기, 멱등)로 채점 → `judge_curve.jsonl`.
- **알려진 수치 특성**: wm_clean 의 frac_reward_zero_std ≈ 0.5 — 단일 varying 신호 + 8 generation 에서 그룹이 같은 답에 수렴하는 경우가 절반. gt_only 는 더 높다(binary 신호라 all_wrong/all_correct 그룹이 다수, mixed 만 학습 기여). 이 값이 높다는 것 자체는 버그가 아니라 **마스킹이 작동 중**이라는 뜻이고 (smoke 게이트가 8 step 중 grad>0 존재를 별도 검증), 대신 유효 배치가 줄어드는 효율 저하로 해석해야 한다.

### 1.8 평가 (extro 공통 — intro 도 동일 러너 사용)

- **`eval_battery.py`**: 학습 프롬프트 빌더를 그대로 사용 + `max_new_tokens 384`(학습 예산과 동일 — 구 evaluate.py 의 256 절단 문제가 여기서 교정됨), greedy. 지표: acc / verb·noun acc / in_joint5 / **wm_follow**(WM top-1 복사율) / **G2**(GT∈top5 ∧ GT≠top1 부분집합 acc — "WM 을 이길 수 있는가", chance 0.2) / parse_rate / belief 통계, 그리고 oracle-subset 3-way (`gt_in_top5_rate`, `acc_given_gt_in_top5`, `oracle_upper_bound_proxy`). 샘플별 completion 전문이 `.records.jsonl` 로 남아 ③의 입력이 된다.
- **`eval_belief_swap.py` (③)**: 완성 trace 에서 reasoning 은 고정, `<task_belief>` 만 다른 샘플 것으로 교체(derangement, offset 250)한 prefix 를 **assistant 응답 강제 prefix** 로 넣고 `<action>` 만 이어서 greedy 재생성. `causal_sensitivity = swap_action_change − control_action_change`. control 군이 디코딩 노이즈 플로어를 제공하는 사전 등록형 개입 실험. GT 무사용.
- **신뢰 경계**: 레거시 `evaluate.py`(v1 플랫 빌더 import, 256 tok) 경로의 과거 수치는 하한으로만 유효. 이 세션의 모든 비교 수치는 eval_battery@384 로 통일되어 있다.

### 1.9 실행 하네스 (`f0_clean_chain.sh` / `f0_span_chain.sh`)

- **smoke 게이트**: 코드 SHA + trl 버전을 키로 8 step(logging 1) 실행 → `reward_log.jsonl` 에서 **grad_norm>0 존재** assert. (B0 no-train 사고의 재발 방지 — 아래 2.7.)
- **부분 체크포인트 오인 방지**: 학습 시작 전 `TRAINING_DONE` 없는 출력 디렉토리는 `rm -rf` — resume 미지원 대신 "부분 산출물은 절대 완료로 취급하지 않음"을 하드 보장.
- 마커 체인: `TRAINING_DONE`(checkpoint-500 검증 후) → 평가/③ → `F0_CLEAN_DONE` → span 체인이 이어받아 `F0_SPAN_DONE`. 실패는 `*_FAILED`. 전 단계 멱등(존재하는 산출물 skip).

---

## 2. intro (B0-MVP) — 현행 구현 상세

파이프라인: **① FAA rollout → ② teacher trace 구성 → ③ routing/검증 → ④ DPO → ⑤ 평가**.

### 2.1 설계 요지

FAA(frozen F0 어댑터)가 자기 프롬프트에서 생성한 full-trace(y⁻)와, teacher 가 미래 GT 궤적의 hindsight 를 시점 t 로 투영해 만든 full-trace(y⁺)를 preference pair 로:

```
L_B0 = -log σ[ β( log π_B0(y⁺)/π_FAA(y⁺) − log π_B0(y⁻)/π_FAA(y⁻) ) ]
```

- **GT 는 오프라인 pair 구성에만** 들어가고 정책의 온라인 입력에는 없다 (extro 의 GT-free 계약과 양립).
- **splicing 금지**: chosen/rejected 는 각각 완결 trace 원문 그대로 — 필드 조합·편집 없음 (reasoning↔action 정합 보존).

### 2.2 ① FAA rollout (`generate_faa_traces.py`)

frozen FAA(LoRA)로 `build_joint_conversation` **동일 빌더** 프롬프트에서 num_generations 4, T 1.0, top_p 0.95 샘플링. adapter safetensors md5 를 `faa_checkpoint_hash` 로 모든 레코드에 스탬프 — 어떤 FAA 가 y⁻ 를 만들었는지 추적 가능. 마스킹은 rollout 에서 비활성(`MASK_FRAME_PROB=0`).

### 2.3 ② teacher (`teacher.py`)

**frozen base VLM(Qwen3-VL-8B-Instruct, FAA LoRA 미적용)** — 외부 대형 모델에 의존하지 않는다는 회의 방침 준수. greedy(결정론)로 preference target 을 안정화. 세 역할:

1. `infer_raw_trace`: **미래 GT action 시퀀스** → overall goal + 설명 (hindsight, 미래 정보 포함 허용).
2. `project_full_trace`: hindsight + **시점 t 까지의 history + 후보 + 시점 t 프레임 이미지** 로 reasoning/belief 를 시점 t 로 재작성. PROJECTION_RULES 가 프롬프트에 명문화: *미래에만 등장하는 개체 금지 / 근거 약하면 belief specificity 하향 / GT next action 을 관측된 사실로 인용 금지 (출력 타깃일 뿐)*. Allowed/Forbidden 예시 포함.
   - 파싱 후 **action 은 무조건 GT 로 강제 치환**(`tr.verb, tr.noun = gt_verb, gt_noun`) 후 `build_full_trace` 로 정규 직렬화 → chosen.action==GT 가 구조적으로 보장.
   - 이미지를 실제로 전달하는 이유(:120 주석): "A bowl is visible" 류 시각 주장이 관측에 근거하게 — offline teacher 는 x≤t 접근이 허용된다는 정보 경계 규약.
3. `equivalence`: FAA belief vs projected belief → SAME/DIFFERENT/UNCERTAIN (**stop-gradient**: 라벨은 routing 에만 쓰이고 어떤 gradient 경로에도 없음). 파서는 DIFFERENT 를 먼저 검사 (SAME 이 substring 인 오탐 방지).

### 2.4 ③ routing (`route_pairs.py` — 순수 로직, smoke 로 단언)

| belief_rel × action_rel | 처리 | 근거 |
|---|---|---|
| DIFFERENT × (SAME\|DIFFERENT) | **KEEP** | belief/action 또는 refinement preference |
| SAME × DIFFERENT | **KEEP** | action/full-trace refinement |
| SAME × SAME | **DROP → audit** | semantic tie — projector 문체 모방 학습 방지 |
| UNCERTAIN × DIFFERENT | **KEEP + tag** | action 신호는 존재 |
| UNCERTAIN × SAME | **DROP → audit** | preference 근거 불확실 |

action_relation 은 `canonical_action`(소문자 정규화 `"verb|noun"` 키) 동치. drop 은 **silent 가 아니라 전건 audit jsonl 로 보존**되고 RoutingStats 가 사유별 집계를 로그로 남긴다.

### 2.5 ④ 데이터셋 구축 (`build_dpo_dataset.py`)

sample 단위 게이트 → pair 단위 routing:

- `gt_in_candidates` 실패(GT∉D_t) → drop 집계 (candidate support, 학습 불능 방지 — extro gt_only 의 oracle-subset 과 동일 논리).
- `future_gt_actions` 가 빈 시퀀스(영상 말미) → projection 실패로 집계 후 drop.
- FAA trace 는 parse 실패·중복 제거(`_dedup_valid_traces`) 후 각각이 독립 pair 가 된다.
- **emit 직전 `validate_record`** — 실패 pair 는 학습에서 빼고 `DROPPED_VALIDATION` 태그로 audit 에 보존.
- 검사 전용 원본(`_leak_check`: GT/future/raw trace 등)은 `_strip_leakcheck` 로 **저장 파일에서 항상 제거**.

### 2.6 무결성 검사 (`validate_dpo_dataset.py` — 2층)

**leakage 층**: policy prompt 텍스트에 raw hindsight / projected belief / FAA belief / equivalence 라벨 / GT·future action 문자열이 substring 으로 존재하면 위반. 단 GT·future 문자열 검사는 **history 섹션을 마스킹한 텍스트**에만 적용 — 반복 태스크에선 과거 히스토리에 GT 와 같은 라벨("stir pan")이 정당하게 존재하고, 시간 경계는 별도의 `stop_time ≤ trigger_time` 구조 검사(HH:MM:SS 문자열도 초 변환)가 보증하기 때문. 오탐으로 반복-행동 샘플을 잃지 않기 위한 설계.

**pair invariant 층**: chosen/rejected 각각 완결 trace 파싱 / `chosen.action==canonical GT`·`rejected.action==canonical FAA` / **verbatim 대조로 splicing 탐지**(빌드가 원문을 그대로 넣었는지) / SAME-SAME 이 훈련셋에 남아있으면 위반 / chosen reasoning 의 미래-지식 표현("actually happens next" 등 FUTURE_LEAK_MARKERS) 스크리닝.

독립 실행(`validate_dataset_file`)으로 전체 데이터셋 사후 재검증도 가능.

### 2.7 ⑤ DPO 학습 (`train_b0_dpo.py`)

- policy = base + **FAA adapter 를 trainable 로 로드** (FAA 에서 초기화). reference = **FAA adapter 를 얹은 별도 frozen 모델을 명시적으로 전달** — TRL 의 PEFT 경로에서 `ref_model=None` 이면 "adapter off = base" 가 reference 가 되는 함정(:110 주석)을 회피. multi-GPU 시 ref 도 `LOCAL_RANK` 고정 device_map (auto 샤딩 충돌 방지).
- VLM DPO 규약: 이미지는 `images` **리스트 컬럼** (GRPO 의 `image` 단수와 다름), `cast_column(Sequence(DSImage()))`.
- **알려진 사고와 현재 상태 (리뷰 필수 확인)**: TRL 1.5.1 의 DPOConfig 에 `max_prompt_length` 가 없어 시그니처 필터(:105)가 이 인자를 **조용히 버렸고**, 남은 `max_length=1536` 이 이미지 토큰(~1000+)에 밀려 **completion 을 전부 절단 → margin 항등 0, loss=ln2, grad=0 인 무학습**이 발생했다. 수정은 실행 체인 층에서: `--max_length 4096` + 학습 후 **safetensors weight-diff no-train guard**(FAA 대비 diff≈0 이면 실패 처리; 건강한 run 실측 diff 0.00035). 리뷰 지적대로 **시그니처 필터 자체를 fail-fast whitelist 로 바꾸는 개선은 수용되었으나 아직 미구현** — 현재 방어선은 체인의 no-train guard 와 smoke 다.

### 2.8 ⑥ 평가

- `evaluate_b0.py`: (A) held-out preference margin `m = logπ(chosen) − logπ(rejected)` 를 relation 별 분해, (B) accuracy 를 candidate_recall / conditional / end-to-end 3분해 + recovery/regression 대차, (C) coherence 프록시(API 0). 순수 계산 함수는 smoke 로 검사.
- **`remeasure_b0_margin.py` (사후 추가)**: 초기 margin 수치(+55.8)가 **길이 미정규화 합산**이라 과대해석 위험 → 단일 토크나이즈 + offset mapping 으로 토큰 logp 를 span(태그 문자 범위)에 귀속시켜 **mean-per-token margin 과 span 별 분해**를 재측정. 결과(906 pairs): B0 +0.287 (reasoning +0.336 / task_belief +0.802 / **action +0.014**), A1 대조 +0.130 (…/ action +0.023). → **margin 이득의 소재는 belief/reasoning 문체이지 action 정렬이 아니다** — 이것이 intro 리팩터의 직접 동기다.

### 2.9 B0-MVP 실측 요약 (eval_battery@384 기준)

acc 0.248(FAA 0.230 대비 +1.8%p), G2 0.342(역대 최고), 스케일링 양성(half-data A2 0.238 < full 0.248), 그러나 **③ causal_sensitivity 0.006** (base 0.016, control floor 0.002–0.006) — belief 가 action 을 인과적으로 조향하지 못하는 문제는 미해결. margin 은 잘 벌어지지만(2.8) 그 정체가 문체 모방이라는 span 증거와 정합.

---

## 3. intro 리팩터 (B0-R1) — **구현 계획** (코드 미작성)

> 이 절 전체는 설계·계획이다. 세부는 `2026-07-19_b0_teacher_refactor_handoff.md`(B0 담당자용, v2 정정 반영) 참조.

### 3.1 문제 진단 (현행 구현의 구조적 한계)

현행 teacher 는 projection 시 **GT 를 보고 그 GT 로 끝나는 trace 를 쓴다**(2.3 의 action 강제 치환). 따라서 chosen 의 "우월성"은 belief/reasoning 텍스트 품질에 있고, belief→action 인과를 담보하는 장치가 없다 — span 분해(+0.802 belief vs +0.014 action)와 ③ 0.006 이 그 증거. equivalence routing 은 pair 선별만 할 뿐 chosen 자체의 인과 품질을 보증하지 못한다.

### 3.2 핵심 변경: hard action gate

- teacher 가 **GT 를 숨긴 채** reasoning/belief/action 을 **공동 생성**하게 하고, `canonical(predicted_action) == canonical(GT)` 인 trace 만 PASS → chosen 후보로. "좋은 belief 였다면 GT 에 도달했어야 한다"를 생성 시점에 검증하는 구조.
- 초안에 있던 "gemini 가 belief→GT 자연스러움을 판정" 방식은 **기각**(v2 정정): GT 를 본 judge 는 모호한 belief 도 사후 합리화한다. gemini 는 **process verifier**(누설·형식·근거 위반 검출)로만 사용.
- goal 추출은 미래 suffix `a_{t+1:}` 기반 + entity provenance(개체가 어느 시점 관측에서 왔는지) 태깅.

### 3.3 예상 코드 변경 지점

| 파일 | 변경 |
|---|---|
| `b0/teacher.py` | `projection_prompt` 를 GT-hidden 공동 생성 프롬프트로 교체, gate 판정 메서드 추가 (기존 `equivalence` 는 유지) |
| `b0/build_dpo_dataset.py` | projection 후 hard gate 단계 삽입 (PASS 실패 시 재시도 n회 → drop 집계), gate 통계 로깅 |
| `b0/validate_dpo_dataset.py` | invariant 추가: chosen 은 gate PASS 원문이어야 함 |
| 체인 | R1(소규모 gate 수율 검증) → R2(pair 재구축) → R3(DPO+평가) staging, `B0_VALIDATED` 게이트 전 full-scale 금지 |

### 3.4 사전 등록 성공 기준 (변경 불가로 문서화됨)

③ causal_sensitivity > 0.03 · action-span margin ≥ +0.023(A1 수준 이상) · acc ≥ 0.248(MVP 비퇴행) · G2/G1 retention ≥ 0.5.

---

## 4. 리뷰 요청 포인트 & 알려진 리스크 요약

**설계 판단의 타당성 검토를 요청하는 항목:**

1. (extro) span-credit 에서 **KL 이 전 토큰에 걸리는 구조적 한계**(1.6)의 수용 가능성 — λ=0 + β=0.04 로 reasoning/belief 는 KL 만으로 보존된다는 가정.
2. (extro) wm_clean 의 frac_zero_std ≈ 0.5 — 마스킹 정상 동작이지만 유효 배치 절반. 재샘플형 dynamic sampling(DAPO 원형) 없이 충분한가.
3. (intro) leakage 검사에서 **history 섹션을 GT-문자열 검사에서 제외**한 설계(2.6) — 시간 경계 검사로 충분히 커버되는가.
4. (intro) teacher = frozen **base**(FAA 아님) 선택 — projection 품질 vs 분포 근접성 트레이드오프.

**알려진 미구현 항목 (수용됐으나 후순위):** DPOConfig fail-fast whitelist(현재 silent-drop + 체인 guard 로 방어) · P0-3 token-survival assertion · B0-6 collator-aligned B0 평가 · legacy `evaluate.py` 폐기 처리 · B0_VALIDATED 게이트 스크립트화.

**수치 신뢰 경계:** 본 문서의 모든 비교 수치는 eval_battery@384 산출물(`runs/f0_battery/*.json`) 기준. 레거시 evaluate.py 경로 수치는 하한으로만 인용.
