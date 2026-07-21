# Retro v2 (HPCTD) 구현·실행 계획 Handoff

> ⚠️ **보류(2026-07-21).** rev2는 폐기·삭제했다. 현재 채택 방향은
> `2026-07-21_step1_night_and_retro_belief_sum_handoff.md` §6 "다음 처방(개선 2 / P3)" —
> belief-swap consistency loss다. 이 문서는 조사 기록(§1 저장소 실측, §3 재사용 가능 코드 목록)으로만 보존한다.

> 작성일: 2026-07-21 KST
> 대상: EGO Step-2 Retrospection 구현/실험 담당자
> 저장소: `/mnt/nvme/migration/jihun/EGO_jihun` (origin `https://github.com/hublemon/EGO`, `main`)
> 기준 커밋: `2c34ea2` (로컬 dirty — 아래 P0.1 참조)
> 선행 문서: `2026-07-21_retro_v2_methodology_and_implementation_handoff.md` (방법론 = **무엇을**)
> 이 문서의 역할: 그 방법론을 **이 저장소에서 실제로 어떻게** 구현·실행·검증할지 (= **어떻게**)

---

## 0. 이 계획의 세 줄 요약

1. 방법론 문서의 설계(HPCTD: hindsight projection → trace SFT → candidate ranking → belief intervention)는 그대로 채택한다. 다만 **코드의 절반은 이미 저장소에 존재**하므로 신규 작성 범위를 재산정했다 (§3).
2. 착수 전에 **실측으로 확인된 두 개의 구조적 위협**이 있다. (a) train/heldout 그룹 분포가 크게 다르고 (G1 70.0% vs 39.9%), (b) belief를 출력하는 것 자체가 정확도에 손해다 (belief 없는 run 0.338 vs belief run 전부 ≤0.280). 둘 다 계획에 대응책과 판정 규칙을 미리 박아두었다 (§2).
3. 학습을 시작하기 전에 **GPU를 거의 쓰지 않는 Phase 0**이 전체 프로젝트의 방향을 바꿀 수 있다. candidate-scored 정확도를 base 모델에서 먼저 재보는 것 하나로, "생성이 병목인가 지식이 병목인가"가 결정된다 (§4.1).

---

## 1. 현재 저장소 상태 (착수 전 사실 확인)

### 1.1 Git

`EGO_jihun`은 `hublemon/EGO`의 클론이고 HEAD는 `2c34ea2`이지만 **워킹트리가 dirty**하다.

수정됨 (4): `scripts/step2/eval_battery.py`, `scripts/step2/eval_belief_swap.py`, `scripts/step2/pro_gr_train.py`, `src/ego/step2_vlm_alignment/retro/train_retro_dpo.py`
추적 안 됨 (주요): `scripts/step2/eval_harness_v2.py`, `src/ego/common/run_provenance.py`, `scripts/step2/retro_overnight_gpu1_v{1,2,3}.sh`, `scripts/step2/{analyze_bo8,rerank_bo8}.py`, `scripts/step2/fix_missing_swap_eval.sh`, `docs/experiments/2026-07-2{0,1}_*.md`, `data/Ego4D`

이 중 `eval_harness_v2.py`와 `run_provenance.py`는 **retro v2가 의존할 측정 인프라**인데 아직 커밋되지 않았다. 참고로 `EGO_jihun2`/`EGO_jihun3`(= origin/main, `f10e327`)에는 이 파일들이 이미 반영되어 있고 `EGO_jihun`(2c34ea2)보다 22커밋 앞서 있다.

> **P0.1 (첫 작업)**: `EGO_jihun`의 로컬 변경분과 origin/main `f10e327`의 차이를 대조해 아직 반영 안 된 것만 커밋/푸시하고, 작업 브랜치를 `f10e327` 기준으로 재정렬한다. retro v2 코드를 dirty 트리 위에 얹으면 나중에 무엇이 무엇의 결과인지 복원 불가능해진다.

### 1.2 데이터 (실측)

| 파일 | 행 수 | 비고 |
|---|---:|---|
| `EGO/runs/f0_battery/train_1f_root/data/grpo_dataset/grpo_train_1f.jsonl` | 4,998 | sha256 `693c0dc2…23a0` |
| `EGO/runs/f0_battery/heldout_1f_root/data/grpo_dataset/grpo_heldout_1f.jsonl` | 1,417 | |

레코드 스키마 (13필드): `image_path, episode_id, frame_id, task_goal, topk_nouns, topk_actions, topk_verbs, topk_actions_with_score, topk_nouns_with_score, memory_context, frame_meta, gt_verb, gt_noun`

- `topk_actions_with_score` = `[{rank, verb, noun, likelihood}] × 5` — **WM rank와 확률이 둘 다 이미 들어 있다.** `p_WM`을 따로 만들 필요 없음.
- WM likelihood는 top-5 합이 1이 아니다 (예: 0.229). `pro_gr_train.py:199-207`이 candidate 정규화하는 방식을 그대로 재사용할 것.
- `sample_id`로 쓰이는 키는 `frame_id`.

### 1.3 ★ 실측된 그룹 분포 — 계획 변경 사유

```
train    n=4998  G1=3498 (0.700)  G2=1120 (0.224)  OUT=380 (0.076)
heldout  n=1417  G1= 566 (0.399)  G2= 321 (0.227)  OUT= 530 (0.374)
```

- **G2 비율은 거의 동일** (0.224 vs 0.227). G2 중심 학습 전략은 분포적으로 안전하다. 좋은 소식.
- **G1과 OUT은 극단적으로 다르다.** train에서는 WM top-1이 70% 맞고 GT가 top-5에 92.4% 들어 있지만, heldout에서는 각각 39.9% / 62.6%다.
- 이것이 기존 메모에 남아 있는 "B0가 WM top-1 베이스라인 미달 — 학습 샘플 선정이 rank prior를 역전시킨 게 근본 원인"의 정량적 실체다. train 분포에서 최적인 정책은 "WM top-1을 따른다"(70% 정답)이고, 그 정책은 heldout에서 39.9%밖에 못 맞는다.
- Candidate ranking loss는 이 prior를 **직접** 학습하므로, 보정 없이 학습하면 이전 실패를 더 강하게 재현한다.

**대응 (§5.2에 반영)**: 그룹 가중치를 분포 정합으로 계산한다. train G1:G2 = 3.12:1, heldout G1:G2 = 1.76:1 → 정합 가중치비 `w_G2/w_G1 = 1.77`. 방법론 문서의 기본값 `w_G1=0.5, w_G2=2.0`은 비율 4.0으로, 정합점을 2.3배 초과하는 **의도적 oversampling**이다. 둘을 구분해서 스윕하고 둘 다 보고한다.

### 1.4 하드웨어·환경

- H200 143GB × 2, 현재 둘 다 유휴. 관례: 한 GPU에 한 잡, `cuda:1` = retro/pro 학습 레인, `cuda:0` = DPO·보조 평가 레인. 스케줄러 없이 `$EGO_ROOT/runs/f0_battery/*` 마커 파일 폴링으로 GPU를 넘긴다.
- Python: `/mnt/nvme/migration/jihun/envs/miniforge3/envs/eve-cu124/bin/python` (torch 2.6.0+cu124, transformers 5.9.0, trl 1.5.1, peft 0.19.1)
- 관측된 실행 시간 (계획 산정 근거): REINFORCE 5,000 샘플 ≈ **3.8h**, DPO 267 step ≈ **2h**, 생성 평가 500샘플 ≈ **4.6분** (32배치 × 8.7s) → 1,417샘플 ≈ **13분**, teacher 데이터 빌드 (MVP 1,500 프롬프트) ≈ **3h**.

---

## 2. 착수 전 위협 등록부 (Threat Register)

각 항목에 **판정 규칙**을 미리 붙였다. 실험 중에 규칙을 바꾸지 않는다.

### T1 ★ belief 출력 자체가 정확도에 손해다

동일 heldout 500 기준:

| run | belief 출력 | acc |
|---|---|---:|
| `f0gr_final` (action-only) | ✗ | **0.338** |
| `base_actiononly` | ✗ | 0.270 |
| `f0wema_final` | ✓ | 0.280 |
| `b0p12_gen_1f` | ✓ | 0.264 |
| `base_1f_strict` | ✓ | 0.242 |
| `belief_sum_wm` | ✓ | 0.230 |

belief를 출력하는 모든 run이 belief를 출력하지 않는 최고 run보다 낮다. retro v2의 전제("좋은 belief가 좋은 action을 만든다")와 정면으로 충돌한다.

**판정 규칙 (사전 등록)**: Stage T+A 종료 시점에 `candidate_scored_acc`가 **동일 파이프라인의 no-belief 대조군**(reasoning만, belief 태그 제거)보다 낮으면, "belief가 정확도를 올린다"는 주장을 폐기하고 논문 주장을 **"정확도 동등 하에서의 belief 제어가능성(controllability)"**으로 전환한다. 이 전환을 지금 미리 승인해 두어야 나중에 데이터에 맞춰 주장을 사후 조정했다는 비판을 피한다.

> 주의: 위 비교는 action-only run이 **더 쉬운 프롬프트 레짐**에서 측정된 것이라 직접 랭킹이 불가능하다(07-20 문서 명시). 그래서 대조군은 반드시 **동일 프롬프트·동일 평가 경로**로 새로 측정해야 한다 (P0.3).

### T2 ★ train/heldout 그룹 분포 불일치

§1.3 참조. **판정 규칙**: 모든 학습 손실은 그룹 가중 적용 후 보고하고, train-side 정확도는 어떤 gate의 근거로도 쓰지 않는다. Stage A ablation에 `w_ratio ∈ {1.0 (무보정), 1.77 (정합), 4.0 (문서 기본)}` 3점을 반드시 포함한다.

### T3 ★ 측정 노이즈가 지금까지의 모든 효과보다 크다

- n=500 이항 SE = 0.020, 관측된 subset 스프레드 = 0.038, `eval_harness_v2`가 계산한 **MDE ≈ 0.064**.
- 지금까지 관측된 모든 효과(0.02~0.03)는 이 아래다. 07-20에 3회 repro가 완전히 동일한 `0.264`를 낸 것은 `do_sample=False` 때문이며, 정보량 0이었다.

**대응**: (a) 항상 full heldout n=1,417, (b) **arm 간 독립 비교가 아니라 동일 샘플 paired delta**로 판정 — paired bootstrap과 McNemar는 두 개의 독립 500샘플 arm보다 SE가 훨씬 작다. `eval_harness_v2`가 이미 paired 구조를 지원한다. (c) 사전 등록 임계값은 arm 절대값이 아니라 **paired delta CI 하한 > 0**으로 쓴다.

### T4 teacher mode collapse

P12 빌드 시 4회 시도 중 **55.9%(219/392)가 전부 동일**했고, `turn-off|tap ×47` 같은 반복이 상위를 차지했다. teacher gate 통과율도 R1 ≈53% → P12 15.3%로 붕괴했다.

**대응**: counterfactual/paraphrase belief 생성에 반드시 diverse decoding (T=1.0, top_p=0.95) + 시도 간 중복 제거 + 동일 belief 템플릿 반복 상한을 건다. 데이터 빌드 리포트에 **belief 유니크율**을 필수 지표로 넣고, 0.7 미만이면 학습을 시작하지 않는다.

### T5 ③ (causal sensitivity)의 표본이 너무 작다

실제 카운트: `swap_b0p12` control 1/499, swap 4/499 → ③=0.006. `swap_abB` control 4/497, swap 6/497 → ③=0.0041. 이 CI들은 거의 확실히 겹친다. `causal_sensitivity`는 이미 반올림된 두 비율의 차라서 정밀도도 잃는다 (`eval_belief_swap.py:173-174`).

**대응**: hard flip rate를 **주지표에서 내리고**, 연속량 Δ_semantic (candidate 확률 이동)을 주지표로 승격한다. hard ③은 보조로만 보고하되 반드시 부트스트랩 CI를 붙인다. 이게 n=1,417에서 검정력을 얻는 유일한 방법이다.

### T6 belief restatement가 ③ 상승을 흉내낸다

reward를 gt→wm으로 바꿨을 때 ③은 0.0135→0.0255로 올랐지만 restatement rate가 0.0191→0.0722로 3.8배 올랐다. belief가 action을 **복창**하면 belief를 바꿀 때 action이 따라 바뀌는 게 당연하다 — 이건 causality가 아니다.

**대응**: 모든 causality 지표를 **lexical-overlap 샘플 제외 서브셋에서도** 보고한다 (`③_excl_restatement`). Gate C는 제외 서브셋 기준으로 판정한다.

---

## 3. 신규 작성 범위 재산정 — 이미 있는 것 / 없는 것

방법론 문서가 "새로 구현"으로 나열한 것 중 상당수가 이미 존재한다. 실제 신규 작성량은 문서가 암시하는 것보다 작다.

### 3.1 이미 있고 그대로 재사용

| 필요한 것 | 이미 있는 위치 | 상태 |
|---|---|---|
| **candidate-normalized 스코어러** | `scripts/step2/pro_gx_train.py:93-118` | ★ 정확히 필요한 것. base + 5개 candidate 완성문을 배치로 만들고 prefix 마스킹 후 `(tokl*mask).sum(dim=1)` → `log_softmax`. **추출만 하면 됨** |
| full heldout + bootstrap CI + subset + MDE | `scripts/step2/eval_harness_v2.py` | 완성. `--limit 0`, `--n_boot 10000`, `mde` 블록 포함 |
| ③ 재집계 + CI | `eval_harness_v2.causal_from_swap_records:156-184` | 완성 (paired, per-sample) |
| run provenance | `src/ego/common/run_provenance.py` | 완성. git SHA/dirty, 데이터 sha256+행수, 패키지 버전까지 기록 |
| 프롬프트 빌더 | `train_grpo_action.build_joint_conversation:442` | 그대로 써야 함 (평가와 학습이 반드시 동일해야) |
| trace 파싱/직렬화 | `retro/trace_utils.py` | `parse_full_trace`, `build_full_trace`, `canonical_action` |
| G1/G2/OUT 라우팅 | `retro/build_dpo_dataset_r1.sample_group:30` | 재사용 (단 `build_pairs_contrastive.py:36`에 중복 구현 있음 — 통합할 것) |
| GT leakage 검사 | `retro/teacher.py:245 goal_leaks` + `validate_dpo_dataset.check_prompt_leakage` | 재사용 |
| future-leak 문구 스크리닝 | `trace_utils.FUTURE_LEAK_MARKERS:90` | 재사용 |
| span별 logp 귀속 | `scripts/step2/rerank_bo8.py:59-89 seq_stats` | char-offset → span 귀속. 진단용으로 재사용 |
| 미래 suffix에서 goal 추출 | `teacher.py:189 goal_prompt` + `GatedTeacherMixin.extract_goal:257` | 재사용 (retry/forbid 로직 포함) |

### 3.2 없어서 새로 만들어야 하는 것

| 필요한 것 | 왜 없는가 | 난이도 |
|---|---|---|
| ★ **token-level span mask** | `retro/` 전체에 tokenizer 인지 마스킹 헬퍼가 **하나도 없다**. 유일한 "mask"는 `validate_dpo_dataset._mask_history_section:39`인데 문자열 치환이다 | **높음 — 최대 리스크** |
| GT 숨긴 prefix-only projection | `teacher.projection_prompt:68`은 GT를 프롬프트에 **넣는다**(`Target next action …`) 그리고 `project_full_trace:157`이 파싱 결과를 GT로 덮어쓴다. GT 숨긴 경로(`gated_trace_prompt:210`)는 있으나 action까지 생성하고 **hard action gate**(예측==GT일 때만 통과, `generate_gated_trace:273-293`)를 건다 | 중간 |
| projection-quality auditor | 현재는 hard action gate가 그 역할을 대신함 (통과율 15~53%, 선택 편향) | 중간 |
| counterfactual / paraphrase belief 생성기 | 없음. `equivalence_prompt:76`(SAME/DIFFERENT 판정)만 있음 — 검증에는 쓸 수 있음 | 중간 |
| 후보 스코어링 **평가 CLI** | 스코어러는 학습 루프 안에만 있고 eval 진입점이 없다. `eval_battery`/`eval_harness_v2`는 **생성 전용 greedy** | 낮음 |
| span-weighted SFT trainer | `train_sft.py`는 TODO scaffold. `train_retro_dpo.py`는 TRL DPO 전용 | 높음 |
| ranking / counterfactual / paraphrase loss | 없음 | 중간 |
| YAML 설정 로딩 | Step-2 하이퍼파라미터는 전부 shell의 CLI 플래그로만 존재 (`configs/step2/b0_full_trace_dpo.yaml`은 존재하지만 미사용) | 낮음 |

**결론**: 실질 신규 코드는 ①span mask ②SFT/ranking/causal trainer ③teacher 4개 함수 ④데이터 빌더 2개 ⑤평가 CLI 2개. 스코어러와 측정 인프라는 재사용이므로, 방법론 문서의 §7 목록보다 약 30~40% 작다.

---

## 4. Phase 0 — GPU 거의 없이 방향을 정하는 단계 (약 3시간)

**이 단계를 건너뛰면 안 된다.** 여기서 나오는 숫자에 따라 Phase 1 이후가 통째로 바뀔 수 있다.

### P0.1 소스 동결 (GPU 0h)
§1.1대로 정리 → 작업 브랜치 `retro-v2` 생성 → SHA 기록. 이후 모든 run은 `run_provenance.write_run_config`를 호출한다 (`pro_gr_train.py`는 이미 호출, `eval_battery`/`eval_belief_swap`은 **호출하지 않음** → 호출 추가 또는 `eval_harness_v2`로 일원화).

### P0.2 베이스라인 full-heldout 재측정 (GPU ~1.5h)
`eval_harness_v2.py --limit 0` (n=1,417)로 아래를 동일 조건 재측정. 일부는 이미 있음.

| arm | adapter | 상태 |
|---|---|---|
| `base` | 없음 | 신규 필요 |
| `pro_wema` | `outputs/step2/f0_wema_fulltrace_1f/checkpoint-final` | 신규 필요 (기존 값은 n=500) |
| `retro_p12` | `outputs/step2/b0_p12_1f/checkpoint-267` | 신규 필요 |
| `retro_belief_sum_wm` | `outputs/step2/retro_belief_sum_wm_1f/checkpoint-final` | **이미 있음** (`EGO/runs/retro_overnight/eval_belief_sum_wm.json`, acc 0.2484) |
| `retro_belief_sum_gt` | 〃 `_gt_1f` | **이미 있음** (acc 0.2371) |

산출물: `BASELINES_RETRO_V2.json` — arm × {acc, acc_ci95, g2_acc (n=321), wm_follow, parse_rate, belief_restatement_rate, mean_reasoning_words}.

### P0.3 ★★ candidate-scored 베이스라인 — 학습 0, 정보량 최대 (GPU ~1h)

`pro_gx_train.py:93-118`의 스코어러를 `retro/action_scoring.py`로 추출하고, 학습 없이 **위 arm들 전부에 대해** 두 지표를 나란히 측정한다.

- `free_action_acc` — 기존 경로 (생성)
- `candidate_scored_acc` — 생성된 `<reasoning>/<task_belief>` prefix를 고정하고 top-5를 teacher forcing으로 스코어링해 argmax

**이 한 번의 측정이 답하는 질문**: 지금까지의 정체가 *모델이 정답을 모르는 것*인지 *알지만 디코딩에서 흘리는 것*인지.

| 관측 | 해석 | 후속 조치 |
|---|---|---|
| candidate_scored ≫ free (예: 0.34 vs 0.26) | 지식은 있고 **디코딩이 병목** | Stage A의 기대값이 크게 올라감. §4의 two-pass 추론만으로도 큰 이득. 계획 그대로 진행 |
| 둘이 비슷 | 후보 구분 정보 자체가 부족 | ranking loss의 상한이 낮다. Stage C(causality)와 데이터 품질에 무게를 옮기고, 정확도 목표를 하향 사전등록 |
| candidate_scored < free | 스코어링 구현 버그 | §6 단위테스트로 회귀 |

동시에 **no-belief 대조군**(동일 프롬프트에서 belief 태그를 제거한 prefix)도 같이 재서 T1의 판정 기준선을 만든다.

### P0.4 데이터 그룹 리포트 (GPU 0h)
§1.3 수치를 스크립트로 고정 산출 (`retro/report_group_stats.py`), heldout G2 n=321 및 OUT 37.4%를 모든 결과표의 각주로 강제한다. OUT은 학습 제외지만 **heldout 정확도 분모에는 포함**되므로 상한이 0.626이다.

**Phase 0 종료 gate (Gate 0)**: `BASELINES_RETRO_V2.json` 존재 + candidate-scored 경로가 단위테스트 5종 통과 + 그룹 리포트 존재. 실패 시 Phase 1로 넘어가지 않는다.

---

## 5. 구현 계획 — 5개 웨이브

의존성 순서. 웨이브 1은 GPU 없이 전부 테스트 가능하므로 **먼저 끝내고** GPU 작업을 시작한다.

### Wave 1 — GPU 없는 코어 (신규 3파일)

**`src/ego/step2_vlm_alignment/retro/action_scoring.py`**
`pro_gx_train.py:93-118`에서 추출. 공개 함수: `serialize_action_candidate(verb, noun)`, `build_candidate_sequences(base_text, candidates)`, `score_candidate_actions(model, processor, conv, candidates, image) -> Tensor[5]`, `candidate_log_softmax(scores, tau)`, `candidate_margin(scores, gt_idx)`.
주의: 원본은 **sum-logp**를 쓴다. 후보 문자열 길이가 다르므로 길이 정규화 여부를 플래그로 노출하고 **둘 다 P0.3에서 측정**한다 (어느 쪽이 맞는지는 경험적 문제다).

**`retro/span_masks.py`** — 최대 리스크 구간
`build_full_trace`가 만든 완성문에서 `<reasoning>/</reasoning>`, `<task_belief>/</task_belief>`, `<action>/</action>` 마커를 **토크나이즈한 뒤 토큰 subsequence 탐색**으로 찾아 span mask를 만든다. 문자 길이→토큰 인덱스 변환 금지 (tokenizer 경계에서 어긋난다). 마커를 못 찾으면 샘플을 drop하고 audit에 기록한다.
참고: `pro_gr_train.py`의 `_action_token_start`가 이진탐색+decode 방식으로 `<action>` 시작점을 찾는 선례다. 다만 그건 단일 태그용이고 3태그 span에는 부족하다.

**`retro/validate_retro_v2_dataset.py`**
검사: prompt 내 GT/future leak 0, GT ∈ candidates, 3태그 완전성, factual/paraphrase SAME, factual/counterfactual DIFFERENT, exact action lexical overlap, candidate ID 일관성, trace 중복, **train/heldout sample_id 교집합 0**, belief 유니크율 ≥ 0.7 (T4).

### Wave 2 — teacher & 데이터 빌더 (GPU 필요)

`teacher.py`에 추가 (기존 함수는 건드리지 않고 병렬 추가 — 기존 baseline 재현성 보존):
- `project_prefix_hidden_gt(goal, memory_context, candidates, image_path)` → `<reasoning>`/`<task_belief>`만 생성. GT 미노출. `gated_trace_prompt`를 기반으로 하되 action 생성 요구를 제거.
- `audit_projected_prefix(prefix, gt, context)` → PASS/FAIL + 태그. **rewrite 금지, GT 열람 허용.** hard action gate를 대체 → G2 보존율이 올라가는지가 핵심 관전 포인트 (기존 gate 통과율 15.3~53%).
- `generate_counterfactual_belief(x, b_plus, a_minus, candidates)` → `a_minus = WM top-1`을 선호하게 만드는 최소 의미 변경 belief. diverse decoding 필수 (T4).
- `generate_belief_paraphrase(b_plus)` → 의미 동일, 표현 변경. candidate 단어 사용 금지.

빌더 2종: `build_hindsight_sft_dataset.py` (G1/G2/OUT 라우팅 → goal 추출 → hidden-GT prefix → GT action 부착 → audit), `build_counterfactual_dataset.py` (G2 대상 `a-`/`b-`/`b~+`). **split을 먼저 고정한 뒤** counterfactual을 만든다 (학습용 belief를 heldout intervention에 재사용 금지).

비용 추정: goal+prefix+audit = 4,618 샘플 × 3콜, cf+para = 1,120 × 2콜 ≈ **16k teacher 콜**. 콜당 ~1.5s면 단일 GPU 6.7h → 2 GPU 샤딩(`--shard/--num_shards`, r1 빌더에 이미 있음)으로 **3~4h**.

### Wave 3 — trainer `retro/train_retro_v2.py --stage trace|rank|causal`

- 초기화: `f0_wema_fulltrace_1f/checkpoint-final` LoRA에서 이어받기 (방법론 문서 결정 사항)
- Stage T: span-weighted SFT (`w_r=0.30, w_b=0.70, w_a=1.00`), span 내부 mean → span 간 weight
- Stage A: `L_rank = -log q(a+|x,r+,b+)`, 그룹 가중치는 §1.3에 따라 `w_ratio` 스윕
- Stage C: `L_cf-ce`, `L_flip`(m=0.2), `L_para`(JS)
- 보호 장치: G1 replay 15~25%, F0 reference KL anchor (G1 0.5 / G2 0~0.05), adapter parameter anchor `‖Δθ−Δθ_T‖²`
- 로깅: `loss/{reasoning,belief,action_sft,rank,cf_ce,flip,paraphrase,f0_anchor,parameter_anchor}` + grad norm, `training_history.csv`, `run_config.json`
- **무학습 가드**: 기존 체인의 safetensors max-abs-diff ≤1e-7 검사를 그대로 이식 (`retro_full_chain.sh` S3에 선례)

### Wave 4 — 추론·평가

`retro/infer_retro_v2.py` (two-pass: prefix 생성 → candidate 스코어링 → trace 조립), `scripts/step2/eval_retro_v2.py` (`eval_harness_v2` 위에 candidate-scored 경로 추가, generated-prefix vs teacher-prefix gap 분리 보고), `scripts/step2/eval_belief_intervention_v2.py` (조건 5종: factual / paraphrase / semantic counterfactual / random / no-belief, 각 조건에서 top-5 확률 기록 → Δ_semantic).

### Wave 5 — 설정·체인

`configs/step2/retro_v2.yaml` (숫자를 shell에서 제거), `scripts/step2/retro_v2_chain.sh` (기존 체인 관례 준수: `say()`/`die()` + `tee` 로그, 스텝별 `[ -s file ]` 멱등 가드, `TRAINING_DONE` 마커, smoke→full 패턴, 마커 파일 버스 `$EGO_ROOT/runs/f0_battery/`, 결과 md 자동 생성).

---

## 6. GPU 전에 통과해야 하는 단위 테스트

`tests/step2/test_retro_v2/`:
1. candidate 순서를 셔플해도 canonical score 매핑이 동일
2. 동일 candidate 두 개는 동일 score
3. 공통 JSON scaffold가 rank를 왜곡하지 않음 (모든 후보에 동일 접두/접미)
4. batch score == single-sample score
5. 길이 정규화 on/off 각각에서 결정적
6. span mask: 마커 subsequence 탐색이 3태그 모두에서 정확한 경계를 반환, 마커 누락 시 drop
7. train/heldout `frame_id` 교집합 0

---

## 7. 실행 일정 (2 GPU, 총 약 26~30h)

| 구간 | 내용 | GPU | 시간 | 산출물/게이트 |
|---|---|---|---:|---|
| A | P0.1 소스 동결 + Wave 1 코딩 + 단위테스트 | — | 4h | Gate 0 (일부) |
| B | P0.2 베이스라인 full heldout ×3 arm | cuda:0 | 1.5h | `BASELINES_RETRO_V2.json` |
| C | ★ P0.3 candidate-scored 베이스라인 + no-belief 대조군 | cuda:1 | 1h | **방향 결정** |
| D | Wave 2 코딩 | — | 4h | |
| E | 데이터 빌드 smoke (200 샘플) + **사람이 50개 육안 검수** | cuda:0/1 | 1h | Gate D |
| F | 데이터 빌드 full (2 GPU 샤딩) | 양쪽 | 3.5h | `validate_retro_v2_dataset` 통과 |
| G | Wave 3 코딩 | — | 5h | |
| H | Stage T smoke(300) → full | cuda:1 | 3h | **Gate T** |
| I | Stage A smoke → full (`w_ratio` 3점) | cuda:1 | 4h | **Gate A** |
| J | Stage C smoke → full | cuda:1 | 3h | **Gate C** |
| K | 최종 평가 전 arm + intervention | 양쪽 | 2h | `RETRO_V2_RESULTS.md` |

구간 F 이후로는 cuda:0이 놀기 때문에, **Phase 0에서 미룬 pro 트랙 숙제**(별도 split 재측정, seed 2회 반복)를 병렬로 돌리는 것을 권한다.

---

## 8. Gate — 사전 등록 판정 기준

측정된 베이스라인이 아직 full heldout으로 갱신되지 않았으므로 (P0.2 후 확정), 아래는 **provisional**이다. P0.2 종료 즉시 숫자를 고정하고 그 후에는 바꾸지 않는다.

**Gate D (데이터)**: train/heldout 교집합 0 · GT leak 0 · future leak hard-rule 위반 0 · counterfactual exact action restatement < 5% · **belief 유니크율 ≥ 0.7** (T4) · G2 보존 수가 hard action gate 대비 유의하게 많음 (기존 15.3~53% 대비 개선 확인)

**Gate T (trace SFT)**: parse_rate ≥ 0.995 · belief_restatement가 pro W-EMA 대비 +1pp 이내 · reasoning 길이 중앙값이 레퍼런스 ±15% · G1 candidate-scored acc 하락 ≤ 1.5pp · generated-prefix vs teacher-prefix gap이 베이스라인보다 악화되지 않음

**Gate A (ranking)** — 아래 중 최소 하나 + 보존 조건 동시 충족:
- 전체 paired delta CI 하한 > 0 (T3에 따라 **paired**로 판정, arm 절대값 비교 아님)
- 또는 G2 acc +3pp 이상 (n=321, SE≈0.026 → 3pp는 약 1.2σ이므로 paired CI 병기 필수)
- 또는 G2 `GT_vs_WM1_margin` paired CI 하한 > 0
보존 조건: G1 regression ≤ 1.5pp · Gate T 유지 · `wm_follow`만 오르고 G2가 안 움직이는 패턴 아님

**Gate C (causality)**: **lexical-overlap 제외 서브셋**에서 Δ_semantic > 0 (주지표, T5·T6) · hard ③ > 0.05 (보조, CI 병기) · paraphrase action-change ≤ 0.02 · control action-change ≤ 0.01 · `Δ_semantic > 2 × paraphrase drift` · Gate A 유지

**중단 조건**: Gate C 실패 시 loss weight를 올리지 말고 **counterfactual 생성 품질을 먼저 감사**한다. Gate A까지 실패하면 남는 후보는 아키텍처 변경(belief를 출력이 아니라 action 디코딩의 **조건 입력**으로 강제)이며, 이번 범위 밖이다.

---

## 9. Ablation 계획

전체 factorial을 돌리지 않는다. 단계별로 한 변수씩 더하는 **누적 ablation**.

### 9.1 주 ablation (누적)

| ID | 설정 | 검증 질문 |
|---|---|---|
| `L0` | 기존 retro DPO (P12) | 기존 방식 baseline |
| `T` | projected-trace SFT | DPO 없이 trace를 직접 학습하면 재생산되는가 |
| `T+A` | + candidate ranking | action correctness가 직접 개선되는가 |
| `T+A+K` | + G1 anchor/replay | selective trust가 보존되는가 |
| `T+A+K+C` | + counterfactual | belief causality가 생기는가 |
| `T+A+K+C+P` | + paraphrase invariance | 복창이 아닌 semantic causality인가 |

### 9.2 데이터 ablation (동일 고정 subset)

1. GT 노출 projection vs **hidden-GT** projection — answer-conditioned rationalization의 영향
2. hard action gate vs **projection-quality gate** — G2 보존율과 최종 성능 (기존 통과율 15.3%/53% 기록이 비교 기준)
3. future goal 포함 vs 제거 — hindsight projection 자체의 기여
4. ★ **그룹 가중치 `w_ratio ∈ {1.0, 1.77, 4.0}`** — §1.3의 분포 불일치 보정 강도 (이번 라운드 신규, T2)

### 9.3 추론 ablation — 학습 비용 0, 최우선

free generation / candidate scoring + teacher prefix / candidate scoring + generated prefix / no-belief prefix / wrong-belief prefix. **P0.3에서 base 모델에 대해 먼저 돌린다.**

### 9.4 causality ablation

random swap only / semantic counterfactual only / +paraphrase / lexical-overlap 포함·제외 / reasoning 고정 vs reasoning+belief 동시 반사실. **첫 구현은 reasoning 고정**(조작변수 명확화).

### 9.5 조건부 (MVP 이후)

trace adapter merge/freeze 후 decision LoRA 추가, layer 분리 LoRA, multi-adapter composition. PEFT composition 복잡도 때문에 원인 분리가 어려워지므로 planning regression이 지속될 때만.

---

## 10. 보고 형식

모든 결과표는 다음을 강제한다.

- n을 명시 (full heldout 1,417 / G2 서브셋 321)
- `free_action_acc`와 `candidate_scored_acc`를 **분리** 기재
- 정확도는 paired delta + 95% CI 병기, 단일 arm 절대값만으로 우열을 주장하지 않음
- causality는 Δ_semantic (주) + hard ③ (보조, CI) + restatement 제외 서브셋 값
- OUT 37.4%로 인한 상한 0.626을 각주로 표기
- 각 run에 `run_config.json` (git SHA, dirty 여부, 데이터 sha256) 첨부

---

## 11. 지금 주장하면 안 되는 것

- 긴 reasoning = faithful reasoning
- LLM judge coherence만으로 belief causality 입증 (judge 스프레드 0.64/14로 이미 진단용 강등됨)
- random belief swap 민감도 하나로 causal intermediate variable 단정
- DPO가 Retrospection의 본질
- WM reward가 hindsight reasoning을 만든다

---

## 12. 착수 체크리스트

**P0 — 코드 작성 전**
- [ ] `EGO_jihun` 로컬 변경분 정리, origin/main `f10e327` 기준 재정렬, `retro-v2` 브랜치
- [ ] baseline adapter 경로/해시 고정
- [ ] heldout sample_id 고정

**P0 — 측정 (GPU)**
- [ ] `BASELINES_RETRO_V2.json` (full heldout, 5 arm)
- [ ] ★ candidate-scored 베이스라인 + no-belief 대조군 → T1 판정선 확정
- [ ] 그룹 분포 리포트 스크립트화

**Wave 1 (GPU 없음)**
- [ ] `action_scoring.py` 추출 + 단위테스트 5종
- [ ] `span_masks.py` + 단위테스트 2종
- [ ] `validate_retro_v2_dataset.py`

**Wave 2 (데이터)**
- [ ] `project_prefix_hidden_gt` / `audit_projected_prefix`
- [ ] `generate_counterfactual_belief` / `generate_belief_paraphrase` (diverse decoding)
- [ ] 빌더 2종 + **사람 육안 검수 50건**

**Wave 3~5**
- [ ] `train_retro_v2.py` 3-stage + 무학습 가드 + 컴포넌트별 로깅
- [ ] `infer_retro_v2.py` / `eval_retro_v2.py` / `eval_belief_intervention_v2.py`
- [ ] `configs/step2/retro_v2.yaml` / `scripts/step2/retro_v2_chain.sh`

---

## 13. 근거 문서·코드

문서: `2026-07-21_retro_v2_methodology_and_implementation_handoff.md` (방법론), `2026-07-20_retro_ab_results_and_next_10h_handoff.md`, `2026-07-21_step1_night_and_retro_belief_sum_handoff.md`, `2026-07-19_b0_teacher_refactor_handoff.md`, `2026-07-20_f0_results_b0_prevalidation_handoff.md`, `docs/NAMING.md`

코드: `retro/{teacher,trace_utils,build_dpo_dataset,build_dpo_dataset_r1,build_pairs_contrastive,route_pairs,train_retro_dpo}.py`, `train_grpo_action.py:442`, `scripts/step2/{pro_gx_train,pro_gr_train,eval_battery,eval_belief_swap,eval_harness_v2,rerank_bo8}.py`, `src/ego/common/run_provenance.py`, `scripts/step2/{retro_full_chain,pro_retro_ab_chain,retro_overnight_gpu1_v3,build_retro_pairs}.sh`

실측 (2026-07-21, 이 문서 작성 시): 그룹 분포 §1.3, 데이터 행 수 §1.2, GPU 유휴 상태 §1.4
