# FAA → HP-DPA 계획 리뷰 — B0 담당자 전달용

2026-07-18 · 평가 대상: `EGO_STEP2_FAA_HP_DPA_HANDOFF.md`
대조 근거: `docs/experiments/2026-07-17_f0_final.md` (F0 실측) · `docs/experiments/2026-07-18_f0_handoff.md` (진단·개선 방향) · `src/ego/step2_vlm_alignment/train_grpo_action.py` (검증 코드)
평가 UI: https://claude.ai/code/artifact/d93184c4-a349-4e27-a38a-9c7a8355010a

---

## 0. 총평

**골격 채택 — 패치 조건부 승인.**

정보 경계 계약(§3.2, §24), 이중 통제 DPO pair, hindsight projection 규칙, freeze 규율,
주장별 ablation 설계는 F0 실측이 요구하는 바로 그 구조다. 다만 **FAA 스펙 3곳이 실측 검증값과
충돌**하고(그중 1곳은 실측으로 확인된 붕괴 설정), **belief pair에 교란 2건**이 있으며,
**F0 진단(학습량·관측 비대칭)과 사전 검증 배터리가 미반영**이다. 아래 C1~C5 반영 없이
FAA를 학습·freeze하면 실패하거나 저품질이 파이프라인 전체에 불가역으로 고정된다.

## 1. 잘 설계된 것 (유지)

1. **이중 통제 pair가 belief 문제의 정면 해법.** F0 실측의 두 결함 — belief 재진술화
   (globality 2.00→1.55)·belief→action 인과 부재 — 에 대해 belief pair는 action을 byte-level로
   고정하고, action pair는 belief/reasoning을 고정한다 (§24.2 assert). DPO가 쉬운 action 토큰
   차이로 belief 감독을 우회하는 것을 구조적으로 차단.
2. **FAA의 퇴화가 HP-DPA의 연료.** FAA의 위축된 belief가 belief pair의 자연 rejected 공급원 —
   결함이 학습 데이터로 재활용되는 설계.
3. **history cutoff 버그가 스펙에서 수정됨.** §24.1 `assert all(a.stop_time <= trigger_time)` —
   F0 run을 오염시킨 4.6% 직전-행동 누설을 구현 전에 계약으로 차단.
4. **projection 규칙(§13.1)** — 미래 첫 등장 명사 금지, specificity 하향, no-projection ablation.
5. **candidate-support 필터 + 3중 지표** (recall / conditional / end-to-end) — 상한선 구조
   (논리 0.620 · 실질 0.374)와 정합.
6. **Qwen2.5-VL-7B 채택** — 모델 효과 실측(−0.018, base G2 0.333)과 일치.
7. **주장 범위의 정직한 축소** (§1.2 "global-task-annotation-free") — F0의 GT-free 주장(리워드 GT 0)과
   겹치지 않게 분리됨.

## 2. 반드시 고칠 것 — C1~C5

### C1 (치명) `num_generations: 4`는 실측으로 확인된 붕괴 설정

F0 final 실측: generation 4에서 `frac_reward_zero_std` **0.5~0.75** — 그룹 리워드 분산이 0이 되어
gradient가 사라진다. 8에서 0으로 해소. §7의 "candidate 수와 별개"라는 해명은 맞지만 이 실측을
반영하지 못했다. (HP-DPA용 trace 생성 4개는 무관 — 문제는 GRPO 학습 그룹 크기.)

→ **패치: `num_generations: 8` + `frac_reward_zero_std` 필수 로그.**

### C2 (치명) FAA 학습량 미지정 — "+0.028짜리 FAA"를 freeze하게 됨

`max_steps`가 "실험별 보존" 목록에만 있다. 실측: 500 step = 프롬프트 방문 1,000회 =
**데이터 5,000개의 20%를 1회** 본 것 (TRL GRPO에서 `per_device_train_batch_size`는 생성 단위 →
GPU당 프롬프트 1개/step). acc(0.228→0.252→0.258)·wm_follow(0.329→0.342→0.351) 곡선이 500에서
아직 단조 상승 중이었다. HP-DPA는 FAA를 초기값·reference·rejected 공급원으로 삼으므로
덜 학습된 FAA를 freeze하면 그 품질이 파이프라인 전체에 고정된다.

→ **패치: `max_steps 2500~5000` + `gradient_accumulation_steps 4~8`, 250 step마다 eval.
G1 구간 일치율(현재 0.49, 목표 0.85)과 G2를 동시 추적, G2가 우연(0.20) 밑으로 무너지기 전
체크포인트를 freeze 대상으로 선정.**

### C3 (높음) `temperature 0.8`은 검증값이 아니라 argparse 기본값

검증 실행 커맨드는 **T=1.0** (1.2는 쓰레기 토큰 실측, 0.8은 미검증이며 다양성 축소로 C1의
zero-std 붕괴를 악화시키는 방향). 문서 자신이 §9에서 경고한 "실험 YAML/command가 source of
truth" 원칙과 충돌.

→ **패치: `temperature 1.0`. 아울러 미지정 세부값 고정 — `loss_type dr_grpo`,
`scale_rewards none`, `epsilon_high 0.28`, `beta 0`, `min_wm_spread 0.05`,
`max_completion_length 384` (Qwen2.5에서 `clipped_ratio` 재실측 후 조정 가능).**

### C4 (높음) belief pair의 문체 교란

chosen의 belief/reasoning은 teacher(base Qwen, LoRA off)가 쓰고 rejected는 FAA가 쓴다.
두 모델의 문체가 체계적으로 다르면 DPO는 "과거-근거 전역 belief 선호"가 아니라 **teacher 문체
판별**을 배울 수 있다 — belief preference accuracy가 올라도 실제 belief 능력 개선이 아니게 된다.

→ **패치: 쌍 구성 전 스타일 정규화(동일 모델로 양쪽 내용 보존 재작성) 또는 길이·어휘 통계 매칭
필터. 검증: "문체만 보고 chosen을 맞히는 판별기"의 정확도가 chance 근처인지 확인.**

### C5 (높음) belief pair rejected의 자기모순 교란

rejected = (FAA reasoning + GT action) 조립인데, FAA가 다른 action을 논증한 trace라면(FAA≠GT)
rejected는 결론이 뒤바뀐 자기모순 완성문이 된다 → 선호 신호가 "belief가 나빠서"가 아니라
"모순이라서"가 되어 belief 감독이 희석된다. 라우팅 표(§20)의 DIFFERENT×(FAA≠GT) 행이 이 경우.

→ **패치: belief pair를 DIFFERENT×(FAA=GT) 행 우선으로 구성하거나, (FAA=GT)/(FAA≠GT) 서브셋을
분리 로깅해 어느 쪽이 preference accuracy를 끌고 가는지 감시.**

## 3. 보완 권고 — M1~M6

| # | 내용 | 패치 |
|---|---|---|
| M1 | **관측 비대칭 결정을 freeze 전에.** FAA 입력이 프레임 1장, 리워드 심판(V-JEPA2)은 4초×32프레임. freeze 후 입력 계약은 불가역 | 배터리 ⑤(클립 8프레임 base 평가, 학습 0회)를 §32 FAA acceptance 게이트로. 상한 상승 확인 시 클립 입력 채택 |
| M2 | 평가에 **G2·wm_follow 부재** (§26·27). F0에서 학습 실효를 가른 핵심 지표 | G2(WM top-1 오답 ∧ GT∈top-5 구간 acc)·wm_follow 추가. "recovery over FAA"를 G2 구간 정의로 표준화 |
| M3 | 주장 3("두 pair가 belief·action 동시 개선")에 **인과성 직접 측정 없음** — §27.3은 상관 지표 | belief-swap 개입 테스트 + judge `belief_action_link` 항목을 FAA baseline과 HP-DPA 사후에 동일 적용. swap 민감도 상승 = 주장 3의 직접 증거 |
| M4 | e_t^proj의 **연성 누설** — 명시적 명사 복사는 막지만 어조·구도 유도 잔존 가능 | leak 프로브: e_proj만 보고 GT action을 맞히는 판별기 정확도가 chance 근처인지. no-projection ablation과 교차 확인 |
| M5 | GT∉D_t **38% 구간의 belief 감독 전량 손실** (§19 필터가 belief pair에도 적용) | 논문 한계에 명시 + drop 카운터의 belief/action별 분리 집계 |
| M6 | **judge 정책 미반영** + 소소한 것들 | judge = gemini-2.5-pro 강제 (Claude 계열 배제 — 자기선호 편향). frozen FAA trace 생성의 T/top_p 스펙화. 같은 chosen이 최대 4쌍에 반복되는 편향 → per-sample pair 상한 또는 가중치 |

## 4. 패치 스펙 요약 (그대로 반영 가능)

| 항목 | 계획서 | 패치 | 근거 |
|---|---|---|---|
| `num_generations` | 4 | **8** | 4는 zero-std 0.5~0.75 붕괴 (실측) |
| `temperature` | 0.8 | **1.0** | 검증 커맨드 값 |
| `max_steps` | 미지정 | **2,500~5,000 + grad_accum 4~8**, 250마다 eval | 500 step = 데이터 20% 1회, 미수렴 |
| GRPO 세부 | "실험별 보존" | **dr_grpo · scale none · ε_high 0.28 · β 0 · min_wm_spread 0.05 · 384tok** | 전부 실측 고정값 |
| FAA 입력 | 프레임 1장 | **배터리 ⑤ 결과로 클립 여부 확정 후 freeze** | 관측 비대칭, freeze 후 불가역 |
| 평가 지표 | GT acc·reranking | **+ G2 · wm_follow · belief-swap · belief_action_link** | 학습 실효·인과성 판별 |
| belief pair | teacher vs FAA 원문 | **스타일 정규화 + (FAA=GT) 서브셋 분리 로깅** | C4·C5 교란 |
| judge | 미지정 | **gemini-2.5-pro 강제** | 자기선호 편향 배제 |
| 사전 게이트 | 없음 | **배터리 ①~⑤를 §32 acceptance에 편입** | 학습 불필요, 반나절 |

### 사전 검증 배터리 (재학습·freeze 전, 전부 학습 불필요)

① `--no_memory` 평가 (히스토리 실제 기여량) ② history-only 베이스라인 (히스토리 단독 예측력)
③ belief-swap 개입 테스트 (belief→action 인과성 baseline) ④ judge `belief_action_link` 추가
(gemini-2.5-pro) ⑤ 클립 8프레임 입력 base 평가 (관측 비대칭 상한 상승분).
**③④는 HP-DPA 성공 판정의 사전값 — 결과 수치를 함께 전달하겠음.**

## 5. 권장 실행 순서

```
① 사전 검증 배터리 (학습 0회, 반나절)
② 패치 반영 FAA 재학습 (gen 8 · T 1.0 · 2,500~5,000 step) → 체크포인트 선정 → freeze
③ 스타일 정규화 포함 pair 구축 + leak 프로브
④ HP-DPA 학습 · ablation (belief-only / action-only / full / no-projection)
⑤ FAA vs HP-DPA — G2·인과성 포함 동일 지표 비교
```

## 6. 참고 — F0 쪽에서 제공할 것

- F0 체크포인트 + immutable snapshot 항목 (§10 목록 그대로 지지)
- `<reasoning>/<task_belief>/<action>` 트레이스 포맷 · `completion_log_every` 롤아웃 로그
- 수정된 memory 생성 규칙 (`< trigger_frame`)
- 배터리 ③④ belief 인과성 baseline 수치
- Ego4D 이식(Phase 0~2)은 F0 측이 병행 — belief 위축이 Ego4D에서 재현되면
  HP-DPA 필요성의 데이터셋-일반성 증거가 됨

명칭 노트: FAA/HP-DPA는 기법 약칭으로 유지하되, 논문 stage 명칭(Ego/Self 등 심리학적 명칭)은
별도 논의 중 — 확정 시 병기 형식 제안 예정.
