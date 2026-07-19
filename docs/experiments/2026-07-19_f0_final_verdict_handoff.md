# F0(extro) 재편 최종 결과와 확정 판정 — 핸드오프

작성: 2026-07-19 23:50 KST · 파이프라인 종료: 20:18 KST (총 3시간 57분, 무인 완주)
목적: **extro(F0 트랙)를 현재 구성으로 확정해도 되는가**에 대한 데이터 기반 판정.
수치 출처: 전부 eval_battery@384 (heldout 1f, n=500) + eval_belief_swap + gemini judge(25-step 간격 20포인트/arm). `runs/f0_battery/F0_CLEAN_RESULTS.md`, `F0_SPAN_RESULTS.md` 원본 보존.

---

## 0. 한눈 요약 (결론 먼저)

**권고: F0-W(wm_clean) 구성으로 extro 를 확정하는 것이 타당하다. 단, 확정의 의미를 "정확도를 올린 방법"이 아니라 "GT 없이 GT-지도와 구별되지 않는 성능에 도달하는 GT-free 방법"으로 자리매김해야 한다.**

세 가지 사실이 근거다:

1. **신호를 무엇으로 바꿔도 acc 는 움직이지 않았다.** 오염 제거(F0-W 0.240), GT 직접 보상(F0-G 0.244), action-span credit(F0-WA 0.244) 전부 base(0.242) 대비 ±0.008 이내 — n=500 의 통계 노이즈(±0.037) 안. **완벽한 신호(GT)조차 못 올린다는 것은 병목이 리워드 신호가 아니라 학습 역학(용량/최적화)이라는 뜻**이고, 따라서 extro 의 리워드 설계를 더 고치는 것은 소득이 없다. 리워드 측 개선 여지는 이번 실행으로 **체계적으로 소진**되었다 (P1-1 제거 → 불변, P1-6 적용 → 불변, GT 교체 → 불변).
2. **F0-W 만이 reasoning 품질을 유일하게 개선했다** (gemini judge 12.12→12.57, +0.45; F0-G −0.18, F0-WA −0.23). acc 동률이라면 부수 지표(judge·G2 0.333·wm_follow)가 가장 좋은 F0-W 가 확정 구성이다.
3. **③ 인과(belief→action)는 어떤 F0 변형으로도 복구되지 않았다** (아래 §3). 이것은 extro 의 한계가 아니라 **intro(B0-R1) 트랙이 담당하기로 이미 분리된 문제**다 — F0 확정을 막을 사유가 아니다.

---

## 1. 실험 구도 (무엇을 왜 돌렸나)

| arm | 리워드 | 검증하려던 가설 | 결과 |
|---|---|---|---|
| F0-N (base) | — (학습 없음) | 기준선 | acc 0.242 |
| F0-L (legacy) | WM + think_convergence 등 복합 | (구 구성) | acc 0.230 |
| **F0-W** | validity floor + **WM likelihood 단독** | "P1-1 오염(사후 언급 보상)을 빼면 WM 신호가 제대로 작동한다" | acc 불변, **judge/G2 개선** |
| **F0-G** | validity floor + **GT 바이너리** (oracle-subset, coverage 92.4%) | "신호가 약해서 못 배운다" (skyline) | acc 불변 → **가설 기각** |
| **F0-WA** | F0-W + **advantage 를 action span 에만** + ref-KL β0.04 | "credit 이 384토큰에 희석되어 못 배운다" (P1-6) | acc 불변 → **가설 기각** |

모든 arm 은 리워드 구성 외 전 조건 동일 (dr_grpo, gen 8, T 1.0, 384tok, LoRA r16, 5000 샘플, 500 step) — 차이는 신호 원천으로 귀속된다.

## 2. 정확도·품질 결과

| 모델 (step500) | acc | G2 (chance 0.2) | cond.acc (GT∈top5) | wm_follow | judge 추이 (전반→후반) |
|---|---|---|---|---|---|
| base | 0.242 | 0.309 | 0.390 | 0.328 | — |
| F0-L | 0.230 | 0.325 | — | 0.330 | — |
| **F0-W** | 0.240 | **0.333** | 0.387 | 0.344 | **12.12 → 12.57 (+0.45)** ↑ |
| F0-G | 0.244 | 0.301 | 0.394 | 0.334 | 12.55 → 12.37 (−0.18) ↓ |
| F0-WA | 0.244 (peak 0.250@250) | 0.325 | 0.394 | 0.346 | 12.35 → 12.12 (−0.23) ↓ |

- 참조 눈금: chance 0.200 / wm_top1 복사 0.374(전샘플 GT∈top1 기준) / oracle 상한 0.620.
- **acc 열은 전부 통계적으로 동률**이다 (§5). 읽어야 할 것은 "차이"가 아니라 "GT 를 줘도 차이가 안 난다"는 패턴.
- F0-WA 의 span-credit **구현 자체는 정상 작동**했다: `<action>` 태그 탐지 99.7%(3575/3584, BPE 폴백 경로), 학습 안정, parse 0.998 유지. 기각된 것은 구현이 아니라 "credit 희석이 병목"이라는 가설이다 — 깨끗한 negative result 로 기록할 가치가 있다.

## 3. ③ belief→action 인과 (Phase 2b 의 목적 지표)

| 모델 | causal_sensitivity | swap / control |
|---|---|---|
| base | 0.016 | 0.022 / 0.006 |
| F0-L | 0.008 | 0.012 / 0.004 |
| F0-W | 0.008 | 0.012 / 0.004 |
| F0-WA | 0.010 | 0.016 / 0.006 |

belief 를 통째로 다른 샘플 것으로 바꿔도 action 이 바뀌는 비율이 노이즈 플로어(control) 수준 — **어떤 학습 변형도 belief 를 인과적으로 만들지 못했고, 학습된 모델은 오히려 base 보다 둔감**하다. 사건 수 자체가 500 중 6~11건이라 세부 순위는 무의미하며, 결론은 "전부 0 수준"이 정확한 독해다. 이 문제의 구조적 레버는 F0 리워드가 아니라 **B0-R1 hard action gate**(teacher 가 GT 를 못 본 채 belief 로부터 action 에 도달한 trace 만 chosen)이며, 사전 등록 기준(③ > 0.03)과 함께 B0 담당자에게 이관되어 있다.

## 4. "extro 확정"이 의미하는 것 — 주장 가능/불가 경계

**확정 구성**: `reward_mode wm_clean` = validity floor(−0.5 constraint) + WM likelihood(candidate 정규화) 단독, joint top-5 셔플·score 은닉, min_wm_spread 0.05, dr_grpo/scale none, LoRA r16. (명명 규율: 레거시 run 은 "F0-L composite WM-GRPO" — "clean WM-only"라 부르지 않는다.)

**주장할 수 있는 것:**
- GT-free WM-분포 신호만으로 학습해 **GT 직접 지도와 구별되지 않는 성능**에 도달한다 (F0-W 0.240 vs F0-G 0.244, 동일 조건). "WM = 분포 인터페이스" 서사와 정합 — 신호의 상한이 아니라 모델/최적화의 상한에 먼저 닿았다.
- 오염 제거는 **reasoning 품질을 실질 개선**한다 (judge +0.45 는 5 arm 중 유일한 상승 추세, belief_action_link 도 +0.067).
- G2(WM top-1 을 이겨야 하는 부분집합) 0.333 으로 base 0.309 대비 방향성 개선 (단, n=123 이라 확정적이지 않음).

**주장할 수 없는 것 (문서·발표에서 금지):**
- "acc 를 개선했다" — 어떤 변형도 base 를 유의하게 넘지 못했다.
- "belief 가 action 을 조향한다" — ③ 이 전 구간 0 수준.
- "WM ≈ GT 동등성 증명" — n=500 에서 ±3.7%p 미만 차이는 검출 불가이므로, 정확히는 "**차이가 검출되지 않았다**(비열등 방향의 증거)". 강한 동등성 주장에는 더 큰 heldout 이 필요하다.

## 5. 통계적 주의 (이 표를 인용할 때)

acc n=500 → 95% CI ≈ ±0.037. 이번 실행의 모든 acc 차이(최대 0.014)는 CI 안이다. G2 n≈123 → CI ≈ ±0.082. judge 는 step 당 3샘플×20포인트의 평균 추세로, 방향성 지표이지 검정된 값이 아니다. **따라서 이 문서의 판정은 "무엇이 올랐다"가 아니라 "무엇을 바꿔도 오르지 않았다"는 반복된 null 패턴에 근거한다** — null 의 반복은 CI 와 무관하게 정보량이 크다 (4개 독립 변형이 같은 자리).

## 6. 남은 선택지와 권고 순서

| 선택지 | 목적 | 비용 | 권고 |
|---|---|---|---|
| **extro 확정 + 결과 문서화** | Phase 1/2b 를 논문 서사(분포 인터페이스 + 클린 방법론)로 고정 | 0 | **즉시** |
| B0-R1 리팩터 (intro) | ③ 인과 확보 — 유일하게 남은 구조 레버 | B0 담당자 이관 완료 | 병행 진행 |
| F0-GA 진단 (action-only 출력 + GT) | "full-trace 생성 자체가 병목" 가설 분리 확인 | ~2h | 선택 — 확정에 필수 아님, 용량 서사 보강용 |
| 용량/최적화 스윕 (LoRA rank↑·full FT·step↑) | acc 상한 자체를 미는 시도 | 수 시간~일 | 후순위 — GT-flat 이 상한이 낮음을 시사, 기대값 낮음 |

`NEEDS_DECISION_F0GA` 마커는 열린 상태로 유지 중 — F0-GA 실행 여부만 결정해 주면 된다.

## 7. 산출물 위치

- 요약: `runs/f0_battery/F0_CLEAN_RESULTS.md` · `F0_SPAN_RESULTS.md`
- 평가 원본: `f0{w,g,wa}_step{125,250,375,500}.json`(+`.records.jsonl`), `swap_{base_1f,step500_1f,f0w,f0wa}.json`
- 학습 로그: `outputs/step2/f0_{clean_wm,gt_only,wa_spancredit}_1f/` — `reward_log.jsonl`(함수별 분해), `judge_curve.jsonl`(20포인트), `completion_samples.jsonl`(25-step trace 원문), `group_stats.jsonl`/`oracle_manifest.json`(F0-G)
- 구현 상세(리뷰용): `2026-07-19_extro_intro_implementation_review_handoff.md`
- B0-R1 계획: `2026-07-19_b0_teacher_refactor_handoff.md`
