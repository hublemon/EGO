# WM 경계 내재화 — 9h 진입 전 precheck 3종 결과 & 방향 결정 Handoff

- 작성: 2026-07-24 KST · EGO_jihun3
- 대상 계획: `2026-07-24_boundary_internalization_9h_plan_handoff.md` + review handoff
- 결정: **9h 원안 진입 보류.** precheck가 "신호는 실재하나 base가 이미 대부분 갖고 있고,
  candidate-free LM 랭킹은 WM보다 약하다"를 보여줌. 아래 두 갈래 중 택일 권고.
- 도구: `tools/precheck_wm_set_jaccard.py` · `tools/precheck_mk_candfree.py` · `tools/precheck_fusion.py`
- 산출 JSON: `runs/retro4/eval/precheck_*.json`

---

## 0. 세 줄 요약

1. **(d) Jaccard 🟢**: WM 집합은 instance-specific(92% 고유, GT 커버리지 prior 21%→WM 44%, +23pp).
   L3(P_true vs P_shuffle) 검정력의 *재료*는 데이터에 실재.
2. **(c) M_K 🟡**: base가 후보 없이도 자기 집합 vs 타 집합을 **0.43** 구별(0=못함, 0.96=presented 상한).
   base가 이미 경계를 상당히 앎. 게다가 학습(r1_sft)이 이 구별력을 **못 올림**(0.43→0.40).
3. **(F) Fusion 🔴(단, vision-blind 단서)**: candidate-free LM 랭킹 acc|cov **0.123** ≪ WM top-1 **0.246**.
   융합 이득 없음(WM 지배). → **VLM이 candidate-free 행동랭킹에서 WM에 못 미침.**

핵심 재구성: "WM 경계를 VLM에 내재화할 수 있나"를 묻기 전에, **"VLM이 이 태스크에서 WM보다
나은가"**부터 서야 하는데 — acc 기준으로는 (적어도 history-only 랭킹에서) 아니다.

---

## 1. Precheck (d) — WM 집합 구별성 (Jaccard)

`support_val.jsonl` 6,960샘플, K=10, action 단위.

| 지표 | 값 | 의미 |
|---|---|---|
| 고유 WM집합 | 6403/6960 (92%) | 입력마다 집합이 거의 다 다름 |
| cross-example Jaccard (mean) | 0.104 | within-video 0.308 / across-video 0.101 |
| 각 집합 ∩ 전역 top-10 | 0.201 | 집합의 80%는 빈출코어 아님 |
| GT ∈ 전역 prior | 0.207 | P_shuffle이 도달하는 GT 천장 |
| GT ∈ WM집합 (cov@10) | 0.439 | P_true 천장 |
| **차이 (instance-specific 정보)** | **+0.232** | shuffle이 못 배우는 부분 = L3 재료 |

판정: **GREEN.** 집합이 diverse하고 prior 너머 +23pp를 담아, P_true>P_shuffle 격차가
나올 재료가 있음. (cross≈null 착시는 빈도-랜덤도 겹침 낮아 생긴 것 — prior-overlap이 옳은 지표.)

## 2. Precheck (c) — candidate-free 집합 구별력 M_K

각 예시에서 own집합 고유원소 vs 무작위 타예시 집합 고유원소에 얹힌 질량(텍스트 전용, 이미지 없이).
`구별력 = own_mass − other_mass` (0=못 가림, presented가 상한).

| model | own_cf | other_cf | **구별력(cf)** | presented 상한 |
|---|---|---|---|---|
| base | 0.716 | 0.284 | **0.432** | 0.955 |
| r1_sft (trace-SFT) | 0.698 | 0.302 | **0.396** | 0.892 |

판정: **AMBER.** base가 이미 0.43(→"완전구별 0.96"에 가까움) → S2가 채울 헤드룸이 작음.
학습(r1_sft)이 오히려 소폭 하락 → **trace-SFT는 candidate-free set-preference를 못 올림.**
단서: 텍스트(history) 전용이라 이 0.43은 "history가 다음행동을 시사"에서 왔을 수 있음(vision 아님).

## 3. Precheck (F) — 학습 0 score-fusion

`F(a) = ℓ_LM(a|candidate-free) + α·log q_WM(a)`, WM top-K argmax. α는 dev(n=500) 보정 → heldout(n=1000).
covered heldout n=422.

| 구성 | acc\|cov |
|---|---|
| LM 단독 (α=0, candidate-free) | **0.123** |
| WM top-1 단독 (α=∞) | **0.246** |
| dev α곡선 | 0.078→…→0.242(α8)→0.255(∞) 단조↑ |
| (참조) retro3 base/r1_sft, candidate-**presented** | 0.223 / 0.234 |

판정: **RED, 단 vision-blind 단서.** dev에서 α가 클수록 좋아져 **순수 WM이 최적** — LM을 섞을수록
나빠짐. 즉 융합 이득 없음, **candidate-free LM 행동랭킹(0.123)이 WM(0.246)의 절반.**
**중대 단서**: 이 LM 스코어링은 Qwen3-VL 배치-RoPE 충돌 회피 위해 **텍스트 전용(이미지 미사용)**.
WM은 vision(V-JEPA feature)을 쓰므로 "vision-blind LM < WM"은 일부 당연. **공정한 F는 vision-grounded
스코어링 필요** — 그 엔지니어링(배치 수정 or B=1+image)은 어차피 S2 학습에도 필요.

---

## 4. 종합 진단

- **일관된 그림**: history(텍스트)만으로 **coarse 집합-수준 지식은 이미 있음**(M_K 0.43), 그러나
  **집합 내부 fine GT-랭킹은 약함**(fusion 0.123 < WM 0.246). 즉 "어느 집합인지는 알지만 그 안에서
  뭐가 정답인지는 못 고름." WM은 vision으로 그 fine 랭킹을 (약하게나마 0.246) 함.
- **S2(경계 내재화)는 fine 랭킹을 안 올림**(set-mass는 집합질량만) → S2 성공해도 acc는 여전히 낮을 것.
- **유일하게 입증된 양성 신호는 S3**: retro3 개입③ causal_sensitivity **0.073→0.387**(SFT가 reasoning의
  인과 기여를 5배). 이건 acc가 아니라 "reasoning이 행동을 조종하는가"의 신호 — 살아있는 자산.

## 5. 권고 — 두 갈래

**옵션 A (F를 공정하게 재검):** vision-grounded candidate-free 스코어링 구현(배치-RoPE 수정 or
B=1+image, ~1h) → F 재측정. vision-grounded LM+WM 융합이 WM 단독(0.246)을 넘으면 F가 baseline이자
방법 후보. 넘지 못하면 Step-2(VLM) 자체의 acc 기여가 없다는 깨끗한 음성결과.

**옵션 B (S3로 내러티브 전환, 권장):** acc 경쟁(S1/S2)을 접고, 입증된 S3(reasoning 인과성)를 headline로.
"WM이 집합을 정의하고 VLM이 그 위에서 **검증가능한 reasoning**을 생성한다 — 그 reasoning이 행동을
인과적으로 조종함(개입③)"이 유일하게 데이터가 지지하는 주장. acc는 WM/fusion에 위임.

제안: **A를 1h만 태워 F 공정판을 확정**하고(음성이든 양성이든 baseline 확정), 그 결과와 무관하게
**B를 논문 spine으로** 가는 하이브리드. S2 9h 원안은 헤드룸(M_K)·필요성(F) 모두 미검증이라 보류.

## 6. 재현

```bash
RETRO3_RUNS=runs/retro4 PYTHONPATH=src PY=.../eve-cu124/bin/python
$PY tools/precheck_wm_set_jaccard.py --runs runs/retro4 --level action
$PY tools/precheck_mk_candfree.py --n 300                       # base
$PY tools/precheck_mk_candfree.py --n 200 --adapter outputs/step2_retrospection/r1_sft/adapter
$PY tools/precheck_fusion.py --n_dev 500 --n_eval 1000          # base F
```

한계: M_K·F 모두 텍스트 전용(vision-blind) — 절대수치는 vision-grounded에서 달라질 수 있음.
Jaccard는 순수 데이터분석이라 무관. n(eval covered)=422로 ±2~3pp 노이즈.
