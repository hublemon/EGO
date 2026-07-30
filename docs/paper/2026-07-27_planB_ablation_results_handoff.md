# Plan B + ablation 결과 handoff — 실행 완료 보고

- 작성일: 2026-07-27 (KST 10:00)
- 실행 대상: `docs/paper/2026-07-27_aaai_reviewer_defense_plan_v2_handoff.md` 순서 1·2·7·8
- 실행 시간: 2026-07-26 17:12 UTC → 2026-07-27 00:5x UTC (약 7시간 40분, 단일 H200)
- 산출물: `runs/cesft_v2_fp_c/eval/`, `runs/cesft_v2_fp_r00/eval/`, `runs/cesft_v2_fp_k/eval/`
- **모든 수치는 실측이다.** 추정치는 이 문서에 없다.

---

## 0. 세 줄 결론

1. **belief 주장이 확정됐다.** 교란 통제 프로토콜에서 `sft_r15_c`의 belief 감수성이 base보다
   **+0.18 [+0.127, +0.233]**, θ_CE보다 **+0.137 [+0.083, +0.190]** 높다. G_CC2 **PASS**.
   문체 교란은 +3.3pp로 전체 효과의 13%에 불과했다 — 나머지 87%는 실재한다.
2. **`main.tex`가 약속한 ablation 두 개가 모두 채워졌고, 둘 다 본문 주장을 지지한다.**
   ρ=0 대조군은 replay anchor가 실제로 prospective 능력을 지킨다는 것을 보였고(SelAcc +2.3pp,
   G₁ +7.0pp), K ablation은 boundary 크기가 판단에 미치는 영향을 정량화했다.
3. **사전등록 규칙이 발동했다. 제목의 `embodied`를 내려야 한다.** no-image 조건에서도
   Cand.-CE의 GT-only 대비 우위가 **+5.48pp [+2.03, +9.57]** 로 유지된다(원조건 +7.70pp의 71%).
   v2 §5의 "정직한 사전 선언"에서 예상한 그대로이며, **rebuttal이 아니라 본문에서 먼저 내린다.**

---

## 1. Plan B — belief 주장 확정 (`harden_paired`, n=300 공통셋, donor=base)

### 1.1 arm별 결과

| arm | `swap_b`<br>(arm 내부 문체) | `swap_b_shared`<br>(전 arm 동일 문자열) | 문체 교란 Δ | `U_b`<br>(belief 단독 효용) | `D_g` | `acc_own` |
|---|---|---|---|---|---|---|
| base | 0.0333 [.010,.060] | 0.0333 [.010,.060] | +0.0000¹ | +0.0090 [−.004,+.022] | 0.2333 | 0.2267 |
| θ_CE | 0.0867 [.057,.120] | 0.0767 [.047,.107] | +0.0100 | +0.0309 [+.014,+.048] | 0.3400 | 0.3367 |
| `sft_r15` | 0.0733 [.047,.107] | 0.0733 [.047,.107] | +0.0000 | +0.0232 [+.011,+.035] | 0.3667 | 0.2933 |
| **`sft_r15_c`** | **0.2467 [.190,.300]** | **0.2133 [.160,.260]** | **+0.0334** | **+0.0460 [+.030,+.062]** | **0.4933** | 0.3233 |

¹ base는 donor arm이므로 두 조건의 주입 문자열이 정의상 동일하다. 교란 Δ는 **비-donor arm에서만**
의미가 있다.

`U_b = p(a_GT | r_own, b_own) − p(a_GT | r_own, b_swap)` — 공격문서 §6이 \(U_g\)라고 부른 지표가
이 도구의 `U_b`다. 이 도구의 `U_g`는 `own − swap_both`(전체 trace)라 다른 양이므로 혼동하지 말 것.

### 1.2 문체 교란의 크기 — Plan B를 돌린 이유

`harden_s3`의 알려진 약점은 swap partner가 arm 내부에서 온다는 것이었다. arm마다 belief 문체가
다르므로 주입되는 정보량 자체가 달라진다. 전 arm 동일 donor 문자열로 통제한 결과:

| arm | 교란 Δ | 전체 효과 대비 |
|---|---|---|
| θ_CE | +0.0100 | 11.5% |
| `sft_r15` | +0.0000 | 0% |
| **`sft_r15_c`** | **+0.0334** | **13.5%** |

C의 교란이 가장 크지만(belief 어휘가 927→1,143으로 가장 다양하므로 예상된 방향),
**전체 효과의 13.5%에 그친다.** 통제 후에도 0.2133 [0.160, 0.260]이 남는다.
사전에 어휘 거리 층화로 논증했던 "정보량 교란이 아니다"가 정식 통제에서 확인됐다.

### 1.3 arm 간 paired 차이 — `harden_s3`로는 불가능했던 것

같은 300건, 같은 순서. `sft_r15_c` 기준:

| 비교 | belief 감수성 | **통제 후(shared)** | `U_b` | `acc_own` |
|---|---|---|---|---|
| C − base | +0.2133 [+.157,+.270] | **+0.1800 [+.127,+.233]** | +0.0369 [+.021,+.054] | +0.0967 [+.043,+.153] |
| C − θ_CE | +0.1600 [+.100,+.217] | **+0.1367 [+.083,+.190]** | +0.0151 [−.007,+.036] | −0.0133 [−.073,+.047] |
| C − `sft_r15` | +0.1733 [+.113,+.233] | **+0.1400 [+.087,+.193]** | +0.0228 [+.006,+.039] | +0.0300 [−.030,+.090] |

**세 비교 모두 통제 후에도 CI가 0을 제외한다.** 이제 논문에서
*"significantly higher than"* 을 쓸 수 있다 — 이것이 Plan B의 가장 큰 소득이다.

### 1.4 G_CC2 정식 판정

```
정의   : 학습 arm 의 belief sensitivity 가 동일 셋 base 보다 유의하게 높은가
delta         = +0.2133 [+0.1567, +0.2700]   verdict = PASS
delta_shared  = +0.1800 [+0.1267, +0.2333]   verdict = PASS
```

### 1.5 인과 채널 이동이 통제 프로토콜에서도 확인된다

| arm | belief | reasoning | both | `flip_para` |
|---|---|---|---|---|
| base | 0.0333 | 0.2567 | 0.7767 | 0.0133 |
| θ_CE | 0.0867 | 0.2067 | 0.7433 | 0.0033 |
| `sft_r15` | 0.0733 | **0.4700** | 0.7833 | 0.0033 |
| `sft_r15_c` | **0.2467** | 0.3333 | 0.6967 | **0.0500** |

C − `sft_r15`의 reasoning 감수성 차이는 **−0.1367 [−0.210, −0.060]** 로 유의하게 낮다.
**fp에서 reasoning으로 쏠려 있던 인과가 belief로 이동했다**는 진단이 두 프로토콜에서
독립적으로 확인됐다.

### 1.6 정직하게 함께 보고할 한계

- **`flip_para`가 C에서 0.0500으로 가장 높다**(다른 arm 0.003~0.013). C는 belief 문구 변화 자체에
  더 민감하다. 감수성 계산에서 이미 차감되지만, "표면 민감도도 함께 올랐다"는 사실은 써야 한다.
- **`U_b`의 C − θ_CE는 +0.0151 [−0.007, +0.036]로 0을 포함한다.** 즉 belief가 **정답을 지지하는
  정도**는 θ_CE보다 유의하게 높지 않다. 유의하게 높은 것은 belief가 **행동을 바꾸는 정도**다.
  `decision-relevant`를 주장할 때 이 구분을 지켜야 한다.
- **`acc_own`의 C − θ_CE는 −0.0133 [−0.073, +0.047]** 로 차이 없음. C가 정확도를 잃지 않았다는
  뜻이라 유리하지만, "더 정확하다"고 쓰면 안 된다.
- `harden_paired`와 `harden_s3`는 **절대값 비교 불가**다(프레임 포함 채점 + assistant 헤더 중복
  수정). 표를 반드시 분리할 것.

---

## 2. ρ ablation — `main.tex` L205 약속 이행

본문: *"the effect of the anchor can be isolated by an ablation with ρ=0 in the experiments."*
ρ=0 arm을 신규 학습했다(294 스텝). 세 arm 모두 **같은 `chosen_train.jsonl`을 공유**하므로
단일변수가 보장된다.

| ρ | SelAcc | G₁ 유지 | G₂ (GADR) | malformed | belief 인과² | G-ACC1 |
|---|---|---|---|---|---|---|
| **0** | 0.2620 | 0.3471 | 0.2348 | **0.009** | **0.4825 [.428,.533]** | +2.12pp [−2.69,+6.68] |
| **0.15** | **0.2850** | **0.4174** | **0.2427** | 0.028 | 0.3825 [.333,.432] | **+5.35pp [−0.18,+10.60]** |
| **0.30** | 0.2760 | 0.4050 | 0.2348 | 0.045 | 0.3700 [.320,.417] | +4.40pp [−0.64,+9.08] |
| *(θ_CE 참조)* | *0.2880* | *0.3554* | *0.2665* | *0.043* | *0.0925* | *+6.06pp [+0.87,+11.22]* |

² `harden_s3` 프로토콜. Plan B(`harden_paired`)와 절대값 비교 불가.

**본문 주장이 실측으로 지지된다.** L205는 "sequential fine-tuning이 1단계에서 얻은 candidate
discrimination을 덮어쓸 수 있으므로 anchor를 넣는다"고 했다. 앵커를 빼면(ρ=0):

- SelAcc **−2.3pp** (0.2850 → 0.2620)
- G₁ 유지 **−7.0pp** (0.4174 → 0.3471) ← 덮어쓰기가 정확히 여기서 일어난다
- G-ACC1 여유 **−3.2pp** (+5.35 → +2.12)

**동시에 정직하게 보고할 대가:** ρ=0의 belief 인과가 **0.4825로 가장 높다.**
앵커는 prospective 능력을 지키는 대신 retrospective belief 인과를 일부 희생한다.
ρ=0.15가 두 축의 균형점이고, ρ=0.30은 어느 축도 개선하지 못한다(앵커를 2.4배로 늘려도 무이득).

> **곡선의 형태**: 정확도는 ρ=0.15에서 정점을 갖는 역U자, belief 인과는 ρ에 대해 단조 감소.
> "최적점이 있는 곡선"이라 단조 곡선보다 설명력이 크다.

---

## 3. K ablation — `main.tex` L289 / `tab:kablation` 약속 이행

본문: *"This ablation directly validates one of this work's core claims---that the boundary
materially affects judgment."*

| arm | K | Coverage@K | 조건부 SelAcc | WM Top-1 (L0) | **SelAcc − L0** | full-set 환산 | malformed |
|---|---|---|---|---|---|---|---|
| θ_CE | 10 | 43.43% | 0.2880 | 0.2420 | +4.60pp | 0.1251 | 0.043 |
| θ_CE | 5 | 30.34% | 0.3820 | 0.3420 | +4.00pp | 0.1159 | 0.042 |
| θ_CE | 3 | 21.97% | 0.5080 | 0.4590 | +4.90pp | 0.1116 | 0.056 |
| `sft_r15_c` | 10 | 43.43% | 0.2850 | 0.2420 | +4.30pp | 0.1238 | 0.028 |
| `sft_r15_c` | 5 | 30.34% | 0.4020 | 0.3420 | **+6.00pp** | 0.1220 | 0.026 |
| `sft_r15_c` | 3 | 21.97% | 0.5390 | 0.4590 | **+8.00pp** | 0.1184 | 0.018 |

### 3.1 세 가지를 동시에 보아야 한다

1. **조건부 SelAcc는 K가 줄면 급격히 오른다** (0.285 → 0.539). 그러나 **WM 바닥(L0)도 같이
   오른다** (0.242 → 0.459). 조건부 수치만 보면 오독이다.
2. **full-set 환산은 거의 평평하다** (C: 0.1238 → 0.1220 → 0.1184). 경계를 좁히면 변별은 쉬워지고
   커버리지는 줄어드는데, 둘이 **거의 상쇄된다.** K=10이 미세하게 최선이다.
3. **바닥 대비 마진이 arm마다 다르게 반응한다.** θ_CE는 K와 무관하게 +4~5pp로 평평한 반면,
   **`sft_r15_c`는 K가 줄수록 마진이 커진다(+4.3 → +6.0 → +8.0).** 회고 학습 arm이 좁은 경계를
   더 잘 활용한다. malformed도 C만 K와 함께 줄어든다(0.028 → 0.018).

### 3.2 사전등록 규칙에 대한 판정 — 주의가 필요하다

v2 §5의 규칙: *"K에 따라 SelAcc가 단조이거나 무반응이면 → 'boundary materially affects
judgment' 하향."*

**문자 그대로 읽으면 조건부 SelAcc는 단조이므로 하향 대상이다.** 그러나 그 단조성은 커버리지
감소가 만드는 기계적 산물이며, 규칙을 쓸 때 이 교란을 예상하지 못한 것이 규칙의 결함이다.
실질적인 판정은 다음과 같이 쓰는 것이 정직하다.

> 경계 크기는 **조건부 변별 난이도를 크게 바꾸지만**(SelAcc 0.285→0.539, L0 0.242→0.459),
> **end-to-end 성능은 거의 바꾸지 않는다**(full-set 0.1238→0.1184). 다만 회고 학습 arm은
> 경계가 좁아질수록 바닥 대비 마진이 커져(+4.3→+8.0pp), 경계 정보를 실제로 활용함을 보인다.

`main.tex` L289의 "materially affects judgment"는 이 결과에 맞게 **"affects the difficulty of
the selection problem, and the retrospective policy exploits a tighter boundary better"**
수준으로 조정할 것을 권한다.

---

## 4. Tier 1 교란 — 제목 판정

4 arm × 3 조건 × n=1000 covered. Δacc = 원조건 − 교란조건 (양수면 그 입력을 사용한다는 뜻).
후보 집합·`wm_scores`·WM Top-1은 **모든 조건에서 불변** — 정책 경로에만 개입했다.

| arm | no-image (C3) | no-hist (C2)³ | no-image ∧ no-hist (C4) | **other-video hist (C6)** |
|---|---|---|---|---|
| base | +6.30 [3.2, 9.3] | +2.90 [−0.1, 5.7] | +14.10 [11.0, 17.0] | **+1.10 [−1.7, 3.8]** |
| cand_free (GT-only) | +6.30 [3.3, 9.3] | +4.50 [1.5, 7.3] | +10.70 [7.7, 13.5] | **+2.20 [−0.6, 5.0]** |
| θ_CE (Cand.-CE) | +8.30 [5.3, 11.3] | +10.10 [6.8, 13.3] | +20.50 [17.2, 23.7] | **+10.00 [6.8, 13.2]** |
| `sft_r15_c` | +7.20 [3.9, 10.7] | +9.80 [6.7, 12.9] | +18.50 [15.1, 21.9] | **+10.60 [7.4, 13.7]** |

³ 기존 `strip_eval` 산출물 재사용(재실행 없음).

### 4.1 A4 / Q2가 결정적으로 닫혔다 — 이번 실행 최대 성과

공격문서 §5의 핵심은 *"history 제거는 OOD intervention이라, 큰 하락이 (1) trajectory reasoning을
배웠기 때문인지 (2) history 누락에 더 취약해졌기 때문인지 구분되지 않는다"* 였다.

**other-video history는 이 둘을 분리한다.** 형식은 완전히 정상이고(같은 길이, 같은 포맷,
video-disjoint 도너) **의미만 틀렸다.**

- **base +1.10 [−1.7, 3.8], GT-only +2.20 [−0.6, 5.0] — 둘 다 귀무.**
  형식이 유지되면 학습 안 된 모델은 아무 영향을 받지 않는다 → **포맷 shift 설명 완전 배제.**
- **θ_CE +10.00, `sft_r15_c` +10.60 — 둘 다 유의.**
  더욱이 이 값이 **no-history 손실(+10.10 / +9.80)과 사실상 같다.**
  즉 **틀린 history는 없는 history와 똑같이 나쁘다** → 학습된 arm은 history의 **의미를 읽는다.**

이 대비는 소제목 `Causal Test`를 유지할 근거가 된다(v1·v2가 `Inference-time History Ablation`
으로 하향을 권했으나, 그 권고는 semantic corruption 통제가 없다는 전제였다).

### 4.2 2×2 교호작용 — 두 입력은 독립적이지 않다

| arm | no-image + no-hist 합 | 실제 C4 | 초가산성 |
|---|---|---|---|
| base | 9.2 | 14.1 | +4.9 |
| cand_free | 10.8 | 10.7 | −0.1 |
| θ_CE | 18.4 | 20.5 | +2.1 |
| `sft_r15_c` | 17.0 | 18.5 | +1.5 |

GT-only만 정확히 가산적이고, 나머지는 초가산적이다. 두 입력이 서로를 부분적으로 보완한다는 뜻.

### 4.3 사전등록 규칙 발동 — `embodied` 하향

v2 §5: *"no-image에서 Cand.-CE−GT-only 우위 CI가 0을 제외하고 유지 → candidate-aligned,
trajectory-conditioned selection으로 강등."*

| 조건 | θ_CE − GT-only | 판정 |
|---|---|---|
| 원조건 | **+7.70pp [+3.73, +11.93]** | 유의 |
| **no-image** | **+5.48pp [+2.03, +9.57]** | **유의 — 우위 71% 유지** |
| no-image, C − GT-only | +6.64pp [+3.04, +10.46] | 유의 |

**규칙이 발동한다. 제목·abstract의 `embodied`를 내린다.**

동시에 정직하게 병기할 것: **이미지는 분명히 기여한다.** 전 arm이 no-image에서 6.3~8.3pp를
잃고, θ_CE의 손실(8.3pp)이 base(6.3pp)보다 크다. 정확한 서술은 다음이다.

> Vision contributes to absolute accuracy for every arm, **but the advantage that candidate-aware
> training confers is not vision-dependent** — 71% of it survives with the frames removed.

---

## 5. 리뷰어 공격 상태 갱신

| 공격 | v2 시점 | 지금 | 근거 |
|---|---|---|---|
| A1 embodied vs script completion | 미식별 | **규칙대로 하향 확정** | §4.3 |
| A2 28.8% headline | 보고 규율 문제 | 변동 없음 (Tier 0 잔여) | — |
| A3 final EGO 개선 없음 | 검정력 문제로 재해석 | 변동 없음 + ρ 곡선이 anchor 기여 입증 | §2 |
| **A4 history 인과 미확립** | 미해결 | **해결** | §4.1 |
| **A5 belief 미입증** | 뒤집힘 | **강점으로 확정** | §1 |
| B3 capability axis | 미대응 | 변동 없음 (Tier 0 잔여) | — |
| Q1 이미지 없어도 향상? | — | **답변 가능**: 71% 유지 | §4.3 |
| Q2 다른 비디오 history? | — | **답변 가능**: −10.0pp, base는 귀무 | §4.1 |
| Q10 sensitivity 유의? | — | **답변 가능**: G_CC2 PASS | §1.4 |
| Q11 GT 확률 방향? | — | **답변 가능**: U_b > 0 (단 θ_CE 대비는 무의) | §1.1·1.6 |

---

## 6. 쓸 수 있는 문장 / 못 쓰는 문장

**쓸 수 있다**

> 회고 학습 arm의 task belief는 base보다 **+0.18 [0.127, 0.233]**, prospection 체크포인트보다
> **+0.137 [0.083, 0.190]** 더 강하게 행동 선택을 좌우한다(전 arm 공통 300건, 동일 donor 문자열,
> 프레임 포함 채점).

> 다른 비디오의 history를 주입하면 학습된 정책은 **−10.0pp / −10.6pp** 를 잃는 반면 base와
> GT-only는 통계적으로 영향을 받지 않는다(**+1.1 / +2.2**, CI가 0 포함). 형식은 보존되고 의미만
> 파괴되므로, 손실은 입력 형식 변화가 아니라 history의 의미 사용에서 온다.

> Replay anchor를 제거하면(ρ=0) SelAcc가 **−2.3pp**, G₁ 유지가 **−7.0pp** 하락한다.

**못 쓴다**

> ~~belief가 정답을 지지하는 정도가 θ_CE보다 높다~~ — `U_b` 차이 +0.0151 [−0.007, +0.036]로 무의.
> ~~embodied visual reasoning~~ — 사전등록 규칙 발동. §4.3.
> ~~최종 arm이 θ_CE보다 정확하다~~ — `acc_own` −0.0133 [−0.073, +0.047].
> ~~boundary가 판단을 좌우한다(현재 표현)~~ — full-set 환산이 거의 평평. §3.2대로 조정.

---

## 7. 남은 일 (GPU 불필요)

| # | 작업 | 닫는 공격 |
|---|---|---|
| 1 | 제목·abstract에서 `embodied` 하향, `visually grounded boundary` → `observation- and history-conditioned proposal boundary` | A1, Q3 |
| 2 | `tab:kablation` 채우고 L289 문구를 §3.2대로 조정 | 본문 약속 |
| 3 | ρ 곡선 표 추가, L205 서술을 실측에 맞춤 | 본문 약속 |
| 4 | G-NH 검정력 서술 추가 (마진 −1pp가 설계 해상도 4.06pp보다 작다) | A3 |
| 5 | `harden_paired` / `harden_s3` 표 분리, 절대값 비교 금지 명시 | 무결성 |
| 6 | full-set 12.5% 병기, Figure 1(b) replay 26.4% 실측점 | A2, A3 |
| 7 | `tools/text_baselines.py` 구현 + 실행 | A1 잔여 |
| 8 | estimand 통일 (`paired_boot.py` 코드 수정 — 플래그 아님) | A2, Q5 |

---

## 8. 재현

```bash
REPO=/mnt/nvme/migration/jihun/EGO_jihun3; cd $REPO
PY=/mnt/nvme/migration/jihun/envs/miniforge3/envs/eve-cu124/bin/python
export PYTHONPATH=$REPO/src HF_HOME=/mnt/nvme/cache
export FRAME_CACHE_DIR=$REPO/runs/cesft_v2/frame_cache
export RETRO_NEXT_GAP_TEXT="after the current action ends"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

bash scripts/step2_retrospection/run_planB.sh          # Plan B 4-arm (3h 35m 실측)
bash scripts/step2_retrospection/run_ablations_v2.sh   # ρ=0 · K · Tier 1 (4h 실측)
$PY tools/ablation_progress.py                          # 진행률·실측 소요
```

신규/수정 코드:
- `tools/oom_opt/perturb_eval.py` (신규) — image/history 교란. noimage 계열은 decord 미진입.
- `src/ego/step2_retrospection/eval/battery.py` — `--top_k` 추가 (K ablation). 기본 10 = 기존 동작.
- `tools/ablation_progress.py` (신규) — 실측 기반 진행률/잔여시간.
- `scripts/step2_retrospection/run_planB.sh`, `run_ablations_v2.sh`, `ablation_progress_refresher.sh`

실측 소요: Plan B arm당 **53.4분**(plan 0.9분 + 4 arm + agg), ρ=0 학습 **17분**,
K ablation 셀당 **9분**, no-image 셀당 **6.5분**(프레임 추출 없음), other-video 셀당 **10분**.
