# interventional belief sensitivity — 지표 handoff (측정법·의미·함정)

- 작성: 2026-07-24 KST · EGO_jihun3
- 코드: `src/ego/step2_retrospection/eval/harden_s3.py` (개입 팩), `.../eval/intervention.py` (`candidate_probs`)
- 관련: [[2026-07-24_ce_sft_methodology_v2_handoff]] §4·§9, [[2026-07-24_cesft_v2_running_state_handoff]]
- 게이트: **G-CC1** (필수, belief sensitivity 실재)

---

## 0. 한 줄 정의

**belief의 *의미*를 바꿔치기(swap)했을 때 모델의 행동 선택이 얼마나 달라지는가** —
단, 의미를 보존한 재작성(paraphrase)이 유발하는 표면적 변화를 **빼서** 순수하게 "의미 변화 때문에 바뀐 정도"만 남긴 값.

```
belief_sensitivity = Pr[action flips | belief를 다른 belief로 swap]
                   − Pr[action flips | belief를 같은 뜻으로 paraphrase]
```
paired bootstrap 95% CI 하한 > 0 이면 "행동이 belief 내용에 **개입적으로 의존**한다"고 판정(G-CC1).

---

## 1. 왜 "interventional"이고 왜 이 지표가 필요한가

모델이 `<reasoning>…</reasoning><task_belief>…</task_belief><action>…</action>`를 순서대로 생성한다고 해서,
belief가 **실제로 action에 쓰였다는 보장은 없다**. 모델은 `context→belief`와 `context→action`을
**병렬로** 뽑고 belief는 장식일 수 있다.

→ 그래서 **관측이 아니라 개입(intervention)**으로 측정한다: belief를 강제로 다른 것으로 바꿔 넣고
action이 반응하는지 본다. 반응하면 "belief가 action 계산에 들어간다"는 **행동적 증거**.

⚠️ **용어 규칙(§9)**: 이건 "belief가 action을 **인과적으로 매개**한다"는 증명이 **아니다**.
"causal pathway/mediation/faithfulness" 표현 **금지**. 허용: *interventional belief dependence·
belief-conditioned action sensitivity·semantic intervention sensitivity*.

---

## 2. 측정 절차 (harden_s3.py 실제 구현)

대상: covered·well-formed 샘플 n=`IV_N`(기본 800). malformed·belief/reasoning 없는 건 제외.

각 샘플 `i`에 대해:

1. **donor 선정**: `partner = recs[(i+7) % n]` — 고정 오프셋 이웃에서 남의 belief `b_other`·reasoning `r_other`를 빌려온다(결정적).
2. **prefix 6종 구성** — (reasoning, belief) 쌍을 바꿔가며:

| variant | reasoning | belief | 무엇을 격리 |
|---|---|---|---|
| `own` | 자기 | 자기 | 기준(개입 없음) |
| `empty` | 자기 | **None** | belief 제거 |
| **`swap_b`** | 자기 | **남의 것** | **belief-only 개입** |
| `swap_r` | **남의 것** | 자기 | reasoning-only 개입 |
| `swap_both` | 남의 것 | 남의 것 | 전체 prefix 개입 |
| **`para`** | 자기 | **자기 belief를 paraphrase** | **통제군(의미 동일, 문체만)** |

3. **각 variant마다 행동 분포 계산**:
   `p = candidate_probs(model, processor, rec, reasoning, belief, device)` →
   K=10 후보에 대한 selection 확률분포. belief/reasoning은 이 분포를 만드는 **프롬프트 prefix**로 들어간다.
   - `top1[variant] = argmax(p)` — 그 prefix로 고른 행동
   - `P[variant] = p[gt_idx]` — GT 후보의 확률
4. **flip 기록**: `F[k] = int(top1[k] != top1["own"])` — 개입으로 top-1 행동이 바뀌었나(0/1).

집계:
```python
causal["belief"] = diff_ci(F["swap_b"], F["para"])   # ← interventional belief sensitivity
causal["reasoning"] = diff_ci(F["swap_r"], F["para"])
causal["both"]      = diff_ci(F["swap_both"], F["para"])
```
`diff_ci`: paired bootstrap(n_boot=2000)로 두 flip-rate 차이의 (point, lo, hi) 반환.

---

## 3. 핵심: 왜 paraphrase를 빼나

`swap_b`만 보면 flip이 두 원인의 합이다:
- (A) belief **의미가 달라져서** action이 바뀐 것 ← 우리가 원하는 신호
- (B) 그냥 **텍스트가 달라져서** 모델이 불안정하게 흔들린 것 ← 노이즈

`para`(같은 뜻·다른 문장)는 (A)=0, (B)만 있는 통제군. 따라서
**`swap_b − para` = 순수 (A)** = 의미 변화에 대한 반응. 이게 "interventional" 지표의 핵심 설계다.

---

## 4. 세 필드 분해의 의미

- **belief** (`swap_b−para`): belief 필드 단독의 행동 관여. **G-CC1의 대상.**
- **reasoning** (`swap_r−para`): reasoning 필드 단독. (para가 belief 통제라 reasoning엔 근사 통제)
- **both** (`swap_both−para`): 전체 prefix의 관여(상한). **G-S3a**의 대상.

훈련이 **belief만 선택적으로** 키우면(=belief↑, both는 비슷/미증) "행동이 특히 belief 내용에 반응"이라는 강한 주장.

---

## 5. 실측 앵커 (이전 세대 retro3, base vs 학습완료 r1_sft, n≈990)

| 필드 | 학습 전(base) | 학습 후 | 해석 |
|---|---|---|---|
| **belief** | 0.058 (CI .043–.075) | **0.390 (CI .358–.421)** | 6.7×↑ — G-CC1 강하게 통과 |
| reasoning | 0.220 | 0.295 | 소폭↑ |
| both(prefix) | 0.820 | 0.776 | 거의 불변(약간↓) |

→ 학습이 **belief 민감도만 크게** 끌어올림. base는 0.058로 사실상 belief 무시 → 학습이 "belief를 실제로 쓰게" 만든다는 증거.

---

## 6. sensitivity ≠ utility (반드시 함께 볼 것)

**sensitivity가 높다 = 행동이 belief에 *반응*한다**. 그러나 **옳은 방향(GT)으로** 반응하는지는 별개다.
belief를 바꾸면 행동이 아무렇게나 흔들려도 sensitivity는 높게 나온다.

→ 방향성은 **belief-only utility U_g**(§4 G-CC3, 필수)로 따로 잰다:
```
U_g = p(a_GT | reasoning 고정, own belief) − p(a_GT | reasoning 고정, swap belief)
    = diff_ci(P["own"], P["swap_b"])            # harden_s3: utility_belief_only_ci
```
U_g>0 = 자기 belief가 남의 belief보다 GT를 **더 잘 지지** = belief가 유용.
관련 신호: `D_g`(directional, `Pr[P_own>P_swap_b]`), `correct_switch`(belief-swap로 top1 바뀐 샘플의 평균 GT-확률 하락).

**"useful belief dependence" 주장 = G-CC1(sensitivity) ∧ G-CC3(U_g) 둘 다 통과.**
sensitivity만 오르고 U_g 실패면 → 분기 8.3(sensitivity-only), "embodied reasoning" 제목 곤란.

⚠️ **레거시 함정**: 예전 utility 지표는 `own − swap_both`(reasoning까지 같이 swap)라 belief 효과가
reasoning 변화에 오염됨. retro3에서 이 값은 0.108→0.067로 **하락**했지만, 올바른 belief-only(`own−swap_b`)로
재보면 0.023→0.042로 **상승**. **반드시 `utility_belief_only_ci`(U_g)를 봐야 한다.**

---

## 7. 게이트 정리

| 게이트 | 코드 | 조건 | 의미 |
|---|---|---|---|
| **G-CC1** | `causal["belief"].lo > 0` | belief sensitivity CI 하한>0 | belief 개입 의존 실재 (필수) |
| G-S3a | `causal["both"].lo > 0` | 전체 prefix 개입 실재 | |
| **G-CC3** | `utility_belief_only_ci.lo > 0` | U_g CI 하한>0 | belief가 유용(필수) |
| G-CC4 | `D_g > chance/base` | 방향성 | |

최소 조건: **G-CC1 ∧ G-CC3**.

---

## 8. 함정·주의

1. **interventional ≠ causal**: 인과 경로/매개 주장 금지(§9). "행동이 belief 내용에 개입적으로 의존"까지만.
2. **sensitivity는 방향맹**: 반드시 U_g와 함께. 위 §6.
3. **donor `(i+7)%n`**: 결정적 이웃. 완전 무작위 아님 — 재현성엔 좋으나, 매우 유사한 이웃이면 swap 효과 과소평가 가능(현행 채택값).
4. **CI**: `diff_ci`는 표본 paired bootstrap. 방법론 §2는 **video-cluster** bootstrap을 요구 — 최종 판정 CI는 cluster 단위로 재확인 권장(현 구현은 sample 단위).
5. **base 동일 포맷 비교(G-CC2)**: 학습된 성질임을 보이려면 같은 포맷의 base가 낮아야 함(retro3 base 0.058이 근거).
6. **malformed 제외**: belief/reasoning 없거나 포맷 깨진 샘플은 집계에서 빠짐 → n 확인.

---

## 9. cesft_v2에서 언제 나오나

Phase A의 **sft_r15 harden** 단계(헤드라인, ~8h 시점)에서 `theta_ce+SFT`에 대해 측정.
`outputs/step2_retrospection/cesft_v2/sft_r15/… harden_s3.json` → `causal_sensitivity_ci.belief`(G-CC1),
`utility_belief_only_ci`(G-CC3). base 앵커와 CI 분리 확인.

논문 문장(§9): *"We test whether action selection is interventionally dependent on the semantic content of the generated task belief."*
