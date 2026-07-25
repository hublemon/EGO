# EGO 방법론 Handoff 2
## 비학습적 Prospection + Projected-Hindsight DPO Retrospection

> 작성일: 2026-07-22 KST
> 목적: strict action anticipation 조건에서 WM Top-K를 non-parametric future support로 사용하고, Base Qwen trace와 projected-hindsight trace 사이의 preference를 DPO로 학습하는 대안적 Retrospection 방법론을 정리한다.
> 핵심 원칙: **DPO 자체가 방법론의 핵심이 아니라, projected trace가 Base trace보다 의미적으로 더 타당하다는 validated preference를 학습하는 것이 핵심이다.**

---

## 0. 세 줄 요약

1. Prospection에서는 별도 학습 없이 strict WM Top-K와 GT completed-action history를 Base Qwen에 제공한다.
2. Base Qwen이 생성한 trace를 rejected candidate로, future trajectory를 과거 evidence 수준으로 projection한 trace를 chosen candidate로 구성한다.
3. 모든 pair를 사용하지 않고, belief 또는 action에 실제 semantic difference가 있으며 projected trace가 quality gate를 통과한 pair만 DPO로 학습한다.

---

# 1. 배경

Projected-trace SFT는 dense하고 안정적인 supervision을 제공하지만, teacher trace를 그대로 모방하게 만들 수 있다.

DPO 방식은 다음 질문에 답한다.

> 동일한 causal context에서 Base Qwen이 실제로 생성한 trace보다 projected hindsight trace를 더 선호하도록 학습하면, reasoning-task belief-action chain이 개선되는가?

그러나 이전 DPO 실험은 다음 문제를 가지고 있었다.

- `end-1s` recognition-like temporal contract
- target action이 이미 영상에 보임
- task belief가 action 선택에 필수적이지 않음
- chosen/rejected pair가 semantic하게 충분히 다르지 않음
- full-trace preference에서 belief credit이 약함
- accepted pair 수가 적어 effective signal이 부족함
- DPO가 action GT correction으로 축소될 위험

따라서 본 방법론은 strict `start-1s`에서 DPO를 다시 정의한다.

---

# 2. 전체 구조

```text
Strict observation: target_start - 1s
        →
Frozen WM Top-K future support
        →
Base Qwen zero-shot trace
        →
y- = Base reasoning-belief-action trace
        →
Future trajectory 기반 hindsight task inference
        →
Past evidence 수준으로 projection
        →
y+ = projected reasoning-belief-GT action trace
        →
Pair quality verification
        →
DPO Retrospection
```

---

# 3. Strict temporal contract

Target action \(a_t^{GT}\)의 시작 시각: \(s_t\)

Decision point: \(t=s_t-1s\)

Causal context: \(c_t=(x_{\le t}, H_{<t}, D_t)\)

조건:

\[ \max \operatorname{time}(x_{\le t}) < s_t \]
\[ H_{<t}=\{a_i \mid e_i \le t\} \]
\[ D_t=\operatorname{TopK} q_{\mathrm{WM}}(a \mid x_{\le t}) \]

필수 assertion:

```python
assert observation_end_sec < target_start_sec
assert all(action.stop_sec <= decision_time_sec for action in history)
```

---

# 4. Prospection: Non-Parametric Support Interface

Prospection에서는 parameter update를 하지 않는다.

Base Qwen은 다음을 입력받는다.

- strict video observation
- completed GT action history
- shuffled WM Top-K
- likelihood/rank 비공개

Base trace:

\[ y_t^{-} = (r_t^{-}, b_t^{-}, a_t^{-}) \]

여기서 \(y_t^{-}\)는 자동으로 rejected가 아니다.

다음 경우에는 DPO pair로 사용할 수 없다.

- Base action이 GT와 같고 belief도 projected belief와 의미적으로 대등
- 단순히 문체 차이만 존재
- Base belief가 더 정확하거나 더 보수적
- projected trace가 future leakage를 포함
- projection이 action label을 복사
- Base trace가 malformed라서 preference가 지나치게 쉬움

---

# 5. Chosen trace 생성

Future trajectory:

\[ F_t=(a_t^{GT},a_{t+1},\ldots,a_{t+m}) \]

Hindsight teacher:

\[ h_t=\Psi(F_t) \]

Projection:

\[ (r_t^{proj},b_t^{proj}) = \Phi(x_{\le t},H_{<t},D_t;h_t) \]

Chosen trace:

\[ y_t^{+} = (r_t^{proj},b_t^{proj},a_t^{GT}) \]

Retrospection training sample은 원칙적으로 \(a_t^{GT}\in D_t\)인 경우에만 사용한다.

---

# 6. Projected trace quality gate

Chosen trace는 다음 조건을 모두 만족해야 한다.

## 6.1 Evidence grounding
Projected reasoning은 당시 observation과 completed history로 뒷받침되어야 한다.

## 6.2 No future leakage
이후 action, object, outcome을 현재 관찰처럼 서술하면 안 된다.

## 6.3 Temporal correctness
Target action을 이미 수행 중이라고 표현하면 안 된다.

## 6.4 Appropriate specificity
당시 evidence가 모호하면 belief를 local procedural stage 수준으로 낮춘다.

## 6.5 Non-restatement
Task belief가 GT action을 직접 paraphrase하면 안 된다.

## 6.6 Belief-action compatibility
Projected belief 아래에서 GT action이 procedural하게 타당해야 한다.

## 6.7 Candidate validity
GT action이 WM Top-K 안에 있어야 한다.

---

# 7. Preference pair 구성

## 7.1 기본 pair

\[ (c_t,y_t^+,y_t^-) \]

- chosen: projected trace
- rejected: Base Qwen trace

## 7.2 Pair acceptance 조건

다음 중 하나 이상이 있어야 한다.

### Belief correction
\[ b_t^{-} \not\equiv b_t^{proj} \]
Base belief가 procedural stage를 잘못 해석하거나 지나치게 구체적 또는 모호하다.

### Action correction
\[ a_t^{-}\neq a_t^{GT} \]

### Reasoning correction
Base reasoning이 observation/history와 맞지 않거나 candidate elimination이 부실하다.

### Temporal correction
Base가 current action과 next action을 혼동한다.

## 7.3 Pair rejection 조건

- belief와 action이 사실상 동일
- style/length 차이만 존재
- chosen이 future information을 노출
- rejected가 parse 실패로 너무 쉬운 negative
- chosen의 belief가 action label restatement
- semantic judge가 차이를 확인하지 못함

---

# 8. Pair taxonomy

Pair를 유형별로 저장한다.

```text
B: belief-only correction
A: action-only correction
BA: belief and action correction
R: reasoning-only correction
T: temporal-semantics correction
```

권장 학습 우선순위:

1. BA
2. B
3. T
4. R
5. A

Action-only pair는 비중을 제한한다.

이유:

> DPO가 GT action correction objective로 축소되는 것을 막기 위해서다.

---

# 9. DPO objective

Reference policy는 pair 생성에 사용한 Base Qwen 또는 fixed initialization policy다.

\[ \pi_{ref} \]

Trainable policy: \(\pi_\theta\)

Standard DPO:

\[
\mathcal L_{DPO} = -\mathbb E \left[ \log\sigma \left( \beta \left[
\log\frac{\pi_\theta(y_t^+\mid c_t)}{\pi_{ref}(y_t^+\mid c_t)}
- \log\frac{\pi_\theta(y_t^-\mid c_t)}{\pi_{ref}(y_t^-\mid c_t)}
\right] \right) \right]
\]

---

# 10. Field-balanced DPO

Standard full-trace DPO는 긴 reasoning field가 preference score 대부분을 차지할 수 있다.

따라서 field별 length-normalized log-probability를 사용한다.

\[ S_\theta(y\mid c) = w_r S_\theta^r + w_b S_\theta^b + w_a S_\theta^a \]

각 field score:

\[ S_\theta^r = \frac{1}{|r|} \log\pi_\theta(r\mid c) \]
\[ S_\theta^b = \frac{1}{|b|} \log\pi_\theta(b\mid c,r) \]
\[ S_\theta^a = \frac{1}{|a|} \log\pi_\theta(a\mid c,r,b) \]

권장 초기값:

\[ w_b=1.0,\quad w_r=0.5,\quad w_a=0.25 \]

Field-balanced DPO:

\[ \mathcal L_{FB-DPO} = -\log\sigma \left( \beta [ \Delta_\theta^{FB} - \Delta_{ref}^{FB} ] \right) \]
\[ \Delta_\theta^{FB} = S_\theta(y^+\mid c) - S_\theta(y^-\mid c) \]

핵심:

- belief preference가 primary signal
- reasoning은 grounding을 담당
- action은 coherence anchor
- action token이 pair preference 전체를 지배하지 않도록 함

---

# 11. Pair weighting

모든 pair의 quality가 같지 않다.

Sample weight:

\[ w_i = w_{proj} \cdot w_{semantic} \cdot w_{type} \]

구성:

- projection confidence
- grounding confidence
- semantic difference confidence
- pair taxonomy weight
- support stability

예시:

| Pair type | Weight |
|---|---:|
| BA | 1.0 |
| B | 1.0 |
| T | 0.8 |
| R | 0.6 |
| A | 0.3 |

Weighted objective:

\[ \mathcal L = \frac{\sum_i w_i \mathcal L_i}{\sum_i w_i} \]

Low-confidence pair를 weight 0으로 넣지 않고 dataset에서 제거한다.

---

# 12. Positive signal이 0이 되지 않도록 하는 조건

DPO는 pair가 충분히 다르지 않으면 effective gradient가 약하다.

따라서 다음을 사전에 측정한다.

## Chosen-rejected reference margin
\[ m_{ref} = S_{ref}(y^+)-S_{ref}(y^-) \]

## Semantic distance
- belief semantic difference
- action difference
- temporal interpretation difference

## Pair acceptance rate
전체 Base trace 중 실제 DPO pair로 남는 비율

필수 보고:

- total samples
- GT-in-support samples
- projected trace accepted
- semantic-difference accepted
- final DPO pairs
- pair type distribution

DPO signal이 너무 적으면 R1 projected-trace SFT를 선행해야 한다.

---

# 13. 권장 학습 순서

## Option D1: Direct DPO

```text
Base Qwen
→ Base trace collection
→ Projection pair generation
→ Field-balanced DPO
```

사용 조건:

- final pair acceptance rate가 충분함
- BA/B pair가 다수
- chosen trace quality가 높음
- Base와 chosen의 semantic gap이 명확함

## Option D2: SFT warm-up + DPO

```text
Base Qwen
→ 소규모 projected-trace SFT
→ 새 policy trace 재생성
→ 고품질 preference pair 재구성
→ Field-balanced DPO
```

추천 조건:

- direct DPO pair가 너무 적음
- Base output format이 불안정
- projected belief를 거의 생성하지 못함
- chosen trace likelihood가 지나치게 낮음

본 방법론의 main DPO variant는 D1과 D2를 모두 비교한다.

---

# 14. DPO 이후 Belief-Action Consistency

DPO가 chosen trace를 선호하게 만들었다고 해서 belief가 action을 기능적으로 조향한다고 볼 수 없다.

따라서 intervention test가 필수다.

\[ p_\theta(a_t^{GT}\mid b_t^{proj}) > p_\theta(a_t^{GT}\mid \varnothing) > p_\theta(a_t^{GT}\mid b_t^{incompatible}) \]

추가 측정:

- GT action margin
- top-1 flip rate
- belief swap sensitivity
- paraphrase-control sensitivity
- random belief sensitivity
- same-belief/different-wording robustness

---

# 15. Checkpoint 계보

```text
QWEN-BASE
  Qwen3-VL-8B-Instruct

BASE-SUPPORT
  strict WM Top-K zero-shot trace generator

DPO-DIRECT
  Base → field-balanced DPO

SFT-WARM
  projected-trace SFT warm-up

DPO-AFTER-SFT
  SFT warm-up → regenerated pair → field-balanced DPO

RETRO-DPO-FINAL
  trace quality와 intervention gate를 통과한 checkpoint
```

---

# 16. 필수 ablation

| 조건 | 목적 |
|---|---|
| Base + WM Top-K | zero-shot support baseline |
| Projected-trace SFT | dense supervision baseline |
| Standard full-trace DPO | 기존 DPO 방식 |
| Field-balanced DPO | belief credit 강화 |
| SFT warm-up + DPO | signal 부족 보완 |
| Action-only DPO | action correction만으로 충분한지 |
| Belief-only pair subset | belief supervision의 독립 효과 |
| Random pair DPO | preference construction의 유효성 |

---

# 17. 핵심 평가

## Action
- overall next-action accuracy
- SelAcc@K
- G1 retention
- G2 correction
- GADR
- correct-to-wrong regression

## Trace
- reasoning grounding
- temporal correctness
- future leakage
- belief specificity
- action-restatement rate
- belief-action semantic coherence

## Preference
- chosen/rejected win rate
- held-out pair accuracy
- field별 DPO margin
- pair type별 성능

## Causality
- correct/no/incompatible belief intervention
- belief swap minus paraphrase-control
- candidate margin shift
- generated belief vs teacher belief gap

---

# 18. 성공 조건

DPO Retrospection을 유지하려면 최소한 다음을 만족해야 한다.

## Signal
- 충분한 final pair count
- BA/B pair가 action-only pair보다 적지 않음
- chosen/rejected semantic difference가 검증됨
- field-balanced margin이 실제 belief field에서 증가

## Quality
- future leakage 증가 없음
- action-restatement 증가 없음
- generated belief quality 개선
- reasoning temporal correctness 개선

## Action
- Base + WM보다 SelAcc@K 또는 G2 correction 개선
- G1 보존
- action-only DPO보다 우수하거나 최소한 belief quality에서 명확한 이점

## Causality

\[ p(a_{GT}\mid b_{proj}) > p(a_{GT}\mid no\ belief) > p(a_{GT}\mid b_{incompatible}) \]

---

# 19. 주장 경계

## 주장 가능

- Projected hindsight trace를 validated preference로 사용하면 Base trace보다 더 coherent한 reasoning-belief-action chain을 선호하도록 학습할 수 있다.
- Field-balanced DPO는 action correction에만 치우치지 않고 projected belief에 더 직접적인 preference signal을 줄 수 있다.
- Strict anticipation에서는 future trajectory가 당시 모호하던 procedural belief를 복원하는 유효한 supervision이 된다.

## 주장 불가

- DPO가 projected-trace SFT보다 본질적으로 우수하다.
- DPO win rate만으로 belief causality가 입증된다.
- 모든 Base trace가 rejected로 적절하다.
- action accuracy 상승만으로 reasoning이 좋아졌다고 결론 내린다.

---

# 20. 최종 확정

DPO는 Retrospection의 정의가 아니다.

\[ \boxed{\text{Retrospection} = \text{projected hindsight preference supervision}} \]

DPO는 그 preference를 학습하는 한 가지 구현이다.

\[ \boxed{\text{DPO} = \text{optional pairwise optimization mechanism}} \]

최종 핵심 비교:

\[ \text{Field-balanced DPO} \quad \text{vs.} \quad \text{Projected-trace SFT} \]

DPO가 우수하다고 주장하려면 action 성능뿐 아니라 다음도 개선해야 한다.

- task belief quality
- field-specific preference margin
- belief intervention sensitivity
- future leakage control
- Base reasoning preservation

---

*이 문서는 2026-07-22 사용자 첨부 원문을 EGO_jihun3에 기록한 것이다. 원문의 Top-10은 인계된 probe 기준 Top-K(K=10 확정, 2026-07-22 변경)로 일반화해 표기했다 — 착수 문서 참조.*
