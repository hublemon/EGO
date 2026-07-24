# EGO 방법론 Handoff 1
## 비학습적 Prospection + Projected-Hindsight Trace Distillation

> 작성일: 2026-07-22 KST
> 목적: strict action anticipation 조건에서 WM이 제공한 future-action support를 비학습적으로 사용하고, Base Qwen의 reasoning 능력을 보존한 채 projected-hindsight trace supervision으로 Retrospection을 수행하는 방법론을 정리한다.
> 핵심 원칙: **Prospection은 future support를 제공하고, Retrospection은 그 support 위에서 reasoning-task belief-action 정합성을 학습한다.**

---

## 0. 세 줄 요약

1. Step 1은 `target_start - 1s` 이전 영상만 사용해 Top-K future-action support를 생성한다.
2. Prospection에서는 별도의 RL 또는 likelihood distillation을 하지 않고, Base Qwen이 영상·과거 action history·셔플된 Top-K 후보를 보고 `reasoning-task_belief-action` trace를 생성한다.
3. Retrospection에서는 실제 이후 trajectory를 통해 당시의 task structure를 복원한 뒤, 이를 과거 시점에 관찰 가능한 evidence 수준으로 projection한 trace를 supervision으로 학습한다.

---

# 1. 배경과 문제 정의

기존 Prospection은 WM likelihood를 reward로 사용해 Qwen을 정렬하려 했다. 그러나 strict `start-1s` action anticipation에서는 WM의 세부 likelihood와 ranking이 부정확할 수 있다.

또한 WM Top-K만 후보로 제공하는 상황에서는 어떤 후보를 선택해도 이미 support 안에 있으므로, uniform reward나 support-membership reward만으로는 support 내부 선택을 학습할 수 없다.

따라서 본 방법론은 Prospection을 다음처럼 재정의한다.

> Prospection은 parameter update를 수행하는 학습 단계가 아니라, independently predicted future-action support를 제공하는 non-parametric interface다.

이후 Retrospection이 실제 parameter alignment를 담당한다.

---

# 2. 전체 구조

```text
Strict observation: target_start - 1s
        →
Frozen V-JEPA2-based action anticipator
        →
Shuffled Top-K future-action support
        →
Base Qwen3-VL-8B-Instruct
        →
reasoning - task_belief - action trace
        →
Future trajectory를 이용한 hindsight task inference
        →
과거 evidence 수준으로 projection
        →
Projected-trace distillation
```

---

# 3. Strict temporal contract

Target action \(a_t^{GT}\)의 시작 시각을 \(s_t\), anticipation horizon을 \(\tau=1.0s\)라 한다.

\[ t = s_t - \tau \]

관찰 영상은 target action 시작 전에 종료되어야 한다.

\[ \max \operatorname{time}(x_{\le t}) < s_t \]

과거 action history는 decision point 이전에 완료된 action만 포함한다.

\[ H_{<t} = \{a_i \mid e_i \le t\} \]

필수 assertion:

```python
assert observation_end_sec <= decision_time_sec
assert observation_end_sec < target_start_sec
assert max(observed_frame_time) < target_start_sec
assert all(action.stop_sec <= decision_time_sec for action in history)
```

금지:

- `target_end - 1s`
- target action 내부 frame
- target action 자체를 history에 포함
- WM과 VLM의 decision point 불일치

---

# 4. Prospection: Non-Parametric Future-Support Interface

## 4.1 WM support

Frozen WM은 strict observation으로 Top-K action support를 생성한다.

\[ D_t = \operatorname{TopK} q_{\mathrm{WM}}(a \mid x_{\le t}) \]

중요한 해석:

- \(D_t\)는 실제 정답 집합이 아니다.
- \(D_t\)는 모든 가능한 행동의 완전한 집합도 아니다.
- \(D_t\)는 현재 관찰에서 WM이 지지하는 near-future action support다.
- WM likelihood와 rank는 Qwen prompt에 노출하지 않는다.

## 4.2 Base Qwen input

Qwen은 다음 causal context를 받는다.

\[ c_t = (x_{\le t}, H_{<t}, D_t) \]

입력 구성:

- strict video frame/grid 또는 short clip
- decision point 이전에 완료된 GT action history
- 순서를 무작위로 섞은 WM Top-K 후보
- candidate likelihood와 rank는 비공개

## 4.3 Base trace

Qwen은 별도 학습 없이 다음 trace를 생성한다.

\[ y_t^{base} = (r_t^{base}, b_t^{base}, a_t^{base}) \]

출력 형식:

```text
<reasoning>
현재 observation, action history, candidates를 바탕으로
후보를 비교하고 다음 action을 추론한다.
</reasoning>

<task_belief>
현재 수행 중인 local procedure 또는 subgoal을 추정한다.
</task_belief>

<action>
Top-K 후보 중 하나를 선택한다.
</action>
```

Constrained decoding 또는 exact candidate matching을 사용해 \(a_t^{base} \in D_t\)를 보장한다.

## 4.4 Prospection에서 학습하지 않는 것

Prospection에서는 다음을 사용하지 않는다.

- WM likelihood reward
- uniform reward
- GRPO
- EMA-REINFORCE
- action GT
- candidate CE
- full-trace policy gradient

이유:

- Top-K 안의 모든 후보가 support에 속하므로 uniform reward는 informative action gradient를 제공하지 않는다.
- raw likelihood는 strict anticipation에서 calibration이 부정확할 수 있다.
- Base Qwen의 reasoning을 잘못된 reward로 훼손할 위험이 있다.

---

# 5. Oracle action history 사용

본 방법론의 첫 검증에서는 decision point 이전에 실제로 완료된 GT action sequence를 history로 사용한다.

\[ H_{<t}^{GT} \]

이 setting은 다음을 확인하기 위한 oracle-history experiment다.

> 과거 task progression 정보가 정확히 주어졌을 때, WM support 위에서 Base Qwen과 Retrospection이 올바른 next-action reasoning을 형성할 수 있는가?

논문에서는 반드시 다음처럼 명시한다.

> ground-truth completed-action history

확장 실험에서는 predicted history를 추가한다.

- oracle history
- predicted history
- no history

세 조건을 비교한다.

---

# 6. Retrospection: Hindsight Task Inference

## 6.1 Future trajectory

Decision point 이후 실제로 발생한 action sequence를 사용한다.

\[ F_t = (a_t^{GT}, a_{t+1}, \ldots, a_{t+m}) \]

단, future trajectory는 online input으로 사용하지 않는다. 오직 offline supervision 생성에만 사용한다.

## 6.2 Hindsight teacher

Frozen teacher는 future trajectory를 보고 당시 진행 중이던 procedural structure를 추론한다.

\[ h_t = \Psi(F_t) \]

Teacher output 예시:

- broad activity
- completed subgoal
- procedural stage
- local next subgoal
- multiple plausible task hypotheses
- uncertainty

---

# 7. Hindsight projection

Future trajectory에서 얻은 정보를 그대로 target trace에 넣으면 future leakage가 발생한다.

따라서 hindsight task interpretation을 과거 decision point에서 실제로 관찰 가능하던 evidence 수준으로 projection한다.

\[ (r_t^{proj}, b_t^{proj}) = \Phi(x_{\le t}, H_{<t}, D_t; h_t) \]

## 7.1 Projection 원칙

### Observability
Projected reasoning의 모든 사실은 당시 영상과 completed history로 뒷받침되어야 한다.

### Temporal validity
Target action이 이미 진행 중이거나 완료됐다고 표현하면 안 된다.

### Appropriate specificity
Future trajectory가 정확한 task를 보여주더라도, 당시 evidence가 모호하다면 belief의 구체성을 낮춘다.

### Non-restatement
Task belief는 GT action label을 문장으로 복사하면 안 된다.

### Belief-action compatibility
Projected belief는 GT next action과 procedural하게 일관되어야 한다.

---

# 8. Projected target trace

최종 supervision은 다음이다.

\[ y_t^{proj} = (r_t^{proj}, b_t^{proj}, a_t^{GT}) \]

Retrospection 학습에는 기본적으로 \(a_t^{GT} \in D_t\)인 sample만 사용한다.

GT가 Top-K에 없으면 policy가 구조적으로 정답을 선택할 수 없으므로 WM support failure로 분리한다.

---

# 9. R1: Projected-Trace Distillation

## 9.1 목적

R1의 목적은 action 정답만 맞히게 하는 것이 아니라, projected task belief를 중심으로 coherent trace를 생성하도록 학습하는 것이다.

## 9.2 Span-normalized loss

\[ \mathcal L_{R1} = \lambda_r \mathcal L_r + \lambda_b \mathcal L_b + \lambda_a \mathcal L_a \]

초기 권장값:

\[ \lambda_b = 1.0,\quad \lambda_r = 0.5,\quad \lambda_a = 0.25 \]

핵심:

- belief가 primary supervision
- reasoning은 belief를 observation과 history에 연결
- action은 belief와 trace coherence를 묶는 anchor
- action CE가 학습 전체를 지배하지 않도록 함

## 9.3 장점

- 모든 accepted sample에서 nonzero gradient
- reasoning 길이가 belief/action loss를 압도하지 않음
- reward 분배가 명시적
- high-variance RL 불필요
- projected trace가 직접 supervision 역할

---

# 10. R2: Belief-Conditioned Action Consistency

Projected trace를 모방한다고 해서 belief가 실제 action preference에 영향을 준다고 볼 수는 없다.

따라서 projected belief prefix 아래에서 후보별 action score를 계산한다.

\[ s_\theta(a \mid c_t, r_t^{proj}, b_t^{proj}) = \frac{1}{|a|} \log \pi_\theta(a \mid c_t, r_t^{proj}, b_t^{proj}) \]

Top-K 내 normalized candidate probability:

\[ p_\theta(a) = \frac{\exp s_\theta(a)}{\sum_{a' \in D_t}\exp s_\theta(a')} \]

Auxiliary loss:

\[ \mathcal L_{BA} = -\log p_\theta(a_t^{GT}) \]

전체 objective:

\[ \mathcal L_{Retro} = \mathcal L_{R1} + \lambda_{BA}\mathcal L_{BA} + \lambda_{pres}\mathcal L_{preserve} \]

---

# 11. Belief intervention evaluation

Retrospection이 성공하려면 projected belief가 action probability에 기능적으로 영향을 주어야 한다.

성공 기준:

\[ p_\theta(a_t^{GT} \mid b_t^{proj}) > p_\theta(a_t^{GT} \mid \varnothing) > p_\theta(a_t^{GT} \mid b_t^{incompatible}) \]

Hard action flip만 보지 않고 candidate margin 변화를 함께 측정한다.

---

# 12. Checkpoint 계보

```text
QWEN-BASE
  Qwen3-VL-8B-Instruct

BASE-SUPPORT
  학습 없이 strict Top-K constrained inference

RETRO-R1
  projected-trace distillation

RETRO-R2
  R1 + belief-conditioned action consistency

RETRO-FINAL
  trace-quality와 intervention gate를 통과한 checkpoint
```

---

# 13. 필수 실험

| 조건 | 목적 |
|---|---|
| Base, candidates 없음 | Qwen 자체 next-action 능력 |
| Base + WM Top-K | support interface 효과 |
| Base + WM Top-K + R1 | projected trace supervision 효과 |
| Base + WM Top-K + R1 + R2 | belief-conditioned action coherence |
| Action-only SFT | projected belief 없이 action GT만 학습한 기준선 |
| Random Top-K | WM support의 유효성 |
| Frequency Top-K | 단순 빈도 prior와 비교 |
| Oracle Top-K | support coverage 상한 |

---

# 14. 평가 지표

## Support
- Coverage@5/10/15
- GT rank
- MRR
- support failure rate

## Action
- overall next-action accuracy
- SelAcc@K
- G1 retention
- G2 correction
- GADR

## Reasoning
- observation grounding
- history grounding
- temporal correctness
- unsupported future reference
- current-action/next-action confusion

## Task belief
- projected belief acceptance rate
- appropriate specificity
- action-label restatement rate
- belief-action semantic consistency
- correct/no/incompatible belief intervention

---

# 15. 주장 경계

## 주장 가능

- WM이 independent future-action support를 제공한다.
- Base Qwen은 그 support 위에서 zero-shot reasoning과 action selection을 수행한다.
- Projected hindsight supervision은 task belief와 action selection의 semantic coherence를 개선한다.
- Retrospection이 action-only supervision보다 더 나은 belief-action relationship을 만든다.

## 주장 불가

- Qwen이 Prospection 단계에서 WM support를 parameter-level로 내재화한다.
- WM likelihood를 학습한다.
- Prospection 자체가 reasoning을 향상시켰다.
- task belief causality가 judge score만으로 입증된다.

---

# 16. 최종 확정

\[ \boxed{\text{Prospection} = \text{non-parametric future-support grounding}} \]

\[ \boxed{\text{Retrospection} = \text{projected task-belief trace supervision}} \]

핵심 비교:

\[ \text{Base + WM TopK + Retro} > \text{Base + WM TopK} \]

그리고

\[ \text{Base + WM TopK + Retro} > \text{Action-only SFT} \]

두 조건을 모두 만족해야 projected hindsight task belief의 독자적 가치를 주장할 수 있다.

---

*이 문서는 2026-07-22 사용자 첨부 원문을 EGO_jihun3에 기록한 것이다. Top-10으로 쓰였던 원문의 support 크기는 인계된 probe의 실측 coverage에 따라 Top-K(K=10 확정, 2026-07-22 변경)로 일반화해 표기했다 — 착수 문서 참조.*
