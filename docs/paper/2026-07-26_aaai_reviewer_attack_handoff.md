# AAAI reviewer attack handoff — Embodied Reasoning Results

- 작성일: 2026-07-26
- 검토 대상:
  - `docs/paper/2026-07-26_embodied_reasoning_results.pdf`
  - `docs/paper/2026-07-26_embodied_reasoning_results.tex`
- 검토 범위: 현재 Results 초안의 주장–증거 정합성, 실험 설계, 통계, 기제 해석
- 가상 판정: **Weak Reject (4/10)**
- confidence: **4/5**

## 0. 한 줄 판정

현재 결과가 직접 지지하는 결론은 다음까지다.

> Candidate-aware cross-entropy training improves conditional selection within
> a frozen world-model proposal set.

반면 아래의 더 강한 주장은 아직 충분히 입증되지 않았다.

1. 개선이 시각적으로 grounded된 **embodied reasoning**에서 온다.
2. candidate alignment가 학습한 이득이 completed-action history를 읽는
   특정 인과 기제로 운반된다.
3. 생성된 task belief가 정답 선택에 유용한 decision-relevant state다.
4. Retrospection, Replay, grounding oversampling을 포함한 최종
   **EGO 전체 방법**이 Cand.-CE보다 낫거나 그 성능을 보존한다.

현재 가장 방어하기 쉬운 논문은 “candidate-aligned,
trajectory-conditioned selection” 논문이다. `embodied reasoning`과
`state-bearing belief`를 headline으로 유지하려면 추가 인과 통제가 필요하다.

---

## 1. 리뷰어가 인정할 강점

공격에 대응할 때 아래 장점은 유지해야 한다.

1. GT-only와 frozen WM Top-1을 모두 비교했다.
2. 후보의 score와 rank를 숨기고 순서를 섞어 단순 rank copying 반론을
   일부 통제했다.
3. 같은 evaluation cohort에서 paired comparison을 수행했다.
4. 같은 비디오 안의 상관을 고려해 video-cluster bootstrap을 사용했다.
5. preregistered non-inferiority 실패와 final belief intervention 미실행을
   숨기지 않았다.
6. covered-set accuracy와 full-set equivalent를 구분해 보고하려는 규율이
   있다.
7. evidence utility가 observational이라는 점을 본문에서 인정한다.

이 강점들은 결과의 신뢰성을 높이지만, 현재의 강한 기제 해석을 자동으로
정당화하지는 않는다.

---

## 2. 치명적 공격 1 — embodied reasoning이 아니라 textual script completion

### 리뷰어 공격

정책은 다음 세 입력을 동시에 받는다.

- first-person observation
- 긴 completed-action history
- 의미가 명시된 Top-10 action candidates

그런데 아래 통제가 없다.

- image 제거 또는 unrelated/shuffled image
- history-only selector
- 후보 텍스트만 받는 selector
- 다른 비디오의 history
- 시간 순서를 섞거나 뒤집은 history
- last-action 또는 action-transition-frequency baseline

따라서 `+7.70pp`가 픽셀에서 얻은 embodied evidence 때문인지, 행동 이름으로
구성된 텍스트 trajectory의 script completion 때문인지 분리되지 않는다.

가장 예상되는 리뷰 문장:

> The experiments establish trajectory-conditioned candidate selection, not
> embodied visual reasoning.

### 추가 논리 문제

Evaluation protocol은 frozen world model도 observation과 completed-action
history를 받는다고 설명한다. 그런데 결론에서는 world model이
“visually grounded action boundary”를 정의하고 language policy가 trajectory
context를 해석한다고 모듈을 분리한다.

후보 경계 자체가 history-conditioned라면 다음 두 정보 경로가 섞인다.

```text
history → world model → candidate set
history → language policy → candidate selection
```

현재 history-strip 실험은 두 번째 경로의 직접 입력만 제거하고, 원래 history로
만들어진 후보 집합은 고정한다. 따라서 정책의 직접 history 사용은 검사하지만
“WM은 visual boundary, LM은 trajectory reasoner”라는 전체 모듈 분해를
입증하지는 않는다.

### 최소 방어

1. image-only, history-only, image+history의 factorial ablation을 수행한다.
2. shuffled-image와 other-video-history control을 추가한다.
3. simple transition-prior와 last-action baseline을 보고한다.
4. WM이 history를 쓰는 설계를 유지한다면 `visually grounded boundary`를
   `observation- and history-conditioned proposal boundary`로 낮춰 쓴다.
5. 강한 embodied claim은 image intervention이 통과할 때만 유지한다.

---

## 3. 치명적 공격 2 — 28.8%는 전체 과제 성능이 아니다

### 핵심 수치

Headline `28.8%`는 GT가 이미 Top-10 후보에 포함된 covered subset에서의
conditional selection accuracy다.

- held-out Coverage@10: `43.4%`
- Cand.-CE/EGO conditional SelAcc: `28.8%`
- full-set equivalent: 약 `12.5%`
- frozen WM Top-1 full-set equivalent: 약 `10.5%`
- 전체 분포의 절대 개선: 약 `+2.0pp`

즉 held-out anticipation point의 `56.6%`에서는 selector가 정답을 고를 기회가
없다.

### 리뷰어 공격

> Why is 28.8% the headline rather than 12.5%, given that the candidate
> generator excludes the ground-truth action on most held-out examples?

Covered-set 분석은 selector 모듈 진단으로 유효하다. 그러나 이를 시스템 전체의
주 성능처럼 전면에 두면 oracle-conditioned evaluation 또는 selection bias라는
공격을 받는다.

### estimand 불일치

현재 표의 headline은 malformed를 오답 처리한 common covered `n=1,000`의
strict accuracy다. 반면 공식 paired CI는 arm별 non-malformed intersection을
사용한다.

- Cand.-CE vs GT-only: `n=948`
- Cand.-CE vs WM Top-1: `n=957`
- final EGO vs Cand.-CE: `n=937`

따라서 점추정과 CI가 서로 다른 모집단과 estimand를 사용한다. 특히
`non-malformed`는 모델 출력에 의해 결정되므로, 이를 조건으로 표본을 제외하면
arm-dependent post-treatment selection이 생길 수 있다.

### 최소 방어

1. 주 표의 첫 열에 full-set accuracy 또는 full-set equivalent를 둔다.
2. `28.8% conditional / 12.5% full-set equivalent`를 항상 함께 쓴다.
3. primary paired CI를 common `n=1,000`에서 malformed=incorrect로 재계산한다.
4. non-malformed intersection 결과는 secondary diagnostic으로 내린다.
5. arm별 malformed rate와 paired attrition 표를 제공한다.
6. 가능하면 sampled 1,000뿐 아니라 held-out 전체 5,326 및 covered 전체
   2,313에서 재평가한다.

---

## 4. 치명적 공격 3 — 최종 EGO가 Cand.-CE보다 좋아진 것이 없다

### 관측 결과

- Cand.-CE strict SelAcc: `28.8%`
- final EGO strict SelAcc: `28.8%`
- official paired SelAcc difference: `−0.64pp`
- official paired \(G_2\) difference: `−2.53pp`
- preregistered `1pp` non-inferiority: **FAIL**
- final EGO belief-swap evaluation: **미실행**

따라서 가장 명확하게 성공한 intervention은 candidate-aligned CE다.
Retrospection-with-Replay와 grounding oversampling을 포함한 최종 방법은
aggregate selection을 개선하지 못했고, 사전 정의한 보존 조건도 통과하지
못했다.

### 예상 리뷰어 질문

> If the final method neither improves over candidate CE nor passes its own
> preservation criterion, why is candidate CE not the entire paper?

### Figure 1(b) 공격

Figure 1(b)는 Cand.-CE `28.8%`에서 final EGO `28.8%`까지 평평한 선을 그려
“holding 28.8%”처럼 보이게 한다. 그러나 본문은 중간
Retrospection-with-Replay checkpoint가 `26.4%`로 하락했다가 grounding
oversampling 후 복구됐다고 밝힌다.

실제로 측정한 중간점을 생략하고 endpoint를 직선으로 연결하면 training path를
과도하게 유리하게 시각화했다는 공격을 받을 수 있다. 더구나 앞 단계의 곡선도
소수 checkpoint 사이를 guide-to-eye로 보간한 것이므로 실제 learning curve처럼
오독될 위험이 있다.

### 최소 방어

1. Figure 1(b)에 replay checkpoint `26.4%`를 실제 측정점으로 추가한다.
2. 측정하지 않은 구간은 곡선 대신 점과 얇은 점선으로만 연결한다.
3. final EGO를 Cand.-CE보다 우월하다고 표현하지 않는다.
4. 최종 단계의 역할을 “improvement”가 아니라
   “recovery after retrospection degradation”으로 정확히 쓴다.
5. Retrospection이 논문의 핵심 기여라면 multi-seed 개선 또는 적어도
   preregistered non-inferiority를 새 독립 run에서 통과해야 한다.
6. 추가 실험이 어렵다면 논문 중심을 candidate alignment로 옮기고
   Retrospection은 exploratory extension으로 낮춘다.

---

## 5. 치명적 공격 4 — history removal만으로 인과 기제를 확립할 수 없다

### 현재 결과

History를 제거했을 때:

- Base: `−2.9pp`
- GT-only: `−4.5pp`
- Cand.-CE: `−10.1pp`
- replay checkpoint: `−6.4pp`

No-history 조건에서 Cand.-CE는 Base 및 GT-only와 통계적으로 구분되지 않는다.

### 리뷰어 공격

History 전체 제거는 학습 때 항상 존재하던 입력 블록을 없애는 OOD
intervention이다. 큰 drop은 다음 둘 중 어느 쪽으로도 설명할 수 있다.

1. Cand.-CE가 유용한 trajectory reasoning을 학습했다.
2. Cand.-CE가 history omission 또는 prompt-format shift에 더 취약해졌다.

현재 실험은 두 해석을 구분하지 않는다.

또한 논문이 실제로 필요한 검정은 각 arm drop의 개별 CI가 아니라 다음
difference-in-differences다.

\[
(\Delta_{\text{Cand.-CE}}-\Delta_{\text{GT-only}})
\quad\text{and}\quad
(\Delta_{\text{Cand.-CE}}-\Delta_{\text{Base}})
\]

이 interaction의 cluster-bootstrap CI가 직접 제시되지 않았다.

### 최소 방어

1. 같은 길이의 shuffled history를 넣는다.
2. other-video history를 넣는다.
3. history 시간 순서만 뒤집는다.
4. 최근 행동 `0, 1, 3, 7, 8+`개로 dose-response를 측정한다.
5. history 없이 별도로 학습한 control을 추가한다.
6. arm×history interaction을 paired video-cluster bootstrap으로 검정한다.
7. 현재 소제목 `Causal Test`를 `Inference-time History Ablation`으로 낮춘다.

강한 인과 주장은 semantic corruption control과 interaction이 통과할 때만
복원한다.

---

## 6. 치명적 공격 5 — belief 결과가 “학습된 유용한 belief”를 보이지 않는다

### 현재 결과

Belief-swap sensitivity:

- Base: `0.073`
- GT-only: `0.085`
- Cand.-CE: `0.093`
- final EGO: 미측정

Cand.-CE와 Base의 차이는 `0.020`에 불과하고, arm 간 차이의 CI나 검정이 없다.
따라서 nonzero sensitivity는 candidate alignment가 학습시킨 성질이 아니라
base model의 일반적인 prefix sensitivity일 수 있다.

### sensitivity의 방향맹 문제

Belief-swap sensitivity는 belief를 바꾸면 top-1 action이 얼마나 자주 바뀌는지를
잰다. 그러나 action이 정답 방향으로 변하는지는 측정하지 않는다. 모델이 무작위로
흔들려도 sensitivity는 높아질 수 있다.

필요한 핵심 지표는 다음이다.

\[
U_g =
p(a_{\mathrm{GT}}\mid r_{\mathrm{own}},b_{\mathrm{own}})
-
p(a_{\mathrm{GT}}\mid r_{\mathrm{own}},b_{\mathrm{swap}})
\]

즉 자기 belief가 donor belief보다 GT를 더 잘 지지하는지 보여야 한다.
현재 Results에는 belief-only utility, directional probability,
correct-switch 결과가 없다.

### intervention의 OOD 문제

- donor는 `(i+7) mod n`의 다른 표본에서 가져온다.
- reasoning은 원래 표본의 것을 유지한 채 belief만 타 표본 것으로 교체한다.
- reasoning과 belief가 의미적으로 모순된 prefix가 될 수 있다.
- 이런 hybrid prefix는 모델의 정상 생성 분포 밖일 수 있다.
- paraphrase flip을 단순 차감하는 것이 semantic effect를 완전히 식별한다는
  보장도 없다.

### echo 해석 문제

낮은 belief–action echo는 belief가 action 문자열을 그대로 복사하지 않았다는
뜻뿐이다. 낮은 echo와 nonzero swap sensitivity를 결합해도 belief가
decision-relevant하거나 정답에 유용하다는 결론은 나오지 않는다.

### 최소 방어

1. arm별 sensitivity CI뿐 아니라 Cand.-CE−Base의 paired CI를 보고한다.
2. belief-only utility \(U_g\)와 cluster-bootstrap CI를 보고한다.
3. directional \(D_g\), correct-switch, GT probability change를 함께 보고한다.
4. donor를 video-disjoint하게 뽑고 여러 random seed에서 반복한다.
5. semantic compatibility를 통제한 hard-negative belief swap을 추가한다.
6. final EGO에 동일 intervention을 실행한다.
7. final 결과가 없으면 belief claim은 Cand.-CE checkpoint에 한정하고
   EGO 전체의 기제로 일반화하지 않는다.
8. 현 문장의 `decision-relevant state`를
   `action-sensitive generated prefix` 수준으로 낮춘다.

---

## 7. 주요 추가 공격

### 7.1 단일 학습 run

Video-cluster bootstrap은 evaluation sampling uncertainty만 다룬다. 학습 seed,
data order, adapter initialization, checkpoint selection에 따른 training
uncertainty는 반영하지 않는다.

`+7.70pp`가 한 번의 학습 run에서 나온 값이라면 일반적인 방법 효과로 주장하기
어렵다.

필요 조치:

- 핵심 Base, GT-only, Cand.-CE를 최소 3 seeds로 반복한다.
- seed 평균, 표준편차, 각 seed의 paired test 결과를 함께 보고한다.
- compute 제약으로 반복할 수 없다면 single-run limitation을 명시한다.

### 7.2 GT-only가 충분한 통제가 아닐 수 있음

Cand.-CE는 10개 후보를 이용한 discriminative objective이고 GT-only는
answer-only objective다. 두 arm은 단순히 “candidate exposure 유무”뿐 아니라
negative supervision, 출력 공간, optimization geometry가 다를 수 있다.

필요 통제:

- random/unrelated candidate set으로 학습
- global-frequency candidate set
- in-batch negatives를 갖는 GT-only
- 후보는 보되 정답 contrast를 사용하지 않는 control
- train candidates × test candidates의 2×2 factorial
- 동일 token budget, target format, number of optimizer updates

### 7.3 capability axis가 독립 능력처럼 제시됨

\(G_1\)과 \(G_2\)는 같은 selection accuracy를 WM correctness로 조건부 분해한
것이다. Continuation과 evidence utility도 표본 특성이 겹친다. 이를 “five
capability axes”로 부르면 독립적으로 검증된 다섯 능력처럼 보일 수 있다.

필요 조치:

- 각 조건부 지표의 denominator와 CI를 표에 병기한다.
- arm×subset interaction을 검정한다.
- confirmatory와 exploratory axis를 분리한다.
- multiple testing 또는 최소한 분석 family를 명시한다.

### 7.4 continuation precision 해석 과장

`68–73%`의 precision이 비슷해 보인다는 것만으로 “looser tendency to repeat”를
배제할 수 없다.

필요 조치:

- precision 차이의 CI
- 반복 예측률
- 직전 action과 GT가 같은 비율
- non-continuation subset의 false repetition rate
- last-action baseline

을 보고한다.

현재 `stable precision rules out`은 `is consistent with`로 낮추는 것이 안전하다.

### 7.5 evidence utility의 selection bias

Evidence를 언급한 trace와 언급하지 않은 trace는 난이도, confidence,
continuation 여부가 다르다. Evidence mention rate도 arm마다 크게 달라져 서로
다른 표본 집합의 accuracy를 빼고 있다.

필요 조치:

- 정규식/분류기의 annotation agreement와 precision/recall
- continuation 여부, history length, WM rank를 통제한 regression
- within-video 또는 matched analysis
- non-continuation subset 결과

현재 evidence utility는 메커니즘 증거가 아니라 descriptive diagnostic으로만
유지해야 한다.

### 7.6 정성 예시가 outcome-conditioned

예시 선정 규칙은 다음을 요구한다.

- 모든 arm valid
- GT-only가 belief에 action을 copy
- GT-only 오답
- EGO 정답

그 결과 31/915를 고르고, 다시 Cand.-CE도 오답인 18개 중 하나를 제시한다.
고정 규칙이라는 사실은 cherry-picking 반론을 없애지 않는다. 규칙 자체가
EGO에 유리한 outcome을 조건으로 한다.

더구나 제시된 예시에서 실제 마지막 completed action은 `cook flatbread`인데
EGO reasoning은 “I have just finished pressing the dough”라고 말한다. 정답
`check heat`를 맞혔더라도 최신 history를 정확히 추적한 state-bearing reasoning의
사례인지 의심받을 수 있다.

필요 조치:

- outcome-independent random examples
- 성공/실패 case를 모두 포함한 stratified sample
- selection criterion을 결과를 보기 전에 고정했다는 기록
- 예시별 image crop 또는 frame을 함께 제시
- history factuality annotation

### 7.7 표준 및 강한 baseline 부재

현재 비교는 주로 자체 Base, GT-only, WM Top-1이다. Full-set equivalent
`12.5%`가 실제 연구 맥락에서 강한지 판단할 수 없다.

필요 baseline:

- 공개 SOTA 또는 강한 frontier VLM selector
- history transition prior
- candidate text similarity
- last-action/repeat baseline
- oracle candidate ranker
- image-only and history-only learned selectors

### 7.8 train–heldout coverage shift

실험 기록상 WM train Coverage@10은 `71.6%`, held-out은 `43.4%`다. 학습은
covered example만 사용하므로 WM이 이미 잘 맞히는 쉬운 train 사례에 supervision이
편중될 가능성이 있다.

필요 조치:

- train/heldout의 GT rank histogram
- rank-stratified SelAcc와 \(G_2\)
- inverse-frequency 또는 heldout-like reweighting
- coverage shift limitation 명시

### 7.9 단일 GT의 과제 모호성

평균적으로 먼 미래 행동을 하나의 exact action label로 평가한다면 여러 행동이
합리적일 수 있다. 후보 간 synonym normalization이 있어도 anticipation 자체의
multi-modality는 남는다.

필요 조치:

- human agreement
- multiple valid next actions 또는 semantic correctness 평가
- time-to-target별 성능
- action-frequency 및 ambiguity별 분석

---

## 8. 리뷰어가 던질 핵심 질문

 rebuttal 또는 본문에서 아래 질문에 한 문장씩 답할 수 있어야 한다.

1. 이미지가 없어도 같은 향상이 나는가?
2. 올바른 history 대신 다른 비디오 history를 넣으면 성능이 떨어지는가?
3. WM도 history를 받는데 왜 WM boundary를 purely visual이라고 부르는가?
4. 왜 28.8%가 headline이고 12.5%는 각주인가?
5. malformed를 오답 처리한 점추정에 왜 non-malformed subset CI를 붙였는가?
6. Cand.-CE의 효과가 여러 training seed에서 재현되는가?
7. 동일한 negative supervision을 가진 GT-only control이 있는가?
8. 최종 EGO가 Cand.-CE보다 나아진 지표는 무엇이며, 그 차이에 CI가 있는가?
9. 최종 EGO가 preregistered non-inferiority에 실패했는데 왜 최종 방법으로
   제시되는가?
10. Belief sensitivity가 Base 0.073에서 Cand.-CE 0.093으로 증가한 것이
    유의한가?
11. Swapped belief가 action을 바꾸는 것뿐 아니라 GT 확률을 올바른 방향으로
    변화시키는가?
12. 정성 예시의 EGO trace가 실제 마지막 action을 잘못 기술하는데 왜
    state-bearing example인가?
13. 개선이 pixel grounding이 아니라 action-history language modeling이라는
    설명을 어떤 실험이 배제하는가?

---

## 9. 수정 경로

### 경로 A — 추가 실험 없이 방어 가능한 축소안

가장 현실적이고 안전한 경로다.

1. 제목과 abstract에서 `embodied reasoning`을 낮춘다.
2. 핵심 기여를 candidate-aware CE의 conditional selection improvement로 둔다.
3. `causal history mechanism`을 `history dependence under inference-time
   ablation`으로 바꾼다.
4. belief 결과를 exploratory diagnostic으로 내린다.
5. final EGO가 Cand.-CE보다 낫다는 인상을 제거한다.
6. Retrospection을 failed/non-conclusive extension으로 정직하게 보고한다.
7. full-set `12.5%`를 conditional `28.8%`와 동일한 가시성으로 보고한다.
8. Figure 1(b)에 실제 replay `26.4%` 측정점을 추가한다.

이 경로의 가능한 중심 주장:

> Exposing a policy to a frozen model’s candidate boundary during supervised
> training improves its ability to select among covered next-action
> hypotheses, whereas answer-only supervision does not.

### 경로 B — embodied reasoning headline 유지안

다음이 최소 패키지다.

1. image × history factorial ablation
2. shuffled image, shuffled history, other-video history
3. transition-prior 및 history-only baseline
4. strict full-heldout 평가와 동일 estimand의 cluster CI
5. Base/GT-only/Cand.-CE 최소 3 training seeds
6. candidate exposure를 격리하는 matched negative-supervision control
7. final EGO의 belief intervention
8. belief utility \(U_g\), directional test, correct-switch
9. Figure 1(b)의 실제 checkpoint 표시
10. final EGO가 Cand.-CE를 개선하거나 사전 non-inferiority를 통과하는 독립 run

이 패키지 없이 강한 embodied/mechanistic 제목을 유지하면 rebuttal로 방어하기
어렵다.

---

## 10. 우선순위별 작업표

| 우선순위 | 작업 | 방어하는 공격 | 비용 예상 |
|---|---|---|---|
| P0 | full-set 12.5%를 headline에 병기 | covered-only 과장 | 낮음 |
| P0 | malformed=incorrect common-set CI | estimand 불일치 | 낮음 |
| P0 | Figure에 replay 26.4% 측정점 추가 | misleading trajectory | 낮음 |
| P0 | causal/embodied/belief 표현 완화 | overclaim | 낮음 |
| P0 | final EGO의 실패를 contribution과 분리 | method identity | 낮음 |
| P1 | shuffled/other-video history | history OOD 반론 | 중간 |
| P1 | image removal/shuffle | embodiment 식별 | 중간 |
| P1 | arm×history interaction CI | 인과 비교 | 낮음 |
| P1 | belief utility와 directional 결과 | sensitivity 방향맹 | 낮음–중간 |
| P1 | final EGO belief intervention | checkpoint mismatch | 중간 |
| P2 | 3-seed Cand.-CE/GT-only | training variance | 높음 |
| P2 | matched negative-supervision control | GT-only 불충분 | 높음 |
| P2 | held-out 전체 5,326 평가 | 표본/외적 타당성 | 중간–높음 |
| P2 | 강한 외부/SOTA baseline | relevance | 중간–높음 |

---

## 11. 예상 리뷰 전문

아래 문단은 실제 AAAI review의 summary/weaknesses에 가까운 형태다.

> The paper presents credible evidence that candidate-aware cross-entropy
> training improves conditional selection within a frozen world-model
> proposal set. The use of a GT-only control, shuffled candidate order, paired
> evaluation, and video-cluster bootstrap are strengths. However, the current
> experiments do not isolate visual grounding from textual trajectory
> completion, while the headline accuracy is measured only on the 43.4% of
> held-out examples for which the ground-truth action is already present in
> the proposal set. The corresponding full-set accuracy is 12.5%, only about
> 2.0 percentage points above the frozen world-model Top-1 policy.
>
> More importantly, the final EGO stage does not improve over candidate CE,
> fails its preregistered 1pp non-inferiority margin, and lacks the
> intervention used to support the belief mechanism. Removing action history
> is an out-of-distribution input ablation and, without shuffled or
> semantically incorrect history controls, does not establish that the gain
> arises from grounded trajectory reasoning. Likewise, nonzero belief-swap
> sensitivity shows action responsiveness but not that the generated belief
> is useful or correct; the reported sensitivity is only slightly larger than
> that of the base model and is not evaluated for the final checkpoint.
>
> I therefore find the candidate-alignment result promising, but the broader
> claims of embodied reasoning, causal history use, and state-bearing belief
> insufficiently supported.

---

## 12. 최종 판정 규칙

### 현재 상태

- Candidate alignment 효과: **지지됨**
- WM Top-1 초과: **covered paired subset에서 지지됨**
- Answer-only보다 우수: **지지됨**
- 전체 시스템 개선: **작고 coverage에 강하게 제한됨**
- 시각적 embodied grounding: **미식별**
- history dependence: **관측됨**
- history의 의미적/인과적 사용: **부분적으로만 지지**
- useful belief mechanism: **미입증**
- final EGO의 Cand.-CE 대비 개선: **기각**
- final EGO의 preregistered 성능 보존: **실패**

### 권고 판정

현재 원고 그대로라면 **Weak Reject**가 타당하다.

다만 논문을 candidate alignment의 명확한 결과로 축소하고 overclaim을 제거하면
경계선 논문으로 방어 가능하다. 강한 embodied reasoning 논문으로 유지하려면
image/history intervention, matched controls, multi-seed, final-checkpoint
mechanism evaluation이 필요하다.
