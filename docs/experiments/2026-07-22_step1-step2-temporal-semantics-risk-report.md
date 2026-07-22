# EGO 시간축 의미 불일치 우려 보고서

> **핵심 판정**  
> 현재 EK100 기반 Step 1과 이를 사용하는 Step 2의 주평가를 **“진짜 next-action anticipation 정확도”로 해석하면 안 된다.**  
> 구현상 관찰 종료점은 target annotation의 `end−1s`이고 정답은 그 동일 annotation이므로, 모델은 대체로 **이미 진행 중인 현재 action을 인식하고 후보 중에서 고르는 문제**를 풀고 있다.

- 작성일: 2026-07-22
- 범위: EK100 기반 Step 1 world model, Step 2 VLM alignment/validation, 논문의 VPA·Planning 주장
- 성격: 논문 결과 해석과 실험 설계를 위한 내부 위험 보고서
- 심각도: **Critical — 핵심 task semantics 및 논문 claim에 직접 영향**

---

## 1. 한눈에 보는 문제

### 우리가 원했던 것

```text
관찰 가능한 과거 영상                      아직 시작하지 않은 target action
───────────────────────────┤ 1초 gap ├─────────────────────────────>
                           t        action_start

입력: t 이전 영상 + 완료된 과거 action
출력/GT: t 이후에 시작할 다음 action
의미: action anticipation
```

### 현재 EK100/Step 2가 실제로 하는 것

```text
                    target annotation action
               ├──────────────────────────────────┤
             start                 trigger       end
                                     ↑
                                  end−1s

입력: target action이 이미 진행 중인 영상
출력/GT: trigger가 속한 동일 annotation action
의미: late-action recognition에 가까운 candidate selection
```

| 질문 | 판정 |
|---|---|
| 실제 annotation GT와 비교해 채점하는가? | **예. 채점 자체는 동일 annotation GT를 기준으로 정상 수행된다.** |
| 그 GT가 관찰 이후의 다음 annotation action인가? | **아니다. trigger가 들어 있는 현재 annotation action이다.** |
| 현재 수치를 next-action anticipation 정확도라고 불러도 되는가? | **아니다.** |
| Step 2 VLM은 다음 action을 잘 고르는지 검증되는가? | **아니다. 현재 action을 top-5 안에서 잘 고르는지에 가깝다.** |
| 이 WM만으로 미래 planning 신호를 제공할 수 있는가? | 제한적이다. **미래 전개는 대부분 LLM의 procedural prior에 의존하게 된다.** |

### 프롬프트와 실제 학습 목표의 직접 충돌

현재 VLM 프롬프트는 분명하게 `NEXT action`을 고르라고 지시한다.

```text
choose the single most likely NEXT action from the five candidates
```

그러나 실제 데이터에서는 관찰 시점이 target annotation의 `end−1s`이므로 target action이
이미 진행 중이고, 학습·검증 정답도 그 동일 annotation action이다. 따라서 모델이 실제로
보상받고 검증되는 행동은 다음과 같다.

```text
프롬프트가 요구하는 것: 아직 시작하지 않은 NEXT action 선택
실제 supervision:       영상에서 이미 진행 중인 CURRENT action 식별
실제 validation:        선택한 action을 CURRENT annotation GT와 비교
```

> **즉, 언어적 instruction은 anticipation이지만 실질적인 optimization target은 recognition이다.**
> 모델이 프롬프트의 의미를 올바르게 따라 정말 다음 action을 선택하면, 오히려 현재 validation에서는
> 오답으로 처리될 수 있다.

---

## 2. 코드로 확인되는 결정적 증거

이 절의 내용은 성능 수치에 대한 해석이 아니라, 현재 구현의 시간 의미를 직접 결정하는 코드 증거다.

### 증거 A — trigger는 target annotation의 `end−1s`

Step 2 샘플 선정 코드는 다음과 같이 trigger를 만든다.

```python
trigger_frame = stop_frame - int(1.0 * fps)
```

출처: [`select_train.py`](../../src/ego/step2_vlm_alignment/data/select_train.py#L86-L94)

동시에 action 길이가 1.5초보다 긴 sample만 남긴다.

```python
stop_frame - start_frame > fps * 1.5
```

따라서 선택된 모든 정상 sample은 구조적으로 다음을 만족한다.

```text
start < end−1s = trigger < end
```

즉 trigger는 target action 시작 전이 아니라 **target action 내부**에 있다.

### 증거 B — 정답은 다음 row가 아니라 동일 annotation row

같은 코드가 `start_frame`, `stop_frame`, `trigger_frame`을 저장한 뒤, 그 **동일한 row**의 `verb_class`와 `noun_class`를 `gt_label`로 저장한다.

출처: [`select_train.py`](../../src/ego/step2_vlm_alignment/data/select_train.py#L119-L142)

다음 annotation row를 찾아 target으로 교체하는 로직은 없다. 따라서 정답은:

```text
next annotation action  (아님)
trigger가 속한 annotation action  (맞음)
```

### 증거 C — WM도 동일 trigger 이전 영상을 본다

Step 2용 V-JEPA 추론은 trigger 직전 약 4초 clip을 구성한다.

```python
indices = np.arange(trigger_frame - nframes, trigger_frame, fstp)
```

출처: [`vjepa_infer_train.py`](../../src/ego/step2_vlm_alignment/data/vjepa_infer_train.py#L58-L69)

trigger 자체가 target action 내부이므로 WM의 clip에도 target action 수행 장면이 포함된다. 이 WM top-5는 엄밀한 미래 action support라기보다 **현재 진행 action을 강하게 반영한 support**가 된다.

### 증거 D — VLM의 최신 프레임도 target action 내부다

VLM 입력은 trigger 시점의 한 프레임, 또는 trigger 기준 `−4.0, −2.67, −1.33, 0초` 네 프레임이다.

출처: [`extract_frame_train.py`](../../src/ego/step2_vlm_alignment/data/extract_frame_train.py#L72-L80)

최신 `0초` 프레임은 항상 trigger 프레임이므로 target action 내부다. 따라서 VLM은 text history에서 target label을 받지는 않더라도, **영상에서 이미 수행 중인 target action을 직접 볼 수 있다.**

### 증거 E — 프롬프트는 “NEXT”라고 하지만 시간축은 바꾸지 않는다

현재 VLM prompt는 다음과 같이 명시한다.

```text
choose the single most likely NEXT action from the five candidates
```

출처: [`train_grpo_action.py`](../../src/ego/step2_vlm_alignment/train_grpo_action.py#L157-L216), [`train_grpo_action.py`](../../src/ego/step2_vlm_alignment/train_grpo_action.py#L442-L505)

그러나 자연어로 `NEXT`라고 부르는 것은 데이터의 trigger와 GT 관계를 바꾸지 않는다. 현재 prompt의 `NEXT action`은 실질적으로 **현재 annotation action에 잘못 붙은 이름**이다.

이는 단순한 용어상의 부정확성이 아니라 instruction–supervision mismatch다. VLM은 문장으로는
미래 행동을 고르라는 요청을 받지만, gradient와 validation score는 현재 행동을 고를수록 좋아진다.
따라서 현재 학습은 프롬프트 의미를 강화하는 것이 아니라, 경우에 따라서는 그 의미를 무시하도록
학습할 수 있다.

### 증거 F — validation도 동일 annotation GT를 exact match한다

Step 2 validation은 dataset의 `gt_verb`, `gt_noun`을 가져와 VLM 출력과 직접 비교한다.

```python
gt_v, gt_n = ex["gt_verb"], ex["gt_noun"]
correct = (pred_verb, pred_noun) == (gt_v, gt_n)
```

출처: [`eval_battery.py`](../../scripts/step2/eval_battery.py#L169-L189)

따라서 현재 validation은 실제 GT를 사용하는 정상적인 분류 채점이지만, 그 GT의 시간 의미가 **미래의 다음 action이 아니라 현재 annotation action**이다.

### 증거 G — 진짜 미래 action은 별도 메타데이터에 존재하지만 주 action GT가 아니다

코드는 trigger 이후 시작하는 action들을 `future_gt_actions`로 별도 추출한다.

출처: [`extract_memory_train.py`](../../src/ego/step2_vlm_alignment/data/extract_memory_train.py#L117-L135)

이 필드는 online policy prompt에서 분리되고 Retrospection의 hindsight belief 구성에만 사용된다. 반면 Retrospection의 chosen action target은 계속 기존 `gt_action_t`, 즉 trigger가 속한 동일 annotation action이다.

출처: [`build_dpo_dataset.py`](../../src/ego/step2_vlm_alignment/retro/build_dpo_dataset.py#L95-L110), [`teacher.py`](../../src/ego/step2_vlm_alignment/retro/teacher.py#L57-L73)

결국 Prospection뿐 아니라 Retrospection과 Step 2 validation까지 같은 시간축 문제를 공유한다.

---

## 3. 실험 결과가 보여주는 정합 증거

코드만으로도 시간 의미는 확정된다. 아래 성능 차이는 그 결론과 강하게 정합되는 **보조 증거**다.

### GoalStep `start−1s`와 `end−1s`의 관측 차이

`end−1s / 8초 / VNA` 진단 실험에서는 다음이 실측됐다.

- train에서 target action 일부가 관찰되는 비율: **99.832%**
- val에서 target action 일부가 관찰되는 비율: **99.917%**
- Action Top-5, epoch 6 full-val: **50.042%**

출처: [`2026-07-21_goalstep-action-end-m1-lobs8-vna-results.md`](2026-07-21_goalstep-action-end-m1-lobs8-vna-results.md)

반면 target 시작 1초 전에 관찰을 끝내는 기존 `start−1s` 계열 `b2_vna`의 full-val Action Top-5는 **20.68%**였다.

출처: [`2026-07-21_step1_night_and_retro_belief_sum_handoff.md`](2026-07-21_step1_night_and_retro_belief_sum_handoff.md#L85-L96)

| 설정 | target action 관찰 | full-val Action Top-5 |
|---|---:|---:|
| `start−1s` 계열 `b2_vna` | target 시작 전 종료 | 20.68% |
| `end−1s / 8s / VNA`, epoch 6 | 약 99.9% sample에서 target 일부 관찰 | 50.042% |
| 차이 | recognition 단서 추가 | **+29.362%p** |

이 결과는 다음 설명과 일관된다.

> `end−1s`에서 성능이 크게 상승한 이유의 상당 부분은 모델이 미래 action을 더 잘 예측해서가 아니라, 이미 영상에 나타난 target action을 인식할 수 있게 되었기 때문일 가능성이 높다.

### 반드시 붙여야 하는 한계

이 비교는 endpoint만 바꾼 완전 통제실험은 아니다. `end−1s` 실험은 최대 8초 관찰, depth-4 probe 등 일부 조건도 달라졌다. 따라서 **+29.362%p 전체를 endpoint 효과라고 단정할 수는 없다.**

그럼에도 다음 두 사실은 분리해야 한다.

1. `end−1s`가 현재 action 내부라는 사실은 코드와 timestamp 관계로 **확정**된다.
2. `start−1s`보다 `end−1s` 성능이 크게 높은 결과는 recognition shortcut 설명을 **강하게 지지**하지만, endpoint 단독 인과효과의 크기는 별도 통제실험이 필요하다.

---

## 4. Step 1에 대한 해석 수정

### 기존 해석

```text
V-JEPA 기반 world model이 과거 관찰로 다음 action 분포를 예측한다.
```

### 현재 코드에 맞는 해석

```text
V-JEPA 기반 probe가 target action이 이미 진행 중인 late observation을 보고,
현재 annotation action의 verb/noun/action 분포를 예측한다.
```

따라서 EK100 `end−1s` 모델은 world model이라는 외형을 갖고 있더라도, 현재 실험 계약에서는 **future-action model보다 late-action recognition probe에 가깝다.**

이는 모델이 물리적 미래를 전혀 학습하지 않았다는 뜻은 아니다. V-JEPA backbone은 temporal representation을 갖고 있고 probe에도 anticipation parameter가 들어간다. 그러나 현재 label–observation 계약으로 측정된 높은 정확도를 **미래 action 예측 능력의 증거로 사용할 수 없다는 뜻**이다.

---

## 5. Step 2에 대한 해석 수정

현재 Step 2는 다음 구조를 사용한다.

```text
현재 action이 보이는 frame/grid
        +
완료된 과거 action history
        +
현재 action을 강하게 포함하는 WM top-5
        ↓
VLM이 후보 하나 선택
        ↓
동일한 현재 annotation GT와 비교
```

여기서 특히 주의할 점은 VLM이 단순히 중립적인 5지선다 문제를 받는 것이 아니라, 명시적으로
`NEXT action`을 선택하라는 instruction을 받는다는 것이다. 하지만 candidate 생성 시점, GT label,
reward 및 validation은 모두 현재 annotation action에 맞춰져 있다. 결과적으로 현재 Step 2 성능은
“VLM이 next-action instruction을 잘 따랐는가”가 아니라 **“next라고 잘못 명명된 current action을
얼마나 잘 골랐는가”**를 나타낸다.

따라서 Step 2의 `acc`, conditional accuracy, G2/GADR은 현재 구현에서 다음을 뜻한다.

- `acc`: 현재 annotation action을 맞힌 비율
- `GT in top-5`: WM 후보에 현재 annotation action이 포함된 비율
- `G2/GADR`: WM top-1이 현재 annotation GT와 다르지만 GT가 top-5에 있을 때, VLM이 현재 GT를 복구한 비율

이 지표들은 candidate reranking 능력과 context 사용 여부를 분석하는 데는 유효하다. 하지만 **다음 action 선택 능력의 증거는 아니다.**

특히 프롬프트에 `NEXT action`이라고 적혀 있고 논문 수식이 `a_t`를 next action이라고 정의해도, validation GT가 다음 annotation으로 이동하지 않는 한 task semantics는 바뀌지 않는다.

---

## 6. 논문 핵심 claim에 미치는 영향

논문은 world model이 현재 관찰로부터 near-future action support를 제공하고, VLM이 그 안에서 task-consistent next action을 선택한다고 설명한다.

출처: [`main.tex`](../../../EGO_paper/EGO_AAAI27_EN/main.tex#L99-L125)

현재 구현과 논문 claim 사이에는 다음 차이가 있다.

| 논문이 주장하려는 것 | 현재 구현이 실제로 입증하는 것 |
|---|---|
| 현재 관찰에서 미래 action 후보 생성 | 진행 중인 현재 action을 포함한 후보 생성 |
| VLM이 task belief로 next action 선택 | VLM이 현재 action 후보를 재선택 |
| GADR이 WM의 미래 예측 오류를 VLM이 복구 | 현재 action 분류에서 WM top-1 오류를 후보 내에서 복구 |
| single-step anticipation이 planning으로 확장 | current-action recognition 결과를 LLM planning 입력으로 사용 |

따라서 현재 결과를 그대로 사용할 경우 논문의 가장 위험한 문장은 다음 유형이다.

> “EGO가 다음 action을 예측한다”, “world model이 미래 action boundary를 제공한다”, “VLM이 미래 action 중 task-consistent action을 고른다.”

현재 증거가 안전하게 뒷받침하는 표현은 더 제한적이다.

> “EGO가 late observation에서 진행 중인 action의 후보 support를 만들고, VLM이 task context를 이용해 후보를 재선택한다.”

---

## 7. VPA·Planning 결과에 대한 핵심 우려

### 원하는 planning 구조

진짜 action anticipation WM이라면 planning은 다음처럼 작동할 수 있다.

```text
현재 상태
  → WM: 실제로 다음에 가능한 action 후보
  → VLM: task goal에 맞는 다음 action 선택
  → 실행/새 관찰
  → 반복
```

이때 WM은 LLM의 상상 범위를 미래의 물리적으로 가능한 action으로 제한하는 실질적인 predictive prior다.

### 현재 구조

현재 WM이 주로 제공하는 것은 이미 진행 중인 action에 대한 recognition support다.

```text
현재 action이 보이는 상태
  → WM: 현재 action 후보
  → VLM: 현재 action 재식별
  → 그 다음에 무엇을 할지는?
  → LLM의 procedural/common-sense prior에 크게 의존
```

따라서 현재 모듈로 VPA나 multi-step planning을 수행하면:

- WM은 “지금 무엇을 하고 있는가”를 알려주는 state/action recognizer 역할은 할 수 있다.
- 그러나 “다음에 무엇이 실제로 일어날 수 있는가”에 대한 독립적인 미래 제약은 충분히 제공하지 못한다.
- 다음 action과 이후 sequence 생성은 사실상 LLM의 사전학습 지식, action history, prompt logic에 의존한다.
- planning 성능이 나오더라도 그것을 **world-model-grounded future planning의 증거로 귀속하기 어렵다.**
- 더 정확한 해석은 `recognition-assisted LLM planning`에 가깝다.

### 가장 중요한 귀속 문제

> 현재 설계에서 planning이 성공하더라도, 성공 원인이 WM의 미래 예측인지 LLM이 이미 알고 있던 procedural prior인지 분리하기 어렵다.

이 문제는 단순한 metric 명칭 문제가 아니다. 논문의 핵심 기여인 “언어모델의 추론을 독립적인 physical predictive prior로 grounding한다”는 인과적 주장에 직접 영향을 준다.

---

## 8. 현재 결과를 폐기해야 하는가?

전부 폐기할 필요는 없다. 다만 결과의 이름과 claim을 분리해야 한다.

### 유지 가능한 결과

- 현재/진행 action joint top-5 coverage
- 후보 내 VLM reranking 정확도
- WM top-1과 VLM 선택 차이
- action history와 task belief가 현재 action 선택에 주는 영향
- Retrospection이 reasoning–belief–action coherence에 주는 영향
- recognition-conditioned planning baseline

### 그대로 주장하면 안 되는 결과

- 진짜 next-action anticipation accuracy
- future action support coverage
- VLM의 미래 action discrimination 능력
- WM이 planning의 미래 전개를 물리적으로 constrain한다는 주장
- 현재 GADR을 미래 action GADR로 해석하는 것

---

## 9. 권고 조치

### P0 — 논문 제출 전 반드시 할 일

1. 논문 Method에서 decision time을 annotation 경계로 명시한다.

   ```text
   t = target_action_start − anticipation_horizon
   assert observation_end <= t < target_action_start
   ```

2. `end−1s` 결과에는 다음 명칭을 사용한다.

   ```text
   late-action recognition / ongoing-action candidate selection
   ```

3. 진짜 anticipation 결과와 `end−1s` 결과를 같은 표의 동등한 anticipation 수치로 비교하지 않는다.

4. Results와 Conclusion에서 VPA·Planning claim을 실제 timing contract에 맞게 재검토한다.

### P1 — 진짜 next-action Step 1 재구축

권장 기준은 target annotation `j`에 대해 다음과 같다.

```text
target label       = action_j
observation end    = start_j − 1.0s
observation frames <= observation end
history            = actions with stop < observation end
```

이 기준으로 feature 추출부터 probe 학습, full validation을 다시 수행해야 한다.

### P1 — 진짜 next-action Step 2 재구축

Step 2의 모든 입력을 동일 decision time으로 맞춰야 한다.

1. VLM frame/grid를 `start_j−1s` 기준으로 재추출
2. WM top-5를 같은 시점의 clip으로 재추론
3. history는 `stop < start_j−1s`만 포함
4. validation GT는 `action_j`
5. Retrospection chosen action도 `action_j`
6. 이후 hindsight sequence는 `j+1, j+2, ...`
7. 모든 sample에 다음 assertion 저장

   ```python
   assert max(observed_frame_time) < target_action_start
   ```

### P2 — Planning 기여를 분리하는 ablation

| 조건 | 목적 |
|---|---|
| LLM only | procedural prior 기준선 |
| LLM + current-action recognizer | 현재 구현의 실제 기여 |
| LLM + true anticipation WM | 미래 WM의 추가 기여 |
| shuffled/unrelated WM candidates | 후보가 실제 물리 정보를 제공하는지 검증 |
| oracle future top-5 | candidate coverage 상한 |

Planning claim은 최소한 `LLM + true anticipation WM`이 `LLM + current-action recognizer`를 유의미하게 이겨야 강하게 주장할 수 있다.

### P2 — endpoint 단독 통제실험

현재 성능 차이의 인과효과를 분리하려면 다음 항목을 모두 고정해야 한다.

- 동일 train/val sample 및 label
- 동일 8초 관찰 길이와 frame 수
- 동일 backbone feature preprocessing
- 동일 probe 구조와 초기화 seed
- 동일 sampler, loss, epoch, checkpoint selection
- 변경 변수는 endpoint만 `start−1s` 대 `end−1s`

이 실험을 통해서만 recognition shortcut이 몇 %p를 설명하는지 정량적으로 말할 수 있다.

---

## 10. 논문 작성 시 권장 문구

### 현재 상태를 정직하게 기술하는 문구

> The released EK100-aligned implementation places the observation endpoint one second before the end of the target action segment. Because the target action may already be underway at this point, we treat this setting as late-action recognition rather than strict action anticipation.

### 사용하면 안 되는 문구

> The model predicts an unseen next action one second before it begins.

현재 코드와 실험으로는 이 문장을 뒷받침할 수 없다.

### 진짜 `start−1s` 재실험 후 사용할 수 있는 문구

> All observed frames precede the start of the target action by at least one second, and evaluation compares the prediction against that unseen future action.

---

## 11. 최종 결론

> **현재 validation은 실제 GT와 비교하므로 채점이 틀린 것은 아니다. 그러나 그 GT가 미래의 다음 action이 아니라 trigger가 속한 현재 annotation action이다.**

따라서 현재 EGO 파이프라인의 가장 정확한 진단은 다음과 같다.

1. EK100 `end−1s` Step 1은 strict action anticipation보다 **late-action recognition**에 가깝다.
2. Step 2 VLM도 다음 action이 아니라 **현재 annotation action을 top-5에서 선택**하도록 검증된다.
3. `start−1s`에서 성능이 크게 낮고, target이 보이는 `end−1s`에서 크게 높아진 결과는 이 해석을 강하게 지지한다.
4. 현재 WM은 planning에서 현재 행동/상태 인식기는 될 수 있지만, 다음 행동을 제한하는 독립적인 future prior 역할은 충분히 입증되지 않았다.
5. 따라서 현재 설정으로 얻은 VPA·Planning 성능은 **온전히 world-model-grounded planning으로 귀속할 수 없으며, 상당 부분 LLM의 기존 procedural reasoning에 의존할 가능성이 높다.**
6. 논문의 본래 목표를 유지하려면 Step 1과 Step 2 모두 `target_start−1s` 기준으로 다시 정렬하고, feature·candidate·validation·Retrospection pair를 재생성해야 한다.

이 문제는 모델의 성능이 낮고 높은 문제보다 먼저 해결해야 한다. 현재 가장 중요한 것은 숫자를 더 높이는 것이 아니라, **그 숫자가 실제로 어떤 task를 측정하는지 정확히 맞추는 것**이다.
