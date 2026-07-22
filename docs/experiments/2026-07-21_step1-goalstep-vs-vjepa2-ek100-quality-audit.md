# Step-1 GoalStep vs. V-JEPA2/2.1 EK100 품질 감사

- 작성일: 2026-07-21
- 대상: `EGO_jihun2`의 GoalStep Z=1 action anticipation
- 비교 기준: Meta 공식 V-JEPA2 저장소 commit `204698b45b3712590f06245fbfba32d3be539812f`, V-JEPA2 및 V-JEPA2.1 논문, EK100 공식 annotation
- 현재 주 run: `outputs/goalstep/runs/z1_jihun2` (action-only, scenario-stratified, depth 4)
- 원인 분해 기준 run: `outputs/goalstep/night/b2_vna` (V/N/A, random, depth 1)

## 1. 결론부터

현재 GoalStep 성능이 EK100에서 보았던 수치보다 낮은 이유는 하나가 아니다. 우선순위는 다음과 같다.

| 우선순위 | 원인 | 판정 | 예상 영향/근거 |
|---|---|---|---|
| P0 | 서로 다른 지표를 같은 숫자로 비교 | 확정 | V-JEPA2/2.1 논문의 EK100 수치는 micro Top-5 accuracy가 아니라 **mean-class recall@5**이다. 사용자가 기억한 action 60%+/verb·noun 90% 근처 수치는 다른 로그의 micro accuracy일 가능성이 높다. |
| P0 | 배포 체크포인트 선택 기준이 목표 지표와 다름 | 확정, 즉시 회수 가능 | `best.pt`는 action CMR@5로 골라 epoch 14를 저장했다. 그러나 epoch 3 full-val action micro Top-5는 25.326%, epoch 14는 20.682%다. 현재 선택만으로 **4.644%p**를 잃었다. |
| P0 | 단 한 번 추출한 고정 view feature를 15 epoch 재사용 | 확정된 구현 차이, 영향 매우 유력 | 공식 학습은 매 epoch random crop/flip/RandAugment/random erasing 및 시간 위치 랜덤화를 적용한다. 현재 train/val 모두 center crop으로 한 번 추출하고 FP16으로 고정한다. 후반 train Top-5 100%, val 하락과 정확히 부합한다. |
| P0 | GoalStep 표적의 시간·의미 단위와 4초 미만 문맥의 부조화 | 데이터로 확인, 영향 매우 유력 | GoalStep target median은 step 21.45초, substep 9.87초로 EK100의 짧은 atomic action보다 훨씬 길다. 다음 recipe step은 직전 3.5~4초보다 장기 진행 상태가 중요하다. |
| P0 | 주 action-only config/run에 결함 sampler가 여전히 적용됨 | 확정 | `z1_jihun2.yaml`과 완료 run metadata 모두 `scenario_stratified`다. random 정상화는 `b2_vna` 등 비교 실험에만 적용됐다. random epoch 3이 기존 15-epoch 최고보다 Top-5 +7.85%p였다. |
| P1 | 공식 probe 탐색을 사실상 1개 설정으로 축소 | 확정 | 공식은 LR 5개 × WD 4개 = 20개 probe를 동시에 학습하고 head/metric별 최고를 보고한다. 현재 핵심 run은 LR/WD 한 조합, depth 1이다. |
| P1 | 모델 크기·해상도 차이 | 확정 | 현재는 V-JEPA2 ViT-L/16 256이고 `use_v2_1: false`다. V-JEPA2.1 최고 표는 ViT-G 2B/384다. 단, 2.1이라는 이름 자체보다 모델 규모·해상도 효과가 크다. |
| P1 | 공식 코드와 논문의 observation endpoint 불일치 | 확정된 재현 모호성 | 논문은 action **start 1초 전** 종료라고 하지만 공개 loader HEAD는 action **end에서 1초 전**을 계산한다. 그대로면 validation action의 상당수에서 action이 이미 시작된 쉬운 조건이 될 수 있다. 정확한 비교 전에 반드시 해소해야 한다. |
| P2 | 자동 생성 taxonomy와 step/substep 혼합 | 확정된 설계 차이, 영향 중간 | GoalStep 문장에서 spaCy+규칙으로 V/N/A를 생성했으며 서로 다른 추상화 수준을 한 action space에 섞었다. EK100은 사람이 부여한 atomic verb/noun annotation이다. |
| P2 | V/N 보조 loss 부재 또는 fusion | 주원인 아님 | 현재 V/N/A run이 action-only보다 약 1%p 좋았고 soft fusion은 약 0.6%p 수준이었다. 보조 신호는 유익하지만 25%→60%를 만들 해결책은 아니다. |
| 제외 | 누락 영상, timestamp 오류, backbone weight 미로딩, token shape 오류, BF16 자체 | 현재 증거상 주원인 아님 | 700개 참조 영상 전부 존재·decode되고 timestamp 범위가 맞으며 encoder/predictor load mismatch는 0이다. BF16 평가가 아니라 학습만 BF16인 것도 성능 급락의 설명이 되기 어렵다. |

따라서 바로 대형 모델로 갈 것이 아니라, **평가/체크포인트 기준 교정 → online 또는 multi-view feature 학습 → 시간 위치 및 장기 문맥 실험 → probe 탐색 → 모델 확대** 순으로 가야 한다.

## 2. 먼저 바로잡아야 하는 숫자: Top-5 accuracy와 mean-class recall@5

V-JEPA2와 V-JEPA2.1 논문의 EK100 표는 verb/noun/action 모두 **mean-class recall-at-5**이다. 클래스 빈도와 무관하게 클래스별 recall을 평균한다. 반면 일반적인 `top5 accuracy`는 모든 validation sample을 동일 가중치로 세는 micro accuracy다. 둘은 특히 long-tail 데이터에서 큰 차이가 날 수 있다.

공식 발표 수치는 다음과 같다.

| 모델 | 해상도 | EK100 Verb R@5 | Noun R@5 | Action R@5 |
|---|---:|---:|---:|---:|
| V-JEPA2 ViT-L/16 | 256 | - | - | 32.7 |
| V-JEPA2 ViT-g/16 1B | 384 | 63.6 | 57.1 | 39.7 |
| V-JEPA2.1 ViT-g/16 1B | 384 | 63.6 | 56.2 | 38.4 |
| V-JEPA2.1 ViT-G/16 2B | 384 | 64.3 | 59.9 | 40.8 |

즉 공식 논문 자체는 action R@5 60% 이상, V/N 90% 근처를 주장하지 않는다. 그런 수치가 실제 공식 실행 로그에 있었다면 `accuracy` 필드, 특정 split/subset, 혹은 서로 다른 probe의 maximum일 가능성이 높다. 정확한 로그 파일을 확보하기 전에는 현재 GoalStep CMR과 직접 비교하면 안 된다.

현재 주 action-only run과 원인 분해용 `b2_vna`의 결과는 다음과 같다.

| run/체크포인트 | 선택/측정 방식 | Verb Top-5 | Noun Top-5 | Action Top-5 | Action CMR@5 |
|---|---|---:|---:|---:|---:|
| `z1_jihun2` epoch 9 `best.pt` | scenario-stratified, subset CMR 선택 후 full val | - | - | **17.757** | 7.966 |
| `b2_vna` epoch 3 | random, 이 감사에서 full val 재평가 | 52.814 | 53.119 | **25.326** | 9.828 |
| `b2_vna` epoch 14 (`best.pt`) | random, subset action CMR@5 최고 후 full val | 42.667 | 43.637 | **20.682** | **11.419** |

해석은 명확하다.

- `b2_vna` 안에서 micro Top-5가 목적이면 epoch 3가 현재 최고 후보다.
- class-balanced 성능이 목적이면 epoch 14가 더 낫다.
- 한 파일을 `best.pt`라고 부르면서 두 목적을 동시에 만족시킬 수 없다.
- 기존 문서의 “subset 25.75 vs full 20.68, subset이 약 5%p 낙관적” 비교는 서로 다른 epoch를 비교한 것이었다. 같은 epoch 3에서는 subset 25.75 vs full 25.326으로 차이가 약 0.424%p다. 이 해석은 정정해야 한다.

공식 evaluator도 micro `accuracy`와 class-mean `recall`을 모두 계산한다. 더 나아가 20개 probe 중 metric/head별 maximum을 각각 취하므로, 공식 출력의 verb·noun·action 최고값이 반드시 하나의 동일한 probe에서 나온 것도 아니다. 공정 비교에는 단일 체크포인트와 metric 정의를 함께 기록해야 한다.

## 3. 공식 EK100과 현재 GoalStep의 end-to-end 비교

### 3.1 데이터와 표적 정의

| 항목 | 공식 EK100 | 현재 GoalStep |
|---|---|---|
| 표적 | 짧은 atomic action의 `(verb, noun)` | 자연어 `step_category` 및 substep을 자동 V/N으로 파싱한 action |
| train/val 규모 | 67,217 / 9,668 annotation | 30,374 / 7,214 sample |
| action class | train pair 3,568개; val 중 train-known pair로 평가 | 293개, val에 270개 등장 |
| train class 빈도 | median 3, min 1, max 1,784 | median 47, min 9, max 855 |
| target duration | train median 약 1.43초, val 약 1.95초 | step median 21.45초, substep median 9.87초 |
| annotation | EPIC-KITCHENS 사람이 부여한 verb/noun/action | spaCy+규칙+작은 synonym map으로 생성 |
| hierarchy | atomic action | step과 substep을 동시에 학습 |
| split overlap | participant/video 기반 공식 split | train/val video overlap 0 |

GoalStep은 class 수가 적고 최소 빈도가 더 높다. validation majority prior의 micro Top-5도 GoalStep 14.17%, EK100 약 9.02%다. 따라서 현재 저성능을 “GoalStep 클래스가 너무 많고 tail이 심해서”라고 설명할 수 없다. 더 가능성 높은 설명은 **label의 예측 가능성, 시간 경계, 필요한 문맥 길이, 학습 regularization**이다.

GoalStep taxonomy의 주요 특성은 다음과 같다.

- raw 39,262 segment 중 157개가 `OTHER`, 713개가 pruning 과정에서 제외됐고 최종 38,392개가 매핑됐다.
- 최종 train은 step 13,062개 + substep 17,973개, val은 step 3,196개 + substep 4,161개다.
- 동일 observation window에 상충 label이 붙은 경우는 train 28개(0.092%), val 10개(0.139%)로 작다. 이것이 주원인은 아니다.
- step action pair 287개, substep pair 289개, 공통 283개라 label ID 자체는 많이 겹친다. 그러나 같은 `(verb,noun)`이라도 step과 substep에서 시간적 의미와 난도가 다를 수 있다.
- train-val action 분포의 total variation은 약 0.158, JS divergence는 약 0.043 bit로 moderate shift다. 293개 중 23개는 val에 없다.

### 3.2 Observation window와 anticipation 정의

| 항목 | 공식 EK100 config | 현재 GoalStep `b2_vna` |
|---|---|---|
| frames | 32 | 32 |
| nominal fps | 8 | 8 |
| nominal context | 약 4초 | index 실측 약 3.5초; `l_obs: 3.875`는 기존 cache window를 바꾸지 않음 |
| validation horizon | 1초 | 1초 |
| train horizon | 0.25~1.75초 random | 1초 고정 |
| train target 위치 | `anticipation_point` 0~0.25 random | target start 기준 고정 |
| temporal augmentation | 매 sample/epoch 랜덤 | 없음, sample마다 하나의 고정 feature |

현재 32개 frame은 3.5초 양 끝점을 포함해 뽑혀 약 8.86fps이며, lobs4 cache는 4초 양 끝점 기준 약 7.75fps다. 한편 공식 loader도 30fps 영상에서 `int(30/8)=3` stride를 써 실효 약 10fps가 될 수 있어 nominal fps와 완전히 일치하지 않는다. 실제 `f_fps` 실험은 Top-5 24.8→24.0, CMR 11.67→13.40으로 metric 간 trade-off만 보였으므로 fps 오차는 현재 micro Top-5 부족의 1순위 원인이 아니다.

더 중요한 문제는 endpoint다. V-JEPA2.1 논문은 clip이 action **start 1초 전** 끝난다고 설명한다. 하지만 감사한 공개 source HEAD의 EK100 loader는 대략 다음 계산을 한다.

```text
anticipation_frame = start * anticipation_point
                   + (1 - anticipation_point) * end
                   - anticipation_frames
```

validation 기본 `anticipation_point=0`이면 `action_end - 1초`가 된다. 공식 EK100 validation annotation의 약 82.6%는 action duration이 1초보다 길어서, 이 코드 그대로라면 target action이 이미 진행 중인 frame을 observation에 포함할 수 있다. 이것은 진짜 anticipation보다 recognition에 가까워져 성능을 크게 올릴 수 있다.

단, 이 불일치를 근거로 “공식 발표가 leakage다”라고 단정해서는 안 된다. 논문 실험에 사용된 내부 버전, released commit, annotation timestamp 해석이 다를 수 있다. 정확한 공식 재현 시 다음을 반드시 해야 한다.

1. 논문 정의인 `action_start - 1.0s`와 공개 코드식 `action_end - 1.0s`를 별도 config로 명시한다.
2. 50개 이상 clip을 frame/time overlay와 함께 육안 감사한다.
3. 두 endpoint 결과를 함께 보고하되 `end-1s` 결과를 GoalStep start-based 결과와 직접 비교하지 않는다.

### 3.3 영상 전처리와 feature 추출

| 항목 | 공식 EK100 | 현재 GoalStep |
|---|---|---|
| backbone 실행 | 학습 중 online | 학습 전 1회 offline cache |
| train spatial crop | random resized crop, scale 0.08~1.0 | deterministic evaluation/center crop |
| flip | random horizontal flip | 없음 |
| appearance augmentation | RandAugment | 없음 |
| random erasing | 0.25 | 없음 |
| val crop | resize/center crop | resize/center crop |
| cache dtype | 해당 없음; online BF16 | token을 FP16으로 저장, load 시 FP32 |
| train view 수/sample | epoch마다 달라짐 | 전체 15 epoch 동안 정확히 1개 |

현재 추출 script는 train과 val 모두 아래처럼 `training=False`를 강제한다.

```python
transform = build_transform(training=False, crop_size=resolution, ...)
```

그 뒤 `[4352, 1024]` token을 FP16으로 저장한다. 4,352는 encoder 4,096 token + predictor 256 token으로 현재 V-JEPA2 ViT-L/256 wrapper 구조에 맞는다. 따라서 token 개수 오류는 아니다.

문제는 FP16 정밀도보다 **고정 view**다. frozen backbone이라도 augmentation을 backbone 앞에서 하면 probe가 매 epoch 다른 feature를 보며 regularization을 얻는다. 현재는 같은 30,374개 tensor를 반복해서 probe에 주므로 작은 transformer head가 이를 암기하기 쉽다. 실제 후반 train V/N/A Top-5가 100%인데 full-val action Top-5는 20.68%인 것은 이 설명과 강하게 일치한다.

FP16 cache가 독립적으로 얼마나 해로운지는 아직 A/B되지 않았다. 저장 양자화 오차는 보통 고정-view 손실보다 작을 것으로 예상하지만, 추측으로 끝내지 말고 동일 1,000 sample의 FP16-cache, BF16-cache, FP32-online output을 비교해야 한다.

### 3.4 Backbone과 anticipation wrapper

| 항목 | 공식 V-JEPA2 ViT-L EK100 | 현재 GoalStep |
|---|---|---|
| 계열 | V-JEPA2 | V-JEPA2 (`use_v2_1: false`) |
| encoder | ViT-L/16, target encoder | 동일 |
| resolution | 256 | 256 |
| predictor | depth 12, 10 mask tokens | 동일 |
| output | encoder token + predicted future token concat | 동일 |
| output frames / steps | 2 / 1 | 2 / 1 |
| checkpoint load | 공식 checkpoint | encoder missing/mismatch 0, predictor missing/mismatch 0 |

현재 `[4352,1024]` shape, encoder/predictor parameter load, predictor concatenation은 공식 ViT-L wrapper와 구조적으로 맞는다. 즉 “V-JEPA predictor를 아예 안 썼다”거나 “checkpoint가 로드되지 않았다”는 설명은 배제된다.

하지만 현재 모델은 V-JEPA2.1이 아니다. V-JEPA2.1 최고 40.8 R@5와 비교하려면 최소한 같은 384 해상도와 ViT-G 2B 조건이 필요하다. 또한 표에서 V-JEPA2.1 ViT-g 1B action 38.4가 V-JEPA2 ViT-g 1B의 39.7보다 낮다. 그러므로 2.1로 이름만 바꾸는 것이 보장된 향상은 아니며, 최고 결과의 일부는 2B scaling에서 온다.

### 3.5 Probe, loss, optimizer, schedule

| 항목 | 공식 EK100 | 주 `z1_jihun2` | 진단 `b2_vna` |
|---|---|---|---|
| probe depth | 4 blocks | 4 blocks | 1 block |
| heads | V/N/A | action only | V/N/A |
| objective | 세 focal loss 합 | action focal loss만 | 세 focal loss 합 |
| focal | alpha 0.25, gamma 2 | 동일 | 동일 |
| LR | 1e-4, 3e-4, 1e-3, 3e-3, 5e-3 | 3e-4 하나 | 3e-4 하나 |
| WD | 1e-4, 1e-3, 1e-2, 1e-1 | 1e-4 하나 | 1e-4 하나 |
| probe 수 | 20개 동시 학습 | 1개 | 1개 |
| epochs | 20 | 15 | 15 |
| global batch | 8 nodes × 8 tasks × local 2 = 128 | 32 | 32 |
| start LR / warmup | ref LR / 0 epoch | 0 / 1 epoch | 0 / 1 epoch |
| train precision | BF16 | BF16 autocast | BF16 autocast |
| val precision | BF16 | FP32 | FP32 |
| sampler | 전체 annotation을 distributed shuffle | **scenario-stratified** | random |

depth 1→4 단독 action-only 실험(`b3_d4`)은 Top-5 24.95로 depth 1의 24.80보다 소폭 높아 depth만으로 격차를 설명하지 못한다. 그러나 공식과 같은 **V/N/A + depth 4 + augmentation + hyperparameter grid** 조합은 아직 시험하지 않았다.

공식 focal loss는 class와 batch를 합산하는 `sum` reduction이고, 현재 구현은 class 합 후 batch 평균이다. Adam 계열은 일정한 gradient scale에 비교적 둔감하지만 weight decay, clipping, scheduler와 결합하면 완전히 동등하지는 않다. 공식 parity 실험에서는 reduction도 맞추는 편이 안전하다.

현재 validation FP32 유지는 오히려 보수적이며 성능 저하의 원인이 아니다. BF16 학습 자체도 공식이 사용하므로 주요 혐의가 아니다.

### 3.6 Sampler

기존 `scenario_stratified`는 작은 scenario를 최대 138.7회 반복하고 큰 scenario sample의 9%만 보는 심각한 왜곡이었다. `random`으로 바꾼 뒤 epoch 3 action Top-5가 16.4→24.8, CMR@5가 9.06→10.64로 개선됐다.

`b2_vna`는 `sampler: random`이다. 따라서 그 run에서는 각 epoch이 cache의 모든 train sample을 한 번씩 무작위 순서로 보고, scenario별 총 노출은 자연 sample 수에 비례하며 sample당 기대 노출은 1회다.

그러나 **현재 주 설정은 아직 정상화되지 않았다.** [`z1_jihun2.yaml`](../../configs/step1/goalstep/z1_jihun2.yaml)과 완료된 `z1_jihun2/run_metadata.json` 모두 `scenario_stratified`다. 즉 random은 실험 옵션/진단 run에만 남아 있고 기본 action-only run에는 반영되지 않았다. 다음 본 학습 전 주 config를 `random`으로 바꿔야 하며, 기존 `z1_jihun2` 결과는 sampler 결함이 있는 결과로 표기해야 한다.

### 3.7 V/N/A auxiliary supervision과 action 결합

공식 EK100은 verb, noun, action 세 head의 focal loss를 모두 사용한다. 현재도 `b2_vna`에서 그렇게 했다. 기존 비교에서는 V/N/A 학습이 action-only보다 action Top-5를 약 1%p 올렸지만 큰 폭의 향상은 아니었다.

V/N prediction으로 action을 재구성한 실험도 이미 다음 결론을 보였다.

- verb top-5 × noun top-5의 hard Cartesian matching은 action Top-5를 낮췄다.
- V/N 확률을 action logit에 soft하게 더한 fusion은 약 0.6%p 개선에 그쳤다.
- 원인은 V와 N의 marginal top-5가 맞더라도 정확한 pair의 joint ranking을 보장하지 않기 때문이다.

따라서 V/N auxiliary supervision은 유지할 가치가 있지만, 60% 수준을 만들 핵심은 feature/context/generalization이다. fusion은 그 뒤의 calibration 단계다.

## 4. 데이터·feature 무결성 감사

다음 항목은 실제 파일과 cache를 검사했다.

| 검사 | 결과 | 판정 |
|---|---|---|
| index 수 | train 30,374 / val 7,214 | 정상 |
| 참조 영상 | 700개 모두 존재; 폴더에는 총 718개 | 정상 |
| video split overlap | 0 | 정상 |
| decode/fps | 700개 모두 decode, 모두 정확히 30fps | 정상 |
| timestamp 범위 | annotation이 video duration을 넘는 항목 0; 최소 여유 2.08초 | 정상 |
| cache 수 | train 30,374 / val 7,214 | 정상 |
| cache tensor | `[4352,1024]`, NaN/Inf 없음 | 정상 |
| dtype | disk FP16, load FP32 | 설계상 정상, 정밀도 A/B 필요 |
| checkpoint load | encoder/predictor missing 0, mismatch 0 | 정상 |
| 해상도 분포 | 540×720: 413, 1440×1920: 172, 540×960: 114, 기타 1 | 다양한 aspect ratio라 random crop 부재 영향 가능 |

즉 “파일을 잘못 읽었다”, “영상이 누락됐다”, “timestamp가 영상 밖이다”, “backbone이 random weight다” 같은 치명적 파이프라인 오류는 발견되지 않았다.

## 5. 왜 GoalStep은 EK100보다 본질적으로 어려울 수 있는가

### 5.1 관찰 길이가 표적 진행 상태를 담지 못한다

GoalStep의 target 간 start gap median은 train 약 11.43초, val 약 11.51초다. 4초 이하인 경우는 약 21~23%뿐이다. 즉 대부분 sample에서 다음 step을 결정하는 직전 step의 핵심 장면이 3.5~4초 observation 밖으로 사라질 수 있다.

예를 들어 “반죽을 섞는다” 다음 “오븐에 넣는다”를 예측하려면 현재 손동작뿐 아니라 재료 상태, recipe 목표, 몇 단계까지 수행했는지 알아야 한다. EK100 atomic anticipation보다 state/history dependence가 훨씬 크다.

해결은 단순히 32 frame을 8초에 균일 샘플하는 것만으로 충분하지 않을 수 있다. 다음 두 스트림이 적합하다.

- short stream: target start 직전 4초, 32 frame
- history stream: 이전 30~60초에서 sparse clip 여러 개 또는 이전 step embedding

두 stream을 probe에서 cross-attention 또는 pooled concatenation으로 합치고, scenario/goal embedding도 보조 조건으로 넣는다.

### 5.2 step와 substep의 예측 시점이 동일하지 않다

step과 substep을 모두 target start 1초 전에 예측하지만, 두 annotation의 추상도와 경계 기준은 다르다. 같은 `(verb,noun)` class라도 step-level에서는 큰 절차 전환을, substep-level에서는 즉각적인 손동작을 의미할 수 있다.

최소한 다음 세 평가를 분리해야 한다.

1. step-only train/eval
2. substep-only train/eval
3. joint train, level별 eval 및 level embedding 추가

분리 결과 substep이 높고 step이 낮으면 장기 문맥 문제이고, 둘 다 낮으면 feature/augmentation/taxonomy 문제일 가능성이 높다.

### 5.3 자동 taxonomy는 EK100 label과 동등하지 않다

GoalStep의 verb/noun은 문장 parser 산출물이다. 표면적으로 class 수와 빈도가 좋아도 다음 noise가 존재할 수 있다.

- 목적어 생략 또는 암묵적 object
- 동일 절차를 다른 동사로 표현
- `step_category`의 추상 명사와 화면의 실제 물체가 불일치
- pruned class에 여러 시각적으로 다른 행동이 합쳐짐
- action 시작 경계가 실제 준비 동작보다 늦거나 빠름

해결은 빈도 통계가 아니라 stratified manual audit이다. train/val, step/substep, high/low confidence, high-frequency/rare class별로 최소 50개씩 영상과 label을 함께 검수하고 noise matrix를 만들어야 한다.

## 6. 해결 계획과 판별 가능한 실험

### Stage 0 — 재학습 없이 즉시 교정

1. `best_top5.pt`, `best_cmr.pt`, `best_loss.pt`를 별도로 저장한다.
2. 기존 `b2_vna/checkpoints/epoch_03.pt`를 micro Top-5 후보로 지정하고 full val 결과를 공식 artifact로 저장한다.
3. 모든 표에 `metric`, `split`, `subset_size`, `epoch`, `checkpoint selection metric`을 필수 기록한다.
4. epoch 3 subset/full 비교로 기존 “subset +5%p” 해석을 정정한다.

기대 효과: 새 학습 없이 action micro Top-5 20.682→25.326, **+4.644%p** 회수.

### Stage 1 — pipeline 원인 분해

#### A. Backbone parity

동일한 raw clip과 anticipation time 100~1,000개에 대해 다음을 비교한다.

- 공식 wrapper output vs 현재 wrapper output
- FP32 online vs BF16 online vs FP16 cache reload
- encoder-only vs predictor-only vs encoder+predictor concat

동일 precision에서 공식/현재 wrapper의 max/mean absolute error와 cosine similarity를 기록한다. 구조 parity는 보이지만 수치 parity를 확인해야 한다.

#### B. Recognition–anticipation horizon curve

같은 split에 대해 observation end를 target start 기준으로 다음처럼 바꾼다.

- `start + 0s`: recognition에 가까운 상한/진단용
- `start - 0.25s`
- `start - 0.5s`
- `start - 1.0s`: 본 benchmark
- `end - 1.0s`: 공개 EK loader 해석 재현용 진단; 정식 benchmark로 사용 금지

`start+0`에서도 낮으면 taxonomy/feature 문제가 크다. `start+0`은 높고 `start-1`만 낮으면 anticipation에 필요한 precondition/history가 부족한 것이다. 이 실험은 원인을 가장 빨리 분리한다.

#### C. Label level 분해

step-only, substep-only, joint+level embedding을 같은 data budget과 seed로 비교한다. class 수 변화가 있으므로 micro Top-5뿐 아니라 CMR@5, top-1, majority/random prior를 함께 낸다.

### Stage 2 — 공식 학습 recipe에 가까운 probe

권장 baseline은 다음과 같다.

```yaml
sampler: random
train_heads: [verb, noun, action]
num_probe_blocks: 4
epochs: 20
precision: bf16
validation_precision: fp32
train_horizon_sec: [0.25, 1.75]
train_target_offset_fraction: [0.0, 0.25]
augmentation:
  random_resized_crop: [0.08, 1.0]
  horizontal_flip: true
  randaugment: true
  random_erasing: 0.25
```

LR `{1e-4, 3e-4, 1e-3, 3e-3, 5e-3}` × WD `{1e-4, 1e-3, 1e-2, 1e-1}`를 최소 6 epoch 먼저 돌리고 상위 설정만 20 epoch까지 연장한다. 현재 epoch 3 부근에서 micro Top-5가 정점이므로 early screening이 합리적이다. seed 3개 평균과 표준편차를 기록한다.

가장 중요한 구현 선택은 online augmentation이다.

- 최선: video decode + frozen backbone을 online으로 실행한다.
- 계산비 절충: sample당 K=4~8 random spatial/temporal view를 미리 cache하고 epoch마다 하나를 고른다.
- 더 나은 절충: encoder context를 여러 view로 cache하되 predictor의 horizon-dependent 부분은 online으로 유지한다.
- 피해야 할 것: train sample당 center-crop FP16 tensor 하나를 모든 epoch에 재사용.

global batch는 가능하면 공식 128에 맞춘다. 32를 유지하면 linear LR scaling을 맹신하지 말고 grid로 다시 고른다. focal reduction도 parity run에서는 공식 `sum`과 맞춘다.

### Stage 3 — GoalStep에 맞는 장기 문맥

공식 EK recipe를 맞춰도 recognition–anticipation curve에서 `start-1s`가 크게 낮으면 다음으로 간다.

1. short 4초 + history 30초 multi-clip
2. short 4초 + history 60초 multi-clip
3. 이전 step/substep label 또는 learned state token 추가
4. scenario/goal 조건 추가
5. 시간 간격(`time since previous step`, target duration prior)을 metadata embedding으로 추가

이때 미래 frame, target text, action-end 이후 frame이 들어가지 않도록 sample-level leakage test를 자동화한다.

### Stage 4 — 모델 확대와 fusion

파이프라인이 검증된 뒤 다음 순서로 확장한다.

1. V-JEPA2 ViT-L/256 현재 모델
2. 동일 recipe에서 384 해상도 가능한 모델
3. V-JEPA2/2.1 ViT-g 1B/384
4. 자원이 충분할 때 V-JEPA2.1 ViT-G 2B/384

V/N soft fusion은 각 모델의 temperature를 validation에서 calibration한 후 마지막 0~1%p 개선 수단으로 사용한다. hard top-5 Cartesian matching은 기본값으로 쓰지 않는다.

## 7. 최소 실험 행렬

| ID | 변경 | 묻는 질문 | 우선순위 |
|---|---|---|---|
| E0 | epoch 3 full-val artifact, best-top5 분리 | 선택 기준만으로 잃은 수치는 얼마인가 | 즉시 |
| E1 | `start+0/-0.25/-0.5/-1/end-1` curve | task가 recognition은 가능한가, endpoint가 공정한가 | 최우선 |
| E2 | fixed cache vs K-view cache vs online | 과적합의 핵심이 고정 feature인가 | 최우선 |
| E3 | step-only/substep-only/joint | 어느 annotation level이 병목인가 | 최우선 |
| E4 | VNA depth4 + LR/WD grid | 공식 probe recipe 부족이 얼마인가 | 높음 |
| E5 | 4초 vs 4+30초 vs 4+60초 | 장기 진행 상태가 필요한가 | 높음 |
| E6 | FP16 cache vs BF16/FP32 | 양자화가 독립적으로 해로운가 | 중간 |
| E7 | ViT-L256 vs g/G384 | 파이프라인 고정 후 모델 scaling 효과는 얼마인가 | 마지막 |

성공/중단 기준도 사전에 고정한다.

- primary metric이 micro Top-5면 그 기준으로 checkpoint를 고르고 CMR@5는 secondary로 보고한다. 반대도 동일하다.
- subset은 same-epoch full-val과 1~2%p 안에서 일관되어야 한다.
- 공식/현재 wrapper parity는 동일 precision 입력에서 cosine similarity가 사실상 1에 가까워야 한다.
- online/K-view에서 train–val gap과 peak epoch가 유의하게 개선되지 않으면 augmentation 가설을 기각한다.
- 4+30/60초가 4초보다 seed 평균 기준 유의하게 낫지 않으면 장기 문맥 확대를 중단한다.
- 모든 핵심 비교는 최소 3 seed와 평균±표준편차로 보고한다.

## 8. 구현상 권고 사항

학습 코드에는 다음 metadata를 강제로 저장하는 편이 좋다.

```json
{
  "metric_definition": "micro_top5 | class_mean_recall_at_5",
  "checkpoint_selection_metric": "action_micro_top5",
  "observation_endpoint": "target_start_minus_1s",
  "train_view_policy": "online_random | k_view_cache | fixed_center",
  "cache_dtype": "fp16",
  "backbone_family": "vjepa2",
  "backbone_size": "vit_large",
  "resolution": 256,
  "use_v2_1": false,
  "label_level": "step | substep | joint",
  "sampler": "random"
}
```

또한 한 epoch마다 `latest.pt`, `epoch_XX.pt` 외에 metric별 best를 독립 저장해야 한다.

- `best_action_top5.pt`
- `best_action_cmr5.pt`
- `best_verb_top5.pt`
- `best_noun_top5.pt`

현재처럼 action CMR 하나로 고른 checkpoint에 모든 head의 최종 수치를 붙이면 모델 선택과 보고 목적이 뒤섞인다.

## 9. 최종 판단

현재 결과가 낮은 것을 V-JEPA feature 자체의 실패로 결론 내릴 근거는 없다. backbone weight와 token 구조는 정상이고, random sampler 진단 run에서 이미 큰 폭의 개선이 있었다. 가장 강한 설명은 다음 조합이다.

1. EK100과 다른 metric/protocol 숫자를 비교했다.
2. micro Top-5가 목표인데 CMR 기준 epoch 14를 최종 모델로 선택해 4.644%p를 버렸다.
3. 결함 sampler가 주 action-only config/run에는 여전히 남아 있다.
4. 공식의 강한 online augmentation과 시간 랜덤화를 제거하고 고정 feature 하나를 반복해 probe가 암기했다.
5. GoalStep은 EK100보다 훨씬 긴 procedural target인데 short context만 사용했다.
6. 공식 20-probe 탐색 대신 단일 hyperparameter probe를 사용했다.
7. 최고 V-JEPA2.1 결과보다 작은 V-JEPA2 ViT-L/256 모델을 사용했다.

probe depth가 1인 것은 `b2_vna`에 해당한다. 주 action-only `z1_jihun2`는 depth 4지만 V/N/A 보조 loss가 없고 scenario-stratified sampler를 사용했다. 두 run의 차이를 섞어서 한 가지 원인으로 해석하면 안 된다.

가장 먼저 해야 할 실험은 대형 모델 교체가 아니라 **E0~E4**다. 특히 recognition–anticipation horizon curve와 online/K-view feature A/B가 원인을 빠르게 분리한다. 이 두 실험 없이 모델만 키우면 계산량을 크게 쓰고도 데이터·시간 정의 문제를 그대로 유지할 위험이 높다.

## 10. 근거 파일과 외부 자료

### 현재 저장소

- GoalStep VNA config: [`configs/step1/goalstep/night/b2_vna.yaml`](../../configs/step1/goalstep/night/b2_vna.yaml)
- 현재 action-only config: [`configs/step1/goalstep/z1_jihun2.yaml`](../../configs/step1/goalstep/z1_jihun2.yaml)
- GoalStep 학습/체크포인트 선택: [`src/ego/step1_action_anticipation/goalstep/train_goalstep_z1.py`](../../src/ego/step1_action_anticipation/goalstep/train_goalstep_z1.py)
- 고정 evaluation transform 추출: [`scripts/step1/ego4d_lta/extract_features.py`](../../scripts/step1/ego4d_lta/extract_features.py)
- FP16 feature cache: [`src/ego/step1_action_anticipation/data/feature_cache.py`](../../src/ego/step1_action_anticipation/data/feature_cache.py)
- taxonomy 방법: [`src/ego/step1_action_anticipation/goalstep/taxonomy/GOALSTEP_TAXONOMY_METHOD.md`](../../src/ego/step1_action_anticipation/goalstep/taxonomy/GOALSTEP_TAXONOMY_METHOD.md)
- 현재 run metadata: [`outputs/goalstep/night/b2_vna/run_metadata.json`](../../outputs/goalstep/night/b2_vna/run_metadata.json)
- epoch별 지표: [`outputs/goalstep/night/b2_vna/metrics_per_epoch.json`](../../outputs/goalstep/night/b2_vna/metrics_per_epoch.json)
- 선택 checkpoint full-val: [`outputs/goalstep/night/b2_vna/final_metrics.json`](../../outputs/goalstep/night/b2_vna/final_metrics.json)
- train–val gap: [`outputs/goalstep/night/b2_vna/gap.json`](../../outputs/goalstep/night/b2_vna/gap.json)

### 공식 자료

- [V-JEPA2 공식 저장소 및 EK100 결과](https://github.com/facebookresearch/vjepa2)
- [공식 ViT-L EK100 config, 고정 commit](https://github.com/facebookresearch/vjepa2/blob/204698b45b3712590f06245fbfba32d3be539812f/configs/eval/vitl/ek100.yaml)
- [공식 EK100 loader, 고정 commit](https://github.com/facebookresearch/vjepa2/blob/204698b45b3712590f06245fbfba32d3be539812f/evals/action_anticipation_frozen/epickitchens.py)
- [공식 evaluator, 고정 commit](https://github.com/facebookresearch/vjepa2/blob/204698b45b3712590f06245fbfba32d3be539812f/evals/action_anticipation_frozen/eval.py)
- [공식 metrics, 고정 commit](https://github.com/facebookresearch/vjepa2/blob/204698b45b3712590f06245fbfba32d3be539812f/evals/action_anticipation_frozen/metrics.py)
- [공식 anticipation wrapper, 고정 commit](https://github.com/facebookresearch/vjepa2/blob/204698b45b3712590f06245fbfba32d3be539812f/evals/action_anticipation_frozen/modelcustom/vit_encoder_predictor_concat_ar.py)
- [V-JEPA2 논문](https://arxiv.org/abs/2506.09985)
- [V-JEPA2.1 논문](https://arxiv.org/abs/2603.14482)
- [EPIC-KITCHENS-100 annotation 저장소](https://github.com/epic-kitchens/epic-kitchens-100-annotations)

공식 소스와 annotation은 감사 과정에서 `/tmp`에 shallow clone해 읽었으며 `EGO_jihun2` git에는 추가하지 않았다.
