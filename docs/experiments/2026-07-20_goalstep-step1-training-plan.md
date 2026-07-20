# GoalStep 기반 Step1 Action Anticipation 학습 — 계획·우려·예상 결과

- 작성일: 2026-07-20
- 상태: **코드·스모크 완료, 본 학습 미실행(사용자 지시 대기).**
- 이 문서의 역할: "왜 이걸 하는가 / 무엇으로 하는가 / 무엇이 걱정되는가 / 어떤 결과를 예상하는가"를
  한 장으로 정리. **구현 상세·실행 로그·재현 명령은
  [`2026-07-19_goalstep-verbnoun-method_handoff.md`](2026-07-19_goalstep-verbnoun-method_handoff.md)**
  (이하 *핸드오프 문서*)에 있고, 이 문서는 그 위의 판단·리스크 레이어다.
- 관련: `2026-07-17_ego4d-lta-full-training-results.md`(FHO-LTA 결과),
  `2026-07-13_vjepa2-action-anticipation-method.md`(V-JEPA2 anticipation 방법론)

---

## 1. 목표

**Ego4D GoalStep(요리 도메인)에서, 관측 구간 직후 tau_a=1초 뒤에 시작할 다음 step을
(verb, noun, action) 3분류로 예측한다.**

부차 목표이자 이 실험의 실질적 가치:

1. **도메인 이전(transfer) 검증.** EK100 → FHO-LTA로 이어온 "frozen V-JEPA2 + attentive probe"
   골격이, 라벨 성격이 다른 데이터(짧은 hand-object interaction → 긴 절차적 step)에서도
   그대로 작동하는지 확인한다.
2. **절차적 예측(procedural anticipation) 난이도 측정.** EK100/FHO는 "다음 손동작"을 맞히는
   과제고, GoalStep은 "레시피의 다음 단계"를 맞히는 과제다. 후자가 정말로 더 어려운지를
   같은 코드·같은 지표로 정량화한다(§5.3의 verb 축).
3. **Step2/Step3 재료 확보.** verb/noun/action likelihood·entropy를 남겨 후속 단계
   (VLM alignment / dynamic planning)의 입력으로 쓴다.

**비목표(하지 않는 것):** 외부 리더보드 제출, EK100/FHO와의 절대 수치 비교,
backbone fine-tuning, probe 구조 변경.

---

## 2. 대원칙 — "골격은 그대로, 데이터만 교체"

> frozen encoder+predictor / attentive probe / 3-head / focal loss / class-mean Recall@5 는 **그대로**.
> 데이터로더 · taxonomy 출력차원 · 인덱스만 GoalStep용으로 교체.

이 원칙 덕분에 **기존 FHO 코드의 diff는 단 1줄**이다
(`scripts/step1/ego4d_lta/extract_features.py`의 argparse에 `"val"` 추가).
GoalStep 전용 로직은 전부 신규 파일로 분리했다.

---

## 3. 데이터셋 — 무엇을 쓰는가

### 3.1 주석

| 항목 | 값 |
|---|---|
| 출처 | Ego4D **GoalStep**, `goalstep_train.json`(583 vid) / `goalstep_val.json`(134 vid) |
| 버전 | **v2_1** (v2에는 GoalStep 주석이 없다) |
| 레벨 | **step + substep 모두** (`--level both`, 사용자 확정 2026-07-20) |
| 평가 | **`goalstep_val.json` 134 vid만.** train은 평가에 일절 미사용 |
| 오염검사 | train∩val video_uid = **0** (겹치면 스크립트가 exit 1) |
| `test_unannotated` | **미사용** (라벨 비공개 + 우리 라벨 공간과 무관) |

### 3.2 라벨 공간 (bespoke)

GoalStep 원본 주석은 `"Cook on a stovetop: Preheat a pan or pot"` 같은 **문장**이다.
이걸 EK100 방법론(spaCy dep-parse → lemma → synonym 병합)으로 파싱해 만든 것이 우리 라벨이다.

| | 값 |
|---|---|
| taxonomy 공간 | verb **100** / noun **190** |
| 등록(train 등장 조합만) | verb **98** / noun **188** / action **390** |
| 파싱 성공률 | 99.60% (OTHER 157건 = 0.40%) |

**중요:** 이 공간은 FHO(117/521)도, GoalStep 공식 step(514)도 아닌 **우리만의 공간**이다.
파싱 규칙을 바꾸면 정답표 자체가 바뀌므로 학습 전에 확정돼야 하고, 외부 SOTA와 직접 비교할 수 없다.

### 3.3 Z=1 인덱스 (학습·채점 단위)

| | train | val |
|---|---|---|
| Z=1 샘플 | **30,804** | **7,425** |
| 영상 | 571 | 130 |
| 레벨 구성 | step 12,622(41%) / substep 17,941(59%) | — |

윈도우 규칙(FHO와 동일): `obs_end = step_start − tau_a(1.0s)`,
`obs_start = obs_end − L_obs(3.5s)`, boundary는 truncate, 영상의 첫 step은 제외.

### 3.4 영상

GoalStep은 clip 레이어가 없어 **원본 영상**이 필요하다(`video_source: full_scale`).

| 매니페스트 | 커버 | 영상당 |
|---|---|---|
| `video_540ss` (v2_1) | 529 / 701 | ~300 MB |
| `full_scale` (**v2_1**) | 나머지 172 (누적 701/701) | ~750 MB |

⚠️ **172개는 v2_1 `full_scale`에만 존재한다.** v2 매니페스트로는 529개밖에 못 받는다.
"540ss 우선 + full_scale 폴백" 전략으로 **총 272 GB**(dry-run 실측).

---

## 4. 코드 — 무엇으로 하는가

### 4.1 그대로 재사용 (수정 없음)

| 구성요소 | 위치 |
|---|---|
| Z=1 윈도우 규칙 | `ego.datasets.ego4d.build_z1_index` |
| action registry 규칙 | `ego.datasets.ego4d.register_action_labels` |
| 비디오 데이터셋 | `ego.datasets.ego4d.Ego4DLTADataset` |
| 피처 캐싱 | `ego.step1_action_anticipation.data.feature_cache` |
| **probe · 3-head** | `ego.step1_action_anticipation.models.AnticipationHead` |
| focal loss · LR/WD 스케줄 | `ego.step1_action_anticipation.train` |
| 지표 | `ego.step1_action_anticipation.metrics` |
| 학습 루프 유틸 | `scripts/step1/ego4d_lta/train_lta_z1.py` (import) |

### 4.2 GoalStep 전용 신규 파일

| 파일 | 역할 |
|---|---|
| `scripts/step1/goalstep/check_overlap.py` | 오염검사 + flat CSV 교차검증 |
| `scripts/step1/goalstep/build_goalstep_taxonomy.py` | taxonomy + step별 라벨 + action registry |
| `scripts/step1/goalstep/build_goalstep_z1_index.py` | Z=1 인덱스(train/val parquet) |
| `scripts/step1/goalstep/download_goalstep_videos.py` | 영상 701개 병렬 다운로드 |
| `scripts/step1/goalstep/train_goalstep_z1.py` | 학습 루프 |
| `configs/step1/goalstep/z1.yaml` / `smoke.yaml` | 본 학습 / 스모크 설정 |

### 4.3 모델 구성 (변경 없음)

```
원본 영상 [obs_start, obs_end]  →  32 frames 균등 샘플 @ 256px
   → V-JEPA2 ViT-L encoder (frozen)      → 4,096 tokens
   → V-JEPA2 predictor    (frozen)       →   256 tokens  ("tau_a초 뒤 mask token")
   → concat [4352, 1024] fp16 캐싱 (샘플당 8.7 MB)
   → attentive probe (4 block / 16 head / query token 3)   ← 유일하게 gradient가 흐름
   → verb(98) / noun(188) / action(390) 3-head
   → sigmoid focal loss (gamma 2.0, alpha 0.25) 합산
```

### 4.4 검증 프로토콜

- 매 epoch: val **500-subset**(seed 42 고정, 전 epoch 공통) →
  verb/noun/action **각각** class-mean Recall@5 · Top-1 · Top-5 전부 로깅
- 학습 후: `best.pt`를 **full val 7,425개로 1회** 재평가
- 체크포인트: `epoch_01.pt … epoch_10.pt` **전부** + `best.pt` + `latest.pt`

### 4.5 비용

| 단계 | 시간 | 디스크 |
|---|---|---|
| 영상 다운로드 (701 / 272 GB) | 1~4 h | 272 GB |
| feature 추출 (38,229 샘플) | ~10 h (1회성, 재개 가능) | ~330 GB |
| 학습 10 epoch | ~25 h (epoch당 ≈2.5 h) | — |
| **합계** | **약 36~40 h** | **~600 GB** |

추출 처리량 **≈66 샘플/분**은 FHO 실측과 동일하다. 병목은 비디오 디코딩이 아니라
**V-JEPA2 ViT-L 순전파(GPU)** 이므로, 영상이 길어져도 처리량은 변하지 않는다.
(FHO가 24h 걸린 건 샘플이 91,573개로 2.4배 많아서일 뿐이다.)

---

## 5. 우려되는 부분

### 5.1 ★ 관측 3.5초가 절차 예측에 너무 짧다 (최대 우려)

**GoalStep step은 EK100/FHO 액션보다 훨씬 길다.**

| 레벨 | 개수 | median 길이 | mean |
|---|---|---|---|
| step | 13,282 | **21.5 s** | 50.1 s |
| substep | 18,186 | **9.9 s** | 19.0 s |

직전 인스턴스 시작 → 다음 인스턴스 시작 간격은 **median 11.3 s**다.
즉 3.5초 창은 대부분 **이전 step의 꼬리 부분만** 본다:

| L_obs | 직전 구간을 완전히 덮는 비율 |
|---|---|
| **3.5 s (현재)** | **21.5%** |
| 8 s | 39.5% |
| 16 s | 61.5% |

**78.5%의 샘플에서 모델은 "직전에 무슨 step이 있었는지"조차 온전히 못 본다.**
게다가 다음 step은 화면보다 **레시피 지식**이 결정하는데("반죽을 다 치댔다 → 발효? 성형? 굽기?"),
우리 probe에는 goal conditioning이 없다.

→ 대응책은 §6에서 별도로 다룬다.

### 5.2 long-tail이 class-mean을 직접 깎는다

train action 390개 중 **94개(24%)가 10샘플 미만**. class-mean Recall@5는 클래스별 평균이라
tail이 0이면 상한이 76%로 잘린다. 스모크에서 tail band가 정확히 `0.0`으로 찍혔다.
focal loss(gamma 2.0)로 완화하지만 근본 해결은 아니다.

### 5.3 EK100/FHO와 절대 수치 비교 불가

클래스 수가 다르면 Recall@5는 정의상 비교할 수 없다.
Recall@5의 우연 수준: GoalStep 5/390 = **1.28%**, EK100 5/3,806 = **0.13%** — **10배** 차이.
**숫자가 EK100보다 높게 나와도 "모델이 더 낫다"는 뜻이 아니다.**

**단, verb head는 거의 공정한 비교축이다: GoalStep 98 vs EK100 97 — 클래스 수가 거의 같다.**
여기서는 "클래스가 적어서 높다"는 설명이 통하지 않는다. 학습 후 이 축을 먼저 읽을 것.

### 5.4 파싱 노이즈 · 포괄어 명사

- 전치사구가 버려진다: `pour water into pot`과 `pour water into bowl`이 같은 (pour, water)로 붕괴.
- 파싱 세그먼트의 **27.4%가 포괄어 noun**이고 "ingredient" 하나가 20%.
  → **Top-1(micro)은 부풀려지고 class-mean은 깎이는 괴리**가 생긴다.
  스모크에서 이미 관측: noun Top-1 27.68% vs noun cmR@5 20.73%.

### 5.5 500-subset과 full-val의 괴리

스모크에서 action cmR@5가 **subset 17.63 → full 10.33**으로 크게 벌어졌다.
class-mean은 표본이 작을수록 낙관적으로 뜬다(390 클래스에 500 샘플 = 클래스당 1.3개).
→ **subset은 모델 선택용으로만, 보고 수치는 full-val로.** 코드·리포트에 명시돼 있다.

### 5.6 운영 리스크

- **decord seek 실패 경고.** 긴 full_scale 원본에서
  `Failed to skip frames effectively … Video might be corrupted or seeking failed` 다발.
  추출은 정상 완료되지만(스모크 1,409/1,409, 실패 0) 처리량이 떨어진다.
  full_scale이 172개 들어가므로 추출이 10h → **12~15h**로 늘 수 있다.
- **디스크 ~600 GB.** 현재 여유 1.7 TB라 여유는 있으나, 기존 FHO 캐시가 이미 761 GB를 쓴다.
- **단일 도메인.** 전부 COOKING. 여기 수치는 "요리 절차 예측" 성능이지 일반 anticipation 성능이 아니다.

---

## 6. L_obs를 3.5초보다 늘릴 것인가

§5.1의 대응책. 결론부터: **해볼 가치가 크고, GPU 비용은 사실상 0이다. 단 숨은 제약이 하나 있다.**

### 6.1 왜 거의 공짜인가

`sample_uniform_frame_indices`는 창 길이와 무관하게 **[obs_start, obs_end]를 32프레임으로 균등 샘플**한다
(`src/ego/datasets/video_sampling.py`). 따라서 L_obs를 늘려도:

- 입력 텐서 `[B, 3, 32, 256, 256]` **불변**
- 캐시 크기 `[4352, 1024]` **불변** (샘플당 8.7 MB)
- GPU 순전파 비용 **불변**

바뀌는 건 **시간 stride뿐**이다.

### 6.2 숨은 제약 — 유효 fps와 predictor 오프셋

`frames_per_second`(config 8)는 장식이 아니다. predictor가 "몇 토큰 뒤가 tau_a초 뒤인가"를
계산하는 데 쓰인다 (`vjepa2_backbone.py:108`):

```python
anticipation_steps = (anticipation_times * self.frames_per_second / self.tubelet_size)
```

| L_obs | 32프레임의 유효 fps | 올바른 `frames_per_second` |
|---|---|---|
| 3.5 s (현재) | 9.14 | 8 (설정값, 14% 오차 — 허용 범위) |
| 4.0 s | 8.00 | 8 (정확히 일치) |
| 8 s | 4.00 | **4** |
| 16 s | 2.00 | **2** |

**L_obs만 바꾸고 `frames_per_second`를 8로 두면 predictor가 tau_a=1초를 2초(L_obs=8 기준)로
착각한다.** 반드시 함께 바꿔야 한다.

두 번째 제약은 **V-JEPA2 사전학습 분포와의 괴리**다. 4 fps / 2 fps 입력은 인접 프레임 간
움직임이 2~4배 커서 encoder에겐 낯선 분포다. 성능이 오히려 떨어질 수 있고, 이건 이론이 아니라
**실측으로만 확인 가능**하다.

### 6.3 진짜 비용은 재추출이다

L_obs는 **feature cache에 구워진다.** 값을 바꾸면 인덱스 재빌드 + **전체 재추출(~10h)** 이 필요하다.
학습(~25h)까지 하면 L_obs 값 하나당 **약 35시간**. GPU 단가는 0이지만 벽시계는 0이 아니다.

### 6.4 대안 비교

| 안 | 방법 | GPU/디스크 | 위험 | 평가 |
|---|---|---|---|---|
| **A. L_obs 8s, 32프레임, fps 4** | config 2줄 | **불변** | 4fps 분포 shift | ★ 권장 ablation |
| B. L_obs 8s, **64프레임**, fps 8 | frames_per_clip 64 | 캐시 2배(**660 GB**), 추출 20h+, 학습 50h+ | "입력 규격 동일" 원칙 위반 | 비권장 |
| C. goal conditioning 추가 | probe에 goal 임베딩 주입 | — | **아키텍처 변경 금지 위반** | 이번 범위 밖 |

### 6.5 권고

1. **먼저 L_obs=3.5s 베이스라인을 완주한다.** FHO/EK100과 같은 조건이라 비교 가능한 유일한 기준점이고,
   §5.1이 실제 문제인지 아닌지 판단할 근거가 된다.
2. **그다음 A안(L_obs 8s / fps 4)을 ablation으로 1회 돌린다.** 캐시를 별도 디렉토리로 분리하면
   베이스라인 캐시는 보존된다. 추가 비용 ~35h.
3. 판단 기준: **action cmR@5가 유의미하게 오르면** 절차 예측에 긴 컨텍스트가 필요하다는 증거이고,
   **떨어지면** 4fps 분포 shift가 이득을 잡아먹은 것 — 어느 쪽이든 보고할 가치가 있는 결과다.

> 지금 config를 바꾸지 않는다. 베이스라인 결과를 보고 결정한다.

---

## 7. 예상 결과

### 7.1 정량 예측

| 지표 | 예상 방향 | 근거 |
|---|---|---|
| **Top-1 / Top-5** | EK100보다 **확실히 높음** | 클래스 10배 적음 + 포괄어 noun("ingredient" 20%) |
| **verb cmR@5** | EK100(63.6)보다 **낮을 것** | 클래스 수는 비슷(98 vs 97)한데 태스크가 더 어렵고, 우리는 ViT-L/256 (논문은 ViT-g/384) |
| **noun cmR@5** | verb보다 낮음 | long-tail + 포괄어 편중 |
| **action cmR@5** | FHO(8.03)보다 **높지만** 기대만큼은 아님 | 클래스 390 vs 5,698로 유리하나, tail 24%가 상한을 깎음 |
| **band** | head ≫ mid > tail ≈ 0 | 스모크에서 이미 tail = 0.0 |
| **best epoch** | 중후반(6~9) | FHO는 12 epoch 중 8이 최적이었다 |

### 7.2 정성 예측

- **verb > noun > action** 순으로 어렵다(클래스 수·모호성 순). 모든 anticipation 실험의 공통 패턴.
- **subset(500) > full-val** 로 class-mean이 낙관 편향될 것. 스모크에서 재현됨.
- scenario별 편차가 클 것. 스모크에서 이미 `MAKE_NOODLE_SOUP` 0.0 vs `MAKE_FLATBREAD` 22.4.

### 7.3 이 실험이 "성공"이라는 기준

절대 수치가 아니라 **아래가 확인되면 성공**이다.

1. 골격 수정 없이 새 도메인에서 학습이 수렴한다(train loss 하락 + 지표 상승).
2. verb 축에서 EK100 대비 격차가 정량화된다 → 절차 예측 난이도를 숫자로 말할 수 있다.
3. long-tail/포괄어의 영향이 band·scenario breakdown으로 분해된다.
4. Step2/Step3가 쓸 likelihood·entropy가 확보된다.

**반대로, 수치가 EK100보다 높게 나왔다고 "더 잘한다"고 쓰면 안 된다.** 그건 라벨 공간이
좁아진 결과다.

---

## 8. 다음 액션

| # | 액션 | 상태 |
|---|---|---|
| 1 | 영상 701개 / 272 GB 다운로드 | ⏸ 지시 대기 |
| 2 | feature 추출 (train 30,804 / val 7,425) | ⏸ |
| 3 | 학습 10 epoch → 핸드오프 문서 §9 결과표 채우기 | ⏸ |
| 4 | (선택) L_obs=8s / fps=4 ablation | 베이스라인 결과 보고 판단 |

실행 명령은 핸드오프 문서 §11에 있다. seed는 전부 42.
