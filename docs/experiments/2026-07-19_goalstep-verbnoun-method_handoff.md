# Ego4D GoalStep → verb/noun/action Taxonomy · Step1 Anticipation 학습 파이프라인

- 작성일: 2026-07-19 (Phase 3 추가: 2026-07-20)
- 목적: Ego4D **GoalStep**(요리 도메인) 문장형 step/substep 주석을 **EK100(EPIC-KITCHENS-100)
  방식**으로 verb/noun/action 클래스로 파싱하고(§1–3), 그 라벨 공간 위에서 **기존 FHO-LTA
  학습 코드를 최대한 그대로 재사용해** Step1 anticipation 파이프라인을 구성한 전 과정을 기록한다(§4–11).
- 관련 문서: `2026-07-13_ego4d-lta-goalstep-join-method.md`,
  `2026-07-16_ego4d-data-download-handoff.md`(원본 Ego4D 다운로드),
  `2026-07-13_vjepa2-action-anticipation-method.md`(학습 파이프라인),
  `2026-07-17_ego4d-lta-full-training-results.md`(FHO-LTA 학습 결과),
  `outputs/goalstep/GOALSTEP_TAXONOMY_METHOD.md`(파싱 방법론 상세),
  **[`2026-07-20_goalstep-step1-training-plan.md`](2026-07-20_goalstep-step1-training-plan.md)
  (목표·우려·예상 결과, L_obs 확장 검토 — 이 문서의 판단 레이어).**
- 스크립트: `scripts/step1/goalstep/` 전체 + `configs/step1/goalstep/{z1,smoke}.yaml`
- 상태: **Phase 1(주석·검토 CSV), Phase 2(verb/noun/action 클래스), Phase 3(오염검사·인덱스·
  다운로드·피처추출·학습 코드) 완료. 본 학습(10 epoch)은 사용자 지시 대기 중 — 미실행.**

---

## 확정 학습 구성 (2026-07-20, 사용자 확정)

실제로 돌아갈 설정. 세부 근거는 §4~9, 한계는 §10 참조.
config: `configs/step1/goalstep/z1.yaml` / 실행: `scripts/step1/goalstep/train_goalstep_z1.py`

### 데이터

| 항목 | 값 | 비고 |
|---|---|---|
| 주석 | `goalstep_train.json` 583 vid / `goalstep_val.json` 134 vid | **v2_1**. video_uid 교집합 **0** 확인(§5.1) |
| 대상 레벨 | **step + substep (`--level both`)** | train 샘플 step 12,622(41%) / substep 17,941(59%). 근거 §10-6 |
| Z=1 샘플 | **train 30,804 / val 7,425** | 영상 571 / 130 (첫 step·1-인스턴스 영상 제외) |
| 출력차원 | **verb 98 / noun 188 / action 390** | train 등장 (verb,noun) 조합만 dense 등록 |
| taxonomy 공간 | verb 100 / noun 190 | bespoke — FHO 117/521도, GoalStep 공식 514도 아님 |
| scenario | GoalStep `goal_category` **79종** | scenario-stratified sampler·breakdown에 사용 |
| 영상 | 701개 / **272 GB** | 540ss 529개 + **full_scale 172개(v2_1에만 존재)** |

### 윈도우 (FHO와 동일)

| 항목 | 값 |
|---|---|
| tau_a (anticipation) | **1.0 s** |
| L_obs (관측창) | **3.5 s** |
| obs 구간 | `obs_end = step_start − 1.0`, `obs_start = obs_end − 3.5` |
| boundary_policy | `truncate` (0 미만은 0으로, `boundary_flag=True`) |
| min_obs_sec | 0.5 |
| 제외 규칙 | 영상의 첫 step(관측 구간 없음), train 미등장 (verb,noun) 조합의 val 샘플 |

### 모델 · 학습 (probe 구조·loss·입력규격 = FHO 그대로, 변경 없음)

| 항목 | 값 |
|---|---|
| backbone | V-JEPA2 ViT-L `checkpoints/vjepa2/vitl.pt`, **frozen** encoder+predictor |
| predictor | `no_predictor: false`, `num_steps: 1` (1초 미래 mask token) |
| 입력 규격 | T=32 frames / 8 fps / 256 px |
| 표현 | encoder ⊕ predictor concat, `[4352, 1024]` fp16 캐싱 (샘플당 8.7 MB) |
| probe | attentive probe **4 block / 16 head / query token 3** (변경 금지) |
| head | 3-head 독립 linear, out_features = 98 / 188 / 390 ← **유일한 구조 변경** |
| loss | sigmoid focal, `gamma=2.0`, `alpha=0.25` |
| optimizer | AdamW, lr 3e-4, wd 1e-4, warmup 1 epoch + cosine |
| batch / sampler | 32 / `scenario_stratified` |
| **epochs** | **10** |
| seed | **42** (experiment.seed = val_subset_seed = index seed) |

### 검증 · 체크포인트

| 항목 | 값 |
|---|---|
| 평가셋 | **`goalstep_val.json`(134 vid)만.** train 절대 미사용 |
| 매 epoch | val **500-subset**(seed 42 고정, 전 epoch 공통) |
| 지표 | verb/noun/action **각각** × (class-mean Recall@5, Top-1, Top-5) 전부 |
| 부가 | head/mid/tail band, scenario breakdown, likelihood·entropy |
| 최종 | 학습 후 `best.pt`를 **full val 7,425개로 1회** 재평가 |
| 체크포인트 | `checkpoints/epoch_01.pt … epoch_10.pt` **전부** + `best.pt` + `latest.pt` |
| run 디렉토리 | `outputs/goalstep/runs/z1/` (config·로그·지표 동봉) |

### 예상 비용 (스모크 실측 기반, §8.1)

| 단계 | 시간 | 디스크 |
|---|---|---|
| 영상 다운로드 | 1~4 h | 272 GB |
| feature 추출 (38,229) | ~10 h (1회성, 재개 가능) | ~330 GB |
| 학습 10 epoch | ~25 h (epoch당 ≈2.5 h) | — |
| best.pt full-val | ~35 min | — |
| **합계** | **약 36~40 h** | **~600 GB** |

---

## 0. 대원칙 (요약)

- 산출물 스키마는 FHO(`fho_lta_taxonomy.json` / `build_lta_z1_index.py` 출력)와 1:1 일치.
- verb/noun/action 추출·클래스 구성 "방법론"은 EK100 방식을 차용.
- **Phase 1·2 단계에서 영상(full_scale/clip)은 다운로드하지 않는다. annotation만 받는다.**
  (영상은 Phase 3의 feature 추출에서 처음 필요해진다 — §5.4.)
- AWS 키는 `~/.aws/credentials`에만 두고, 스크립트·로그·리포트엔 placeholder만.
  region은 이메일상 us-west-1이나 boto3 직접 접근 시 us-west-2로 검증(기존 FHO 실측).

---

## 1. 어떤 데이터를 어떻게 받았나 (annotation only)

### 1.1 다운로드 명령 (실측)

```bash
ego4d --datasets annotations --benchmarks goalstep --version v2_1 -o data/Ego4D -y
```

- **`--datasets annotations`만** 지정(영상 데이터셋 full_scale/clips/clip_* 미포함).
- **`--version v2_1` 필수.** ⚠️ 함정: GoalStep 주석은 **Ego4D v2가 아니라 v2.1**에 추가되었다.
  `--version v2`로 받으면 v2 annotation manifest에 benchmarks 컬럼이 없어
  `--benchmarks goalstep` 필터가 조용히 무시되고 goalstep이 아닌 파일만 받아진다.
  v2_1로 받아야 콘솔에 `Filtering by benchmarks: ['goalstep']`가 뜨고 goalstep 파일만 내려온다.
- AWS 자격증명은 `~/.aws/credentials`의 `[default]`(Ego4D 발급 IAM 키, access key `AKIA...P27W`)를 사용.
  이 키는 GetObject만 가능하고 ListBucket은 불가(403). CLI annotation 다운로드는 정상 동작.

### 1.2 받은 파일 (`data/Ego4D/v2/annotations/`)

| 파일 | 영상 수 | 내용 | 스키마 |
|---|---|---|---|
| `goalstep_train.json` | 583 | 요리 dense 주석 (goal→step→substep) | `videos[].segments[].segments[]` |
| `goalstep_val.json` | 134 | 요리 dense 주석 (goal→step→substep) | 동일 |
| `goalstep_test_unannotated.json` | 134 | **uid만, 라벨 없음** (벤치마크용 비공개) | `videos[].video_uid`만 |
| `goalstep_trainval.json` | 7219 | **goal-level만** (train+val 포함 superset, 다도메인) | `videos[].annotations[]`(다른 스키마) |

- **dense 요리 step 세트 = train(583)+val(134)+test(134) = 851 영상 ≈ 430h**(논문 수치).
  이 중 라벨 공개된 건 train+val 717영상(영상시간 305.2h). test 134영상(≈125h)은 라벨 비공개.
- 영상 파일은 **한 개도 받지 않았다**(대원칙 준수). 기존 `clip_256ss`(71G)는 이전 FHO 작업물.

### 1.3 주석 계층 구조

```
video (video_uid, goal_category, goal_description, is_procedural, ...)
 └─ segments (step)   : step_category, step_description, start/end_time, is_relevant/procedural/continued
     └─ segments (substep) : (step과 동일 필드)
```

- `step_category`는 **`"<coarse category>: <specific step>"` 콜론 구조**
  (예: `"Cook on a stovetop: Preheat a pan or pot on the stovetop"`).
- 정답(주석)의 원천은 `goalstep_train.json` / `goalstep_val.json`. 이후 모든 산출물은 이 둘을 가공한 것.

---

## 2. Phase 1 — 사람이 검토할 평탄화 CSV

스크립트: `scripts/step1/goalstep/dump_goalstep_annotations.py`
→ `outputs/goalstep/inspection/goalstep_annotations_flat.csv` (46,646행, 고유 영상 7,219개)
   + `goalstep_annotations_sample.csv` (300행 표본).

- goal→step→substep을 행 단위로 펼침. **원문 텍스트 그대로 보존(파싱 없음).**
- "모든 주석 데이터 포함" 정책: train/val dense + trainval goal-only(중복 영상은 dense 우선).
- 정합성 확인: **단일 COOKING 도메인**(전부 `COOKING:*`), train+val 305.2h,
  goal_category 80/86개·step_category 501/514개 노출(나머지는 test 전용).

---

## 3. EK100 유사 메소드로 verb/noun/action 클래스 만들기 (Phase 2)

스크립트: `scripts/step1/goalstep/parse_goalstep_to_verbnoun.py` (spaCy `en_core_web_sm` 필요).

### 3.1 파싱 소스·레벨 (확정값)

| 결정 | 값 | 근거 |
|---|---|---|
| 소스 필드 | **`step_category` 콜론 뒤 구절** | step_category 100%가 콜론 구조, 고유 라벨 501개(통제 어휘) → 자유서술 step_description보다 깨끗 |
| 파싱 레벨 | **step + substep** | 두 레벨이 같은 501-라벨 어휘 공유 → 커버리지 극대화 |
| trainval | **제외** | goal-level·다도메인이라 요리 step taxonomy 범위 밖 |

### 3.2 EK100 4단계 방법론 (차용)

1. **추출 (POS/dependency parse).** spaCy로 각 라벨 파싱. GoalStep 라벨은 명령문이라
   문두 동사가 명사로 오태깅됨 → **명령문 강제**(`"Store ingredients"` → `"I store ingredients"`:
   주어 prepend + 첫 글자 소문자화)로 root 동사 + 목적어 추출. 짧은 명령문 오태깅 보정용
   POS-관대 fallback: 목적어는 dobj/obj면 POS 무관 수용(bare object "eggplant"가 ADJ로 태깅됨),
   동사는 VERB/AUX 없으면 첫 내용어로 fallback(manner 부사 skip → "Deep fry"→fry).
2. **레마화.** verb/noun을 lemma로(복수·시제 정규화: ingredients→ingredient).
3. **의미 클러스터링.** lemma 기준 그룹 + 소규모 **수동 synonym 병합맵**. 각 class = 대표 key + members,
   FHO식 `key_(member1,_member2)` 표기.
   - verb 병합: `put(place,set,lay,drop)`, `cut(chop,slice,dice,mince)`, `mix(combine,blend,whisk,beat)`,
     `get(grab,take,fetch,retrieve,collect,pick)`, `dispose(discard,throw,toss)`, `organize(arrange,tidy)`,
     `wash(rinse)`, `shape(form)`.
   - noun 병합: `tool(utensil,cookware,equipment)`, `spice(seasoning)`, `vegetable(veggie,veg)`.
   - 의도적 미병합: `stir` vs `mix`, `wash` vs `clean` (빈도 크고 의미 구분됨).
4. **action = (verb_class, noun_class).** **train split에 실제 등장한 조합만** dense index화
   (FHO `register_action_labels` 규칙 동일).

### 3.3 결과 통계

- 파싱 세그먼트(train+val step+substep): **39,262** (train 31,468 · val 7,637).
- 파싱 성공률 **99.60%**, **OTHER 0.40%(157건)** — 전부 "동사O 목적어X",
  그중 101건이 비행위성 "Non cooking miscellaneous". (임계값 15% 훨씬 미만.)
- **N_verb = 100, N_noun = 190** (taxonomy 어휘, train+val 기준).
- **N_action = 390** (train-등장 조합).
- dense registry(train-only, val 전용 클래스 제외 — FHO 117→116과 동일 패턴):
  `num_verbs=98, num_nouns=188, num_actions=390`.
- 상위 verb: add(6120)·stir(3379)·wash(3054)·cut(2800)·cook(1420) /
  상위 noun: ingredient(7878)·dough(4235)·dish(1585)·flatbread(1541)·tool(1323).

### 3.4 EK100와의 비교 및 action이 적은 이유

| 항목 | EK100 | GoalStep(우리) |
|---|---|---|
| 영상 시간 | ~100h(전체) | train+val 305h(test 125h 라벨없음 제외) |
| 파싱 입력 텍스트 종류 | narration ~20,000종 | step_category **495종(파싱가능)** |
| 세그먼트 | ~90,000 | 39,262 |
| verb / noun / action | 97 / 300 / **3,806** | 100 / 190 / **390** |

action이 390뿐인 이유(데이터 확인):
- **① 통제 어휘가 상한을 결정.** 각 라벨은 1개 (verb,noun)로 결정적 매핑 → action 상한 ≤ 라벨 수(≈495).
  EK100은 ~2만 종 자유 narration → 40배 다양.
- **② 반복.** 39,105 세그먼트가 390 action으로 뭉침(action당 100개; EK100은 24개).
- **③ 포괄어 명사.** 파싱 세그먼트의 27.4%가 generic noun("ingredient" 하나가 20%). GoalStep 라벨은
  "Add **ingredients**"처럼 구체 객체 대신 포괄어 → noun 다양성 낮음.
- 결론: 버그가 아니라 "깨끗한 통제 라벨을 소스로 택한" 트레이드오프의 구조적 결과.

### 3.5 한계 (중요)

- **전치사구(second argument) 미표현.** EK100/FHO의 (verb, 단일 noun) 2-슬롯을 따르므로,
  고유 라벨의 **61%(301/495)가 전치사구 포함**이지만 전부 버려진다:
  `Add oil to a pan → add|oil`(~~to a pan~~), `add wood to fire → add|wood`(~~to fire~~).
  전치사 목적어는 직접목적어가 없을 때만 fallback noun으로 쓰인다("Cook on a stovetop"→cook|stovetop).
  → `Add oil to a pan`과 `Add oil to a bowl`이 같은 (add,oil)로 붕괴(도착점/도구/장소 손실).
  이는 EK100 자체의 한계와 동일(EK100도 "add wood to fire" 표현 불가).
- 다중 객체 라벨은 syntactic head noun만 취함("a small amount of filling"→amount).
  복합동사는 root 동사 유지("check and adjust"→check).
- 이 taxonomy는 FHO(117/521)도 GoalStep 공식 step(514)도 아닌 **bespoke 공간** →
  **외부 SOTA와 직접 수치 비교 불가.**
- 병합맵은 수동 큐레이션이 얕고(EK100은 더 촘촘함) 희소 클래스 빈도 컷오프 미적용(롱테일 유지).

---

## 4. Phase 3 — GoalStep Step1 학습 파이프라인 (FHO-LTA 코드 재사용)

대원칙: **골격은 그대로.** frozen V-JEPA2 encoder+predictor → attentive probe(4 block, 16 head,
query token 3개) → verb/noun/action 3-head, sigmoid focal loss, class-mean Recall@5.
데이터로더 / taxonomy 출력차원 / 인덱스만 GoalStep으로 교체했다.

### 4.1 재사용 범위 (수정 없음)

| 구성요소 | 위치 | 변경 |
|---|---|---|
| Z=1 윈도우 규칙 | `ego.datasets.ego4d.build_z1_index` | **없음** (그대로 호출) |
| action registry 규칙 | `ego.datasets.ego4d.register_action_labels` | **없음** |
| train-seen 조합으로 val 제한 | `ego.datasets.label_mapping.filter_to_known_pairs` | **없음** |
| 비디오 데이터셋 | `ego.datasets.ego4d.Ego4DLTADataset` | **없음** (`video_source=full_scale` 경로로 사용) |
| 피처 캐싱 | `ego.step1_action_anticipation.data.feature_cache` | **없음** |
| probe / 3-head | `ego.step1_action_anticipation.models.AnticipationHead` | **없음** (out_features만 registry에서 주입) |
| focal loss · LR/WD 스케줄 | `ego.step1_action_anticipation.train` | **없음** |
| 지표 | `ego.step1_action_anticipation.metrics` (`class_mean_recall`, `top_k_recall`, `prediction_entropy`) | **없음** |
| 학습 루프 유틸 | `scripts/step1/ego4d_lta/train_lta_z1.py` (`train_one_epoch`, `evaluate`, `ScenarioStratifiedSampler`, `_build_*_loader`, `save_likelihood_entropy`) | **없음** (import해서 사용) |
| 특징 추출 CLI | `scripts/step1/ego4d_lta/extract_features.py` | `--split`에 `"val"` 추가 (1줄) |

즉 **기존 FHO 코드의 diff는 `extract_features.py`의 argparse choices 한 줄뿐**이다.

### 4.2 GoalStep 전용 신규 파일

| 파일 | 역할 |
|---|---|
| `scripts/step1/goalstep/check_overlap.py` | 작업1 — train/val 오염검사 + flat CSV 교차검증 |
| `scripts/step1/goalstep/build_goalstep_taxonomy.py` | 작업2 — verb/noun 클래스 → FHO 스키마 taxonomy + step별 라벨 + action registry |
| `scripts/step1/goalstep/build_goalstep_z1_index.py` | 작업3 — Z=1 인덱스(train/val parquet) |
| `scripts/step1/goalstep/download_goalstep_videos.py` | 인덱스가 참조하는 영상 701개 병렬 다운로드 |
| `scripts/step1/goalstep/train_goalstep_z1.py` | 작업6 — 학습 루프(전 epoch 체크포인트, 500-subset val, 최종 full-val) |
| `configs/step1/goalstep/z1.yaml` / `smoke.yaml` | 본 학습 / 스모크 설정 |

### 4.3 FHO 대비 바뀐 지점 (전부)

1. **clip 레이어 없음.** GoalStep 타임스탬프는 video-relative이고 clip이 없다 →
   `clip_uid := video_uid`(로그로 명시), `video_source: full_scale`로 원본 영상에서 디코딩.
   FHO는 `clip_256ss` 클립을 쓴다.
2. **scenario = GoalStep `goal_category`.** Ego4D scenario 태그 대신 요리 goal(예: `COOKING:MAKE_BREAD`)
   79종을 scenario 컬럼에 넣어, 기존 scenario-stratified sampler와 per-scenario breakdown을 그대로 살렸다.
3. **split 구성.** FHO는 val을 dev(0.8)/heldout(0.2)로 재분할하지만, GoalStep은
   `goalstep_val.json`(134 vid)이 곧 평가셋이라 **train/val 2분할**만 쓴다. train은 평가에 일절 미사용.
4. **출력차원.** verb 98 / noun 188 / action 390 (registry에서 자동 주입).
5. **parquet 컬럼에 `action_label` 추가.** FHO 컬럼
   `[video_uid, clip_uid, obs_start_sec, obs_end_sec, verb_label, noun_label, scenario, boundary_flag]`에
   dense action id 한 칼럼을 덧붙였다(감사용). `verb_label`/`noun_label`은 FHO와 동일하게
   **raw taxonomy id**이고, dense 인코딩은 기존대로 `LabelMapping`이 데이터셋 단계에서 수행한다.
6. **step/substep 중복 창 제거.** `--level both`에서는 step과 그 첫 substep이 같은 시각에 시작하는 경우가 있어
   `(video, obs 창, verb, noun)`이 완전히 같은 행을 제거한다(train 13행, val 3행).

---

## 5. 데이터 — 작업 1~3 실행 결과

### 5.1 오염검사 (작업 1, `outputs/goalstep/index/overlap_report.json`)

| 항목 | train | val |
|---|---|---|
| 영상 수 | 583 (unique 583, 중복 0) | 134 (unique 134, 중복 0) |
| step 수 | 13,342 | 3,349 |
| step+substep 수 | 31,566 | 7,696 |
| flat CSV 교차검증 | 583/583 영상 존재, 누락 0 | 134/134 영상 존재, 누락 0 |

**`video_uid` 교집합 = 0.** (겹치면 스크립트가 목록을 출력하고 exit 1로 중단한다.)
flat CSV 행 수(train 32,149 / val 7,830)는 step+substep 수 + 영상당 goal 레벨 1행 = 정확히 일치.

### 5.2 taxonomy · action registry (작업 2)

- `goalstep_verbnoun_taxonomy.json` — **N_verb = 100, N_noun = 190** (FHO와 동일 스키마
  `{"verbs":[...], "nouns":[...]}`, 리스트 인덱스 = 클래스 id).
- step 인스턴스 매핑: 39,262건 중 **39,105건 매핑(99.60%)**, **OTHER 157건(0.40%)** →
  `taxonomy_other_segments.csv`에 전량 로깅. 157건은 전부 Phase 2 파싱 단계에서 이미 OTHER로
  판정된 것(주로 `General activity: Non cooking miscellaneous`)이며, 재매핑(members 경유)이
  필요했던 인스턴스는 0건 — 즉 **파싱 때 저장된 per-step 할당을 100% 재사용**했다.
- `action_registry.json` — **train에 등장한 (verb,noun) 조합만** dense index로 등록
  (FHO `register_action_labels` 규칙 동일): **N_verb=98, N_noun=188, N_action=390.**
  (taxonomy 100/190 중 train 미등장 verb 2개·noun 2개는 등록되지 않는다.)
- 이 스크립트는 Phase 2가 만든 `goalstep_verbnoun_taxonomy.json` / `action_registry.json`을
  **바이트 단위로 동일하게 재생성**하는 것을 확인했다(회귀 검증).

### 5.3 Z=1 인덱스 (작업 3, `outputs/goalstep/index/`)

규칙은 FHO와 동일: `tau_a=1.0s`, `L_obs=3.5s`, `obs_end = step_start - tau_a`,
`obs_start = obs_end - L_obs`, `boundary_policy=truncate`(0 미만은 0으로 자르고
`boundary_flag=True`), `min_obs_sec=0.5` 미만은 제외, 영상의 **첫 step은 관측 구간이 없어 제외**.

| | train | val |
|---|---|---|
| 대상 step 인스턴스 | 31,468 | 7,637 |
| 첫 step 제외 | 582 | 134 |
| min_obs 미달 제외 | 69 | 7 |
| 경계 truncate(유지) | 75 | 23 |
| 중복 창 제거 | 13 | 3 |
| **kept** | **30,804** | 7,493 |
| train-seen 조합 제한 후 | — | **7,425** (68건 제외) |
| 영상 수(최종) | **571** | **130** |
| scenario(goal_category) | 79종 (train+val 합) | |

- train 583 → 571: 1개 영상은 라벨 인스턴스가 전부 OTHER, 11개 영상은 인스턴스가 1개뿐이라
  첫-step 제외 규칙에서 사라진다. val 134 → 130도 동일 사유(1-인스턴스 영상 4개).
- 산출 컬럼: `[video_uid, clip_uid, obs_start_sec, obs_end_sec, verb_label, noun_label,
  action_label, scenario, boundary_flag]` (FHO 컬럼 + `action_label`).
- `video_uids.txt` — 인덱스가 참조하는 **701개** video_uid (다운로드 입력).

### 5.4 영상 다운로드 (작업 4 전제)

GoalStep은 원본 영상이 필요하다. 매니페스트 실측(2026-07-20):

| 매니페스트 | 701개 중 커버 | 영상당 평균 |
|---|---|---|
| `video_540ss` (v2 / v2_1 동일, 9,645개) | 529 | ~300 MB |
| `full_scale` (**v2_1**, 9,821개) | 나머지 172 (누적 701/701) | ~750 MB |
| `full_scale` (v2, 9,611개) | 529만 | — |

**v2 매니페스트로는 172개를 받을 수 없다 — 반드시 `--version v2_1`.**
(Phase 1에서 확인한 "GoalStep은 v2_1" 사실이 영상 매니페스트에도 그대로 적용된다.)
→ 기본 전략은 "540ss 우선, 없으면 full_scale": **701개 / 272.1 GB** (`--dry-run` 실측).
전량 full_scale이면 ~530 GB. 해상도가 섞여도 V-JEPA2 transform이 256px로 리사이즈/크롭하므로 무해하다.

---

## 6. 학습 설정 (작업 5·6)

`configs/step1/goalstep/z1.yaml` — 값은 FHO `full.yaml`과 동일하며, 다른 것은 경로·epoch·val 서브셋뿐.

| 항목 | 값 | FHO 대비 |
|---|---|---|
| backbone | V-JEPA2 ViT-L `checkpoints/vjepa2/vitl.pt`, **frozen** encoder+predictor | 동일 |
| predictor | `no_predictor: false`, `num_steps: 1` (1초 미래 mask token) | 동일 |
| 입력 규격 | T=32 frames, 8 fps, 256px | 동일 (EK100/FHO와 같음) |
| 표현 | encoder ⊕ predictor concat, `.pt`로 캐싱(레코드 id = `{video_uid}_{row}`), 재실행 시 스킵 | 동일 |
| probe | attentive probe 4 block / 16 head / query token 3 | **변경 금지, 동일** |
| head out_features | verb 98 / noun 188 / action 390 | ← 유일한 구조 변경 |
| loss | sigmoid focal, `gamma=2.0`, `alpha=0.25` (config 노출) | 동일 |
| optimizer | AdamW, lr 3e-4, wd 1e-4, warmup 1 epoch + cosine | 동일 |
| batch | 32 | 동일 |
| sampler | `scenario_stratified` (scenario = goal_category) | 동일 코드 |
| epochs | **10** | FHO는 12 |
| seed | 42 | 동일 |

### 6.1 검증 프로토콜

- **매 epoch 종료 시** `val.parquet`로 검증하고 verb/noun/action **각각** 대해
  **class-mean Recall@5 / Top-1 accuracy / Top-5 accuracy** 전부를 출력·로깅한다.
- val 7,425개는 매 epoch 돌리기엔 크므로, **고정 시드(42)로 뽑은 500개 subset**을
  **전 epoch 공통**으로 사용한다(sample_id 목록은 `val_subset_sample_ids.json`에 저장 → 재현 가능).
- 학습 종료 후 **`best.pt`를 전체 val 7,425개로 1회 최종 평가**하고, subset 대비 편차를
  `final_metrics.json`에 함께 남긴다.
- 체크포인트: **매 epoch 전부 저장** `checkpoints/epoch_01.pt … epoch_10.pt`,
  더불어 `best.pt`(val-subset action class-mean Recall@5 기준 최고) / `latest.pt`.
- 부가 산출물: epoch별 `likelihood_entropy_epoch_NN.jsonl` (예측 likelihood·entropy),
  head/mid/tail band breakdown, scenario별 breakdown.

### 6.2 run 디렉토리 구조 (`outputs/goalstep/runs/z1/`)

```
config_resolved.yaml            그대로 다시 실행 가능한 설정 스냅샷
run_metadata.json               seed/tau_a/L_obs/focal/샘플수/출력차원
val_subset_sample_ids.json      500-subset의 sample_id 전체 (재현용)
training_history.csv            epoch,train_loss, {verb,noun,action}×{cmr@5,top1,top5}, seconds
metrics_per_epoch.json          위 + band/scenario breakdown
final_metrics.json              best_epoch, 500-subset 지표, full-val 지표
checkpoints/epoch_01.pt … 10.pt / best.pt / latest.pt
likelihood_entropy_epoch_NN.jsonl, likelihood_entropy_full_val_best.jsonl
```

---

## 7. 학습·채점에 실제로 쓰이는 파일

### 7.1 학습

| 단계 | 파일 |
|---|---|
| 인덱스 | `outputs/goalstep/index/train.parquet` (30,804행) |
| 클래스 | `outputs/goalstep/index/action_registry.json` ← **학습이 읽는 것은 index 디렉토리 쪽** |
| 영상 | `data/Ego4D/v2/goalstep_videos/<video_uid>.mp4` |
| 피처 | `data/Ego4D/goalstep_feature_cache/train/*.pt` |
| 모델 | `checkpoints/vjepa2/vitl.pt`(frozen) + `AnticipationHead` |

`outputs/goalstep/taxonomy/action_registry.json`(세그먼트 레벨)과
`outputs/goalstep/index/action_registry.json`(Z=1 인덱스 레벨)은 별개 산출물이다.
현재 둘 다 98/188/390으로 일치하지만, **학습·채점의 단일 진실은 index 쪽**이다
(FHO와 동일하게 Z=1 train 인덱스에서 등록되므로).

### 7.2 채점 — 정답표

| 우선순위 | 파일 | 역할 |
|---|---|---|
| **① 정답표** | **`outputs/goalstep/index/val.parquet`** | 평가 샘플의 `verb_label`/`noun_label` 정수 = 정답 (goalstep_val.json 134 vid 유래) |
| ② 유효 조합 | `outputs/goalstep/index/action_registry.json` | (verb,noun)→action_id |
| ③ 이름 | `outputs/goalstep/taxonomy/goalstep_verbnoun_taxonomy.json` | id→텍스트 |

- 정답·예측·채점이 전부 우리 bespoke 라벨 공간 안에서 자기완결적이다. 파싱/병합맵을 바꾸면
  **정답표 자체가 바뀐다** → 학습 전에 클래스를 확정해야 한다.
- GoalStep 공식 서버(EvalAI)는 우리 라벨을 모르므로 무의미. `test_unannotated`는 사용하지 않는다.

---

## 8. 스모크 검증 (작업 1~6 end-to-end)

본 학습 전에 **영상 7개**(train 4 + val 3)로 작업 1~6을 1회 완주했다.
설정은 `configs/step1/goalstep/smoke.yaml`, run 디렉토리는 `outputs/goalstep/runs/smoke/`.
**수치 자체는 무의미**하다(영상 7개, 1 epoch) — 확인 대상은 배관이다.

| 단계 | 결과 |
|---|---|
| 인덱스 (`--video-uid-subset`) | train 1,409 / val 495 (train-seen 제한으로 784→495), verb 37 / noun 37 / action 64 |
| 영상 다운로드 | 7/7 성공 (540ss 6 + full_scale 1) |
| feature 추출 train | **1,409/1,409 saved, 실패 0**, 22분 (≈64 샘플/분) |
| feature 추출 val | **495/495 saved, 실패 0**, 7.4분 (≈67 샘플/분) |
| 학습 1 epoch | train_loss 1.2842, 466초 (캐시 1,409 샘플 + 100-subset 검증) |
| 체크포인트 | `checkpoints/epoch_01.pt`, `best.pt`, `latest.pt` 생성 확인 |
| 산출물 | `training_history.csv`(12컬럼), `metrics_per_epoch.json`, `final_metrics.json`, `val_subset_sample_ids.json`, `likelihood_entropy_epoch_01.jsonl`, `likelihood_entropy_full_val_best.jsonl` |

검증 지표가 verb/noun/action **각각** class-mean Recall@5 / Top-1 / Top-5로 전부 찍히는 것,
subset→full-val 최종 평가가 도는 것, band(head/mid/tail)·scenario breakdown이 나오는 것을 확인했다.

| (스모크, 참고용) | verb | noun | action |
|---|---|---|---|
| val 100-subset cmR@5 | 21.45 | 21.43 | 17.63 |
| val 전체(495) cmR@5 | 15.44 | 20.73 | 10.33 |
| val 전체 Top-1 / Top-5 | 14.14 / 51.92 | 27.68 / 69.29 | 18.59 / 44.24 |

→ **subset과 full의 괴리(action 17.63 → 10.33)가 §9.1에서 경고한 그대로 재현됐다.**
표본이 작을수록 class-mean 지표가 낙관적으로 뜬다. 본 학습에서도 subset은 **모델 선택용**으로만
쓰고 보고 수치는 full-val을 써야 한다.

### 8.1 본 학습 소요 시간 추정 (스모크 실측 기반)

| 단계 | 규모 | 추정 |
|---|---|---|
| 영상 다운로드 | 701개 / 272 GB | **1~4시간** (단일 스트림 실측 125 Mbps, 16 workers 병렬) |
| feature 추출 | 38,229 샘플 | **~10시간** (≈65 샘플/분), 1회성, 재개 가능 |
| 학습 10 epoch | epoch당 train 30,804 + val 500 | **~25시간** (epoch당 ≈2.5시간) |
| best.pt full-val 최종 평가 | 7,425 | ~35분 |
| **합계** | | **약 36~40시간, 디스크 ~600 GB** |

epoch당 2.5시간 추정 근거: 스모크가 1,409 샘플에 466초(≈3.0 샘플/초)였고, 동일 코드의
FHO-LTA 본 학습이 epoch당 85k 샘플에 20,000~27,000초(≈3.9 샘플/초)로 같은 대역이다.
병목은 캐시 `.pt`(샘플당 8.7 MB) 로드 + 4,352 토큰에 대한 attentive probe 순전파다.

**시간을 줄여야 하면** `--level step`(substep 제외)이 유일하게 안전한 knob이다.
샘플이 30,804 → 약 13,000으로 줄어 추출·학습이 대략 절반 이하가 되고, 시간적으로 겹치는
관측 창도 사라진다. 다만 taxonomy를 만든 레벨(step+substep)과 달라지므로 리포트에 명시해야 한다.
probe 구조·focal loss·입력 규격은 변경 금지 대상이라 손대지 않는다.

---

## 9. epoch별 결과 — **미실행 (사용자 지시 대기)**

본 학습(10 epoch)은 아직 돌리지 않았다. 사용자가 시작을 지시하면 아래 표를
`outputs/goalstep/runs/z1/training_history.csv` / `final_metrics.json`에서 채운다.

| epoch | train_loss | verb cmR@5 | verb Top-1 | verb Top-5 | noun cmR@5 | noun Top-1 | noun Top-5 | action cmR@5 | action Top-1 | action Top-5 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | | | | | | | | | | |
| … | | | | | | | | | | |
| 10 | | | | | | | | | | |

- **best epoch** = val-500-subset의 action class-mean Recall@5 최고 epoch (`best.pt`).
- 참고 기준선(같은 코드·다른 데이터라 **직접 비교 불가**): FHO-LTA 전체 학습에서 epoch 8이
  최종 채택, action class-mean Recall@5 = 8.03
  (`2026-07-17_ego4d-lta-full-training-results.md`). GoalStep은 action 클래스가 390개로
  FHO보다 훨씬 적으므로 수치가 더 높게 나오는 것이 정상이며, 이는 성능 향상이 아니라
  **라벨 공간이 좁아진 결과**다.

### 9.1 500-subset validation vs 최종 full-val

| 지표 | val 500-subset (epoch별) | val 전체 7,425 (best.pt 1회) | 차이 |
|---|---|---|---|
| verb class-mean Recall@5 | | | |
| noun class-mean Recall@5 | | | |
| action class-mean Recall@5 | | | |
| verb / noun / action Top-1 | | | |
| verb / noun / action Top-5 | | | |

- subset은 **500개 고정**(seed 42, `val_subset_sample_ids.json`)이라 epoch 간 비교는 정확하지만,
  **class-mean** 지표는 표본이 작을수록 희소 클래스가 몇 개만 등장해 분산이 크다.
  특히 action 390 클래스에 500 샘플이면 클래스당 평균 1.3개 → subset의 action cmR@5는
  full-val 값과 상당히 어긋날 수 있다. **모델 선택 기준으로만 쓰고, 보고 수치는 full-val을 쓴다.**
- Top-1/Top-5(micro accuracy)는 클래스 가중이 없어 subset↔full 편차가 훨씬 작다.

---

## 10. 관찰 · 한계

1. **단일 도메인.** GoalStep은 전부 요리(COOKING) 영상이다. scenario 축은 79개 goal_category로
   나뉘지만 도메인 다양성은 FHO-LTA(수백 scenario)보다 훨씬 좁다. 여기서 나온 수치는
   "요리 절차 예측" 성능이지 일반 egocentric anticipation 성능이 아니다.
2. **bespoke taxonomy.** verb 100 / noun 190 / action 390은 우리 파싱이 만든 공간이며
   FHO의 117/521도, GoalStep 공식 step 514개도 아니다. **어떤 외부 리더보드와도 직접 비교 불가.**
3. **EK100과 절대 수치 직접 비교 불가.** 클래스 수(EK100 verb 97/noun 300/action ~3.8k)와
   도메인·세그먼트 길이 분포가 모두 다르다. 같은 것은 *지표 정의*(class-mean Recall@5)뿐이다.
4. **파싱 노이즈.** step_category 후반부 구문을 spaCy로 파싱해 얻은 (verb, noun)이므로,
   전치사구를 버리는 등 정보 손실이 있다("pour water **into the pot**" → `pour/water`).
   OTHER 0.40%는 낮지만, 잘못 붙은 noun(문장 첫 명사구를 목적어로 오인)까지 잡아내지는 못한다.
5. **long-tail.** train 390 action 중 **94개가 10샘플 미만**, 최빈 action도 전체의 2.8%에 불과하다.
   focal loss(gamma 2.0)로 완화하지만 tail band Recall@5는 낮게 나올 것으로 예상된다.
6. **step/substep 혼합 — `--level both` 채택 (사용자 확정 2026-07-20).**
   verb/noun 클래스를 step+substep 양쪽에서 만들었으므로 인덱스도 동일 레벨을 쓴다.
   train Z=1 샘플 구성은 **step 12,622(41%) / substep 17,941(59%)** 로 substep이 다수다.

   채택 전 우려했던 "step과 첫 substep이 같은 시각에 시작 → 동일 관측 창 중복"은 실측 결과
   무시 가능한 수준이었다:

   | 항목 | 실측 |
   |---|---|
   | step·substep이 정확히 같은 시각에 시작 | **27건** / 고유 시작시각 31,436개 (0.09%) |
   | ↳ verb·noun 정답까지 동일 (완전중복, 제거됨) | 13건 |
   | ↳ 같은 창인데 정답이 다름 (라벨 노이즈로 잔존) | 14건 |
   | 직전 샘플과 관측 창이 부분적으로 겹침 | 6,449건 (20.9%) |
   | ↳ 시작시각 차이 0.5초 미만 | 1,730건 (5.6%) |

   즉 GoalStep 주석에서 substep은 대체로 부모 step보다 늦게 시작한다. 부분 겹침 20.9%는
   step 경계가 조밀한 구간에서 자연히 생기는 것으로, FHO-LTA도 같은 성질을 가진다
   (액션 간격 ~2초 < 관측창 3.5초). train/val이 **영상 단위**로 분리돼 있어 누수는 아니다.

   대안 `--level step`은 샘플이 30,804 → 약 12,600으로 줄어 추출·학습 시간이 2.5배 절약되지만,
   (a) taxonomy를 만든 레벨과 어긋나고 (b) 이미 심한 long-tail(390 action 중 94개가 10샘플 미만)이
   더 악화되므로 채택하지 않았다. 시간이 문제라면 feature는 `both`로 1회 추출해 두고
   (다른 레벨 실험에도 그대로 재사용 가능) epoch 수를 조절하는 편이 낫다.
7. **디스크·시간 비용(사전 확보 필요).**
   - 영상 272 GB(701개) + **feature cache 약 320 GB**
     (샘플당 `[4352, 1024]` fp16 ≈ 8.7 MB × 38,229 — FHO와 동일한 규격/크기).
     참고로 기존 FHO 캐시 `data/Ego4D/feature_cache_full`가 이미 761 GB를 쓰고 있다.
   - 클립이 아닌 20~80분짜리 원본에서 3.5초 창을 랜덤 액세스하므로 feature 추출이 파이프라인의
     지배적 비용이다(스모크 실측 **약 64 샘플/분** → 38,229 샘플 ≈ **10시간**, 1회성).
     `.pt` 캐시가 존재하면 스킵하므로 중단/재개는 안전하다. 학습 자체는 캐시에서 읽으므로 저렴하다.
8. **decord seek 경고(무해).** full_scale 원본 중 긴 것(예: `d2e05761…`, 1.2 GB)에서
   `Failed to skip frames effectively at frame N … Video might be corrupted or seeking failed`
   경고가 다수 뜬다. keyframe 간격이 넓어 decord가 순차 skip으로 폴백하는 것이며,
   **추출은 정상 완료된다**(스모크 train 1,409/1,409 saved, 실패 0). 다만 해당 영상 구간에서
   처리량이 눈에 띄게 떨어지므로 위 시간 추정에 여유를 두는 편이 좋다.
9. **val 68건 손실.** train에 없는 (verb,noun) 조합을 가진 val 샘플은 FHO 규칙대로 제외된다
   (7,493 → 7,425, 0.9%). 이 조합들은 애초에 예측 가능한 클래스가 아니다.

---

## 11. 재현 방법

```bash
source ~/ml_env/bin/activate
cd ~/Project/EGO

# (0) 주석 (v2_1 필수)
ego4d --datasets annotations --benchmarks goalstep --version v2_1 -o data/Ego4D -y

# (1) Phase 1·2 — 평탄화 CSV + verb/noun 클래스 (spaCy 필요)
python scripts/step1/goalstep/dump_goalstep_annotations.py \
  --annotations-dir data/Ego4D/v2/annotations --output-dir outputs/goalstep/inspection
python scripts/step1/goalstep/parse_goalstep_to_verbnoun.py \
  --annotations-dir data/Ego4D/v2/annotations --output-dir outputs/goalstep/taxonomy

# (2) 작업 1 — 오염검사 (교집합≠0이면 exit 1)
python scripts/step1/goalstep/check_overlap.py

# (3) 작업 2 — taxonomy + step별 라벨 + action registry
python scripts/step1/goalstep/build_goalstep_taxonomy.py

# (4) 작업 3 — Z=1 인덱스 (tau_a/L_obs/level 모두 인자 노출)
python scripts/step1/goalstep/build_goalstep_z1_index.py \
  --tau-a 1.0 --l-obs 3.5 --level both --output-dir outputs/goalstep/index

# (5) 영상 매니페스트 (v2_1) — uid 필터가 안 맞아도 manifest.csv는 받아진다
ego4d --datasets video_540ss --version v2_1 --video_uids 00000000-0000-0000-0000-000000000000 -o data/Ego4D -y
ego4d --datasets full_scale  --version v2_1 --video_uids 00000000-0000-0000-0000-000000000000 -o data/Ego4D -y

# (6) 영상 다운로드 (272 GB; --dry-run으로 먼저 확인)
python scripts/step1/goalstep/download_goalstep_videos.py \
  --uid-list outputs/goalstep/index/video_uids.txt \
  --manifest data/Ego4D/v2/video_540ss/manifest.csv \
  --manifest data/Ego4D/v2/full_scale/manifest.csv \
  --out-dir data/Ego4D/v2/goalstep_videos

# (7) 작업 4 — 피처 캐싱 (FHO 스크립트 재사용, 재실행 시 스킵)
python scripts/step1/ego4d_lta/extract_features.py --config configs/step1/goalstep/z1.yaml --split train
python scripts/step1/ego4d_lta/extract_features.py --config configs/step1/goalstep/z1.yaml --split val

# (8) 작업 6 — 학습 (10 epoch, 매 epoch 체크포인트 + 500-subset val, 종료 후 full-val)
python scripts/step1/goalstep/train_goalstep_z1.py --config configs/step1/goalstep/z1.yaml
```

- seed는 전부 42 (`experiment.seed`, `training.val_subset_seed`, 인덱스 `--seed`).
- AWS 키는 `~/.aws/credentials`에만 둔다. Ego4D 버킷 region은 **us-west-2** 고정
  (IAM 유저에 `s3:GetBucketLocation` 권한이 없어 자동 탐지 불가).
- 스모크 재현: 위 (4)~(8)에서 `--video-uid-subset`/`configs/step1/goalstep/smoke.yaml` 사용
  (§8 참조).
