# Step 1 (EK100 / Assembly101) 리팩토링 리포트

- 날짜: 2026-07-11
- 범위: Step 1 — Action Anticipation (V-JEPA2, EK100 + Assembly101)
- 브랜치: `feat/step1-ek100-assembly101-baseline`

## 배경

`/home/hogun/Project/EvE/V-JEPA2`에서 V-JEPA2를 이용해 EK100·Assembly101 데이터로
action anticipation 학습/추론까지 결과를 확인했던 프로토타입 코드
(`evals/action_anticipation_frozen/*`, `scripts/*.py`)를, [hublemon/EGO](https://github.com/hublemon/EGO)
저장소의 구조에 맞춰 리팩토링했다. EGO 저장소는 클론 시점에 전부 스캐폴드(`TODO` 주석만 있는
placeholder) 상태였고, 이번 작업으로 Step 1 부분을 실제로 구현했다.

## 한 일

### 1. 저장소 세팅
- `https://github.com/hublemon/EGO.git`을 `/home/hogun/Project/EGO`에 클론
- `feat/step1-ek100-assembly101-baseline` 브랜치 생성

### 2. 데이터/모델 이동
- EK100(`data/EPIC-KITCHENS`, 10GB)·Assembly101(`data/Assembly101`, 293GB)·EK100 annotation
  (`data/annotations`)을 V-JEPA2 저장소에서 EGO로 **cut(mv)** — 같은 파티션이라 즉시 이동, 복사 없음
- V-JEPA2 backbone 소스(encoder/predictor/attentive pooler 등 구동에 필요한 최소 파일 집합,
  ~14개 파일)를 `third_party/vjepa2/`로 **실제 복사(vendor)** — 참조가 아니라 물리적으로 저장소에 포함
- V-JEPA2 체크포인트(`vitl.pt`, 4.8GB)와 기존에 학습되어 있던 EK100 분류기 체크포인트
  (`ek100-vitl-256.pt`, 204MB)를 `checkpoints/`로 복사
- 기존 V-JEPA2 저장소의 프로토타입 코드는 그대로 두고 손대지 않음 (참고용으로 보존)

### 3. Step 1 구현
| 영역 | 파일 | 내용 |
|---|---|---|
| 공통 유틸 | `src/ego/common/*` | config 로딩, JSONL/YAML IO, `[Step 1][Phase] ...` 로그 포맷, seed, path 해석 |
| 계약(contract) | `src/ego/contracts/*` | `Observation`, `ActionLabel`, `ActionCandidate`, `StepOneCandidateRecord` |
| 데이터셋 | `src/ego/datasets/{base,label_mapping,video_sampling,ek100,assembly101}.py` | 공통 인터페이스, 결정론적 라벨 매핑, observation/target 불변식을 보장하는 클립 샘플링 |
| 모델 | `step1_action_anticipation/models/{vjepa2_backbone,attentive_probe,anticipation_head}.py` | frozen encoder+predictor wrapper, attentive pooling, verb/noun/action head |
| 데이터 파이프라인 | `step1_action_anticipation/data/{build_samples,transforms,collator,feature_cache}.py` | dataset-agnostic 빌더, V-JEPA2 전처리 transform, Assembly101용 feature cache |
| 실행 스크립트 | `step1_action_anticipation/{prepare,train,infer,evaluate,metrics}.py` | `ego step1 prepare|train|infer|evaluate` 로 연결 |
| 레거시 호환 | `step1_action_anticipation/legacy_checkpoint.py` | 프로토타입에서 학습된 체크포인트를 새 구조로 로드 |
| 설정 | `configs/step1/{ek100_vjepa2,assembly101_vjepa2,inference}.yaml` | 실제 하이퍼파라미터/경로로 채움, `assembly101_vjepa2.yaml` 신규 작성 |
| 파이프라인 연결 | `pipelines/step1_to_step2.py` | `action_candidates.jsonl`을 스키마 검증하며 로드 |
| 테스트 | `tests/unit/*`, `tests/integration/*`, `tests/smoke/*` | 라벨 매핑/샘플링/지표 단위 테스트, 스키마 통합 테스트, tiny end-to-end smoke test |
| 문서 | `docs/*.md` | 설계 결정과 사용법 반영 |

### 4. 기존 프로토타입과 의도적으로 다르게 만든 부분
- **WebDataset 스트리밍 → map-style `Dataset`**: 멀티노드 SLURM 전제가 없는 단일 머신 구조에 맞춰 단순화
- **observation/target 불변식 강제**: 기존 학습 샘플링(`anticipation_point`)은 observation이 target
  action 구간 안으로 들어갈 수 있었음. `video_sampling.build_clip_window`는 항상
  `observation_end_sec <= target_start_sec`를 만족하도록 재설계 (`tests/unit/test_video_sampling.py`로 검증)
- **결정론적 라벨 매핑**: 기존 EK100 코드는 `enumerate(set(...))`이라 verb/noun 순서가
  이론상 비결정적. `label_mapping.build_label_mapping`은 항상 정렬 후 부여
- **verb/noun/action 체크포인트 독립 저장**: `best_verb.pt`/`best_noun.pt`/`best_action.pt`를
  각각의 최고 validation 지표 기준으로 저장 (기존엔 `latest.pt` 하나만 사용)
- **분류기 hyperparameter sweep 제거**: 기존엔 프로세스 하나가 학습률/weight decay 조합
  ~20개를 병렬로 학습. 이번엔 config 하나당 분류기 1개로 단순화 (필요하면 config를 여러 개 만들어 스윕)
- **Assembly101 feature cache 학습**: 293GB를 매 epoch 디코딩하는 대신, `training.use_feature_cache: true`
  설정 시 프리징된 backbone을 한 번만 돌려 토큰을 캐싱하고 그 이후 분류기만 학습 (기존
  `extract_features_a101.py` + `train_probe_a101.py` 2단계 구조를 config 하나로 통합)

## 검증

### 유닛/스모크/통합 테스트
```
21 passed, 4 skipped (skipped = step2/step3 placeholder, 이번 범위 아님)
```

### 실데이터 검증
- `EK100Dataset`: 실제 다운로드된 44개 영상(참가자 17명)에 대해 annotation 파싱 → 경로 해석 →
  라벨 매핑 → decord 디코딩 → V-JEPA2 transform까지 전 과정 정상 동작 확인
  (`video.shape == [3, 32, 256, 256]`, `missing_videos == 0`)
- `Assembly101Dataset`: 로컬에 있는 293GB 전체 데이터에 대해 manifest 빌드 성공
  (train 102개 영상 / 24,743 샘플, val 36개 영상 / 9,148 샘플, missing_videos = 0)
- V-JEPA2 체크포인트(`vitl.pt`) 로딩: encoder/predictor 모두 **missing key 0, shape mismatch 0**
- backbone → `AnticipationHead` forward pass 실제 실행, 출력 shape 확인
- `ego step1 train → infer → evaluate` CLI를 tiny 실데이터(P01_13 6개 세그먼트)로 전체 실행,
  스키마에 맞는 `action_candidates.jsonl` 생성 및 지표 계산까지 확인

### 학습된 모델로 실제 정확도 확인 (레거시 체크포인트, 새로 실행)
기존 리포지토리에서 실제 EK100 전체로 20 epoch 학습된 체크포인트
(`checkpoints/step1/legacy_ek100_vitl256/best_action.pt`)를 새 `AnticipationHead` 구조로
key remapping해서 로드(strict 로딩, missing/unexpected key 0개)하고, 옛 `sliding_window_anticipation.py`와
동일한 방식(1초 간격 슬라이딩 윈도우, 1초 뒤 action anticipate, GT와 top1/top5 비교)의
`scripts/step1/sliding_window_demo.py`를 새로 작성해 영상 2개에 대해 실행했다.

| 영상 | 샘플 수 | Verb Top1 / Top5 | Noun Top1 / Top5 |
|---|---|---|---|
| P01_13 | 93 | 40.9% / 86.0% | 46.2% / 83.9% |
| P02_13 | 29 | 27.6% / 69.0% | 31.0% / 86.2% |
| **전체** | **122** | **37.7% / 82.0%** | **42.6% / 84.4%** |

결과 CSV: `outputs/step1/legacy_demo/{P01,P02}/*_1sec.csv` (로컬에만 존재, git에는 미포함 — 아래 커밋 정책 참고)

## 알려진 제약사항
- 이 환경의 `data/EPIC-KITCHENS`에는 EK100 학습 스플릿 전체가 아니라 예전 데모용
  validation-subset 영상(44개)만 실제로 다운로드되어 있다. 따라서 지금 상태로
  `ego step1 train --config configs/step1/ek100_vjepa2.yaml`을 실행하면 학습용 영상이 0개라
  에러가 난다. 나머지 EK100 참가자 영상을 받아야 실제 baseline 학습이 가능하다
  (Assembly101은 전체가 이미 로컬에 있어 바로 가능).
- 레거시 체크포인트의 action_id 매핑은 원본 학습 코드의 `enumerate(set(...))` 알고리즘을
  그대로 재현해서 복원했다 (verb/noun은 새 코드의 정렬 방식과 우연히 100% 일치함을 확인했지만,
  action pair는 정렬 순서가 아니므로 원본 알고리즘을 별도로 복제함, `legacy_checkpoint.py` 참고).

## 다음 단계 제안
1. EK100 나머지 참가자 영상 다운로드 → `ego step1 prepare`로 전체 데이터 검증 → `ego step1 train`
2. Assembly101은 `training.use_feature_cache: true`로 feature 추출부터 시작 (293GB 전체 디코딩은
   비현실적이므로 캐시 우선)
3. Step 1 baseline 학습이 끝나면 `ego step1 infer` → `evaluate`로 정식 `action_candidates.jsonl`과
   `metrics.json` 생성, Step 2 연결(`pipelines/step1_to_step2.py`) 검증

## 커밋 정책
- 코드(데이터셋 로더, 모델 wrapper, 학습/추론/평가 스크립트, config, 테스트, vendored V-JEPA2 소스,
  문서)는 커밋 — 다른 사람이 데이터셋/체크포인트만 각자 준비하면 git에서 받은 그대로 바로 실행 가능
- `data/`, `checkpoints/*.pt`, `outputs/*` 등 대용량/생성물은 `.gitignore`로 로컬에만 유지 (커밋 안 함)
