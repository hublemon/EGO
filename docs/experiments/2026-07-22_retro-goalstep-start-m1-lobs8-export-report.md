# RETRO GoalStep `action_start-1s / 8s` 모델 이식 보고서

- 작성일: 2026-07-22
- Export ID: `RETRO-goalstep-start-m1-lobs8-best-action-top5`
- 송신 저장소: `/root/nvme/migration/jihun/EGO_jihun2`
- 수신 저장소: `/root/nvme/migration/jihun/EGO_jihun3`
- 상태: 복사 및 SHA-256 무결성 검사 완료

## 1. 결론

학습에 실제 사용된 **가공 annotation/index는 모델과 함께 복사됐다.** 구체적으로
train 30,374개와 validation 7,214개의 관찰 구간, 정수 라벨, scenario 및
boundary 정보가 각각 `train.parquet`와 `val.parquet`에 포함돼 있다.

다만 Ego4D가 배포한 원본 `goalstep_train.json`과 `goalstep_val.json` 자체를
export 폴더에 다시 복제한 것은 아니다. 원본 annotation, 영상 및 feature cache는
두 저장소의 공용 상위 데이터 디렉터리에 이미 있으며 `EGO_jihun3`에서도 같은
상대경로로 접근할 수 있다.

즉 다음 두 목적을 구분해야 한다.

- 모델 로딩, 동일 index 기반 평가 및 downstream 연결: 현재 export만으로 필요한
  모델·가공 annotation·registry가 전달됐다.
- 원본 GoalStep 계층 annotation에서 index를 처음부터 재생성: 공용
  `datasets/Ego4D/v2/annotations`도 함께 참조해야 한다.

## 2. 정확한 수신 위치

모든 export 파일은 다음 폴더에 있다.

```text
/root/nvme/migration/jihun/EGO_jihun3/
└── outputs/goalstep/exports/
    └── RETRO-goalstep-start-m1-lobs8-best-action-top5/
        ├── EXPORT_CONTRACT.md
        ├── SHA256SUMS
        ├── best_action_top5.pt
        ├── config_resolved.yaml
        ├── final_metrics.json
        ├── metrics_per_epoch.json
        ├── run_metadata.json
        ├── training_history.csv
        ├── val_subset_sample_ids.json
        └── index/
            ├── action_registry.json
            ├── build_stats.json
            ├── train.parquet
            ├── val.parquet
            └── video_uids.txt
```

전체 export의 논리 크기는 약 572MB다. `outputs/*`는 `.gitignore` 대상이므로
체크포인트와 산출물은 Git에 추가되지 않으며 `EGO_jihun3`의 Git 상태를 오염시키지
않는다.

## 3. 함께 전달된 annotation/index

### `index/train.parquet`

- 표본 수: 30,374
- 목적: 학습에 실제 투입된 sample-level annotation

### `index/val.parquet`

- 표본 수: 7,214
- 목적: 전체 validation sample-level annotation

두 parquet의 스키마는 동일하다.

| 필드 | 의미 |
|---|---|
| `video_uid` | 원본 Ego4D 영상 식별자 |
| `clip_uid` | clip 식별자 |
| `obs_start_sec` | 모델이 보는 관찰 구간 시작 시각 |
| `obs_end_sec` | 모델이 보는 관찰 구간 종료 시각 |
| `verb_label` | upcoming target action의 verb 정수 라벨 |
| `noun_label` | upcoming target action의 noun 정수 라벨 |
| `action_label` | upcoming target의 `(verb, noun)` 조합 라벨 |
| `scenario` | GoalStep scenario/category |
| `boundary_flag` | 영상 시작 경계 때문에 관찰 구간이 잘렸는지 여부 |

`build_stats.json`에 기록된 index 조건은 다음과 같다.

- anticipation gap (`tau_a`): 1.0초
- observation length (`l_obs`): 8.0초
- 최소 관찰 길이: 0.5초
- 경계 처리: `truncate`
- 중복 관찰 창 제거: 활성화
- seed: 42
- taxonomy: verb 81, noun 140, action 293
- train 영상: 570개
- validation 영상: 130개
- scenario: 79개

`action_registry.json`에는 verb/noun ID와 `(verb, noun) -> action ID` 대응이
들어 있다. `video_uids.txt`는 사용된 영상 UID 집합이다.

고정된 model-selection subset 2,000개의 ID는 index 폴더 밖
`val_subset_sample_ids.json`에 별도로 포함돼 있다.

## 4. 시간 및 정답 계약

target action을 `A2`라고 할 때 전달된 모델과 index의 의미는 다음과 같다.

```text
관찰: [A2.start - 9초, A2.start - 1초]
정답: A2
```

따라서 관찰 구간에 어떤 이전 행동이나 annotation 공백이 포함되더라도 정답은
관찰 구간의 행동이 아니라 **아직 시작하지 않은 다음 target action `A2`**다.
이는 action recognition이 아니라 action anticipation 설정이다.

영상 시작부에서 8초를 모두 확보할 수 없는 표본은 `boundary_policy=truncate`에
따라 짧아질 수 있으며 `boundary_flag=true`로 표시된다.

## 5. 모델 및 결과 산출물

`best_action_top5.pt`는 validation subset action Top-5를 기준으로 선택된 epoch 4
체크포인트다.

- subset 크기/seed: 2,000 / 42
- subset action Top-5: 26.90%
- 체크포인트 heads: verb, noun, action
- 클래스 수: 81 / 140 / 293
- 체크포인트 SHA-256:
  `b10ae8ffd125060d63cd8725c3986e26d48485452e3fcc0428f049581aad37b7`

원래 학습 프로세스가 종료 단계에서 자동 생성한 full-validation 결과도
`final_metrics.json`에 들어 있다. 이번 이식 과정에서 별도의 full-validation을
추가 실행하지 않았다.

## 6. export에 복제하지 않은 공용 자산

다음 자산은 크기가 크고 동일 저장공간에서 공유되므로 export에 복제하지 않았다.

| 자산 | 공용 절대경로 | 크기 |
|---|---|---:|
| Ego4D 원본 annotation 전체 | `/root/nvme/migration/jihun/datasets/Ego4D/v2/annotations` | 약 6.0GB |
| GoalStep 원본 영상 | `/root/nvme/migration/jihun/datasets/Ego4D/v2/goalstep_videos` | 약 256GB |
| start-1s/8s VNA feature cache | `/root/nvme/migration/jihun/datasets/Ego4D/goalstep_feature_cache_start_m1_lobs8_vna` | 약 313GB |
| V-JEPA2 backbone | `/root/nvme/migration/jihun/EGO_jihun/checkpoints/vjepa2/vitl.pt` | config에서 참조 |

특히 원본 dense annotation 파일은 다음 공용 위치에 존재한다.

- `datasets/Ego4D/v2/annotations/goalstep_train.json` (약 13MB)
- `datasets/Ego4D/v2/annotations/goalstep_val.json` (약 3.2MB)

`EGO_jihun3`에서 export의 `config_resolved.yaml`에 기록된 `../datasets/Ego4D/...`
경로를 사용하면 이 공용 자산을 그대로 참조할 수 있다.

## 7. 중복 및 무결성 확인

export의 다음 다섯 index 파일은 `EGO_jihun3`에 기존에 있던
`src/ego/step1_action_anticipation/goalstep/index_start_m1_lobs8/`의 파일들과
바이트 단위로 동일하다.

- `action_registry.json`
- `build_stats.json`
- `train.parquet`
- `val.parquet`
- `video_uids.txt`

중복 사본을 export 안에도 유지한 이유는 모델 파일 하나만 이동해도 정확히 어떤
sample/label registry로 학습됐는지가 분리되지 않도록 하기 위해서다.

`SHA256SUMS`에 등록된 모델·설정·지표·index 파일은 모두 무결성 검사를 통과했다.
