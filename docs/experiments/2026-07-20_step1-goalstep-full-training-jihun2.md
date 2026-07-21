# Step-1 GoalStep Z=1 본 학습 실행 및 산출물 안내 (`jihun2`)

- 작성일: 2026-07-20
- 실행 저장소: `EGO_jihun2`
- 실행 설정: `configs/step1/goalstep/z1_jihun2.yaml`
- 실행 스크립트: `scripts/step1/goalstep/run_full_jihun2.sh`
- 학습 코드: `src/ego/step1_action_anticipation/goalstep/train_goalstep_z1.py`
- 학습 종류: 스모크가 아닌 전체 GoalStep train/val 본 학습

## 1. 현재 실행 흐름

다운로드가 끝난 `../datasets/Ego4D/v2/goalstep_videos`의 영상을 재사용한다. 전체 파이프라인은
다음 순서로 실행된다.

1. GPU 0에서 전체 train 30,374개 피처 추출
2. GPU 1에서 전체 val 7,214개 피처 추출
3. 두 피처 추출이 모두 성공하면 GPU 0에서 15 epoch probe 학습
4. 매 epoch 고정된 val 500개 표본으로 평가 및 체크포인트 저장
5. 학습 종료 후 가장 좋은 체크포인트를 전체 val 7,214개로 최종 평가

피처 추출은 기존 `.pt`를 건너뛰므로 중단 후 동일 스크립트를 다시 실행해 이어갈 수 있다.
반면 현재 trainer에는 epoch 체크포인트를 지정하는 `--resume` CLI가 구현되어 있지 않다.
epoch `.pt` 안에는 optimizer state가 포함되지만, 학습 재개에는 trainer의 resume 로직 추가가 필요하다.

## 2. 학습 설정

| 항목 | 값 |
|---|---:|
| Seed | 42 |
| Epoch | 15 |
| Train batch size | 32 |
| Feature extraction batch size | 8 |
| Learning rate | 0.0003 |
| Weight decay | 0.0001 |
| Warmup | 1 epoch |
| Sampler | scenario-stratified |
| Focal loss gamma / alpha | 2.0 / 0.25 |
| Observation | 32 frames, 8 fps, 256 px |
| Anticipation time | 1.0 sec |
| 클래스 | verb 81 / noun 140 / action 293 |
| 매 epoch val | 고정 500개, seed 42 |
| 최종 val | 전체 7,214개 |

V-JEPA2 encoder/predictor는 frozen 상태다. 이번 실행은
`training.train_heads: [action]`인 **완전한 action-only probe**다. attentive probe query와
classifier를 action용으로 하나씩만 생성하며, 학습 loss·validation·확률/entropy 산출물도
action만 대상으로 한다. 캐시에는 원래 index의 verb/noun ID가 남아 있지만 trainer가 읽어
loss 또는 지표 계산에 사용하지 않는다.

Action class ID는 taxonomy에서 `(verb_id, noun_id)` 조합에 대응하지만, 이는 하나의 293-way
정답 클래스를 정의하기 위한 매핑이다. 최적화 objective는 action focal loss 하나뿐이며,
별도의 verb loss, noun loss, verb-noun matching loss는 존재하지 않는다.

## 3. epoch별 `.pt` 체크포인트

모든 epoch의 체크포인트가 별도 파일로 보존된다.

```text
outputs/goalstep/runs/z1_jihun2/checkpoints/
├── epoch_01.pt
├── epoch_02.pt
├── ...
└── epoch_15.pt
```

각 `epoch_NN.pt`에는 다음 값이 들어간다.

- `epoch`: 완료된 epoch 번호
- `model_state`: attentive-probe와 세 classification head 가중치
- `optimizer_state`: AdamW optimizer state
- `metric`: 해당 epoch의 val-subset action CMR@5
- `num_classes`: verb/noun/action 클래스 수

추가 대표 체크포인트는 run 디렉터리 바로 아래에 저장된다.

| 파일 | 의미 |
|---|---|
| `latest.pt` | 가장 최근에 완료된 epoch의 체크포인트로 매 epoch 덮어쓴다. |
| `best.pt` | 고정 val 500개에서 **action CMR@5**가 가장 높은 epoch로 갱신한다. |

`best.pt` 선정 기준은 Top-1이나 Top-5 accuracy가 아니라 action class-mean Recall@5임에
주의한다. Top-1/Top-5는 모든 epoch에서 함께 측정·기록한다.

## 4. 전체 학습 결과 저장 위치

모든 학습 결과의 기준 디렉터리는 다음과 같다.

```text
outputs/goalstep/runs/z1_jihun2/
├── checkpoints/
│   └── epoch_01.pt ... epoch_15.pt
├── best.pt
├── latest.pt
├── config_resolved.yaml
├── run_metadata.json
├── val_subset_sample_ids.json
├── training_history.csv
├── metrics_per_epoch.json
├── likelihood_entropy_epoch_01.jsonl
├── ...
├── likelihood_entropy_epoch_15.jsonl
├── likelihood_entropy_full_val_best.jsonl
├── final_metrics.json
└── logs/
    ├── pipeline.log
    ├── extract_train.log
    ├── extract_val.log
    ├── train.log
    ├── dashboard.log
    └── cloudflared.log
```

학습 단계가 아직 시작되지 않았으면 위 파일 중 `logs/`만 먼저 존재한다. 나머지는 피처 추출
완료 후 trainer가 순차적으로 생성한다.

### 주요 결과 파일

| 파일 | 내용 |
|---|---|
| `config_resolved.yaml` | 해당 실행에 사용한 전체 설정 사본 |
| `run_metadata.json` | 표본 수, taxonomy, seed, batch size 등 실행 메타데이터 |
| `val_subset_sample_ids.json` | 매 epoch 평가에 공통으로 사용하는 val 500개의 정확한 ID |
| `training_history.csv` | epoch별 action loss, action CMR@5/Top-1/Top-5, 소요 시간 |
| `metrics_per_epoch.json` | epoch별 상세 지표와 head/mid/tail 및 scenario breakdown |
| `likelihood_entropy_epoch_NN.jsonl` | 해당 epoch 예측의 likelihood/entropy 분석값 |
| `final_metrics.json` | best epoch, val-subset 지표, 전체 val 최종 지표 |

`training_history.csv`의 열은 다음과 같다.

```text
epoch, train_loss,
action_cmr@5, action_top1, action_top5,
seconds
```

Action Accuracy Top-1과 Top-5는 **모든 epoch가 끝날 때마다** 실행되고 같은 행에 기록된다.
Verb/noun metric은 계산하거나 저장하지 않는다.

## 5. 최종 평가

15 epoch가 끝나면 `best.pt`를 다시 로드해 전체 val 7,214개를 한 번 평가한다. 결과는
`final_metrics.json`의 `val_full`에 저장되고, 다음 지표를 포함한다.

- action class-mean Recall@5
- action instance-level Accuracy Top-1
- action instance-level Accuracy Top-5
- head/mid/tail band breakdown
- GoalStep goal-category 기반 scenario breakdown

## 6. 피처 캐시와 원본 데이터

피처 캐시는 저장소 밖의 공유 데이터 영역에 저장한다.

```text
../datasets/Ego4D/goalstep_feature_cache_jihun2/
├── train/<sample_id>.pt
└── val/<sample_id>.pt
```

각 캐시 `.pt`에는 frozen V-JEPA2 feature와 verb/noun/action label ID가 함께 들어 있다.
Action-only trainer는 이 중 `features`, `action_id`, `sample_id`를 사용한다. taxonomy를
변경하면 기존 캐시를 재사용하면 안 된다.

입력과 백본 경로는 다음과 같다.

| 종류 | 경로 |
|---|---|
| 원본 영상 | `../datasets/Ego4D/v2/goalstep_videos/*.mp4` |
| Train/val index | `src/ego/step1_action_anticipation/goalstep/index/` |
| V-JEPA2 backbone | `../EGO_jihun/checkpoints/vjepa2/vitl.pt` |

## 7. 로그 확인

```bash
cd /root/nvme/migration/jihun/EGO_jihun2

# 전체 단계 전환
tail -f outputs/goalstep/runs/z1_jihun2/logs/pipeline.log

# 피처 추출
tail -f outputs/goalstep/runs/z1_jihun2/logs/extract_train.log
tail -f outputs/goalstep/runs/z1_jihun2/logs/extract_val.log

# 15 epoch 학습 및 평가
tail -f outputs/goalstep/runs/z1_jihun2/logs/train.log
```

긴 Ego4D 영상에서 decord의 `Failed to skip frames effectively` 경고가 발생할 수 있다.
경고 자체는 프로세스 중단을 뜻하지 않으며, 캐시 개수와 최종 return code를 함께 확인한다.

## 8. 실시간 웹 UI

대시보드 구현은 `tools/goalstep_live_dashboard.py`에 있다. 다음 실측값을 5초마다 읽는다.

- train/val 피처 `.pt` 실제 파일 수와 진행률
- GPU utilization, memory, temperature
- 현재 epoch와 train loss
- epoch별 action CMR@5
- epoch별 action Accuracy Top-1/Top-5
- 파이프라인·추출·학습 로그

현재 임시 공개 주소:

<https://pound-archive-editor-sierra.trycloudflare.com>

이 주소는 계정 없는 Cloudflare Quick Tunnel이므로 실행 환경이나 tunnel 프로세스가 종료되면
만료된다. 장기 고정 주소가 필요하면 Cloudflare named tunnel 또는 별도 서버 배포가 필요하다.

## 9. 전체 파이프라인 재실행

```bash
cd /root/nvme/migration/jihun/EGO_jihun2
bash scripts/step1/goalstep/run_full_jihun2.sh
```

이 명령은 스모크 설정을 사용하지 않는다. 기존 feature cache는 건너뛰지만,
`training_history.csv`와 학습 체크포인트는 새 trainer 실행 시 같은 output directory에 다시
기록될 수 있으므로 이미 완료된 결과를 보존해야 한다면 먼저 run directory를 별도 이름으로
복사하거나 새 `experiment.output_dir`을 사용한다.

## 10. SSH 연결과 독립된 백그라운드 실행

VS Code를 닫거나 SSH 연결이 끊겨도 학습을 유지하려면 detached tmux 세션을 사용한다.

```bash
cd /root/nvme/migration/jihun/EGO_jihun2
bash scripts/step1/goalstep/start_tmux_jihun2.sh
```

세션 이름은 `ego_goalstep_jihun2`이며 다음 세 창으로 구성된다.

| 창 | 역할 |
|---|---|
| `pipeline` | 전체 피처 추출 후 action-only 15 epoch 학습 및 최종 평가 |
| `dashboard` | `0.0.0.0:7860` 실시간 UI |
| `tunnel` | 외부 공개 Cloudflare Quick Tunnel |

상태 확인과 접속 방법:

```bash
tmux ls
tmux list-windows -t ego_goalstep_jihun2
tmux attach -t ego_goalstep_jihun2
```

tmux에서 빠져나오되 작업은 유지하려면 `Ctrl-b`를 누른 뒤 `d`를 누른다. SSH 연결 종료는
tmux server를 종료하지 않으므로 내부 프로세스가 계속 실행된다. 단, 서버가 재부팅되면 tmux
세션도 종료되며 Cloudflare Quick Tunnel URL은 tunnel을 다시 시작할 때마다 변경된다.
