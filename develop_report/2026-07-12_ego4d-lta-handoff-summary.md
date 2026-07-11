# Ego4D LTA Z=1 작업 인수인계 요약 (라이선스 발급 후 새 세션에서 이어가기 위한 문서)

- 작성일: 2026-07-12
- 브랜치: `feat/step1-ek100-assembly101-baseline` (커밋됨, **미푸시**)
- 관련 이전 리포트: `develop_report/2026-07-11_step1-ek100-assembly101-refactor.md`,
  `develop_report/2026-07-12_ego4d-lta-z1-scaffold.md`
- 이 문서의 목적: Ego4D License Agreement 승인 + AWS 자격증명 발급 후,
  **새로운 대화 세션**에서 바로 이어서 진행할 수 있도록 (1) 원래 요청한 전체
  내용, (2) 데이터 없이 이미 끝낸 것, (3) 다음에 할 것을 한 곳에 정리한다.

---

## Part 1. 처음에 요청한 내용 (원본 스펙 요약)

### 목표

V-JEPA2(frozen encoder + frozen predictor) 위에 attentive probe를 얹어
"1초 후 다음 action의 verb/noun/action"을 예측하는 기존 EK100 anticipation
파이프라인을, **아키텍처 변경 없이** Ego4D LTA 데이터의 Z=1(다음 1개
action만 예측) 형태로 재현한다.

> 대원칙: "골격(frozen encoder+predictor, attentive probe, 3-head, focal
> loss, class-mean Recall@5)은 그대로 두고, 데이터 로더 / 출력 차원
> (taxonomy) / loss 파라미터만 교체."

### 전제 / 입력 자산 (사용자가 준비해야 하는 것)

- Ego4D LTA 주석: `fho_lta_train.json`, `fho_lta_val.json`, `fho_lta_taxonomy.json`
- Ego4D 클립 원본(또는 full_scale 비디오) — Ego4D License Agreement 서명 후
  발급되는 AWS 자격증명으로 공식 `ego4d` CLI로 다운로드
- 각 클립의 scenario 메타데이터 소스: `ego4d.json`

### 요청했던 6개 작업

**작업 1 — LTA JSON 파서 & Z=1 샘플 생성 (`build_lta_z1_index.py`)**
- 각 action segment를 하나의 anticipation 타깃으로 변환.
- `tau_a=1.0`s(anticipation time), `L_obs=3.5`s(관찰 윈도우)를 config
  기본값으로, 둘 다 인자로 노출.
  - `obs_end_sec = action_start_sec - tau_a`
  - `obs_start_sec = obs_end_sec - L_obs`
- 경계 처리: `obs_start_sec < clip 시작`이면 (a) 가능한 만큼만 잘라서 flag
  기록, 또는 (b) 최소 관찰 길이(`min_obs_sec`) 미달 시 제외. 어떤 정책을
  썼는지 통계에 남길 것.
- 산출물 컬럼: `[video_uid, clip_uid, obs_start_sec, obs_end_sec, verb_label,
  noun_label, action_label, scenario, boundary_flag]` (parquet/csv).
- `action_label`: train에 등장한 `(verb,noun)` 조합만 dense index로 등록
  (등록 테이블도 저장). 조합 수를 콘솔·로그에 출력.
- taxonomy 로더: `fho_lta_taxonomy.json`에서 verb/noun id<->text 매핑,
  개수(`N_verb`, `N_noun`) 출력.
- 공식 split을 따르되, val을 내부 dev(80%)/heldout(20%)로 재분할하는 옵션
  (시드 고정).

**작업 2 — 클래스 분포 통계 (`analyze_lta_stats.py`)**
- verb/noun/action 각각 클래스별 빈도, head/mid/tail 대역 구분, Gini 또는
  imbalance ratio(max/min) → json + 막대그래프 png.
- verb-noun co-occurrence 행렬 저장(후속 단계용, 이번엔 모델 미반영).
- scenario별 샘플 수 분포.
- "pilot taxonomy" 옵션(`--top_noun 150 --top_verb 80`): 나머지는 제외
  (권장) 또는 "other" 매핑. 전체 taxonomy와 비교 불가함을 경고로 출력.

**작업 3 — 특징 캐싱 (`extract_features.py`)**
- `[obs_start_sec, obs_end_sec]` 구간을 V-JEPA2 encoder 입력 규격(고정
  프레임 수 T, 균등 샘플링)으로 리샘플. T/fps는 기존 EK100 값과 일치.
- frozen encoder+predictor forward → (encoder 토큰 ⊕ predictor 토큰) 캐싱.
  predictor는 EK100과 동일하게 "1초 미래 프레임 mask token" 구조 재사용.
- 레코드 id로 인덱싱해 디스크 캐시, 재실행 시 스킵.

**작업 4 — 모델(probe), 기존 코드 최소 교체**
- 기존 EK100 attentive probe를 그대로 import.
- 교체할 것은 오직: (a) 3개 linear head의 `out_features`를
  `N_verb/N_noun/N_action`으로, (b) focal loss의 `gamma`/`alpha`를 config로
  노출(EK100보다 long-tail이 심하므로 gamma를 키우는 A/B 가능하게).
- attention block, query token 수(=3)는 변경 금지.

**작업 5 — 학습 스크립트 (`train_lta_z1.py`), config 기반**
- config로 제어: `tau_a, L_obs, taxonomy(full/pilot), focal_gamma,
  focal_alpha, batch, lr, epochs, sampler(random|scenario_stratified), seed`.
- `scenario_stratified` 샘플러: 대형 시나리오가 배치를 지배하지 않도록.
- 평가: verb/noun/action 각각 class-mean Recall@5(기존 구현 재사용).
- 필수 로깅: 전체 class-mean Recall@5, head/mid/tail 대역별 breakdown,
  scenario별 breakdown(멀티도메인 편차 진단), focal 파라미터·taxonomy
  모드를 run 메타데이터로 기록.
- 체크포인트·config·지표를 run 디렉토리에 함께 저장.
- **likelihood와 entropy 값도 산출물로 저장.**

**작업 6 — 파일럿 우선 검증 절차 (`PILOT.md`)**
1. train의 10~20% subset + pilot taxonomy(top_noun~150, top_verb~80)로
   파이프라인 정합성·수렴·처리시간 실측(작업1~5 관통 스모크 테스트).
2. 정합성 확인되면 full taxonomy로 확장, focal loss로 재학습, 전체/대역별/
   시나리오별 지표 로깅.
3. 결과 해석 주의문(README/PILOT.md에 명시):
   - 클래스 수가 EK100의 수 배라 Recall@5 절대치를 EK100과 직접 비교 불가
   - pilot taxonomy 지표는 개발 가속용, 전체 taxonomy 결과와 비교 불가
   - 정확한 action 조합 수/샘플 수/시나리오 분포는 실제 집계값으로 표에 기재

### 산출물 요구사항
- 위 스크립트 + config 예시 + PILOT.md + 통계 결과(json/png).
- 각 스크립트는 `--help`로 인자 확인 가능, 소규모 subset에서 end-to-end
  1회 완주 확인.
- 기존 EK100 코드 수정 시 diff 최소화, LTA 전용 신규 파일로 분리 우선.

---

## Part 2. 데이터 없이 이미 완료한 것 (2026-07-12 세션)

Ego4D 데이터(annotations + clips)는 Ego4D License Agreement 서명 후 발급되는
AWS 자격증명이 있어야 공식 `ego4d` CLI로 다운로드할 수 있는데, 이 환경에는
아직 그 자격증명이 없어(발급 대기 중) **다운로드를 제외한 코드 작성 전체를
먼저 진행**했다.

### 구현 파일 목록

```
src/ego/datasets/ego4d.py          # taxonomy 로더, JSON 파서, Z=1 인덱스 빌더,
                                    # dev/heldout 분할, Ego4DLTADataset
src/ego/datasets/ego4d_stats.py    # 빈도/head-mid-tail/Gini/co-occurrence/pilot taxonomy
src/ego/datasets/video_sampling.py # sample_uniform_frame_indices() 추가 (기존 함수 불변)

scripts/step1/ego4d_lta/build_lta_z1_index.py   # 작업 1
scripts/step1/ego4d_lta/analyze_lta_stats.py    # 작업 2
scripts/step1/ego4d_lta/extract_features.py     # 작업 3
scripts/step1/ego4d_lta/train_lta_z1.py         # 작업 4+5 (probe 재사용 + 학습 루프)
scripts/step1/ego4d_lta/PILOT.md                # 작업 6

configs/step1/ego4d_lta/pilot.yaml
configs/step1/ego4d_lta/full.yaml

tests/unit/test_ego4d_lta_index.py   (12 tests)
tests/unit/test_ego4d_stats.py       (9 tests)
tests/unit/test_scenario_sampler.py  (5 tests)
tests/unit/test_video_sampling.py    (+3 tests, 기존 6개는 그대로)
```

### "아키텍처 변경 없음" 원칙 준수 현황 — 재사용 vs 신규

기존 EK100 코드에서 **한 줄도 수정하지 않고 그대로 import**한 것:
- `AnticipationHead` (`src/ego/step1_action_anticipation/models/anticipation_head.py`) —
  attentive probe(3-query-token) + verb/noun/action linear head. 출력
  차원(`num_verb_classes` 등)만 Ego4D 값으로 바꿔서 인스턴스화.
- `load_vjepa2_backbone` (frozen encoder+predictor wrapper)
- `sigmoid_focal_loss`, `_WarmupCosineLR`, `_CosineWD`
  (`src/ego/step1_action_anticipation/train.py`에서 import)
- `class_mean_recall`, `per_class_recall`, `prediction_entropy`
  (`src/ego/step1_action_anticipation/metrics.py`)
- `extract_and_cache_features`, `FeatureCacheDataset`, `anticipation_collate`
  (`src/ego/step1_action_anticipation/data/{feature_cache,collator}.py`) —
  Ego4D 전용 코드 0줄, 데이터셋의 `__getitem__` 스키마만 맞추면 그대로 동작
- `build_label_mapping` (`src/ego/datasets/label_mapping.py`) — "train에 등장한
  (verb,noun) 조합만 dense index로 등록"이 이미 이 함수가 하는 일이라 재사용

기존 파일에 **순수 추가만** 한 것 (기존 함수/동작 변경 없음):
- `src/ego/datasets/video_sampling.py`에 `sample_uniform_frame_indices()` 함수 추가

완전히 새로 만든 파일: 위 파일 목록의 나머지 전부 (LTA 전용, EK100 코드
비수정 원칙 준수).

### 검증 상태

- **유닛 테스트**: 신규 28개 전부 통과, 전체 스위트 21→52 passed(4 skipped
  동일 유지, 기존 EK100/Assembly101 테스트 회귀 없음). 합성 fixture로
  Z=1 경계 처리(truncate/exclude), `dev_fraction`에 따른 clip 단위
  dev/heldout 분할(결정성+미누출), label registry가 `build_label_mapping`과
  정확히 일치하는지, 통계 함수, `ScenarioStratifiedSampler`의 라운드로빈
  동작을 검증.
- **합성 데이터 end-to-end 드라이런**: 실제 Ego4D 스키마를 흉내 낸 가짜
  `fho_lta_{train,val}.json`/`taxonomy.json`/`ego4d.json`(60 클립, 20
  verb x 30 noun, 4 scenario)을 생성해 `build_lta_z1_index.py`→
  `analyze_lta_stats.py`(일반+pilot 모드)를 실제로 실행, 정상 동작 확인.
  가짜 feature cache(랜덤 텐서)를 만들어 `train_lta_z1.py`도 2 epoch 실제
  실행 — focal loss, scheduler, `scenario_stratified` 샘플러, band/scenario
  breakdown, checkpoint 저장, `likelihood_entropy.jsonl` 생성까지 배관이
  안 새는지 확인(수치 자체는 랜덤 특징이라 무의미, 파이프라인 완주 여부만
  검증한 것). 4개 스크립트 모두 `--help` 정상 확인.
- **검증하지 못한 것 (실데이터 필요)**: 실제 `fho_lta_*.json`의 정확한
  필드명이 `ego4d.py`의 `_FIELD_CANDIDATES` 가정과 일치하는지, 실제 clip
  비디오 디코딩, 실제 V-JEPA2 체크포인트로의 feature 추출, 실제 유의미한
  정확도 수치.

### 커밋 상태

```
3b768fd feat(step1): scaffold Ego4D LTA Z=1 pipeline reusing the EK100 architecture
77f2d4d feat(step1): implement EK100 + Assembly101 V-JEPA2 action anticipation baseline
```
브랜치 `feat/step1-ek100-assembly101-baseline`에 커밋됨, **origin에 푸시되지
않음**. 이 저장소의 git identity는 로컬(전역 아님)로
`user.name=Pumpkin0527`, `user.email=hogunpark1700@gmail.com`로 설정되어 있음.

---

## Part 3. 다음 세션에서 할 일 (자격증명 발급 후)

자세한 절차는 `scripts/step1/ego4d_lta/PILOT.md`를 그대로 따르면 된다. 여기선
순서만 요약:

### 0단계 — 자격증명 확인 및 데이터 다운로드
```bash
# Ego4D 홈페이지에서 발급받은 AWS access key/secret 설정 후
ego4d --output_directory <경로> --datasets lta annotations --version v2
# clips(또는 full_scale) 다운로드, 필요 시 ego4d.json도 함께
```
필요 파일: `fho_lta_train.json`, `fho_lta_val.json`, `fho_lta_taxonomy.json`,
`ego4d.json`, LTA 클립(또는 full_scale) 비디오.

### 1단계 — 파일럿 인덱스로 파서부터 검증
```bash
python scripts/step1/ego4d_lta/build_lta_z1_index.py \
    --taxonomy <path>/fho_lta_taxonomy.json \
    --train-json <path>/fho_lta_train.json \
    --val-json <path>/fho_lta_val.json \
    --ego4d-json <path>/ego4d.json \
    --train-clip-fraction 0.15 --top-verb 80 --top-noun 150 --pilot-mode exclude \
    --output-dir outputs/ego4d_lta/index_pilot
```
**여기서 실패한다면 십중팔구 실제 JSON 필드명이 가정과 다른 것** —
`src/ego/datasets/ego4d.py`의 `_FIELD_CANDIDATES` 딕셔너리에 실제 필드명을
한 줄 추가하면 된다 (에러 메시지에 그 레코드의 실제 키 목록이 그대로
출력되도록 이미 만들어둠).

### 2단계 — 통계로 클래스 분포 확인
```bash
python scripts/step1/ego4d_lta/analyze_lta_stats.py \
    --index outputs/ego4d_lta/index_pilot/train.parquet \
    --output-dir outputs/ego4d_lta/stats_pilot
```

### 3단계 — 파일럿 feature 추출 + 학습 스모크 테스트
```bash
python scripts/step1/ego4d_lta/extract_features.py --config configs/step1/ego4d_lta/pilot.yaml --split train
python scripts/step1/ego4d_lta/extract_features.py --config configs/step1/ego4d_lta/pilot.yaml --split dev
python scripts/step1/ego4d_lta/train_lta_z1.py --config configs/step1/ego4d_lta/pilot.yaml
```
`configs/step1/ego4d_lta/pilot.yaml`의 `dataset.video_root`,
`dataset.feature_cache_dir` 등 실제 다운로드 경로로 수정 필요. 클립당 처리
시간을 여기서 실측해서 전체 taxonomy 추출이 현실적인지 판단.

### 4단계 — 전체 taxonomy로 확장
`configs/step1/ego4d_lta/full.yaml` 사용, `--train-clip-fraction`/
`--top-verb`/`--top-noun` 없이 인덱스 재생성 → 동일 절차 반복.
Long-tail이 EK100보다 심하므로 `training.focal_gamma`를 2.0(EK100 기준값)
대비 3~4로 올리는 A/B도 시도해볼 것(이미 config에 주석으로 남겨둠).

### 확인해야 할 것 (완료 기준)
- [ ] `build_lta_z1_index.py`가 실제 파일로 정상 완주, `N_verb`/`N_noun`/
      등록된 action 조합 수가 콘솔에 출력됨
- [ ] `Ego4DLTADataset.__getitem__`이 실제 clip 비디오를 정상 디코딩
      (`video.shape == [3, frames_per_clip, resolution, resolution]`)
- [ ] `extract_features.py`가 실제 V-JEPA2 체크포인트로 feature 추출 완료
- [ ] `train_lta_z1.py` pilot 학습이 loss 감소 + 유의미한 Recall@5 도달
- [ ] `likelihood_entropy.jsonl`이 실제 값(NaN 아님)으로 채워짐
- [ ] full taxonomy로 확장 후 head/mid/tail, scenario별 breakdown 기록

### 알아두면 좋은 것
- 이 저장소(`/home/hogun/Project/EGO`)의 EK100 학습도 아직 전체 데이터로는
  못 돌렸다 (`data/EPIC-KITCHENS`에 validation-subset 44개 영상만 있음) —
  Ego4D LTA 작업과 별개 이슈이니 헷갈리지 말 것.
- 메모리 파일 `project_ego_step1_lta.md`(Claude 메모리 시스템)에도 이
  진행 상황이 요약되어 있어, 새 세션에서 관련 memory가 자동으로 로드되면
  이 문서와 같은 내용을 다시 보게 될 것이다.
