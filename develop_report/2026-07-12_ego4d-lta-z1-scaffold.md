# Ego4D LTA Z=1 파이프라인 스캐폴드 (코드만, 실데이터 미검증)

- 날짜: 2026-07-12
- 범위: Step 1 확장 — Ego4D LTA를 Z=1(다음 1개 action) 형태로, 기존 EK100 아키텍처 재사용
- 브랜치: `feat/step1-ek100-assembly101-baseline`
- 상태: **코드/테스트 완료, 실제 Ego4D 데이터로는 미검증** (AWS 자격증명 발급 대기 중)

## 배경

2026-07-11에 사용자가 EK100과 동일한 골격(frozen V-JEPA2 encoder+predictor,
attentive probe, 3-query-token, verb/noun/action 3-head, focal loss,
class-mean Recall@5)을 **아키텍처 변경 없이** Ego4D LTA 데이터의 Z=1 형태로
재현하는 6개 작업(파서/인덱스 빌더, 통계 스크립트, 특징 캐싱, probe 재사용,
학습 스크립트, PILOT.md)을 상세 스펙으로 요청했다. 확인 결과 이 환경에는
Ego4D 접근에 필요한 AWS 자격증명이 없어(라이선스 승인 대기 중) 데이터
다운로드는 보류하기로 했고, 이번 작업은 "데이터 다운로드를 제외한 코드
작성"으로 범위를 좁혀 진행했다.

## 한 일

### 라이브러리 코드 (`src/ego/`)
| 파일 | 내용 |
|---|---|
| `datasets/video_sampling.py` | `sample_uniform_frame_indices()` 추가 (기존 함수는 그대로, 순수 추가) |
| `datasets/ego4d.py` | taxonomy 로더, `fho_lta_*.json` 파서(방어적 필드 매칭), Z=1 인덱스 빌더(경계 정책 truncate/exclude), dev/heldout 재분할, `Ego4DLTADataset` |
| `datasets/ego4d_stats.py` | 클래스 빈도, head/mid/tail 대역, Gini, imbalance ratio, verb-noun co-occurrence, scenario 분포, pilot taxonomy 생성 |

### 스크립트 (`scripts/step1/ego4d_lta/`)
| 파일 | 내용 |
|---|---|
| `build_lta_z1_index.py` | 인덱스 생성 CLI. `--train-clip-fraction`(파일럿 subset), `--top-verb/--top-noun/--pilot-mode`(파일럿 taxonomy)까지 지원 |
| `analyze_lta_stats.py` | 클래스 분포 리포트(json + png), pilot 모드에서 "비교 불가" 경고 출력 |
| `extract_features.py` | `Ego4DLTADataset` + 기존 `extract_and_cache_features`를 **그대로 재사용**해 캐싱 |
| `train_lta_z1.py` | config 기반 학습 스크립트. `ScenarioStratifiedSampler`, head/mid/tail·시나리오별 Recall@5 breakdown, likelihood/entropy 저장 |
| `PILOT.md` | 파일럿→풀 taxonomy 검증 절차 + 결과 해석 주의사항 |

### 설정
`configs/step1/ego4d_lta/{pilot,full}.yaml` — `frames_per_clip=32`,
`frames_per_second=8`, `resolution=256`, `num_probe_blocks=4`,
`num_heads=16`는 EK100과 동일 값으로 고정(아키텍처/입력 규격 불변 원칙).

## 기존 EK100 코드 재사용 현황 (아키텍처 변경 없음 원칙 검증)

다음은 **한 줄도 수정하지 않고 import만** 했다:
- `AnticipationHead` (attentive probe, 3-query-token, verb/noun/action head) — 출력 차원만 Ego4D 것으로 교체
- `load_vjepa2_backbone` (frozen encoder+predictor)
- `sigmoid_focal_loss`, `_WarmupCosineLR`, `_CosineWD` (train.py에서 import)
- `class_mean_recall`, `per_class_recall`, `prediction_entropy` (metrics.py)
- `extract_and_cache_features`, `FeatureCacheDataset`, `anticipation_collate` (feature_cache.py, collator.py) — Ego4D 전용 코드 0줄

label mapping(`build_label_mapping`)도 재사용했다 — "train에 등장한 (verb,noun)
조합만 dense index로 등록"이 이미 그 함수가 하는 일이라 새로 만들지 않았다.

`ego.datasets.video_sampling.py`에 함수 하나를 **추가**한 것을 제외하면
기존 EK100/Assembly101 코드는 전혀 건드리지 않았다 (diff 최소화 원칙).

## 검증 (실데이터 없이 가능한 범위)

### Unit test — 42개 신규, 전부 통과
```
tests/unit/test_ego4d_lta_index.py   (12) — 파서, Z=1 경계 처리(truncate/exclude), 라벨 registry가
                                              build_label_mapping과 정확히 일치하는지, dev/heldout
                                              분할 결정성 및 clip 단위 미누출, scenario lookup 일관성
tests/unit/test_ego4d_stats.py        (9) — 빈도/대역/Gini/imbalance/co-occurrence/pilot taxonomy
tests/unit/test_scenario_sampler.py   (5) — 라운드로빈 샘플러 에폭 길이/결정성/소수 시나리오 비기아
tests/unit/test_video_sampling.py    (+3) — sample_uniform_frame_indices 추가분

전체 스위트: 21 -> 52 passed, 4 skipped 동일 유지 (기존 EK100/Assembly101 테스트 회귀 없음 확인)
```

### 합성(synthetic) 데이터로 4개 스크립트 전부 end-to-end 1회 완주
실제 Ego4D 스키마를 흉내 낸 가짜 `fho_lta_{train,val}.json` / `taxonomy.json` /
`ego4d.json`(60개 클립, 20 verb x 30 noun, 4개 scenario)을 생성해서:
1. `build_lta_z1_index.py` — 216개 action 중 168개 Z=1 샘플로 변환, 151개
   (verb,noun) 조합 등록, dev=5/heldout=1 분할까지 정상 동작
2. `analyze_lta_stats.py` — 일반 모드 + pilot 모드(top_verb=5, top_noun=8)
   둘 다 json/csv/png 정상 생성, pilot 경고 문구 정상 출력
3. 가짜 feature cache(랜덤 텐서, 실제 backbone/비디오 없이)를 만들어
   `train_lta_z1.py` 2 epoch 실행 — focal loss, scheduler, scenario-stratified
   sampler, band/scenario breakdown, checkpoint 저장, `likelihood_entropy.jsonl`
   까지 전부 정상 동작 확인 (수치 자체는 랜덤 특징이라 의미 없음, **배관이 안
   새는지**만 확인한 것)
4. 4개 스크립트 모두 `--help` 정상 출력 확인

### 검증하지 못한 것 (실데이터 필요)
- 실제 `fho_lta_*.json` 스키마가 `_FIELD_CANDIDATES`에 가정한 필드명과
  맞는지 — 안 맞으면 어떤 필드가 문제인지 콘솔에 정확히 나오도록
  방어적으로 작성해뒀음 (`ego.datasets.ego4d` 모듈 docstring 참고)
- 실제 clip 비디오 디코딩 (`Ego4DLTADataset.__getitem__`의 decord 경로)
- 실제 V-JEPA2 체크포인트로 Ego4D 클립 feature 추출
- 실제 타당한 정확도 수치

## 다음 단계

AWS 자격증명 도착 시 `scripts/step1/ego4d_lta/PILOT.md`의 절차대로:
1. `fho_lta_{train,val}.json`, `fho_lta_taxonomy.json`, `ego4d.json`, LTA 클립 다운로드
2. `build_lta_z1_index.py`를 실제 파일로 1회 실행 — `_FIELD_CANDIDATES`
   불일치가 있다면 여기서 바로 에러로 드러남 (수정은 `ego4d.py` 한 곳만
   고치면 됨)
3. PILOT.md 절차대로 파일럿(10~20% subset + top_verb=80/top_noun=150) →
   전체 taxonomy 순으로 진행
