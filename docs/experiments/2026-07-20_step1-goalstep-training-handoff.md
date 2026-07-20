# Step-1 (Action Anticipation) — Ego4D GoalStep 학습 Handoff

- 작성일: 2026-07-20
- 목적: 저장소를 clone 받은 사람이 **Ego4D GoalStep 데이터로 Step-1 anticipation 모델을
  바로 학습**할 수 있도록, 필요한 것과 실행 순서만 간단히 정리한다.
- 관련: `docs/experiments/2026-07-20_goalstep-step1-training-plan.md`(설계·우려),
  `src/ego/step1_action_anticipation/goalstep/taxonomy/GOALSTEP_TAXONOMY_METHOD.md`(라벨 생성 방법론)

---

## 1. 지금 상태 — 무엇이 준비되어 있나

**저장소에 이미 포함(추가 작업 불필요)**

| 항목 | 경로 | 내용 |
|---|---|---|
| 코드 | `src/ego/step1_action_anticipation/goalstep/*.py` | 파서·taxonomy/인덱스 빌더·트레이너 |
| **라벨 공간** | `.../goalstep/taxonomy/` | **verb 81 / noun 140 / action 293** + 클래스 CSV·registry |
| **학습 인덱스** | `.../goalstep/index/` | `train.parquet` **30,374** / `val.parquet` **7,214** |
| 설정 | `configs/step1/goalstep/{z1,smoke}.yaml` | 본 학습 / 스모크 |

> 라벨과 인덱스를 저장소에 넣어 둔 이유: 예전엔 gitignore된 `outputs/`에 있어서
> clone 하면 **코드는 있는데 학습할 라벨 공간이 없는** 상태였다.

**직접 받아야 하는 것**

| 항목 | 크기 | 비고 |
|---|---|---|
| V-JEPA2 백본 | 5.1GB | `checkpoints/vjepa2/vitl.pt` |
| GoalStep 영상 700개 | **~272GB** | `data/Ego4D/v2/goalstep_videos/<video_uid>.mp4` |
| 피처 캐시 | 수백 GB | 아래 2단계에서 생성 |
| (재생성 시에만) 주석 | 22MB | `ego4d --datasets annotations --benchmarks goalstep --version v2_1 -o data/Ego4D -y` |

---

## 2. 실행 순서 (4단계)

```bash
source ~/ml_env/bin/activate
cd ~/Project/EGO
```

### ① 영상 다운로드 (~272GB, 가장 오래 걸림)
```bash
python scripts/step1/goalstep/download_goalstep_videos.py \
  --plan goalstep_download_plan.json \
  --out-dir data/Ego4D/v2/goalstep_videos
```
인덱스가 참조하는 uid 목록은 `src/ego/step1_action_anticipation/goalstep/index/video_uids.txt`(700개).
재개 안전(`.part` → rename, 크기 일치 시 스킵)하므로 중단해도 다시 실행하면 이어받는다.

### ② 피처 추출 — **별도 단계다(트레이너가 대신 해주지 않는다)**
```bash
python scripts/step1/ego4d_lta/extract_features.py --config configs/step1/goalstep/z1.yaml --split train
python scripts/step1/ego4d_lta/extract_features.py --config configs/step1/goalstep/z1.yaml --split val
```
frozen V-JEPA2로 관측창을 1회 인코딩해 `data/Ego4D/goalstep_feature_cache/{train,val}/*.pt`에 저장.
이후 epoch마다 영상을 다시 디코딩하지 않는다.

### ③ 학습
```bash
python src/ego/step1_action_anticipation/goalstep/train_goalstep_z1.py \
  --config configs/step1/goalstep/z1.yaml
```
frozen 백본 + attentive-probe 헤드(verb/noun/action 3-way). 결과는 `outputs/goalstep/runs/z1/`.

### ④ (선택) 인덱스를 다시 만들고 싶을 때
```bash
python src/ego/step1_action_anticipation/goalstep/parse_goalstep_to_verbnoun.py \
  --min-action-count 10 --prune-on train          # spaCy en_core_web_sm 필요
python src/ego/step1_action_anticipation/goalstep/build_goalstep_taxonomy.py --level both
python src/ego/step1_action_anticipation/goalstep/build_goalstep_z1_index.py
```

---

## 3. 먼저 스모크로 검증하고 싶다면

영상 몇 개만 받아서 전체 배관을 확인할 수 있다:
```bash
ls data/Ego4D/v2/goalstep_videos/*.mp4 | xargs -n1 basename | sed 's/\.mp4$//' > /tmp/smoke_uids.txt
python src/ego/step1_action_anticipation/goalstep/build_goalstep_z1_index.py \
  --video-uid-subset /tmp/smoke_uids.txt --output-dir outputs/goalstep/index_smoke
python scripts/step1/ego4d_lta/extract_features.py --config configs/step1/goalstep/smoke.yaml --split train
python scripts/step1/ego4d_lta/extract_features.py --config configs/step1/goalstep/smoke.yaml --split val
python src/ego/step1_action_anticipation/goalstep/train_goalstep_z1.py --config configs/step1/goalstep/smoke.yaml
```
**실측 참고치**(영상 7개 · 1 epoch · 서브셋 63 action): FULL val 495샘플에서
verb cmr@5 13.86 / noun 20.64 / action 10.24. *품질 지표가 아니라 배관 확인용*이다.

---

## 4. 함정 (실제로 겪은 것들)

1. **taxonomy를 바꾸면 피처 캐시를 반드시 지울 것.**
   `.pt`에 `verb_id/noun_id/action_id`가 **구워져** 저장된다. 가지치기 전(390 action) 캐시를
   가지치기 후(293) 레지스트리로 쓰면 **에러 없이 엉뚱한 라벨로 학습**된다.
   ```bash
   rm -rf data/Ego4D/goalstep_feature_cache*      # taxonomy 변경 시
   ```
2. **트레이너는 캐시를 만들지 않는다.** ②를 건너뛰면
   `EgoConfigError: No cached features found under .../train` 로 즉시 실패한다.
3. **decord seek 경고.** 긴 영상에서
   `Failed to skip frames effectively ... Video might be corrupted` 가 나올 수 있다.
   `skipped=0`이면 저장 자체는 됐지만 관측창 프레임이 틀어졌을 수 있으니 **빈도를 확인**할 것.
4. **GoalStep에는 clip 레이어가 없다.** 타임스탬프가 video-relative라 FHO처럼
   `clip_256ss`를 쓸 수 없고 **원본 영상 전체**가 필요하다(`video_source: full_scale`).
   그래서 용량이 272GB로 크다.
5. **split은 train/val 뿐**이다(FHO의 train/dev/heldout 재분할 아님).

---

## 5. 라벨 공간 요약 (평가 해석에 필요)

- **verb 81 / noun 140 / action 293**, action은 **train에 실제 등장한 (verb,noun) 조합만** 등록.
- 롱테일 가지치기: **train 지원 ≤10인 108개 클래스 제거**(713 세그먼트, 1.82%).
  기준이 train 지원뿐이라 **val은 라벨 공간 결정에 관여하지 않는다(누출 없음)**.
- ⚠️ 293개 중 **23개는 val 샘플이 0**이라 class-mean 지표에 잡히지 않는다(support 0은 NaN 처리되어
  평균에서 제외되므로 지표가 오염되진 않는다). val ≤2인 클래스도 34개다.
  → **instance-level Recall@5를 함께 보고**하고, 보조로 "val 지원 ≥5 클래스 한정" 매크로를 병기 권장.
- 이 taxonomy는 FHO(117/521)도 GoalStep 공식 step(514)도 아닌 **자체 공간**이라
  **외부 SOTA와 직접 수치 비교 불가**.
