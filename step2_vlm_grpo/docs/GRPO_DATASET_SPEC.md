# GRPO 학습용 데이터셋 생성 명세

## 목적

Step 2 (GRPO 강화학습) 에서 VLM을 파인튜닝하기 위한 오프라인 데이터셋 생성.
WM(V-JEPA2)의 action anticipation 출력을 reward signal로 활용해
VLM이 WM 예측과 정렬되도록 학습시키는 것이 목표다.

---

## 데이터셋 설계 원칙

### Train set 사용 이유
- EK100 validation set은 Step 3 (WM+VLM 최종 성능 평가) 에 보존
- V-JEPA2가 이미 학습한 train set에서 WM 출력을 추출해 GRPO 학습 데이터로 사용
- validation set으로 step 3 평가 시 WM+VLM 성능을 오염 없이 측정 가능

### Anticipation 시점
- action 끝나기 **1초 전** = `stop_frame - int(1.0 * fps)` 를 trigger frame으로 사용
- 이 시점에서 V-JEPA2 encoder + predictor를 실행해 Top-5 예측 추출

### 샘플 수
- 목표: **5,000 samples** (GRPO 검증을 위한 최소 규모)
- EK100 train set 전체 약 67,000개 중 랜덤 샘플링
- 단, 아래 필터링 조건 적용 후 5,000개 확보

### 필터링 조건
```python
# 1. action 길이가 충분해야 1초 전 프레임이 존재
#    stop_frame - start_frame > fps * 1.5 (최소 1.5초 이상 액션)
# 2. trigger frame이 0보다 커야 함
# 3. 프레임 파일이 실제로 존재해야 함
```

---

## 샘플 포맷 (JSONL)

각 라인 = 하나의 학습 샘플

```json
{
  "sample_id": "P01_01_0123",
  "split": "train",
  "video_id": "P01_01",
  "narration_id": "P01_01_0123",
  "trigger_frame": 3540,
  "trigger_timestamp": "00:01:59.00",
  "frame_path": "data/grpo_dataset/frames/P01_01_0123.jpg",
  "task_goal": "make scrambled eggs",
  "gt_label": {
    "action": "crack egg",
    "verb": "crack",
    "noun": "egg",
    "verb_class": 2,
    "noun_class": 17,
    "action_class": 42
  },
  "wm_output": {
    "top5_verb": [
      {"rank": 1, "verb": "crack",  "verb_class": 2,  "likelihood": 0.412},
      {"rank": 2, "verb": "put",    "verb_class": 0,  "likelihood": 0.198},
      {"rank": 3, "verb": "take",   "verb_class": 1,  "likelihood": 0.143},
      {"rank": 4, "verb": "open",   "verb_class": 3,  "likelihood": 0.089},
      {"rank": 5, "verb": "wash",   "verb_class": 8,  "likelihood": 0.041}
    ],
    "top5_noun": [
      {"rank": 1, "noun": "egg",    "noun_class": 17, "likelihood": 0.523},
      {"rank": 2, "noun": "bowl",   "noun_class": 4,  "likelihood": 0.201},
      {"rank": 3, "noun": "pan",    "noun_class": 22, "likelihood": 0.134},
      {"rank": 4, "noun": "fridge", "noun_class": 11, "likelihood": 0.078},
      {"rank": 5, "noun": "knife",  "noun_class": 14, "likelihood": 0.031}
    ],
    "top5_action": [
      {"rank": 1, "action": "crack egg",   "verb_class": 2,  "noun_class": 17, "action_class": 42, "likelihood": 0.387},
      {"rank": 2, "action": "put egg",     "verb_class": 0,  "noun_class": 17, "action_class": 18, "likelihood": 0.156},
      {"rank": 3, "action": "take egg",    "verb_class": 1,  "noun_class": 17, "action_class": 31, "likelihood": 0.098},
      {"rank": 4, "action": "crack bowl",  "verb_class": 2,  "noun_class": 4,  "action_class": 43, "likelihood": 0.072},
      {"rank": 5, "action": "open fridge", "verb_class": 3,  "noun_class": 11, "action_class": 67, "likelihood": 0.044}
    ],
    "gt_in_top5_verb":   true,
    "gt_in_top5_noun":   true,
    "gt_in_top5_action": true
  },
  "memory_context": {
    "task_history": ["open fridge", "take egg", "close fridge"],
    "temporal_proximity": {
      "t-0.5s": "take egg",
      "t-1.0s": "take egg",
      "t-2.0s": "open fridge"
    }
  }
}
```

---

## 생성 파이프라인

```
EK100 train CSV
      ↓
① select_train.py    — 필터링 후 5,000 샘플 선택
      ↓
② vjepa_infer_train.py  — V-JEPA2로 trigger frame에서 Top-5 예측 + likelihood 추출
      ↓
③ extract_frame_train.py — trigger frame 이미지 저장 (JPEG)
      ↓
④ extract_memory_train.py — task_history + temporal_proximity 추출
      ↓
⑤ assemble_train.py  — 위 결과 합쳐 grpo_dataset.jsonl 생성
      ↓
⑥ analyze_train.py   — GT hit rate, likelihood 분포 등 통계 출력
```

---

## 파일 구조 (완료 시 기대 상태)

```
~/work/jihun/EGO/
├── data/
│   └── grpo_dataset/
│       ├── grpo_dataset.jsonl        # 5,000 샘플 (메인 출력)
│       ├── frames/                   # trigger frame 이미지
│       │   ├── P01_01_0123.jpg
│       │   └── ...
│       └── stats/
│           ├── hit_rate.json         # GT in Top-5 비율
│           └── likelihood_dist.png   # likelihood 분포 시각화
└── make_grpo_dataset/
    ├── select_train.py
    ├── vjepa_infer_train.py
    ├── extract_frame_train.py
    ├── extract_memory_train.py
    ├── assemble_train.py
    └── analyze_train.py
```

---

## 각 스크립트 구현 명세

### ① select_train.py
```python
# 입력: EPIC_100_train.csv
# 출력: data/grpo_dataset/selected_train.jsonl (5,000개)

# 로직:
# 1. EPIC_100_video_info.csv에서 fps 매핑 테이블 구성
# 2. 필터링: (stop_frame - start_frame) > fps * 1.5
# 3. trigger_frame = stop_frame - int(1.0 * fps)
# 4. trigger_frame > 0 확인
# 5. 필터링 통과한 샘플에서 random seed=42로 5,000개 샘플링
# 6. task_goal은 같은 video_id의 첫 번째 narration으로 설정
```

### ② vjepa_infer_train.py
```python
# 입력: selected_train.jsonl
# 출력: data/grpo_dataset/predictions_train.jsonl

# 기존 make_samples/vjepa_infer.py를 train 버전으로 수정
# 핵심 변경: likelihood 값을 반드시 추출해서 저장
# V-JEPA2 probe의 softmax 출력값 = likelihood
# verb / noun / action 각각의 Top-5 class + score 저장
```

### ③ extract_frame_train.py
```python
# 입력: selected_train.jsonl
# 출력: data/grpo_dataset/frames/{sample_id}.jpg

# trigger_frame을 실제 영상에서 추출
# 기존 extract_frame.py와 동일 로직, 경로만 변경
```

### ④ extract_memory_train.py
```python
# 입력: selected_train.jsonl + EPIC_100_train.csv
# 출력: data/grpo_dataset/memory_train.jsonl

# MEMORY_CONTEXT_SPEC.md의 get_task_history(), get_temporal_context() 재사용
# train CSV 기준으로 동일하게 적용
```

### ⑤ assemble_train.py
```python
# 입력: selected_train.jsonl + predictions_train.jsonl
#        + frames/ + memory_train.jsonl
# 출력: data/grpo_dataset/grpo_dataset.jsonl

# 위 포맷 기준으로 병합
# gt_in_top5_verb/noun/action 플래그 계산해서 포함
```

### ⑥ analyze_train.py
```python
# 출력 예시:
# === GRPO Dataset Stats ===
# Total samples:         5,000
# GT in Top-5 verb:      X,XXX / 5,000 (XX.X%)
# GT in Top-5 noun:      X,XXX / 5,000 (XX.X%)
# GT in Top-5 action:    X,XXX / 5,000 (XX.X%)
# Mean action likelihood (rank-1): 0.XXX
# Samples where rank-1 == GT:      X,XXX (XX.X%)
```

---

## 주의사항

### 1. Likelihood 추출 방법
V-JEPA2 probe의 출력은 logit 형태이므로 softmax를 적용해 확률값으로 변환:
```python
import torch.nn.functional as F
probs = F.softmax(logits, dim=-1)  # shape: (num_classes,)
top5_probs, top5_indices = probs.topk(5)
```
likelihood 값이 추출 불가능한 경우 `null`로 저장하고 계속 진행.

### 2. 배치 처리로 속도 최적화
5,000 샘플을 1개씩 처리하면 매우 느림.
기존 vjepa_infer.py의 배치 처리 구조를 그대로 활용:
```python
# batch_size = 기존 설정값 유지
# 단, 진행 상황을 tqdm으로 표시할 것
```

### 3. 프레임 저장 용량
5,000장 × 약 50KB = 약 250MB → 문제 없음

### 4. task_goal 설정
EK100은 per-video narration이 연속으로 있어 전체 태스크 목표가 명시되지 않음.
같은 video_id 내 첫 번째 narration의 동사+명사로 task_goal을 대리 정의:
```python
task_goal = f"{first_narration['verb']} {first_narration['noun']}"
# 예: "make scrambled eggs" 대신 "take knife" 가 될 수 있음
# 데모 단계에서는 이걸 수동으로 오버라이드 가능
```
