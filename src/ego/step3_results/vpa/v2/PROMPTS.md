# VPA v2 — arm별 프롬프트 · 코드 경로 원문

작성 2026-07-26 · 이 문서는 **실제 실행에 들어간 문자열을 코드에서 그대로 덤프**한 것이다.
수치는 `develop_report/2026-07-26_vpa_v2_results_handoff.md`, 설계 배경은 `METHODS.md`.

---

## 0. 한눈에 — 무엇이 arm을 가르는가

**단 하나의 함수** `common.build_prompt(sample, vocab, T, *, with_frames, candidate_mode)` 가
전 arm의 프롬프트를 만든다. arm 차이는 **인자 두 개**뿐이다.

| arm | 모델 | `with_frames` | `candidate_mode` | 이미지 전달 |
|---|---|---|---|---|
| `qwen_backbone` | Qwen3-VL-8B (무학습) | `True` | `vocab` | PIL → chat template |
| `qwen_blind` | Qwen3-VL-8B (무학습) | **`False`** | `vocab` | 없음 |
| `ours_full` | Qwen3-VL-8B + **sft_r15 LoRA** | `True` | `vocab` | PIL → chat template |
| `ours_wm1st` | 동일 (**같은 가중치**) | `True` | **`wm10_first`** | PIL → chat template |
| `frontier` | gemini-2.5-pro (API) | `True` | `vocab` | **base64 data URI** |

→ `qwen_backbone` vs `ours_full` = **가중치만** 다름 (LoRA 유무)
→ `ours_full` vs `ours_wm1st` = **프롬프트 한 블록만** 다름 (WM 힌트)
→ `qwen_backbone` vs `qwen_blind` = **이미지 유무 + 시스템 프롬프트 한 구절**만 다름
→ `qwen_backbone` vs `frontier` = **모델만** 다름 (프롬프트 문자열은 완전히 동일)

---

## 1. SYSTEM 프롬프트

### 1-1. 프레임을 보는 arm 전부 (`qwen_backbone` · `ours_full` · `ours_wm1st` · `frontier`)

```
You are a procedural planning assistant for egocentric cooking videos. You are given:
8 frames sampled at 2 fps from the last 4 seconds of first-person video, ending 1 second
before the next action begins; the goal; and the actions the camera-wearer has already
COMPLETED. Predict the next 3 actions, in temporal order, starting with the action that
begins immediately after the observation ends. Each action is '<verb> <noun>'. You MUST
copy labels verbatim from the candidate list and output EXACTLY 3 of them as a JSON array
of strings, and nothing else.
```

시간 계약(4초 창 · 2fps · **1초 전 종료**)을 프롬프트에 **명시**한다 — 모델이 "관측 직후가 아니라
1초 뒤부터 시작하는 행동"을 예측해야 함을 알아야 하기 때문.

### 1-2. `qwen_blind` — 한 구절만 교체

```diff
- 8 frames sampled at 2 fps from the last 4 seconds of first-person video, ending 1 second
- before the next action begins; the goal; and ...
+ no video (text context only); the goal; and ...
```

**나머지는 한 글자도 다르지 않다.** 프레임 기여분을 재는 대조군이므로 다른 변인을 두지 않는다.

---

## 2. USER 프롬프트

### 2-1. 공통 골격 (`vocab` 모드 — `qwen_backbone` · `ours_full` · `frontier` · `qwen_blind`)

```
GOAL: Make deep fried dish

ACTIONS ALREADY COMPLETED (in order):
  1. organize_(arrange) tool_(utensil)
  2. wash potato
  ...
  15. cut potato

CANDIDATE ACTION LABELS (choose only from these):
- add batter
  ... (293개) ...
- wrap leaf

Predict the next 3 actions as a JSON array of exactly 3 labels copied verbatim from the
candidate list, in order. Output ONLY the JSON array.
```

- `GOAL` = `goal_text` (GoalStep video-level goal)
- 완료 행동은 **최근 15개**만 (`observed[-15:]`) — 프롬프트 길이 제어
- 후보는 **293개 전체** (GoalStep verb×noun 어휘)

### 2-2. `ours_wm1st` — 후보 목록 뒤에 WM 블록 삽입

```
CANDIDATE ACTION LABELS (choose only from these):
- add batter
  ... (293개) ...
- wrap leaf

A video world model has ranked these 10 candidates for the FIRST next action only:
- check heat
- stir ingredient
- organize_(arrange) tool_(utensil)
- wash dish
- add oil
- add water
- clean surface
- wipe dish
- dispose item
- organize_(arrange) ingredient
Your 1st predicted action MUST be one of these 10. Actions 2 onward are unconstrained —
pick them from the full candidate list.

Predict the next 3 actions as a JSON array of exactly 3 labels copied verbatim from the
candidate list, in order. Output ONLY the JSON array.
```

**설계 의도**: EGO 프레임워크의 배포 형태("WM 제시 → LM 선택")를 재현한다. 단 WM 은 **바로 다음
1개 action 만** 예측하므로 2스텝 이후에는 제약을 걸 수 없다. 그래서 1스텝만 묶고 나머지는 열어 둔다.

**후보 10개의 출처** — `build_support.py`에서 Step1 anticipation 모델을 forward 해
`softmax(logits)["action"].topk(10)` 으로 뽑은 **실제 WM 출력**이다(`wm_scores` = 대응 확률).

**순서는 셔플이며, 이것이 반드시 유지되어야 하는 이유** —
`random.Random(f"shuffle:{sid}").shuffle(order)` (`build_support.py:157`, **split 분기 밖**이라
train·val 에 동일 적용). step2 SFT 는 `context_train.jsonl` 의 **이 셔플된 candidates 를 그대로
프롬프트에 넣어 학습**했다(`sft_r1.py:47,101` · `vlm.py:fmt_candidates`).

→ **VPA 에서 후보를 점수순으로 정렬해 주면 학습 분포 밖 입력이 된다.** 모델은 "목록 순서에 정보가
없다"는 전제로 학습됐으므로, 정렬된 목록을 주면 순위 정보가 있다고 오인해 첫 줄로 쏠릴 수 있다.
셔플 유지는 **step2 학습 조건과 평가 조건을 일치시키는 필수 요건**이며, 부수적으로 rank 비공개
원칙(`contracts.py`)도 만족한다. 실측으로 915/915 전부 점수 내림차순이 아니다.

**앵커링 실측** (703건): 예측 1번째가 **목록 첫 줄**과 같은 비율 17.9%(우연 10%) <
**실제 WM top-1**과 같은 비율 20.9% → 모델은 위치가 아니라 내용을 보고 고른다.
프롬프트 문구(`has ranked these 10 candidates`)는 그대로 둔다 — 2026-07-26 사용자 확정.

> ⚠ **폐기된 초기안**: T스텝 전부를 top-10 으로 제한하는 방식. 2·3번째 정답이 top-10 안에 있는
> 비율이 각각 53% / 56% 뿐이라 **절반은 정답을 출력할 수조차 없는** 불공정 설정이었다.

**검증 (실측)** — 예측이 WM top-10 안에 있는 비율:

| arm | 1번째 | 2번째 | 3번째 |
|---|---:|---:|---:|
| `ours_wm1st` | **98.5%** ← 제약 준수 | 87.2% | 79.5% |
| `ours_full` | 54.4% | 54.2% | 48.1% |
| `qwen_backbone` | 47.0% | 49.2% | 46.1% |

`ours_full` 이 WM 힌트를 받지 않았다는 **실측 증거**다(54.4% ≈ 무학습 백본 47.0%, wm1st 98.5%와 확연히 다름).

---

## 3. 이미지 전달 방식 — 여기만 arm 계열별로 다르다

프레임 자체는 **완전히 동일**하다: 같은 캐시(`frame_cache_w4_g1_n8_s336/{video_uid}/{sample_id}/f{0..7}.jpg`),
같은 8장, 같은 336px. 전달 형식만 API/로컬이 다르다.

### 3-1. 로컬 (`run_local_vlm.py`) — PIL 이미지를 chat template 에

```python
content = [{"type": "image", "image": im} for im in images]   # PIL.Image 8개
content.append({"type": "text", "text": user})
messages = [{"role": "system", "content": [{"type": "text", "text": system}]},
            {"role": "user",   "content": content}]
inputs = processor.apply_chat_template(messages, add_generation_prompt=True,
                                       tokenize=True, return_dict=True, return_tensors="pt")
out = model.generate(**inputs, max_new_tokens=96, do_sample=False)   # greedy
```

### 3-2. frontier (`run_frontier.py`) — base64 data URI 를 OpenAI 호환 포맷으로

```python
uris = ["data:image/jpeg;base64," + base64.b64encode(p.read_bytes()).decode() for p in paths]
payload = {
  "model": "gemini-2.5-pro", "temperature": 0,
  "messages": [
    {"role": "system", "content": system},
    {"role": "user", "content": [*[{"type": "image_url", "image_url": {"url": u}} for u in uris],
                                 {"type": "text", "text": user}]},
  ],
}
requests.post(f"{base_url}/chat/completions", headers={"Authorization": f"Bearer {key}"}, json=payload)
```

**이미지가 실제로 쓰이는지 검증함** — 텍스트를 고정하고 프레임만 다른 영상 것으로 바꾸자 출력이
그 영상 내용(`flip flatbread`)으로 바뀌었다. 텍스트에 `flatbread` 라는 단어는 없었다.

---

## 4. 코드 경로

```
common.py
  ├── build_prompt(sample, vocab, T, with_frames=, candidate_mode=)  ← 전 arm 단일 진입점
  ├── parse_prediction(raw, T)      코드펜스/산문 허용, JSON 배열 추출
  ├── normalize_label(x)            NFKC · lower · 공백정규화 · 양끝 문장부호 제거
  └── map_to_vocab(x, vset, vlist)  정확일치 → 정규화 → difflib 최근접(cutoff .6)

frames.py
  ├── cache_dirname()               "frame_cache_w4_g1_n8_s336"  ← 창 규격이 키에 포함
  └── extract(sample)               decord 로 [start−5s, start−1s] 8장 균등 · 짧은변 336

run_local_vlm.py   --mode {frames,blind} --adapter <peft> --candidates {vocab,wm10_first}
  └── load_model() → (base + LoRA) → gb10_compat.apply()  ⚠ 반드시 로딩 **후**

run_frontier.py    --max-calls 500 --subset <json>
  └── resume: records.jsonl 의 ok=true sample_id 만 skip (실패 행 보존)

evaluate.py        SR / mAcc / mIoU + video-cluster bootstrap
paired.py          두 arm 차이의 CI (같은 표본에서 나란히 재표집)
```

### 4-1. 생성 설정

| | 로컬 arm | frontier |
|---|---|---|
| 디코딩 | `do_sample=False` (greedy) | `temperature: 0` |
| 최대 토큰 | 96 | 기본값 |
| 재시도 | 없음(예외는 오답 기록) | 429/5xx 지수 백오프 최대 5회 |

**전 arm 결정론적 디코딩** — 샘플링 분산이 arm 비교에 섞이지 않게 한다.

---

## 5. 출력 파싱 · 채점

1. 모델 출력에서 **JSON 배열 추출** (` ```json ` 코드펜스, 앞뒤 산문 허용)
2. 각 라벨 `normalize_label` → 어휘에 없으면 `map_to_vocab` 으로 최근접 매핑
3. 길이가 T 미만이면 빈 문자열 패딩(= 오답)
4. `per_sample(gt, pred, T)` → `(SR 0/1, 위치일치 수, IoU)`

**실측 매핑 품질 (frontier)**: 정확일치 668 · 퍼지매핑 13(`add_water`→`add water` 등 밑줄→공백) ·
미매핑 0 → 채점이 매핑 아티팩트에 오염되지 않았다.

---

## 6. 재현

```bash
export PYTHONPATH=src
P=~/ml_env/bin/python

# qwen_backbone
$P -m ego.step3_results.vpa.v2.run_local_vlm --gt runs/vpa_v2/vpa_v2_T3.json \
   --mode frames --out runs/vpa_v2/preds/qwen_backbone_T3

# qwen_blind
$P -m ego.step3_results.vpa.v2.run_local_vlm --gt runs/vpa_v2/vpa_v2_T3.json \
   --mode blind  --out runs/vpa_v2/preds/qwen_blind_T3

# ours_full  (WM 힌트 없음)
$P -m ego.step3_results.vpa.v2.run_local_vlm --gt runs/vpa_v2/vpa_v2_T3.json \
   --mode frames --adapter outputs/step2_retrospection/cesft_v2/sft_r15/adapter \
   --candidates vocab --out runs/vpa_v2/preds/ours_sft_r15_T3

# ours_wm1st (1스텝만 WM top-10 제약)
$P -m ego.step3_results.vpa.v2.run_local_vlm --gt runs/vpa_v2/vpa_v2_T3.json \
   --mode frames --adapter outputs/step2_retrospection/cesft_v2/sft_r15/adapter \
   --candidates wm10_first --out runs/vpa_v2/preds/ours_sft_r15_wm1st_T3

# frontier (키는 환경변수 전용)
FRONTIER_API_KEY=... $P -m ego.step3_results.vpa.v2.run_frontier \
   --gt runs/vpa_v2/vpa_v2_T3.json --subset runs/vpa_v2/frontier_subset_T3.json \
   --out runs/vpa_v2/preds/frontier_T3 --max-calls 500
```
