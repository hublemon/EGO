# VPA v2 — 평가 방법 정리 (arm · 프롬프트 · 계약)

작성 2026-07-26 · 코드 `src/ego/step3_results/vpa/v2/` · 산출물 `runs/vpa_v2/`
결과·수치는 `REPORT_v2.md`, 설계 근거는 `develop_report/2026-07-25_vpa_v2_frame_conditioned_plan.md`.

---

## 1. 태스크

**VPA (Patel et al., ICCV 2023)** — 목표(goal)와 지금까지의 관찰이 주어지면 **다음 T개 action을
순서대로** 예측한다. T=3, 4.

우리 이식은 Ego4D GoalStep(요리)에 올렸고, v1(text-conditioned)과 달리 **실제 영상 프레임을 입력**한다.

## 2. 전 arm 공통 계약 — 여기서만 분기한다

`common.py` 하나에 모아 두었다. arm마다 창이나 프롬프트가 달라지면 표 내부 비교가 무너지기 때문이다.

| 항목 | 값 | 근거 |
|---|---|---|
| 관측창 | `[target_start − 5s, target_start − 1s]` (4초) | 원본 δ=4s 유지 + **1초 안전 간격** |
| 프레임 | 8장 @ **2 fps**, 짧은 변 336px | step2 학습 입력(8프레임@336)과 **토큰 형상 동일** |
| 미래 오염 | **없음** — 관측이 예측 대상 시작 1초 전 종료 | 빌더 assert로 강제 |
| 후보 어휘 | GoalStep verb×noun **293** (step2와 동일 계보) | 라벨 공간 통일 |
| 출력 형식 | `["label1", ..., "labelT"]` JSON 배열 | 전 arm 동일 |

> 원본 VLaMP 공식 구현은 `[start−2s, start+2s)`를 써서 **첫 예측 대상의 시작 후 2초가 입력에 들어간다**
> (코드 주석 `# ISSUE: This is a hack`). 우리는 논문 §3 정의를 따르고 그 구현을 따르지 않는다.

## 3. 표본 선정

`runs/cesft_v2/data/context_val.jsonl` → 네 조건:

| 조건 | T=3 남은 수 |
|---|---:|
| **heldout split** (dev 제외 — dev는 step2 학습 중 probe 대상) | 5,326 |
| **covered**: GT(다음 action) ∈ WM top-10 | 2,313 (43.4%) |
| 미래 action ≥ T개 | 1,064 |
| anticipation gap ≤ 5초 | **915** (71영상) |

T=4는 504샘플 / 54영상.

**covered 필터는 1번째 action에만 걸린다.** 실측 포함률 — 1번째 100%(강제), 2번째 53%, 3번째 56%.
→ 첫 스텝이 인위적으로 쉬우므로 **절대 수치를 "GoalStep VPA 성능"으로 일반화하지 말 것.**
편향이 전 arm에 동일하게 걸리므로 **arm 간 비교(특히 paired)는 유효**하다.

## 4. arm 목록

### 4-1. 모델 arm — 모두 같은 8프레임·같은 프롬프트

| arm | 모델 | 프레임 | 후보 제시 | 역할 |
|---|---|---|---|---|
| `frontier_gemini25pro` | gemini-2.5-pro (API) | ✅ 8장 | 전체 293 | 외부 기준점 |
| `qwen3vl_8b_frames` | Qwen3-VL-8B-Instruct (무학습) | ✅ 8장 | 전체 293 | **어댑터를 뗀 대조군** |
| `qwen3vl_8b_blind` | 동일 (무학습) | ❌ 없음 | 전체 293 | **프레임 기여분 분리** |
| `ours_sft_r15_vocab` | **EGO step2 sft_r15** (LoRA on 위 백본) | ✅ 8장 | 전체 293 | 우리 프레임워크 |
| `ours_sft_r15_wm1st` | 동일 | ✅ 8장 | 전체 293 **+ WM top-10 제약(1스텝)** | **EGO 배포 형태** |

- `qwen3vl_8b_frames`는 임의의 오픈웨이트가 아니라 **`ours`의 정확한 베이스**다. 학습은 그 위의
  LoRA(r=16)이므로 `ours − backbone` = **CE/SFT 학습이 계획 능력에 기여한 몫**.
- `blind`는 프레임을 언급하는 **한 문장만** 빼고 사용자 메시지까지 완전히 동일하다.
  `frames − blind` = **프레임 기여분** = v2 재작성의 정당성.

### 4-2. `wm10_first` — EGO 배포 형태를 재현하는 방식

EGO 프레임워크는 "world model이 다음 action 후보를 제시 → LM이 선택"이다. 그런데
**WM은 바로 다음 1개 action만 예측**하므로 2번째 이후 스텝의 후보를 만들 수 없다. 그래서:

```
CANDIDATE ACTION LABELS (choose only from these):
- ... 293개 전체 ...

A video world model has ranked these 10 candidates for the FIRST next action only:
- ... WM top-10 ...
Your 1st predicted action MUST be one of these 10.
Actions 2 onward are unconstrained — pick them from the full candidate list.
```

- **1스텝만 WM 제약, 2스텝부터 자유.** 라벨 공간은 여전히 293이라 `ours_vpa`와 **직접 비교 가능**하고,
  두 arm의 차이가 곧 **WM prior의 기여분**이 된다.
- ⚠ 초기 구현은 T스텝 전부를 top-10으로 제한했으나 폐기했다 — 2·3번째 정답이 top-10 안에 있는 비율이
  53%/56%뿐이라 **절반은 정답을 출력할 수조차 없는** 불공정 설정이었다.

### 4-3. baseline (프레임 불필요, 비용 ~0)

| baseline | 예측 방식 |
|---|---|
| `random` | 어휘에서 균등 샘플 T개 |
| `most_probable` | 관측 history의 전역 빈도 top-T |
| `most_probable_goal` | goal(scenario)별 빈도 top-T |
| `wm_top1_repeat` | WM top-1을 T회 반복 (WM은 시퀀스 모델이 아님 — 정직한 퇴화형) |
| `wm_topk_rank` | WM 후보를 점수 내림차순 T개 |

빈도 통계는 **평가셋 정답이 아니라 관측된 history에서만** 뽑아 누출을 막는다.

## 5. 지표

원본 VPA 정의 그대로이며, VLaMP 공식 구현과 의미 등가임을 실사 확인했다.

| 지표 | 정의 | 성격 |
|---|---|---|
| **SR** | 예측 T-시퀀스가 **순서까지** 완전 일치한 샘플 비율 | 가장 엄격 |
| **mAcc** | 위치별 `1[pred_i == gt_i]` 평균 | 순서 민감, 부분점수 |
| **mIoU** | 예측 집합 vs 정답 집합의 IoU 평균 | 순서 무시 |

- **CI는 video 클러스터 부트스트랩**. 영상당 평균 ~13샘플이 상관돼 있어 샘플 단위 재표집은 CI를 과소추정한다.
- **arm 비교는 paired** (`paired.py`) — 같은 표본에서 두 arm을 나란히 재표집해 **차이의 분포**를 직접 구한다.
  arm별 CI가 겹치는지 보는 것은 틀린 판정이다 (difference of significance ≠ significance of difference).
- 어휘 밖 예측은 정확일치 → 정규화 → difflib 최근접으로 매핑하고 매핑 횟수를 기록한다.
  (실측: frontier 668건 정확일치 · 13건 밑줄→공백 매핑 · 미매핑 0)

## 6. 핵심 비교 세 가지

| 비교 | 답하는 질문 |
|---|---|
| `ours_vpa` − `qwen3vl_backbone` | **CE/SFT 학습이 VPA 계획 능력에 기여하는가** |
| `ours_wm1st` − `ours_vpa` | **WM prior가 계획에 기여하는가** |
| `frames` − `blind` | **프레임 입력이 기여하는가** (v2 재작성의 정당성) |
| (참고) `ours_vpa` − `frontier` | 8B 학습 모델이 frontier 대비 어디에 서는가 |

## 7. 재현

```bash
PYTHONPATH=src python -m ego.step3_results.vpa.v2.build_dataset --split heldout
PYTHONPATH=src python -m ego.step3_results.vpa.v2.frames --gt runs/vpa_v2/vpa_v2_T3.json
PYTHONPATH=src python -m ego.step3_results.vpa.v2.evaluate --gt runs/vpa_v2/vpa_v2_T3.json --baselines
bash runs/vpa_v2/rerun_pipeline.sh     # frontier + 백본 + blind
bash runs/vpa_v2/ours_arms.sh          # EGO arm 2종 + paired
```

## 8. 운영 안전장치

- **프레임 캐시 키에 창 규격 포함** (`frame_cache_w4_g1_n8_s336`) — 창을 바꿔도 옛 프레임을 재사용하는 사고 방지.
- **부분 결과 보고 차단** — 러너가 `complete=false`, 채점기가 `reportable=false`를 남긴다.
- **실패 행 보존 재개** — 성공한 sample_id만 건너뛴다.
- **빌더 계약 assert** — 미래 action이 관측창을 침범하면 빌드가 즉시 실패한다.
- **API 키는 환경변수 전용** — 코드·로그·파일에 기록하지 않는다.
