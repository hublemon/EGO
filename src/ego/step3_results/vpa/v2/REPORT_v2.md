# VPA v2 — 프레임-조건 · 무오염 GoalStep VPA 챌린지

- 작성: 2026-07-25 · 코드: `src/ego/step3_results/vpa/v2/`
- 설계 근거: `develop_report/2026-07-25_vpa_v2_frame_conditioned_plan.md` (승인본)
- 원본 실사: `develop_report/2026-07-25_vlamp_official_code_review_applicability.md`

v1(`../`)은 **text-conditioned**(프레임 미사용)였다. v2는 원본 VPA(Patel ICCV2023)의 취지대로
**실제 영상 프레임을 입력**하고, 원본 공식 구현이 가진 **경계 리키지를 제거**한 이식이다.

---

## 1. 시간 계약 — 원본 구현과의 차이

| | 원본 VLaMP 공식 구현 | **VPA v2 (본 이식)** |
|---|---|---|
| 관측 윈도우 | `[start−2s, start+2s)` (δ=4s) | `[start−5s, start−1s)` (4s) |
| 예측 대상 침범 | **있음** — 첫 예측 action 시작 후 2초가 입력에 포함 | **없음** — 관측이 시작 1초 전에 종료 |
| 근거 | `observation_type: "pre"` + `(action_prefix + [action_target[0]])`, 코드 주석 `# ISSUE: This is a hack` | `common.observation_window()` + 빌더 assert |

원본 논문 §3의 정의(`V_t`는 k개 완료 action까지)는 깨끗하나 **공개 구현이 그 정의를 경계에서
지키지 않는다.** v2는 논문 정의를 따르며, 이는 우리 V-JEPA2 end−1s late-recognition 제보와
동일한 원칙이다. 관측창 길이 4초는 원본 δ=4s와 맞추되 **1초 안전 간격**을 둔 경계-안전 버전이다.

프레임: 4초 창에서 **8프레임 @2fps**, 짧은 변 336 — step2 학습 입력(8프레임@336)과 토큰 형상 동일.

## 2. 표본 (step2 검증셋 계보)

`runs/cesft_v2/data/context_val.jsonl` (6960샘플 / 128영상)에서:

| 필터 | T=3 | T=4 |
|---|---:|---:|
| covered (GT ∈ WM top-10) | 3057 (43.9%) | 3057 |
| ∧ future ≥ T | 1406 | 768 |
| ∧ gap ≤ 5s | **1194 / 99영상** | **663 / 70영상** |

- **WM top-10은 표본 선정에만** 사용하고 예측 후보로 주지 않는다. 후보는 **전체 어휘 293**
  (`goalstep_step_labels.csv` 계보 = step2와 동일) — VPA로서 원본과 정합.
  (예외: `ours_wm10` arm 만 EGO 배포 형태 재현을 위해 top-10 을 제시 — §4-2)
- gap 캡 5s: 미적용 시 p95가 30.8s로, 관측이 무의미해지는 표본이 섞인다.

### 2-2. covered 필터의 정확한 범위 — **첫 스텝에만 걸린다**

heldout 기준 표본 선정 경로:

| 단계 | 남은 샘플 |
|---|---:|
| heldout 전체 | 5,326 (90영상) |
| **covered (GT ∈ WM top-10)** | 2,313 (**43.4%**) |
| ∧ future ≥ 3 | 1,064 |
| ∧ gap ≤ 5s | **915** (71영상) |

covered 필터는 `gt_next_action`(= step2의 예측 대상)에만 적용된다. `future[0] == gt_next_action`이
915개 중 911개로 일치하므로 **1번째 예측 대상은 정의상 항상 WM 후보 안에 있고, 2·3번째는 자유**다.
실측 포함률:

| 예측 스텝 | 정답이 WM top-10 안 |
|---|---:|
| 1번째 | **100%** (필터가 강제) |
| 2번째 | 53% |
| 3번째 | 56% |

**함의 3가지:**
1. **첫 스텝이 인위적으로 쉽다** — mAcc의 1/3이 유리한 구간. `wm_top1_repeat`의 mAcc가 15%대로
   높게 나오는 주된 이유다.
2. **val 전체 분포가 아니다** — heldout의 43.4%만 본다.
3. WM 체크포인트가 val로 선택됐으므로(§2-1) covered 필터 자체가 val 선택 의존성을 물려받는다.

**보고 규칙**: 편향이 전 arm에 동일하게 걸리므로 **arm 간 비교(특히 paired)는 영향받지 않는다.**
다만 절대 수치를 "GoalStep VPA 성능"으로 일반화하지 말고 **"WM이 다음 행동을 커버하는 구간에서의
계획 성능"**으로 한정해 서술할 것. 편향 없는 수치가 필요하면 covered 필터를 뺀 변형을 별도 실행한다.

## 2-1. 오염 검사 (2026-07-26, 사용자 요청)

step1·step2 학습 코드와 대조해 VPA 평가 표본의 오염 여부를 전수 확인했다.

| 검사 | 결과 | 판정 |
|---|---|---|
| VPA 영상 ⊂ `goalstep_val.json` | 99/99 (val 밖 0개) | ✅ |
| VPA 영상 ∩ `goalstep_train.json` | **0** | ✅ |
| `goalstep_train` ∩ `goalstep_val` | 0 (583 / 134) | ✅ |
| step2 SFT 학습 데이터 | `context_train.jsonl` = goalstep **train** split (`sft_r1.py:101`) | ✅ 평가셋 미사용 |
| **`context_val.jsonl` 의 split 구성** | **`heldout` 5326 + `dev` 1634 혼재** | ⚠️ **문제 발견** |

### [정정] dev split 혼입 — 발견 및 수정

초판 빌더는 `context_val.jsonl`을 통째로 썼는데, 이 파일에는 **두 개의 split이 섞여 있다**:

- **`dev`** — `probe_gen.py:26`이 `split == "dev"` 행만 뽑아 **step2 학습 중 probe 세트**로 쓴다.
  즉 학습 루프가 들여다본 표본이다.
- **`heldout`** — `battery.py`의 기본 평가 split(`--split_name heldout`). cesft_v2가 논문에
  보고한 수치는 전부 이쪽이다.

초판 T3 1194개 중 **279개(28영상)가 dev**였다. 영상 단위로는 dev/heldout이 서로 겹치지 않아
(dev 28영상 ∩ heldout 71영상 = 0) 누출 자체는 경미하지만, **cesft_v2 보고 수치와 표본이
달라져 비교가 깨진다**. 또 향후 `ours` arm(sft_r15)을 넣으면 학습 중 관찰한 표본으로 평가하는
셈이 된다.

**수정**: 빌더에 `--split`(기본 `heldout`)을 추가하고 데이터를 재생성했다.

| | 초판 (dev 혼재) | **수정판 (heldout 전용)** |
|---|---|---|
| T=3 | 1194 샘플 / 99 영상 | **915 / 71** |
| T=4 | 663 / 70 | **504 / 54** |

**§4의 표는 초판(dev 혼재) 기준이므로 폐기 대상이다.** 영상 확보 후 heldout 기준으로 전량 재실행한다.

### 남은 주의 — WM은 val로 체크포인트를 골랐다

step1 설정(`z1_end_m1_lobs8_vna_ep10.yaml`)에 `val_subset_size: 2000`이 있어 **WM 학습 중
val 부분집합으로 모니터링·선택**했다. 따라서 `wm_top1_repeat` · `wm_topk_rank` arm과
covered 필터(WM top-10)는 val에 대한 선택 의존성을 물려받는다 — 낙관적으로 읽힐 수 있다.
**VLM arm(frontier · Qwen 백본 · blind)에는 영향이 없다**(어느 쪽도 이 데이터를 본 적 없음).
논문에 WM baseline을 넣을 때 각주로 명기할 것.

## 3. 지표

SR(순서까지 완전일치) · mAcc(위치별 정확도) · mIoU(집합 IoU). VLaMP 공식 구현과 의미 등가임을
실사 확인했다. **부트스트랩은 video 클러스터 단위** — 영상당 ~12샘플이 상관돼 있어 샘플 단위
재표집은 CI를 과소추정한다.

## 4. 결과 (전체 셋, 값=%, [ ]=video-cluster bootstrap 95% CI)

### T=3 — heldout (n=915, 71 videos)

| arm | SR | mAcc | mIoU |
|---|---:|---:|---:|
| random | 0.00 | 0.40 [0.2, 0.6] | 0.66 [0.5, 0.9] |
| most_probable | 0.11 [0.0, 0.3] | 8.01 [2.8, 14.4] | 13.14 [3.8, 26.4] |
| most_probable_goal | 0.66 [0.0, 1.6] | 13.48 [8.1, 19.0] | 24.13 [14.2, 35.1] |
| **wm_top1_repeat** | 0.98 [0.2, 2.0] | **15.34** [10.3, 21.3] | 14.92 [10.0, 20.7] |
| wm_topk_rank | 0.00 | 12.60 [9.9, 15.4] | 18.65 [15.0, 22.4] |

### T=4 — heldout (n=504, 54 videos)

| arm | SR | mAcc | mIoU |
|---|---:|---:|---:|
| random | 0.00 | 0.10 [0.0, 0.2] | 0.59 [0.3, 0.9] |
| most_probable | 0.00 | 6.15 [1.3, 10.6] | 15.16 [4.3, 27.2] |
| most_probable_goal | 0.00 | 10.96 [7.4, 14.2] | **25.65** [15.4, 34.8] |
| **wm_top1_repeat** | 0.60 [0.0, 1.5] | **15.62** [9.0, 23.2] | 15.18 [8.8, 22.8] |
| wm_topk_rank | 0.00 | 11.81 [8.3, 15.4] | 20.66 [15.9, 25.1] |

**읽는 법**: `wm_top1_repeat`의 mAcc가 baseline 중 가장 높은 것은 covered 표본이라 WM top-1이
첫 스텝을 자주 맞히기 때문이다(2·3번째는 같은 라벨 반복이라 거의 틀린다). 반대로
`most_probable_goal`은 mIoU가 높다 — 순서를 못 맞혀도 집합이 겹친다.
**baseline 의 SR이 전 arm 1% 미만**이라는 것이 핵심: 3~4개 시퀀스를 순서까지 맞히는 것은
빈도·prior 휴리스틱으로는 불가능하다.

> ⚠ 이전 판의 T=3 n=1194 / T=4 n=663 수치는 **dev split 혼재**(§2-1) 상태였으므로 폐기.

## 4-1. 모델 arm 결과 — **최종** (915샘플 / 71영상, 전 arm 동일 표본)

| arm | 가중치 | 입력 | 후보 | SR | mAcc | mIoU |
|---|---|---|---|---:|---:|---:|
| `ours_wm1st` | EGO sft_r15 | 8프레임 | 293 + WM 1스텝 | 9.40 [0.8, 23.5] | **21.38** [13.2, 32.7] | **31.55** [19.3, 48.2] |
| `ours_full` | EGO sft_r15 | 8프레임 | 293 | **12.46** [1.1, 31.7] | 20.62 [9.8, 37.9] | 30.82 [17.8, 48.1] |
| `qwen_backbone` | Qwen3-VL-8B 무학습 | 8프레임 | 293 | 11.58 [0.4, 30.4] | 17.49 [6.5, 35.1] | 25.99 [13.2, 44.8] |
| `qwen_blind` | Qwen3-VL-8B 무학습 | 텍스트만 | 293 | 0.55 [0.0, 1.7] | 9.18 [2.9, 18.5] | 13.75 [6.5, 23.5] |
| `frontier` | gemini-2.5-pro | 8프레임 | 293 | 3.06 [0.4, 6.1] | 11.55 [8.4, 14.4] | 28.09 [15.0, 45.0] |

frontier 는 **915/915 완주 · 실패 0 · 전건 HTTP 200 · 전건 8이미지 전송**
(`frontier_T3.status.json` `complete: true`). 라벨 매핑은 정확일치 위주 · 퍼지매핑 소수(밑줄→공백) ·
미매핑 0 이라 채점 아티팩트가 없다.

### 짝지은 대조 (video-cluster **paired** bootstrap, 2000 resample · n=915 / 71클러스터)

| 비교 | 무엇이 다른가 | SR Δ | mAcc Δ | mIoU Δ | 판정 |
|---|---|---:|---:|---:|---|
| `ours_full` − `qwen_backbone` | **LoRA 어댑터만** | +0.87 [+0.2, +1.6] | **+3.13** [+1.3, +5.1] | **+4.84** [+2.3, +7.7] | ✅ 3/3 유의 |
| `qwen_backbone` − `qwen_blind` | **프레임만** | +11.04 [+0.3, +29.3] | **+8.31** [+2.3, +16.4] | **+12.23** [+5.4, +21.0] | ✅ 3/3 유의 |
| `ours_wm1st` − `ours_full` | **WM 힌트만** | −3.06 [−8.2, +0.8] | +0.77 [−5.3, +5.9] | +0.72 [−1.5, +2.9] | ❌ 전부 비유의 |
| `ours_full` − `frontier` | 모델 | +9.40 [−0.6, +25.0] | +9.07 [−1.4, +24.3] | +2.73 [−0.7, +6.2] | ❌ 비유의 |
| `qwen_backbone` − `frontier` | 모델 | +8.52 [−1.5, +24.1] | +5.94 [−5.1, +21.4] | −2.11 [−6.3, +1.6] | ❌ 비유의 |

**arm 사다리** — 한 번에 하나씩만 더해 각 부품의 기여를 분리한다:

```
qwen_blind ──(+프레임)──▶ qwen_backbone ──(+EGO 학습)──▶ ours_full ──(+WM 힌트)──▶ ours_wm1st
  mAcc 9.18      +8.31 ✅      17.49        +3.13 ✅      20.62      +0.77 ✗      21.38
```

앞의 두 단계는 유의하고 마지막 WM 힌트만 비유의다 — **성능을 만든 것은 프레임 입력과 EGO 학습**이며,
WM 후보를 추가로 알려주는 것은 도움이 되지 않았다. `frontier` 는 사다리 밖의 외부 기준점(mAcc 11.55).

> 상세 해석·원인 분석(§3-5 frontier vs 백본), 오염 검사, SR 결함은
> **`docs/experiments/2026-07-26_vpa_v2_results_handoff.md` 가 정본**이다.

> ⚠ **폐기**: 초판(2026-07-26 오전)의 230샘플 / 3영상 수치는 dev split 혼재 + 클러스터 3개라
> 전부 무효다. `runs/vpa_v2/_superseded_devmixed/` 에 격리했다.

## 5. arm 실행 현황 — 전량 완주

| arm | 상태 | 비고 |
|---|---|---|
| `ours_full` · `ours_wm1st` | ✅ 915 | EGO step2 `sft_r15` 어댑터 (로컬 `outputs/step2_retrospection/cesft_v2/`) |
| `qwen_backbone` · `qwen_blind` | ✅ 915 | 어댑터를 뗀 동일 베이스 · 프레임 유무 |
| `frontier` | ✅ 915 | gemini-2.5-pro, 비용 상한 500을 두 번에 나눠 전량 |
| baseline 5종 | ✅ 915 / 504 | T=3 · T=4 |
| T=4 모델 arm | 미실행 | 프레임 캐시 재사용 가능(sample_id 가 T3 부분집합) |

### 5-1. [정정] 게이트웨이는 이미지를 받는다

`2026-07-25_paper_capability_evidence_crosscohort_handoff.md` §2-7과 기존
`frontier_select_eval.py` docstring은 **"게이트웨이 모델이 text-only(vision 미지원)"** 로 판정하고,
그 근거로 main.tex에 "frontier는 이미지 미열람 → 직접 비교 아님" 각주를 달아 두었다.

**2026-07-26 실측으로 이 판정은 뒤집혔다.** `https://gw.letsur.ai/v1` · `gemini-2.5-pro`에
OpenAI 호환 멀티모달 포맷(`image_url` + base64 data URI)으로 **8프레임을 보내 정상 응답**을 받았다.

→ **VPA v2에서는 frontier와 로컬 arm이 같은 8프레임을 본다.** "text-only라 직접 비교 불가"라는
각주는 VPA v2 트랙에 한해 **불필요**하다. (select 트랙은 여전히 텍스트로 실행된 과거 결과이므로
각주 유지 — 재실행 시 동일하게 이미지 전송 가능.)

**"받기만" 하는 게 아니라 "쓴다"는 확증 (frame-swap 테스트)**: 요청이 200으로 통과하는 것만으로는
모델이 이미지를 실제로 참조했다고 말할 수 없다. 그래서 **텍스트(goal·history·vocab)를 완전히
고정한 채 프레임만 다른 영상 것으로 바꿔** 두 번 호출했다:

| 조건 | 출력 |
|---|---|
| 자기 프레임 (goal="cooking general") | `add oil` · `add cabbage` · `stir vegetable` |
| 다른 영상 프레임 (플랫브레드 영상) | `flip flatbread` · `stir dish` · `transfer flatbread` |

텍스트 프롬프트에는 **"flatbread"라는 단어가 없다**(goal은 "cooking general"). 그럼에도 출력이
플랫브레드로 바뀌었으므로 **이미지가 출력을 지배적으로 좌우한다**. 재현: `frontier` 러너의
`encode_frames`/`build_payload`를 그대로 쓰고 프레임 경로만 교체.

**프레임 가용성 — 해결됨**: 초기에는 로컬에 GoalStep 영상이 7개뿐이라 T3 기준 230샘플 / 3영상만
가능했다. 2026-07-26 v2_1 매니페스트로 **필요 영상 99개를 전량 확보**(540ss 우선 + `grp-` 계열은
`v2_1/full_scale`, 총 36.5 GiB)해 **915샘플 / 71영상**으로 확대했다. 클러스터가 3 → 71로 늘면서
초판에서 전부 비유의였던 짝비교가 판정 가능해졌다(§4-1).

> **초판(3영상)의 교훈**: 그 부분집합은 상위 5개 행동이 정답의 63%를 차지할 만큼 편중돼
> `most_probable_goal` mAcc가 전체 셋 13.48 대비 23.77로 부풀려졌다. **부분집합의 절대 비교는
> 무의미**하며, 이는 2026-07-24 frontier select 사고(32영상 중 8영상만 커버)와 같은 종류의 함정이다.

## 6. 실행 환경 이슈 — GB10(sm_121) × torch

이 장비의 GPU는 compute capability **12.1**인데 설치된 torch(2.11.0+cu128)가 담고 있는
아키텍처는 `sm_80/90/100/120`뿐이다. 사전 컴파일 커널(matmul 등)은 동작하지만 **런타임
nvrtc JIT로 생성되는 리덕션 커널**이 `compute_121`을 거부한다:

```
nvrtc: error: invalid value for --gpu-architecture (-arch)
```

Qwen3-VL 경로에서 이를 밟는 것은 `prod` 계열(작은 정수 텐서)뿐이라 `gb10_compat.py`로
**prod만 CPU 우회**하는 패치를 두었다. 단 **패치를 모델 로딩 전에 적용하면 로딩이 크게
느려지므로 로딩 후에 적용해야 한다**(로딩 경로가 prod를 반복 호출). 근본 해결은 sm_121을
포함한 torch 빌드로 갱신하는 것.

## 7. 재현 명령

```bash
# (1) 데이터셋 — T3/T4 + frontier 500 부분집합(seed 고정)
PYTHONPATH=src python -m ego.step3_results.vpa.v2.build_dataset

# (2) 프레임 (4s·8frames@2fps·336, 전용 캐시)
PYTHONPATH=src python -m ego.step3_results.vpa.v2.frames \
  --gt runs/vpa_v2/vpa_v2_T3.json --video-root data/Ego4D/v2/goalstep_videos

# (3) baseline + WM arm 채점 (전체 셋)
PYTHONPATH=src python -m ego.step3_results.vpa.v2.evaluate \
  --gt runs/vpa_v2/vpa_v2_T3.json --baselines

# (4) Frontier (vision) — 500샘플 상한, 중단해도 같은 명령으로 재개
export FRONTIER_API_KEY=...   # 하드코딩 금지
PYTHONPATH=src python -m ego.step3_results.vpa.v2.run_frontier \
  --gt runs/vpa_v2/vpa_v2_T3.json --subset runs/vpa_v2/frontier_subset_T3.json \
  --out runs/vpa_v2/preds/frontier_T3 --max-calls 500

# (5) Qwen3-VL 백본 / blind
PYTHONPATH=src python -m ego.step3_results.vpa.v2.run_local_vlm \
  --gt runs/vpa_v2/vpa_v2_T3.json --mode frames --out runs/vpa_v2/preds/qwen_backbone_T3
PYTHONPATH=src python -m ego.step3_results.vpa.v2.run_local_vlm \
  --gt runs/vpa_v2/vpa_v2_T3.json --mode blind  --out runs/vpa_v2/preds/qwen_blind_T3

# (6) 예측 채점 (동일 표본 비교는 --subset 로 강제)
PYTHONPATH=src python -m ego.step3_results.vpa.v2.evaluate \
  --gt runs/vpa_v2/vpa_v2_T3.json --pred runs/vpa_v2/preds/qwen_backbone_T3.json \
  --run-name qwen_backbone --subset runs/vpa_v2/frames_subset_T3.json
```

## 8. 안전장치 (과거 사고 반영)

- **프레임 캐시 키에 창 규격 포함** (`frame_cache_w4_g1_n8_s336`) — step2 캐시는 sample_id만
  써서 창을 바꿔도 옛 프레임을 조용히 재사용하는 함정이 있었다.
- **부분 보고 차단** — frontier 러너가 `complete=false`와 잔여 건수를 status에 남기고,
  채점기는 `reportable=false`를 표시한다. (2026-07-24 select 런 972/1520 실패 사고 재발 방지)
- **실패 행 보존 재개** — 성공한 sample_id만 건너뛴다. api_error 행을 지워 "완료"로 오판정한
  `resume_select_429.sh` 사고를 구조적으로 막는다.
- **빈도 baseline 누출 차단** — 빈도 통계를 평가셋 정답이 아니라 관측된 history에서만 뽑는다.
