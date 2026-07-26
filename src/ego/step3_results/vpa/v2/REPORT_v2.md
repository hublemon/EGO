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

### T=3 (n=1194, 99 videos)

| arm | SR | mAcc | mIoU |
|---|---|---|---|
| random | 0.00 [0.0, 0.0] | 0.31 [0.2, 0.5] | 0.56 [0.4, 0.8] |
| most_probable | 0.08 [0.0, 0.2] | 5.03 [1.8, 9.0] | 11.48 [4.2, 21.0] |
| most_probable_goal | 0.59 [0.0, 1.3] | 12.00 [8.4, 15.5] | 21.04 [15.2, 26.6] |
| **wm_top1_repeat** | 0.92 [0.3, 1.7] | **16.11 [11.5, 20.5]** | 15.33 [11.0, 19.4] |
| wm_topk_rank | 0.08 [0.0, 0.3] | 13.12 [10.7, 15.3] | 19.77 [16.1, 23.1] |

### T=4 (n=663, 70 videos)

| arm | SR | mAcc | mIoU |
|---|---|---|---|
| random | 0.00 [0.0, 0.0] | 0.30 [0.1, 0.6] | 0.65 [0.5, 0.9] |
| most_probable | 0.00 [0.0, 0.0] | 7.43 [3.0, 10.9] | 15.08 [5.9, 23.2] |
| most_probable_goal | 0.00 [0.0, 0.0] | 11.12 [7.5, 14.0] | 25.04 [16.5, 31.9] |
| **wm_top1_repeat** | 0.45 [0.0, 1.2] | **16.59 [10.3, 22.4]** | 15.42 [9.8, 21.3] |
| wm_topk_rank | 0.00 [0.0, 0.0] | 12.33 [9.3, 15.1] | 21.53 [17.2, 25.2] |

**읽는 법**: `wm_top1_repeat`의 mAcc가 가장 높은 것은 covered 표본이라 WM top-1이 첫 스텝을
자주 맞히기 때문이다(2·3번째는 같은 라벨 반복이라 거의 틀린다). 반대로 `most_probable_goal`은
mIoU가 높다 — 순서를 못 맞혀도 집합이 겹친다. **SR이 전 arm 1% 미만**이라는 것이 핵심:
3~4개 시퀀스를 순서까지 맞히는 것은 빈도·prior 휴리스틱으로는 불가능하다.

## 4-1. 프레임 arm 결과 (동일 230샘플 / **3영상** — 파이프라인 검증용, 논문 보고 불가)

| arm | 입력 | SR | mAcc | mIoU |
|---|---|---:|---:|---:|
| **Frontier gemini-2.5-pro** | 8프레임 + 텍스트 | **2.17** | **17.10** | **26.63** |
| Qwen3-VL-8B 백본 | 8프레임 + 텍스트 | 0.43 | 9.71 | 18.70 |
| Qwen3-VL-8B blind | 텍스트만 | 0.00 | 8.26 | 18.21 |
| (참고) most_probable_goal | — | 1.74 | 23.77 | 43.40 |
| (참고) wm_top1_repeat | — | 1.74 | 20.72 | 18.91 |

frontier는 **230/230 완주, 실패 0건, 전건 HTTP 200, 전건 8이미지 전송**(`frontier_T3.status.json`
`complete: true`). 라벨 형식은 668건 정확일치 · 13건 밑줄→공백 퍼지매핑(`add_water` → `add water`) ·
미매핑 0건이라 채점 아티팩트가 없다.

### 짝지은 대조 (video-cluster **paired** bootstrap, 2000 resample)

| 비교 | 지표 | Δ | CI95 | 유의 |
|---|---|---:|---|---|
| **frames − blind** (Qwen3-VL, 프레임 기여분) | SR | +0.43 | [+0.00, +1.67] | 아니오 |
| | mAcc | **+1.45** | [−3.33, +7.22] | 아니오 |
| | mIoU | +0.49 | [−9.40, +11.06] | 아니오 |
| **frontier − Qwen3-VL 백본** (같은 8프레임) | SR | +1.74 | [−1.67, +5.71] | 아니오 |
| | mAcc | **+7.39** | [−1.67, +12.86] | 아니오 |
| | mIoU | +7.93 | [−3.33, +17.90] | 아니오 |

**판정: 모든 점추정의 부호는 기대와 일치하나(프레임 > blind, frontier > 백본) 전부 비유의.**
클러스터가 **3개**뿐이라 검정력이 없다 — CI 폭이 mAcc 10~15pp에 달한다.
→ **이 로컬 실행으로는 "프레임이 도움이 되는가", "frontier가 더 나은가" 어느 쪽도 답할 수 없다.**
답하려면 원격 전량 실행(T3 99 클러스터)이 필요하다. 지금 확인된 것은 **파이프라인이 끝까지
돌아간다는 것과, 필요한 표본 규모가 로컬 가용치를 훨씬 넘는다는 것**이다.

> arm별 CI가 겹치는지 보는 방식이 아니라 **차이를 직접 재표집**했다
> (difference of significance ≠ significance of difference). 도구: `paired.py`.

**주의**: 위 표의 빈도 baseline(most_probable_goal mAcc 23.77)이 프레임 arm보다 높아 보이지만,
§5의 라벨 편중 경고대로 **3영상 부분집합에서 부풀려진 값**이다(전체 셋에서는 12.00).
학습 없는 모델과의 절대 비교는 무의미하며, 짝지은 대조만 읽을 것.

## 5. 미완 arm (실행 차단 — 코드는 준비됨)

| arm | 상태 | 비고 |
|---|---|---|
| Qwen3-VL-8B 백본 (frames) | **실행** | 230샘플 / 3영상 — GB10 이슈 해소(§6) |
| blind control | **실행** | 동일 230샘플, 프레임만 제거 |
| Frontier gemini-2.5-pro (vision) | **실행** | 동일 230샘플에 8프레임 전송. 500 상한 미달이라 전량 시도 |
| ours (θ_CE/sft_r15) | 범위 외 | 어댑터가 로컬에 없음(원격) |

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

**프레임 가용성 한계**: 로컬에 GoalStep 영상이 7개뿐이라 프레임을 추출할 수 있는 표본이
T3 기준 **230개 / 3영상**에 그친다. 클러스터가 3개면 CI가 무의미하므로(예: most_probable mIoU
[0.0, 62.4]) **로컬 프레임 결과는 파이프라인 검증용이며 논문 보고 대상이 아니다.**
전량 실행은 원격 호스트(272GB 영상)에서 해야 한다.

**⚠ 3영상 부분집합의 라벨 편중 — 빈도 baseline이 과대평가된다**: 이 230샘플의 정답 라벨은
상위 5개 행동이 **63%**를 차지한다(`cook flatbread` 단독 23.6%, `roll dough` 14.9%, `check heat` 13.8%;
distinct 62종). goal도 3종뿐이라 `most_probable_goal`이 사실상 그 3개 영상의 행동 분포를 외운다.
실제로 같은 지표가 전체 셋에서는 mAcc 12.00인데 이 부분집합에서는 **23.77**로 부풀려진다.

→ **이 부분집합에서 arm 간 절대 비교(특히 학습 없는 모델 vs 빈도 baseline)는 무의미하다.**
의미 있는 것은 **동일 표본에서 짝지어진 대조**뿐이다: ⑴ frames vs blind(같은 모델, 프레임만 차이),
⑵ frontier vs 백본(둘 다 같은 8프레임을 봄).

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
