# VPA v2 — 프레임-조건 · 무오염 · Frontier(vision) 재작성 계획 [검토 요청]

> 작성: 2026-07-25 KST · **상태: 사용자 검토 대기 (승인 전 코드 미작성)**
> 목적: `src/ego/step3_results/vpa`를 ⑴ 이미지 프레임 입력, ⑵ 무오염 시간 계약,
> ⑶ Frontier VLM(gemini-2.5-pro, vision) 비교군을 갖춘 **논문용 검증 트랙**으로 재작성한다.
>
> 선행 문서: `2026-07-25_vlamp_official_code_review_applicability.md` (원본 코드 실사) ·
> `2026-07-25_paper_capability_evidence_crosscohort_handoff.md` §2-7 (기존 Frontier 결과)

---

## 0. 판정 요약

**설계 골격은 타당하다.** 4초×2fps=8프레임은 step2 학습 모델의 입력 형상(8프레임)을 그대로
보존하는 최적 선택이고, WM top-10 covered 필터는 실측 **43.9%**(3057/6960)로 사용자 추정과 일치한다.

**단 착수 전 해소해야 할 블로커 2건, 설계 수정 1건, 승인 필요한 결정 5건이 있다.**

| # | 항목 | 판정 |
|---|---|---|
| B1 | 게이트웨이 vision 지원 **미확인** | **블로커** — 실패 시 계획 전체 무효 |
| B2 | 로컬에 val 영상 **7/128개**뿐 | **블로커** — 전량은 원격 실행 필요 |
| F1 | `[start−4, start]`는 계약 위반 + 온셋 누출 | **수정 필수** → `[start−5, start−1]` |
| D1~D5 | 후보 제시 · gap 캡 · 우리 모델 arm · 라벨 계보 · T4 포함 | **사용자 승인 필요** |

---

## 1. 사용자 설계 검증 — 맞는 부분

| 지시 | 검증 | 근거 |
|---|---|---|
| 이미지 프레임 입력 | ✅ 타당 | 현 이식의 최대 한계(REPORT.md §5 "text-conditioned") 해소. 원본 VPA 정의(`V_t`)에 부합 |
| 4초 × 2fps = 8이미지 | ✅ **최적** | step2 학습은 8프레임(`vlm.py:18 N_FRAMES=8`). 창만 8s→4s로 줄이면 **토큰 수 불변** = 분포 shift 최소. 앞선 논의의 (a)안과 일치 |
| Frontier도 동일 8이미지 | ✅ 필수 | 동일 입력 계약이라야 트랙 내 비교 성립. 기존 text-only 각주 제거 가능 |
| WM top-10 covered 데이터 | ✅ 실측 부합 | `context_val.jsonl` 6960샘플 중 GT∈top10 = **3057 (43.9%)**, 128 영상 |
| 데이터 오염 금지 | ✅ 유지 필요 | §2 F1 참조 — 사용자 안대로면 오히려 오염이 생긴다 |
| step2 validation 기준 사용 | ✅ 자산 확보 | `runs/cesft_v2/data/context_val.jsonl`(37MB) **로컬 존재**. future 필드(최대 5 action, 24s 캡)가 이미 있어 **VPA 라벨을 그대로 유도 가능** |

---

## 2. 착수 전 블로커

### B1. 게이트웨이(gw.letsur.ai)의 이미지 입력 지원 미확인 — **최우선**

- `frontier_select_eval.py` docstring과 handoff §2-7이 공통으로 **"게이트웨이 모델이 text-only(vision 미지원)"** 로 명시.
- gemini-2.5-pro 자체는 vision 모델이나, **게이트웨이가 `image_url` 파트를 통과시키는지는 미검증**.
- 현재 셸에 `FRONTIER_API_KEY` **미설정** (키는 `EGO_jihun3/.env.local`, 원격 호스트).
- → **Phase 0에서 1샘플 이미지 프로브가 전제조건.** 실패 시 대안:
  ⑴ 게이트웨이의 vision 지원 모델 탐색, ⑵ 직접 Gemini API 키 사용,
  ⑶ frontier arm을 text-only로 유지하고 "우리 모델만 vision"임을 각주(현행 유지 = 후퇴).

### B2. 로컬 영상 부족 — 스모크와 본런의 실행 위치 분리

| | 영상 | covered∩fut≥3 | covered∩fut≥4 |
|---|---:|---:|---:|
| **로컬 보유** | **7** (5.9GB) | **273** | **187** |
| 전체 val | 128 | 1406 | 768 |

- 로컬 `data/Ego4D/v2/goalstep_videos`에 7개만 존재. 나머지는 원격(~272GB).
- → **로컬 = 파이프라인 스모크 전용**(273샘플로 충분), **본런 = 원격 호스트**.
- 프레임 캐시는 반드시 **새 디렉토리**(`FRAME_CACHE_DIR`)로. 기존 캐시는 8초 창 기준이고
  **캐시 키에 창 길이가 없어 조용히 8초 프레임을 재사용**한다(`vlm.py:34`).

---

## 3. [F1] 설계 수정 필수 — 관측창을 `[start−5, start−1]`로

사용자 안 `[start−4, start]`는 세 가지 문제가 있다:

1. **계약 위반** — `contracts.py:86`의 `assert_strict_contract`가 `obs_end < target_start`를 강제하고
   `gap ≥ tau_a(1s)`를 요구한다. gap=0이면 **ContractViolation raise**로 파이프라인이 멈춘다.
2. **온셋 누출** — 마지막 프레임이 action 시작 순간이라 동작 개시가 화면에 들어온다.
   **이것이 정확히 V-JEPA2 end−1s late-recognition 문제**이자, 우리가 VLaMP 원본 구현에서
   비판한 `start+2s` 리키지의 축소판이다. 같은 논문에서 이를 재현하면 자기모순.
3. **학습 조건 불일치** — step2 모델은 `"starting 1 second after the last frame"` 프롬프트로 학습됨.

**권고안**: `obs = [target_start − 5, target_start − 1]` — **4초 창 + 1초 안전 간격**, 8프레임 = 2fps.
계약·학습조건·무오염을 동시에 충족하며, 원본 VLaMP δ=4s의 **경계-안전 버전**이다.

> ⚠ 구현 주의: 현 `context_val.jsonl`의 `obs_end_sec`는 A2.end−1s(retro4 가변 horizon)라
> gap이 1.0~1308초로 퍼져 있다(p50 1.3s, p95 30.8s). **VPA용 창은 `obs_end := target_start − 1`로
> 재정의**해야 한다(기존 `obs_start/obs_end` 필드를 그대로 쓰면 안 됨).
> `target_start < 5s`인 샘플은 0건이라 클램프 문제 없음.

---

## 4. 승인이 필요한 설계 결정

### D1. 예측 후보를 무엇으로 줄 것인가 — **권고: 전체 vocab 제시, WM top-10은 표본 선정만**

- 원본 VPA는 전체 action 공간에서 예측(우리 이식은 252 vocab 제시).
- step2 select는 WM top-10 후보 제시 — **다른 프로토콜**.
- 둘을 섞으면 "VPA를 한 것도, step2를 한 것도 아닌" 표가 된다.
- → **VPA로서 정합하려면 후보 = 전체 vocab.** top-10은 표본 선정 기준으로만 사용.
- **명기 필수**: covered 필터 자체가 "첫 action이 WM top-10 안"이므로 **첫 스텝이 쉬운 쪽으로 편향**된다.
  (선택 시) top-10을 후보로 주는 변형은 "step2 select의 T-step 확장"이라는 **별도 표**로.

### D2. anticipation gap 캡 — **권고: ≤5s**

gap이 30초면 "관측 후 30초 뒤부터 3개를 예측"이라 관측이 사실상 무의미해진다.

| 캡 | T3 (샘플/영상) | T4 (샘플/영상) |
|---|---|---|
| 3s | 1093 / 98 | 613 / 67 |
| **5s (권고)** | **1194 / 99** | **663 / 70** |
| 10s | 1262 / 105 | 695 / 71 |
| 없음 | 1406 / 110 | 768 / 76 |

5s 캡은 T3의 85%를 보존하면서 태스크를 정직하게 만든다. 캡 적용 사실과 드롭 수를 `log`·REPORT에 명기.

### D3. 우리 모델(step2 cesft_v2) arm을 어떻게 채울 것인가 — **권고: autoregressive rollout**

step2 모델은 **단일 next-action selector**라 3~4 시퀀스를 직접 못 낸다. 표를 채우려면 선택 필요:

| 옵션 | 내용 | 평가 |
|---|---|---|
| **(a) rollout (권고)** | 프레임 고정 + 예측 action을 history에 append해 T회 재질의 | **VLaMP의 latent future 생성과 동형** — 원논문 정합. 추가 학습 0 |
| (b) 첫 스텝만 | mAcc@1만 보고 | 표 불완전, VPA 아님 |
| (c) T개 출력 프롬프트 | 학습 분포 밖 | 형식 파탄 위험 |

이번 지시에는 frontier만 언급되었으나, **"우리 검증 result"로 쓰려면 우리 모델 행이 필수**다.

### D4. 라벨 계보 통일 — **권고: context_val 계보(goalstep taxonomy)**

기존 VPA vocab(252, `goalstep_parsed_segments.csv`)과 context_val의 verb/noun(NUM_ACTIONS=293,
`goalstep_step_labels.csv`)은 **다른 계보**다. step2 모델과 표본·라벨을 공유해야 비교가 성립하므로
**context_val 계보로 통일**하고, vocab을 그 계보에서 재생성한다.

### D5. T4 포함 여부 — **권고: 포함하되 [시사] 등급**

T4는 캡 적용 시 663샘플 / **70 영상**. video-cluster bootstrap에는 충분하나 T3보다 약하다.
포함하되 표본 규모를 병기.

### D6. 비교군(arm) 구성 — 2026-07-25 사용자 지시로 Qwen3 백본 추가

**전제**: 모든 arm은 **동일 입력 계약**(4초 창 8프레임@336 + 동일 goal/history/vocab 프롬프트)을 공유한다.
프롬프트가 arm마다 다르면 표 내부 비교가 무너지므로, step2의 3-태그 형식이 아니라 **VPA 공통 프롬프트**를 쓴다.

| # | arm | 실행 위치 | 비용 | 판정 |
|---|---|---|---|---|
| A1 | **Qwen3-VL-8B-Instruct (무학습 백본)** | 로컬 GPU | 낮음 | **필수** — 아래 참조 |
| A2 | **Blind control** (동일 모델, 프레임 제거) | 로컬 GPU | ~0 | **필수** |
| A3 | **WM-only rollout** (Step-1 단독 top-1 × T) | 로컬 CPU/GPU | ~0 | **필수** |
| A4 | ours (θ_CE / sft_r15 / wise_a050) | **원격** (어댑터 미로컬) | 낮음 | D3 승인 시 |
| A5 | Frontier gemini-2.5-pro | API | **높음** | 지시 확정 |
| A6 | Frontier claude-sonnet-4-6 (동일 게이트웨이) | API | 중간 | 권장 |
| A7 | gemini-2.5-flash | API | 낮음 | 선택 (비용 백업) |
| A8 | Qwen2.5-VL-7B / InternVL3-8B | 로컬 GPU (다운로드 필요) | 중간 | 여유 시 |
| — | Most-Probable · Most-Probable-w-Goal · Random | 로컬 | ~0 | 기본 baseline |

**A1이 특별한 이유**: `vlm.py:17`의 `MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"` — 우리 step2 모델의
**정확한 베이스**이고, 학습은 LoRA(PEFT) 어댑터다(`battery.py:30`). 따라서 A1은 임의의 오픈웨이트
비교군이 아니라 **어댑터를 뗀 완전 대조군** = CE/SFT 학습 기여분을 분리하는 유일한 arm이다.
가중치는 **로컬 HF 캐시에 이미 존재**(17GB)하므로 다운로드 불필요.

> ⚠ `run_qwen_baseline.py`의 기본값이 `Qwen3-VL-**7B**-Instruct`로 낡았다(가중치 미도착 시절 잔재).
> 재작성 시 **8B-Instruct로 통일**해야 A1↔A4 대응이 성립한다.

**A2(blind)가 필수인 이유**: 본 재작성의 명분이 "프레임을 넣는다"인데, 프레임이 실제로 기여하는지를
증명하는 arm이 없으면 재작성 자체가 정당화되지 않는다. 기존 이식을 text-only라고 비판한 이상
**"프레임이 +X pp 기여한다"는 실측이 논문의 필수 근거**다. 러너 재사용에 이미지만 제거하므로 비용 ~0.

**A3(WM-only)**: `RETRO4-goalstep-end-m1-history-k8-phase1/best_action_top5.pt`의 top-1을 T회 반복.
"언어모델 없이 비디오 prior만"의 하한선이자, GADR 논증(모방으로는 못 얻는다)의 VPA판 대조.

**포함하지 않는 것**: VLaMP 원 모델 — S3D 특징 + AllenNLP 스택 + GoalStep 재학습이 필요하고,
선행 리뷰 문서 §3-3에서 포팅 비권장으로 판정. arm 총수는 **4~6행**으로 제한(429·비용 리스크).

---

## 5. 구현 계획

### Phase 0 — 전제조건 프로브 (~15분, 코드 최소)
1. **게이트웨이 vision 프로브**: 1샘플 8이미지(base64 data URI) + 텍스트로 `chat/completions` 호출.
   성공/실패를 `runs/vpa_v2/probe_vision.json`에 기록. **실패 시 여기서 중단하고 보고.**
2. 로컬 7영상에서 프레임 추출 1샘플 검증(decord 디코드·2fps 인덱스·336 리사이즈).

### Phase 1 — 데이터 빌더 `build_vpa_v2.py`
- 입력: `runs/cesft_v2/data/context_val.jsonl`
- 필터: GT∈candidates(top-10) ∧ `len(future) ≥ T` ∧ `gap ≤ CAP`
- 창 재정의: `obs_end = target_start − 1.0`, `obs_start = obs_end − 4.0`
- 출력 샘플 스키마:
  ```json
  {"sample_id","video_uid","goal_text","observed_actions":[...],
   "obs_start_sec","obs_end_sec","target_start_sec",
   "future_actions":["verb noun", ...T],   // context_val.future에서 유도
   "horizon":3, "gap_sec":1.3, "eval_split":"test"}
  ```
- **계약 assert 재사용**: `assert_strict_contract` 동등 검사 + `obs_end + eps ≤ target_start` 강제.
- 산출: `data/vpa_v2_T{3,4}.json`, `vocab_v2.json`, `manifest.json`(필터별 드롭 수 포함)

### Phase 2 — 프레임 추출 `vpa_frames.py`
- `vlm.extract_frames` **재사용**(step2와 동일 경로 = 공정성 보장), `n_frames=8`, `FRAME_SHORT_SIDE=336`
- **전용 캐시 디렉토리**(`runs/vpa_v2/frame_cache`) — B2의 캐시 오염 방지
- 실패 샘플은 `skip_decode`로 태깅해 별도 집계(무응답을 오답으로 세지 않기 위해)

### Phase 3 — Frontier(vision) 러너 `run_frontier_vpa_v2.py`
- 8이미지를 base64 data URI로 `content` 배열에 삽입 + goal/observed/vocab 텍스트
- **resume**: `records.jsonl`의 sample_id 기준. **`api_error` 행 삭제 금지**(handoff §6-3 사고 재발 방지)
- rate limit: 동시성·백오프 파라미터화. 429는 실패로 남기고 재시도 큐에 유지
- **부분 결과 보고 금지 원칙**을 코드에 반영: 커버리지 100% 미만이면 metrics 파일에
  `"reportable": false`와 결손 목록을 기록

### Phase 4 — 우리 모델 러너 `run_ours_vpa_v2.py` (D3 승인 시)
- `battery.py` 로딩 경로 재사용(어댑터: θ_CE / sft_r15 / wise_a050)
- rollout: 프레임 고정, 예측 action을 history에 append → T회

### Phase 5 — 채점 `eval_vpa_v2.py`
- SR / mAcc / mIoU 유지(원본 metric과 의미 등가 검증 완료)
- **video-cluster bootstrap으로 교체** — 영상당 평균 ~12.8샘플이라 샘플 단위 CI는 과소추정
- baseline 행: Most-Probable, Most-Probable-w-Goal, **Random**(신규)
- 산출: `runs/vpa_v2/metrics_T{3,4}.json` + `REPORT_v2.md`

### Phase 6 (선택) — 리키지 진단 ablation
`obs = [start−4, start]`(원본 VLaMP pre-window 등가) 변형 1행을 **진단 목적**으로 추가 →
"경계 침범이 첫-스텝 지표를 몇 pp 부풀리는가"를 정량화. 본 표와 분리, 오염 재현이 아니라 **오염의 측정**.

---

## 6. 위험·비용

| 위험 | 완화 |
|---|---|
| 게이트웨이 vision 미지원 | Phase 0 프로브로 조기 판정 |
| API 비용/429 (이전 64% 실패) | 이미지 8장이면 토큰 급증. 336 리사이즈로 억제 + 사전 비용 추정 + resume + 부분보고 차단 |
| 원격 영상 의존 | 로컬 273샘플 스모크로 파이프라인 완결 후 원격 이관 |
| 프레임 캐시 오염 | 전용 캐시 디렉토리 강제, 창 길이를 캐시 키에 포함 |
| 오염 검사 누락 | step2 SFT 학습 표본과 VPA 평가 표본의 video_uid 교집합 0 재확인 + WM(Step-1) train/val 분리 재확인 |

---

## 7. 승인 요청 (이 5개만 확정해 주시면 착수)

1. **F1 관측창** `[target_start−5, target_start−1]` (4초+1초 간격) — 동의?
2. **D1 후보 제시** 전체 vocab (top-10은 표본 선정만) — 동의? 아니면 top-10 후보 버전도 병행?
3. **D2 gap 캡** 5초 (T3 1194 / T4 663) — 동의? 다른 값 선호?
4. **D3 우리 모델 arm** autoregressive rollout으로 포함 — 이번 범위에 포함? 아니면 frontier만 먼저?
5. **D5 T4** 포함 — 동의?
6. **D6 arm 구성** — A1(Qwen3-VL-8B 백본)·A2(blind)·A3(WM-only)은 필수로 넣고, A6(claude-sonnet-4-6)·
   A7(flash)·A8(타 오픈웨이트) 중 무엇까지 포함할지?

승인 후 Phase 0(프로브) → 결과 보고 → Phase 1~5 순으로 진행합니다.
