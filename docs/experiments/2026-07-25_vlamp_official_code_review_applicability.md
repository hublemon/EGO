# VLaMP(원본 VPA) 공식 코드 리뷰 — baseline 구성 확인 · GoalStep 이식 적용성 평가

> 작성: 2026-07-25 KST. **목적: 원본 VPA(Patel et al., ICCV 2023, VLaMP) 공식 코드를 실사하여
> ⑴ VLM API baseline 존재 여부, ⑵ 시각 입력 계약(프레임 수·윈도우·리키지), ⑶ 우리
> `src/ego/step3_results/vpa` 이식에의 적용 가능성을 판정한다.**
>
> 검증 소스: [facebookresearch/VLaMP](https://github.com/facebookresearch/VLaMP) (shallow clone 실사) ·
> ICCV 2023 본문/부록 PDF(pdftotext 실사) ·
> [dhruvdcoder/VideoFeatureExtractor](https://github.com/dhruvdcoder/VideoFeatureExtractor) (S3D 추출 설정).
> 관련 문서: `2026-07-25_paper_capability_evidence_crosscohort_handoff.md` §2-7 (Frontier VPA),
> `src/ego/step3_results/vpa/data/REPORT.md` (이식 설계).

---

## 0. 다섯 줄 요약

1. **원본 공식 코드에 VLM API baseline은 없다.** 비교군은 휴리스틱(Random·Most-Probable) +
   로컬 재구현(DDN·P3IV) + VLaMP 변형(p3lm-T/ST/GT/AT/GAT, GPT-2 로컬 로드)뿐.
   `openai`/`anthropic`/HTTP API 호출이 저장소 전체에 부재 → **우리 frontier API baseline은 원본에 없는 확장**.
2. **시각 입력 계약 확정**: 관찰 = observed action별 시작 기준 **δ=4초 윈도우** `[start−2s, start+2s)`
   (best config `observation_length: 4, observation_offset: -2`), S3D 특징 **초당 1벡터**(추출 16fps)
   → 윈도우당 특징 4개(원시 ~64프레임 압축). 전체 영상이 아니라 **action 경계 주변만 성기게** 본다.
3. **[중대] 원본 구현에 경계 리키지 존재**: `observation_type: "pre"`(best config 전원)에서
   history 마지막 관찰 = **첫 예측 대상 `a_{k+1}`의 `[start−2s, start+2s)`** — 논문 §3 정의(V_t는
   k개 action까지)와 모순되며 코드 주석에 `# ISSUE: This is a hack` 자인. **end−1s late-recognition과
   동형 문제.** 첫 스텝 예측은 부분적 recognition, 2스텝부터 순수 예측.
4. **지표 등가성 검증 통과**: VLaMP `success_rate.py`(전열 일치)·`mean_intersection_over_union.py`
   (샘플별 집합 IoU 평균)·mAcc(위치별)의 의미가 우리 `eval_vpa.py` 구현과 **k=1 기준 동일**.
   우리 채점기는 그대로 신뢰 가능.
5. **판정: 전체 포팅 비권장, 부분 차용 권장.** AllenNLP(아카이브됨) 기반 학습 프레임워크는 이식 가치
   없음. 차용할 것: 관찰 윈도우 스펙(경계-안전 수정판), (G, A_k) ablation 변형(우리 text-conditioned
   설계의 원논문 내 선례), Random baseline 1행. 리키지 발견은 논문 각주/제보 소재로 활용 가능.

---

## 1. 질문 ①: 원본 공식 코드는 VLM API를 비교군으로 썼나 — **아니오**

저장소 전수 grep 결과 (`import openai`, `api_key`, `chat.completion`, `anthropic`, `requests` 호출):

| 검색 대상 | 결과 |
|---|---|
| 외부 LLM/VLM API 호출 | **0건** (grep 무일치) |
| "gpt" 히트의 정체 | `gpt2_decoder.py` 등 — HuggingFace `AutoConfig.from_pretrained`로 **GPT-2 로컬 로드** |
| 비교군 (논문·부록·코드) | Random · Most-Probable(휴리스틱), DDN(LSTM, 부록에서 S3D로 공정 재구현), P3IV(적응 재구현), VLaMP 변형 5종(`best_models_configs/p3lm-{T,ST,GT,AT,GAT}.jsonnet`) |

ICCV 2023 시점 논문이라 frontier VLM API 비교 관행 자체가 없던 때다. 함의:

- 우리 `run_frontier_baseline.py`(frontier API 채점)는 **원본 프로토콜에 없는 우리의 확장**이며,
  원본과의 프로토콜 충돌은 없다. 논문에서 "원본 VPA에는 없는 frontier 비교를 추가했다"로 서술 가능.
- 반대로 "원본도 VLM을 썼다"는 식의 서술은 불가.

## 2. 질문 ②: 원본의 시각 입력 계약 (수치 확정)

논문 본문에는 δ 수치가 없고(그림에 "Observation Window of δ frames"만), **공식 코드가 SSOT**:

| 항목 | 값 | 근거 (코드 좌표) |
|---|---|---|
| 관찰 윈도우 | **4초** = `[start−2s, start+2s)` | `best_models_configs/p3lm-*.jsonnet` `observation_length: 4, observation_offset: -2`; `common.py` `pre_observation_start = floor(start + offset)` |
| 코드 기본값 | 2초 (config가 4로 override) | `common.py:160` `observation_length: int = 2` |
| 특징 밀도 | S3D 초당 1벡터 → 윈도우당 4벡터(1024-d) | 1초 클립 단위 전처리(논문 §4.2) + per-second 인덱스 슬라이스 |
| 원시 프레임 환산 | 16fps → 윈도우당 ~64프레임 | VideoFeatureExtractor `extract.py` `FRAMERATE_DICT = {'s3dg': 16}` |
| 히스토리 총량 | observed segment k개 × 4벡터 | 전체 미편집 영상이 아니라 **경계 주변 샘플링** |
| 최소 segment 길이 | 2초 | `common.py` `minimum_segment_length = 2` |

### 2-1. [중대] 경계 리키지 — 정의와 구현의 불일치

논문 §3 정의는 깨끗하다: *"V_t contains k actions {a1,...,ak}"* — 관측·예측 분리.
그러나 구현(`procedural_planning.py`)은:

```python
# _pick_observations, observation_type == "pre" (best config 전원)
# ISSUE: This is a hack. The next actions pre is passed as the obs after current action.
observations = [ ... for step in steps[1:]]        # 각 step의 관찰 = "다음 step"의 pre-window

# text_to_instance, history(prefix) 관찰 구성
(action_prefix + [action_target[0]])               # ← 첫 예측 대상을 관찰 목록에 붙임
```

→ history 마지막 action에 짝지어지는 관찰 = **`a_{k+1}`(첫 예측 대상)의 `[start−2s, start+2s)`**.
정의상 V_t 밖인 **미래 2초가 입력에 들어간다.**

- i<k에서 "다음 action 시작 직전 구간 = a_i 완료 후 상태"를 쓰는 것 자체는 정당한 모델링 선택.
  문제는 **마지막 윈도우 하나가 경계를 +2초 침범**하는 것뿐(offset을 −4로 잡았으면 무결했다).
- 결과: **첫 스텝 예측은 부분적으로 진행-중-action recognition**. 2스텝부터는 순수 예측
  (미래 관찰은 autoregressive 생성 — 부록 Algo 1). SR/mAcc는 l스텝 평균이라 희석되지만
  **nAcc(다음-스텝 정확도) 계열이 가장 부풀려지는 구조**.
- **우리 V-JEPA2 end−1s late-recognition 확정 건과 동형 문제** — 공개 리더보드 원본 구현에도
  같은 범주의 시간 계약 위반이 있다는 실증. 논문 각주 또는 별도 제보 소재로 가치 있음.

## 3. 질문 ③: 우리 상황(step3_results/vpa)에의 적용성 리뷰

### 3-1. 이미 정합한 것 — 손대지 않아도 됨

| 항목 | 판정 | 비고 |
|---|---|---|
| **지표 구현** | ✅ 등가 확인 | VLaMP SR=전열 일치(`eqs.min(dim=-1)`)·mIoU=샘플별 집합 IoU 평균·mAcc=위치별. 우리 `eval_vpa.py` `per_sample_scores`와 k=1 기준 의미 동일. 원본은 top-k(max-over-k)·t-절단 지원 — 향후 top-k plan 비교 시 그 관례(k개 중 최대) 채택하면 됨 |
| **per-timestep 샘플 생성** | ✅ 동형 | 원본 `create_prefixes: "all"` ≈ 우리 t별 샘플 전개 |
| **Most-Probable baseline** | ✅ 보유 | 원본 라인업과 겹침. Random 1행만 추가하면 라인업 완비(비용 ~0) |
| **시간 계약** | ✅ **원본 구현보다 엄격** | 우리 observed_steps는 경계 이전 완료 step의 GT 라벨만 — 원본의 +2s 리키지 없음 |

### 3-2. 원본에서 차용할 것

1. **(G, A_k) 변형의 선례** — VLaMP 부록은 관찰 없이 action history만 쓰는 변형
   (VLaMP (G, A_k))과 GT-history config(`p3lm-GT`)를 명시적으로 평가한다.
   **우리 text-conditioned 이식 = 원논문의 이 ablation arm을 주 설정으로 승격한 것**으로
   자리매김 가능 → "자체 변형"이 아니라 **원논문 내 선례 있는 설정**이라는 인용 근거.
2. **FRAME HOOK 구현 시 관찰 윈도우 스펙** (경계-안전 수정판):
   - observed step i(완료분)마다 `[start_i−2s, start_i+2s)` 4초 윈도우 — 원본 그대로.
   - **마지막 관찰은 경계 이전에서 절단** — 원본의 `action_target[0]` pre-window 차용 금지.
     대안: 마지막 완료 step의 end 기준 `[end−4s, end)` 또는 boundary−1s 이전으로 제한
     (cesft end−1s 교훈과 일관).
   - 프레임 수 등가: segment당 3~8프레임 샘플링이면 원본(초당 특징 1개 × 4초)과 정보량 등가.
3. **Random baseline 1행** — vocab 균등 샘플 T개. REPORT.md 표 완결성용.

### 3-3. 포팅하지 않을 것 (장애 요인)

| 항목 | 사유 |
|---|---|
| 학습 프레임워크 전체 | **AllenNLP 기반(2022 아카이브)** + 구 torch 고정 + wandb sweep 결합 — 유지보수 불가 계열. 우리 preds-json 분리 설계가 우월 |
| GPT-2+S3D 스택 | 우리 프로젝트는 WM 후보 + CE/SFT VLM 스택으로 이미 상회. VLaMP 재현 arm의 한계효용 낮음 |
| segmentation 모듈(VideoCLIP) | "라벨 접근 불가" 완전 VPA를 하려면 필요하나, HowTo100M(3인칭) 사전학습 ↔ Ego4D egocentric 도메인 갭 + 파이프라인 비용. 현 단계 비권장 — 필요 시 별도 과제로 |
| dataset reader | CrossTask/COIN 전용 — GoalStep 등가물은 이미 `build_goalstep_vpa.py`가 수행 |
| `observation_type: "pre"` 경로 | §2-1 리키지 — **차용 금지** |
| 라이선스 | CC-BY-NC — 연구 사용 OK, 코드 재배포 시 명기 |

## 4. 논문 서술 반영 지시

- **주장 가능**: "우리 text-conditioned VPA는 원논문의 action-history ablation(G, A_k)과 구조 동형이며,
  시간 계약은 원본 공개 구현보다 엄격하다(경계 침범 없음)." / "frontier VLM 비교는 원본 프로토콜에
  없는 본 연구의 확장이다."
- **주장 불가**: 원본 리더보드 수치와의 직접 비교(라벨 공간·데이터셋 상이 — REPORT.md §5 유지) /
  "원본도 VLM baseline을 썼다"류 서술.
- **선택적 기여**: 원본 공식 구현의 첫-스텝 경계 리키지(+2s) 발견 — end−1s 제보 건과 묶어
  "anticipation 벤치마크의 시간 계약 위반" 각주 또는 독립 제보 가능. 제보 시 재현 좌표:
  `procedural_planning.py` `_pick_observations`(ISSUE 주석) + `text_to_instance`
  `(action_prefix + [action_target[0]])` + `p3lm-*.jsonnet` `observation_type: "pre"`.

## 5. 근거 파일 좌표

| 무엇 | 위치 |
|---|---|
| VLaMP clone (실사본) | scratchpad `vlamp_repo/` (세션 임시 — 필요 시 재클론) |
| 관찰 윈도우 산식 | `src/vlamp/dataset_readers/common.py` L160(기본값)·L409-424(pre/post 산출) |
| 리키지 지점 | `src/vlamp/dataset_readers/procedural_planning.py` `_pick_observations`(ISSUE 주석)·`text_to_instance` prefix_observations |
| best config | `best_models_configs/p3lm-*.jsonnet` (`observation_length: 4, observation_offset: -2, observation_type: "pre"`) |
| 지표 원본 구현 | `src/vlamp/training/metrics/{success_rate,accuracy,mean_intersection_over_union}.py` |
| S3D 추출 fps | VideoFeatureExtractor `extract.py` `FRAMERATE_DICT = {'s3dg': 16}` |
| 논문 PDF 실사 | scratchpad `vlamp_{main,supp}.{pdf,txt}` |
| 우리 이식 | `src/ego/step3_results/vpa/` (`REPORT.md`·`eval_vpa.py`·`run_frontier_baseline.py`·`run_qwen_baseline.py` FRAME HOOK L69) |
