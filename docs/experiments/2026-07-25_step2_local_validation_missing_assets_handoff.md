# Step-2 (cesft_v2) 로컬 검증 — 다른 서버에서 가져와야 할 자산 Handoff

> 작성: 2026-07-25 KST. 대상: `EGO_jihun3`(원격, `/mnt/nvme/migration/jihun/`)에서
> 로컬 워크스테이션(`~/Project/EGO`, NVIDIA GB10 / aarch64)으로 자산을 옮겨줄 사람.
> 선행 문서: `develop_report/2026-07-25_step2_cesft_sft_r15_model_usage_handoff.md` (§ 번호는 이 문서 기준)

---

## 0. 요약 — 무엇이 됐고 무엇이 막혔나

**모델은 로컬에서 정상 동작한다.** Qwen3-VL-8B-Instruct + LoRA `sft_r15`를 로드해 100샘플을
끝까지 돌렸고, 3태그 trace 생성·후보 매칭·집계까지 shipped 코드(`build_context`, `eval.battery`)
그대로 통과했다. 즉 **코드/모델/GPU 쪽은 더 받을 게 없다.**

**막힌 것은 데이터 한 가지다.** WM prior가 만든 **Top-10 후보(`context_val.jsonl`)**가 로컬에
없고, 그걸 로컬에서 재생성할 수도 없다(§3). 그래서 이번 검증은 후보를 합성해서 돌렸고,
그 결과 SelAcc는 원격 `sft_r15.json`과 **직접 비교할 수 없다**.

→ **§2의 Tier 1 네 덩어리만 받으면 정식 프로토콜로 재현·수치 대조가 가능해진다.**

---

## 1. 이번 로컬 스모크 결과 (참고용 — 후보 분포가 다름)

`runs/cesft_v2/eval/sft_r15_local100.{json,records.jsonl}`

| 지표 | 로컬 스모크 (n=100) | 원격 정식 (`sft_r15.json`) | 비교 가능? |
|---|---|---|---|
| SelAcc (covered) | 31.0% | 31.7% | **불가** — 후보 분포 상이 |
| malformed rate | 2.0% | — | 참고 |
| cov@10 | 100% (합성이라 GT 항상 포함) | ≈43% | 불가 |
| L0 (WM top-1) | 20.0% (합성 점수 = 무작위) | 23.7% | **불가** — 실제 WM 아님 |
| frame/decode 오류 | 0 / 100 | — | — |
| 처리 속도 | 6.19 s/sample (batch 8), 총 415s | — | — |

**trace 품질 계약은 100% 준수** (n=100): reasoning 3–6문장 100%, task_belief 1문장 100%,
task_belief에 정답 행동 verbatim 노출 0%. 평균 reasoning 492자 / 3.69문장.
→ sft_r15의 목적(검증가능한 추론 trace)은 로컬에서도 그대로 재현된다.

숫자 31.0%가 원격 31.7%와 우연히 가깝지만 **의미 없는 일치**다. 로컬은 GT가 항상 후보에
있고(cov@10=100%) distractor가 빈도 기반 무작위라 WM prior의 hard negative보다 쉽다.
반대로 표본이 3개 영상 100샘플뿐이라 분산이 크다. 두 편향의 방향이 달라 상쇄 여부를 알 수 없다.

---

## 2. 다른 서버에서 가져와야 할 것

원격 루트: `R = /mnt/nvme/migration/jihun`

### Tier 1 — 정식 프로토콜 재현에 **필수** (이것만 받으면 됨)

| # | 원격 경로 | 로컬 배치 위치 | 크기 | 무엇이 풀리나 |
|---|---|---|---|---|
| 1 | `R/EGO_jihun3/runs/cesft_v2/data/context_val.jsonl` | `runs/cesft_v2/data/` | 미확인(수십 MB 예상) | **핵심.** WM prior Top-10 후보 + history/future. 이게 없으면 정식 평가 자체가 불가 |
| 2 | `R/EGO_jihun3/runs/cesft_v2/frame_cache/` | `runs/cesft_v2/frame_cache/` | 미확인(수십 GB 예상) | 관측창 8프레임 사전추출본. **이걸 받으면 goalstep 원본 영상(≈112 GiB)을 안 받아도 됨** |
| 3 | `R/EGO_jihun3/runs/cesft_v2/eval/{base,theta_ce,sft_r15}.{json,records.jsonl}` | `runs/cesft_v2/eval/` | 수 MB | 대조 기준선. 같은 샘플에서 로컬 재현치와 1:1 대조 |
| 4 | `R/EGO_jihun3/runs/cesft_v2/overrides.json` | `runs/cesft_v2/` | 1 KB | `eval_covered_only` 플래그. 없으면 covered-only 평가가 자동으로 안 켜짐 (`battery.py:62-67`) |

**2번(frame_cache)을 못 받는 경우에만** 아래를 대신 받는다:

| 대안 | 원격 경로 | 크기 | 비고 |
|---|---|---|---|
| goalstep 원본 영상 | `R/datasets/Ego4D/v2/goalstep_videos/` | **≈112 GiB** (val 133편 × 평균 0.84 GiB) | 로컬엔 현재 7편(val 3 / train 4)뿐. 로컬 디스크 여유 1.7 TB로 수용 가능 |

> `context_val.jsonl`의 heldout split에 등장하는 `video_uid`만 추리면 133편보다 줄일 수 있다.
> 전송 전에 원격에서 `jq -r 'select(.split=="heldout").video_uid' context_val.jsonl | sort -u | wc -l`로 확인 권장.

### Tier 2 — 새 데이터에 WM prior를 **직접 돌려야 할 때만**

| # | 원격 경로 | 크기 | 비고 |
|---|---|---|---|
| 5 | `R/datasets/Ego4D/goalstep_history_context_store/` | 미확인(대용량) | `[N,17,1024]` fp16 summaries + frozen visual logits. **§3의 병목** |
| 6 | `R/EGO_jihun3/outputs/goalstep/exports/RETRO4-goalstep-end-m1-history-k8-phase1/index/` | 미확인 | `{train,val}.parquet` + `action_registry.json`. sample_id·target_start의 SSOT |
| 7 | `R/EGO_jihun3/runs/cesft_v2/data/context_train.jsonl` | 미확인 | 재학습/추가 SFT 할 때만 |

Tier 2는 §5(새 비디오·새 구간 사용) 시나리오 전용이다. **단순 검증·수치 대조가 목적이면 불필요.**

### 이미 로컬에 있는 것 — 다시 보낼 필요 없음

| 자산 | 로컬 경로 | 상태 |
|---|---|---|
| LoRA 어댑터 `sft_r15` | `outputs/step2_retrospection/cesft_v2/sft_r15/adapter/` | ✅ 59 MB, r=16 α=32, peft 0.19.1 |
| LoRA 어댑터 `theta_ce` | `outputs/step2_retrospection/cesft_v2/theta_ce/adapter/` | ✅ 59 MB |
| WM prior 체크포인트 | `outputs/goalstep/exports/RETRO4-.../best_action_top5.pt` | ✅ 452 MB, **sha256 검증 통과** (`1d20c942…a61cd`, cesft_v2.yaml:16 고정값과 일치) |
| 베이스 VLM | `~/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct` | ✅ 17 GB, snapshot `0c351dd0…` |
| step2 코드 전체 | `src/ego/step2_retrospection/` | ✅ 2026-07-25 pull(`656554d`)로 확보 |
| goalstep 주석·taxonomy | `data/Ego4D/v2/annotations/`, `src/.../goalstep/taxonomy/` | ✅ |

---

## 3. 왜 로컬 재생성이 불가능한가 (조사 결과)

`context_val.jsonl`을 로컬에서 만들려면 `build_support_hk8.py` → `build_context.py` 순인데,
**첫 단계에서 막힌다.** 세 가지가 동시에 필요하다:

1. **파생 store** (`derived_store_dir`) — `DerivedStore`가 `manifest.json` + `{split}/*.pt`에서
   `summaries [N,17,1024]`와 `visual_logits`를 읽는다 (`build_support_hk8.py:39-69`). 로컬에 없음.
2. **retro4 index parquet** — `sample_id`·`target_start_sec`·`history_{1..8}_*` 컬럼이 필요하다
   (`build_support_hk8.py:101-112`). 로컬 `outputs/goalstep/index_smoke/val.parquet`은 495행이지만
   컬럼이 `video_uid/clip_uid/obs_{start,end}_sec/verb_label/noun_label/action_label/scenario/boundary_flag`
   뿐 — **history 컬럼도 sample_id도 없고 start−1s 계약이라 대체 불가.**
3. **frozen visual logits** — 이게 결정적이다. export 체크포인트 메타데이터가
   `"visual_path": "frozen_precomputed_logits"`라고 명시한다. 즉 `best_action_top5.pt`는
   **contextual/fusion 부분만** 담고 있고 visual head는 들어있지 않다.
   raw feature cache(`data/Ego4D/goalstep_feature_cache_smoke`, `features [4352,1024]`)가
   로컬에 있어도 visual logits를 만들 수 없다.

→ 체크포인트만으로는 Top-10을 재생성할 수 없다. **파생 store(#5)를 받거나, 이미 만들어진
결과물(#1 `context_val.jsonl`)을 받는 수밖에 없다.** 후자가 훨씬 작고 빠르다.

---

## 4. 전송 후 실행 절차

```bash
cd ~/Project/EGO
# (Tier 1 배치 후) — frame_cache를 받았으면 영상 없이도 동작
RETRO3_RUNS=runs/cesft_v2 \
RETRO_NEXT_GAP_TEXT="after the current action ends" \
HF_HOME=$HOME/.cache/huggingface \
PYTHONPATH=src <VENV>/bin/python -m ego.step2_retrospection.eval.battery \
  --config <로컬경로판 cesft_v2.yaml> --arm sft_r15_repro \
  --adapter outputs/step2_retrospection/cesft_v2/sft_r15/adapter \
  --eval_n 100 --batch_size 8
# 대조: runs/cesft_v2/eval/sft_r15.json 의 acc / L0_wm_top1 / coverage_at_k 와 비교
```

config는 원격 경로(`../datasets/...`)를 쓰므로 로컬 경로판 사본을 쓴다 —
`configs/step2_retrospection/cesft_v2_local.yaml` (`shared_assets`의 `annotations_dir`,
`video_root`, `hf_home` 세 줄만 로컬로 바꾼 것. `derived_store_dir`은 Tier 2를 받기 전까지 미해결로 남겨둠).

이번 스모크 실행에 쓴 보조 스크립트 (Tier 1 도착 전까지만 필요):

| 스크립트 | 용도 |
|---|---|
| `tools/local_eval/build_smoke_support.py` | WM prior 없이 합성 Top-10으로 `support_val.jsonl` 생성 — **Tier 1 #1이 오면 폐기** |
| `tools/local_eval/extract_frames_av.py` | PyAV로 `frame_cache` 생성 (decord 대체) — **Tier 1 #2가 오면 불필요** |
| `tools/local_eval/report_eval.py` | arm별 지표 + trace 품질(문장수·1문장·verbatim leak) 요약 |

`--arm` 이름이 같으면 기존 records를 이어받으므로(resume) 새 실험은 새 arm 이름을 쓸 것.

---

## 5. 로컬 환경 재현 메모 (원격 `eve-cu124`와 다름 — 하드웨어가 aarch64/GB10)

원격 env를 그대로 못 쓴다. 이번에 동작 확인된 조합:

| 패키지 | 버전 | 메모 |
|---|---|---|
| transformers | 5.14.1 | 4.56.1은 **Qwen3-VL 미지원**(`Qwen3VLForConditionalGeneration` ImportError). 5.x 필요 |
| peft | 0.19.1 | 어댑터 `adapter_config.json`의 `peft_version`과 일치 |
| torch / torchvision | 2.12.1+cu130 / **0.27.1+cu130** | **짝을 맞춰야 함.** torchvision 없거나 버전 불일치 시 `Qwen3VLVideoProcessor requires the Torchvision library` 또는 `operator torchvision::nms does not exist`로 프로세서 로딩 실패 |
| decord | **설치 불가** | aarch64 휠 없음 ⇒ 아래 우회 필수 |

**decord 우회**: `vlm.py`는 decord를 모듈 최상단이 아니라 `extract_frames`/`_video_reader` **안에서**
import한다. `_load_cached`(`vlm.py:29-44`)가 8장 전부 히트하면 decord 경로에 진입하지 않는다.
따라서 **`FRAME_CACHE_DIR`에 프레임이 완비되어 있으면 decord 없이 평가가 돌아간다** — 이번 실행이 그 증거다.
캐시 규칙: `{FRAME_CACHE_DIR}/{video_uid}/{sample_id}/f{0..7}.jpg`, 8장 전부 존재 + 크기 동일.
로컬에서 원본 영상으로 캐시를 만들 때는 decord와 동일 인덱싱을 재현해야 한다:
`idx[i] = clamp(round((t0 + (t1-t0)·i/7) · fps), 0, n_total-1)`, 짧은 변 336 축소(확대 금지).
PyAV로 재현한 스크립트: `tools/local_eval/extract_frames_av.py`.

> **원격 frame_cache(#2)를 받으면 이 재현 자체가 불필요해진다** — Tier 1에서 #2를 권하는 이유.

---

## 6. 요청 체크리스트 (원격 담당자용)

- [ ] `runs/cesft_v2/data/context_val.jsonl` 전송 — **최우선**
- [ ] `runs/cesft_v2/frame_cache/` 전송 (크면 heldout split에 해당하는 video_uid만)
- [ ] `runs/cesft_v2/eval/{base,theta_ce,sft_r15}.{json,records.jsonl}` 전송
- [ ] `runs/cesft_v2/overrides.json` 전송
- [ ] 전송 전 각 항목 크기 회신 (`du -sh`) — 로컬 여유 1.7 TB
- [ ] frame_cache가 없거나 불완전하면 → `datasets/Ego4D/v2/goalstep_videos/` 중 heldout video_uid만
- [ ] (새 데이터에 WM prior를 돌릴 계획이 있을 때만) `goalstep_history_context_store/` + `exports/RETRO4-.../index/`
