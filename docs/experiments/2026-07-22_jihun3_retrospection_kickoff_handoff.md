# EGO_jihun3 착수 Handoff — 인터페이스 계약 · 자산 맵 · 사전 등록

> 작성일: 2026-07-22 KST
> 저장소: `/mnt/nvme/migration/jihun/EGO_jihun3` (main @ `e79d2ac`)
> 방법론 원문: `2026-07-22_nonparametric_prospection_projected_trace_retrospection_handoff.md` (Handoff 1) ·
> `2026-07-22_nonparametric_prospection_dpo_retrospection_handoff.md` (Handoff 2)
> Step-1 인계 보고: `2026-07-22_retro-goalstep-start-m1-lobs8-export-report.md`

---

## 0. 세 줄 요약

1. **jihun2 → jihun3 인계는 완료 상태다.** strict start−1s probe checkpoint(epoch 4)·index·registry가
   `outputs/goalstep/exports/RETRO-goalstep-start-m1-lobs8-best-action-top5/`에 SHA 검증까지 끝나 있고,
   공용 자산(annotation·영상·feature cache·Qwen 가중치) 접근 경로를 전부 실측 확인했다.
2. **Phase-0 coverage 게이트는 사실상 선(先)통과다**: probe full-val(n=7,214) 실측
   cov@15 47.9% / **cov@10 = 39.8%** / cov@5 27.5% / top-1 9.4%. → **Top-K = 10** (07-22 저녁 사용자 확정, 15→10 변경).
3. Step-2 신규 골격 `src/ego/step2_retrospection/`을 스캐폴딩했다 (구 `step2_vlm_alignment` 의존 0).
   `contracts.py`(시간 계약 assertion + support 스키마)는 구현·스모크 통과, 나머지는 단계별 TODO 스텁.

---

## 1. 트랙 원칙 (확정 사항)

| 항목 | 확정 내용 |
|---|---|
| 독립성 | Step-2는 EGO_jihun과 **완전 독립**. `step2_vlm_alignment` import 금지, 코드 이식 금지 |
| Prospection | **non-parametric interface** — 학습 없음. shuffled Top-10, 점수/rank 비공개 |
| Retrospection | Handoff 1 (R1 SFT + R2) 과 Handoff 2 (field-balanced DPO D1/D2) **둘 다** 실험 |
| 성공 판정 | **belief 개입 테스트(③)만**. LLM judge를 성공 지표로 사용 금지 |
| judge 용도 | gemini-2.5-pro는 **pair 품질 게이트 전용** (규칙 기반 선필터 후 의미 판정만). gw.letsur.ai, `LETSUR_API_KEY` |
| history | 1차 검증은 **oracle GT completed-action history** (논문 명시 필수), 확장에서 predicted/no-history |
| Top-K | **10** (07-22 저녁 사용자 확정. cov@10 39.8% — 상한 지표는 coverage@10 기준) |
| Base VLM | Qwen/Qwen3-VL-8B-Instruct (Handoff 명시 그대로. 참고: 구 트랙 EK100에서 Qwen2.5 우위 −0.018 이력 있음 — 문제 시 재론) |

## 2. Step-1 → Step-2 인터페이스 계약

### 2.1 인계 산출물 (불변, 읽기 전용)

```text
outputs/goalstep/exports/RETRO-goalstep-start-m1-lobs8-best-action-top5/
├── best_action_top5.pt        # epoch 4, val-subset action Top-5 26.90% 선택
│                              # sha256 b10ae8ff…37b7 (검증 완료)
├── config_resolved.yaml       # 학습 당시 경로·하이퍼 (backbone 경로만 jihun3 로컬로 대체)
├── final_metrics.json         # full-val 지표 — coverage 참조값의 출처
├── index/{train,val}.parquet  # train 30,374 / val 7,214 (schema: 인계 보고서 §3)
├── index/action_registry.json # verb 81 / noun 140 / action 293, (v,n)→action id
└── val_subset_sample_ids.json # 고정 model-selection subset 2,000
```

### 2.2 시간 계약 (probe와 완전 동일해야 함)

```text
관찰: [A2.start − 9s, A2.start − 1s]   (l_obs 8s, τ_a 1s, 32frames@4fps)
정답: A2 — 관찰창에 등장하지 않은, 1초 뒤 시작하는 다음 action
```

`src/ego/step2_retrospection/contracts.py`의 `assert_strict_contract()`가 강제하는 것:
관측 종료 < target 시작 · gap = 1.0s · history 전 항목 stop ≤ decision time ·
future 전 항목 start ≥ target 시작 · |candidates| = 10 · wm_scores 길이 일치.
**생산자(build_support/build_context)와 소비자(base_trace/train) 양쪽에서 호출한다.**

### 2.3 Step-2가 생성할 파생 산출물 (support jsonl)

`contracts.SupportSample` 직렬화 — 샘플당:
`sample_id · video_uid/clip_uid · obs_start/end · target_start · gt_verb/noun ·`
`candidates(셔플된 10개, "verb noun") · wm_scores(eval 전용, 프롬프트 노출 금지) ·`
`history(oracle 완료 action렬) · future(offline 전용) · boundary_flag`

### 2.4 공용 자산 맵 (접근 실측 확인, 2026-07-22)

| 자산 | 경로 | 상태 |
|---|---|---|
| GoalStep 원본 annotation | `../datasets/Ego4D/v2/annotations/goalstep_{train,val}.json` | ✓ 존재 |
| GoalStep 영상 (256GB) | `../datasets/Ego4D/v2/goalstep_videos` | ✓ 존재 |
| feature cache (313GB) | `../datasets/Ego4D/goalstep_feature_cache_start_m1_lobs8_vna` | ✓ 존재 |
| V-JEPA2 backbone | **`checkpoints/vjepa2/vitl.pt` (jihun3 로컬 사본)** | ✓ 5.1GB 복사, sha256 원본과 일치 |
| Qwen3-VL-8B / Qwen2.5-VL-7B | `HF_HOME=/mnt/nvme/cache` 공용 캐시 | ✓ 존재 |
| gemini 게이트웨이 | `https://gw.letsur.ai/v1`, env `LETSUR_API_KEY` | 키는 셸에서 export |

> 백본만 로컬로 복사한 이유: export config가 `../EGO_jihun/checkpoints/…`를 참조하고 있었는데
> 이는 독립성 원칙 위반 경로라 jihun3 사본으로 대체했다 (`configs/step2_retrospection/*.yaml` 반영).

## 3. jihun2 Plan A 런 상태 (2026-07-22 20:4x KST 기준)

| 런 | 상태 |
|---|---|
| `z1_start_m1_lobs8_vna` | **완료 + export 인계 완료** — 이 문서의 기준 checkpoint |
| `z1_start_m1_lobs16_vna` | 진행 중 (11:19 착수, val feature 추출 단계) — 완료 시 l_obs 16s 비교점으로 인계 요청 가능 |
| `z1_end_m6_lobs8_vna_ep10` | 진행 중 (옵션 8 계열) — 본 트랙과 무관, anchor는 strict start−1s 유지 |

**jihun3은 lobs8 checkpoint만으로 전 단계 착수 가능** — lobs16은 도착하면 ablation 열에 추가.

## 4. 신규 골격 (스캐폴딩 완료)

```text
src/ego/step2_retrospection/
├── contracts.py            ✅ 구현 + 스모크 통과 (계약 assertion·스키마·TOP_K=10)
├── data/build_support.py      probe → Top-10 support 덤프 + coverage 리포트   [Phase-0]
├── data/build_context.py      goalstep json → history/future 조인             [Phase-1]
├── prospection/base_trace.py  Base Qwen zero-shot trace (y−)                  [Phase-1]
├── hindsight/teacher.py       Ψ: future → task structure                      [Phase-1]
├── hindsight/projection.py    Φ: 과거 evidence 수준 재작성 (y+)               [Phase-1]
├── hindsight/quality_gate.py  규칙 게이트 + 공용 직렬화(serialize_trace) ✅   [Phase-1]
├── hindsight/semantic_gate.py gemini 의미 판정 (게이트 전용)                  [Phase-1]
├── pairs/build_pairs.py       acceptance·taxonomy·weight + §12 리포트         [Phase-1]
├── train/sft_r1.py            span-normalized SFT (λ_b 1.0/λ_r 0.5/λ_a 0.25) [Phase-2]
├── train/consistency_r2.py    belief-conditioned aux (순환성 주의 명기)       [Phase-2]
├── train/dpo_fb.py            field-balanced DPO D1/D2 + 가드레일             [Phase-2]
├── eval/battery.py            coverage·SelAcc@10·G1/G2·L0 행 강제             [Phase-0~]
└── eval/intervention.py       개입 테스트 ③ — 유일한 성공 판정               [Phase-3]
configs/step2_retrospection/goalstep_start_m1_lobs8.yaml   ✅ 실측 경로 반영
```

구현 순서: **build_support(+coverage) → build_context → base_trace → teacher/projection/gate →
sft_r1 → pairs → dpo_fb(D1) → D2 → intervention.** (D2가 SFT를 전제, §12 fallback 판정에도 필요)

## 5. 사전 등록 (실행 전 고정)

### 게이트

| # | 게이트 | 기준 | 미달 시 |
|---|---|---|---|
| G0 | support coverage | 자체 덤프의 cov@10이 probe 실측(39.8%)과 ±2pp 내 재현 | 파이프라인 버그 — 학습 착수 금지 |
| G1 | projected trace 수율 | gate 통과율 ≥ 30% (drop-not-patch 유지) | projection 프롬프트 재설계, teacher 교체 검토 |
| G2 | DPO pair 신호 | final pairs ≥ 2,000 **AND** BA+B ≥ A 개수 | D1 폐기, D2만 진행 (§12) |
| G3 | 문체-학습 가드 | 학습 중 belief-margin↑ & action-margin 정체 감지 시 | 해당 run 자동 중단 |

### 성공 기준 (heldout 1회 측정, dev와 video-disjoint)

- **주장 1 (support)**: Base+Top10 > Base(후보 없음) — SelAcc 비교
- **주장 2 (retrospection)**: R1(+R2) 또는 DPO가 Base+Top10 대비 SelAcc@10 **또는** G2 correction 개선, G1 보존
- **주장 3 (belief 인과, 핵심)**: **생성-belief 기준** 개입 부등식
  `p(a_GT|b_gen) > p(a_GT|∅) > p(a_GT|b_incompat)` + paraphrase-control 차감 후 유의
- **비교 주장**: Retro > Action-only SFT (belief의 독자 가치, Handoff 1 §16)

### 보고 규율 (구 트랙 실수의 재발 방지)

- 모든 결과표에 **L0 행**(무학습 WM top-1 추종, 참조 top-1 9.4%) 강제
- n·split id 병기, n 혼재 표 금지 · 부분 데이터로 방향 판정 금지 (최소 표본 코드에 명시)
- 승자 arm은 시드 2회 + 별도 split 재측정 후에만 순위 주장

## 6. Ablation 계획 (요약 — 상세는 이전 논의)

핵심 7 arm 우선: Base(후보 없음) · Base+Top10 · +R1 · +R1+R2 · Action-only SFT ·
FB-DPO(D1 또는 D2) · Random Top-10. 방어용(Standard DPO·Action-only DPO·Belief-only ·
Random pair · Frequency/Oracle Top-K)은 승자 확정 후.
예상: 핵심 7 arm 1차 결론까지 ~2일, 전체 3–5일 (GPU 2장, 무인 체인 기준).

### 6.1 DPO pair-gate 3-arm ablation (07-22 저녁 사용자 확정, 사전 등록)

"DPO가 아니라 **validated preference**가 핵심"(Handoff 2 §0)을 직접 검증하는 축.
공통: 같은 chosen/rejected 원료, 같은 FB-DPO 트레이너 — 차이는 pair 게이트뿐.

| arm | pair 구성 | pairs 파일 | G3 가드 |
|---|---|---|---|
| **DPO-all** | 무조건 B≻A — acceptance 조건 없음 (identical·style-only 포함) | `pairs_train_all.jsonl` | **warn** (실패 관측이 목적) |
| **DPO-rule** | 규칙 판정만 (Jaccard·leakage·restatement·malformed) | `pairs_train.jsonl` | abort |
| **DPO-sem** | 규칙 + gemini 4항목(belief_equivalent·style_only·chosen_grounded·restates) | `pairs_train_sem.jsonl` | abort |

**사전 등록 예측**: DPO-all은 문체-학습 신호(belief margin↑·action margin 정체,
`G3_STYLE_WARN_dpo_all` 마커)를 보이고 ③·G2에서 열세 — B0 실패의 통제된 재현.
gate가 강할수록(all→rule→sem) belief 품질·③이 개선되면 게이트의 순기여 입증.
**판정 지표**: ③(생성-belief 개입) + G2 correction + restatement rate. acc 단독 비교 금지.
**실행**: `scripts/step2_retrospection/retro3_ablation_chain.sh` — 메인 체인 완료 후 자동,
DPO-sem은 `LETSUR_API_KEY` 있을 때만 (없으면 `ABLATION_SEM_PENDING` 마커로 보류).

## 7. 운영 — 무인 실행·대시보드·자가 복구 (2026-07-22 저녁 가동)

```bash
# 체인 기동 (ssh 분리, nohup+setsid; marker 기반 resume이라 재실행 안전)
bash scripts/step2_retrospection/start_chain.sh
# 대시보드 (stdlib http.server)
python3 tools/retro3_dashboard.py --port 7867   # http://<host>:7867
# 로컬에서: ssh -L 7867:localhost:7867 <host>
```

- **체인**: `scripts/step2_retrospection/retro3_chain.sh` — S0 support → S1 context →
  S1b subset(6,000) → S2 base trace → S3 Ψ→Φ→gate → (S4 semantic, 키 있을 때) →
  S5 pairs → S6 R1 SFT → S6 FB-DPO(D1) → S7 배터리×3 arm + 개입③×3 arm.
  스테이지 성공 판정은 exit code가 아니라 **marker** (`runs/retro3/markers/`).
- **자가 복구**: `supervisor.sh`가 체인 사망 시 자동 재시작 (같은 지점 연속 5회 실패 →
  `CHAIN_STUCK`, GPU 접근 불가 → 정지). 각 모듈은 sample 단위 resume.
- **대시보드**: 스테이지 진행/실측 ETA(처리율 기반)·전체 남은 시간·coverage/gate/pair/
  margin/eval/③ 실측값·GPU·로그 tail, 5초 자동 갱신. 데이터 원천은
  `runs/retro3/status/*.json`(원자적 쓰기) + markers + eval json.
- **가드레일 내장**: G2(pair 부족 시 D1 중단) · G3(belief margin만 상승 시 자동 중단,
  `G3_STYLE_ABORT`) — 대시보드 배너로 표시.

### 확정된 값 (07-22 저녁)

- future m: **최대 5 action, 24s 캡** (`FUTURE_MAX_ACTIONS=5`, `FUTURE_MAX_SECONDS=24`)
- Top-K: **10** · dev/heldout: val video-disjoint 30/70 (`splits.json`)
- train 코퍼스: 6,000 샘플 (video당 ≤30, seed 42)

## 8. 남은 열린 항목

- [ ] `LETSUR_API_KEY` export 후 S4 semantic gate 재실행 (현재 키 없으면 규칙 게이트만)
- [ ] D2 (SFT warm-up → trace 재생성 → pair 재구성 → DPO) — 1차 체인 결과 확인 후
- [ ] Action-only SFT · Random Top-10 arm — 1차 결과 후 추가
- [ ] lobs16 checkpoint 도착 시 ablation 열 추가 여부
- [ ] 스캐폴딩 커밋: `git add src/ego/step2_retrospection configs/step2_retrospection scripts/step2_retrospection tools/retro3_dashboard.py docs/experiments/2026-07-22_*.md`
