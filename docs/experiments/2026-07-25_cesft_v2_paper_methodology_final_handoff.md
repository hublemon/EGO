# cesft_v2 논문 방법론 확정 Handoff — CE → Projected-Trace SFT (DPO-free two-stage)

> 작성: 2026-07-25 KST · EGO_jihun3. **cesft_v2 실측(2026-07-24)으로 방법론을 논문 확정판으로 승격.**
> 성격: 이 문서가 **논문 §Method 의 SSOT(확정)**. 이전 방법론 문서(계획·상태)는 §7 참조 지도에.
> 실측 근거: `2026-07-25_cesft_v2_quantitative_evidence_handoff.md` (능력별 정량 지표 전량).
> 시각 보고서: https://claude.ai/code/artifact/922dc65e-5fd9-4b8b-b7b6-48e2b02734d2

---

## 0. 결론 — 확정 판단

**골격 확정 가능. 단 서사는 "분업"으로 정직하게.**

- **확정**: 2단계 구조(candidate-CE → projected-trace SFT), 평가 프로토콜(정적+개입+strip).
  핵심 주장 = "정확도와 인과적 추론은 서로 다른 단계가 만든다" — 게이트로 실증(§2).
- **수치는 full 학습으로 갱신**: 현 N=2,500 파일럿. 방법 **구조**는 확정, **수치**는 full-val로 대체.
- **정직성 3원칙** (논문 서사 강제):
  1. **SFT는 acc를 안 올린다** (G-NH FAIL, +1.1pp 비유의). "SFT의 목적은 정확도가 아니라 검증가능한
     추론" — 역할 분리로 프레이밍. "SFT가 정확도도 개선" 주장 **금지**.
  2. **주장 금지선**: "egocentric reasoning"(1인칭 침식 반증, SFT 화법 이관 전) · "causal mediation"
     (interventional dependence까지만) · 두 코호트(cesft_v2↔JIHUN) 수치 직접 비교.
  3. **확정 전 보완 2건**: G-DELTA 본셋 실증 · full 학습 (§6).

---

## 1. 확정 방법론

### 1.1 문제 정의

next-action anticipation. 컨텍스트 `c_t = (영상 프레임 x≤t, 완료 행동 이력 H<t, WM Top-K 후보 D_t)`.
VLM(Qwen3-VL-8B + LoRA)이 trace `y_t = (reasoning r, task_belief g, action a ∈ D_t)`를 생성.
**WM prior** = jihun2 Phase-1 history-context K8 (`RETRO4-...-phase1/best_action_top5.pt`, cov@10 43.4%,
end−1s 계약, 읽기전용 export). VLM은 8초 관측창 8프레임(336px)을 vision-grounded 입력으로 실제로 봄.

### 1.2 Stage 1 — θ_CE (Predictive-Boundary Selection, candidate-CE)

- **입력**: 프레임 + history + **후보 D_t 제시** (셔플, WM rank/prob 비공개)
- **손실**: 후보 span CE
  ```
  s(a)   = (1/|a|) Σ log π(aⱼ | c, a<ⱼ)      # length-norm 후보 점수
  p(a|D) = softmax over D_t ( s(a)/τ )
  L_sel  = − log p(a_GT | c, D_t)
  ```
- covered 샘플만 학습 (GT ∈ D_t)
- **산출**: 판별력 높은 정책 θ_CE. **이것이 정확도 엔진.**

### 1.3 Stage 2 — θ_CE + Projected-Trace SFT (+ CE replay)

- **초기화**: **θ_CE에서** (base 아님 — CE의 acc를 보존한 채 인과성만 추가)
- **목표**: teacher가 hindsight로 만든 뒤 결정시점에 투영한 이상적 `(r_proj, g_proj, a_GT)` 재현
  (누출·restatement 게이트 통과분)
- **손실**: field-weighted span CE — **belief 1.0 / reasoning 0.5 / action 0.25**
  (belief 최고가중 = belief→action 인과 경로를 심는 핵심)
- **CE replay 15~20%**: SFT step의 일부를 candidate-CE 배치로 섞어 판별력(SelAcc/GADR) 퇴화 방지
  (continual-replay). 기본 15%, non-harm 실패 시 상향.
- **산출**: 인과 제어 강 + 판별력 보존 정책 θ_CE+SFT

### 1.4 순서·금지

- **순서**: CE → SFT(+CE replay). SFT를 마지막에 둬 belief-인과를 fresh하게, CE는 replay로 보호.
- **금지**: CE → SFT → **full-CE** (마지막 full-CE가 action을 직접 최적화해 belief-인과 워시아웃).

### 1.5 논문 매핑

| 논문 자리 | 구현 | 학습 신호 |
|---|---|---|
| **Prospection** (선택 정렬) | candidate-CE (θ_CE) | 후보 위 선택 CE |
| **Retrospection** (의미 정렬) | projected-trace SFT | hindsight 투영 trace SFT |

---

## 1.6 구현 세부 (EGO_jihun3 cesft_v2 실제 코드)

공통: LoRA `r=16, α=32, dropout=0.05, bias=none, target=[q,k,v,o]_proj`, gradient checkpointing,
AdamW, grad-clip 1.0, cosine schedule, accum 8, epochs 1. 모델 Qwen3-VL-8B-Instruct.
데이터: `context_train.jsonl`(frames·history·candidates·gt) · `train_subset.json`(video-disjoint covered 6k) ·
`chosen_train.jsonl`(projection trace, gate=pass). config `configs/step2_retrospection/cesft_v2.yaml`.

### ① Stage 1 — θ_CE : `train/select_ce.py` (arm=wm_cand)

- **후보 채점** `_candidate_logps()`: K개 후보를 **하나의 배치로 forward** (동일 frames+history 프롬프트,
  후보만 다른 completion). **video-grounded** — 8초 관측창 8프레임(336px) + history를 VLM이 실제로 봄
  (기존 tok()-only 경로는 vision-blind였음, 2026-07-24 교정). teacher reasoning/belief는 미사용(순수 action selection).
- **점수·손실** `selection_ce_step()`:
  ```
  s(a) = mean logp of action-span tokens  (<action>\n 접두 뒤 후보 토큰만, length-norm)
  L_sel = − log softmax(s / τ)[gt_idx]     over D_t (WM Top-10 후보, 셔플)
  ```
  τ=1.0. 후보는 `rec["candidates"]`(셔플 저장), gt_idx로 정답 위치.
- **학습 데이터**: `context_train.jsonl` 중 `gt_rank ≤ 10` ∧ train_subset(covered)만.
  비디오별 그룹 정렬 + `prefetch_chunks`(decord 캐시 히트, 500MB 영상 재디코드 방지).
- **하이퍼**: lr 1e-5, τ 1.0, accum 8, epochs 1.
- **강건성**: OOM·decode 실패·Qwen3-VL patch 타일링 실패(size 69395200 등)는 **샘플 스킵**(로그만) — 4h 학습 보호.
- **산출**: `outputs/step2_retrospection/cesft_v2/theta_ce/adapter`.

### ② Stage 2 — projected-trace SFT : `train/sft_v2.py`

- **초기화**: `--init_adapter theta_ce/adapter` (θ_CE의 LoRA state 로드 = warm-start, acc 보존).
- **SFT 손실** `sft_step()` → `common.span_ce_loss()`:
  - completion 직렬화(`common.completion_pieces`): `<reasoning>…</reasoning>` `<task_belief>…</task_belief>` `<action>…</action>`.
  - **field별 토큰 마스크**: 프롬프트는 processor(chat template+이미지), completion은 조각별
    `tokenizer(add_special_tokens=False)`로 이어붙여 **조각 경계=span 경계 일치** → field 마스크 정확.
  - **field 가중** `FIELD_WEIGHTS = {task_belief 1.0, reasoning 0.5, action 0.25}`:
    ```
    L_sft = Σ_f w_f · ( − mean logp of field-f tokens )
    ```
    belief 최고가중 = belief→action 인과 경로를 심는 핵심.
  - Qwen3-VL M-RoPE: 프롬프트의 per-token 텐서(mm_token_type_ids)를 completion 길이만큼 0 연장.
- **학습 데이터**: `chosen_train.jsonl` gate=pass (teacher Ψ → projection Φ → 누출·restatement gate 통과분).
- **하이퍼**: lr 5e-5, accum 8, epochs 1.
- **산출**: `.../sft_r15/adapter` (기본 arm).

### ③ CE replay (Stage 2 내 인터리브) — `sft_v2.py` 핵심

**방식 A — 합산손실이 아니라 micro-step 인터리브** (gradient 충돌 회피, 순차 적층의 덮어쓰기 방지):

```
매 micro-step:
  do_ce = (rng.random() < ρ)                        # 기본: 확률 혼합
        = (micro % period == period-1)  if --alternate  # period=round(1/ρ) 결정적 교대
  if do_ce:  loss = select_ce.selection_ce_step(...wm_cand...)   # CE 판별 채널 앵커
  else:      loss = sft_step(...)                                # projected-trace SFT
  (loss / accum).backward()                          # 두 손실을 합치지 않고 스텝 단위 교대
```

- **ρ = 0.15 기본**(sft_r15). arm 스윕: `sft_r0(ρ=0)` · `sft_r15(ρ=0.15)` · `sft_r30(ρ=0.30)`.
  ρ=0은 순차 적층(앵커 없음), G-NH 비열등 실패 시 ρ 상향.
- **CE 스트림 재사용**: `select_ce.selection_ce_step(arm="wm_cand")` 그대로 호출 — θ_CE와 **동일 분포**
  (covered pool, video-grounded, 같은 후보 채점). 별도 CE 데이터 없음.
- 두 손실이 **같은 옵티마·같은 accum 버퍼**에 누적되지만 스텝마다 하나만 활성 → belief-인과(SFT)와
  판별(CE)이 서로를 워시아웃하지 않음.

### ④ 체인·산출 (`scripts/step2_retrospection/cesft_v2_chain.sh`)

```
E1  select_ce theta_ce (wm_cand)     → G-ACC1 (vs L0)
E1b select_ce cand_free / no_history → G-DELTA·strip 대체 arm
①  sft_v2 {r0,r15,r30} (init=theta_ce, ρ 스윕) → G-NH (vs theta_ce)
    battery 평가 + harden_s3 개입(U_g) → verdict
```
marker 멱등, preflight(GPU<30GB), setsid 분리 기동(`start_cesft_v2.sh`).

---

## 2. 실측이 지지하는 것 (cesft_v2 게이트 — 확정 근거)

| 게이트 | 판정 | 수치 | 방법론적 의미 |
|---|---|---|---|
| **G-ACC1** CE > WM top-1 | **PASS** | SelAcc 30.8 vs L0 23.7, Δ+7.2pp[0.5,15.0] | CE가 모방을 넘는 선택 엔진 |
| **개입 U_g** reasoning 유용 | **PASS** | +9.8pp[7.2,12.4] · both flip 81.8% vs para 4.5% | SFT가 인과적 추론 채널 (spine 확정) |
| **G-CC1∧CC3** belief 인과 | **PASS** | causal 0.296 · U_g belief-only +5.0[3.3,6.6] | belief→action 경로 실재 |
| **strip** history 인과 | **PASS** | Δ+3.1pp[1.1,5.2] · H8 유의 | history를 판단에 사용 |
| **G-NH** SFT 판별 비손상 | **FAIL(중립)** | +1.1pp[−5.7,+7.1] | **acc는 CE 몫** — 정직 서사의 핵심 |

**분업 지도** (방법론의 실측 정당화):

| 능력 | candidate-CE | projected-SFT |
|---|---|---|
| 후보 선택·정확도 (SelAcc) | ✓ 20→30.8 | 중립 |
| 판별 (GADR) | ✓ G2 16.4→25.6 | 중립 |
| history 사용 (strip) | ✓ +3.1 · 자매 DiD +8.4 | — |
| belief→action 인과 | ✗ 미형성 | ✓ 0.296 · 자매 6.7× |
| reasoning 인과·유용 | — | ✓ flip 81.8 · U_g +9.8 |
| 소거 서술 | ✗ 침식 10.4→2.4 | ✓ 회복 →25.5 |

→ **두 단계가 서로 다른 능력을 심는다** = two-stage 방법론의 정량 근거.

---

## 3. 논문 §Method 서사 (권장 골격)

1. **문제·경계**: WM(Phase-1)이 next-action 후보 경계 D_t를 제공. "LM은 WM의 가능성 경계 안에서 추론".
2. **Stage 1 (Prospection/CE)**: 후보 위 선택 정렬로 판별 엔진 — SelAcc가 WM top-1 모방을 초과(G-ACC1),
   GADR로 hard-case 교정, history를 판별 단서로 사용(strip/DiD).
3. **Stage 2 (Retrospection/SFT)**: hindsight-투영 trace로 belief→action 인과를 심음 — 개입으로
   검증(swap flip 82% vs paraphrase 4.5%, U_g>0). **acc는 CE가 이미 확보, SFT는 검증가능한 추론 담당.**
4. **분업 주장**: 두 단계는 다른 능력(정확도 vs 인과추론)을 만든다 — 절제(no acc-inflation claim).

---

## 4. 평가 프로토콜 (확정)

- **정적(선택)**: `eval_candidate_scored.py` — K=10 teacher-forcing sum-logp argmax. **full acc + covered
  SelAcc + G1/GADR 분해** 병기. GT = (verb,noun) strict. 항상 **L0(WM top-1) 병기**.
- **개입(인과·유용)**: `harden_s3.py` — belief/reasoning swap vs paraphrase(문체 통제).
  belief는 **G-CC1(민감도)∧G-CC3(방향, U_g belief-only)** 쌍으로만 보고.
- **history**: paired strip Δacc (WM 후보 고정, H-bin 층화).
- **게이트**: G-ACC1(CE>L0) · G-NH(SFT 비열등) · 개입 U_g. G-DELTA(후보제시>자유생성)는 보완 대상(§6).
- **CI**: 최종은 **video-cluster bootstrap**(같은 비디오 프레임 상관 → sample-CI 과소분산).

---

## 5. 주장 가능 / 불가 (실측 경계)

**주장 가능**:
- "candidate-CE는 WM top-1 모방을 넘는 선택·판별 엔진을 만든다" (G-ACC1, GADR)
- "projected-trace SFT는 belief→action 인과를 심으며, 개입으로 검증된다" (U_g, swap vs para)
- "두 단계는 다른 능력(정확도/인과추론)을 분업한다" (분업 지도)

**주장 불가**:
- "SFT가 정확도를 개선한다" (G-NH FAIL)
- "egocentric reasoning" (자매 1인칭 침식 반증 — SFT 화법 이관 전)
- "causal mediation" (interventional dependence까지만)
- 두 코호트 수치 직접 비교 (방향·구조만)

---

## 6. 확정 전 보완 2건 (권장 순서)

1. **G-DELTA 본셋 실증** (~2.5h) — cand_free arm 학습 후 full-set + 모드 명기로 "후보제시>자유생성".
   현재 SKIP(자매 대체치 +2.4pp full / +19.2pp covered만). 후보-제시 정당화의 본셋 공백.
2. **full 학습** — N=2,500 → full. §Method 구조 불변, 수치만 갱신. 자매 추산: full +3~8pp vs strict
   pre-onset 전환 손실 −5~10pp 상쇄 → **의의는 수치가 아니라 분해(coverage 돈줄·GADR·DiD)에**.
3. (선택) **egocentric 화법 SFT 이관** — projected-SFT 화법 타깃 명시 후 H-bin별 1인칭율 회복 검증
   (embodied 주장을 살리려면).

---

## 7. 방법론 참조 지도 — 기존 handoff 어디에 있나

### 확정 방법론의 근거 체인 (EGO_jihun3)

| 문서 | 무엇 | 상태 |
|---|---|---|
| **2026-07-25_cesft_v2_paper_methodology_final_handoff.md** | ← **이 문서. §Method SSOT(확정)** | 확정 |
| 2026-07-25_cesft_v2_quantitative_evidence_handoff.md | 능력별 정량 지표 전량 (실측 근거) | 실측 |
| 2026-07-24_dpo_free_ce_sft_methodology_handoff.md | CE→SFT 방법론 원안 (사용자 결정) | 계획 |
| 2026-07-24_ce_sft_methodology_v2_handoff.md | 방법론 수식·게이트 상세(v2) | 계획 |
| 2026-07-24_s3_pivot_plan_handoff.md | S3 인과성(개입) spine 전환 결정 | 결정 |
| 2026-07-24_wm_boundary_precheck_results_handoff.md | WM 경계 precheck (acc 헤드룸 진단) | 실측 |
| 2026-07-24_cesft_v2_running_state_handoff.md | cesft_v2 실행 상태·OOM 수정 | 운영 |
| 2026-07-24_cesft_v2_time_oom_optimization_handoff.md | 시간·OOM 최적화 (cgroup RAM gate) | 운영 |

### 평가 지표 정의 (EGO_jihun3)

| 문서 | 무엇 |
|---|---|
| 2026-07-24_evaluation_metrics_handoff.md | SelAcc/G1/GADR/L0/full-set 환산 규약 |
| 2026-07-24_interventional_belief_sensitivity_metric_handoff.md | belief 개입(swap vs para)·G-CC1∧CC3 정의 |
| 2026-07-24_history_usage_verified_handoff.md | history strip 인과 검증 |

### 방법론 정당화·문헌·자매 실험 (EGO_jihun)

| 문서 | 무엇 |
|---|---|
| 2026-07-24_embodied_reasoning_methodology_justification_handoff.md | embodied reasoning 방법론 정당화 |
| 2026-07-24_ce_sft_combination_literature_handoff.md | CE+SFT 조합 선행연구(continual-replay·순서) |
| 2026-07-24_reasoning_quality_quantitative_evidence_handoff.md | **자매 실험** 정량 지표 (goalstep_v3_boundary 코호트) |

### 초기 설계 계보 (참고)

| 문서 | 무엇 |
|---|---|
| 2026-07-22_nonparametric_prospection_projected_trace_retrospection_handoff.md | projected-trace retrospection 최초 설계 |

### 논문 소스

- `EGO_paper/EGO_AAAI27_EN/main.tex` (영문) · `EGO_paper/EGO_AAAI27/main.tex` (국문)
- 반영 위치: §Next-Action Selection(full+covered+GADR 병기) · §Ablation(history strip) ·
  §Reasoning and Belief Analysis(개입 U_g)

---

## 8. 근거 파일 좌표

| 무엇 | 위치 |
|---|---|
| arm 별 평가 | `runs/cesft_v2/eval/{base,theta_ce,sft_r15}.{json,records.jsonl}` |
| 게이트 | `eval/paired_{G-ACC1,G-DELTA,G-NH}_*.json` · `eval/strip_verdict.json` |
| 개입 (spine) | `eval/sft_r15.harden_s3.json` (verdict "PASS — spine 확정 (U_g)") |
| WM prior | `RETRO4-goalstep-end-m1-history-k8-phase1/best_action_top5.pt` |
| 학습 스크립트 | `scripts/step2_retrospection/` (select_ce · sft_v2 · harden_s3) |
