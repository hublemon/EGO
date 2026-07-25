# cesft_v2 실행 상태 handoff — Predictive-Boundary Selection → Projected Retrospection

- 작성: 2026-07-24 KST · EGO_jihun3 · `runs/cesft_v2`
- **성격**: 현재 *실행 중인* 런의 운영·방법론 handoff (스냅샷). SSH가 끊겨도 지속되도록 구성됨.
- **방법론 SSOT**: [`2026-07-24_ce_sft_methodology_v2_handoff.md`](2026-07-24_ce_sft_methodology_v2_handoff.md) — 수식·게이트 상세는 그 문서가 정본. 본 문서는 그 위에 **이번 런의 결정·변경·상태**를 얹는다.
- **리포트 아티팩트**: https://claude.ai/code/artifact/9310c32d-a120-4bef-9250-71be16ca5b98

---

## 0. 한 줄 요약

WM(jihun2 Phase-1)이 만든 Top-10 후보 경계 안에서 VLM(Qwen3-VL-8B)이 **top-1 모방을 넘어 task-conditioned 선택**을 하도록 2단계 지도학습(DPO 없음)하고, 생성된 belief가 행동에 **유용하게 관여**하는지를 개입(intervention)으로 측정한다.

---

## 1. 현재 상태 (2026-07-24 ~09:55 KST 스냅샷)

| 항목 | 값 |
|---|---|
| 현재 스테이지 | **E1 · θ_CE (wm_cand, selection CE)** 학습 중 |
| 진행 | ~3.8% (step 19 / total 4189 samples), ETA ~4h |
| 완료 마커 | 없음 (E0 앵커만 측정 완료) |
| 프로세스 | supervisor pid 7364 (PPID=1) → orchestrator 7387 → select_ce 7392 |
| RAM | cur ~58G / peak 96G / limit 240G — **안정** |
| GPU | 단일 GPU, MAX_PARALLEL=1 |

---

## 2. 이번 세션의 결정·변경 (중요 — v2 문서에 없는 델타)

### 2.1 OOM-kill 수정 (240GiB cgroup)
- **원인**: 이전 실행(08:12)에서 orchestrator가 MAX_PARALLEL=2로 비디오 arm(theta_ce+cand_free)을 동시 기동 → 각자의 decord VideoReader 스파이크 합산이 208G→240G cgroup 한도 초과 → SIGTERM(rc=143). `psutil`은 호스트(2TB)를 봐서 무력했음.
- **수정** (durable):
  - `tools/parallel_orchestrator.py`: **MAX_PARALLEL 기본 2→1**, `cgroup_ram_free_gb()`(=`memory.max−current`) 기반 **RAM admission gate `RAM_FLOOR_GB=100`** 추가 (GPU gate와 병렬).
  - `scripts/step2_retrospection/{cesft_v2_parallel,start_cesft_v2}.sh`: 동일 기본값 반영.
- **한계**: admission control은 *기동 시점*만 보호. 실행 중 arm의 런타임 스파이크는 **MAX_PARALLEL=1(직렬)**로 격리하는 게 결정적 방어. → 관련 메모리: `cesft-video-arm-oom-cgroup`.

### 2.2 baseline arm 축소 — random_cand · no_video 제거
- **유지**: `cand_free`, `no_history`만. **제거**: `random_cand`, `no_video` (train/eval/gate 모두).
- **근거**: WM prior 후보는 `context_train.jsonl`에 **이미 셔플되어 저장**됨(확인: `gt_rank`≠candidates 내 위치, 예 rank3→pos7). 즉 위치/객관식(multiple-choice) confound가 `wm_cand` 안에서 이미 통제되므로 `random_cand`를 별도 arm으로 둘 필요 없음. `no_video`는 이번 스코프에서 제외.
- 반영: `tools/parallel_orchestrator.py` GPU_TASKS, `runs/cesft_v2/chain.json`(대시보드), serial `cesft_v2_chain.sh`.

### 2.3 확인된 사실 (질문 대응)
- **jihun2 Phase-1 모델 사용 = YES, 단 WM prior로서**. 후보 생성기 = `RETRO4-goalstep-end-m1-history-k8-phase1/best_action_top5.pt` (SHA 검증 읽기전용 export, cov@10 43.9%). 학습되는 VLM은 별개(Qwen3-VL-8B-Instruct, LoRA).
- **VLM이 비디오 prefix를 실제로 봄 = YES**. 8초 관측창의 8프레임(336px)을 image content로 삽입 → processor → model forward (vision-grounded). `select_ce.py:81, :102-106`.

---

## 3. 방법론 (요약 — 상세는 v2 SSOT)

**Stage 1 — θ_CE (Predictive-Boundary Selection)**
```
s(a)   = (1/|a|) Σ log π(aⱼ | c, a<ⱼ)          # length-norm 후보 점수
p(a|D) = softmax over D ( s(a)/τ )              # 후보집합 정규화
L_sel  = − log p(a_GT | c, D)
```
후보 K=10, 후보 셔플, WM rank/prob 비공개, covered 샘플만 학습.

**Stage 2 — θ_CE + SFT (Projected Retrospection + CE Replay)**
```
L = λ_g·L_g + λ_r·L_r + λ_a·L_sel   (λ_g=1.0, λ_r=0.5, λ_a=0.25, field 내부평균)
batch ~ B_proj (1−ρ) | B_CE (ρ),    ρ ∈ {0, 0.15, 0.30}  (배치 혼합 방식 A)
```
θ_CE에서 초기화. 순서 CE→SFT(+replay) 고정. `CE→SFT→full-CE` 금지.

**성립 3조건**: ① wm_cand > cand_free 그리고 > WM-top1 ② SFT가 belief sensitivity ∧ belief-only utility 둘 다 ③ 비손상.

---

## 4. 설정 · 하이퍼파라미터

| | 값 |
|---|---|
| config | `configs/step2_retrospection/cesft_v2.yaml` |
| 학생 VLM | `Qwen/Qwen3-VL-8B-Instruct` + LoRA (r=16, α=32, dropout 0.05, q/k/v/o_proj) |
| WM prior | jihun2 Phase-1 HistoryContextResidualHead, Top-10, cov@10 43.9% |
| 후보 K / 관측창 | 10 / ≤8s 가변 (평균 gap 12.8s, strict-next A3) |
| 프레임 | 8 frames · 짧은변 336px · 1fps |
| optimizer | AdamW lr=1e-5, cosine, accum=8, epochs=1, τ=1.0, seed=42 |
| probe | probe_every=100 step |
| 데이터 | `context_train.jsonl` ∩ (gt_rank≤10) ∩ train_subset → ~4189 covered |
| 평가 | EVAL_N=1000 (battery), IV_N=800 (harden_s3 개입) |
| SFT | ρ∈{0,.15,.30}, BEST_R=sft_r15, 부록A CSTACK_STEPS=150 |
| adapter 출력 | `outputs/step2_retrospection/cesft_v2/<run_name>/adapter` |

---

## 5. 파이프라인 순서 · 게이트 — **core=sft_r15 우선 재배치 (2026-07-24)**

DAG runner: `parallel_orchestrator.py`의 `GPU_TASKS`. marker 멱등(완료 skip).
MAX_PARALLEL=1이라 **리스트 순서 = 실행 우선순위**. 재배치는 결과 불변, headline 도달 시점만 앞당김(~22h→~8h).

**Phase A — core spine (fail-fast, ~8h)**
1. **E0** base 공정 앵커 — 완료 (VLM 0.213 / WM 0.244 / fusion 0.223)
2. **θ_CE** 학습 → battery → **G-ACC1** `SelAcc(θ_CE) − WM-top1` CI 하한>0 *(생사)*
3. **sft_r15** (ρ=0.15, 기본 조합; θ_CE init) → battery → harden_s3(**belief-only U_g**) + **G-NH**
   → ~8h에 헤드라인 확정 (분기 8.1/8.2/8.3/8.4)

**Phase B — 귀속 ablation (core 성공 시, ~6h)**
4. **cand_free** 학습 → battery → **성립부등식** `wm_cand > cand_free` (성립조건 1)
5. **no_history** 학습 → battery (G-ACC2 보조)

**Phase C — ρ ablation/강건성 (~10h)**
6. **sft_r0**(replay 없음 ablation) · **sft_r30**(G-NH fallback) + 각 battery/harden
7. **WiSE-FT** α∈{.25,.5,.75} frontier (학습 0)
8. **부록A** P-UTIL 게이트 통과 시 C-stack/C-ctrl equal-budget → T-ACC (harden_sft_r15 후 동적 추가)
9. **리포트** 아티팩트 확정

**필수 게이트**: G-ACC1(생사), 성립부등식(wm>cand_free), **G-CC3 belief-only U_g CI>0(필수)**, G-NH(비손상).

**분기(4-outcome)**: 8.1 강한성공 / 8.2 GADR·hard-case 헤드라인 / 8.3 sensitivity-only(embodied reasoning 제목 곤란) / 8.4 실패(방법 재검). 상세 v2 SSOT §8.

---

## 6. 앵커 (판정 기준선)

**E0 base — 이번 run 실측** (n=1000 heldout, covered):
- SelAcc@10 0.200 · WM-top1 0.242 (beats_L0=false) · GADR 0.164 (n=758) · G1 0.314 (n=242) · cov@10 0.434 · malformed 0.011
- → base의 acc 기여 ≈ 0. **E1 θ_CE가 WM-top1(0.242)을 넘어야 생존.**

**belief 개입 — 이전 세대 실측** (retro3, base vs 학습완료 r1_sft, n≈990):
- belief sensitivity 0.058 → 0.390 (6.7×↑, G-CC1 강함)
- utility(own−swap_both) 0.108 → 0.067 **하락** — 그러나 **belief-only(own−swap_b) 0.023 → 0.042 상승**
- → v2가 utility 지표를 **belief-only U_g로 승격**한 근거. θ_CE+SFT 완료 후 재측정 필수.

---

## 7. 운영 (모니터 · 재기동 · 정지)

**기동** (setsid, PPID=1 재부모화 — SSH 끊겨도 지속, 자동 재시작):
```bash
MAX_PARALLEL=1 RAM_FLOOR_GB=100 bash scripts/step2_retrospection/start_cesft_v2.sh
```
**모니터**:
```bash
tail -f runs/cesft_v2/logs/chain.log        # orchestrator/스테이지
tail -f runs/cesft_v2/logs/mem.log          # cgroup RAM (cur가 240G 근처면 위험)
cat runs/cesft_v2/status/S_CE_*.json        # 스테이지별 진행률
python3 tools/retro3_dashboard.py           # :7867 대시보드
```
**정지** (supervisor 먼저 — 안 그러면 자동 재기동):
```bash
kill $(cat runs/cesft_v2/chain.pid)         # supervisor
pkill -f parallel_orchestrator.py
pkill -f 'ego.step2_retrospection.train'
```
**재개(resume)**: marker 기반. 완료 스테이지는 skip. 단, **select_ce/sft_v2는 중간 체크포인트 없음**(끝에서 1회 save) → 학습 중 죽으면 해당 arm은 step 0부터 재시작. GPU가 <30000MiB로 비어야 preflight 통과.

---

## 8. 알려진 리스크 · 열린 항목

- **중간 체크포인트 부재**: 긴 arm(θ_CE ~4h)이 후반에 죽으면 전량 재학습. 필요 시 select_ce에 주기 저장 추가 검토.
- **admission gate는 기동만 보호**: MAX_PARALLEL을 2 이상으로 올릴 경우 실행 중 스파이크 재발 가능 — 비디오 arm은 1 유지 권장.
- **θ_CE ETA 변동**: 초반 워밍업으로 rate가 흔들림(0.19~0.21/s). 안정화 후 재추정.
- **아티팩트의 "학습 후" 정성/정량은 이전 세대(retro3/retro4) 실측** — v2 θ_CE·SFT 완료 시 동일 URL로 갱신 필요.
- **다음 결정 지점**: E1 종료 시 G-ACC1 판정(생사). 통과 시 E1b 성립부등식 → SFT로 진행, 실패 시 v2 §8.3/8.4 분기 검토.

---

## 관련
- 방법론 정본: [[2026-07-24_ce_sft_methodology_v2_handoff]]
- OOM 인프라 교훈: 메모리 `cesft-video-arm-oom-cgroup`
