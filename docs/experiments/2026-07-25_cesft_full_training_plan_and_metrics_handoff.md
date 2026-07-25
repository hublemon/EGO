# cesft full 학습 계획 · 평가 지표 총정리 Handoff (3-arm 확정판)

> 작성: 2026-07-25 KST · EGO_jihun3. **방법론은 [[2026-07-25_cesft_v2_paper_methodology_final_handoff]]로 확정.
> 이 문서는 그 방법론을 full 데이터로 1회 완주하는 실행 계획 + 논문에 들어갈 평가 지표의 SSOT.**
> 사용자 확정 3건 (2026-07-25):
> 1. **ablation 종료** — 학습 arm은 base / stage1 / stage1+stage2 **3개뿐**.
> 2. 평가 항목은 EGO_jihun `2026-07-24_results_section_update_evidence_map_handoff.md`에서 **항목만** 차용(그 문서의 실측값은 다른 코호트이므로 전량 폐기, 이번 런의 수치로 채운다).
> 3. **신규 정량 지표 2종 추가** — 1인칭 사용률 · 추론 간결화.
> 운영 요구: **OOM 금지 · 서버 강제종료 금지**(§5·§6이 그 설계).

---

## 0. 결론 요약

| 항목 | 내용 |
|---|---|
| 학습 arm | **base(무학습) · θ_CE(stage 1) · θ_CE+SFT ρ=0.15(stage 1+2)** — 그 외 전부 제거 |
| 데이터 | 파일럿 6k 서브샘플(비디오당 30 캡) → **full 전량**. CE 4,189→**20,985** · SFT 2,945→**~14,700** |
| 신규 run dir | `runs/cesft_full` (파일럿 `runs/cesft_v2`는 **동결·보존**, frame_cache만 공유 재사용) |
| GPU 예산 | **약 53h** (실측 rate 기반, §3-6) · CPU 프레임 추출 ~2h는 GPU와 겹침 |
| 학습 내구성 | **중간 체크포인트/resume 구현·검증 완료**(2026-07-25) — kill 시 최대 손실 0.7h (§6-3, §6-5) |
| 프레임 추출 | **전량 오프라인 사전 추출** — 학습·평가 경로의 on-the-fly 디코드 0회 (§3-2) |
| OOM 방어 | 프레임 디스크 캐시 100% 선행 + hard-RAM admission + 단일 GPU 잡 + decord 가드 (§5) |
| 강제종료 방어 | setsid 분리 · supervisor 자동 resume · **스톨 워치독** · @reboot 재기동 · 단계 마커 멱등 (§6) |

---

## 1. 확정 범위 — 무엇을 하고 무엇을 안 하나

### 1.1 하는 것 (3-arm)

| arm | 정체 | 산출 어댑터 |
|---|---|---|
| **base** | 학습 0 (Qwen3-VL-8B-Instruct) | — |
| **θ_CE** | stage 1 = candidate-CE (arm=wm_cand, τ=1.0, lr 1e-5, accum 8, 1 epoch) | `outputs/step2_retrospection/cesft_full/theta_ce/adapter` |
| **θ_CE+SFT** | stage 2 = projected-trace SFT (init=θ_CE, **ρ=0.15 고정**, lr 5e-5, accum 8, 1 epoch) | `.../cesft_full/sft_r15/adapter` |

하이퍼·손실·게이트 정의는 방법론 handoff §1.2~§1.6에서 **한 글자도 바꾸지 않는다**. full 런에서 바뀌는 것은 **표본 수뿐**이다.

### 1.2 안 하는 것 (전부 삭제, 재제안 금지)

| 제거 | 사유 | 논문 영향 |
|---|---|---|
| `cand_free` 학습 arm | ablation 종료 | **G-DELTA(후보제시>자유생성) same-pipeline 측정 불가** → 자매 실험(JIHUN full +2.4pp / covered +19.2pp) 인용 + "본 파이프라인 미측정" 각주로 처리 |
| `no_history` 학습 arm | 동상 | history 인과는 **추론-시 strip**(학습 0, §4-B)이 더 강한 paired 근거로 대체 |
| ρ 스윕 (r0 / r30) | 동상 | "CE replay가 필요하다"는 주장은 못 함 → **ρ=0.15는 설계 선택으로 서술**, 효과 주장 금지 |
| WiSE-FT α frontier | 동상 | §Method에서 삭제 |
| 부록A C-stack / C-ctrl | 동상 | 삭제 |
| DPO 전 경로 | 방법론 확정(DPO-free) | 이미 §Method에서 배제 |
| VPA · closed-loop planning | step-3 별건 | 이 계획 범위 밖. 단 evidence-map §3의 **split 불일치(v2 heldout ↔ VPA val-134)** 는 미해소로 남음 → 논문 각주 필요 |

> **판단 1건 (플래그)**: history-strip과 belief/reasoning 개입(harden_s3)은 *학습 arm을 추가하지 않는 추론-시 개입*이라 "ablation 종료"에 걸리지 않는다고 보고 **유지**했다. 두 지표가 논문 §Ablation·§Reasoning and Belief Analysis의 유일한 인과 근거라 빼면 인과 주장이 전부 사라진다. 개입까지 포함해 없애길 원하면 §4-B·§4-C를 삭제하고 예산 −6.2h.

---

## 1.3 계승 선언 — 이 학습은 EGO_jihun3 cesft core 조합 학습의 스케일업이다

새 파이프라인이 아니다. **`runs/cesft_v2`에서 완주한 candidate-CE → projected-trace SFT(+CE replay) 조합 학습을 코드·계약 그대로 두고 표본 수만 full로 올린 것**이다.

| 축 | 계승 여부 | 실체 |
|---|---|---|
| 학습 코드 | **그대로** | `train/select_ce.py`(arm=wm_cand) · `train/sft_v2.py`(micro-step 인터리브) |
| 손실 | **그대로** | CE: length-norm 후보 span logp softmax, τ=1.0 · SFT: field 가중(belief 1.0/reasoning 0.5/action 0.25) |
| 조합 구조 | **그대로** | θ_CE → SFT warm-start(`--init_adapter`) + CE replay ρ=0.15 (micro-step 교대, 합산손실 아님) |
| 하이퍼 | **그대로** | LoRA r16/α32/dropout.05/[q,k,v,o] · AdamW · clip 1.0 · cosine · accum 8 · epochs 1 · CE lr 1e-5 / SFT lr 5e-5 |
| 시간 계약 | **그대로** | retro4: 관측 [max(A2.start, A2.end−9s), A2.end−1s] ≤8s, target = strict-next A3 |
| 관측 형식 | **그대로** | **8프레임 @336** (변경 금지 — 계약의 구성 요소) |
| WM prior | **그대로** | jihun2 Phase-1 K8 `RETRO4-…-phase1/best_action_top5.pt` (읽기전용, sha 고정) |
| 프롬프트 | **그대로** | `vlm.SYSTEM_PROMPT` · `RETRO_NEXT_GAP_TEXT` |
| teacher/projection/게이트 | **그대로** | Ψ→Φ→`quality_gate.check_chosen`, drop-not-patch |
| config | **그대로** | `configs/step2_retrospection/cesft_v2.yaml` |
| **바뀌는 것** | — | **① 표본 수(서브셋→full) ② run dir(`runs/cesft_full`) ③ 중간 체크포인트/resume ④ 평가 셋 확대·신규 언어 지표** |

**가중치는 계승하지 않는다 (중요)**: θ_CE는 **base에서 fresh**로 다시 학습한다. 파일럿 θ_CE(서브셋 4,189 학습본)를 warm-start로 쓰면 그 4,189개를 두 번 보게 되어 "full 1 epoch"이 아니게 되고, 파일럿과 full 사이에 샘플별 노출 횟수가 불균등해져 게이트 해석이 오염된다.
**데이터는 계승한다**: 파일럿이 이미 생성한 `chosen_train.jsonl` 4,189행(teacher 생성 트레이스)은 그대로 이어받고 나머지 ~16.8k만 증분 생성한다 — 같은 Ψ/Φ/게이트로 만든 결정적 산출물이라 재생성해도 동일하며, 8.9h를 아낀다.

---

## 2. 데이터 스케일 — 파일럿 → full (전량 실측)

| 층위 | 파일럿(cesft_v2) | **full** | 비율 |
|---|---:|---:|---|
| `context_train.jsonl` 전체 | 6,000 (subset) | **29,293** | 20.5% |
| CE 학습 풀 = covered(`gt_rank≤10`) | **4,189** | **20,985** | 20.0% |
| SFT 학습 풀 = gate `pass` | **2,945** | **~14,700** (pass율 70.3% 가정) | 20.0% |
| heldout 평가 (`context_val` split=heldout) | 1,000 샘플링 | **5,326 전량** (covered 2,313, cov@10 43.4%) | — |
| 학습/평가 video-disjoint | ✓ | **✓ 검증** (train 564 vid ∩ val 128 vid = **0**) | — |

파일럿 서브샘플의 형태: `make_subset.py --per_video 30 --n 6000 --seed 42` → 564개 비디오 중 550개는 남고(97.5%) **비디오당 깊이만 1/5**(평균 51.9→10.9). 즉 도메인 편향이 아니라 **표본 수 부족**이었고, full은 그 5배로 CI가 좁아진다 — 특히 지금 `+1.1pp [−5.7, +7.1]`로 "비유의"인 **G-NH의 판정이 뒤집힐 여지**가 있다(방향은 어느 쪽이든 논문 서사에 반영).

### 2-1. full 데이터셋 구성 전량 (실측)

**학습 소스 — `runs/cesft_full/data/context_train.jsonl` : 29,293 샘플 / 564 비디오 (split=train 단일)**

| 풀 | 정의 | 개수 | 비고 |
|---|---:|---:|---|
| 전체 | Ego4D GoalStep train 비디오의 모든 결정 시점 | **29,293** | 후보는 전 샘플 K=10 고정 |
| **CE 학습 풀** | `gt_rank ≤ 10` (= GT∈후보, covered) | **20,985 (71.6%)** | θ_CE가 실제로 보는 전량 |
| 제외 | uncovered (GT가 WM Top-10 밖) | 8,308 (28.4%) | 선택 CE가 정의되지 않음 — 구조적 제외 |
| **SFT 학습 풀** | CE 풀 중 hindsight 게이트 `pass` | **≈14,700** (파일럿 pass율 70.3% 적용) | 상한 20,985 |
| **CE-replay 풀** | CE 학습 풀과 동일(복원추출) | 20,985 | ρ=0.15 → CE micro-step ≈ 2,600회 |
| 미사용 | `base_trace_train.jsonl` | 6,000 | DPO 경로 잔재 — full에서 생성하지 않음 |

**평가 소스 — `context_val.jsonl` : 6,960 샘플 / 128 비디오 (train과 video-disjoint, 교집합 0 검증)**

| split | 전체 | covered(cov@10) | 용도 |
|---|---:|---:|---|
| **heldout** | **5,326** | **2,313 (43.4%)** | **논문의 모든 수치**. battery·strip 전량, harden 800·freegen 500은 이 안의 부분집합 |
| dev | 1,634 | 744 (45.5%) | 프로브·개발 전용 — **논문 수치 산출 금지** |

**history 길이 분포** (두 셋 모두 H8이 지배): train H8 25,600(87.4%) / H0 151 · heldout 계열 H8 6,083(87.4%) / H0 38. → strip의 H-bin 층화에서 H0(플라시보)은 표본이 얇으므로 **합성 H0**(H8의 history를 지운 조건)을 보조로 병기한다.

**⚠ WM coverage 비대칭 — 논문 각주 필요**: train cov@10 **71.6%** vs heldout **43.4%**. WM prior(jihun2 Phase-1)가 train 비디오로 학습되어 **in-sample recall이 부풀려진** 것이다(train gt_rank=1이 전체의 32.7%, heldout은 10.4%). 학습은 covered만 쓰므로 절차상 문제는 없지만, ① 학습 분포가 "WM이 이미 맞히는" 쉬운 샘플 쪽으로 기울어 있고 ② 따라서 **학습 coverage를 논문의 coverage 수치로 쓰면 안 된다**(논문 수치는 heldout 43.4%). 진단으로 `gt_rank` 구간별 GADR를 병기한다.
**단, 이 편향은 파일럿과 동일하다** — 서브셋의 covered율 4,189/6,000 = 69.8% ≈ 전체 71.6%. 즉 파일럿→full 사이 **분포 이동은 없고 규모만 5배**다.

---

## 3. 실행 계획

### 3-0. 원칙

1. **파일럿 산출물 불변**: `runs/cesft_v2/**`, `outputs/.../cesft_v2/**` 는 읽기만. 새 런은 `runs/cesft_full`.
2. **프레임은 디스크 캐시만 사용**(저장소 규약). 학습·평가 경로의 on-the-fly 디코드는 0회여야 하며, 캐시 커버리지 게이트(§3-2)를 통과하지 못하면 착수하지 않는다.
3. **모든 단계는 마커 멱등** — 재기동 시 완료 단계 skip, 진행 단계는 sample_id resume.
4. GPU 잡은 **항상 1개** (MAX_PARALLEL=1). CPU 추출기는 hard-RAM 게이트 하에서만 동시 실행.

### 3-1. P0 — 준비 (CPU, ~0.5h)

```bash
REPO=/mnt/nvme/migration/jihun/EGO_jihun3; cd $REPO
mkdir -p runs/cesft_full/{data,eval,logs,markers,status}
# 데이터: context는 그대로 재사용(동일 계약), chosen은 파일럿 4,189행을 이어받아 증분 생성
cp runs/cesft_v2/data/context_train.jsonl runs/cesft_v2/data/context_val.jsonl runs/cesft_full/data/
cp runs/cesft_v2/data/chosen_train.jsonl runs/cesft_full/data/     # projection이 sample_id resume
# 프레임 캐시는 공유 (재추출 금지) — 환경변수로 지정
export FRAME_CACHE_DIR=$REPO/runs/cesft_v2/frame_cache
```
`train_subset.json`은 **복사하지 않는다** (= full 풀 신호). §7의 코드 변경 후 `--subset_file none`으로도 동일.

스모크: CE 3샘플 · SFT 3샘플 · projection 8샘플 · battery 8샘플을 `--limit`로 1회씩 통과시키고 마커를 지운다.

### 3-2. P1 — 프레임 캐시 완성 = **오프라인 사전 추출** (CPU, 6 shards, ~2.1h)

> **질문 답: 이 계획의 프레임 추출은 100% 오프라인이다.** 학습·평가가 시작되기 전에 26,311 샘플 ×8프레임@336을
> JPEG로 디스크에 굽고(§3-2), 학습 루프는 **이미지 파일만 읽는다**. `vlm.extract_frames`에 on-the-fly 디코드
> 폴백(`vlm.py:146` 캐시-히트 분기)이 남아 있지만, full 런에서는 커버리지 게이트로 **폴백 발생 0**을 강제한다.
> (파일럿 θ_CE는 on-the-fly였고 그게 ±25G RAM 진동·reshape 크래시·7h 스톨의 원인이었다 — 반복하지 않는다.)

대상 = CE 풀 20,985 ∪ heldout 5,326 = **26,311** (기존 ok 6,284 → **~20,027 신규**). 실측 rate 2.3 s/샘플(단일 워커) → 6 shard 병렬 ~2.1h. 디스크 ~10GB (여유 27T).

```bash
for i in 0 1 2 3 4 5; do
  FRAME_RAM_FLOOR_GB=80 RETRO3_RUNS=runs/cesft_full \
  setsid nohup $PY tools/oom_opt/frame_extractor.py --pool both --shard $i --shards 6 \
    >> runs/cesft_full/logs/frame_extract_$i.log 2>&1 & disown
done
```
`--shard/--shards`는 §7-4 신규 인자(video_uid 해시로 분할 — 같은 VideoReader를 두 워커가 열지 않게).

**게이트(필수)**: `manifest.jsonl`의 ok 집합 ⊇ (CE 풀 ∪ heldout). 미스 0이 아니면 P2 착수 금지.
```bash
$PY tools/oom_opt/check_cache_coverage.py --run runs/cesft_full --cache $FRAME_CACHE_DIR   # §7-5 신규
```

### 3-3. P2 — projected trace 전량 생성 (GPU, ~8.9h)

```bash
RETRO3_RUNS=runs/cesft_full $PY -m ego.step2_retrospection.hindsight.projection \
  --config configs/step2_retrospection/cesft_v2.yaml --batch_size 32     # --subset 미지정 = full
```
- Ψ(teacher, 텍스트 192tok) → Φ(projection, 8프레임 320tok) → **규칙 게이트**(`quality_gate.check_chosen`) → `chosen_train.jsonl`.
- 파일럿 4,189행은 done_ids로 skip → 실제 신규 ~16,800 × ~1.9s.
- 게이트 정책 **drop-not-patch 유지**. 유의어 패러프레이즈 누출(2026-07-25 보류 결정)은 **이번에도 손대지 않는다** — 한계로 §논문에 명시만.
- 산출 검증: pass율이 파일럿 70.3% ±5pp 밖이면 정지하고 원인 조사(분포 이동 신호).

### 3-4. P3 — θ_CE full 학습 (GPU, ~18.8h)

```bash
RETRO3_RUNS=runs/cesft_full $PY -m ego.step2_retrospection.train.select_ce \
  --config configs/step2_retrospection/cesft_v2.yaml --run_name theta_ce \
  --arm wm_cand --tau 1.0 --epochs 1 --subset_file none --ckpt_every 100 --resume auto
```
20,985 샘플 × 3.23 s/샘플(실측) = 18.8h · opt.step ≈ 2,623(accum 8). `--ckpt_every/--resume`은 §6-3 신규.

### 3-5. P4 — projected-trace SFT full (GPU, ~12.8h)

```bash
RETRO3_RUNS=runs/cesft_full $PY -m ego.step2_retrospection.train.sft_v2 \
  --config configs/step2_retrospection/cesft_v2.yaml --run_name sft_r15 \
  --init_adapter outputs/step2_retrospection/cesft_full/theta_ce/adapter \
  --ce_replay_rho 0.15 --ce_tau 1.0 --epochs 1 --subset_file none --ckpt_every 100 --resume auto
```
~14,700 × 3.14 s/샘플 = 12.8h. CE-replay 풀도 **full covered**(20,985)로 확장된다(파일럿은 subset).

### 3-6. P5 — 평가 (GPU, ~11.5h) · P6 — 집계 (CPU)

| 단계 | 대상 | 명령 | 예산 |
|---|---|---|---|
| battery ×3 arm | heldout 5,326 전량 | `eval.battery --arm {base,theta_ce,sft_r15} --eval_n 5326` | 3×1.8h = **5.3h** |
| strip ×3 arm | 동일 셋, history만 제거 | `tools/oom_opt/strip_eval.py --adapter … --eval_n 5326` | 3×1.8h = **5.3h**(covered 2,313로 줄이면 2.3h) |
| harden ×3 arm | IV_N=800 | `eval.harden_s3 --arm … --n 800` | 3×0.3h = **0.9h** |
| freegen 화법 ×3 arm ×2 레짐 | n=500 | `eval.freegen --mode {presented,cand_free}` (§7-6 신규) | 6×0.17h = **1.0h** |
| 게이트·텍스트지표·정성로그 | CPU | `tools/paired_boot.py`, `tools/trace_text_metrics.py`(신규) | ~0 |

### 3-7. 전체 예상 시간 (파일럿 실측 rate 기반)

근거 rate(모두 `runs/cesft_v2/status/*.json` 실측): CE **3.23 s/샘플**(4,189→3.76h) · SFT **3.14 s/샘플**(2,945→2.57h) · battery **1.22 s/샘플**(1,000→0.34h) · strip 1.37 s/샘플 · harden **1.36 s/샘플**(396→0.15h) · 프레임 추출 **2.3 s/샘플**(단일 워커).

| # | 단계 | 자원 | 물량 | 낙관 | **기준** | 비관 |
|---|---|---|---:|---:|---:|---:|
| P0 | 준비·스모크 | CPU+GPU | — | 0.3h | **0.5h** | 1h |
| P1 | **오프라인 프레임 추출** (6 shards) | CPU | 20,027 신규 | 1.7h | **2.1h** | 3.5h |
| P2 | projection Ψ→Φ→gate | GPU | 16,796 신규 | 7.5h | **8.9h** | 12h |
| P3 | **θ_CE full** | GPU | 20,985 | 17.5h | **18.8h** | 22h |
| P4 | **SFT r15 full** | GPU | ~14,700 | 11.5h | **12.8h** | 15h |
| P5a | battery ×3 arm | GPU | 5,326×3 | 4.8h | **5.3h** | 6.5h |
| P5b | strip ×3 arm | GPU | 5,326×3 | 4.8h | **5.3h** | 6.5h |
| P5c | harden ×3 arm | GPU | 800×3 | 0.8h | **0.9h** | 1.2h |
| P5d | freegen 화법 ×3×2 | GPU | 500×6 | 0.9h | **1.0h** | 1.4h |
| P6 | 게이트·텍스트지표·정성·아티팩트 | CPU | — | 0.2h | **0.3h** | 0.5h |
| | **GPU 합** | | | 47.8h | **53.0h** | 64.6h |
| | **벽시계 합** (P1은 GPU와 겹치지 않게 선행) | | | 49.8h | **55.9h** | 69.6h |

→ **기준 약 56시간 = 2일 8시간.** 무인 연속 운전 전제(§6), 사람 개입은 게이트 판정 보고 시점뿐.
불확실성 큰 항목은 **P2 projection**(1.9 s/샘플은 battery 1.22 + teacher 텍스트 생성 추정치라 실측 아님) — P2 첫 30분의 `status/S3_hindsight.json:sec_per_sample`으로 즉시 재추정한다.
단축 레버: strip을 covered 2,313으로 줄이면 −3.0h, freegen을 제시 레짐만 하면 −0.5h. **P3/P4는 줄이지 않는다**(full 1 epoch 주장의 본체).

**총 GPU ≈ 53h · 벽시계 ≈ 56h.**

### 3-8. 파일럿 자산 재사용으로 이미 줄인 시간 / 더 줄일 수 있는 것

이 예산표는 **파일럿(`runs/cesft_v2`) 산출물을 최대한 재사용한 뒤의 숫자**다. 재사용을 안 했다면 다음이 더 붙는다:

| 재사용 자산 | 물량 | 아낀 시간 | 상태 |
|---|---:|---:|---|
| `chosen_train.jsonl` (Ψ→Φ→gate 완료분) | 4,189행 | **−2.2h** | ✅ 계획 반영 (P2가 16,796만 생성) |
| `frame_cache` JPEG | 6,284 샘플 | **−4.0h** (CPU) | ✅ 계획 반영 (P1이 20,027만 추출) |
| `context_{train,val}.jsonl` + WM export | 36,253행 | **−수 h** (S1 조인·WM 추론 전량) | ✅ 그대로 사용, 재빌드 없음 |
| `base.records.jsonl` | heldout 1,000 | −0.34h | ⚠ **권장 안 함** — 파일럿은 raw decode, full은 JPEG 캐시라 입력이 미세하게 다르다. 같은 표 안에서 두 조건을 섞느니 재생성 |
| 파일럿 θ_CE 어댑터 warm-start 후 **나머지 16,796만** 학습 | — | −3.8h | ❌ **권장 안 함** — 사유 아래 |
| 파일럿 `sft_r15` 어댑터 | — | — | ❌ **불가** — 파일럿 θ_CE에서 갈라져 나온 가중치라 full θ_CE와 계보가 끊긴다 |

**θ_CE warm-start를 왜 안 쓰나** (−3.8h, 전체의 7%): ① 파일럿은 **optimizer/scheduler state를 저장하지 않았다** — AdamW 모멘트를 0에서 다시 쌓으면 이어붙인 학습이 아니다. ② 서브셋 4,189를 먼저 다 보고 나머지 16,796을 보는 **순서 편향**이 생긴다(현 설계는 video-group 셔플로 전 구간을 고르게 통과). ③ cosine LR이 두 사이클로 쪼개진다. ④ 파일럿 프레임은 raw decode, 나머지는 JPEG 캐시라 **입력 조건이 학습 도중 바뀐다**. 7% 아끼자고 "full 1 epoch을 균질하게 1회 통과" 주장을 흐리는 거래는 손해다.

**데이터 재사용 외의 단축 레버**:
- ❌ **gradient checkpointing 해제 — 실측 기각(2026-07-25)**. H200 143GB면 여유가 있을 줄 알고 40샘플 프로브를 돌렸으나 **40/40 전부 `skip_oom`**(steps=0). CE 스텝은 *후보 10개 × 8프레임@336을 한 배치로* forward 하므로 활성값이 checkpointing 없이는 143GB를 넘는다. → **gradient checkpointing은 선택이 아니라 필수 조건**이다. (부수 확인: OOM 가드가 런을 죽이지 않고 샘플 스킵으로 흡수하는 것도 이때 실증됐다.)
- ✅ 평가 축소: strip을 covered 2,313로 −3.0h, freegen 제시 레짐만 −0.5h.
- **P3/P4의 표본 수는 줄이지 않는다** — full 1 epoch 주장의 본체.

**결론: 데이터 재사용으로 이미 ~6.2h(+ S1 조인 수 h)를 뺐고, 남은 53h는 안전하게 더 줄일 여지가 크지 않다.** 추가 단축은 (a) 평가 축소 −3.5h, (b) 방법론을 흐리는 warm-start −3.8h 뿐이며 (b)는 권장하지 않는다.

> 축소 레버(기본 아님): 시간 압박 시 `make_subset --per_video 60 --n 12000`으로 절반 스케일 → GPU ~28h. 단 "full 1 epoch" 주장이 사라지고 파일럿과 같은 성격의 캡 표본이 된다. **권장하지 않음.**

---

## 4. 평가 지표 총정리 (논문 §Results SSOT)

측정 모집단: **heldout n=5,326** (video-disjoint, cov@10 43.4%, covered 2,313). 개입류는 그 부분집합.
CI 규약: **video-cluster bootstrap**(같은 비디오 프레임 상관 → sample-CI는 과소분산). 5,000 resample, seed 123.
모든 지표는 **3 arm(base / θ_CE / θ_CE+SFT) 전부**에서 측정한다.

### A. 선택 정확도 — Table reasoning (battery)

| # | 지표 | 정의 | 산출 |
|---|---|---|---|
| A1 | **full acc** | heldout 전체 정확도 (uncovered는 구조적 0) | `{arm}.json:acc` |
| A2 | **SelAcc (covered)** | GT∈Top-10 부분집합 정확도 = 선택 능력 | records에서 `gt_in_support` 필터 |
| A3 | **L0 (WM top-1)** | WM prior top-1 추종 기준선 — **모든 표에 강제 병기** | `L0_wm_top1` |
| A4 | **Coverage@10** | GT∈후보 비율 (SelAcc→full 환산 계수) | `pool_coverage` |
| A5 | **G1 retention** | WM top-1이 정답일 때 유지율 | `G1_retention` |
| A6 | **GADR (G2 correction)** | GT∈후보 ∧ WM top-1≠GT일 때 교정률 — **모방 전략은 정의상 0점** | `G2_correction` |
| A7 | **malformed rate** | 형식 파탄/후보 매칭 실패 | `malformed_rate` |
| A8 | beats_L0 | A1 > A3 여부 | `beats_L0` |

**게이트**: **G-ACC1** = SelAcc(θ_CE) − L0, paired CI 하한 > 0 · **G-NH** = SelAcc(θ_CE+SFT) − SelAcc(θ_CE) 비열등.
`tools/paired_boot.py --gate {G-ACC1,G-NH} --arm_a … --arm_b …`

### B. history 인과 — Table ablation (strip, 학습 0)

| # | 지표 | 정의 |
|---|---|---|
| B1 | **Δacc (paired)** | acc(history 있음) − acc(history 제거), **WM 후보 고정**·같은 체크포인트 |
| B2 | **H-bin 용량-반응** | H0(플라시보, Δ≈0이어야 함) · H1–3 · H4–7 · H8 별 Δ |
| B3 | 무-history vs L0 | strip 조건에서 acc < L0 이면 "LM 우위는 history에서 온다"의 대우 |

`strip_verdict.json`. **H0 Δ≈0 실패 시 나머지 해석 보류**(건전성 검사).

### C. belief · reasoning 인과 — Table intervention (harden_s3, IV_N=800)

| # | 지표 | 정의 |
|---|---|---|
| C1 | flip율 | swap_b / swap_r / swap_both 각각의 top-1 변경률 |
| C2 | **paraphrase 대조** | 뜻 보존 재서술의 flip률 = 문체 노이즈 바닥 |
| C3 | **causal sensitivity** | flip(swap_X) − flip(para), paired CI |
| C4 | **U_g (belief-only utility)** | p_gt(own) − p_gt(swap_b) — **G-CC3** |
| C5 | G-CC1 belief sensitivity | belief swap 민감도 CI 하한 > 0 |
| C6 | directional D_g | Pr[p_gt(own) > p_gt(swap_b)] |
| C7 | correct-switch | belief-swap로 flip된 샘플의 평균 GT확률 하락 |
| C8 | acc 직교성 | flip이 정답/오답 샘플에서 다른가 |

용어 규율: **"interventional dependence"까지만.** "causal mediation" 금지.

### D. 트레이스 언어 지표 — Table trace_metrics (**신규 정량화**, CPU)

`tools/trace_text_metrics.py`(신규)가 records의 `reasoning`/`task_belief`를 재계산. 전부 video-cluster bootstrap CI 병기.

| # | 지표 | 정의 | 비고 |
|---|---|---|---|
| D1 | **1인칭 사용률** ★신규 | `\b(I\|my\|me\|I'm\|I've)\b` 검출률, reasoning·belief 각각 | **§4-D-1 필독** |
| D2 | **추론 간결화** ★신규 | reasoning 평균·중앙 단어수, 문장 수, base 대비 압축률 `1 − len(arm)/len(base)` | belief 길이도 병기 |
| D3 | 소거 서술률 | 선택 외 후보를 비교·배제 언급한 비율 (비교추론 프록시) | |
| D4 | 후보 거명 수 | reasoning이 언급한 후보 개수 `n_mentioned` | |
| D5 | 장면-묘사율 | 관찰 서술 문장 비율 | evidence-map 항목 |
| D6 | 미래-지향율 | 다음 행동을 향한 서술 비율 | evidence-map 항목 |
| D7 | 인과 연결어율 | `since/because/having just…` 패턴 | |
| D8 | in_support | **비제시 생성**이 WM Top-10 안에 안착한 비율 (경계 내재화) | freegen 전용 |

#### 4-D-1. 1인칭 지표는 현행 템플릿에서 측정 불가 — freegen 패스가 필요하다 (실측 확인)

파일럿 records에서 직접 계산한 결과:

| arm | 1인칭(reasoning) | 1인칭(belief) | reasoning 평균 단어 |
|---|---:|---:|---:|
| base | **0.0%** | 0.0% | 69.3 |
| θ_CE | **0.0%** | 0.0% | 57.7 |
| θ_CE+SFT | **0.0%** | 0.0% | 80.6 |

원인은 학습이 아니라 **프롬프트**다 — `vlm.SYSTEM_PROMPT`가 *"a list of actions **the person** already COMPLETED"* 로 3인칭 관찰자 프레임을 못박아 전 arm이 3인칭으로 쓴다. 이 템플릿 위에서 1인칭율은 항상 0이라 **arm을 구분하지 못한다.**

→ **해결: 화법-중립 freegen 평가 패스 신설**(§7-6). 학습 프롬프트는 **절대 건드리지 않고**, 평가에서만 화자 중립 SYSTEM(“the person” 프레이밍 제거, 1인칭 강제·금지 어느 쪽도 하지 않음)으로 n=500 생성. 두 레짐 모두:
- **presented**(후보 제시 = 배포 레짐) — 자매 실험에서 CE가 1인칭을 *강화*한 조건(52.4→61.4, +8.9pp)
- **cand_free**(후보 비제시 = 비교군) — 자매 실험에서 *침식*이 관찰된 조건(74.0→31.6/21.2/7.4)

**정직성 규칙(강제)**: ⑴ 1인칭율은 **템플릿 종속**이므로 동일-템플릿·동일-레짐 내 arm 비교만 유효. ⑵ SFT 학습 타깃(Φ 트레이스)이 3인칭이므로 **SFT가 1인칭을 가르치지 않는다** — 결과가 어떻든 "화법을 학습했다"가 아니라 "타깃 문체를 모방했다"로 서술. ⑶ "egocentric reasoning" 주장은 **후보-제시 레짐 근거로만**, 그마저 조건부. ⑷ 침식/강화의 일반화 금지(레짐-종속, 2026-07-24 정정).

D8(in_support)도 이 freegen 패스에서 함께 나온다 — 후보를 안 보여줬을 때 WM 경계 안에 들어오는가.

### E. 정성 근거 — Table trace

base 오답 → θ_CE+SFT 정답으로 교정된 케이스 중 (GT∈후보 ∧ WM top-1≠GT)인 **GADR 실물** 3–4건 자동 추출(`tools/pick_trace_examples.py`, records 조인). 시각 단서 문구와 belief 변화를 나란히 제시.

### F. 산출 파일 좌표 (전부 `runs/cesft_full/eval/`)

```
{base,theta_ce,sft_r15}.json / .records.jsonl              # A
paired_G-ACC1_theta_ce.json · paired_G-NH_sft_r15_vs_theta_ce.json
strip_{base,theta_ce,sft_r15}.json · strip_verdict.json     # B
{arm}.harden_s3.json / .harden_s3.records.json              # C
freegen_{arm}_{presented,cand_free}.records.jsonl           # D
text_metrics.json                                           # D 집계
trace_examples.md                                           # E
```

---

## 5. OOM 방지 설계 (5중)

사고 이력: 2026-07-24 동시-arm 208G→SIGTERM · decord 진동 ±25G · `memory.current` 오판으로 추출기 자기교착(19:57~20:23) · reshape 디코드 크래시.

| # | 방어 | 구현 | 성격 |
|---|---|---|---|
| 1 | **프레임 디스크 캐시 100%** | P1에서 26,311 전량 선추출 + 커버리지 게이트. 학습·평가 경로 decord **0회** | **근본** — 진동·디코드 크래시가 원천 소멸 |
| 2 | **단일 GPU 잡** | `MAX_PARALLEL=1`, `MIN_FREE_MB=60000` preflight (GPU<30GB 대기) | 근본 (240G peak 주범 제거) |
| 3 | **hard-RAM admission** | `RAM_FLOOR_GB=100`. 판정은 `anon+unevictable+slab_unreclaimable+dirty+writeback` — **`memory.current` 금지**(page cache 포함, JPEG 쓰기만으로 한도에 붙어 교착) | 근본 (오판 제거) |
| 4 | **decord 가드** | 캐시 미스 폴백 경로 한정: workers 2, prefetch chunk 4, 프레임 개수(짝수)·크기 검증 후 불합격은 **샘플 스킵** | 완화 (폴백에서만 유효) |
| 5 | **CUDA/학습 설정** | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` · **gradient checkpointing 필수**(끄면 40/40 GPU OOM — §3-8 실측) · batch 32·accum 8 유지 | 유지 |

경보: `tools/oom_opt/ram_alarm.sh`(hard ≥ 225G 2틱) 상시 기동. **관측 전용 — 아무것도 죽이지 않는다.**
디스크: frame_cache +10GB, records/adapters 수 GB → 27T 여유로 무관. 단 `df` < 500GB면 체인 정지 게이트 추가.

---

## 6. 강제종료·중단 방지 설계

### 6-1. 방어 4중

| # | 실패 모드 | 방어 |
|---|---|---|
| 1 | SSH/VS Code 세션 종료 | `setsid nohup … & disown` + PPID=1 재부모화 확인 (`start_cesft_full.sh`) |
| 2 | 프로세스 크래시 | `supervisor.sh` 자동 재시작 + 마커 멱등 resume. 같은 지점 MAX_RETRY=5 연속 실패 → `CHAIN_STUCK`(사람 호출) |
| 3 | **스톨(굳음)** | **일반화 스톨 워치독**(§6-2) — 2026-07-24에 status 무갱신 7시간을 놀린 실패 모드. supervisor는 죽으면 살리지만 **멈추면 아무것도 안 한다** |
| 4 | 컨테이너 재시작 | `@reboot` cron으로 `start_cesft_full.sh` 등록(중복 기동 거부 내장) + GPU preflight 대기 |

### 6-2. 스톨 워치독 (기존 serverA 전용 → 전 단계 일반화)

`runs/cesft_full/status/*.json`의 `updated_at`을 60s 폴링. `state=running`인데 **STALL_SEC=1800s** 무갱신이면 해당 단계 프로세스만 kill → supervisor가 마커 resume으로 재기동. 최대 `MAX_RESTART=3`, 초과 시 `CHAIN_STUCK`. (정상 rate: CE 0.31 샘플/s, SFT 0.32 샘플/s — 매 초 갱신된다.)

### 6-3. 학습 중간 체크포인트 — **구현 완료 (2026-07-25)**

> 이 절의 "리스크"는 해소됐다. 아래 스펙 그대로 `train/ckpt.py`(신규) + `select_ce.py`/`sft_v2.py` 수정으로 들어갔고,
> 60샘플 스모크로 **저장 → 강제 kill → resume**을 실제 검증했다(§6-5).

현재 `select_ce.py:316` / `sft_v2.py:197`은 **끝날 때 한 번만** `save_pretrained` 한다. full 스케일에서 이건 치명적이다:

| 런 | 길이 | 지금 죽으면 | 필요 |
|---|---|---|---|
| θ_CE full | 18.8h | **0에서 재시작** | 주기 체크포인트 |
| SFT full | 12.8h | **0에서 재시작** | 동상 |

**요구 스펙** (두 파일 동일):
- `--ckpt_every N`(opt.step 기준, 기본 100 ≈ 0.7h): `adapter_ckpt/`(LoRA ~40MB) + `train_state.json{step, n_seen, consumed_sample_ids_hash, epoch}` + `opt.state_dict()` + `sched.state_dict()`를 **원자적 교체**(tmp→rename)로 저장.
- `--resume auto`: `train_state.json`이 있으면 LoRA·optimizer·scheduler 복원 후 **소비한 샘플 수만큼 스킵**. 데이터 순서는 `seed=42` + video-group 정렬로 결정적이라 재현된다.
- 저장 비용 ~2s → 총 오버헤드 1분 미만.
- resume 정합성 로그: 재개 시 `step/lr/n_seen`을 `train_log.jsonl`에 남겨 학습곡선이 이어지는지 눈으로 확인.

### 6-3-1. 구현 실체

| 파일 | 내용 |
|---|---|
| `src/ego/step2_retrospection/train/ckpt.py` (신규) | `save_ckpt`(원자적 tmp→ckpt→old 교체) · `load_ckpt`(LoRA+opt+sched 복원) · `restore_rng` · `clear_ckpt` |
| `train/select_ce.py` | `--ckpt_every`(기본 100) · `--resume auto|off` · `--subset_file` · `--no_grad_ckpt`. 스트림 슬라이싱(`sample_stream(epoch, start_pos)`)으로 **이미 본 구간은 프레임 로딩조차 하지 않고** 건너뜀 |
| `train/sft_v2.py` | 동일 인자. 큐가 `pop()` 소비라 `sft_queue[: len−n_sft]` 슬라이스로 정확히 이어붙음 |

저장 내용: LoRA adapter + AdamW state + cosine scheduler state + `random.Random` state + 스트림 위치(`stream_pos`/`n_sft`) + ema. 크기 176MB, 저장 ~2s. accum 버퍼는 이어받지 않는다(최대 7샘플 gradient 손실 — 무시 가능).

### 6-5. 검증 기록 (2026-07-25, 실제 실행)

θ_CE 경로 60샘플 스모크(`--accum 2 --ckpt_every 2`):
1. 학습 중 `ckpt/`에 주기 저장 확인 → `train_state.json`: `step 22 / stream_pos 44`
2. **강제 kill**(SIGTERM) → 체크포인트 온전, `step 26 / stream_pos 52`로 남음
3. 같은 명령 재실행 → 로그에 `[ckpt] resume from step=26 n_seen=52 stream_pos=52`
4. 남은 8샘플만 처리하고 `steps=30 / seen=60`으로 정상 완료 → `adapter` 저장 + `ckpt/` 자동 정리

즉 **kill → 재기동 시 최대 손실은 `ckpt_every`(=100 opt-step ≈ 0.7h) 이하**다. 스모크 산출물은 삭제했고 파일럿 마커·status는 오염되지 않았다.

### 6-4. resume 매트릭스 (수정 후 최종 상태)

| 단계 | resume 단위 | 최대 손실 |
|---|---|---|
| frame cache | sample_id (manifest) | 1 샘플 |
| projection | sample_id (chosen_train.jsonl append) | 1 chunk(32) |
| θ_CE / SFT | **opt.step 100** (신규) | ~0.7h |
| battery / strip / freegen | sample_id (records append) | 1 chunk |
| harden_s3 | 단계 재실행 | 0.3h |

---

## 7. 필요한 코드 변경 (착수 전, 총 6건)

| # | 파일 | 변경 | 상태 |
|---|---|---|---|
| 1 | `train/select_ce.py` | `--subset_file` (기본 `train_subset.json`, `none`=full 풀) | **✅ 완료 (2026-07-25)** |
| 2 | `train/sft_v2.py` | 동일 (CE-replay 풀에도 적용) | **✅ 완료** |
| 3 | `train/ckpt.py`(신규) + 두 트레이너 | **`--ckpt_every` / `--resume auto`** (§6-3) | **✅ 완료 · kill→resume 실검증(§6-5)** |
| 3b | `train/select_ce.py` | `--no_grad_ckpt` (속도 레버, 수치 불변) | ✅ 완료 (SFT 쪽은 채택 시 동일 추가) |
| 4 | `tools/oom_opt/frame_extractor.py` | `--shard i --shards N` (video_uid 해시 분할) | P1 2.1h로 단축 |
| 5 | `tools/oom_opt/check_cache_coverage.py` (신규) | 학습·평가 풀 ⊆ manifest(ok) 검증, 미스 리스트 출력 | OOM 방어 1의 게이트 |
| 6 | `eval/freegen.py` (신규) · `tools/trace_text_metrics.py` (신규) · `tools/pick_trace_examples.py` (신규) | §4-D·§4-E 지표 산출 | 신규 지표 2종 |

기존 학습·평가 손실/프롬프트/하이퍼는 **불변**. 변경은 전부 진입점·내구성·측정 도구에 한정된다.

---

## 8. 실패 모드 런북

| 증상 | 판정 | 조치 |
|---|---|---|
| `ram_alarm.log`에 ALARM 연속 | hard RAM 225G↑ | GPU 잡 유지, **추출기 shard 수를 6→2로** 감축. 그래도 오르면 추출 일시정지 |
| status 무갱신 30분 | 스톨 | 워치독이 자동 처리. 3회 초과 시 `CHAIN_STUCK` → 로그의 마지막 sample_id로 원인 샘플 특정 |
| `CHAIN_FAILED` 반복 같은 지점 | 코드/데이터 문제 | supervisor가 5회 후 정지. 해당 sample_id를 `skip_ids.json`에 추가하고 재기동 |
| projection pass율 <65% 또는 >76% | 분포 이동 | 정지. gate 사유 분포(restatement/future_leak/…)를 파일럿과 비교 |
| θ_CE SelAcc < L0 | G-ACC1 실패 | **논문 서사 재검토 사안** — 즉시 보고, 후속 단계 진행 보류 |
| GPU 접근 불가 | 드라이버/컨테이너 | supervisor가 `CHAIN_STUCK` 남기고 정지 (자동 재시도 안 함) |

---

## 9. 논문 매핑

| 논문 자리 | 이 계획의 지표 |
|---|---|
| §Next-Action Selection (Table reasoning) | A1–A8, 3 arm + L0 병기, G-ACC1 |
| §Ablation (history) | B1–B3 (strip, 학습 0) |
| §Reasoning and Belief Analysis | C1–C8 (개입), U_g·G-CC1∧CC3 |
| §Trace Quality (Table trace_metrics) | D1–D8 — **1인칭은 freegen 레짐 분리 필수** |
| §Qualitative (Table trace) | E |
| §Limitations | G-DELTA 본셋 미측정 · 유의어 게이트 누출 · VPA split 불일치 · 1인칭 템플릿 종속 |

**주장 금지선 (방법론 handoff §5 계승 + 2026-07-25 갱신)**
- "SFT가 정확도를 개선한다" — G-NH가 full에서 PASS로 바뀌기 전에는 금지.
- "causal mediation" — interventional dependence까지만.
- "CE가 egocentric 화법을 죽인다/살린다" — **레짐-종속**이므로 조건 명시 없는 일반화 금지.
- "CE replay가 필요하다" — ρ 스윕을 안 하므로 효과 주장 불가(설계 선택으로만 서술).
- 두 코호트(cesft ↔ JIHUN) 수치 직접 비교 — 방향·구조만.

---

## 10. 부록 — 기동 스크립트 (신규, 아직 미설치)

### A. `scripts/step2_retrospection/start_cesft_full.sh`

```bash
#!/usr/bin/env bash
# cesft full 체인 기동 — setsid 분리, 중복 기동 거부, @reboot 재사용 가능.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$REPO"
RUNS=runs/cesft_full; mkdir -p "$RUNS"/{logs,markers,status}
if [[ -f "$RUNS/chain.pid" ]] && kill -0 "$(cat "$RUNS/chain.pid")" 2>/dev/null; then
  echo "[start] 이미 실행 중 — 거부"; exit 0; fi
rm -f "$RUNS/markers/CHAIN_FAILED" "$RUNS/markers/CHAIN_STUCK"
export FRAME_CACHE_DIR="$REPO/runs/cesft_v2/frame_cache"
RETRO3_RUNS="$RUNS" CHAIN_SCRIPT="scripts/step2_retrospection/cesft_full_chain.sh" \
MAX_PARALLEL=1 MIN_FREE_MB=60000 RAM_FLOOR_GB=100 MAX_RETRY=5 \
  setsid nohup bash scripts/step2_retrospection/supervisor.sh \
  >> "$RUNS/logs/chain_stdout.log" 2>&1 < /dev/null & 
echo $! > "$RUNS/chain.pid"; disown -a
setsid nohup bash tools/oom_opt/ram_alarm.sh >> "$RUNS/logs/ram_alarm.log" 2>&1 < /dev/null & disown
setsid nohup bash scripts/step2_retrospection/stall_watchdog.sh >> "$RUNS/logs/stall.log" 2>&1 < /dev/null & disown
sleep 2; ps -eo pid,ppid,stat,cmd | grep -E "supervisor.sh|cesft_full" | grep -v grep || true
```
crontab: `@reboot /bin/bash /mnt/nvme/migration/jihun/EGO_jihun3/scripts/step2_retrospection/start_cesft_full.sh`

### B. `scripts/step2_retrospection/cesft_full_chain.sh` (스테이지 골격)

```
preflight(GPU<30GB) 후 각 단계 run_stage(마커 멱등):
 F1  frame_cache 커버리지 게이트          CACHE_OK
 S3  projection full                      S3_HINDSIGHT_DONE
 E1  select_ce theta_ce (--subset_file none --ckpt_every 100 --resume auto)   S_CE_THETA_CE_DONE
 E1e battery theta_ce (eval_n 5326)       S7_EVAL_THETA_CE_DONE  → gate G-ACC1
 E0e battery base                         S7_EVAL_BASE_DONE
 E2  sft_v2 sft_r15 (init=theta_ce, ρ.15) S6_SFT_R15_DONE
 E2e battery sft_r15                      S7_EVAL_SFT_R15_DONE   → gate G-NH
 IV  harden_s3 ×3 (n=800)                 S3H_{arm}_DONE
 ST  strip_eval ×3                        S_STRIP_{arm}_DONE
 FG  freegen ×3 ×2 레짐 (n=500)           S_FREEGEN_{arm}_{mode}_DONE
 TX  trace_text_metrics + pick_examples   S_TEXT_METRICS_DONE
 RP  artifact 재굽기                       CESFT_FULL_CHAIN_DONE (+ RETRO3_CHAIN_DONE 정지신호)
```

### C. `scripts/step2_retrospection/stall_watchdog.sh`

`runs/cesft_full/status/*.json` 전수 폴링(60s) → `state=running` ∧ `now−updated_at ≥ 1800` 이면 해당 stage의 프로세스만 `pkill -f`(stage명 매칭) → 30s 후 `-9` → supervisor 재기동에 맡김. `MAX_RESTART=3` 초과 시 `markers/CHAIN_STUCK` 기록 후 종료.

---

## 11. 착수 체크리스트

- [x] ~~#1·#2 `--subset_file`~~ · ~~#3 체크포인트/resume(kill→resume 실검증)~~ — **2026-07-25 완료**
- [ ] §7의 잔여 코드 변경 3건 (#4 shard · #5 캐시 커버리지 게이트 · #6 freegen·텍스트지표·정성추출)
- [ ] `runs/cesft_full` 생성 + context/chosen 복사 + `train_subset.json` 미복사 확인
- [ ] 스모크 4종(CE·SFT·projection·battery) 통과 후 마커 삭제
- [ ] P1 프레임 캐시 26,311 전량 + **커버리지 게이트 미스 0**
- [ ] `ram_alarm.sh` · `stall_watchdog.sh` 기동 확인, `@reboot` cron 등록
- [ ] 파일럿(`runs/cesft_v2`) 읽기전용 확인 — 덮어쓰기 0
- [ ] 첫 1h 후 mem.log의 hard 추이·status rate 점검(3.2 s/샘플 근처인지)
