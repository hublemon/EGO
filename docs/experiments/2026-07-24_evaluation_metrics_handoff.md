# cesft_v2 평가지표 종합 handoff — 정확도 · 게이트 · belief 개입

- 작성: 2026-07-24 KST · EGO_jihun3
- 코드: `eval/battery.py`(정확도), `tools/paired_boot.py`(게이트 CI), `eval/harden_s3.py`(belief 개입), `tools/precheck_fair_fusion.py`(fusion)
- 관련: [[2026-07-24_interventional_belief_sensitivity_metric_handoff]](belief 심화), [[2026-07-24_ce_sft_methodology_v2_handoff]] §2·§4, [[2026-07-24_cesft_v2_running_state_handoff]]
- **모든 판정은 같은 샘플의 paired 차이 + bootstrap 95% CI 하한**으로 한다.

---

## 0. 지표 가족 3개 (한눈에)

| 가족 | 무엇 | 코드 | bootstrap 단위 |
|---|---|---|---|
| **A. 정확도/선택** | 모델이 정답 행동을 고르나 | `battery.py` | — (point) |
| **B. 게이트 Δ** | arm 간 차이가 유의한가 | `paired_boot.py` | **video-cluster** |
| **C. belief 개입** | belief가 행동에 쓰이나/유용한가 | `harden_s3.py` | sample |

---

## 1. 평가 세트 규약 (공통)

- **covered-only**: GT∈Top-10(`gt_rank≤10`) 샘플만 평가. uncovered는 후보 매칭 구조상 **전 arm 강제 0점**이라 비교력만 희석 → 제외(2026-07-22 사용자 확정).
- **K=10 고정**: Coverage/SelAcc/GADR/G1/WM-top1/fusion 전부 **동일 K=10 후보셋**에서 계산. (Phase-1 "K=8"은 visual history 길이 — 별개 축.)
- **full-set 환산**: covered 지표에 `pool_coverage`(=split 전체의 cov@10, ≈0.434)를 곱해 full-set 등가 병기.
  `acc_full_equiv = acc × pool_coverage`.

---

## 2. 가족 A — 정확도/선택 지표 (`battery.py`)

생성 평가: 모델이 `<action>verb noun</action>`을 생성 → 후보에 매칭 → GT와 비교.

| 지표 | 정의 (코드) | 의미 |
|---|---|---|
| **SelAcc@10** (`acc`) | `mean(correct)` | 후보 중 정답을 고른 비율 (핵심 정확도) |
| **WM-top1** (`L0_wm_top1`) | `mean(wm_top1_correct)` | WM `argmax(wm_scores)`의 정확도 = **모방 상한/바닥선** |
| `beats_L0` | `acc > L0` | 모델이 WM 모방을 넘었나 (bool) |
| **G1 retention** | `acc | wm_top1 정답` 부분집합 | WM이 이미 맞힌 걸 **유지**하나 |
| **GADR** (`G2_correction`) | `acc | (GT∈support ∧ wm_top1 오답)` | WM이 **틀린 hard-case를 교정**하나 (gain beyond imitation) |
| `coverage_at_k` | `mean(gt_in_support)` | GT가 후보 K개 안에 있나 (covered셋≈1.0) |
| `pool_coverage` | split 전체 `gt_rank≤10` 비율 | 후보 풀 커버리지 (≈0.434) |
| `malformed_rate` | 포맷 깨진 출력 비율 | 생성 품질 |
| `acc_full_equiv` | `acc × pool_coverage` | full-set 등가 정확도 |

**G1/G2 분해가 핵심**: SelAcc는 두 체제의 혼합이다 —
G1(WM 정답 유지) + GADR(WM 오답 교정). 평균 SelAcc가 WM에 못 미쳐도 **GADR이 살아있으면**
"hard-case 교정" 스토리(분기 8.2)가 가능. 그래서 SelAcc 하나가 아니라 **G1·GADR을 분리 보고**한다.

**fusion** (`precheck_fair_fusion.py`): VLM prob와 WM prob를 결합한 최종 선택 정확도. G-ACC3용.
단 fusion은 WM prob를 직접 쓰므로 "rank/prob 비공개" 설계와 충돌 → **보조 분석**으로 규정.

---

## 3. 가족 B — 게이트 Δ (`paired_boot.py`, **video-cluster** bootstrap)

**왜 cluster인가**: 한 영상의 연속 프레임은 상관됨 → per-sample bootstrap은 분산 과소평가.
resampling 단위를 **video_uid(클러스터)**로 하고, 두 arm을 **같은 리샘플셋**에서 채점 → Δ가 제대로 paired.

지표(cluster 위): `SelAcc=mean(correct)` · `GADR=mean(correct|wm_top1 오답)` · `WMtop1=mean(wm_top1_correct)` · `G1`.

| 게이트 | Δ 정의 | 통과 조건 | 무엇 |
|---|---|---|---|
| **G-ACC1** (생사) | `SelAcc(arm) − WMtop1(arm)` | `Δ.lo > 0` | θ_CE가 모방 초과 |
| **성립부등식** (G-DELTA) | `SelAcc(arm_a) − SelAcc(arm_b)` | `Δ.lo > 0` | wm_cand > cand_free (WM 경계 가치) |
| **G-ACC2** (G-DELTA/GADR) | `GADR(θ_CE) − GADR(base)` | `Δ.lo > 0` | 학습 효과 (+ > no_history) |
| **G-ACC3** | `acc(fusion) − acc(WM)` | `Δ.lo > 0` | fusion이 정보 추가 |
| **G-NH** (비손상) | A=학습후, B=학습전 | `[SelAcc(A)−SelAcc(B)].lo ≥ −0.01` **∧** `[GADR(A)−GADR(B)].point ≥ −0.02` | SFT가 판별 안 망침 |

CLI: `paired_boot.py --run runs/cesft_v2 --arm_a <A> [--arm_b <B>] --gate {G-ACC1,G-DELTA,G-NH} [--metric SelAcc|GADR|G1]`.

---

## 4. 가족 C — belief 개입 지표 (`harden_s3.py`) — 요약

*(측정 절차·swap variant·paraphrase 통제의 상세는 [[2026-07-24_interventional_belief_sensitivity_metric_handoff]] 참조)*

prefix `(reasoning, belief)`를 강제로 바꿔(open/empty/swap_b/swap_r/swap_both/para) 행동 분포 변화를 잰다.

| 지표 | 정의 | 게이트 |
|---|---|---|
| **belief sensitivity** | `flip(swap_b) − flip(para)` | **G-CC1** `.lo>0` (필수) |
| reasoning / both sensitivity | `flip(swap_r/both) − flip(para)` | G-S3a(both) |
| **belief-only utility U_g** | `p(a_GT|own) − p(a_GT|swap_b)` | **G-CC3** `.lo>0` (필수) |
| directional D_g | `Pr[p_gt(own) > p_gt(swap_b)]` | G-CC4 |
| correct-switch | belief-swap로 top1 바뀐 샘플의 평균 GT-확률 하락 | 보조 |

⚠️ **sensitivity ≠ utility**: 민감도가 높아도(반응함) 방향이 GT로 맞는지는 U_g가 따로 판정.
**useful belief dependence = G-CC1 ∧ G-CC3.**
⚠️ **레거시 함정**: 구 utility `own−swap_both`(reasoning까지 swap)는 오염됨 → 반드시 `utility_belief_only_ci`(U_g) 사용.
⚠️ **용어**: interventional만, "causal 매개/경로" 금지(§9).
※ harden의 CI는 **sample** bootstrap(`diff_ci`) — 최종 판정은 게이트(B)의 cluster CI로 재확인 권장.

---

## 5. 앵커 (판정 기준선)

**A. E0 base — 이번 run 실측** (n=1000, covered):
`SelAcc 0.200 · WM-top1 0.242(beats_L0=false) · GADR 0.164(n=758) · G1 0.314(n=242) · cov@10 0.434 · malformed 0.011`
→ base acc 기여 ≈ 0. **θ_CE가 WM-top1(0.242)을 넘어야 생존.**

**C. belief 개입 — 이전 세대 retro3** (base vs 학습완료, n≈990):
`sensitivity(belief) 0.058→0.390` · `U_g(belief-only) 0.023→0.042` (구 own−swap_both는 0.108→0.067 하락 — 오염).

---

## 6. 게이트 → 분기 매핑 (§8 4-outcome)

| 분기 | 조건 | 헤드라인 |
|---|---|---|
| **8.1 강한 성공** | G-ACC1·G-ACC2·G-CC1·U_g·G-NH 전부 PASS | full claim (모방 초과 + useful belief) |
| **8.2 제한적 성공** | G-ACC1 FAIL·G-ACC2 PASS·G-CC1·U_g·G-NH PASS | **GADR/hard-case** 교정 중심 |
| **8.3 sensitivity-only** | G-ACC1·G-ACC2 FAIL·G-CC1 PASS·U_g FAIL | belief-prefix 민감도만 — embodied reasoning 제목 곤란 |
| **8.4 실패** | CE 판별 실패 ∨ belief 의존 실패 ∨ non-harm 위반 | 2단계 방법 재검 |

---

## 7. cesft_v2에서 언제/어디서 나오나

| Phase | 산출 게이트 | 파일 |
|---|---|---|
| A · θ_CE eval | **G-ACC1** (생사) | `eval/theta_ce.json` + `eval/paired_G-ACC1_theta_ce.json` |
| A · sft_r15 eval+harden | **G-CC1·G-CC3(U_g)·G-NH** (헤드라인) | `sft_r15.json`, `sft_r15.harden_s3.json`, `paired_G-NH_sft_r15_vs_theta_ce.json` |
| B · cand_free eval | **성립부등식** | `paired_G-DELTA_theta_ce_vs_cand_free.json` |
| B · no_history eval | G-ACC2 보조 | `no_history.json` |
| C · WiSE/부록A | frontier · T-ACC · P-UTIL | `wise_ft_frontier.json`, `paired_TACC_*.json` |

---

## 8. 8h 헤드라인 시점 체크리스트

sft_r15까지 끝나면 이 숫자들을 순서대로 본다:
1. `theta_ce.json`: `acc` > `L0_wm_top1`(0.242)? → G-ACC1 방향
2. `paired_G-ACC1_theta_ce.json`: `delta.lo > 0`? → **생사 확정**
3. `sft_r15.harden_s3.json`: `causal_sensitivity_ci.belief.lo > 0`? → G-CC1
4. 같은 파일 `utility_belief_only_ci.lo > 0`? → **G-CC3(필수)**
5. `paired_G-NH_sft_r15_vs_theta_ce.json`: SelAcc.lo ≥ −0.01 ∧ GADR.point ≥ −0.02? → G-NH
6. → §6 표로 분기(8.1~8.4) 판정.

논문 문장(§9): *"We test whether action selection is interventionally dependent on the semantic content of the generated task belief."*
