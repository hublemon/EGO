# EGO Step-2 — DPO-free 방법론 (CE → projected-SFT) 및 acc·인과제어 지표 Handoff

- 작성: 2026-07-24 KST · EGO_jihun3
- 결정(사용자): **DPO 완전 배제.** Step-2 = ① candidate-CE(정확도 엔진) → ② projected-trace SFT
  (reasoning·belief 인과적 행동 제어). VLM은 영상 prefix를 본다. WM prior = **EGO_jihun2 Phase-1 K8**.
- 선행 SSOT: `2026-07-24_wm_boundary_precheck_results_handoff.md` · `2026-07-24_s3_pivot_plan_handoff.md`
- 논문: `EGO_paper/EGO_초안_영문.pdf` ("The Limits of the World Model Mean the Limits of the LM")

---

## 0. 요지

- **왜 DPO 배제**: Retrospection(논문 eq 11 DPO)은 두 번 붕괴(문체학습 G3-abort ×5 → margin +70 collapse).
  faithfulness는 DPO 없이 **projected-trace SFT만으로 이미 달성**(belief-채널 0.058→0.390). DPO는 불필요.
- **역할 분리**: **CE = acc 엔진**(GT 판별) · **SFT = 인과 제어**(belief→action 인과 경로). 둘은 다른 목적, 다른 지표.
- **논문 매핑**: CE ↔ Prospection 자리(단 RL→지도로 철학 변경) · SFT ↔ Retrospection 자리(DPO→SFT).
  method 섹션 재작성 필요.

## 1. 방법론 (2단계, DPO 없음)

컨텍스트 `c_t = (x≤t frames, H<t history, D_t=WM Top-K)`. VLM 출력 trace `y_t=(reasoning r, task_belief g, action a∈D_t)`.

### ① candidate-CE — 정확도 엔진
- 입력: 영상 프레임 + 완료행동 history + **후보 D_t 제시**
- 목표: GT를 D_t 중에서 고르도록 지도학습 (후보 span CE, p(GT)↑)
- 산출: 판별력(SelAcc·GADR) 높은 정책 체크포인트 `θ_CE`
- WM prior: Phase-1 K8 (`RETRO4-...` export, cov@10 0.44)

### ② projected-trace SFT (+CE replay) — 인과적 행동 제어
- **θ_CE에서 초기화**(base 아님 — CE의 acc를 보존한 채 인과성만 추가)
- 목표: teacher가 hindsight로 만들어 결정시점에 투영한 이상적 `(r_proj, g_proj, a_GT)` 재현
- 손실(field-weighted span CE): `1.0·belief + 0.5·reasoning + 0.25·action`
  → belief 최고가중 = belief→action 인과 경로를 심는 핵심
- **CE replay 15~30%**: SFT step의 15~30%를 candidate-CE 배치로 섞어 판별력(SelAcc/GADR)
  퇴화 방지 (Ibrahim 2024 continual-replay). 기본 20%, non-harm 실패 시 30%로 상향.
- 산출: 인과 제어 강 + 판별력 보존 정책 `θ_CE+SFT`

### 순서 원칙 (2024+ 선행 기반)
- **기본: CE → SFT(+CE replay 20%)**. SFT를 마지막에 둬 belief-인과(2차 자산)를 fresh하게 남기고,
  CE는 replay로 보호. (근거: replay가 forgetting 완화 · SFT-후-최적화 컨벤션 DeepSeek-R1/Cosmos-Reason1 2025)
- **금지: CE → SFT → full-CE.** 마지막 full-CE는 action을 직접 최적화해 belief-인과를 워시아웃.
- **조건부(수리): G-NH 실패 시에만 CE → SFT → light-CE(경량 샌드위치)** — few steps·낮은 LR로
  판별 재선명화 후 **belief-인과 재측정 필수**(씻기면 롤백). full 재학습 아님.
- **헤지: 순차 간섭이 심하면(acc↔인과 trade-off 큼) 병렬 학습 후 weight merge**
  (Task Arithmetic/TIES 2023) — 둘 다 보존.

> 두 단계 모두 **candidate-presented** (후보를 프롬프트에 제시). inference도 D_t를 받음.
> "후보 비제시 내재화"는 이 방법론에 없음(precheck에서 헤드룸 작아 보류).

---

## 2. 정확도(acc) — 판단·지표화

### 2.1 지표 (논문 eq 12-14 + 확장)

| 지표 | 정의 | 의미 |
|---|---|---|
| Coverage@K | `Pr(GT ∈ D_t)` | WM 천장 (Phase-1: 0.44) |
| **SelAcc@K** | `Pr(a=GT \| GT∈D_t)` | 조건부 선택 정확도 (= 우리 acc\|cov) |
| **GADR** | `Pr(a=GT \| GT∈D_t, WM-top1≠GT)` | **모방으로 개선 불가한 하드 케이스 판별** |
| G1-retention | `Pr(a=GT \| GT∈D_t, WM-top1=GT)` | WM 정답을 지키나 (낮으면 WM 정답 버림) |
| acc(full) | `SelAcc × Coverage` | 전체분포 환산 |
| **WM-top1 ref** | `Pr(WM-top1=GT \| GT∈D_t)` | "그냥 WM top-1 따르기" 기준선 |
| fusion | `argmax_a [ℓ_VLM(a\|frames,r,g) + α·log q_WM(a)]` | α는 dev(calib) 보정 후 heldout 적용 |

- 측정: `tools/measure_gadr.py`(records 기반, GPU 불필요) · `tools/precheck_fair_fusion.py`(vision-grounded 채점+fusion)
- 모든 지표 **bootstrap 95% CI** 병기. 평가는 heldout covered-only, α는 반드시 dev/calib에서만 선택.

### 2.2 판정 게이트

- **G-ACC1 (모방 초과)**: `SelAcc(θ_CE)`의 CI 하한 > `WM-top1 ref` → VLM이 그냥 WM 따르기보다 낫다
- **G-ACC2 (하드케이스)**: `GADR(θ_CE) > 0` **그리고** `GADR(θ_CE) > GADR(base)` (CI 분리) → 학습이 모방 너머 판별을 민다
- **G-ACC3 (융합 가치)**: `fusion` CI 하한 ≥ `WM-top1 ref` → VLM 점수가 WM에 정보 추가 (G1 유지 낮음 보완)

### 2.3 현재 실측 앵커 (아직 CE 미학습 — 모두 base / projected-SFT-on-base)

| arm | SelAcc | GADR | G1-ret | WM-top1 | G-ACC1 |
|---|---|---|---|---|---|
| base (r4, Phase-1) | 0.200 | 0.164 | 0.314 | 0.242 | FAIL |
| projected-SFT (r4, n480) | 0.215 | 0.214 | 0.217 | 0.221 | FAIL |
| base (r3) | 0.223 | 0.213 | 0.257 | 0.230 | FAIL |
| projected-SFT (r3) | 0.234 | 0.222 | 0.274 | 0.230 | FAIL |

**공정 fusion 예고편 (E0 완료, base, Phase-1 prior, vision-grounded reasoning + WM fusion, n=475):**

| | acc\|cov |
|---|---|
| VLM 단독 (vision-grounded 후보 채점) | 0.213 |
| WM top-1 | 0.244 |
| VLM + WM fusion (best α on dev) | 0.223 (WM 대비 **−0.021**) |

> **base는 공정하게도 acc 기여 0/음수** — vision·후보·추론·fusion 다 줘도 WM top-1(0.244)을 못 넘음.
> (이전 vision-blind 0.123은 불공정이었고, 공정 수치 0.213도 WM 아래지만 격차는 ~0.03로 작음.)
>
> **핵심 미결**: 위는 전부 base/projected-SFT — **candidate-CE 체크포인트는 아직 없음**(v2 CE는 jihun2 별도 셋업).
> 이 방법론의 acc 생사는 "**CE-학습 VLM이 SelAcc > WM 0.244 를 통과하나**"(E1)이며 아직 미측정.

---

## 3. 인과적 행동 제어 — 판단·지표화

### 3.1 무엇을 재나
행동 `a`가 reasoning·belief의 **함수**인가(=내용을 바꾸면 행동이 바뀌나), 아니면 병렬 생성된 장식인가.
상관이 아닌 **개입(intervention)**으로 측정.

### 3.2 개입 프로토콜 (`tools`/`eval/harden_s3.py`)
프롬프트 뒤 `(r, g)` prefix 아래 D_t 후보 채점 → argmax = 그 조건의 선택. 변형별 flip 측정:

| 변형 | 조작 | 역할 |
|---|---|---|
| own | 자기 r·g | 기준 |
| swap | 다른 샘플의 r·g로 교체 | 개입(내용 변경) |
| paraphrase | 같은 g를 동의어 재작성 | **통제군**(문체만) |

### 3.3 지표

| 지표 | 정의 |
|---|---|
| **causal_sensitivity** | `flip(swap) − flip(paraphrase)` (통제군 차감이 핵심) |
| 필드 분해 | belief-only / reasoning-only / both swap — 어느 필드가 인과를 나르나 |
| utility | `p_gt(own) − p_gt(swap_both)` — 인과 채널이 **유용**(정답 방향)한가 |
| 직교성 | 정답/오답 샘플별 flip율 — acc와 독립 축인지 |

모두 bootstrap 95% CI. `both`-swap은 포화(prefix 통째 교체)라 판별력 없음 — **belief-only 채널이 헤드라인**.

### 3.4 판정 게이트

- **G-CC1 (인과 실재·학습효과)**: `causal_belief(θ_CE+SFT)` CI 하한 > `causal_belief(θ_CE)` 상한
  → SFT가 belief→action 인과를 **증가**시킴 (현재 base 0.058 → projected-SFT 0.390, CI 분리 = 강함)
- **G-CC2 (기계적 아님)**: 동일 포맷 base가 낮음(0.058) → 학습된 성질임을 입증
- **G-CC3 (유용성, 부차)**: `utility` CI 하한 > 0. ⚠️ projected belief는 GT 미포함 설계라 약할 수 있음
  (실측 base 0.108 → SFT 0.067, 하락). faithfulness는 강하나 utility는 약함 — 정직 보고.

### 3.5 현재 실측 앵커 (projected-SFT-on-base, retro3 n≈1000)

| 채널 | base | projected-SFT | 판정 |
|---|---|---|---|
| **belief-only** | 0.058 [.043,.075] | **0.390** [.358,.421] | G-CC1 PASS (6.7배, CI 분리) |
| reasoning-only | 0.220 | 0.295 | 소폭 |
| utility | 0.108 | 0.067 | G-CC3 약함(하락) |

---

## 4. Non-harm 게이트 — SFT가 CE의 acc를 깎지 않는가

- **G-NH**: `SelAcc(θ_CE+SFT) ≥ SelAcc(θ_CE) − 1pp` **그리고** `GADR` 하락 ≤ 2pp (CI로)
- 근거: SFT가 utility를 낮춘 흔적(0.108→0.067) → CE의 판별력을 훼손할 위험. 반드시 측정.
- **1차 방어 = CE replay 20%** (SFT에 내장, §1-②). 그래도 실패 시 순서대로:
  1. replay 30%로 상향
  2. **light-CE 샌드위치** (E3b, few steps·낮은 LR) — 판별 재선명화 후 belief-인과 재측정
  3. λ_belief 하향(제어↓ acc보존↑) 또는 SFT를 별도 adapter로 격리 / weight merge 헤지

---

## 5. 실험 순서 (retro4, Phase-1 prior, 전부 frames)

| # | 단계 | 내용 | 게이트/산출 |
|---|---|---|---|
| E0 | (완료) base 공정 fusion | vision-grounded 후보채점 + WM fusion | VLM 0.213 / WM 0.244 / fusion 0.223 — **base는 acc 기여 0** |
| **E1** | **candidate-CE 학습 (θ_CE)** | frames + 후보 제시, GT span CE | **생사 게이트: SelAcc(θ_CE) CI하한 > WM-top1 0.244?** → G-ACC1/2 |
| E1b | θ_CE 공정 fusion | E0 도구 재사용, θ_CE로 | G-ACC3 (fusion ≥ WM) |
| **E2** | **projected-SFT +CE replay 20% (θ_CE+SFT)** | θ_CE 초기화, belief 최고가중 | G-CC1 (belief-인과 ↑, base 0.058→목표 0.3+) |
| E3 | θ_CE+SFT non-harm 검사 | SelAcc·GADR 회귀 측정 | **G-NH: SelAcc 회귀 ≤1pp, GADR ≤2pp** |
| E3b | (조건부) G-NH 실패 시 light-CE 샌드위치 | few steps·낮은 LR | G-NH 재통과 **AND** belief-인과 생존 재확인 |
| E4 | 헤드라인 리포트 | acc(SelAcc/GADR/fusion) + 인과(causal_sensitivity/필드/직교) | 분기(§6) |

**의사결정 흐름**: E1이 이 방법론의 생사. `SelAcc(θ_CE) > WM 0.244` 통과 → acc 스토리 유효,
E1b~E4 진행. 미통과 → §6 재프레이밍(GADR+faithfulness)으로 직행, SFT는 인과 자산으로만 유지.

## 6. 분기 (게이트 종합)

- **G-ACC1∨G-ACC3 PASS**: acc 스토리 유효 → 논문 "VLM이 모방 넘어 선택" 유지.
- **G-ACC 전부 FAIL, G-CC1 PASS**: acc는 WM/fusion 위임, 논문을 **"GADR(하드케이스) + 인과적 grounding"** 으로 재프레이밍. 제목("WM 한계=VLM 한계")은 생존.
- **G-CC1 FAIL** (CE 위 SFT에서 인과 안 오름): SFT 스테이지 재고 (λ·데이터).
- **G-NH FAIL**: SFT가 acc 훼손 → 격리/재조정.

## 7. 도구 · 산출물

- `tools/measure_gadr.py` — SelAcc/GADR/G1 단계별 (records, GPU 불필요)
- `tools/precheck_fair_fusion.py` — vision-grounded 후보 채점 + WM fusion (records의 저장 reasoning 재사용)
- `eval/harden_s3.py` — 개입③ 인과 지표 (필드분해·CI·직교성)
- `eval/battery.py` — free-gen SelAcc/G1/GADR + records 생성
- 결과: `runs/{retro3,retro4}/eval/*.json`
- 미구현: **candidate-CE 학습 모듈**(E1) — retro4 Phase-1 prior용 신규 필요 (jihun2 v2 CE 참조)

## 8. 리스크 · 한계

- acc 생사가 "CE-VLM이 WM-top1 넘나"에 걸림 — 미측정. 넘으면 논문 강함, 못 넘으면 faithfulness/GADR 논문.
- 인과 지표는 candidate-presented 텍스트 채점 기반 — free-gen과 다를 수 있음(양쪽 병기 권장).
- projected belief의 GT 미포함 설계가 utility를 눌러 "제어 강·유용 약"의 구조적 tension 존재.
- CE는 GT 지도라 논문 Prospection(WM-분포 RL, GT 미사용)과 철학이 다름 — method 서술 정합 필요.
