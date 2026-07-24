# EGO Step-2 방법론 v2 — Predictive-Boundary Selection → Projected Retrospection (DPO-free)

- 작성: 2026-07-24 KST · EGO_jihun3
- **supersedes**: `2026-07-24_dpo_free_ce_sft_methodology_handoff.md` (v1)
- 반영: `2026-07-24_ce_sft_methodology_feedback_handoff.md` (peer review) 전면 수용
- 핵심 변경(v1→v2): ① CE를 **candidate-normalized selection CE**로 수식 명세 ② "belief 최고가중=인과경로"
  과잉해석 제거(학습은 belief 품질 강화, 인과 사용은 **측정**) ③ **belief-only utility를 필수 게이트**로 승격
  ④ 게이트를 **paired-bootstrap Δ**로 ⑤ **candidate-free CE 등 핵심 baseline** 추가 ⑥ 용어(causal→interventional)
  ⑦ 분기 4-outcome 재구성 ⑧ 선행연구 인용을 EGO-내부 가설로.

---

## 0. 성립 3조건 (이게 안 서면 논문 없음)

1. **WM-candidate CE > candidate-free CE 그리고 > WM top-1** (paired Δ, CI 하한>0)
   → WM이 만든 경계가 단순 GT fine-tuning·multiple-choice 효과를 넘어 학습 가치를 준다
2. **projected-SFT가 belief sensitivity + belief-only utility 둘 다** (utility만으로도 CI 하한>0)
   → belief가 행동에 쓰일 뿐 아니라 **정답 방향으로 유용**
3. **SFT 후 SelAcc·GADR 비손상**(non-harm)

sensitivity만 오르고 acc·GADR·utility가 정체하면 "embodied reasoning"이 아니라 **"belief-prefix sensitivity"**까지만 주장.

---

## 1. 2단계 objective (DPO 없음)

컨텍스트 `c_t=(x≤t frames, H<t history, D_t)`. trace `y=(r,g,a∈D_t)`. **candidate K=10**(cov@10 0.44).
※ Phase-1 "K=8"은 **visual history 길이**로 별개 축 — 모든 지표는 candidate-K=10에서 일관 계산(§6).

### Stage 1 — Predictive-Boundary Selection (구 candidate-CE)
후보 span의 단순 token CE가 아니라 **후보 집합 내 정규화 selection CE**:

```
s_θ(a;c) = (1/|a|) Σ_j log π_θ(a_j | c, a_<j)          # length-norm 후보 점수
p_θ(a|c,D) = exp(s_θ(a)/τ) / Σ_{a'∈D} exp(s_θ(a')/τ)   # 후보 집합 softmax
L_sel = − log p_θ(a_GT | c, D)
```
- 명칭: **candidate-normalized selection CE** (단순 "GT span CE" 표현 금지 — 길이·토크나이저·짧은후보 편향)
- 후보 shuffle, WM rank/prob 비공개, teacher reasoning/belief 미사용, covered 샘플 학습
- 산출: `θ_CE` (판별 엔진)

### Stage 2 — Projected Retrospection with CE Replay (구 projected-SFT)
- **θ_CE 초기화.** projected target `y^proj=(r_proj, g_proj, a_GT)` (hindsight를 결정시점에 투영)
- **field 내부 평균** loss (긴 reasoning이 gradient 지배하는 것 방지):
```
L_r = −(1/|r|)Σ log p(r_j)   L_g = −(1/|g|)Σ log p(g_j)   L_a = L_sel
L_proj = λ_r·L_r + λ_g·L_g + λ_a·L_a ,   λ_g=1.0, λ_r=0.5, λ_a=0.25
```
- ⚠️ **해석 주의**: λ_g를 크게 주는 것은 **projected belief 재현을 강화**할 뿐,
  "belief→action 인과 경로를 만든다"고 **단정 금지**. 모델은 `c→g`와 `c→a`를 병렬 학습할 수 있음.
  belief가 실제로 action에 쓰이는지는 **§4 개입으로 측정**한다.
- **CE replay (batch mixture, ρ=0.2)**:
```
매 step  B ~ B_proj (확률 1−ρ) | B_CE (확률 ρ),  ρ=0.2 (dev pilot로 결정, 기본값)
```
  방식 B(joint `L_proj+γL_CE`)가 아니라 방식 A(배치 혼합) 채택. 논문에 sampling ratio·각 데이터셋 크기·
  epoch 기준·optimizer step 수 명시. non-harm 실패 시 ρ 0.3 상향.
- 산출: `θ_CE+SFT`

### 순서 원칙 (EGO-내부 가설 — 외부 컨벤션 아님)
- 기본 **CE → SFT(+replay)**: projected-SFT가 최종 r-g-a trace 구조를 형성하므로 CE 뒤 배치, CE replay로 판별 보존.
  (DeepSeek-R1/Cosmos-Reason1을 **직접 근거로 인용하지 않음** — 해당 논문은 이 순서를 일반규칙으로 지지 안 함)
- 금지 **CE→SFT→full-CE**: 마지막 full-CE가 action loss만 재가해 belief-dependence 워시아웃
- 조건부 **light-CE 샌드위치**(G-NH 실패 시만): few-step·low-LR repair, 이후 belief sensitivity+utility 재측정, 씻기면 롤백
- 헤지 **weight merge**(Task Arithmetic/TIES): 순차 간섭 심할 때만 **exploratory fallback(appendix)**, 본문 방법 아님

---

## 2. 정확도 지표 · paired-bootstrap 게이트

지표(§6 K=10 공통): Coverage@10 · SelAcc@10 · GADR · G1-retention · WM-top1 ref · fusion.
**모든 판정은 같은 샘플의 paired 차이 + video-cluster bootstrap 95% CI 하한.**

- **G-ACC1** (모방 초과): `Δ_WM = SelAcc(θ_CE) − Acc(WM-top1)`, `CI_low(Δ_WM) > 0`
- **G-ACC2** (학습 효과): `Δ_GADR = GADR(θ_CE) − GADR(base)`, `CI_low > 0` (+ no-video/no-history baseline 초과)
- **G-ACC3** (정보 추가): `Δ_fusion-WM = Acc(fusion) − Acc(WM)`, `CI_low > 0`; `Δ_fusion-VLM`도 병기.
  fusion이 **최종 inference면 WM prob를 직접 사용** → "rank/prob 비공개" 설계와 충돌하므로 **보조 분석**으로 규정(본 selection은 hidden 유지).

현재 앵커(E0, base, 공정): VLM 0.213 / WM 0.244 / fusion 0.223 → **base는 acc 기여 0.** CE가 넘어야 함(E1 생사).

---

## 3. 핵심 baseline — "WM 경계 때문인가, 그냥 GT CE 때문인가"

리뷰어 1순위 질문 대응. 모든 arm 동일 프레임·history·학습설정.

| Arm | 입력 | 격리하는 것 |
|---|---|---|
| Base VLM | frames + history | 무학습 기준 |
| **Candidate-free CE** | frames + history (후보 無) | **GT CE 자체 효과** |
| Random-candidate CE | frames + history + 랜덤 후보 | 후보 제시(multiple-choice) 효과 |
| **WM-candidate CE** | frames + history + WM Top-K | **WM 경계 효과** |
| No-video WM-cand CE | history + WM Top-K | 영상 사용 검증 |
| No-history WM-cand CE | frames + WM Top-K | task progression 검증 |
| WM top-1 | WM ranking | 단순 모방 |

**필수 부등식**(paired Δ, CI 하한>0):
- `WM-candidate CE > candidate-free CE` → WM 경계가 실제 학습 가치 (핵심 주장)
- `WM-candidate CE > random-candidate CE` → WM 후보가 "보기 중 고르기" 효과가 아님

이 둘이 없으면 성립조건 1 미충족.

---

## 4. interventional belief dependence — 지표·게이트 (승격)

용어: **causal 아님 → interventional.** "belief→action 인과 경로/매개" 표현 금지. §9 참조.

### 개입 (harden_s3.py)
own / **belief-only swap**(reasoning 고정) / paraphrase(통제) / reasoning-only / both.
**belief utility는 반드시 reasoning 고정 + belief만 swap**:
```
U_g = p(a_GT | r_own, g_own) − p(a_GT | r_own, g_swap)     # belief-only utility
D_g = Pr[ p(a_GT|g_own) > p(a_GT|g_swap) ]                  # directional
correct-switch = swap belief가 특정 후보를 지지할 때 그 후보로 전환하는 비율 (아무 flip 아님)
```
(현행 harden_s3는 `own−swap_both`로 utility를 잘못 계산 — **swap_b(belief-only)**로 교체.
P["own"], P["swap_b"]는 이미 기록되므로 재계산만 필요.)

### 게이트 (전부 bootstrap CI)
- **G-CC1**: belief sensitivity 증가 (base 0.058 → 목표 ↑, CI 분리)
- **G-CC2**: 동일 포맷 base 낮음 → 학습된 성질
- **G-CC3 (필수 승격)**: **belief-only utility U_g CI 하한 > 0**
- **G-CC4**: directional utility D_g > chance 또는 > base
- **최소 조건: G-CC1 ∧ G-CC3** 둘 다 통과해야 "useful belief dependence" 주장 가능

현재 앵커: sensitivity 0.058→0.390(G-CC1 강함) · utility(own−swap_both) 0.108→0.067(하락).
→ **belief-only U_g로 재측정 필수.** 하락이 belief-only에서도 확인되면 G-CC3 FAIL = sensitivity-only 주장으로 제한.

---

## 5. Non-harm

- **G-NH**: `SelAcc(θ_CE+SFT) − SelAcc(θ_CE) ≥ −1pp` **∧** `GADR 하락 ≤ 2pp` (paired CI)
- 1차 방어 = CE replay ρ=0.2 → 실패 시 ρ=0.3 → light-CE 샌드위치(§1) → λ_g 하향/격리/merge

---

## 6. K 일관성

- **candidate set K=10** 고정. Coverage/SelAcc/GADR/G1/WM-top1/fusion **전부 K=10 동일 후보셋**에서 계산.
- Phase-1 "K=8"(visual history 길이)와 혼동 금지 — 별개 축.
- 논문 ablation(#1)에서 candidate K∈{5,10,15} 변화로 coverage↔discrimination trade-off 별도 보고.

---

## 7. 실험 순서

| # | 단계 | 게이트 |
|---|---|---|
| E0 (완료) | base 공정 fusion | VLM 0.213/WM 0.244/fusion 0.223 — base acc 기여 0 |
| **E1** | **WM-candidate CE 학습 θ_CE** | **생사: G-ACC1** `Δ_WM CI>0` |
| E1b | baseline arms: candidate-free/random-candidate/no-video/no-history CE | **§3 부등식**(WM-cand>cand-free, >random) + G-ACC2 |
| E1c | θ_CE 공정 fusion | G-ACC3 |
| E2 | projected-SFT +CE replay ρ0.2 (θ_CE+SFT) | G-CC1 |
| E3 | 개입 재측정: belief-only U_g, D_g, correct-switch | **G-CC3(필수)**, G-CC4 |
| E3n | non-harm | G-NH (실패 시 E3s light-CE) |
| E4 | 헤드라인 리포트 | 분기(§8) |

---

## 8. 분기 (4-outcome)

- **8.1 강한 성공** (G-ACC1·G-ACC2·G-CC1·U_g·G-NH PASS): "VLM이 WM 경계 안에서 top-1 모방을 넘어
  task-conditioned selection 수행 + projected belief가 행동에 유용하게 관여" — full claim.
- **8.2 제한적 성공** (G-ACC1 FAIL·G-ACC2 PASS·G-CC1·U_g·G-NH PASS): "평균 SelAcc는 WM 미달이나
  **WM top-1이 틀린 hard case에서 의미있는 교정(GADR)** + useful belief dependence" — headline=GADR/hard-case.
- **8.3 sensitivity-only** (G-ACC1·G-ACC2 FAIL·G-CC1 PASS·U_g FAIL): "belief-prefix sensitivity만 증가,
  acc·grounding 개선 미입증" — **embodied reasoning 제목 유지 곤란.**
- **8.4 실패** (CE 판별 개선 실패 ∨ SFT belief-dependence 실패 ∨ non-harm 위반): 2단계 방법 전체 재검.

(v1의 "G-ACC 전부 FAIL → GADR 재프레이밍"은 모순 — G-ACC2(GADR)도 FAIL이면 GADR로 못 감. 8.2/8.3로 분리.)

---

## 9. 용어

- 금지: "belief→action 인과 경로", "belief가 action을 인과적으로 매개", "causal faithfulness 달성"
- 권장: **interventional belief dependence · belief-conditioned action sensitivity · semantic intervention sensitivity · belief utility**
- 논문 문장: *"We test whether action selection is interventionally dependent on the semantic content of the generated task belief."*

---

## 10. 도구 · 미구현

- 있음: `measure_gadr.py`(paired Δ 확장 필요) · `precheck_fair_fusion.py`(E0/E1c) · `harden_s3.py`(belief-only U_g 슬라이스로 교체) · `battery.py`
- **미구현(신규)**:
  1. Stage-1 candidate-normalized selection CE 학습 모듈 (τ, softmax-over-D)
  2. baseline arm 스위치 (candidate-free / random-candidate / no-video / no-history)
  3. CE-replay batch mixture(ρ) 를 projected-SFT에 통합
  4. paired video-cluster bootstrap Δ 유틸 (G-ACC1/2/3, G-NH)
  5. belief-only U_g · D_g · correct-switch 지표 (harden_s3 확장)

---

## 11. 논문 재서술 (P2)

- Prospection(RL, WM분포 정렬) → **Predictive-Boundary Selection(지도 selection CE)** 로 명칭·수식 교체.
  "WM distribution alignment" 주장 제거.
- Retrospection(DPO) → **Projected Retrospection SFT(+CE replay)**.
- 헤드라인: "predictive boundary + task-conditioned discrimination" 중심, sensitivity는 interventional dependence로.
- weight merge는 appendix contingency.
