# cesft_v2 — 능력별 정량 지표·실측 근거 종합 Handoff (CE → projected-SFT, DPO 배제)

> 작성: 2026-07-25 KST · EGO_jihun3 · **실행 2026-07-24 (runs/cesft_v2, CHAIN 완료).**
> **갱신: 2026-07-26 — 전 지표를 full validation 산출물로 재계산.** 초판(07-25 09:26)은
> heldout n=1,000 부분셋·sample bootstrap 기준이었고, 본 갱신은 **full covered(θ_CE/+SFT n=2,313,
> covered 판정 후 paired n=2,218) · video-cluster bootstrap · 개입 n=800 재실행** 기준이다.
> 수치 출처는 전부 `runs/cesft_v2/eval/*.json` 원본(§6 좌표).
> **목적: 2단계 지도학습(candidate-CE θ_CE → projected-trace SFT + CE replay)이 만드는 능력들을
> "능력 → 정량 지표(도출법) → 실측치 → 근거 로그 → 원천 학습 신호"로 완결 매핑.**
> 증거 등급: **[확정]**=paired/CI 통과 실측 · **[시사]**=실측이나 교란/검정력 한계 ·
> **[미검증]**=측정 설계만 · **[반증]**=기대와 반대로 실측.
> 자매 문서(EGO_jihun, 다른 코호트): `2026-07-24_reasoning_quality_quantitative_evidence_handoff.md`
> — 두 실험은 같은 CE→SFT 계열을 **다른 코호트·시각계약**에서 검증(방향 일치, 크기는 파이프라인차). 수치 직접 비교 금지, 방향·구조만 교차.
> 방법론 SSOT: `2026-07-24_dpo_free_ce_sft_methodology_handoff.md` · `2026-07-24_ce_sft_methodology_v2_handoff.md`
> 시각 보고서(artifact, full): https://claude.ai/code/artifact/e4bafc93-9e5e-480c-8ef4-5adcd6a646f3
> 구판 artifact(부분셋, 참고용): https://claude.ai/code/artifact/922dc65e-5fd9-4b8b-b7b6-48e2b02734d2

---

## 0-A. 초판 대비 **판정이 바뀐 항목** (읽기 전 필독)

| 항목 | 초판 (부분셋) | **full validation** | 성격 |
|---|---|---|---|
| **G-NH (SFT 비손상)** | **FAIL(비유의)** +1.1pp[−5.7,+7.1] | **PASS(비열등)** +2.2pp[−0.2,+4.7] · GADR +3.0pp[−0.1,+6.2] | **판정 전환** |
| **GADR 국면** | θ_CE 정점 25.6 → SFT 24.0 (**SFT 감소**) | θ_CE 22.9 → SFT **26.0** (**SFT 증가**) | **방향 반전** |
| **G1 retention** | 41.2 → SFT **48.5** (SFT가 올림) | 40.5 → SFT **39.6** (소폭 내림) | **방향 반전** |
| **§1-1 해석** | "SFT는 G1 retention만 추가" | "SFT는 GADR을 올리고 G1은 소폭 내림" | **결론 교체** |
| **history 원천 귀속** | candidate-CE 단독 | **CE와 SFT 둘 다** (Stage 2 DiD +5.4pp[0.2,10.3] 유의) | **귀속 확장** |
| strip Δ 크기 | +3.1pp[1.1,5.2] | +11.6pp[7.5,16.0] | **규약 변경**(full-set·sample → covered·cluster). 값 변화 아님 |
| 개입 채널 대칭성 | belief≈reasoning (34.1≈33.1) | **arm마다 배분이 다름** (r15 belief 우위 / r0 reasoning 우위) | **해석 정교화** |
| SelAcc θ_CE | 30.8 | **27.0** | 부분셋 편향 제거 |

신규로 추가된 것: **CE-replay 직접 paired 근거**(r0 vs r15) · **arm 3종 개입 전개**(r0 · WiSE-FT) ·
**3 체크포인트 history strip + DiD** · **텍스트 지표 4종 전 arm 재계산**.

---

## 0. 다섯 줄 요약

1. **CE = 정확도 엔진 [확정]**: candidate-CE(θ_CE)가 WM top-1 모방을 넘어 task-conditioned 선택 —
   SelAcc 20.0→**27.0%** (L0 23.3% 초과, **G-ACC1 +4.8pp[1.3,8.5] PASS**, n=2,218·cluster 87).
2. **SFT = 인과적 추론 채널 [확정]**: belief/reasoning 개입 시 행동 flip **both 80.0%** vs 대조(paraphrase)
   3.5% · 유용성 U_g **+10.5pp[8.7,12.3]** — reasoning이 장식이 아니라 행동을 인과. **개입 verdict "PASS — spine 확정 (U_g)"**.
3. **SFT의 acc 기여는 비손상 [PASS]**: G-NH SelAcc +2.2pp[−0.2,+4.7] · **GADR +3.0pp[−0.1,+6.2]** —
   초판의 FAIL 판정이 full셋에서 뒤집혔다. acc 도약은 여전히 CE 몫이나 **SFT는 손상하지 않으며 GADR은 오히려 올린다**.
4. **reasoning 품질: CE가 지우고 SFT가 되살림 [확정]**: 배제 언명 27.0→CE **10.5**→SFT **28.6%** ·
   평균 단어 69.3→57.6→80.3 · 장면 묘사 55.0→53.3→**34.3%**(SFT가 −19.0pp).
5. **history 인과 [확정 · 3 체크포인트]**: strip Δ가 Base **+1.1pp[−1.6,+4.2] 비유의** → θ_CE +6.2 →
   +SFT **+11.6pp[7.5,16.0]**로 단조 증가. **DiD 3쌍 모두 유의**. strip 조건에서는 세 체크포인트가
   18.9/19.6/18.3%로 동일 — **학습 이득이 거의 전부 이력 채널에서 나온다**.

**배경**: 논문 eq.11 Retrospection의 **DPO는 두 번 붕괴**(G3 문체학습 abort ×5 → margin collapse → malformed 100%)해
사용자 결정으로 배제. projected-trace SFT만으로 faithfulness 달성.

---

## 1. 능력 → 지표 → 실측 매핑 (마스터 표)

측정 기반: **covered(GT∈WM Top-10) 부분집합** — θ_CE/+SFT는 heldout n=5,326 중 covered 2,313
(paired 판정 n=2,218, cluster 87), base는 covered-only 평가 n=1,000. 개입은 **harden_s3 재실행
n=800(r15) · 405(r0) · 388(WiSE)**. CI는 전부 **video-cluster bootstrap**(개입만 sample bootstrap).
WM prior = jihun2 Phase-1 K8(cov@10 **43.43%**, end−1s 계약).
arm: **base**(무학습) · **θ_CE**(candidate-CE) · **+SFT**(θ_CE 초기화 → projected-trace SFT + CE replay ρ=0.15).

| 능력 | 정량 지표 (도출법) | 실측 | 등급 | 원천 학습 신호 |
|---|---|---|---|---|
| **후보 선택·정확도** | SelAcc = covered 선택정확도 · G-ACC1 = SelAcc vs L0(WM top-1) | 20.0→**27.0**→29.2% (+4.8pp[1.3,8.5]) · L0 23.3 초과 | **확정** | **candidate-CE** |
| **모방 초과 판별 (GADR)** | Pr(선택=GT \| GT∈후보 ∧ WM top-1≠GT), g2 부분집합 | base 16.4→θ_CE 22.9→SFT **26.0%** | **확정** | **candidate-CE**(도약) + **SFT**(추가) |
| **history 사용** | paired strip Δacc = acc(hist)−acc(strip), WM후보 고정 | Base +1.1[−1.6,4.2] n.s. → θ_CE +6.2 → **+SFT +11.6**[7.5,16.0] | **확정** | **CE와 SFT 둘 다** (DiD §2-3) |
| **belief→action 인과** | belief sensitivity = flip(swap_b)−flip(para); U_b(belief-only) | flip **32.6%** · sensitivity **0.291**[0.259,0.325] · U_b +5.4pp[4.3,6.6] | **확정** | **SFT** |
| **reasoning 인과·유용** | both flip; U_g = p(GT\|own)−p(GT\|swap_both) | both flip **80.0%** · U_g **+10.5pp**[8.7,12.3] | **확정** | **SFT** |
| **CE-replay의 판별 보존** | GADR(sft_ρ)−GADR(θ_CE) paired · r0 vs r15 직접 paired | r0 GADR **−6.5pp[−12.4,−0.8] 유의 붕괴** · r15 −1.4 n.s. · 직접 −4.4pp[−8.1,−1.1] | **확정** | **CE replay (ρ=0.15)** |
| **소거 서술 (비교·배제)** | reasoning의 배제 언명률 (후보-제시 생성) | base 27.0 → CE **10.5** → SFT **28.6%** | **확정** | **SFT** (CE가 침식) |
| **장면 묘사 억제** | 묘사 정규식 검출률 ↓ | 55.0 → 53.3 → **34.3%** (SFT −19.0pp) | **확정** | **SFT** |
| **SFT의 acc 기여** | G-NH = SelAcc(+SFT) − SelAcc(θ_CE) paired | +2.2pp[−0.2,+4.7] · GADR +3.0pp[−0.1,+6.2] | **확정(비열등)** | — (도약은 CE) |
| **간결화** | reasoning 평균 단어 수 | 69.3→CE 57.6→SFT 80.3 (CE 압축·SFT 확장) | **시사** | CE 압축 / SFT 확장 |
| **egocentric 화법** | 1인칭 검출률 `\b(I\|my\|me)\b` | 후보-제시 조건 전 arm **0%** (자유생성 미측정) | **미검증** | 자매 JIHUN: 반증(74→7%) |

**로그 좌표**: 정확도·GADR = `eval/{base,theta_ce,sft_r15,sft_r0,wise_a050}.json` + `.records.jsonl`;
게이트 = `eval/paired_{G-ACC1,G-NH,direct}_*.json`; history = `eval/strip_metrics.json` + `{arm}_nohist.records.jsonl`;
개입 = `eval/{sft_r15,sft_r0,wise_a050}.harden_s3.{json,records.json}`; 텍스트 지표 = records `reasoning` 필드 재집계.

### 1-1. 정확도 3층위 분해 (이득의 출처)

> 판독: `eval_candidate_scored.py` — WM 후보 K=10 teacher-forcing sum-logp argmax. GT = (verb,noun) strict 일치.
> L0 = WM top-1 그대로 따르기. base 열은 n=1,000 covered-only 서브셋이라 L0가 24.2로 다르게 잡힌다.

| 지표 | base | θ_CE | +SFT | 읽는 법 |
|---|---:|---:|---:|---|
| **full-eq acc** % (=SelAcc×cov) | 8.7 | 11.7 | **12.7** | coverage 43.4%가 천장 |
| **SelAcc** % (covered) | 20.0 | **27.0** | **29.2** | **CE의 도약** — L0(23.3) 초과. base는 L0 아래 |
| G1 retention % (WM top-1=GT) | 31.4 | **40.5** | 39.6 | WM 맞을 때 지키기 — **θ_CE 정점, SFT 소폭↓** |
| G2 correction % (GADR) | 16.4 | 22.9 | **26.0** | hard 교정 — **SFT가 더 올림** |
| L0 (WM top-1) % | 24.2 | 23.3 | 23.3 | 모방 바닥선 |
| malformed rate % (covered) | 1.1 | 4.1 | 3.2 | 형식 파탄 — CE↑ SFT 일부 회복 |

**해석**: **CE가 acc 도약(SelAcc 20.0→27.0)을 만들고, SFT는 그 위에 GADR(22.9→26.0)을 추가**한다 —
"WM이 틀린 hard-case를 교정하는" 능력. G1(WM이 맞을 때 지키기)은 θ_CE가 정점이고 SFT에서 −0.9pp
소폭 내려간다. 초판은 이 두 지표의 방향을 반대로 적었으므로 **인용 시 본 표를 SSOT로 사용**할 것.

### 1-2. 사전 등록 게이트 판정

| 게이트 | 기준 | 결과 | 수치 |
|---|---|---|---|
| **G-ACC1** | θ_CE SelAcc > L0(WM top-1) paired CI 하한>0 | **PASS** | Δ+4.82pp CI[1.31, 8.45] · SelAcc 28.13 vs L0 23.31 · n=2,218 cluster 87 |
| **G-NH** | +SFT SelAcc ≥ θ_CE 비열등(하한>−1pp) | **PASS** ⟵ *초판 FAIL* | SelAcc Δ+2.22pp CI[−0.15, +4.68] · GADR Δ+3.01pp CI[−0.06, +6.19] · n=2,165 cluster 87 |
| **CE-replay (r0)** | ρ=0 arm이 θ_CE 대비 비열등인가 | **FAIL(붕괴)** | SelAcc −4.17pp[−9.67,+0.91] · **GADR −6.48pp[−12.40,−0.76] 유의** · n=384 cluster 76 |
| **CE-replay 직접** ★신규 | r0 vs r15 직접 paired | **r0 유의 열세** | SelAcc −4.34pp[−8.33,−0.59] · GADR −4.36pp[−8.11,−1.12] · n=392 cluster 76 |
| **G-HIST (+SFT)** | acc(hist) > acc(strip) paired CI 하한>0 | **PASS** | Δ+11.6pp CI[7.52, 16.02] · n=1,000 cluster 86 |
| **G-HIST (θ_CE)** | 동일 | **PASS** | Δ+6.79pp CI[2.83, 11.05] · n=1,223 cluster 87 |
| **G-HIST (base)** | 동일 | **비유의**(기대대로) | Δ+1.1pp CI[−1.61, +4.18] · n=1,000 |
| **G-CC1** | belief sensitivity CI 하한>0 | **PASS** | 0.291 CI[0.259, 0.325] · n=800 |
| **G-CC3 (U_b)** | belief-only utility CI 하한>0 | **PASS** | +5.41pp CI[4.27, 6.55] · n=800 |
| **G-S3a / S3b** | both sensitivity · own>swap_both | **PASS** | 0.765[0.735,0.796] · U_g +10.48pp[8.66,12.29] |
| **G-CC2** | 동일 포맷 base의 sensitivity가 낮은가 | **SKIP** | base·θ_CE harden 패스 부재 → 계산 불가 |
| **G-DELTA** | 후보제시 > 자유생성 | **SKIP** | cand_free arm 미학습(`no covered records for arm_b='cand_free'`) · 자매 matched +1.2pp[−5.2,+6.7] 비유의 |

---

## 2. 능력 상세

### 2-1. 후보 선택·정확도 [확정] — CE 엔진

- **도출**: SelAcc = 정답이 WM 후보 안에 있을 때(covered)의 선택 정확도. G-ACC1은 이를 L0(WM top-1
  그대로 따르기)와 paired 비교 — LM이 WM 순위 모방을 넘어 task-conditioned 재선택을 하는지의 판정.
- **실측**: base 20.0% < L0 24.2% (안 배운 VLM은 WM보다 못함) → θ_CE **27.0%**
  (paired 서브셋 28.13 vs L0 23.31, +4.82pp, CI 하한 1.31>0) → +SFT 29.2%.
- **의미**: candidate-CE가 정확도의 도약을 단독으로 만든다. full-eq(8.7→11.7)는 coverage(43.4%)가
  천장이라 이득이 희석 — "coverage가 지배 변수" 서사와 정합.
- **⚠ 초판 대비**: θ_CE SelAcc가 30.8 → **27.0**으로 내려갔다(부분셋 편향 제거). 게이트는 여전히 PASS이나
  **L0 초과 마진이 +7.2pp → +4.8pp로 축소**되었으므로, 논문에서 이 마진 크기에 기대는 서술은 재작성 필요.

### 2-2. 모방 초과 판별 GADR [확정]

- **도출**: GADR = Pr(선택=GT | GT∈후보 ∧ WM top-1≠GT). WM top-1이 틀린 hard-case만 채점 —
  WM 순위 복사 전략은 정의상 0점이므로 점수는 전부 맥락 기반 재선택에서 온다.
- **실측**: base 16.4 → θ_CE 22.9 (+6.5pp) → **+SFT 26.0%** (+3.1pp 추가).
  **초판과 방향이 반대다** — 초판은 SFT에서 24.0으로 감소한다고 적었으나, full셋에서는 SFT가 GADR을 더 올린다.
- **자매 교차**: JIHUN GADR base 18.0→C 41.3% (+23.3pp, n=584). 방향 일치(코호트차로 크기 다름).

### 2-3. history 사용 [확정] — 3 체크포인트 실측 + DiD

- **도출**: paired history-strip — 같은 샘플에서 프롬프트의 완료-행동 이력만 `(history withheld)`로
  치환하고 나머지(프레임·WM 후보·후보 순서)는 고정한 **단일-요인 개입**. Δ = acc(hist) − acc(strip).
  별도 no-history arm 학습(B_nohist) 없이 per-sample paired이므로 "다른 체크포인트" 교란이 없다.
  covered · video-cluster bootstrap · malformed는 오답 처리.

| 체크포인트 | acc(hist) | acc(strip) | Δ SelAcc | 95% CI | Δ GADR | Δ G1 | 판정 |
|---|---:|---:|---:|---|---:|---:|---|
| Base (n=1,000) | 20.0 | 18.9 | +1.1 | [−1.61, +4.18] | +0.5 | +2.9 | **비유의** |
| θ_CE (n=1,223) | 26.4 | 19.6 | +6.8 | [+2.83, +11.05] | +6.8 | +6.6 | **PASS** |
| **+SFT (n=1,000)** | **29.9** | 18.3 | **+11.6** | [+7.52, +16.02] | +10.7 | +14.5 | **PASS** |

- **핵심 — strip 수평선**: history를 빼면 세 체크포인트가 **18.9 / 19.6 / 18.3%로 사실상 동일**(모두
  L0=23.3보다 낮다). 즉 §1-1의 학습 이득(20.0→26.4→29.9)은 **거의 전부 이력 채널에서 나온다** —
  학습이 만든 것은 "이력을 판별에 쓰는 능력"이고, 프레임·후보만으로 얻는 성능은 학습 전후가 동일하다.
- **DiD (동일 covered 1,000셋 · cluster 86)**:

  | 비교 | Δ_strip(A) | Δ_strip(B) | DiD | 95% CI | 판정 |
  |---|---:|---:|---:|---|---|
  | θ_CE − Base (Stage 1) | +6.2 | +1.1 | **+5.1** | [+1.0, +9.8] | 유의 |
  | +SFT − θ_CE (Stage 2) | +11.6 | +6.2 | **+5.4** | [+0.2, +10.3] | 유의 |
  | +SFT − Base (전체) | +11.6 | +1.1 | **+10.5** | [+6.1, +15.3] | 유의 |

- **history 길이 층화**: H8+에서 Δ가 가장 크다 — θ_CE +7.4[3.3,11.7] · +SFT +11.9[7.1,16.6].
  H0(n=5)은 표본 부족으로 제외. **다만 fine-grained 재버킷(실제 이력 길이 중앙값 37·최대 250)에서는
  단조 용량-반응이 성립하지 않는다** — +SFT는 H0-7 +9.3부터 H96+ +9.3까지 전 구간에서 CI 하한>0으로
  **국면에 무관하게 균일**하다. "길수록 더 쓴다"가 아니라 "학습 단계에 따라 켜진다"가 맞는 진술.
- **반증 방어**: malformed는 strip에서 **줄어든다**(Base 1.1→0.7 · θ_CE 4.4→2.1 · +SFT 3.1→0.8,
  프롬프트가 짧아짐) → Δ가 포맷 붕괴의 산물이 아니다.
- **⚠ 초판 대비**: 초판의 +3.1pp[1.1,5.2]는 **full-set(uncovered 포함)·sample bootstrap** 값이라
  구조적 0점에 희석된 수치다. 본 갱신은 covered·cluster 규약(`strip_metrics.json`). 값이 커진 것은
  모델 변화가 아니라 **규약 변경**이므로 두 수치를 나란히 인용하지 말 것.
- **원천 귀속 변경**: 초판은 history 사용을 candidate-CE 단독 귀속으로 적었으나, Stage 2 DiD가
  +5.4pp[0.2,10.3]로 유의하므로 **CE와 SFT 둘 다** 기여한다. 실제로 이것이 본 런에서
  **SFT가 유의하게 개선한 유일한 지표**다.

### 2-4. belief→action 인과 & reasoning 유용성 [확정] — SFT 채널

- **도출** (`harden_s3.py`): 각 arm이 생성한 `(reasoning, belief)`를 강제 교체(swap)하고 K=10 선택 flip 측정.
  핵심 통제군 = **paraphrase**(같은 뜻 재서술)가 유발하는 flip(문체 노이즈)을 빼 "의미 변화로 바뀐 정도"만 남김.
  paired bootstrap CI, donor 결정적 선정(`(i+7) % n`).
- **실측 — +SFT ρ=0.15 (Ours, n=800 재실행)**:

  | 채널 | flip율 | 95% CI | causal sensitivity | 유용성 Δp(GT) |
  |---|---:|---|---:|---:|
  | belief만 swap | 32.6% | [29.5, 35.9] | **0.291** [0.259, 0.325] | **+5.4pp**[4.3,6.6] (U_b) |
  | reasoning만 swap | 29.1% | [26.1, 32.4] | 0.256 [0.224, 0.289] | — |
  | **both** swap | **80.0%** | [77.2, 82.9] | **0.765** [0.735, 0.796] | **+10.5pp**[8.7,12.3] (U_g) |
  | paraphrase (대조) | 3.5% | [2.2, 4.8] | — | — |

- **arm 전개 (기존 harden 산출물 재집계 · 신규 추론 0)**:

  | 지표 | +SFT ρ=0 (n=405) | **+SFT ρ=0.15 (n=800)** | WiSE-FT α=0.5 (n=388) |
  |---|---:|---:|---:|
  | belief sensitivity | 0.284 | **0.291** | 0.116 |
  | reasoning sensitivity | **0.346** | 0.256 | 0.242 |
  | both sensitivity | 0.763 | 0.765 | 0.765 |
  | 초가법 gap (both − b − r) | +0.133 | +0.218 | **+0.407** |
  | U_b (belief-only) | +4.1pp | **+5.4pp** | +4.7pp |
  | U_g (own − swap_both) | +6.2pp | +10.5pp | **+12.4pp** |
  | directional D_g | 0.435 | **0.463** | 0.350 |
  | correct-switch (flip 시 p(GT) 하락) | −4.4pp (n_flip 134) | −4.6pp (n_flip 261) | −4.9pp (n_flip 52) |
  | acc 직교성 flip(both) 정답 vs 오답 | 73.5 vs 83.4 (Δ−9.9) | 76.2 vs 81.6 (Δ−5.5) | 79.5 vs 77.9 (Δ+1.6) |
  | acc_own (개입 셋 내) | 24.2 | **29.9** | 28.9 |

- **핵심 3가지**:
  ① **채널 배분이 arm마다 다르다** — r0는 reasoning 우위(0.346 > 0.284), r15는 역전되어 belief 우위
  (0.291 > 0.256). **CE-replay는 정확도만 지키는 게 아니라 개입 반응을 reasoning 채널에서 belief 채널로 옮긴다.**
  초판의 "belief≈reasoning 대칭" 해석은 부분셋(n=396)에서 두 값이 우연히 가까웠던 것이므로 폐기.
  ② **both sensitivity는 세 arm 모두 0.763~0.765로 포화** — prefix 전체 상한은 학습으로 거의 변하지 않고,
  학습이 바꾸는 건 그 상한을 belief/reasoning 중 어느 채널로 쪼개 쓰느냐이다.
  ③ **WiSE는 belief 채널만 선택적으로 깎인다** — belief 0.116(r15의 40%)인데 reasoning 0.242·both 0.765는
  r15와 사실상 동일하고 **정확도도 28.9 vs 29.9로 동률**. 즉 채널은 정확도의 부산물이 아니다.
  그런데도 U_g는 +12.4pp로 최고 → **U_g가 belief 채널 능력의 대리 지표가 될 수 없다는 직접 반례**이고,
  U_b/G-CC1로 갈아탄 결정을 사후적으로 정당화한다.
- gate `S3a_causal_real`·`S3b_utility_real`·`CC1_belief_sensitivity`·`CC3_belief_only_utility` **전 arm true**,
  verdict **"PASS — spine 확정 (U_g)"**.
- **미측정**: `p_empty`(belief 제거 조건)는 주변 CI만 있고 paired diff·acc가 집계되지 않았다
  (r15 p_own 0.2303 vs p_empty 0.2164). base·θ_CE는 harden 패스 자체가 없어 **G-CC2 계산 불가**.
- **자매 교차**: JIHUN belief sensitivity base 0.058→SFT 0.390(6.7×), U_g 0.023→0.042. 방향 일치.
- **용어 규율**: "interventional dependence"까지만 — "causal mediation" 금지.

### 2-5. reasoning 품질 (텍스트 지표 4종) [확정] — CE가 지우고 SFT가 되살림

- **도출**: covered ∧ non-malformed records의 `reasoning` 필드 정규식 채점
  (`tools/trace_text_metrics.py`, 파일럿과 동일 정규식). n = base 989 / θ_CE 2,218 / +SFT 2,239.

| 지표 | base | θ_CE | +SFT | Δ₁ | Δ₂ |
|---|---:|---:|---:|---:|---:|
| 장면 묘사율 % ↓ | 55.0 | 53.3 | **34.3** | −1.7 | **−19.0** |
| 미래-지향율 % | 94.8 | 95.4 | 57.9 | +0.6 | −37.5 |
| 배제 언명률 % | 27.0 | **10.5** | **28.6** | −16.5 | +18.1 |
| 1인칭율 % (cand-present) | 0.0 | 0.0 | 0.0 | 0 | 0 |
| reasoning 길이 (단어) | 69.3 | 57.6 | 80.3 | −11.7 | +22.7 |
| 인과 연결어율 % | 7.9 | 3.1 | 5.2 | −4.8 | +2.1 |

- **핵심**: 배제 언명은 CE가 침식(27.0→10.5)하고 SFT가 회복·강화(→28.6)한다. 동시에 SFT는
  장면 묘사를 −19.0pp 줄이면서 길이는 +22.7단어 늘린다 — **"화면에 무엇이 보이는가"에서
  "대상이 어떤 상태인가"로 서술 축이 이동**. 미래-지향율의 −37.5pp 하락은 SFT trace가 예측 어휘
  대신 상태 기술로 옮겨간 결과로, 해석 시 장면 묘사율과 함께 읽어야 한다.
- **⚠ 초판 대비**: 배제 언명률의 절대값이 전부 달라졌다(10.4→2.4→25.5 ⟹ 27.0→10.5→28.6).
  초판은 부분셋 + `elim_mention_rate`(후보 거명 ≥2개) 정의였고, 본 갱신은 full covered +
  `elim_lang_rate`(대시보드 §2 footnote b 패턴)다. **정의가 다르므로 두 계열을 섞어 인용 금지.**
  후보 거명 수(D4) 계열은 base 0.44 → θ_CE 0.19 → +SFT 0.06으로 별도 보고.
- **정성** (selection log, `records.jsonl`): base는 "가장 논리적인 다음 단계는…" 관성 추론,
  +SFT는 손·도구의 시각 단서("holding a small container while stirring", "rolling pin set aside")에서
  belief를 세워 정답으로 이끔. checkpoint별 원문 발췌 20건 = artifact §7.
  단 REGRESSION 사례(세밀 관찰이 그럴듯한 오답)도 존재.

### 2-6. egocentric 화법 [미검증 · 자매 반증]

- **이 실험**: 후보-제시 조건에서 1인칭율 전 arm 0% (측정 무의미 — 프롬프트가 3인칭 유도).
  자유생성(freegen) 패스가 없어 cand-free 1인칭율·in-support율은 **미측정**.
- **자매 JIHUN [반증]**: 자유생성에서 학습이 1인칭 침식 (base 74.0→C 31.6→Q 21.2→P 7.4%, H8 66→0).
  원인: memory_context가 비인칭 리스트라 CE가 트레이스를 그 문체로 끌어당김. → projected-SFT 화법 타깃 이관 필요.
- **리스크**: 현 상태로 "egocentric reasoning" 주장 시 트레이스 실물이 반박.

---

## 3. 종합 — 실측 분업 지도

| 능력 | candidate-CE (θ_CE) | projected-SFT (+CE replay) | 증거 |
|---|---|---|---|
| 후보 선택·정확도 | ✓ SelAcc 20.0→27.0 | 비열등 (+2.2 [−0.2,+4.7]) | §2-1 |
| 판별 (GADR) | ✓ 16.4→22.9 | ✓ **추가 상승 →26.0** (+3.0) | §2-2 |
| G1 retention | ✓ 31.4→40.5 (정점) | 소폭 ↓ 39.6 | §1-1 |
| history 사용 | ✓ strip +6.2 (DiD +5.1) | ✓ **+11.6 (DiD +5.4 유의)** | §2-3 |
| belief→action 인과 | ? 미측정 (harden 부재) | ✓ 0.291 · U_b +5.4 | §2-4 |
| reasoning 인과·유용 | ? 미측정 | ✓ both flip 80.0 · U_g +10.5 | §2-4 |
| 채널 배분 | — | ✓ **replay가 reasoning→belief로 이전** | §2-4 |
| 소거 서술 | ✗ 침식 (27.0→10.5) | ✓ 회복 (→28.6) | §2-5 |
| 장면 묘사 억제 | ✗ 거의 없음 (−1.7) | ✓ −19.0pp | §2-5 |
| egocentric 화법 | ✗ 침식 (자매) | 이관 대상 (미검증) | §2-6 |

**결론**: two-stage(Prospection=선택 정렬 / Retrospection=의미 정렬)가 **서로 다른 능력을 심는다는
정량 근거 — DPO 없이**. 다만 초판의 "acc는 CE, 추론은 SFT" 이분법은 full셋에서 수정된다:
**SFT는 정확도를 손상하지 않으면서 GADR과 이력의 인과적 사용을 늘린다.** 허용되는 최대 주장은
"CE가 정확도 도약을 만들고, SFT가 그 위에 hard-case 교정·이력 사용·belief 인과 채널을 얹는다"이다.

**논문 매핑**: CE ↔ Prospection 자리(RL→지도학습으로 철학 변경) · SFT ↔ Retrospection 자리(DPO→projected-trace SFT) ·
CE replay ρ=0.15 ↔ Method §Prospection Replay(현재 논문에 대응 ablation 없음 — §5 참조).

---

## 4. 인용·보고 규율 (자매 handoff §5 팀 규약 이월)

1. G1/GADR 분리 보고 · **L0(WM top-1) 상시 병기** — beats_L0 없으면 순효과 없음.
2. K=10 고정 · full-eq 환산(SelAcc×pool_coverage) 병기 · covered SelAcc와 구분.
3. 최종 CI는 **video-cluster bootstrap** — 본 갱신에서 정확도·history 전 지표에 적용 완료.
   개입(harden_s3)만 아직 sample bootstrap이다.
4. belief는 **G-CC1(민감도) ∧ G-CC3(방향) 쌍** 통과로만 보고 · U_b는 `utility_belief_only` 사용
   (구 own−swap_both = U_g는 reasoning 스케일 희석 — WiSE 반례가 이를 실증, §2-4 ③).
5. 용어 **interventional dependence까지만** — "causal mediation" 금지.
6. covered-only 규약은 **정적-모드 한정**(자유생성 arm은 uncovered에서도 득점).
7. **두 코호트(cesft_v2 ↔ JIHUN) 수치 직접 비교 금지** — 방향·구조만 교차 확인.
8. **★신규 — 초판(부분셋) 수치와 본 갱신 수치를 혼용 금지.** 특히 SelAcc(30.8 vs 27.0),
   GADR 방향, strip Δ(3.1 vs 11.6, 규약 상이), 배제 언명률(정의 상이)은 계열이 다르다.

---

## 5. 남은 것

- [x] ~~video-cluster bootstrap으로 최종 CI 재계산~~ — 정확도·history 완료(2026-07-26). 개입은 sample CI 유지.
- [x] ~~3 체크포인트 history strip + DiD~~ — 완료(`strip_metrics.json`).
- [ ] **base·θ_CE harden 패스(6-variant)** — 없으면 G-CC2 계산 불가. belief 채널의 **출발점**이
      자매 코호트 앵커(0.058)뿐이라 모집단 상이로 직접 비교 금지 상태. 채널 지도의 최대 공백.
- [ ] **`empty` 변형 집계 보강** — 현재 p_gt 주변 CI만 있고 paired diff·flip·acc가 없다.
      `(∅,∅)`(trace 전체 제거) 변형을 추가하면 "trace가 판단을 바꾸는가"를 Δacc 단위로 잴 수 있다.
      θ_CE는 애초에 trace 없이 학습(action span만 채점)되므로 **θ_CE ≈ 0 / +SFT > 0** 이 예측된다.
- [ ] G-DELTA 본셋 실증 — cand_free arm 학습(~2.2h+eval) 후 비교. **freegen 패스 1회로
      G-DELTA + in-support율 + cand-free 1인칭율 3종이 동시에 채워진다.**
- [ ] egocentric 화법 SFT 이관 실험 — projected-SFT 화법 타깃 명시 후 H-bin별 1인칭율 회복 검증.
- [ ] 논문 반영: **CE-replay ablation(ρ=0)이 Method §Prospection Replay에 대응 실험 없이 남아 있다.**
      §2-3 CE-replay 행이 그 자리를 채울 1순위 근거.

## 6. 근거 파일 좌표

| 무엇 | 위치 |
|---|---|
| arm 별 평가 (5 arm) | `runs/cesft_v2/eval/{base,theta_ce,sft_r15,sft_r0,wise_a050}.{json,records.jsonl}` |
| 게이트 판정 | `eval/paired_G-ACC1_theta_ce.json` · `eval/paired_G-NH_{sft_r15,sft_r0}_vs_theta_ce.json` · `eval/paired_direct_sft_r0_vs_sft_r15.json` · `eval/paired_G-DELTA_theta_ce_vs_cand_free.json`(error=SKIP) |
| history strip (3 체크포인트 + 층화) | `eval/strip_metrics.json` · `eval/{base,theta_ce,sft_r15}_nohist.records.jsonl` · 산출 `tools/strip_metrics.py` · DiD `tools/did_history.py` |
| 개입 (인과·유용, 3 arm) | `eval/{sft_r15,sft_r0,wise_a050}.harden_s3.{json,records.json}` (verdict 전부 "PASS — spine 확정 (U_g)") · 산출 `src/ego/step2_retrospection/eval/harden_s3.py` |
| 텍스트 지표 | records `reasoning` 재집계 · 정규식 SSOT `tools/trace_text_metrics.py` (파일럿과 동일 잣대) |
| 이력 길이·video_uid 조인 | `runs/cesft_v2/data/context_val.jsonl` (`history`, `video_uid`, `wm_scores`) |
| WM prior 후보 | `RETRO4-goalstep-end-m1-history-k8-phase1/best_action_top5.pt` (읽기전용, cov@10 43.43%) |
| 방법론 SSOT | `2026-07-24_dpo_free_ce_sft_methodology_handoff.md` · `2026-07-24_ce_sft_methodology_v2_handoff.md` |
| DPO 배제 배경 | `2026-07-24_s3_pivot_plan_handoff.md` (G3 붕괴·pivot) |
| 자매 실험 (다른 코호트) | EGO_jihun `2026-07-24_reasoning_quality_quantitative_evidence_handoff.md` |
| 시각 보고서 (full) | https://claude.ai/code/artifact/e4bafc93-9e5e-480c-8ef4-5adcd6a646f3 |
