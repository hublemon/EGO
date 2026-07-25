# cesft_v2 — 능력별 정량 지표·실측 근거 종합 Handoff (CE → projected-SFT, DPO 배제)

> 작성: 2026-07-25 KST · EGO_jihun3 · **실행 2026-07-24 (runs/cesft_v2, CHAIN 완료).**
> **목적: 2단계 지도학습(candidate-CE θ_CE → projected-trace SFT + CE replay)이 만드는 능력들을
> "능력 → 정량 지표(도출법) → 실측치 → 근거 로그 → 원천 학습 신호"로 완결 매핑.**
> 증거 등급: **[확정]**=paired/CI 통과 실측 · **[시사]**=실측이나 교란/검정력 한계 ·
> **[미검증]**=측정 설계만 · **[반증]**=기대와 반대로 실측.
> 자매 문서(EGO_jihun, 다른 코호트): `2026-07-24_reasoning_quality_quantitative_evidence_handoff.md`
> — 두 실험은 같은 CE→SFT 계열을 **다른 코호트·시각계약**에서 검증(방향 일치, 크기는 파이프라인차). 수치 직접 비교 금지, 방향·구조만 교차.
> 방법론 SSOT: `2026-07-24_dpo_free_ce_sft_methodology_handoff.md` · `2026-07-24_ce_sft_methodology_v2_handoff.md`
> 시각 보고서(artifact): https://claude.ai/code/artifact/922dc65e-5fd9-4b8b-b7b6-48e2b02734d2

---

## 0. 다섯 줄 요약

1. **CE = 정확도 엔진 [확정]**: candidate-CE(θ_CE)가 WM top-1 모방을 넘어 task-conditioned 선택 —
   SelAcc 20.0→**30.8%** (L0 23.7% 초과, **G-ACC1 +7.2pp[0.5,15.0] PASS**).
2. **SFT = 인과적 추론 채널 [확정]**: belief/reasoning 개입 시 행동 flip **both 81.8%** vs 대조(paraphrase)
   4.5% · 유용성 U_g **+9.8pp[7.2,12.4]** — reasoning이 장식이 아니라 행동을 인과. **개입 verdict "PASS — spine 확정 (U_g)"**.
3. **SFT의 acc 기여는 중립 [FAIL]**: G-NH +1.1pp[−5.7,+7.1] 비유의 — **acc는 CE 몫, SFT는 추론 담당**의 분업.
4. **reasoning 품질: CE가 지우고 SFT가 되살림 [확정]**: 소거 서술 10.4→CE **2.4**→SFT **25.5%** ·
   평균 단어 69→58→81 — selection log에서 "관성 추론 → 시각근거 추론" 전환 육안 확인.
5. **history 인과 [확정]**: paired strip Δ+**3.1pp[1.1,5.2]** (H8 +3.19[1.03,5.47] 유의).
   자매 JIHUN은 +12.6pp/DiD+8.4 — 두 코호트 모두 인과 확정.

**배경**: 논문 eq.11 Retrospection의 **DPO는 두 번 붕괴**(G3 문체학습 abort ×5 → margin collapse → malformed 100%)해
사용자 결정으로 배제. projected-trace SFT만으로 faithfulness 달성.

---

## 1. 능력 → 지표 → 실측 매핑 (마스터 표)

측정 기반: heldout n=1,000(제시 채점) · 개입 n=396 · WM prior = jihun2 Phase-1 K8(cov@10 43.4%, end−1s 계약).
arm: **base**(무학습) · **θ_CE**(candidate-CE) · **+SFT**(θ_CE 초기화 → projected-trace SFT + CE replay 15%).

| 능력 | 정량 지표 (도출법) | 실측 | 등급 | 원천 학습 신호 |
|---|---|---|---|---|
| **후보 선택·정확도** | SelAcc = covered 선택정확도 · G-ACC1 = SelAcc vs L0(WM top-1) | 20.0→**30.8%** (+7.2pp[0.5,15.0]) · L0 23.7 초과 | **확정** | **candidate-CE** |
| **모방 초과 판별 (GADR)** | Pr(선택=GT \| GT∈후보 ∧ WM top-1≠GT), g2 부분집합 | base 16.4→θ_CE **25.6**→SFT 24.0% | **확정** | **candidate-CE** |
| **history 사용** | paired strip Δacc = acc(hist)−acc(strip), WM후보 고정 | +**3.1pp**[1.1,5.2] · H8 +3.19[1.03,5.47] | **확정** | **candidate-CE** |
| **belief→action 인과** | belief sensitivity = flip(swap_b)−flip(para); U_g(belief-only) | flip **34.1%** · causal **0.296** · U_g +5.0pp[3.3,6.6] | **확정** | **SFT** |
| **reasoning 인과·유용** | both flip; U_g = p(GT\|own)−p(GT\|swap_both) | both flip **81.8%** · U_g **+9.8pp**[7.2,12.4] | **확정** | **SFT** |
| **소거 서술 (비교·배제)** | reasoning이 선택 외 후보 거명률 (후보-제시 생성) | base 10.4→CE **2.4**→SFT **25.5%** | **확정** | **SFT** (CE가 침식) |
| **SFT의 acc 기여** | G-NH = SelAcc(+SFT) − SelAcc(θ_CE) paired | +1.1pp[−5.7,+7.1] | **FAIL**(중립) | — (acc는 CE) |
| **간결화** | reasoning 평균 단어 수 | 69.3→CE 57.7→SFT 80.6 (CE 압축·SFT 확장) | **시사** | CE 압축 / SFT 확장 |
| **egocentric 화법** | 1인칭 검출률 `\b(I\|my\|me)\b` | 후보-제시 조건 전 arm **0%** (자유생성 미측정) | **미검증** | 자매 JIHUN: 반증(74→7%) |

**로그 좌표**: 정확도·GADR = `eval/{base,theta_ce,sft_r15}.records.jsonl`; 게이트 = `eval/paired_{G-ACC1,G-NH}_*.json`·`strip_verdict.json`;
개입 = `eval/sft_r15.harden_s3.{json,records.json}`; 텍스트 지표 = 위 records `reasoning`/`task_belief` 필드 재집계.

### 1-1. 정확도 3층위 분해 (이득의 출처)

> 판독: `eval_candidate_scored.py` — WM 후보 K=10 teacher-forcing sum-logp argmax. GT = (verb,noun) strict 일치.
> L0 = WM top-1 그대로 따르기 = 23.7%(covered).

| 지표 | base | θ_CE | +SFT | 읽는 법 |
|---|---:|---:|---:|---|
| **full acc** % (전체) | 8.7 | 12.0 | **12.2** | coverage 43.4%가 천장. ⚠ base는 covered-only 평가라 근사 |
| **SelAcc** % (covered) | 20.0 | **30.8** | **31.7** | **CE의 도약** — L0(23.7) 초과. base는 L0 아래 |
| G1 retention % (WM top-1=GT) | 31.4 | 41.2 | **48.5** | WM 맞을 때 지키기 — **SFT가 더 올림** |
| G2 correction % (GADR) | 16.4 | **25.6** | 24.0 | hard 교정 — θ_CE 정점, SFT 중립 |
| malformed / 1000 | 11 | 52 | 39 | 형식 파탄 — CE↑ SFT 일부 회복 |

**해석**: **CE가 acc 도약(SelAcc 20→30.8)을 만들고, SFT는 G1 retention(41→48.5)만 추가** — "WM이
확신하고 맞을 때 그걸 지키는" 능력. hard-case 교정(G2)은 θ_CE 정점이라 §1-2 G-NH 비유의와 정합.

### 1-2. 사전 등록 게이트 판정

| 게이트 | 기준 | 결과 | 수치 |
|---|---|---|---|
| **G-ACC1** | θ_CE SelAcc > L0(WM top-1) paired CI 하한>0 | **PASS** | Δ+7.2pp CI[0.5, 15.0] (SelAcc 30.8 vs 23.7) |
| **G-NH** | +SFT SelAcc ≥ θ_CE 비열등(하한>−1pp) | **FAIL(비유의)** | Δ+1.1pp CI[−5.7, +7.1] · GADR −1.4pp |
| **strip** | acc(hist) > acc(no-hist) paired CI 하한>0 | **PASS** | Δ+3.1pp CI[1.1, 5.2] · H8 +3.19[1.03,5.47] |
| **개입 U_g** | own reasoning p(GT) > swap_both, CI 하한>0 | **PASS** | Δ+9.8pp CI[7.2, 12.4] |
| **G-DELTA** | 후보제시 > 자유생성 | **SKIP** | cand_free arm 미학습 (본셋 실측 없음, 자매 파일럿 대체치) |

---

## 2. 능력 상세

### 2-1. 후보 선택·정확도 [확정] — CE 엔진

- **도출**: SelAcc = 정답이 WM 후보 안에 있을 때(covered)의 선택 정확도. G-ACC1은 이를 L0(WM top-1
  그대로 따르기)와 paired 비교 — LM이 WM 순위 모방을 넘어 task-conditioned 재선택을 하는지의 판정.
- **실측**: base 20.0% < L0 23.7% (안 배운 VLM은 WM보다 못함) → θ_CE **30.8%** (L0 +7.2pp, CI 하한 0.5>0).
  +SFT 31.7%로 소폭 더 오르나 G-NH 비유의.
- **의미**: candidate-CE가 정확도의 도약을 단독으로 만든다. full acc(8.7→12.0)는 coverage(43.4%)가 천장이라
  이득이 희석 — "coverage가 지배 변수" 서사와 정합.

### 2-2. 모방 초과 판별 GADR [확정]

- **도출**: GADR = Pr(선택=GT | GT∈후보 ∧ WM top-1≠GT). WM top-1이 틀린 hard-case만 채점 —
  WM 순위 복사 전략은 정의상 0점이므로 점수는 전부 맥락 기반 재선택에서 온다.
- **실측**: base 16.4 → θ_CE **25.6%** (+9.2pp). +SFT 24.0으로 소폭↓ (SFT는 판별 중립).
- **자매 교차**: JIHUN GADR base 18.0→C 41.3% (+23.3pp, n=584). 방향 일치(코호트차로 크기 다름).

### 2-3. history 사용 [확정]

- **도출**: paired history-strip — 같은 샘플에서 프롬프트의 완료-행동 이력 하나만 제거하고 재추론
  (WM 후보 고정). Δacc = acc(hist)−acc(strip). B_nohist 학습-arm 대체(같은 θ_CE, history-only 개입).
- **실측**: acc(hist) 0.120 vs acc(strip) 0.089 → Δ+**3.1pp**[1.1,5.2]. history 길이별: H8(n=878) +3.19[1.03,5.47] 유의,
  H1–7은 방향 일치·검정력 부족. `gate_history_causal_H8=True`.
- **자매 교차**: JIHUN Δ+12.6pp[10.5,14.7] · DiD(C−Q) +8.4pp[8.0,8.8] 이중 해리. 두 코호트 모두 history 인과 확정.

### 2-4. belief→action 인과 & reasoning 유용성 [확정] — SFT 채널

- **도출** (`harden_s3.py`): +SFT가 생성한 `(reasoning, belief)`를 강제 교체(swap)하고 K=10 선택 flip 측정.
  핵심 통제군 = **paraphrase**(같은 뜻 재서술)가 유발하는 flip(문체 노이즈)을 빼 "의미 변화로 바뀐 정도"만 남김.
  paired bootstrap CI, donor 결정적 선정.
- **실측** (n=396):

  | 채널 | flip율 | causal sensitivity | 유용성 Δp(GT) |
  |---|---:|---:|---:|
  | belief만 swap | 34.1% | 0.296 | **+5.0pp**[3.3,6.6] (belief-only U_g) |
  | reasoning만 swap | 33.1% | 0.285 | — |
  | **both** swap | **81.8%** | **0.773** | **+9.8pp**[7.2,12.4] |
  | paraphrase (대조) | 4.5% | — | — |

- **핵심**: belief·reasoning 어느 하나가 인과를 독점하지 않음(34.1≈33.1). 둘을 함께 바꾸면 **초가법**
  (34+33=67 < 82) — 결합해 행동을 결정. belief 단독 유용성 +5.0pp 유의 → belief→action 인과 경로 실재.
  gate `S3a_causal_real`·`S3b_utility_real`·`CC1_belief_sensitivity`·`CC3_belief_only_utility` **전부 true**,
  verdict **"PASS — spine 확정 (U_g)"**. directional_dg 0.434.
- **자매 교차**: JIHUN belief sensitivity base 0.058→SFT 0.390(6.7×), U_g 0.023→0.042. 방향 일치.
- **용어 규율**: "interventional dependence"까지만 — "causal mediation" 금지.

### 2-5. reasoning 품질 (소거 서술) [확정] — CE가 지우고 SFT가 되살림

- **도출**: 후보-제시 프롬프트 생성 trace에서 선택 외 후보를 비교·배제 언급한 비율(비교추론 프록시), n≈960.
- **실측**: base 10.4 → θ_CE **2.4**(CE가 침식) → +SFT **25.5%**(되살림+강화). 평균 단어 69→58→81.
- **정성** (selection log, `records.jsonl`): base는 "가장 논리적인 다음 단계는…" 관성 추론,
  +SFT는 손·도구의 시각 단서("holding a small container while stirring", "rolling pin set aside")에서
  belief를 세워 정답으로 이끔. 성공 4케이스(add rice·wash ingredient·stir·roll dough) 전문 = artifact §3.
  단 REGRESSION 사례(세밀 관찰이 그럴듯한 오답)도 존재 → G-NH 비유의(acc 중립)와 정합.

### 2-6. egocentric 화법 [미검증 · 자매 반증]

- **이 실험**: 후보-제시 조건에서 1인칭율 전 arm 0% (측정 무의미 — 프롬프트가 3인칭 유도).
- **자매 JIHUN [반증]**: 자유생성에서 학습이 1인칭 침식 (base 74.0→C 31.6→Q 21.2→P 7.4%, H8 66→0).
  원인: memory_context가 비인칭 리스트라 CE가 트레이스를 그 문체로 끌어당김. → projected-SFT 화법 타깃 이관 필요.
- **리스크**: 현 상태로 "egocentric reasoning" 주장 시 트레이스 실물이 반박.

---

## 3. 종합 — 실측 분업 지도

| 능력 | candidate-CE (θ_CE) | projected-SFT | 증거 |
|---|---|---|---|
| 후보 선택·정확도 | ✓ SelAcc 20→30.8 | 중립 (+1.1 n.s.) | §2-1 |
| 판별 (GADR) | ✓ G2 16.4→25.6 | 중립 | §2-2 |
| history 사용 | ✓ strip +3.1 · 자매 DiD +8.4 | — | §2-3 |
| belief→action 인과 | ✗ 미형성 | ✓ 0.296 · U_g↑ · 자매 6.7× | §2-4 |
| reasoning 인과·유용 | — | ✓ both flip 81.8 · U_g +9.8 | §2-4 |
| 소거 서술 | ✗ 침식 (10.4→2.4) | ✓ 회복 (→25.5) | §2-5 |
| egocentric 화법 | ✗ 침식 (자매) | 이관 대상 (미검증) | §2-6 |

**결론**: two-stage(Prospection=선택 정렬 / Retrospection=의미 정렬)가 **서로 다른 능력을 심는다는
정량 근거 — DPO 없이**. acc는 CE, "검증가능한 reasoning에의 정렬"은 SFT. 이 프로젝트에서 처음으로
한 정책이 정확도 + 인과적 추론 두 채널을 동시 보유.

**논문 매핑**: CE ↔ Prospection 자리(RL→지도학습으로 철학 변경) · SFT ↔ Retrospection 자리(DPO→projected-trace SFT).

---

## 4. 인용·보고 규율 (자매 handoff §5 팀 규약 이월)

1. G1/GADR 분리 보고 · **L0(WM top-1) 상시 병기** — beats_L0 없으면 순효과 없음.
2. K=10 고정 · full-set 환산(acc×pool_coverage) 병기 · covered SelAcc와 구분.
3. 최종 CI는 **video-cluster bootstrap** 권장(같은 비디오 프레임 상관 → sample-CI 과소분산).
4. belief는 **G-CC1(민감도) ∧ G-CC3(방향) 쌍** 통과로만 보고 · U_g는 `utility_belief_only` 사용
   (구 own−swap_both는 reasoning 스케일 희석).
5. 용어 **interventional dependence까지만** — "causal mediation" 금지.
6. covered-only 규약은 **정적-모드 한정**(자유생성 arm은 uncovered에서도 득점).
7. **두 코호트(cesft_v2 ↔ JIHUN) 수치 직접 비교 금지** — 방향·구조만 교차 확인.

---

## 5. 남은 것

- [ ] G-DELTA 본셋 실증 — cand_free arm 학습(~2.2h+eval) 후 full-set 비교 (현재 SKIP, 자매 대체치만).
- [ ] egocentric 화법 SFT 이관 실험 — projected-SFT 화법 타깃 명시 후 H-bin별 1인칭율 회복 검증.
- [ ] full 학습 이동 추산 검증 (자매 §3-4: full +3~8pp vs strict pre-onset 전환 손실 −5~10pp 상쇄).
- [ ] video-cluster bootstrap으로 최종 CI 재계산 (현 sample-CI는 과소분산).

## 6. 근거 파일 좌표

| 무엇 | 위치 |
|---|---|
| arm 별 평가 (records 포함) | `runs/cesft_v2/eval/{base,theta_ce,sft_r15}.{json,records.jsonl}` |
| 게이트 판정 | `eval/paired_{G-ACC1,G-DELTA,G-NH}_*.json` · `eval/strip_verdict.json` |
| 개입 (인과·유용) | `eval/sft_r15.harden_s3.{json,records.json}` (verdict "PASS — spine 확정 (U_g)") |
| 텍스트 지표 | 위 records `reasoning`/`task_belief` 재집계 |
| WM prior 후보 | `RETRO4-goalstep-end-m1-history-k8-phase1/best_action_top5.pt` (읽기전용, cov@10 43.4%) |
| 방법론 SSOT | `2026-07-24_dpo_free_ce_sft_methodology_handoff.md` · `2026-07-24_ce_sft_methodology_v2_handoff.md` |
| DPO 배제 배경 | `2026-07-24_s3_pivot_plan_handoff.md` (G3 붕괴·pivot) |
| 자매 실험 (다른 코호트) | EGO_jihun `2026-07-24_reasoning_quality_quantitative_evidence_handoff.md` |
| 시각 보고서 (완전판) | https://claude.ai/code/artifact/922dc65e-5fd9-4b8b-b7b6-48e2b02734d2 |
