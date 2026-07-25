# 논문 제시용 능력 → 정량 지표 교차-코호트 종합 Handoff (최신화)

> 작성: 2026-07-25 KST · EGO_jihun3. **목적: 두 코호트(cesft_v2 본셋 · EGO_jihun 파일럿)의
> "정성 능력 → 정량 지표 → 실측치 → 등급 → 근거 로그"를 하나의 표로 통합하고,
> 2026-07-24 야간 산출물(sft_r0 ablation · WiSE · Frontier VPA · GADR 귀인 · G-DELTA 재계산)로 등급을 갱신한다.**
>
> 증거 등급: **[확정]**=paired/CI 통과 실측 · **[시사]**=실측이나 교란/검정력 한계 ·
> **[미검증]**=측정 설계만 · **[반증]**=기대와 반대로 실측 · **[정정]**=이전 handoff 판정 변경.
>
> **이 문서가 갱신하는 기존 handoff** (§4에 행 단위 대조):
> · EGO_jihun `2026-07-24_reasoning_quality_quantitative_evidence_handoff.md` (§1 마스터 표)
> · EGO_jihun `2026-07-24_history_strip_ablation_results_handoff.md` (§2 결과)
> · EGO_jihun3 `2026-07-25_cesft_v2_quantitative_evidence_handoff.md` (§1 마스터 표)
>
> 방법론 SSOT: `2026-07-25_cesft_v2_paper_methodology_final_handoff.md` (변경 없음)

---

## 0. 다섯 줄 요약

1. **[신규·확정] CE-replay가 방법론 부품으로 정당화됨** — replay 없는 `sft_r0`(ρ=0)에서 GADR이
   **−6.5pp CI[−12.4, −0.8] 유의 감소**(26.6→20.1%, base 16.4 수준으로 회귀). ρ=0.15는 −1.4pp[−6.4,+3.6] 비유의.
   **전체 r0/r15 비교에서 유의성에 도달한 유일한 셀.** 무너지는 것은 hard-case 판별(GADR)뿐이고
   G1 retention은 오히려 유지(41.2→44.3) — **손상 지점이 특정된다.** 인과 채널(U_g)은 r0도 PASS → **replay와 직교.**
2. **[정정·중대] G-NH FAIL의 해석이 확정됨** — "SFT는 정확도에 무관"(중립설)이 아니라
   **"SFT는 판별을 덮어쓰고, CE-replay가 그것을 막는다"**. r0 대조 없이는 두 해석을 가를 수 없었다.
3. **[확정] 인과 추론 채널은 두 코호트 공통** — 본셋 개입 U_g +9.8pp[7.2,12.4]·both flip 81.8% vs para 4.5%;
   history 인과는 본셋 +3.1pp[1.1,5.2]·파일럿 +12.6pp[10.5,14.7]·DiD +8.4pp[8.0,8.8].
4. **[정정] 두 건 강등** — ⑴ G-DELTA(후보제시>자유생성): 표본 정합 시 **+1.2pp CI[−5.2,+6.7] 비유의**
   (기존 +2.4pp/+19.2pp 폐기). ⑵ GADR 귀인: hard-case 중심(8.2)이 아니라 **retention·GADR 동반 상승(8.1)**.
5. **[신규] Frontier(gemini-2.5-pro)** — VPA T3 mAcc 3.07%·T4 3.19%, SR≈0 (**보고 가능**).
   select는 972/1520(64%) API 실패 + 성공분이 **32개 video 중 8개만 덮는 연속 prefix**라
   **저장 집계·유효분 모두 보고 불가**(§2-7). 재시도 후 재집계가 유일한 경로. text-only라 직접 비교도 불가.

---

## 1. 능력 → 지표 → 실측 마스터 표 (교차 코호트)

**코호트 정의 — 수치 직접 비교 금지, 방향·구조만 교차**

| | **본셋 (cesft_v2)** | **자매 파일럿 (EGO_jihun)** |
|---|---|---|
| 과제·시간 계약 | end−1s 관측 → strict-next A3, 평균 12.8s 미래 | end−1s 관측 → current action (인식 성향) |
| 관측 | 8프레임@336 | 1프레임 |
| WM prior | jihun2 Phase-1 K8, cov@10 **43.4%** | V-JEPA2 probe 계열, cov@10 **49.2%** |
| arm | base · θ_CE · sft_r0(ρ=0) · sft_r15(ρ=.15) · wise_a050 | base · C/S(cand-CE) · Q(GT-CE) · P(WM 증류) |
| eval | 제시 채점 n=1000 · 개입 n=396 | 제시 n=1520 · 자유생성 n=500 |

| 능력 | 정량 지표 (도출법) | 본셋 실측 | 파일럿 실측 | 등급 | 원천 신호 |
|---|---|---|---|---|---|
| **후보 선택·정확도** | SelAcc(covered) · G-ACC1 = SelAcc vs L0(WM top-1) | 20.0→**30.8%** (L0 23.7, **+7.2pp[0.46,15.0] PASS**) | 21.1→**41.4%** (C, n=1520) | **확정** | candidate-CE |
| **모방 초과 판별 (GADR)** | Pr(선택=GT ǀ GT∈후보 ∧ WM top-1≠GT) | 16.4→**25.6**→(r0 **19.2**)→(r15 24.0)% | 18.5→**37.0%** (C) | **확정** | candidate-CE |
| **판별의 취약성 (신규)** | GADR(sft_rρ) − GADR(θ_CE) paired, ρ=0 vs 0.15 | ρ=0 **−6.5pp[−12.4,−0.8] 유의** vs ρ=.15 −1.4pp[−6.4,+3.6] 비유의 | — | **확정** | **CE-replay** |
| **history 사용** | paired strip Δacc (WM 후보 고정) | +**3.1pp**[1.1,5.2] · H8 +3.19[1.03,5.47] | C +**12.6pp**[10.5,14.7] · H8 +15.1 · **DiD(C−Q) +8.4pp[8.0,8.8]** | **확정** | candidate-CE |
| **국면 상보성 (이중 해리)** | 무-history WM vs LM · history 축적 시 역전 | 미측정 | 합성H0: WM 25.1 vs LM 20.8 (**+4.2pp[0.2,8.3]**) / hist: LM 48.4 vs WM 25.1 | **확정** | 구조 |
| **belief→action 인과** | belief sensitivity = flip(swap_b)−flip(para) · U_g(belief-only) | **0.296**[.248,.343] · U_g +**5.0pp**[3.3,6.6] | 0.058→**0.390**[.358,.421] · U_g .023→.042 | **확정** | **SFT** |
| **reasoning 인과·유용** | both flip vs para · U_g = p(GTǀown)−p(GTǀswap_both) | both **81.8%** vs para **4.5%** · U_g +**9.8pp**[7.2,12.4] | (본셋 위임) | **확정** | **SFT** |
| **방향성** | D_g · correct-switch | D_g **0.434**[.384,.482] · flip_swap_b 34.1%(n=135) | — | **확정** | SFT |
| **후보 내재화 (창발)** | in_support = 자유생성 ∈ 비제시 WM top-K | 미측정 | base 28.2→**Q 47.6 · P 49.6%** (+**21.4pp**[17.2,25.8]) | **확정** | GT-CE (WM 증류 불필요, I-1 FAIL) |
| **소거 서술** | reasoning이 선택 외 후보 거명률 | 10.4→CE **2.4**→SFT **25.5%** | 제시조건 94.4→86.4% · n_mentioned 4.68→4.00 | **확정** | SFT (CE가 침식) |
| **간결화 (창발)** | reasoning 평균 단어 수 | 69.3→CE 57.7→SFT 80.6 | 80.8→C 69.0→Q 64.2→P 65.3 | **시사** | CE 압축 |
| **SFT의 acc 기여** | G-NH = SelAcc(+SFT) − SelAcc(θ_CE) paired | +1.1pp[−5.7,+7.1] | — | **FAIL(중립)** | — (§2-2 해석 주의) |
| **성립부등식 (G-DELTA)** | 후보제시 vs 자유생성 | **SKIP** (cand_free 미학습) | matched 500: **+1.2pp[−5.2,+6.7] 비유의** | **[정정] 미확립** | — |
| **egocentric 화법** | 1인칭 검출률 `\b(I\|my\|me)\b` | **전 arm 0%** (템플릿이 3인칭 고정) | 비제시 74.0→31.6/21.2/7.4% · 제시 52.4→**61.4%**[3.3,14.6] | **[정정] 지표 무효** | — (§2-6) |

### 1-1. GADR과 SelAcc의 관계 — 읽는 법 (혼동 방지)

`GADR < SelAcc` 는 오류가 아니라 **정의상 정상**이다. covered 표본은 두 층으로 쪼개진다:

| 층 | 정의 | 본셋 비중 | θ_CE 정확도 |
|---|---|---:|---:|
| **G1** (easy) | WM top-1 == GT | 97/410 = **23.7%** | 41.2% (retention) |
| **G2** (hard) | GT ∈ 후보 ∧ WM top-1 ≠ GT | 313/410 = **76.3%** | 25.6% (**GADR**) |

SelAcc는 둘의 가중평균 → (97×41.2 + 313×25.6)/410 = **29.3%**. 즉 항상 `GADR < SelAcc < retention`.
GADR은 **어려운 층에만 조건부로 잰 정확도**이므로 전체 SelAcc보다 낮아야 정상이다
(GADR > SelAcc면 오히려 이상 — 어려운 층을 쉬운 층보다 잘 맞춘다는 뜻).

**부수 함의**: G2가 covered의 76%를 차지하므로 **본셋 SelAcc는 사실상 GADR이 지배**한다
(파일럿도 281/379 = 74%로 동일 구조). SelAcc 개선을 보고할 때 GADR 분해를 반드시 병기할 것.

### 1-2. GADR 수치가 문서마다 다른 이유 — top-K 컷 차이 + 코호트 차이

파일럿에 GADR 값이 **두 개 공존**한다. 정의(top-K 컷)가 달라서이며 둘 다 맞다:

| 출처 | G2 정의 | n | base → C |
|---|---|---:|---|
| jihun `reasoning_quality` §1 | GT ∈ top-**10** ∧ wm1≠GT | 584 | 18.0 → **41.3%** |
| `GADR_ATTRIBUTION.json` | GT ∈ top-**5** ∧ wm1≠GT | 281 | 18.5 → **37.0%** |
| **본셋 cesft_v2** | GT ∈ top-**10** ∧ wm1≠GT | 313 | 16.4 → **25.6%** |

**짝이 맞는 비교는 41.3% (파일럿) ↔ 25.6% (본셋)** 이며, 이득은 **+23.3pp → +9.2pp 로 약 60% 축소**됐다.
G-ACC1이 +19.7pp → +7.2pp 로 1/3 축소된 것과 **같은 패턴**이고, 원인도 동일하다
(`2026-07-24_overnight_frontier_and_pilot_reanalysis_handoff` §4):
⑴ 과제 난이도 — 현재행동 인식 → 평균 12.8s 미래 예측, ⑵ WM coverage 51.4 → 41%, ⑶ 표본 1/4.

**논문 반영 규칙**: 확정셋은 **본셋 25.6%**. 파일럿 41.3%를 본문 수치로 쓰지 말 것(코호트 직접 비교 금지).
다행히 `main.tex`의 GADR 수치는 현재 전부 *illustrative* 표기(29.5/34.2/31.8 등)라 파일럿 값이 박혀 있지 않다.
**GADR의 논리적 지위(§main.tex L249 "모방으로는 점수를 얻을 수 없다")는 효과 크기와 무관하게 유효**하나,
효과가 파일럿의 절반이므로 **서술 톤을 낮출 것**.

**로그 좌표**: 본셋 `runs/cesft_v2/eval/{base,theta_ce,sft_r0,sft_r15,wise_a050}.{json,records.jsonl}` ·
`eval/paired_{G-ACC1,G-NH}_*.json` · `eval/strip_verdict.json` · `eval/sft_r15.harden_s3.json`;
파일럿 `EGO_jihun/runs/goalstep_v3_boundary/eval/{HSTRIP_VERDICT,I_VERDICT,GADR_ATTRIBUTION,G_DELTA_FULLSET_VERDICT,TRACE_TEXT_METRICS}.json`.

---

## 2. 신규·정정 항목 상세

### 2-1. [신규·확정] CE-replay 필요성 — sft_r0 ablation (2026-07-25 05:24 완주)

동일 θ_CE에서 출발해 `--ce_replay_rho`만 0.0 vs 0.15로 갈린 쌍둥이. 학습 설정(2945 샘플·1 epoch·τ=1.0) 동일.
평가 subset 동일(covered n=410, G1 n=97, G2 n=313) — **arm 간 직접 비교 가능**.

| arm | full acc | G1 retention | **GADR (G2)** | malformed |
|---|---:|---:|---:|---:|
| base | 8.7%* | 31.4% | 16.4% | 1.1% |
| **θ_CE** | **12.0%** | 41.2% | **25.6%** | 5.2% |
| **sft_r0** (ρ=0) | **10.3%** | 44.3% | **19.2%** | 0.9% |
| **sft_r15** (ρ=.15) | **12.2%** | **48.5%** | 24.0% | 3.9% |
| wise_a050 (α=.5) | 11.6% | 42.3% | 24.0% | 5.7% |

\* base는 covered-only 평가라 full 환산 근사(다른 denominator).

**판정 3가지:**
1. **replay는 장식이 아니다** — 없으면 GADR이 25.6→19.2%로 무너지고, 이는 base(16.4)에 근접한다.
   θ_CE가 벌어놓은 판별 이득의 **약 70%가 소실**된다. ρ=0.15면 손실이 −1.6pp에 그친다.
2. **손상 지점이 특정된다** — G1 retention은 r0에서 오히려 **오른다**(41.2→44.3). 즉 SFT가 덮어쓰는 것은
   "WM이 맞을 때 지키기"가 아니라 **"WM이 틀렸을 때 교정하기"**다. 방법론 서술에 그대로 쓸 수 있는 국소화다.
3. **부수 관찰** — malformed는 r0가 0.9%로 가장 깨끗하다(θ_CE 5.2%). CE 신호가 형식 파탄의 원인이며
   SFT trace가 형식을 되돌린다는 §2-5와 정합. **단 형식 회복과 판별 보존은 상충** — replay를 넣으면
   malformed가 0.9→3.9%로 오른다. 논문에 트레이드오프로 명기 가능.

**paired 게이트 (video-cluster bootstrap, 76 클러스터) — 2026-07-25 05:49 확정**

| 비교 (vs θ_CE) | 지표 | θ_CE → arm | Δ | CI95 | 유의 |
|---|---|---|---:|---|---|
| **sft_r0** (ρ=0, n=384) | SelAcc | 30.73 → 26.56% | −4.17pp | [−9.67, +0.91] | 아니오 |
| **sft_r0** (ρ=0, n=384) | **GADR** | 26.62 → **20.14%** | **−6.48pp** | **[−12.40, −0.76]** | **예** |
| sft_r15 (ρ=.15, n=378) | SelAcc | 30.69 → 31.75% | +1.06pp | [−5.70, +7.06] | 아니오 |
| sft_r15 (ρ=.15, n=378) | GADR | 26.74 → 25.35% | −1.39pp | [−6.43, +3.57] | 아니오 |

**전체 r0/r15 비교에서 유의성에 도달한 유일한 셀이 "ρ=0의 GADR 손실"이다.** 이것이 CE-replay를
방법론 부품으로 정당화하는 실측 근거다.

> ⚠ **통계적 주의 (반드시 준수)**: 위는 각각 θ_CE 대비 **독립 검정** 2개이며, r0와 r15를 **직접 paired
> 비교하지 않았다.** 두 CI가 겹치므로(−12.4~−0.8 vs −6.4~+3.6) **"replay가 유의하게 더 낫다"고 쓸 수 없다.**
> 쓸 수 있는 문장은 **"ρ=0에서는 유의한 GADR 손실이 관측되고, ρ=0.15에서는 관측되지 않는다"**까지다
> (difference of significance ≠ significance of difference). 직접 주장이 필요하면 r0 vs r15 paired 검정 추가.

**harden (개입, IV_N=400) — 인과 채널은 replay와 무관하게 형성된다**

| 지표 | sft_r0 (ρ=0) | sft_r15 (ρ=.15) |
|---|---|---|
| causal sensitivity (both) | 0.763 [0.720, 0.808] | 0.773 [0.730, 0.813] |
| U_g (own − swap_both) | +6.2pp [4.2, 8.3] | +9.8pp [7.2, 12.4] |
| U_g belief-only | +4.0pp [2.8, 5.5] | +5.0pp [3.3, 6.6] |
| D_g (방향성) | 0.438 [0.390, 0.485] | 0.434 [0.384, 0.482] |
| verdict | **PASS — spine 확정 (U_g)** | **PASS — spine 확정 (U_g)** |

**r0도 전 게이트 PASS**(S3a·S3b·CC1·CC3 모두 true). U_g 점추정은 r0가 낮으나 **CI가 겹쳐 차이는 비유의**.
→ **projected-trace SFT가 인과 채널을 심는 것은 CE-replay와 무관**하다. 두 신호가 담당 능력에서 **직교**한다:
**replay = 판별(GADR) 보존 · trace = 인과(U_g) 형성.** 이것이 §3 분업 지도의 실측 기반이다.

### 2-2. [정정·중대] G-NH FAIL 해석의 확정

기존 서술(`2026-07-25_cesft_v2_quantitative_evidence_handoff` §0-3)은 G-NH +1.1pp[−5.7,+7.1]을
**"acc는 CE 몫, SFT는 추론 담당의 분업"**으로 읽었다. 이 해석은 두 가지 중 하나였고 데이터로 못 갈랐다:

- **(A) 중립설**: SFT는 애초에 정확도에 영향이 없다 → replay는 불필요
- **(B) 방어설**: SFT는 판별을 덮어쓰는데 replay가 막았다 → replay는 필수 부품

**r0 결과가 (B)를 지지한다.** replay를 빼자 GADR이 실제로 무너졌으므로, r15의 "중립"은 무개입의 결과가
아니라 **개입(replay)의 성과**다. 논문 서술을 다음으로 교체할 것:

> ~~"SFT는 정확도에 중립이며 추론만 담당한다"~~
> → **"SFT는 판별 능력을 침식하는 경향이 있으며(r0: GADR −6.4pp), CE-replay 앵커가 이를 상쇄해
>    판별을 보존한 채 인과적 추론 채널을 추가한다(r15: GADR −1.6pp, U_g +9.8pp)."**

**여전히 주의**: G-NH CI 폭이 12.8pp(하한 −5.7pp)로 넓어 "비열등성"을 **정식으로 주장할 수는 없다.**
검정력 부족이지 중립의 증명이 아니다. §6-1 재측정 권고.

### 2-3. [신규] WiSE-FT α=0.5 — 학습 0의 frontier 점

merge만으로 얻은 보간 어댑터(θ_CE ⊕ sft_r15, α=0.5). 서버 B에서 2026-07-25 04:12/04:22 완주.

- acc 11.6%(r15 12.2 대비 −0.6pp) · G1 42.3% · GADR 24.0%(r15과 동일)
- 개입: **U_g +12.4pp[9.1, 15.9]** — **r15(+9.8pp[7.2,12.4])보다 높다** · causal both 0.766 · verdict PASS

**함의**: 정확도를 0.6pp 내주고 reasoning 유용성을 2.6pp 얻는 지점이 존재한다. 학습 비용 0이므로
논문 부록의 "정확도–인과성 frontier" 한 점으로 제시 가능. 단 α 단일 점이라 곡선 주장은 불가.

### 2-4. [정정] G-DELTA — 성립부등식은 표본 정합 시 유의하지 않다

`G_DELTA_FULLSET_VERDICT.json`(matched 500 sample_id, video-cluster 31개 paired bootstrap):

| 비교 | Δ | CI95 | 판정 |
|---|---:|---|---|
| C-presented − Q-freegen (시스템 비교, 주 지표) | **+1.2pp** | [−5.2, +6.7] | **비유의** |
| C-presented − C-freegen (동일 체크포인트 배포효과) | 0.0pp | — | 이득 없음 |
| Q-presented − Q-freegen | −7.2pp | [−10.8, −3.8] | **후보 제시가 Q를 해침** |

기존 "full +2.4pp"는 C를 1520개·Q를 다른 500개로 잰 **표본 불일치 산물**, "covered-only +19.2pp"는
uncovered=0점 규약을 자유생성 arm에 오적용한 **편향치 — 인용 금지**.

**단, 살아 있는 대안 근거 하나**: 같은 제시 조건에서 **C 41.4% vs Q(GT-CE) 26.9%** (SelAcc covered,
n=1520 동일 표본, `GADR_ATTRIBUTION.json`). "후보를 보여주는 것"이 아니라 **"후보 위에서 학습하는 것"**의
효과는 크고 표본 정합적이다. 논문의 정당화 축을 *제시 여부*에서 ***학습 신호***로 옮기면 근거가 선다.

### 2-5. [정정] GADR 귀인 — 8.2가 아니라 8.1

`GADR_ATTRIBUTION.json`(WM 고정 cell 내 correct rate 분해, base→C):

- retention(G1) 28.57 → 54.08% (**+25.5pp**) · GADR(G2) 18.51 → 37.01% (**+18.5pp**) · 비율 **0.73 (<1)**
- → hard-case 교정만 크게 오른 게 아니라 **retention·GADR 동반 상승**

**논문 수정 지시**: §er-next/ablation에서 GADR을 "모방 불가한 hard-case 교정 능력"으로만 프레이밍하면
과장. "retention·correction 동반 상승"이 정직한 서술. (GADR>0·base 대비 상승 자체는 유효 — 방향만 조정.)

### 2-6. [정정·강등] egocentric 화법 — 지표 자체를 폐기 권고

본셋 실측(2026-07-25, `eval/*.records.jsonl` 재집계, 후보-제시 조건 n≈1000):

| arm | 1인칭 | 3인칭 |
|---|---:|---:|
| base(무학습) · θ_CE · sft_r15 | **0.0% / 0.0% / 0.0%** | 99.8 / 100 / 100% |

trace 실물: *"**The person** is holding a spoon and a small container…"*

**결정적 문제**: **무학습 base**의 1인칭율이 템플릿에 따라 **74.0 / 52.4 / 0.0%**로 갈린다. 학습을 안 한
모델이므로 이 차이는 전부 프롬프트 템플릿 효과다. 이 지표는 능력이 아니라 **표면형**을 재고 있다.
또 같은 "후보 제시" 레짐에서 파일럿 C는 61.4%, 본셋 candidate-CE는 0%로 **정면 모순**한다.

**판정**: 파일럿 §7의 "레짐-종속" 정정은 유효하나, 여기서 더 나아가 **논문 본문에서 1인칭율을 능력
지표로 쓰지 말 것.** "egocentric reasoning" 주장은 근거가 없다. 창발 서사가 필요하면 **in_support
후보 내재화(+21.4pp[17.2,25.8])**를 쓸 것 — 행동 어휘 집합 소속 여부라 템플릿에 불변하고 CI가 강하다.

### 2-7. [신규] Frontier (gemini-2.5-pro, text-only 게이트웨이)

| Track | n | 결과 |
|---|---:|---|
| VPA T3 (test) | 1042 | SR 0.10% · mAcc **3.07%**[2.37,3.71] · mIoU **4.02%**[3.34,4.70] |
| VPA T4 (test) | 988 | SR **0.0%** · mAcc **3.19%**[2.56,3.85] · mIoU **4.57%**[3.93,5.30] |
| candidate-scored select | 1520 | **972건(64%) API 429 실패** → 저장 집계(full 6.3%/SelAcc 16.9%) **무효** |

**select 수치 — 어느 것도 현재 보고 불가 (2건 모두 결함)**

| | 분모 | Acc_full | SelAcc(cov,**top5**) | G1 retention | GADR(G2) |
|---|---|---:|---:|---:|---:|
| 저장 집계 `frontier_select.json` | 1520 (실패=오답) | 6.32% | 16.88% (n=545) | 37.88% | 4.90% |
| 유효 548건만 | 548 | 17.52% | **31.19%** (92/295) | **60.98%** (75/123) | **9.88%** (17/172) |

⚠ **top-5 / top-10 혼동 주의**: 저장 필드명이 `SelAcc_covered_top5`다. 유효분의 top-10 기준값은
25.13%(96/382)로 다르다. **논문 규약은 top-5** — 비교 시 반드시 컷을 맞출 것.

⚠ **유효 548건은 무작위 표본이 아니다**: 429 실패가 파일 순서 ~560번째부터 전량 발생해, 성공분은
**파일 위치 0–547의 연속 prefix**다. 결과적으로 **전체 32개 video 중 8개(25%)만** 포함되고
H8 비율도 80.7% vs 전체 73.5%로 치우친다. video-cluster bootstrap을 하면 **클러스터 8개**뿐이라
CI가 무의미하게 넓다. → **저장 집계도 유효분도 논문 표에 넣지 말 것. 972건 재시도 후 재집계가 유일한 경로.**

**[시사] 다만 정성적 프로파일은 주목할 만하다** (같은 데이터셋 v2 heldout·top-5 기준 대조):

| arm | SelAcc(cov,top5) | G1 retention | GADR |
|---|---:|---:|---:|
| base (무학습) | 21.11% | 28.57% | 18.51% |
| **Frontier (548 부분집합)** | 31.19% | **60.98% ← 최고** | **9.88% ← 최저** |
| Q (GT-CE) | 26.91% | 42.86% | 21.35% |
| **C (candidate-CE)** | **41.42%** | 54.08% | **37.01%** |

Frontier는 **retention 최고 · GADR 최저** — WM top-1이 맞을 때는 가장 잘 따라가고, WM이 틀렸을 때는
무학습 base보다도 못 고친다. 이는 **WM 추종(모방)의 전형적 시그니처**이며, `main.tex` L249의
"GADR은 WM이 틀린 구간만 재므로 모방으로는 점수를 얻을 수 없다"는 논증의 **실증 사례**가 될 수 있다.
**단 클러스터 8개 기반이므로 [시사]에 머문다** — 재시도 완료 후 [확정] 승격 여부 재판정.

**[정정] "이미지 미열람"의 실제 원인 — 게이트웨이가 아니라 하네스 설계**

기존 handoff는 "게이트웨이 모델이 text-only(vision 미지원)"라고 적었으나 **부정확하다.** 실측 확인 결과:

1. `run_frontier_baseline.py` docstring L1–3: *"**Text-conditioned VPA**: each sample's goal + observed step
   history + the candidate vocabulary are sent as a prompt"* — **이미지 코드 경로가 아예 없다.**
   `messages`는 text content만 싣는다. 게이트웨이 vision 지원 여부와 무관하게 **보낼 수가 없다.**
2. **우리 로컬 baseline도 동일하게 text-only** — `run_qwen_baseline.py` L4–6: *"**Same text-conditioned VPA
   prompt and output format as the frontier baseline.** The history is text-only (goal + observed step labels);
   **a frame-input hook is left as a commented stub**"*. L69: `# FRAME HOOK: to add video later…`
3. **VPA 데이터 스키마에 프레임 접근 정보가 없다** — 필드는 `sample_id · video_uid · goal_text ·
   observed_steps · future_steps · horizon · eval_split`뿐. **관측 구간 타임스탬프가 없어** 프레임을 넣으려면
   GoalStep 원본 어노테이션에서 `observed_steps ↔ 시각` 매핑을 다시 붙여야 한다.
4. 게이트웨이의 vision 지원 여부는 **테스트된 적 없다**(로그에 흔적 없음). 검증되지 않은 주장이었다.

**두 트랙의 공정성이 다르다 — 구분해서 각주할 것:**

| 트랙 | 우리 arm | frontier | 공정성 |
|---|---|---|---|
| **Track A · VPA(planning)** | Qwen baseline도 **text-only** | text-only | **대등 — 직접 비교 가능** |
| Track B · candidate-scored select | θ_CE는 **8프레임 관측** | text-only | **불공정 — 직접 비교 불가** |

→ VPA 표에 "frontier가 이미지를 못 봐서 낮다"는 각주를 달면 **틀린 변명**이 된다. 양쪽 다 못 봤다.
낮은 이유는 §2-7의 어휘 미매칭(57.8%/55.5%)과 strict 채점이지 modality 격차가 아니다.

> ⚠ **논문 서술 불일치 (조치 필요)**: `main.tex` L309는 *"predicting the next three actions from the
> previous three actions **and the video**"*라고 쓰여 있으나 **구현은 video를 쓰지 않는다.**
> 문구를 실제 구현(text-conditioned)에 맞추거나, 프레임 훅을 구현해 재실행할 것. 둘 중 하나는 반드시.

---

## 3. 능력 분업 지도 (r0 반영 최종본)

| 능력 | candidate-CE | CE-replay 앵커 | projected-SFT |
|---|---|---|---|
| 후보 선택·정확도 (SelAcc) | ✓ 20→30.8 | 보존 | 중립 |
| **hard-case 판별 (GADR)** | ✓ 16.4→25.6 | **✓ 보존 (없으면 19.2로 붕괴)** | ✗ 침식 |
| easy retention (G1) | ✓ 31.4→41.2 | — | ✓ 41.2→48.5 |
| history 사용 (strip) | ✓ +3.1 · 자매 DiD +8.4 | — | — |
| belief→action 인과 | ✗ 미형성 | **무관** (r0도 PASS) | ✓ 0.296 · 자매 6.7× |
| reasoning 인과·유용 | — | **무관** (r0 U_g +6.2[4.2,8.3] PASS) | ✓ flip 81.8 · U_g +9.8 |
| 소거 서술 | ✗ 침식 10.4→2.4 | — | ✓ 회복 →25.5 |
| 형식 안정 (malformed) | ✗ 1.1→5.2% | ✗ 0.9→3.9% | ✓ 5.2→3.9% |

→ **세 신호가 서로 다른 능력을 담당한다** = two-stage + replay 방법론의 정량 근거.

---

## 4. 기존 handoff 대비 등급 변경 이력

| 기존 문서 · 행 | 기존 판정 | **갱신 판정** | 근거 |
|---|---|---|---|
| jihun `reasoning_quality` §1 egocentric 화법 | [반증] SFT로 이관 | **[정정] 지표 무효 — 본문 사용 금지** | 본셋 base 0% (§2-6) |
| jihun `reasoning_quality` §3-2 성립부등식 +2.4/+19.2pp | 실측 | **[정정] 폐기 — matched +1.2pp 비유의** | G_DELTA_FULLSET (§2-4) |
| jihun `reasoning_quality` §5-3 GADR 8.1 방향 | 시사 | **[확정] 8.1** (ratio 0.73) | GADR_ATTRIBUTION (§2-5) |
| jihun `history_strip` §4-1 "Q 맹목 아님" | 갱신 필요 | **유지** — graded dissociation | 변경 없음 |
| jihun3 `cesft_v2_quantitative` §0-3 "SFT acc 중립" | FAIL(중립) | **[정정] 방어설 확정 — replay의 성과** | sft_r0 (§2-1, §2-2) |
| jihun3 `cesft_v2_quantitative` §1-2 G-DELTA SKIP | SKIP | **유지 SKIP** + 대안축(학습신호) 제시 | §2-4 |
| jihun3 `cesft_v2_quantitative` Frontier 빈칸 | 미측정 | **[신규] VPA 실측 · select 부분** | §2-7 |
| (신규 행) CE-replay 필요성 | — | **[확정]** | §2-1 |
| (신규 행) WiSE α=0.5 frontier | — | **[신규·시사]** | §2-3 |

---

## 5. 주장 가능 / 불가 (실측 경계)

**주장 가능**
- "candidate-CE는 WM top-1 모방을 넘는 선택·판별 엔진을 만든다" (G-ACC1 +7.2pp[0.46,15.0], GADR)
- **"CE-replay 없이 projected-trace SFT를 얹으면 hard-case 판별이 유의하게 손실된다(GADR −6.5pp[−12.4,−0.8]);
  ρ=0.15에서는 그 손실이 관측되지 않는다(−1.4pp[−6.4,+3.6])"** — 신규. **단 "replay가 유의하게 더 낫다"는 불가**(§2-1 주의)
- **"인과 추론 채널의 형성은 CE-replay와 무관하다"** (r0 harden 전 게이트 PASS, U_g +6.2pp[4.2,8.3]) — 신규
- "SFT는 개입으로 검증되는 belief→action 인과를 심는다" (U_g +9.8pp[7.2,12.4], swap vs para 18배)
- "history는 선택에 인과적으로 기여한다" (본셋 +3.1pp, 파일럿 +12.6pp·DiD +8.4pp — 두 코호트 재현)
- "WM과 LM은 상보적 국면에서 이긴다" (무-history WM>LM +4.2pp / history 축적 시 LM≫WM)
- "후보 경계가 명시적 증류 없이 내재화된다" (in_support +21.4pp[17.2,25.8], WM 증류와 통계적 동률)

**주장 불가**
- "SFT가 정확도를 개선한다" (G-NH FAIL) — 또한 "비열등하다"도 CI 폭 때문에 정식 주장 불가
- "egocentric reasoning" (§2-6 — 지표 자체가 템플릿 함수)
- "후보 제시가 자유생성보다 낫다" (§2-4 — 표본 정합 시 비유의)
- "causal mediation" (interventional dependence까지만)
- Frontier와의 직접 성능 비교 (text-only)
- 두 코호트 수치 직접 비교 (방향·구조만 교차)

---

## 6. 잔여 공백 (우선순위)

1. **G-NH CI 축소 (~1h, 학습 0)** — EVAL_N 1000→2500~3000 재측정. 현재 CI 폭 12.8pp로는
   "SFT가 판별을 해치지 않는다"는 최소 주장조차 방어 불가. **최우선.**
2. ~~sft_r0 paired CI~~ — **완료 (2026-07-25 05:49, §2-1)**. 후속으로 **r0 vs r15 직접 paired 검정**이
   필요하면 추가(현재는 각각 θ_CE 대비 독립 검정만 존재 — §2-1 통계 주의 참조). 학습 0, 분석만.
3. **Frontier select 972건 재시도 (~1h, API)** — 예산 복구 후. `resume_select_429.sh` 사용 금지
   (api_error 행이 이미 제거돼 "완료"로 오판정) → `frontier_select_eval.py` 직접 호출.
4. **cand_free arm (~2.1h, GPU)** — G-DELTA 본셋 공백. 다만 §2-4 대안축(학습신호)으로 우회 가능하므로
   우선순위 하향 가능.
5. **부록A C-stack/C-ctrl (~3h)** — T-ACC. 핵심 주장 3개와 무관, 부록 한정.
6. (선택) **본셋 무-history WM>LM** — 파일럿의 이중 해리를 본셋에서 재현하면 서사가 크게 강해진다.

---

## 7. 근거 파일 좌표

| 무엇 | 위치 |
|---|---|
| 본셋 arm별 평가 | `EGO_jihun3/runs/cesft_v2/eval/{base,theta_ce,sft_r0,sft_r15,wise_a050}.{json,records.jsonl}` |
| 본셋 게이트 | `eval/paired_{G-ACC1,G-NH}_*.json` · `eval/strip_verdict.json` |
| 본셋 개입 (spine) | `eval/{sft_r15,wise_a050}.harden_s3.json` (verdict "PASS — spine 확정 (U_g)") |
| sft_r0 실행 로그 | `runs/cesft_v2/logs/serverA_r0.log` · 런처 `run_serverA_r0.sh` |
| 파일럿 history-strip | `EGO_jihun/runs/goalstep_v3_boundary/eval/HSTRIP_VERDICT.{md,json}` |
| 파일럿 내재화 | `.../eval/I_VERDICT.{md,json}` (I-1 FAIL · I-2 PASS · I-3 PASS) |
| GADR 귀인 · G-DELTA · trace 텍스트 | `EGO_jihun3/runs/overnight_20260724/{GADR_ATTRIBUTION,G_DELTA_FULLSET_VERDICT,TRACE_TEXT_METRICS}.json` |
| Frontier VPA · select | `runs/overnight_20260724/frontier/{metrics_T3_test,metrics_T4_test,frontier_select}.json` |
| WM prior | `RETRO4-goalstep-end-m1-history-k8-phase1/best_action_top5.pt` |
| 논문 소스 | `EGO_paper/EGO_AAAI27_EN/main.tex` (영문) · `EGO_paper/EGO_AAAI27/main.tex` (국문) |

**운영 주의 (2026-07-25 사고)**: `runs/cesft_v2`는 두 호스트가 공유한다. 착수 전 `OWNER.lock`의 **ts 신선도**를
확인할 것(PID 생존 판정은 교차-호스트에서 무효). 상세: `runs/cesft_v2/owner_lease.sh` 주석.
