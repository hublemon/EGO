# cesft 지표 대시보드 artifact + 09:15 게이트 갱신 Handoff (G-NH PASS 전환)

> 작성: 2026-07-25 KST · EGO_jihun3.
> **목적: ① 논문 §Embodied Reasoning 지표 목록을 cesft 코호트(runs/cesft_v2) 실측으로 채운 시각
> 대시보드(artifact) 제작 기록, ② 09:15 재계산으로 발생한 게이트 판정 변경(중대: G-NH FAIL→PASS)의 정리,
> ③ 논문(main.tex)에 없는 확장 지표의 실측 정리.**
> 갱신 대상: [[2026-07-25_cesft_v2_quantitative_evidence_handoff]] §1-2 게이트 표(G-NH FAIL → **PASS**),
> [[2026-07-25_paper_capability_evidence_crosscohort_handoff]] §6 잔여 공백 #1·#2 (**해소됨**).
> 지표 정의 SSOT: [[2026-07-24_evaluation_metrics_handoff]] · full 계획: [[2026-07-25_cesft_full_training_plan_and_metrics_handoff]] §4.
>
> **시각 대시보드 (완성본)**: https://claude.ai/code/artifact/e4bafc93-9e5e-480c-8ef4-5adcd6a646f3

---

## 0. 다섯 줄 요약

1. **[중대·판정 전환] G-NH FAIL → PASS** — full covered 재계산(n_paired 2,165, cluster 87)에서
   SelAcc Δ+2.2pp[−0.2,+4.7] ∧ GADR Δ+3.0pp[−0.1,+6.2], 비열등 기준(하한≥−1pp) **양쪽 통과**.
   파일럿 n=1,000의 FAIL(중립, +1.1pp[−5.7,+7.1])이 표본 5배로 뒤집혔다(§2). 단 **"SFT가 acc 개선" 주장은 여전히 불가**(CI가 0 포함).
2. **G-ACC1 재계산** — +7.2pp[0.5,15.0](n=389) → **+4.8pp[1.3,8.5]**(n=2,218, cluster 87). 효과 축소·CI 협소화, PASS 유지.
3. **[신규] r0 vs r15 직접 paired** — r0가 SelAcc −4.3pp[−8.3,−0.6]·GADR −4.4pp[−8.1,−1.1] **양지표 유의 열세**
   → CE-replay 필요성의 직접 paired 근거 완결(crosscohort §6 공백 #2 해소).
4. **harden n=800 재실행(sft_r15)** — sensitivity 0.291[0.259,0.325]·U_b +5.4pp[4.3,6.6]·para 3.5%. 전 게이트 PASS 유지, CI 협소화.
5. **3-arm 지표 전량을 동일 조건(covered)으로 재계산**해 artifact에 표+그래프로 고정, 논문에 없는
   확장 지표(개입 팩·텍스트 신규 3종·게이트/아블레이션)와 anchor trace 2건 포함(§3~§5).

---

## 1. 대시보드 실측 — 3-arm 학습 스텝별 지표 (재계산, covered 통일)

**측정 조건**: heldout covered(GT∈Top-10, K=10) 부분집합으로 통일.
⚠ **모집단 차이**: base는 07-24 covered-only 평가(n=1,000 서브샘플), θ_CE·sft_r15는 07-25 full heldout
재평가(n=5,326)의 covered 2,313 전량 — base 열은 근사 비교이며 full 런에서 3-arm 동일 셋으로 재산정 예정.
텍스트 지표는 covered ∧ non-malformed의 `reasoning` 필드 재계산.

| 지표 | Base | θ_CE (S1) | θ_CE+SFT (S2) | Δ₁ | Δ₂ |
|---|---:|---:|---:|---:|---:|
| SelAcc@10 (covered %) | 20.0 | 27.0 | **29.2** | +7.0 | +2.2 |
| G1 retention (%) | 31.4 | **40.5** | 39.6 | +9.1 | −0.9 |
| GADR (%) | 16.4 | 22.9 | **26.0** | +6.5 | +3.1 |
| Full-eq (×cov 0.434, %) | 8.7 | 11.7 | **12.7** | +3.0 | +1.0 |
| L0 (WM top-1, %) | 24.2 | 23.3 | 23.3 | — | — |
| malformed (covered %) | 1.1 | 4.1 | 3.2 | +3.0 | −0.9 |
| 장면 묘사율 (%) ↓ | 55.0 | 53.3 | **34.3** | −1.7 | −19.0 |
| 미래-지향율 (%) | 94.8 | 95.4 | 57.9 | +0.6 | −37.5 |
| 배제 언명률 (%) | 27.0 | 10.5 | **28.6** | −16.5 | +18.1 |
| 1인칭율 (cand-present, %) | 0.0 | 0.0 | 0.0 | 0 | 0 |
| reasoning 길이 (단어) | 69.3 | 57.6 | 80.3 | −11.7 | +22.7 |

- **파일럿 handoff 수치와 다른 이유**: handoff의 SelAcc 30.8/31.7 등은 n=1,000 스냅샷·paired 부분집합(n=389) 기준.
  이 표는 현행 records 전량 재계산값(theta_ce/sft_r15 records는 07-25 08:03/08:56 재생성분).
- **배제 언명률 정의 변경 주의**: `(other|remaining|alternative) (candidates?|options?|actions?)` 패턴 검출률.
  구 소거 서술률(10.4→2.4→25.5%)과 정의·records 스냅샷이 달라 절대값 상이, **방향(CE 침식→SFT 회복)은 동일 재현**.
- 미래-지향율의 S2 급락(95.4→57.9)은 신규 관찰 — projected trace 문체가 `will/next/likely` 계열 표현을
  줄이는 것으로 보이며 full 런에서 재확인 필요.
- **미측정 2종**: Candidate in-support rate · 1인칭(cand-free) — freegen(후보-비제시 생성) 패스 부재.
  cesft_full 계획 §7-6(`eval/freegen.py`)에서 채워질 항목.

---

## 2. 09:15 게이트 갱신 (paired_boot 재실행, video-cluster bootstrap)

| 게이트 | 구판 (n=1,000 era) | **신판 (full covered)** | 판정 |
|---|---|---|---|
| **G-ACC1** | +7.2pp[0.5,15.0] n=389·cl76 | **+4.8pp[1.3,8.5]** n=2,218·cl87 (SelAcc 28.1 vs L0 23.3) | PASS 유지 |
| **G-NH** | +1.1pp[−5.7,+7.1] **FAIL(중립)** | **SelAcc +2.2pp[−0.2,+4.7] ∧ GADR +3.0pp[−0.1,+6.2]** n=2,165·cl87 | **PASS 전환** |
| r0 vs θ_CE (참고) | GADR −6.5pp[−12.4,−0.8] 유의 붕괴 | (05:49 산출 유지) | r0 FAIL |
| **r0 vs r15 직접** ★신규 | — (독립 검정만 존재) | **SelAcc −4.3pp[−8.3,−0.6] · GADR −4.4pp[−8.1,−1.1]** n=392·cl76 | r0 유의 열세 |

**서사 영향 (주장 경계 갱신)**:
- 허용: "projected-SFT는 판별(SelAcc·GADR)을 **손상하지 않는다**" (비열등 PASS) — 기존 "중립(FAIL)" 서술을 대체.
- **여전히 금지**: "SFT가 정확도를 개선한다" — SelAcc·GADR Δ 모두 CI가 0을 포함.
- CE-replay: "replay 없이는 판별이 유의하게 손실되고(r0 붕괴 + 직접 검정), replay는 그 손실을 막는다" —
  이제 **직접 paired 근거**로 말할 수 있음. "replay가 유의하게 더 낫다"는 r0−r15 직접 검정 통과로 **주장 가능해짐**(구판에서는 불가였음).
- harden n=800: sensitivity 0.291[0.259,0.325] · U_b +5.4pp[4.3,6.6] · D_g 0.463 · para 3.5% — G-CC1∧CC3 PASS 유지.

---

## 3. 논문(main.tex)에 없는 지표 — 실측 정리 (artifact §4)

### A. 개입 확장 팩 (sft_r15, n=800) — 논문은 belief sensitivity·U_b만 수록

| 지표 | 실측 | 비고 |
|---|---|---|
| flip율 4종 | swap_b 32.6 · swap_r 29.1 · swap_both 80.0 · para 3.5% | sens 계산 원재료 |
| reasoning sensitivity | 0.256[0.224,0.289] | |
| both sensitivity | 0.765[0.735,0.796] | G-S3a PASS · **초가법**(0.291+0.256<0.765) |
| U_g(own−swap_both) | +10.5pp[8.7,12.3] | 레거시(U_b로 대체) |
| directional D_g | 0.463[0.428,0.496] | G-CC4 |
| correct-switch | −4.6pp (n_flip=261) | |
| acc 직교성 | flip(both) 정답 76.2 vs 오답 81.6% | |
| G-CC2 | **미측정** (base harden 없음) | |

### B. 텍스트 확장 지표 (이번 세션 신규 계산, Base → θ_CE → +SFT)

| 지표 | 정의 | 실측 |
|---|---|---|
| 인과 연결어율 (D7) | `since\|because\|having just\|given that\|therefore…` 검출률 | 7.9 → 3.1 → 5.2% |
| 후보 거명 수 (D4) | 후보 문자열(동의어 괄호 전개) 완전-포함 매칭 평균 개수 | 0.44 → 0.19 → 0.06 |
| 추론 압축률 (D2 파생) | 1 − len(arm)/len(base) | 0 → +16.9 → −15.9% |

⚠ 후보 거명 수는 굴절형("cutting") 미포착의 보수적 하한 — 배제 언명률(패턴 기반)과 괴리 나는 이유.

### C. 논문 밖 게이트·아블레이션

| 항목 | 상태 | 처리 |
|---|---|---|
| **CE-replay 취약성** (r0 붕괴 + 직접 검정) | 실측 완료 (§2) | **논문 반영 후보 1순위** — "replay가 판별 보존"의 유일한 정량 근거인데 main.tex에 없음 |
| G-ACC2 (GADR vs base) | point +6.5pp, paired CI 미산출 | 논문은 부등식 서술만 |
| G-DELTA | SKIP (cand_free 미학습) · 자매 matched +1.2pp 비유의 | 논문의 "생성 대비 선택 +12.1pp"는 파일럿 코호트 — 본셋 공백 각주 필요 |
| G-ACC3 fusion | 보조 분석 격하 · 미측정 | rank/prob 비공개 설계와 충돌 |
| WiSE-FT α=0.5 | SelAcc(cov) 28.3 · GADR 24.0 · **belief sens 0.116**(r15의 40%) | 3-arm 확정으로 논문 제외 — merge가 belief 채널을 깎는 참고 실측 |
| T-ACC · P-UTIL | 폐기 (3-arm 확정, 2026-07-25) | |

---

## 4. Anchor trace (artifact §5)

- 논문 국문판의 anchor `110352df…_497`은 **EGO_jihun 파일럿 코호트 샘플이라 cesft records에 부재**.
- 선정 기준: GADR 실물 = covered ∧ WM top-1 오답 ∧ base 오답 ∧ **θ_CE·sft_r15 양쪽 정답** → 교집합 50건 중 선별.
- **주 anchor** `864fa3d8-9b18-44cb-a8e9-9b40765e2d0c_2357` (MAKE_FLATBREAD, GT `roll dough`, WM top-1 `cook flatbread`):
  base는 WM top-1 추종 오답 → 두 학습 스텝 모두 사이클 구조를 읽어 교정, S2는 손 위치 시각 단서 + 후보 배제 언명 명시.
- **보조 anchor** `13c76616-f168-4af0-8d2a-fe82ce232d6a_516` (MAKE_STIR_FRIED_RICE, GT `check heat`):
  이력의 주기 패턴("repeatedly checking the heat")을 판별 단서로 사용 — history 인과(strip)의 실물 예시.

---

## 5. 재현 좌표

| 무엇 | 위치 |
|---|---|
| 시각 대시보드 | https://claude.ai/code/artifact/e4bafc93-9e5e-480c-8ef4-5adcd6a646f3 |
| arm 평가 records | `runs/cesft_v2/eval/{base,theta_ce,sft_r15}.{json,records.jsonl}` (θ_CE/sft는 07-25 08시대 full heldout 재평가분) |
| 게이트 (신판) | `eval/paired_G-ACC1_theta_ce.json` · `paired_G-NH_sft_r15_vs_theta_ce.json` · **`paired_direct_sft_r0_vs_sft_r15.json`(신규)** (전부 07-25 09:15) |
| 개입 | `eval/sft_r15.harden_s3.json` (n=800, 09:15) · `{sft_r0,wise_a050}.harden_s3.json` (n=396) |
| 후보 조인 | `runs/cesft_v2/data/context_val.jsonl` (`candidates` 필드, 동의어 `verb_(alt) noun` 형식) |
| 텍스트 지표 정규식 | 1인칭 `\b(I\|I'm\|I've\|I'll\|my\|me\|myself)\b` · 장면 `\b(shows?\|appears?\|the frame\|visible\|can be seen\|depicts?\|image)\b` · 미래 `\b(will\|next\|should\|going to\|about to\|likely\|plan to\|intend)\b` · 인과 `\b(since\|because\|having just\|given that\|as a result\|therefore\|thus\|so that)\b` · 배제 `(other\|remaining\|alternative) (candidates?\|options?\|actions?)` — EGO_jihun `scripts/step2/trace_text_metrics.py` 정의 계승 + 배제 신설 |

## 6. 남은 것

- [ ] full 런(`runs/cesft_full`)에서 3-arm 동일 셋 재산정 → artifact 수치 교체 (base 모집단 차이 해소)
- [ ] freegen 패스 후 in-support·1인칭(cand-free) 채우기 (계획 §7-6)
- [ ] base/θ_CE harden 측정 → G-CC2 및 belief 지표의 학습 전 앵커 확보 (현재 자매 파일럿 앵커만, 코호트 상이로 비교 금지)
- [ ] G-NH PASS 전환·CE-replay 직접 근거를 main.tex(§er-next non-harm 항목, §Method replay 서술)에 반영
- [ ] 미래-지향율 S2 급락(95.4→57.9)의 원인 확인 (projected trace 문체 vs 실제 성질 변화)
