# 서버 B 대조군 평가 핸드오프 — base·cand_free 4종 배터리 (2026-07-26)

**코호트**: cesft_v2_fp (1인칭 일원화 최종본). 평가 풀 covered-only (val, n=1000; harden n=400; freegen n=500).
**시각**: 13:37–14:36 KST, 서버 B 병렬 분업. **남은 절반**: θ_CE(~18시), sft_r15(~19시 반) — 이 문서는 대조군 절반만 다룬다.
**원자료**: `runs/cesft_v2_fp/eval/{base,cand_free}.records.jsonl`, `*.harden_s3.records.json`,
`freegen_*.records.jsonl`, markers `S7_EVAL_*`, `S_STRIP_*`, `S3H_*`, `S_FREEGEN_*`.

---

## 1. 무엇을 검증하려 했나 (사전 등록된 목적)

| 평가 | 질문 | 대조 구조 |
|---|---|---|
| E1 battery (SelAcc) | 경계 내 **판별력**이 어디서 오는가 — 후보-대조 selection CE 구조인가, 단순 GT 노출인가 | base=무학습 하한, cand_free=**G-DELTA 대조군**(후보 없이 GT-span CE만 학습), L0=WM top-1 |
| E2 strip | history가 행동 선택에 **인과적**으로 기여하는가 | 같은 모델·같은 후보에서 history만 제거한 paired 개입 |
| E3 harden | task belief가 행동에 **인과적**으로 기여하는가 | belief swap vs paraphrase 개입, likelihood 채점 |
| E4 freegen | 학습이 만든 능력의 **소재** 확인 — 후보 비제시 레짐에서는 무엇이 변했나 | 자유 생성 후 support 매칭 |

cand_free는 θ_CE와 **동일 subset·epochs·seed**(G-EQ)로 학습된 대조군이다. 학습 자체는 정상 수렴했다
(GT-span CE 8.44→~1.75; probe 자유생성 정확도는 0.156→0.125로 무개선).

## 2. 실측 결과 (전부 1인칭 fp 코호트 실측)

### E1 battery — 판별력은 생기지 않았다
| arm | SelAcc | vs L0(24.2%) | copy(top-1 복사)율 |
|---|---|---|---|
| base | 21.0% | −3.2pp | 19.3% |
| **cand_free** | **21.9%** | **−2.3pp** | 16.0% |

- cf−base = **+0.9pp**의 실체 (paired n=1000): 둘 다 오답 685 / **cf만 정답 105 / base만 정답 96** / 둘 다 정답 114.
  → 201건의 예측이 뒤바뀌는 **churn** 끝에 순이득 +9건. 판별력 형성이 아니라 재배열.
- WM top-1이 정답인데 **둘 다 놓친 145건** — 경계가 정답을 들고 있어도 무학습/GT-CE LM은 활용 못 함.
- copy율 16–19% < L0 24.2%: top-1만 복사해도 더 잘했을 수준 — **모방 경로는 애초에 없음** (C2의 전제 확인).

### E4 freegen — GT-CE의 효과는 '판별'이 아니라 '명명 정렬'
| arm | in_support | gt_correct | malformed |
|---|---|---|---|
| base | 43.0% | 37.6% | 4.0% |
| cand_free | **60.6% (+17.6pp)** | 41.2% (+3.6pp) | 0.4% |

cand_free는 자신이 학습한 레짐(후보 없이 GT 명명)에서는 뚜렷이 개선 — taxonomy 형식의 행동 명명·어휘 정렬.
같은 모델이 battery에서는 +0.9pp: **능력의 소재가 레짐에 갇혀 있고 판별로 전이되지 않는다.**

### E3 harden — belief 인과는 두 대조군 모두 부재
| 지표 (n=400) | base | cand_free | (참조: 파일럿 sft_r15) |
|---|---|---|---|
| belief-only 인과 민감도 | 0.073 | 0.085 | 0.296 |
| belief-only utility | 0.011 | 0.033 | — |
| flip(paraphrase) | 0.75% | 0.75% | 4.5% |
| flip(swap both) | 87.5% | 82.5% | — |

개입 설계 자체는 대조군에서도 특이적으로 작동(paraphrase 0.75%). belief만 바꿨을 때의 행동 변화가
base·cf 모두 0.07–0.09로 미미 — **belief→행동 인과는 학습 설계 없이 생기지 않는다.**

### E2 strip — history는 무학습에도 인과적
| arm | ΔAcc(all) | ΔAcc(H8, n=893) |
|---|---|---|
| base | +2.9pp CI[−0.1, 5.7] | **+4.0pp CI[1.0, 7.2]** |
| cand_free | +4.5pp CI[1.5, 7.3] | **+4.4pp CI[1.2, 7.4]** |

WM 후보를 고정한 채 history만 제거하는 개입에서 두 arm 모두 H8(충분한 이력) 구간 유의 하락 —
WM 단계에서 확인했던 history 기여(+1.94pp)가 하류 LM 판단에서도 재현.

## 3. 추론 로그가 보여주는 도출 기전 (원문 발췌)

### 기전 A — "다음 행동 예측"이 "현재 행동 재기술"로 붕괴 (145건 실패군의 전형)
`04fe8f4d..._228` GT=**mix ingredient**, WM top-1=**mix ingredient** (정답이 경계 안에 있음):

> **cand_free** (action: *stir ingredient* ✗): "The current action involves stirring the mixture in the
> pan, which is part of the cooking process. The candidate 'stir ingredient' **matches this activity
> directly**. ... the person is actively cooking and stirring, so **continuing this action** is logical."

> **base** (action: *check heat* ✗): "...The most logical next step is to **continue monitoring** the
> cooking process by checking the heat..."

지금 보이는 것과 가장 닮은 후보를 고르거나(재기술), 언어적으로 매끄러운 관리 행동(check heat, wash hand)
으로 이탈 — **시각 경계가 정답을 제시해도 시간 계약(다음 행동)을 지키는 판별이 없다.**

### 기전 B — GT-CE는 task_belief를 action-echo로 형해화
cand_free가 맞춘 사례조차 belief가 행동 문자열의 반복으로 붕괴:

`002c3b5c..._7` GT=**cut potato** → cf: task_belief = **"cut potato"** (action과 동일 문자열)
`04fe8f4d..._228` → cf: task_belief = **"stir ingredient"**; `0a01978c..._285` → cf: **"dispose item"**

base의 belief는 문장형("Wash the peeled potato before cutting.")인데 cf는 GT 명명 훈련의 부산물로
belief 슬롯에 행동명을 복창 — §2 harden의 낮은 belief 인과(0.085)와 정확히 부합하는 표층 신호.

### 기전 C — cf의 이득 105건은 '명명 양식 정렬'의 산물
`04fe8f4d..._166` GT=**mix_(beat,_blend,_whisk) ingredient**, WM top-1=*stir ingredient*(오답):

> **base** (✗ *stir ingredient*): "...the most logical next step is to continue stirring..."
> **cand_free** (✓): "...The candidate action **'mix_(beat,_blend,_whisk) ingredient'** aligns with
> this ongoing task."

stir/mix처럼 표면상 유사한 후보 쌍에서 cf는 **GT 데이터의 taxonomy 표기 쪽으로 정렬**된 선택을 함 —
freegen in_support +17.6pp와 같은 원천. 판별이 아니라 어휘·양식 사전(prior)의 이동.

### 기전 D — 같은 크기의 파괴: base가 맞던 96건을 cf가 깨뜨림
`04fe8f4d..._236` GT=**stir ingredient**: base ✓ "Among the candidates, 'stir ingredient' is the most
appropriate..." / cf ✗ *add rice* — "Given the context ... **adding rice** to the pot is a logical next
step" (그럴듯한 절차 진행 상상으로 이탈). `_247`에서도 cf는 *add ingredient*로 이탈.
→ 이득(기전 C)과 손실(기전 D)이 상쇄되어 순 +0.9pp.

### 기전 E — belief 인과의 부재와 예외 (harden 원자료)
cf에서 own-correct 217건 중 belief swap으로 flip된 것은 **7건뿐**. flip 사례는 belief가 실제 문장일 때만
발생: `eb1b6e62..._4212` own_belief "Shape the remaining dough into balls." → swap "check heat" 시
p(GT) 0.716→0.061. 반면 echo형 belief(`_4218` "Shape dough")는 swap해도 0.631→0.631 무변화 —
**belief가 정보를 담지 않으면 개입도 무효**, 인과 부재의 미시적 근거.

## 4. 시사점

1. **GT 정답 노출만으로는 경계 내 판별력이 생기지 않는다** — 4,189샘플 GT-CE의 순효과 +0.9pp,
   L0조차 넘지 못함. G-DELTA의 대조군 절반이 1인칭 최종 코호트에서 확보됨.
2. GT-CE가 만든 것은 **명명 양식·어휘 정렬**(freegen +17.6pp)이며, 이는 판별(battery)로 전이되지 않는다
   — "능력의 소재" 분해가 실측으로 갈라짐.
3. **belief→행동 인과는 설계 없이 창발하지 않는다** (0.07–0.09) — Retrospection의 projected-trace
   지도 + gate + replay가 belief 인과(파일럿 0.296)의 원천이라는 C3 주장의 대조 축.
4. history는 무학습 LM에도 +4pp 인과 기여 — WM history 설계의 하류 정당화.
5. 판정 유보: θ_CE(오늘 ~18시)가 L0를 유의하게 넘으면 C1/C2가, sft_r15 harden(~19시 반)이 belief 인과
   0.2대를 재현하면 C3가 1인칭 코호트에서 완성된다.

## 5. 연구 의의(C1–C3)와의 연결

- **C1 (분업)**: 판별력의 원천이 '후보-대조 selection CE 구조'임을 대조군 부재로 입증하는 절반 완료.
  cand_free는 동일 데이터·동일 스텝에서 판별을 만들지 못했다.
- **C2 (경계 위의 추론)**: copy율(16–19%) < L0(24.2%) — top-1 모방 경로가 실재하지 않음을 확인.
  145건 실패군은 "경계를 받아도 추론하지 못하는 LM"의 실체이며, θ_CE의 개선이 어디서 와야 하는지 규정.
- **C3 (검증 가능한 인과)**: 개입 설계의 특이성(paraphrase 0.75%)이 대조군에서도 유지되므로,
  sft_r15의 belief 인과가 재현되면 그 차이는 설계(projection+gate+replay)에 귀속된다.

## 6. 한계·주의

- covered-only(n=1000) 조건부 수치 — full-pool 환산치(acc_full_equiv ~0.09)와 혼용 금지.
- strip base의 all-집계 CI는 0을 스침([−0.1,5.7]) — H8 층화 수치로 인용할 것.
- 파일럿(3인칭) 수치와 fp(1인칭) 수치는 코호트가 다름 — 비교 시 명시 필수.
- probe(n=32)는 노이즈가 크므로 참고 지표로만.

---

## 7. (추가 2026-07-26 15:2x) 정성 지표 — GT-CE가 화법·belief 표층에 남긴 변화

도구: `tools/trace_text_metrics.py` (정규식은 EGO_jihun 파일럿과 **글자 그대로 동일** — 잣대 유지).
채점: non-malformed & reasoning 존재분. 산출: `runs/cesft_v2_fp/eval/text_metrics.json`.
보조 지표(belief-echo)는 이번에 신규 정의(아래 명시) — 파일럿 잣대와 구분해 인용할 것.

### 7.1 파일럿 잣대 지표

| 지표 | base bat.(n=983) | **cf bat.(n=988)** | base fg.(n=480) | cf fg.(n=498) | Φ 학습타깃(참조) |
|---|---|---|---|---|---|
| 1인칭율 (reasoning) | 93.5% | **30.7% ↓** | 99.2% | 91.0% | 99.8% |
| 3인칭 표지("the person") | 3.9% | **15.1% ↑** | — | — | — |
| causal 접속사율 | 18.9% | **5.4% ↓** | 20.6% | 8.2% | — |
| scene 서술율 | 34.6% | 31.1% | 24.4% | 9.6% | 42.8% |
| future 표지율 | 96.6% | 93.8% | 90.6% | 90.2% | — |
| 후보 거명(elim_mention) | 86.2% | 87.4% | 79.4% | 75.5% | — |
| reasoning 평균 단어 | 63.2 | 56.7 | 53.0 | 47.1 | 81.6 |

### 7.2 보조 지표 (신규 정의: belief 문자열 vs 선택 action 문자열)

| 지표 | base bat. | **cf bat.** | base fg. | **cf fg.** |
|---|---|---|---|---|
| belief == action (완전 echo) | 0.0% | **28.9%** | 15.8% | **64.5%** |
| belief ≤3단어 | 0.1% | **33.8%** | 23.1% | **70.7%** |
| belief 평균 단어 | 8.4 | 5.8 | 6.0 | 3.4 |

### 7.3 독해

1. **1인칭 정렬조차 레짐-국소적**: cf는 자기 학습 레짐(freegen)에서는 1인칭 91.0%를 유지하지만,
   비학습 레짐(battery)으로 옮기면 30.7%로 붕괴하고 3인칭 표지가 4배 증가. 판별력이 전이되지 않던
   패턴(§2)과 동형 — GT-CE가 만든 모든 변화는 학습 형식에 갇힌다.
2. **belief 형해화의 정량 확인**: 기전 B(로그)의 action-echo가 base 0% → cf 28.9%로 측정됨.
   freegen에서는 64.5%까지 — belief 슬롯이 사실상 action 복창 채널로 전락. harden의 낮은
   belief 인과(0.085)의 표층 대응물.
3. **인과 접속 구조의 약화**: causal 접속사율 18.9→5.4%. 짧은 GT-span 학습이 긴 추론의 인과적
   연결 표현을 침식 — G-NH/replay가 방어하려는 침식의 대조군 버전.
4. 형식 준수(후보 거명 87%, future 표지 94%)는 유지 — 무너진 것은 **화법·인과·belief 내용**이지
   출력 형식이 아니다.
5. sft_r15 판정 관전점: 이 지표들이 Φ 타깃(1인칭 99.8%, scene 42.8%, 81.6단어)으로 회복되는지가
   Retrospection 설계 효과의 정성 축 검증이 된다.

### 7.4 (정정·보강) 선행 기록과의 대조 — 침식의 '자리 이동'

**선행**: [[2026-07-25_first_person_pronoun_erosion_candidate_vs_gt_ce_handoff]] (EGO_jihun) —
파일럿(goalstep_v3_boundary)에 history 이중 해리용 **Q(GT-CE, 후보 무) arm**이 존재했고
1인칭율이 측정돼 있었다. §7.3의 "최초 측정" 뉘앙스는 부정확 — 정정한다.

| | 파일럿 Q (비인칭 memory 혼재) | fp cand_free (1인칭 일원화) |
|---|---|---|
| freegen(후보 무·학습 레짐) | **21.2%** (base 74.0% 대비 −52.8pp 침식) | **91.0%** (−8.2pp, 침식 미미) |
| 후보 제시 레짐 | 미측정 (선행 §9-3 과제) | **30.7%** (붕괴) |

독해: ① 파일럿 침식의 확정 원인(비인칭 memory_context 용량-반응, H0 무침식)이었으므로,
07-25 1인칭 일원화가 학습-레짐 내 침식을 실제로 제거했음이 대조군에서 확인됨 — 일원화의 의도 효과.
② 침식은 소멸이 아니라 **비학습 레짐(battery)으로 이동** — "1인칭율은 레짐 종속 표면 지표,
같은 템플릿 내 arm 비교만 유효"라는 선행 판정은 유지·강화됨. ③ 파일럿 Q(21.2%)와 fp
cand_free(91.0%)의 수치 직접 비교는 템플릿·페르소나 상이로 금지 (선행 §3-4 판정 준용).

### 7.5 (정정 — 서버 B측 15:15 재집계 반영) §2 freegen 수치의 strict 교체

서버 B 세션이 freegen 요약 지표의 관대 매칭을 발견·재집계함 (`tools/freegen_strict_recount.py`,
산출: `runs/cesft_v2_fp/eval/freegen_strict_recount.json`). 원인: 후보 리스트가 단일 원소일 때
`match_candidate`의 토큰-겹침 폴백이 항상 성립 (verb 또는 noun 하나만 겹쳐도 정답 처리).

- **§2 E4 표의 인용 교체**: gt_correct 37.6/41.2% → **strict 11.8/15.2%** ·
  in_support 43.0/60.6% → **strict 29.2/44.0%** (격차 +17.6 → **+14.8pp**, 부호·순서 유지 → 결론 불변).
- battery(제시 레짐) 수치는 영향 없음 — 후보 10개 유일-최대겹침 + matched==gt 판정.
- 추가 신규 집계 (같은 세션): 게이트 규약 paired CI — cand_free−base SelAcc **+0.62pp
  CI[−2.24,+3.38] 비유의**, 두 대조군 모두 L0 paired 미달; gt_rank 층화 — GT-CE 이득은 2–10위
  표기 정렬(+2.7/+1.8pp), 1위 구간은 −3.2pp; coverage@K 사다리 10.1→43.4%.
- 이후 인용은 아티팩트(1d3ee191) 최신판 또는 본 절 기준.

### 7.6 학습 신호의 3인칭 문체 — 출처·영향·잔존 (신규 측정 포함)

**출처**: history(파일럿 memory_context)는 Ego4D GoalStep 주석에서 온다. 제3자 주석자가 구간에
verb–noun **분류 라벨**을 붙인 것이라 문법적 주어가 애초에 없다. 실물:
- 파일럿: `Completed actions before the current frame (oldest first):\n- (53s ago) stir soup`
- fp(현재): `Your completed actions so far (oldest to newest):\n- measure ingredient\n- stir ingredient`

**파일럿에서의 영향 (용량-반응, 선행 §3-2)**: 후보-무 자유생성 1인칭율이 H0 92.6%(무침식) → H1–3 73.3%
→ H4–7 37.7% → H8+ 21.0%. 침식은 학습 일반 효과가 아니라 **프롬프트에 쌓인 무주어 줄 수의 함수**.
극단 사례가 cesft_v2 — 시스템 프롬프트("the person")와 Φ 타깃까지 3인칭이어서 전 arm 0%,
지표 자체가 무효화(본문 사용 금지 판정의 근거).

**fp가 바꾼 것 / 안 바꾼 것** (07-25 일원화는 *프레임*만 교체):

| 구성요소 | 이전 | fp |
|---|---|---|
| SYSTEM_PROMPT (vlm.py:54) | 관찰자 "the person does next" | **행위자 "you do next"** |
| history 헤더 (vlm.py:195) | "Completed actions so far" | **"Your completed actions so far"** |
| SFT 타깃 Φ (projection.py:27) | "a careful observer…" | **1인칭 "I" (타깃 1인칭율 99.8%)** |
| **history 항목 (vlm.py:183)** | `- verb noun` | **`- verb noun` 그대로 (미변경)** |

→ 비인칭 압력은 잔존하고 반대 앵커만 강해진 상태. base 93.5/99.2%가 앵커의 힘.

**잔존의 실측 (신규, 서두 정형구 집계)**:

| reasoning 서두 | base bat. | cf bat. | base fg. | cf fg. |
|---|---|---|---|---|
| "I …" | 92.7% | **30.2%** | 97.3% | 90.0% |
| 관찰자·주석 문체 | 2.5% | **48.2%** | 0.0% | 2.8% |
| 그중 "The current action involves…" | 0.4% | **41.5%** | 0.0% | 2.8% |

cand_free는 비학습 레짐에서 10건 중 4건을 단일 정형구로 시작 — 무작위 붕괴가 아니라 **주석자 언어로 수렴**.

**기각된 가설 (정직 공시)**: "주석 문체 복귀 → 현재 행동 재기술 → 오답"이라는 연결은 이 데이터로
성립하지 않는다. cand_free 내부 서두별 SelAcc: 주석문체 서두 **29.7%**(n=451) > 1인칭 서두
**16.1%**(n=298) — 반대 방향(관찰적 분할, 인과 해석 불가). **화법 축과 능력 축은 같은 학습 신호에서
갈라지지만 서로를 설명하지 않는다** → 선행의 "1인칭율=표면 지표, 능력 지표 사용 금지" 판정 유지.

**§7.4 정정**: "침식이 레짐 밖으로 이동" 서술은 근거 초과였다 — 파일럿 Q의 **제시 레짐 1인칭율은
미측정**(선행 §9 남은 과제 3). 4칸 중 한 칸이 비어 있으므로 fp 내부 사실까지만 진술한다.

---

## 8. 능력 축 재분석 — 우리 학습이 형성/강화하고 GT-CE가 침식시키는 것 (2026-07-26 16:1x)

**아티팩트**: https://claude.ai/code/artifact/018e1b2d-67e4-4fac-bb21-2dc7509d898c
**목적**: 1인칭율 같은 표면 지표 대신 **정오와 연결되는** 능력 축을 로그에서 찾는다.
**비교 구조 (중요)**: fp의 본 방법 arm은 미도착 → 파일럿 코호트(cesft_v2, 3인칭)의 본 방법 계보
(base→θ_CE→sft_r15)와 fp(1인칭)의 대조군 계보(base→cand_free)를 병치. **절대값 비교 금지,
코호트 내 Δ만 유효.** 정당화 조건: 두 base가 근접(SelAcc .202/.214, G1 .318/.297, 근거 전환력
+12.9/+13.5pp). 저녁에 fp θ_CE·sft_r15 도착 시 동일 코호트 내 재확인 필수.
**집계**: battery records × context_val 조인(history·후보·gt_rank), 각 코호트 공통 covered n=1000,
non-malformed. 신규 추론 0.

### 8-1. 다섯 축 실측

| 축 | base(파) | θ_CE | sft_r15 | Δ(우리) | base(fp) | cand_free | Δ(대조) |
|---|---|---|---|---|---|---|---|
| **1. G1 유지** (top-1 정답 보존) | .318 | .395 | .418 | **+10.0pp** | .297 | .265 | **−3.2pp 침식** |
| **2. G2 교정** (top-1 오답 재선택) | .165 | .224 | .274 | **+10.9pp** | .187 | .208 | +2.1pp |
| **3. 연속 판별 회수율** | .271 | .410 | .529 | **+25.8pp** | .284 | .329 | +4.5pp |
| ─ 연속선택 정밀도 | .712 | .676 | .722 | 유지 | .727 | .713 | 유지 |
| **4. 근거 전환력**(조건부 이득) | +12.9pp | +15.3pp | **+19.6pp** | 강화 | +13.5pp | **+3.2pp** | **소멸** |
| ─ 근거화 언급률 | .285 | .417 | .264 | — | .224 | **.378** | 증가 |
| **5. belief=action 복창** | .001 | .119 | **.000** | 복원 | .000 | **.288** | 고착 |
| ─ belief 인과(개입) | — | — | **0.296** | — | 0.073 | 0.085 | — |

기준값: GT의 **31.0%**가 직전 행동의 연속(310/1000). 연속 선택은 어느 arm에서나 정밀도 0.71–0.72로
고정밀 신호 → 관건은 **회수율**.

### 8-2. 핵심 발견 — 축 4 (근거의 판별 전환력)

두 계보가 **정반대**로 갈리는 유일한 축.
- cand_free: 이전 행동 패턴을 **더 자주 언급**(22.4→37.8%)하지만 언급의 이득은 **소멸**(+13.5→+3.2pp).
- sft_r15: 언급률은 base 수준(26.4%)인데 **언급 시 이득 최대**(+19.6pp).
→ 우리 학습이 키운 것은 *근거를 늘어놓는 습관*이 아니라 **근거를 판단으로 바꾸는 힘**.
선행 핸드오프의 "어휘 인용의 역설"(C는 history를 덜 인용하며 더 씀)과 동형이며, 이번엔 **대조군에서
정반대(더 인용하고 못 씀)**가 실증됨.

로그 예 (cc575a16, GT=stir ingredient, WM top-1=stir dough 오답):
- base ✗ "…they are about to add something… adding water is the most logical next step" (눈앞 물체에서 장면 상상)
- sft_r15 ✓ "…**repeatedly stirring**… only stirring aligns with… the **established pattern of action**"

### 8-3. 기각된 축 (정직 공시)

**대조 접속 표현**("rather than"/"but not"): sft_r15에서 0.4%→**10.4%** (26배, 대조군 0.2% 무변화)로
가장 눈에 띄지만, sft_r15 **내부** 유무별 SelAcc가 31.7% vs 30.8%(**Δ+0.9pp**)로 무의미 →
Φ 타깃 문체의 이식 흔적일 뿐 **능력 아님**. 축 채택 기준("정오와 연결되지 않으면 능력 아님")의 적용례.

### 8-4. 의의 정합

- **C1 분업**: 축 5가 직접 증거 — 1단계는 판별을 만들되 belief를 일시 형해화(11.9%), 2단계가 복원(0.0%)
  하면서 판별을 더 끌어올림(G2 +5.9→+10.9pp). 두 단계가 서로 다른 것을 담당.
- **C2 경계 위의 추론**: 축 1·2. 경계가 준 정답을 지키는 힘(+10.0pp)과 top-1을 넘어 고르는 힘(+10.9pp)이
  함께 성장. 대조군은 전자가 침식(−3.2pp) → **경계를 받는 것만으로는 경계를 쓸 수 없다.**
- **C3 검증 가능한 인과**: 축 4(관찰)와 축 5의 belief 인과(개입)가 같은 방향으로 수렴 —
  근거 전환력 +19.6 vs +3.2pp, belief 인과 0.296 vs 0.085.
- 논문 서술 후보: "우리 학습이 만든 것은 *더 많이 말하는 추론*이 아니라 *말이 판단을 이끄는 추론*" —
  rationale faithfulness 문제에 대해 경계(외부 제약)+사후 투영(지도 신호)으로 연결을 만든다는 주장.

### 8-5. 한계

1. 코호트 교차 — 절대값 금지, Δ만. fp θ_CE·sft_r15 도착 시 동일 코호트 재확인(오늘 저녁).
2. 축 4·5의 문구 분할은 **관찰적**(인과 아님). 개입 근거는 harden belief 인과뿐.
3. 파일럿 θ_CE·sft_r15는 원래 covered_only=false(n=5326) 평가 → 여기서는 base와의 공통 covered
   교집합(n=1000)으로 재집계한 값.
4. 근거화·대조접속 정규식은 **이번 분석 신규 정의** — 파일럿 잣대(trace_text_metrics.py) 지표와 구분해 인용.
