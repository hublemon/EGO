# Ablation 계획 감사 handoff — attack/defense 두 문서 정합성 검토

- 작성일: 2026-07-27 (KST 01:40)
- 감사 대상
  - `docs/paper/2026-07-26_aaai_reviewer_attack_handoff.md` (이하 **공격문서**)
  - `docs/paper/2026-07-26_aaai_reviewer_defense_plan_handoff.md` (이하 **방어문서**)
- 감사 방법: 두 문서의 모든 실행 가능 주장을 코드베이스·산출물과 직접 대조. 추정치는 마커·파일
  타임스탬프로 재측정.
- 제약 조건: 제출까지 시간 부족, 논문 지면 부족 (Results 초안 이미 533행 / 표 4개 + 그림 2개)

---

## 0. 한 줄 결론

두 문서는 **서로는 정합적이다**. 방어문서는 공격문서 14건을 빠짐없이 받아 비용 오름차순으로
정리했고 대응 매핑도 정확하다. 문제는 다른 데 있다.

1. **두 문서 모두 2026-07-26 12:20 시점 상태로 굳어 있다.** 그 뒤 실행된 개선안 C가 공격문서
   A5(belief)의 전제를 뒤집었고, "final EGO = `sft_r15_gx`"라는 두 문서 공통의 전제도
   오늘 밤 안에 바뀐다. **A5 대응 전체를 재작성해야 한다.**
2. **논문이 스스로 약속한 ablation 두 개가 계획에 없거나 최하위다.** `main.tex` L205의
   ρ=0 대조군은 **아예 존재하지 않는 arm**이고, L289/302의 K ablation은 placeholder인데
   방어문서에서 Tier 3 우선순위 6이다. 반면 Tier 1의 10조건 스윕은 리뷰어 방어용
   add-on이 대부분이다. **핵심 ablation으로만 구성돼 있지 않다.**
3. **G-NH 실패의 성격을 두 문서 모두 오독했다.** 이것은 방법의 실패가 아니라 **검정력 부족**이며,
   그렇게 쓰면 A3의 실점 폭이 크게 줄어든다.
4. 실행 계획에 **즉시 실패하는 명령이 하나** 있고(§5.1), 중복 셀이 하나 있으며(§5.2),
   시간 추정에 최대 2배 과대가 있다(§6).

**가장 비용 대비 이득이 큰 조치는 ρ=0 arm 학습 17분이다.** 논문 본문이 이미 약속한 실험이고,
지금 재료(`chosen_train.jsonl`)가 준비돼 있어 다른 어떤 항목보다 싸다.

---

## 1. 검증한 사실 — 계획을 바꾸는 것들

### 1.1 "final EGO = `sft_r15_gx`" 전제가 오늘 밤 무너진다

방어문서 §1.1은 `sft_r15_gx`를 final EGO로 두고 "개입 실험 2종 미실행 = 20 GPU-분" 문제로
정리했다. 정확한 진단이었으나, 그 뒤 두 개의 arm이 추가됐다.

| arm | 개입 | 상태 | belief 인과 |
|---|---|---|---|
| `sft_r15` | fp 프롬프트, ρ=0.15 | 완료 | 0.0975 [.070, .128] |
| `sft_r15_gx` | fp 프롬프트 + 오버샘플, ρ=0.15 | battery·freegen만 | 미측정 |
| **`sft_r15_c`** | **규칙 4·5 수정 프롬프트**, ρ=0.15 | **battery·strip·harden·freegen 완주** | **0.3825 [.333, .432]** |
| `sft_r30_c` | 같은 프롬프트, ρ=0.30 | 학습 완료, 평가 중 | 측정 예정 |

`sft_r15_c`는 방어문서가 "미사용 adapter"로 분류한 것과 달리 **완결된 arm**이며, 최종 arm
후보다. 최종 arm 확정은 `sft_r30_c` 평가 종료 시점(01:58 KST경)이다.

**따라서 방어문서 §1.1의 "gx harden/strip 20분"은 최우선이 아니다.** gx는 최종 arm이 아닐
가능성이 높고, 그 경우 gx의 개입 실험은 ablation 재료로만 의미가 있다.

### 1.2 A5(belief)의 전제가 뒤집혔다 — 가장 약한 축이 가장 강한 축이 됐다

공격문서 §6은 belief를 최대 약점으로 지목했다: *"Cand.-CE `0.093` vs Base `0.073`, 차이
`0.020`뿐이고 arm 간 CI도 없다. base model의 일반적 prefix sensitivity일 수 있다."*

`sft_r15_c` 실측으로 이 지적의 사실 근거가 사라진다.

| arm | belief 인과 | para 대조 | n_flip |
|---|---|---|---|
| base | 0.0725 | 0.0075 | 32 |
| cand_free (GT-only) | 0.0850 | 0.0075 | 37 |
| theta_ce (Cand.-CE) | 0.0925 | 0.0025 | 38 |
| sft_r15 | 0.0975 | 0.0125 | 44 |
| **sft_r15_c** | **0.3825 [.333, .432]** | 0.0425 | **170** |

추가로, 공격문서가 "없다"고 지적한 두 통제를 실측으로 채웠다.

**(a) arm 간 paired CI** — 공격문서 §6 및 Q10이 요구한 것. harden_s3는 arm마다 샘플 셋이
달라(실측 겹침 **27%**, θ_CE ∩ C = 170/400) paired 비교가 불가능하다는 것이 원래 한계였다.
3-arm 공통 78건으로 재계산했다.

| 비교 | paired Δflip | CI95 |
|---|---|---|
| C − θ_CE | **+0.3205** | [+0.1923, +0.4487] |
| C − sft_r15 | +0.3077 | [+0.1795, +0.4359] |
| sft_r15 − θ_CE | +0.0128 | [−0.0897, +0.1154] |

McNemar 불일치 30 대 5, **정확검정 p = 2.2×10⁻⁵**. 그리고 `sft_r15 − θ_CE`가 귀무라는 점이
중요하다 — 효과가 SFT 자체가 아니라 **프롬프트 수정에서 왔음**이 분리된다.

**(b) OOD/정보량 교란 배제** — 공격문서 §6 "hybrid prefix가 생성분포 밖일 수 있다"에 대한
직접 반박. own/swap belief의 어휘 거리별로 flip을 갈랐다.

| arm | 거리 하위 33% | 중간 | 거리 상위 33% |
|---|---|---|---|
| `sft_r15` | 0.083 | 0.105 | **0.142** ↗ |
| `sft_r15_c` | **0.459** | 0.414 | 0.403 → |

`sft_r15`는 거리에 비례해 오르는 **정보량 반응** 패턴이고, C는 **거리와 무관하게 평평**하다.
자기 belief와 거의 같은 문자열을 주입해도 46% 뒤집힌다. 주입 문자열의 정보량이 원인이면
나올 수 없는 모양이다.

> **방어문서 §3 A5 대응(RPT+RUN+EXP 7항목)을 전면 재작성해야 한다.** T2-2(video-disjoint
> donor)·T2-3(hard-negative swap)의 필요성은 남지만, "sensitivity가 base와 구분 안 된다"는
> 전제로 짜인 부분은 폐기 대상이다.

### 1.3 U_g·D_g가 이미 있다는 방어문서 §1.2의 지적은 정확하다

`harden_s3`의 표준 출력에 `utility_belief_only_ci`, `directional_dg_ci`, `correct_switch`가
전 arm 존재함을 확인했다. `sft_r15_c` 기준 `utility_belief_only = 0.0405 [0.0246, 0.0573]`,
`directional_dg = 0.4675 [0.4175, 0.5175]`. **GPU 0의 보고 문제라는 판단은 유효하다.**

### 1.4 G-NH 실패는 방법 실패가 아니라 검정력 부족이다 — 두 문서 모두 미포착

공격문서 §4와 Q9는 "사전등록 1pp 비-열등성 FAIL"을 방법 정체성 공격의 근거로 쓰고, 방어문서
A3는 이를 "서술 재정의"로만 받는다. 그런데 게이트 정의를 보면 성격이 다르다.

`tools/paired_boot.py:343` — `pass = selacc_ok AND gadr_ok`,
`selacc_ok = (SelAcc Δ의 CI 하한 ≥ −0.01)`.

이 코호트(n=932, 클러스터 86)의 SelAcc Δ **CI 반폭이 4.06pp**다. 하한이 −1pp 위로 오려면

```
point − 0.0406 ≥ −0.01   →   point ≥ +3.06pp
```

즉 **"−1pp 마진의 비열등"을 선언하려면 θ_CE를 3pp 이상 이겨야 한다.** 비열등 게이트가
우월 게이트로 작동한다. 반폭을 0.01까지 줄이려면 클러스터가 약 1,400개 필요하다(현재 86).

세 arm 전부 같은 벽에 걸렸다는 것이 방증이다.

| arm | SelAcc Δ vs θ_CE | GADR Δ | G-NH |
|---|---|---|---|
| `sft_r15` | −3.00pp [−6.37, +0.83] | −4.94pp [−10.59, +0.70] | FAIL |
| `sft_r15_gx` | −0.64pp [−3.88, +2.91] | −2.53pp [−6.98, +2.29] | FAIL |
| `sft_r15_c` | −1.18pp [−5.24, +2.57] | −3.39pp [−7.40, −0.27] | FAIL |

**이 사실을 논문에 명시하면 A3의 실점 폭이 크게 줄어든다.** "사전등록 마진이 설계 해상도보다
작았다"는 것은 정직한 방법론적 한계이지, 방법이 성능을 보존하지 못했다는 증거가 아니다.
사전등록 실패를 숨기지 않으면서 그 해석을 정확히 하는 것이므로 방어문서 §6의 정직성 원칙과도
어긋나지 않는다.

**단, GADR은 다르다.** `sft_r15_c`의 GADR Δ는 CI 상한이 −0.27pp로 0을 배제한다. 이것은
검정력 문제가 아니라 **실재하는 유의한 손실**이며, 그대로 한계로 보고해야 한다.
(원인 분석: WM 오답 구간이 평가셋의 76%이고, SFT는 G1 유지력 +5.5pp를 사고 GADR −3.8pp를
파는 교환을 한다. `0.242×5.5 + 0.758×(−3.8) = −1.55pp`로 관측된 SelAcc Δ −1.18pp와 일치한다.)

---

## 2. 두 문서의 정합성 판정

### 2.1 정합적인 부분

- 공격 14건 → 대응 매핑에 **누락 없음**. Q1–Q13이 전부 특정 대응에 배정돼 있다.
- 유형 분류(RPT/RUN/AGG/EXP/TXT)와 비용 오름차순 사다리는 올바른 설계다.
- §6 사전등록 결정규칙 표는 이 계획의 가장 좋은 부분이다. "실험 전에 반증조건을 커밋한다"는
  원칙은 B6(사후선택) 방어까지 겸한다. **Tier 1 착수 전 커밋 지시도 그대로 유지해야 한다.**
- §6의 "정직한 사전 선언"(C3에서 우위가 살아남을 가능성이 낮지 않다고 본다 → 그러면 본문에서
  먼저 제목을 내린다)은 전략적으로 옳다.

### 2.2 어긋나는 부분

| # | 문제 | 영향 |
|---|---|---|
| D1 | 두 문서 모두 `sft_r15_c` 완주 이전 상태. 방어문서 §1.3 표는 이를 "미사용 adapter"로 분류 | A5 대응 전면 재작성 필요 |
| D2 | 방어문서 §8-4가 `sft_r15_c`/`sft_r30_c`를 "ρ=0 대조군 승격 가능성"으로 추정 | **사실 오인.** 둘은 ρ=0.15/0.30이고 프롬프트도 fp와 다르다. ρ=0 arm은 존재하지 않는다 |
| D3 | G-NH FAIL을 검정력 문제로 인식하지 못함 | A3에서 불필요하게 실점 |
| D4 | 공격문서 §6의 belief 수치(0.093 vs 0.073)가 최신 상태가 아님 | 공격 자체가 무효화됨 |
| D5 | 방어문서가 harden_s3의 **arm 간 샘플 셋 불일치(겹침 27%)**를 지적하지 않음 | A5-2 "paired CI 재계산"이 그대로는 불가능. `harden_paired`가 필요한 이유 |

---

## 3. 연구 의의와의 부합 — **여기가 가장 큰 문제다**

### 3.1 논문이 스스로 약속한 ablation 두 개

`main.tex`를 직접 확인했다. 본문이 명시적으로 약속한 ablation이 둘 있고, **둘 다 계획에서
제대로 다뤄지지 않는다.**

**약속 1 — ρ=0 대조군 (`main.tex` L205)**

> "Because the interleaving inserts anchor steps without replacing any retrospective updates,
> the amount of trace supervision is identical across replay ratios, and **the effect of the
> anchor can be isolated by an ablation with ρ=0 in the experiments.**"

현재 fp 코호트에 ρ=0 arm은 **없다**. 있는 것은 ρ=0.15 두 개(`sft_r15`, `sft_r15_c`)와
ρ=0.30 하나(`sft_r30_c`)다. 파일럿의 `sft_r0`는 3인칭 프롬프트·다른 코호트라 대체 불가다.

본문이 약속한 실험이 비어 있는 것은 어떤 리뷰어 공격보다 치명적이다. 심사자는 "저자가 하겠다고
쓴 것을 안 했다"를 가장 낮은 비용으로 지적할 수 있다.

**비용은 사실상 없다.** `sft_v2.py:115`의 `total = ceil(len(chosen)/accum/(1−ρ))`에 따라
ρ=0이면 스텝이 **294개**로 가장 적고, CE 마이크로스텝(3.305s)이 0이라 전부 SFT
마이크로스텝(0.393s)이다.

```
294 스텝 × 8 마이크로 × 0.393s × 1.09(오버헤드) ≈ 17분
```

여기에 battery 10분 + harden 10분을 더해도 **37분**이면 본문 약속이 닫힌다. 그리고
`sft_r15_c`(ρ=0.15)·`sft_r30_c`(ρ=0.30)와 합치면 **3점 ρ 곡선**이 되어 ablation의 격이
올라간다. 세 arm이 같은 `chosen_train.jsonl`을 공유하므로 단일변수도 보장된다.

**약속 2 — K ablation (`main.tex` L289, `tab:kablation`)**

> "This ablation **directly validates one of this work's core claims**---that the boundary
> materially affects judgment."

본문이 *core claim의 유일한 직접 증거*라고 부르는 표가 placeholder다. 방어문서는 이를
T3-4로 두고 우선순위 **6번**에 배치했다.

**이것도 대부분 무료다.** `context_val.jsonl`에 `wm_scores`와 `gt_rank`(1-index)가 있어
coverage 축은 GPU 0으로 즉시 나온다.

| K | Coverage@K | n_covered |
|---|---|---|
| 3 | 21.97% | 1,170 |
| 5 | 30.34% | 1,616 |
| 10 | 43.43% | 2,313 |

(`gt_rank ≤ 10` = 43.43%로 `battery.py:40`의 `pool_coverage=0.4343` 및 논문 인용 43.4%와
정확히 일치함을 확인했다. 이 부분은 문제없다.)

정확도 축만 GPU가 필요하다. `battery.py`에 후보 절단 인자가 없으므로 `--top_k` 한 줄
추가가 선행된다. 최종 arm + θ_CE × K∈{3,5} = 4회 × 10분 = **40분**.

### 3.2 그래서 "핵심 ablation으로만 구성돼 있는가?"

**아니다.** 현재 계획의 무게중심은 리뷰어 방어(embodiment 식별)에 있고, 논문 자신의 주장
구조를 검증하는 ablation은 뒤로 밀려 있다.

| 성격 | 항목 | 현재 우선순위 |
|---|---|---|
| **논문 본문이 약속** | ρ=0 대조군 | **계획에 없음** |
| **논문이 "core claim 직접 증거"라 명시** | K ablation | Tier 3 / 6번 |
| 리뷰어 방어 (제목 유지용) | Tier 1 10조건 스윕 | Tier 1 / 4번 |
| 리뷰어 방어 (통제 보강) | T3-1 matched negative, T3-2 3-seed | Tier 3 / 7번 |

지면 제약을 고려하면 이 배분은 더 나빠진다. Results 초안은 이미 533행에 표 4개·그림 2개다.
**4 arm × 10 조건 표는 물리적으로 들어가지 않는다.** 반면 ρ 3점 곡선과 K 3점 표는 각각
작은 표 하나로 끝나고, 둘 다 본문이 이미 지면을 배정해 둔 자리가 있다.

---

## 4. 축소안 — 핵심만 남기기

시간과 지면이 부족하다는 전제에서, 아래 배분을 권고한다.

### 4.1 승격 (본문 필수)

| 항목 | 비용 | 근거 |
|---|---|---|
| **ρ=0 arm 학습 + battery + harden** | **37분** | `main.tex` L205 약속. ρ 3점(0/0.15/0.30) 곡선 완성 |
| **K ablation** (coverage 무료 + acc 4회) | **40분** + 코드 1줄 | `main.tex` L289가 "core claim 직접 증거"로 명시 |
| **G-NH 검정력 서술** | 0 | §1.4. A3 실점을 서술만으로 줄임 |
| **T0-3 U_g·D_g·correct-switch 보고** | 0 | 이미 측정됨 |
| **T0-6 full-set 12.5% 병기 / T0-7 Figure 1(b) 실측점** | 0 | A2·A3 무료 방어 |
| **T0-9 문구 하향 일괄** (특히 Q3) | 0 | 안 하면 반드시 잡힘 |

### 4.2 유지하되 축소 (Tier 1 스윕 10조건 → 4조건)

| 조건 | 판정 | 사유 |
|---|---|---|
| C3 no-image | **유지** | 제목을 결정하는 유일한 실험. 전 arm 필요 |
| C6 other-video history | **유지** | 의미 사용 vs OOD 분리. Q2를 닫는 유일한 실험 |
| C4 no-image ∧ no-history | **유지** | 2×2 factorial의 교호작용 셀. 이게 없으면 C3가 factorial이 아니다 |
| C2 no-history | **유지** (gx/최종 arm만 추가) | 이미 4 arm 중 3개 완료 |
| C5 shuffled history | **강등** | C6의 약한 버전. C6가 통과하면 불필요, 실패하면 C5도 실패 |
| C7 reversed history | **삭제** | C5와 중복 |
| C8 other-video image | **강등** | C3가 제목 판단에 충분. 프레임 캐시 미스로 비용도 큼(§7.2) |
| C9~C12 dose-response | **삭제 또는 2 arm 한정** | 4 arm × 4 셀 = 16회로 스윕 비용의 40%. 판정을 바꾸지 않음 |

4조건 × 4 arm = 16회 × 10분 ≈ **2.7시간**(10조건 40회 6.7시간 대비 60% 절감).

### 4.3 삭제 또는 limitation 대체

| 항목 | 판정 |
|---|---|
| T3-2 3-seed (10.4h, §6 정정치) | **limitation 명시로 대체.** 제출 전 불가 |
| T3-6 history-free WM 후보 재생성 | **삭제.** Step-1 재학습 필요. 방어문서 §8 열린판단 2도 같은 결론 |
| T3-8 frontier VLM baseline | **삭제.** 시간 대비 반박력 낮음 |
| T3-5 held-out 전체 5,326 재평가 | **강등.** covered 2,313 재평가만으로 A2 상당 부분 커버 |

---

## 5. 실행 계획의 오류

### 5.1 즉시 실패하는 명령 — `paired_boot.py` 플래그 부재 (방어문서 §9 순서2)

```bash
$PY tools/paired_boot.py --run runs/cesft_v2_fp --arm_a theta_ce --arm_b cand_free \
    --common_set --malformed_as_incorrect \        # ← 두 인자 모두 존재하지 않음
    --out .../paired_commonset_ce_vs_gtonly.json
```

`tools/paired_boot.py`의 `add_argument`는 `--run --arm_a --arm_b --gate --metric --n_boot
--seed --out` 뿐이다. **T0-1(estimand 통일)은 플래그가 아니라 코드 수정 작업이다.**
A2/Q5가 리뷰어 핵심 질문임을 감안하면 이 오분류의 대가가 크다 — 무료 작업으로 잡아둔 것이
실제로는 구현 항목이다.

### 5.2 중복 셀 — C9 `last0` = C2 `nohist`

방어문서 §7 Tier 1 표에서 C9~C12는 "최근 0/1/3/7/all"인데, **최근 0개 = no-history**로
C2와 동일하다. `--mode last0`과 `--mode nohist`가 같은 셀을 두 번 돌린다. dose-response를
유지한다면 1/3/7만 남겨야 한다.

### 5.3 미구현 도구 (문서도 인정하나 §9는 실행형으로 기술)

| 도구 | 상태 |
|---|---|
| `tools/text_baselines.py` | **없음** — T0-5에서 "신규"로 명시했으나 §9는 바로 실행하는 형태 |
| `tools/oom_opt/perturb_eval.py` | **없음** — "구현 후"라 단서가 있으나 Tier 1 전체가 여기 의존 |
| `tools/did_history.py` | ✅ 존재. `--arm_a/--arm_b` 인자 확인 |
| `tools/strip_metrics.py` | ✅ 존재. 단 `--interaction --cluster` 인자는 미확인 |

**T0-5(텍스트-only baseline)가 경로 A/B 판단의 1차 신호라고 방어문서 스스로 밝혔는데
(§7 "T0-5를 가장 먼저 돌린다"), 그 도구가 없다.** 이것이 전체 계획의 임계 경로다.

### 5.4 해소된 우려 — gx 코호트 동일성

방어문서 §9의 "선행 확인: `runs/cesft_v2_fp_gx/data/context_val.jsonl`이 `runs/cesft_v2_fp`와
동일 코호트인지" — 확인 완료. **심볼릭 링크로 동일 파일**이다.

```
runs/cesft_v2_fp_gx/data/context_val.jsonl -> .../runs/cesft_v2_fp/data/context_val.jsonl
```

`RETRO3_RUNS=runs/cesft_v2_fp_gx`로 그대로 실행해도 paired 비교가 성립한다.
단 `overrides.json`(`{"eval_covered_only": true}`) 존재도 함께 확인할 것 — gx 재실행 때
이것을 빠뜨려 잘못된 표본을 평가한 전례가 있다.

---

## 6. 시간 추정 정정 (adapter_step 타임스탬프 실측)

방어문서 §1.4의 앵커 중 학습 항목에 과대가 있다. **CE 학습이 후보 채점 유무로 2배 이상
갈리는데 하나로 묶여 있다.**

| 학습 | 스텝 | 방어문서 | **실측** | 차이 |
|---|---|---|---|---|
| `cand_free` (GT-only CE) | 522 | "select-CE 약 4~5h" | **95분** | 3배 과대 |
| `theta_ce` (candidate CE) | 522 | 동상 | **218분 (3.6h)** | 근사 |
| `sft_r15` (retro SFT) | 368 | 44분 | **46분** | 정확 |
| `sft_r15_c` (retro SFT) | 347 | — | **41분** | — |
| `sft_r30_c` (ρ=0.30) | 420 | — | **69분** | — |
| **ρ=0 arm (신규)** | **294** | — | **~17분 (추정)** | — |

단가: SFT 마이크로스텝 **0.393s**, CE 마이크로스텝 **3.305s** (8.4배). `accum=8` 확인.

**파생 정정:**

- T3-2 3-seed: 방어문서 20h → **실측 기반 10.4h** (2 seed × (3.6h + 1.6h)). 여전히 제출 전
  불가지만 절반이다.
- T3-1 matched negative 3 arm: `rand_cand`/`freq_cand`는 candidate-CE(3.6h), `gt_inbatch`는
  GT-only급(1.6h) → **~9h**. 방어문서 14h는 다소 과대.
- Tier 1 "~8 GPU-h": 조건당 battery급 10분 가정은 타당하나 **모델 로드 오버헤드와 프레임
  캐시 미스가 빠져 있다**(§7.2). 40회 실행이면 로드만 +30~60분.

---

## 7. OOM·실행 리스크

### 7.1 OOM — 전례가 있고, 원인과 방어가 코드에 있다

`outputs/.../theta_ce/train_log.jsonl.bak_serverB_oom`(07-25 19:51)이 실제 OOM 흔적이다.
수정은 `vlm.py`의 **cache-first + `close_readers()`**(L103)로 들어가 있고, 서버 B 스크립트
헤더도 이 커밋을 전제로 명시한다.

**신규 `perturb_eval.py`에서 반드시 지킬 것:**

1. **`vlm.close_readers()`를 반드시 호출한다.** `projection.py:175`, `battery.py`가 모두
   종료 시 호출한다. 빠뜨리면 decord 리더가 누적되어 07-25 OOM이 재현된다. 특히 40회 연속
   실행이라 누수가 있으면 후반부에서 반드시 터진다.
2. **모드 간 프로세스를 분리한다.** `harden_paired.py`가 arm마다 프로세스를 나눈 이유가
   "GPU 메모리 반납"이다(§헤더). 40 셀을 한 프로세스에서 돌리지 말 것.
3. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`를 유지한다(기존 체인 전부 사용 중).

### 7.2 프레임 캐시 — `othervideo_image`가 캐시를 무력화한다

`FRAME_CACHE_DIR=runs/cesft_v2/frame_cache`는 **평가 코호트 비디오의 프레임만** 담고 있다.
방어문서 §9는 "재추출 금지"를 명시했으나, C8(`othervideo_image`)은 정의상 **다른 비디오의
프레임**을 요구한다.

- 도너를 **평가 코호트(heldout 5,326) 안의 비디오로 제한**하면 캐시 히트가 유지된다.
- 제한하지 않으면 1,000샘플 전부 디코드가 발생해 10분짜리가 40~60분이 된다. 40 셀 계획에서
  이 한 셀이 전체 일정을 흔든다.
- **C6(`othervideo_history`)는 텍스트만 바꾸므로 이 문제가 없다.** C8을 강등하고 C6를
  유지하라는 §4.2 판단의 실행상 근거이기도 하다.

### 7.3 no-image 구현 리스크

방어문서 §7의 판단(blank 이미지 + 마스킹 system prompt)이 옳다. 근거를 보강하면:

- `battery.py`는 `wm_top1`을 `rec["wm_scores"]`에서 계산하므로 **이미지를 지워도 후보 집합과
  WM Top-1은 불변**이다. 정책 경로만 끊긴다 — 이것이 A1이 요구하는 개입이다.
- 프레임 수 8은 시간 계약과 함께 불변 제약이므로, **프레임 개수를 줄이지 말고 내용만 blank로**
  채워야 한다. 개수를 바꾸면 프롬프트 포맷 shift가 생겨 A4가 지적한 OOD 반론을 자초한다.

### 7.4 마커·resume

기존 체인의 멱등 규약(`S_PERTURB_{ARM}_{MODE}_DONE`)을 따르라는 지시는 옳다. 추가로:

- **`overrides.json`을 새 run dir에 반드시 복사한다.** gx 재실행 때 이것을 빠뜨려 covered
  410건만 평가된 전례가 있다. 심볼릭 링크 구성 시 데이터·eval 링크만 걸고 이 파일을 잊기 쉽다.
- 스텝 체크포인트를 남기려면 `CKPT_KEEP_STEP_ADAPTERS=1`이 필요하다. gx에는 이것이 빠져
  중간 스텝 어댑터가 없다(최종만 존재).
- `RETRO3_RUNS`가 정확한 환경변수명이다(`RETRO3_RUNS_ABS` 아님). 틀리면 `runs/retro3`를
  읽어 ZeroDivisionError로 죽는다 — 실제 발생한 사고다.

---

## 8. 권장 실행 순서 (수정판)

| 순서 | 작업 | 비용 | 생략 시 |
|---|---|---|---|
| **1** | **ρ=0 arm 학습 + battery + harden** | **37분** | `main.tex` L205 약속 미이행 |
| **2** | **K ablation** (coverage 무료 / `battery --top_k` 1줄 + 4회) | **40분** | 본문이 "core claim 직접 증거"라 부른 표가 placeholder |
| **3** | Tier 0 무료분: T0-3 / T0-6 / T0-7 / T0-9 / T0-10 + **G-NH 검정력 서술** | CPU 수 시간 | A2·A3·A5 무료 실점 |
| **4** | §6 결정규칙 표 커밋 (타임스탬프 고정) | 0 | B6 사후선택 반론 |
| **5** | `tools/text_baselines.py` 구현 + T0-5 실행 | CPU | **경로 A/B 판단 불가 — 임계 경로** |
| **6** | T0-1 estimand 통일 (**코드 수정 필요**, 플래그 아님) | CPU + 구현 | A2/Q5 |
| **7** | Plan B: `harden_paired` 4-arm (base/θ_CE/sft_r15/최종) | 4h (A/B 분업 2h10m) | belief 주장에 arm 간 CI 부재 |
| **8** | 축소 Tier 1: C3/C4/C6 + C2 잔여, 4 arm | 2.7h | 제목의 `embodied` 포기 |
| **9** | Tier 2 잔여 (T2-2 video-disjoint donor, T2-3 hard-negative) | 2h | A5 OOD 반론 잔존 |

**순서 1~4가 2시간이 안 되는데 논문 본문의 두 약속을 닫고 무료 방어를 전부 회수한다.**
현재 방어문서의 순서 1(gx harden/strip 20분)은 최종 arm이 gx가 아닐 가능성이 높으므로
**순서 8의 ablation 재료로 강등**한다.

### 열린 판단

1. **최종 arm 확정 후 Plan B를 돌려야 한다.** `harden_paired.py:190-196`의 공통 셋이 plan
   시점 arm 목록의 교집합으로 고정되고 `:240`이 plan에 없는 arm을 막으므로, **plan은 확장
   불가**다. 최종 arm이 정해지기 전에 돌리면 전량 재실행이다.
2. **ρ 곡선을 3점으로 만들 것인가 4점으로 만들 것인가.** ρ=0/0.15/0.30이 기본이고,
   `sft_r15`(fp 프롬프트, ρ=0.15)를 넣으면 프롬프트 축이 섞이므로 **별도 표로 분리**해야 한다.
3. **`sft_r15_gx`(오버샘플)의 위치.** 정확도 게이트는 통과하되(G-ACC1 PASS) belief 주장이
   없는 arm이다. `sft_r15_c`는 정반대다. 두 축이 서로 다른 개입으로 각각 달성 가능함을
   보이는 재료로 쓸 수 있으나, 지면이 없으면 부록으로 내린다.

---

## 9. 검증 커맨드 부록

```bash
REPO=/mnt/nvme/migration/jihun/EGO_jihun3; cd $REPO
PY=/mnt/nvme/migration/jihun/envs/miniforge3/envs/eve-cu124/bin/python
export PYTHONPATH=$REPO/src HF_HOME=/mnt/nvme/cache
export FRAME_CACHE_DIR=$REPO/runs/cesft_v2/frame_cache
export RETRO_NEXT_GAP_TEXT="after the current action ends"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CKPT_KEEP_STEP_ADAPTERS=1
CFG=configs/step2_retrospection/cesft_v2_fp.yaml
ADAPT=outputs/step2_retrospection/cesft_v2_fp
```

### 순서 1 — ρ=0 대조군 (본문 L205 약속)

```bash
# run dir 구성: 타깃은 최종 arm과 동일한 chosen_train.jsonl 을 공유해야 단일변수가 성립
R=runs/cesft_v2_fp_r00
mkdir -p $R/{data,logs,status,markers,eval,probe}
cp runs/cesft_v2_fp_c/overrides.json $R/            # ← 절대 빠뜨리지 말 것
ln -sf $REPO/runs/cesft_v2_fp_c/data/chosen_train.jsonl $R/data/
for f in context_train.jsonl context_val.jsonl train_subset.json; do
  ln -sf $REPO/runs/cesft_v2_fp/data/$f $R/data/$f; done
for a in base cand_free theta_ce sft_r15; do
  ln -sf $REPO/runs/cesft_v2_fp/eval/$a.records.jsonl $R/eval/$a.records.jsonl; done

export RETRO3_RUNS=$R
$PY -m ego.step2_retrospection.train.sft_v2 --config $CFG --run_name sft_r00_c \
    --init_adapter $ADAPT/theta_ce/adapter --ce_replay_rho 0.0 \
    --epochs 1 --seed 42 --ckpt_every 50 --resume auto     # ~17분
$PY -m ego.step2_retrospection.eval.battery --config $CFG --arm sft_r00_c \
    --adapter $ADAPT/sft_r00_c/adapter --eval_n 1000        # ~10분
$PY -m ego.step2_retrospection.eval.harden_s3 --config $CFG --arm sft_r00_c \
    --adapter $ADAPT/sft_r00_c/adapter --n 400              # ~10분
```

### 순서 2 — K ablation

```bash
# coverage 축: GPU 0
$PY - <<'EOF'
import json
h=[json.loads(l) for l in open('runs/cesft_v2_fp/data/context_val.jsonl') if l.strip()]
h=[r for r in h if r['split']=='heldout']
for K in (3,5,10):
    c=sum(1 for r in h if r['gt_rank']<=K)   # gt_rank 는 1-index
    print(f'Coverage@{K} = {100*c/len(h):.2f}%  n_covered={c}')
EOF

# 정확도 축: battery.py 에 --top_k 추가 후 (rec["candidates"] 를 wm_scores 상위 K 로 절단)
for K in 3 5; do
  for arm in theta_ce <최종arm>; do
    $PY -m ego.step2_retrospection.eval.battery --config $CFG --arm ${arm}_k$K \
        --adapter $ADAPT/$arm/adapter --eval_n 1000 --top_k $K
  done
done
```

### 검증에 쓴 명령 (재현용)

```bash
# 학습 실소요 — adapter_step 타임스탬프
for a in cand_free theta_ce sft_r15 sft_r15_c; do
  d=$ADAPT/$a; f=$(ls -d $d/adapter_step* | sed 's/.*step//' | sort -n | head -1)
  l=$(ls -d $d/adapter_step* | sed 's/.*step//' | sort -n | tail -1)
  echo "$a: $(( ($(stat -c %Y $d/adapter_step$l)-$(stat -c %Y $d/adapter_step$f))/60 ))분 / $((l-f))스텝"
done

# 마이크로스텝 단가
$PY -c "
import json,collections
g=collections.defaultdict(list)
for l in open('$ADAPT/sft_r15_c/train_log.jsonl'):
    r=json.loads(l)
    if isinstance(r.get('sec'),(int,float)): g[r.get('tag')].append(r['sec'])
for k,v in g.items(): print(k, len(v), sum(v)/len(v))"

# harden_s3 arm 간 샘플 겹침
$PY -c "
import json
S={a:{r['sample_id'] for r in json.load(open(p))} for a,p in [
 ('theta_ce','runs/cesft_v2_fp/eval/theta_ce.harden_s3.records.json'),
 ('sft_r15_c','runs/cesft_v2_fp_c/eval/sft_r15_c.harden_s3.records.json')]}
a,b=S['theta_ce'],S['sft_r15_c']; print(len(a&b), '/', len(a|b))"
```

---

## 10. 참조

- 공격문서: `docs/paper/2026-07-26_aaai_reviewer_attack_handoff.md`
- 방어문서: `docs/paper/2026-07-26_aaai_reviewer_defense_plan_handoff.md`
- 오늘 밤 실행 종합: `docs/experiments/2026-07-26_results_owner_handoff_fp_run_and_planC.md`
- 본문 약속: `../EGO_paper/EGO_AAAI27_EN/main.tex` L205 (ρ=0), L289·L302 (`tab:kablation`)
- 산출물: `runs/cesft_v2_fp/eval/`, `runs/cesft_v2_fp_gx/eval/`, `runs/cesft_v2_fp_c/eval/`,
  `runs/cesft_v2_fp_r30/eval/`
