# AAAI reviewer defense — 대응 계획 v2 (감사 반영 개정판)

- 작성일: 2026-07-27
- **v1을 대체한다**: `docs/paper/2026-07-26_aaai_reviewer_defense_plan_handoff.md` (superseded)
- 반영한 감사: `docs/paper/2026-07-27_ablation_plan_audit_handoff.md`
- 공격 원문: `docs/paper/2026-07-26_aaai_reviewer_attack_handoff.md`
- 방어 대상: `2026-07-26_embodied_reasoning_results.tex`, `../EGO_paper/EGO_AAAI27_EN/main.tex`
- 제약: 제출까지 시간 부족, **지면 부족** (Results 초안 533행 / 표 4 · 그림 2)

---

## 0. 개정의 핵심

v1은 "리뷰어 공격 14건을 비용순으로 막는 계획"이었다. 감사의 가장 무거운 지적은
**그 무게중심이 틀렸다**는 것이다.

> 논문 본문이 스스로 약속한 ablation 두 개(ρ=0 대조군, K ablation)가 각각
> **아예 없거나 우선순위 6번**인데, Tier 1의 10조건 스윕은 대부분 리뷰어 방어용 add-on이다.

이 지적을 전면 수용한다. 심사자는 "저자가 하겠다고 쓴 것을 안 했다"를 가장 낮은 비용으로
지적할 수 있고, 그것은 어떤 embodiment 반론보다 치명적이다. **새 순서 1·2가 그 둘이며,
합쳐서 77분이다.**

동시에 v1 작성 이후 실행된 arm들이 **A5(belief)의 전제를 뒤집었고**, 이번 개정에서 확인한
`sft_r30_c` 완주로 **최종 arm이 확정됐다**. 두 변화가 계획을 다시 바꾼다.

---

## 1. v1의 오류 — 명시적 정정

| # | v1의 서술 | 정정 |
|---|---|---|
| E1 | §8-4 "`sft_r15_c`/`sft_r30_c`를 ρ=0 대조군으로 승격 가능한지 확인" | **사실 오인.** 둘은 ρ=0.15 / 0.30이고 프롬프트도 fp와 다르다. **ρ=0 arm은 존재하지 않는다.** 파일럿 `sft_r0`는 3인칭 프롬프트·다른 코호트라 대체 불가 |
| E2 | §1.3 표에서 `sft_r15_c`를 "미사용 adapter"로 분류 | **`sft_r15_c`는 battery·strip·harden·freegen 완주한 완결 arm**이며, §2.2에서 최종 arm으로 확정됐다 |
| E3 | 순서 1 = "gx의 harden/strip 20분이 비용 대비 이득 최대" | **강등.** gx는 최종 arm이 아니다. gx 개입 실험은 ablation 재료로만 의미가 있다 |
| E4 | §9 커맨드 `paired_boot.py --common_set --malformed_as_incorrect` | **두 플래그 모두 없다.** `--run --arm_a --arm_b --gate --metric --n_boot --seed --out` 뿐. T0-1은 무료 재집계가 아니라 **코드 수정 작업**이다 (직접 확인함) |
| E5 | Tier 1 조건 C9 `last0` | **C2 `nohist`와 동일 셀.** dose-response를 남긴다면 1/3/7만 |
| E6 | §1.4 "select-CE 523 steps ≈ 4\~5h" | **후보 채점 유무로 2배 이상 갈린다.** `cand_free` 95분, `theta_ce` 218분. 3-seed 20h → **10.4h**, matched negative 14h → **~9h** |
| E7 | A3를 "서술 재정의"로만 대응 | **G-NH 실패는 검정력 부족이다**(§2.3). 그렇게 쓰면 실점 폭이 크게 준다 |
| E8 | A5-2 "paired CI를 records에서 재계산" | harden_s3는 **arm마다 샘플 셋이 달라 겹침 27%**다. 그대로는 불가능하고 `harden_paired`가 필요하다 |

v1에서 **유지되는 것**: 공격 14건 → 대응 매핑(누락 없음), 유형 분류(RPT/RUN/AGG/EXP/TXT),
그리고 **§6 사전등록 결정규칙 표와 "Tier 1 착수 전 커밋" 지시**. 감사도 이 부분을 계획의
가장 좋은 부분으로 평가했고, B6(사후선택) 방어까지 겸하므로 그대로 간다.

---

## 2. 계획을 바꾼 새 사실

### 2.1 A5(belief)의 전제가 뒤집혔다 — 가장 약한 축이 가장 강한 축이 됐다

공격문서 §6은 belief를 최대 약점으로 지목했다: *"Cand.-CE `0.093` vs Base `0.073`,
차이 `0.020`뿐이고 arm 간 CI도 없다."* 이 지적의 **사실 근거가 사라졌다.**

| arm | belief 인과 | para 대조 | n_flip |
|---|---|---|---|
| base | 0.0725 | 0.0075 | 32 |
| cand_free (GT-only) | 0.0850 | 0.0075 | 37 |
| theta_ce (Cand.-CE) | 0.0925 | 0.0025 | 38 |
| sft_r15 | 0.0975 | 0.0125 | 44 |
| **sft_r15_c** | **0.3825 [.3325, .4325]** | 0.0425 | **170** |

그리고 공격문서가 "없다"고 지적한 두 통제가 실측으로 채워졌다.

**(a) arm 간 paired CI** (Q10이 요구한 것). 3-arm 공통 78건 기준:

| 비교 | paired Δflip | CI95 |
|---|---|---|
| C − θ_CE | **+0.3205** | [+0.1923, +0.4487] |
| C − sft_r15 | +0.3077 | [+0.1795, +0.4359] |
| sft_r15 − θ_CE | +0.0128 | [−0.0897, +0.1154] |

McNemar 불일치 30 대 5, 정확검정 **p = 2.2×10⁻⁵**. `sft_r15 − θ_CE`가 귀무라는 점이 중요하다 —
효과가 SFT 자체가 아니라 **프롬프트 수정에서 왔음**이 분리된다.

**(b) OOD·정보량 교란 배제** (공격문서 §6 "hybrid prefix가 생성분포 밖일 수 있다"에 대한 직접 반박).
own/swap belief의 어휘 거리별 flip:

| arm | 거리 하위 33% | 중간 | 거리 상위 33% |
|---|---|---|---|
| `sft_r15` | 0.083 | 0.105 | **0.142** ↗ |
| `sft_r15_c` | **0.459** | 0.414 | 0.403 → |

`sft_r15`는 거리에 비례해 오르는 **정보량 반응** 패턴이고, C는 **거리와 무관하게 평평**하다.
자기 belief와 거의 같은 문자열을 주입해도 46%가 뒤집힌다. 주입 문자열의 정보량이 원인이면
나올 수 없는 모양이다.

> **A5 대응은 "약점 방어"에서 "강점 제시"로 성격이 바뀐다.** v1의 RPT+RUN+EXP 7항목 중
> "sensitivity가 base와 구분 안 된다"는 전제로 짜인 부분은 폐기한다. T2-2(video-disjoint
> donor)·T2-3(hard-negative swap)의 필요성은 남는다 — 다만 이제 **강한 결과를 굳히는 통제**이지
> 없는 효과를 찾는 실험이 아니다.

### 2.2 최종 arm 확정 — `sft_r15_c` (이번 개정에서 확인)

감사는 `sft_r30_c` 평가 종료(01:58 KST경)를 최종 arm 확정 시점으로 남겨뒀다.
**완주했고(마커 `S_STRIP_SFT_R30_C_DONE` 07-26 17:06 UTC = 01:57 KST), 결과를 직접 읽었다.**

| arm | ρ | SelAcc | G₁ | G₂ | malformed | belief 인과 | U_g |
|---|---|---|---|---|---|---|---|
| **`sft_r15_c`** | **0.15** | **28.5%** | **41.7%** | **24.3%** | **2.8%** | **0.3825** | **0.0405** |
| `sft_r30_c` | 0.30 | 27.6% | 40.5% | 23.5% | 4.5% | 0.3700 | 0.0274 |

**ρ=0.15가 전 축에서 ρ=0.30을 이긴다.** 따라서 최종 arm은 `sft_r15_c`로 확정하고,
감사의 열린 판단 1(Plan B 착수 가능 시점)이 해소된다 — **지금 돌릴 수 있다.**

부수 효과: ρ=0을 추가하면 **0 / 0.15 / 0.30 3점 곡선**에서 0.15가 정점인 형태가 나온다.
`main.tex` L205가 약속한 "anchor 효과의 격리"가 단조 곡선이 아니라 **최적점이 있는 곡선**으로
나오면 ablation의 설명력이 더 커진다.

### 2.3 G-NH 실패는 방법 실패가 아니라 검정력 부족이다

`tools/paired_boot.py`의 게이트는 `selacc_ok = (SelAcc Δ의 CI 하한 ≥ −0.01)`이다.
이 코호트(n≈930, 클러스터 86)의 **CI 반폭이 4.06pp**다. 하한이 −1pp 위로 오려면

```
point − 0.0406 ≥ −0.01   →   point ≥ +3.06pp
```

즉 **"−1pp 마진의 비열등"을 선언하려면 θ_CE를 3pp 이상 이겨야 한다.** 비열등 게이트가
사실상 우월 게이트로 작동했다. 반폭을 0.01로 줄이려면 클러스터가 약 1,400개 필요하다(현재 86).
세 arm 전부 같은 벽에 걸린 것이 방증이다.

| arm | SelAcc Δ vs θ_CE | GADR Δ | G-NH |
|---|---|---|---|
| `sft_r15` | −3.00pp [−6.37, +0.83] | −4.94pp [−10.59, +0.70] | FAIL |
| `sft_r15_gx` | −0.64pp [−3.88, +2.91] | −2.53pp [−6.98, +2.29] | FAIL |
| `sft_r15_c` | −1.18pp [−5.24, +2.57] | −3.39pp [−7.40, **−0.27**] | FAIL |
| `sft_r30_c` | −1.74pp [−5.68, +1.85] | — | FAIL |

**이 사실을 논문에 명시하면 A3의 실점 폭이 크게 준다.** "사전등록 마진이 설계 해상도보다
작았다"는 것은 정직한 방법론적 한계이지, 방법이 성능을 보존하지 못했다는 증거가 아니다.
사전등록 실패를 숨기지 않으면서 해석만 정확히 하는 것이므로 §5의 정직성 원칙과도 어긋나지 않는다.

**단, GADR은 다르다.** `sft_r15_c`의 GADR Δ는 CI 상한이 **−0.27pp로 0을 배제**한다.
이건 검정력 문제가 아니라 **실재하는 유의한 손실**이며 그대로 한계로 보고한다.
(원인: WM 오답 구간이 평가셋의 76%. SFT는 G₁ 유지력 +5.5pp를 사고 GADR −3.8pp를 파는 교환을 한다.
`0.242×5.5 + 0.758×(−3.8) = −1.55pp`로 관측 SelAcc Δ −1.18pp와 일치.)

---

## 3. 무게중심 재조정 — 논문 자신의 약속이 먼저다

### 3.1 약속 1 — ρ=0 대조군 (`main.tex` L205)

> "the effect of the anchor can be isolated by **an ablation with ρ=0 in the experiments.**"

**현재 fp 코호트에 ρ=0 arm은 없다.** 있는 것은 ρ=0.15 둘, ρ=0.30 하나다.

비용은 사실상 없다. `sft_v2.py`의 `total = ceil(len(chosen)·epochs/accum/(1−ρ))`에 따라
ρ=0이면 스텝이 **294개**로 가장 적고, CE 마이크로스텝(3.305s)이 0이라 전부 SFT
마이크로스텝(0.393s)이다.

```
294 스텝 × 8 마이크로 × 0.393s × 1.09(오버헤드) ≈ 17분
```

battery 10분 + harden 10분을 더해 **37분**이면 본문 약속이 닫히고, `sft_r15_c`·`sft_r30_c`와
합쳐 **3점 ρ 곡선**이 된다. 세 arm이 같은 `chosen_train.jsonl`을 공유하므로 단일변수도 보장된다.

### 3.2 약속 2 — K ablation (`main.tex` L289, `tab:kablation`)

> "This ablation **directly validates one of this work's core claims**---that the boundary
> materially affects judgment."

본문이 *core claim의 유일한 직접 증거*라고 부르는 표가 placeholder다. v1은 이것을
우선순위 6번에 뒀다. **대부분 무료다** — `context_val.jsonl`의 `wm_scores`·`gt_rank`로
coverage 축은 GPU 0으로 즉시 나온다.

| K | Coverage@K | n_covered |
|---|---|---|
| 3 | 21.97% | 1,170 |
| 5 | 30.34% | 1,616 |
| 10 | 43.43% | 2,313 |

(`gt_rank ≤ 10` = 43.43%가 `battery.py`의 `pool_coverage=0.4343` 및 논문 인용 43.4%와 일치함을 확인.)

정확도 축만 GPU가 필요하고, `battery.py`에 후보 절단 인자가 없으므로 **`--top_k` 한 줄 추가**가
선행된다. `theta_ce` + `sft_r15_c` × K∈{3,5} = 4회 × 10분 = **40분**.

### 3.3 지면 제약이 이 재조정을 강화한다

Results 초안은 이미 533행에 표 4개·그림 2개다. **4 arm × 10 조건 표는 물리적으로 안 들어간다.**
반면 ρ 3점 곡선과 K 3점 표는 각각 작은 표 하나로 끝나고, **본문이 이미 지면을 배정해 둔
자리가 있다.**

---

## 4. Tier 1 축소 — 10조건 → 4조건

| 조건 | 판정 | 사유 |
|---|---|---|
| **C3 no-image** | **유지** | 제목을 결정하는 유일한 실험. 전 arm 필요 |
| **C4 no-image ∧ no-history** | **유지** | 2×2의 교호작용 셀. 이게 없으면 C3가 factorial이 아니다 |
| **C6 other-video history** | **유지** | 의미 사용 vs OOD 분리. Q2를 닫는 유일한 실험. 텍스트만 바꾸므로 프레임 캐시 문제도 없다 |
| **C2 no-history** | **유지** (최종 arm만 추가) | 4 arm 중 3개 이미 완료 |
| C5 shuffled history | 강등 | C6의 약한 버전. C6가 통과하면 불필요, 실패하면 C5도 실패 |
| C7 reversed history | **삭제** | C5와 중복 |
| C8 other-video image | 강등 | C3가 제목 판단에 충분. **프레임 캐시 미스로 10분짜리가 40\~60분**이 된다(§6.2) |
| C9\~C12 dose-response | **삭제 또는 2 arm 한정** | 4 arm × 4 셀 = 16회로 스윕 비용의 40%인데 판정을 바꾸지 않는다. 남긴다면 `last0`을 빼고 1/3/7만 |

**4조건 × 4 arm = 16회 × 10분 ≈ 2.7시간** (10조건 40회 6.7시간 대비 60% 절감).

### 삭제 · limitation 대체

| 항목 | 판정 |
|---|---|
| 3-seed (10.4h) | **limitation 명시로 대체.** 제출 전 불가 |
| history-free WM 후보 재생성 | **삭제.** Step-1 재학습 필요 |
| frontier VLM baseline | **삭제.** 시간 대비 반박력 낮음 |
| held-out 전체 5,326 재평가 | **강등.** covered 2,313 재평가만으로 A2 상당 부분 커버 |

---

## 5. 사전등록 결정규칙 (v1에서 유지 + 개정)

Tier 1 착수 전에 커밋해 타임스탬프를 남기고 논문 부록에 그대로 싣는다.

| 유지하려는 주장 | 식별 실험 | 반증조건 → 강등 후 표현 |
|---|---|---|
| embodied **visual** reasoning | image × history 2×2 (C3·C4) | no-image에서 Cand.-CE−GT-only 우위 CI가 0을 제외하고 유지 → *candidate-aligned, trajectory-conditioned selection* |
| history의 **의미적** 사용 | other-video history (C6) | 의미붕괴 손실이 no-history 손실과 구분 불가 → *inference-time history ablation* |
| **state-bearing belief** | ~~U_g 유의성~~ → **video-disjoint donor + hard-negative swap** | 거리-무관 평탄성이 무너지거나 disjoint donor에서 효과가 사라지면 → *action-sensitive generated prefix* |
| replay anchor의 효과 | **ρ ∈ {0, 0.15, 0.30} 곡선** | ρ=0이 ρ=0.15와 구분되지 않으면 → anchor 주장 철회, 본문 L205 서술 수정 |
| **boundary가 판단을 좌우** | **K ∈ {3, 5, 10}** | K에 따라 SelAcc가 단조이거나 무반응이면 → "boundary materially affects judgment" 하향 |
| candidate exposure가 원인 | rand_cand / freq_cand / gt_inbatch | random-candidate CE가 동등 이득 → *negative supervision 효과*로 재서술 |
| WM 경계가 **visual** | history-free WM (삭제) | **미실행 확정** → *observation- and history-conditioned proposal boundary*로 하향 (비용 0, 필수) |

belief 행이 v1에서 바뀐 지점을 주목할 것: 효과가 **이미 크게 관측됐으므로**, 반증조건이
"효과가 있는가"에서 "그 효과가 교란인가"로 이동했다.

**정직한 사전 선언**(v1에서 유지) — no-image에서 Cand.-CE의 우위가 상당 부분 살아남을 가능성이
낮지 않다고 본다. 그 경우 위 규칙대로 **제목·abstract에서 `embodied`를 내리는 선택을
rebuttal이 아니라 본문에서 먼저 한다.**

---

## 6. 실행 리스크 (감사 §7 반영)

### 6.1 OOM — 전례가 있다

`outputs/.../theta_ce/train_log.jsonl.bak_serverB_oom`(07-25 19:51)이 실제 OOM 흔적이다.
수정은 `vlm.py`의 cache-first + `close_readers()`(L103)로 들어가 있다.
**신규 `perturb_eval.py`에서 반드시 지킬 것:**

1. **`vlm.close_readers()`를 반드시 호출한다.** 빠뜨리면 decord 리더가 누적된다.
   16\~40회 연속 실행이라 누수가 있으면 후반부에서 반드시 터진다.
2. **모드 간 프로세스를 분리한다.** `harden_paired.py`가 arm마다 프로세스를 나눈 이유가
   GPU 메모리 반납이다. 한 프로세스에서 전 셀을 돌리지 말 것.
3. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 유지.

### 6.2 프레임 캐시 — C8이 캐시를 무력화한다

`FRAME_CACHE_DIR`는 평가 코호트 비디오의 프레임만 담고 있다. C8(`othervideo_image`)은 정의상
다른 비디오의 프레임을 요구하므로, 도너를 **평가 코호트 안의 비디오로 제한**하지 않으면
1,000샘플 전부 디코드가 발생한다. C8을 강등하고 C6를 유지하는 §4 판단의 실행상 근거다.

### 6.3 no-image 구현

- `battery.py`는 `wm_top1`을 `rec["wm_scores"]`에서 계산하므로 **이미지를 지워도 후보 집합과
  WM Top-1은 불변**이다. 정책 경로만 끊긴다 — A1이 요구하는 개입이 정확히 이것이다.
- **프레임 개수(8)를 줄이지 말고 내용만 blank로 채운다.** 개수를 바꾸면 프롬프트 포맷 shift가
  생겨 A4가 지적한 OOD 반론을 자초한다.

### 6.4 마커 · 환경변수

- **`overrides.json`을 새 run dir에 반드시 복사한다.** gx 재실행 때 빠뜨려 covered 410건만
  평가된 전례가 있다. 심볼릭 링크 구성 시 데이터·eval 링크만 걸고 이 파일을 잊기 쉽다.
- 스텝 체크포인트를 남기려면 `CKPT_KEEP_STEP_ADAPTERS=1`.
- **`RETRO3_RUNS`가 정확한 변수명**이다. 틀리면 `runs/retro3`를 읽어 ZeroDivisionError로 죽는다 —
  실제 발생한 사고다.
- **해소됨**: `runs/cesft_v2_fp_gx/data/context_val.jsonl`은 `runs/cesft_v2_fp`의 **심볼릭 링크**로
  동일 파일이다(직접 확인). v1 §9의 선행 확인 항목은 닫혔다.

### 6.5 미구현 도구 — 임계 경로

| 도구 | 상태 | 영향 |
|---|---|---|
| `tools/text_baselines.py` | **없음** | v1이 "가장 먼저 돌린다"고 한 경로 A/B 판단 신호. **전체 계획의 임계 경로** |
| `tools/oom_opt/perturb_eval.py` | **없음** | Tier 1 전체가 의존 |
| `battery.py --top_k` | **없음** | K ablation 정확도 축이 의존. 1줄 |
| `paired_boot.py`의 estimand 플래그 | **없음** | T0-1은 코드 수정 작업 |
| `tools/did_history.py` | 존재 | — |
| `tools/strip_metrics.py` | 존재 (`--interaction --cluster` 인자는 미확인) | — |

---

## 7. 실행 순서 (개정)

| 순서 | 작업 | 비용 | 생략 시 |
|---|---|---|---|
| **1** | **ρ=0 arm 학습 + battery + harden** | **37분** | `main.tex` L205 약속 미이행 — 가장 싼 공격 표면 |
| **2** | **K ablation** (coverage 무료 / `--top_k` 1줄 + 4회) | **40분** | 본문이 "core claim 직접 증거"라 부른 표가 placeholder |
| **3** | Tier 0 무료분 + **G-NH 검정력 서술** + **A5 강점 재서술** | CPU 수 시간 | A2·A3·A5 무료 실점 |
| **4** | 결정규칙 표 커밋 (타임스탬프 고정) | 0 | B6 사후선택 반론 |
| **5** | `tools/text_baselines.py` 구현 + 실행 | CPU | **경로 A/B 판단 불가 — 임계 경로** |
| **6** | T0-1 estimand 통일 (**코드 수정**, 플래그 아님) | CPU + 구현 | A2 / Q5 |
| **7** | Plan B: `harden_paired` 4-arm (base / θ_CE / sft_r15 / **sft_r15_c**) | 4h (A/B 분업 2h10m) | belief 주장에 arm 간 CI 부재 |
| **8** | 축소 Tier 1: C3 / C4 / C6 + C2 잔여, 4 arm | 2.7h | 제목의 `embodied` 포기 |
| **9** | Tier 2 잔여: video-disjoint donor, hard-negative swap | 2h | A5 OOD 반론 잔존 |

**순서 1\~4가 2시간이 안 되는데 본문의 두 약속을 닫고 무료 방어를 전부 회수한다.**
v1의 순서 1(gx harden/strip)은 순서 8의 ablation 재료로 강등한다.

순서 7은 최종 arm이 `sft_r15_c`로 확정됐으므로 **지금 착수 가능**하다.
`harden_paired.py`의 공통 셋이 plan 시점 arm 목록의 교집합으로 고정되고 plan에 없는 arm을
막으므로 **plan은 확장 불가**다 — 최종 arm이 정해진 지금이 정확한 착수 시점이다.

### 열린 판단

1. **ρ 곡선을 3점으로 낼 것인가.** ρ=0/0.15/0.30이 기본. `sft_r15`(fp 프롬프트, ρ=0.15)를
   넣으면 프롬프트 축이 섞이므로 **별도 표로 분리**한다.
2. **`sft_r15_gx`(오버샘플)의 위치.** 정확도 게이트는 통과하되(G-ACC1 PASS) belief 주장이 없다.
   `sft_r15_c`는 정반대다. 두 축이 서로 다른 개입으로 각각 달성됨을 보이는 재료로 쓸 수 있으나,
   지면이 없으면 부록으로 내린다.
3. **C9\~C12 dose-response 존치 여부.** 삭제가 기본. 남긴다면 2 arm × {1,3,7}로 제한하고
   `last0`은 반드시 뺀다(C2와 중복).

---

## 8. 실행 커맨드

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
  for arm in theta_ce sft_r15_c; do
    $PY -m ego.step2_retrospection.eval.battery --config $CFG --arm ${arm}_k$K \
        --adapter $ADAPT/$arm/adapter --eval_n 1000 --top_k $K
  done
done
```

### 순서 8 — 축소 Tier 1

`tools/oom_opt/perturb_eval.py` 구현 후 (§6.1의 세 규칙 준수):

```bash
export RETRO3_RUNS=runs/cesft_v2_fp
for arm in base cand_free theta_ce sft_r15_c; do
  case "$arm" in base) AD="";; *) AD="$ADAPT/$arm/adapter";; esac
  for mode in noimage nohist_noimage othervideo; do
    $PY tools/oom_opt/perturb_eval.py --config $CFG --arm "$arm" ${AD:+--adapter "$AD"} \
        --mode "$mode" --eval_n 1000 --covered_only     # 프로세스 분리 — 셀마다 새 프로세스
  done
done
# C2(nohist)는 최종 arm만 추가
$PY tools/oom_opt/strip_eval.py --config $CFG --arm sft_r15_c \
    --adapter $ADAPT/sft_r15_c/adapter --eval_n 1000 --covered_only
```

마커 규약 `S_PERTURB_{ARM}_{MODE}_DONE`, 기존 `supervisor.sh` + `run_stage` 멱등 패턴 준수.

---

## 9. 참조

- 감사: `docs/paper/2026-07-27_ablation_plan_audit_handoff.md`
- 공격 원문: `docs/paper/2026-07-26_aaai_reviewer_attack_handoff.md`
- v1 (superseded): `docs/paper/2026-07-26_aaai_reviewer_defense_plan_handoff.md`
- 본문 약속: `../EGO_paper/EGO_AAAI27_EN/main.tex` L205 (ρ=0), L289·L302 (`tab:kablation`)
- 산출물: `runs/cesft_v2_fp/eval/`, `runs/cesft_v2_fp_c/eval/`, `runs/cesft_v2_fp_r30/eval/`,
  `runs/cesft_v2_fp_gx/eval/`
