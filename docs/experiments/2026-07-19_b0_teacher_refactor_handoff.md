# B0 Teacher 2단계 리팩터 핸드오프 — 구현 목표·근거·계획

- **작성:** 2026-07-19
- **대상:** B0 담당자
- **선행 문서:** `2026-07-19_code_review_response_handoff.md` (코드리뷰 회신), `2026-07-19_f0_b0_validation_results_handoff.md` (실측 결과)
- **전제:** F0 확정 작업(클린 WM ∥ GT-only skyline → P1-6 계열)은 별도 트랙으로 선행 중.
  본 리팩터는 그 다음 단계이며, **풀 스케일 B0는 이 리팩터 완료 전에 실행하지 않는 것을 권고**한다.

---

## ⚠ v2 정정 — 통합 구현 핸드오프(리뷰 리드, 2026-07-19) 반영

아래 4가지는 본 문서 초판(v1)의 설계를 **대체**한다. 충돌 시 통합 핸드오프
(`2026-07-19_B0_F0_consolidated_implementation_handoff.md`)가 우선한다.

1. **인과 게이트 변경 (v1 §4·§5 대체)** — "gemini가 belief→GT 자연스러움 판정"은 인과 게이트로 **불충분**
   (GT를 본 판정자는 모호한 belief에도 사후적으로 GT를 연결 가능 — "모순 없음" ≠ "예측력 있음").
   최소 인과 조건은: **teacher가 GT 은닉 상태에서 reasoning/belief/action을 함께 생성하고,
   그 독립 예측이 `canonical(predicted)==canonical(GT)`일 때만 hard PASS.**
   gemini는 **process verifier**(past-grounding·future leakage·과잉 구체성·reasoning-belief 일관성·스키마)로만 사용.
   가능하면 `teacher_action_rank`·`action_margin` 기록, validation subset에서 `causal_delta`(belief mask 시 GT 확률 감소) 측정.
2. **goal 누출 방어 강화 (v1 §5-1 대체)** — 문자열 검사만으론 부족("preparing a salad"는 exact 누출 없이 강한 힌트).
   (a) goal 추출 **입력에서 target action 제외** — future suffix `a_{t+1:}`만 사용 (`assert target ∉ goal_source`),
   (b) goal entity **provenance** 기록(observed_in_history / current_frame / future_only), (c) future-only entity는 상위 추상어로 치환.
3. **G2 selection bias 게이트 신설 (v1 미비)** — hard PASS는 쉬운(G1) 샘플에 편중될 수 있음. B0의 가치는 G2 회복이므로
   **`G2 retention / G1 retention ≥ 0.5`**를 데이터 게이트로 추가하고 retention 통계(G1/G2·GT rank·WM margin 등)를 필수 보고.
   미달 시: G2 teacher rollout 수 증대 → goal 구체성 미세 상향(누출 audit 유지) → 그래도 미달이면 구조 한계로 보고.
4. **단계화 (v1 §5 실행 순서 대체)** — R1: teacher 리팩터만(기존 DPO 유지) → R2: same-teacher **failed trace를 hard negative**로
   추가(문체 shortcut 통제) + prompt별 pair weight 1/N → R3: action auxiliary CE(belief-swap↑인데 action-span<+0.023일 때만).
   성공 기준에 **G2 recovery ↑**·style-only separability probe 추가.

## 1. 한 줄 요약

현행 B0 teacher는 **정답(GT action)을 미리 본 상태로 reasoning/belief를 작성**한다 → chosen이
"사후 합리화" 텍스트가 되어, DPO가 **belief 문체는 가르치지만 belief→action 인과는 가르치지 못한다.**
리팩터는 teacher를 "**정답을 모르는 상태에서 작성 + 실제로 정답에 도달한 trace만 채택**" 구조로 바꾼다.

---

## 2. 근거 — 왜 지금 구조로는 안 되는가 (전부 실측)

### 2-1. MVP 실행 결과 (1,500 프롬프트 × 4 롤아웃, DPO 4,117쌍, 2026-07-19 완료)

| 지표 | FAA (baseline) | B0 | A1 (action-patch 통제군*) |
|---|---|---|---|
| 생성 acc (n=500, @384) | 0.230 | 0.248 | **0.254** |
| G2 (WM-틀림 구간) | 0.325 | 0.342 | 0.333 |
| **③ belief-swap 인과 민감도** | 0.008 | **0.006 (개선 없음)** | 0.012 |
| ④ judge belief_action_link | ~2.0 (만점 포화) | — | — |

\* A1 = chosen을 teacher 투영 대신 "FAA 자기 trace에 GT action만 패치"로 구성한 ablation. teacher의 기여를 분리한다.

### 2-2. 결정적 증거 — margin의 span 분해 (heldout 906쌍, 길이정규화, `remeasure_b0_margin.py`)

| 개선(vs FAA, 토큰당) | 전체 | reasoning span | task_belief span | **action span** |
|---|---|---|---|---|
| B0 | +0.287 | +0.336 | **+0.802** | **+0.014** |
| A1 | +0.130 | +0.129 | +0.543 | **+0.023** |

- B0가 A1보다 margin을 2.2배 올렸지만, 그 우위는 **전부 belief/reasoning span**(teacher 문체)에 있고
  **action span은 +0.014로 A1(+0.023)보다 오히려 낮다.**
- 해석: **full-trace projection이 추가로 가르친 것은 "인과적으로 무력한 belief 텍스트 모방"뿐.**
  생성 acc(B0≈A1)·③(0.006, 개선 없음)과 완전히 정합한다.
- 참고: pref_acc≈0 — 절대 선호는 여전히 chosen<rejected (teacher trace를 "덜 비선호"하게 됐을 뿐).

### 2-3. 구조적 원인 (코드 위치)

`src/ego/step2_vlm_alignment/b0/teacher.py` — `project_full_trace()`:

```
현행 정보 흐름:
  입력:  과거 이미지·히스토리 + 미래 궤적 + [exact GT action]   ← 정답을 통째로 봄
  출력:  reasoning/belief 생성 → 코드가 action 을 GT 로 덮어씀
```

정답을 아는 작성자는 결론에서 역산한 글을 쓴다. 이런 chosen으로 DPO를 하면 모델이 배우는 것은
"belief로부터 action을 도출하는 법"이 아니라 "**정답 옆에 자연스럽게 어울리는 belief 문체**"다.
2-2의 span 분해가 이 예측을 정확히 확인한다. **규모를 키워도 ③는 0 근처에 머물 것** — 이것이
풀 스케일 전 리팩터가 필수인 이유다.

---

## 3. 구현 목표

**hindsight(미래를 아는 것)는 유지하되, 정답의 해상도를 낮춰 사후 합리화를 차단한다.**

```
리팩터 정보 흐름 (3단계):

  1단계  미래 궤적 → "상위 목표/latent task state"만 추출
          예: "샐러드 준비" (exact 'wash lettuce'는 포함 금지)

  2단계  teacher가 [과거 이미지·히스토리 + 후보 5 + 상위 목표]만 보고
          reasoning/belief 작성                       ← GT action 은닉

  3단계  verifier: "이 belief에서 출발하면 GT action이 자연스럽게 도출되는가?"
          PASS → GT action append, chosen 채택
          FAIL → 드랍 (audit 보존 + 탈락률 로깅)
```

**왜 인과가 생기나**: chosen이 되는 기준이 "정답을 아니까 항상 맞아 보임"에서
"**belief가 실제로 GT를 예측했을 때만**"으로 바뀐다. DPO가 선호하게 되는 belief는 정의상
행동 예측력이 있는 belief이고, 그런 belief를 쓰는 모델은 belief가 바뀌면 action도 바뀐다 — ③가 측정하는 그것.

**방법론 정합**: "hindsight-to-past projection"이라는 B0 정체성은 유지된다. 미래에서 가져오는 정보가
"정답 레이블"→"상위 목표"로 약해질 뿐이며, 오히려 belief 정의(여러 액션에 걸친 전역 목표)에 더 충실해진다.

---

## 4. 모델 구성

| 단계 | 현행 | 제안 |
|---|---|---|
| teacher (생성) | frozen Qwen3-VL-8B base (`build_teacher`, 로컬) | **동일 유지** (1·2단계 모두) |
| equivalence judge (belief SAME/DIFF) | 같은 teacher가 겸임 (`teacher.equivalence`) | 유지 가능 |
| **verifier (신규)** | 없음 | **1안: gemini-2.5-pro** (LETSUR 게이트웨이, ④ judge로 이미 연동) — 생성자와 독립이라 자기동의 편향 회피. **2안(보수적/전량 로컬)**: teacher가 후보 5 중 직접 선택 → GT 일치 시만 PASS (판정이 문자열 비교로 환원, 단 통과율 = teacher 자체 acc가 상한) |

권장: **1차는 1안**, 탈락 사례 ~30건 수동 audit 후 품질 문제 있거나 API 의존 부담이면 2안 전환.
참고: 현행 equivalence judge의 `uncertain=0` 이력은 자기동의 편향 가능성의 정황 — verifier만큼은 독립 모델 권장.

---

## 5. 구현 계획 (파일·단계)

| # | 작업 | 파일 | 예상 |
|---|---|---|---|
| 1 | 상위 목표 추출 프롬프트 + `extract_goal()` (미래 액션 시퀀스 → 1문장 목표; **exact next action 누출 금지 검사** 포함) | `b0/teacher.py` | 1h |
| 2 | `project_full_trace()` 2단계화: GT를 프롬프트에서 제거, goal 주입 | `b0/teacher.py` | 1h |
| 3 | `verify_belief_action()` (1안 gemini / 2안 로컬 선택 플래그) + PASS/FAIL·사유 기록 | `b0/teacher.py` 또는 신규 `b0/verifier.py` | 1–1.5h |
| 4 | 빌드 파이프라인 통합: 탈락률 통계(`num_verifier_dropped`), audit manifest에 goal·verdict 보존 | `b0/build_dpo_dataset.py` | 0.5h |
| 5 | smoke: 2샘플 전 경로 (goal 추출→belief 생성→verify→pair) — **goal에 GT 누출 없는지 assert** | 체인 | 0.5h |
| 6 | MVP 재빌드 — **기존 FAA 롤아웃(rejected) 전량 재사용**, teacher 산출만 재생성 (2-way 샤딩) | 체인 | ~3h GPU |
| 7 | DPO 재학습 (동일 하이퍼파라미터, max_length 4096) | 체인 | 0.5h |
| 8 | 평가: 생성 acc + **③ belief-swap** + span-margin 재측정(`remeasure_b0_margin.py` 재사용) | 체인 | ~1h |

**총 예상: 구현 4–5h + GPU 5h ≈ 캘린더 1일 이내** (기존 체인 인프라·멱등 마커 재사용 시).

---

## 6. 성공 기준 (사전 등록)

| 지표 | 현행 B0 | 목표 |
|---|---|---|
| **③ 인과 민감도** (주 지표) | 0.006 | **> 0.03** (control floor 0.004~0.006의 5배 이상; base 0.016도 상회) |
| action-span margin 개선 | +0.014 | belief-span 편중 해소 — action span이 A1(+0.023) 이상 |
| 생성 acc | 0.248 | ≥ 0.248 유지 (인과를 얻으려고 acc를 희생하지 않음) |
| verifier 탈락률 | — | 로깅 필수. >50%면 상위 목표 해상도 조절(아래 리스크) 후 재시도 |

코드리뷰 B0-8 연계: 위 평가를 **`B0_VALIDATED` 게이트**로 체인에 넣어, margin·acc만으로 "완료" 판정되지 않게 할 것.

---

## 7. 리스크와 손잡이

- **verifier 탈락률 과다** → chosen 부족: 상위 목표의 구체성이 손잡이다(너무 추상적이면 belief가 GT에 못 닿고, 너무 구체적이면 GT 누출로 회귀). "동사 없는 명사구 목표"부터 시작해 단계적으로 조절 권장.
- **1단계의 GT 누출**: goal 문자열에 GT verb/noun이 그대로 들어가면 리팩터가 무효. smoke assert(#5) + audit manifest에 goal 저장으로 상시 검사.
- **verifier 편향**: 1안(gemini)이라도 PASS 사례만 남으므로, 탈락/통과 각 ~30건 수동 audit 1회 필수.
- **비용**: teacher 호출 1→2회 + verifier 1회 → 빌드 ~2.5–3배 (MVP 기준 ~3h). FAA 롤아웃 재사용으로 상쇄.

---

## 8. 하지 않기로 한 것 (scope 명시)

- **B0-2/3 (pair 불균형·near-dup)**: 유효한 지적이나 본 리팩터와 변수 분리 — 리팩터 효과 측정 후 별도 반영.
- **F0 측 P1-6 (action-span credit / warm-start)**: F0 트랙에서 별도 진행 중. B0 리팩터와 목적(인과)은 같으나 레버가 다름.
- **풀 스케일 B0**: 본 리팩터의 ③ 성공 기준 통과 전 실행 금지 권고.

---

*근거 파일: `b0/teacher.py:163` `project_full_trace`, `b0/build_dpo_dataset.py:14`,
`runs/f0_battery/remeasure_b0.json`·`remeasure_a1.json`(span 분해 원자료),
`swap_b0_1f.json`(③), `b0_gen_1f.json`·`abl_actpatch_gen_1f.json`(생성 acc),
코드리뷰 `EGO_step2_code_handoff_2026-07-19.md` §B0-1/B0-8.*
