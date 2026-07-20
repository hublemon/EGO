# extro(F0) 학습 결과 종합 · intro(B0) 사전검증과 개선 방향

- 작성 2026-07-20 (KST) · 대상: 방법론/코드 리뷰어
- 인터랙티브 리포트(그래프·실제 추론 로그 포함): https://claude.ai/code/artifact/ce06e35a-2cbd-4cba-b5bf-8d3bf84f90a7
- 코드 기준 커밋 `61ae561` (+ 미커밋: `build_pairs_contrastive.py`, `retro_p12_chain.sh`)

---

## 0. 한 문단 요약

보상 설계를 네 번 바꾸는 동안 heldout 정확도는 base 0.242에서 움직이지 않았다. 원인은 보상이 아니라
**그룹-상대 advantage**였다 — 롤아웃끼리 같으면 advantage가 0이 되고 그 스텝은 학습이 사라진다.
기준선을 그룹 평균에서 **보상 EMA**로 바꾸자 같은 모델·같은 데이터에서 정확도가 **0.280**, G2 구간이
**0.382**까지 올랐다(GT 라벨 없이). intro(B0)는 사전검증 R1이 사전 등록 기준 4개 중 1개만 통과했고,
원인은 DPO 쌍이 **문체 차이**를 가르치고 있었다는 것이다(belief span margin +0.802 vs action +0.014).
쌍을 최소대조(P1+P2)로 재설계해 현재 무인 실행 중이다.

---

## 1. 평가 눈금 — 숫자를 읽는 기준

| 기준자 | 값 | 의미 |
|---|---|---|
| 우연 | 0.200 | 후보 5개 중 무작위 |
| WM top-1 복사 | 0.374 | world model의 1순위를 그대로 따라 하기 |
| 오라클 상한 | 0.620 | GT가 top-5 안에 있는 비율 |
| G2 구간 | n≈123, 우연 0.2 | GT는 top-5에 있으나 **WM top-1은 틀린** 부분집합 |
| ③ 인과민감도 | swap−control | belief를 바꿔치기했을 때 action이 바뀐 비율의 순증 |

**G2가 오르는지**가 "WM을 베끼는 것"과 "WM을 이기는 것"을 가르는 유일한 지표다.
전체 평가는 heldout 500샘플·1프레임·단일 시드.

---

## 2. extro(F0) — 무엇을 했고 무엇을 알아냈나

### 2.1 Phase 1/2b: 보상을 바꿔도 움직이지 않았다

| 정책 | 보상 · 기준선 | acc | G2 | cond.acc | wm_follow | ③ |
|---|---|---|---|---|---|---|
| base | — | 0.242 | 0.309 | — | 0.328 | 0.016 |
| F0-W | WM likelihood · 그룹 adv | 0.240 | 0.333 | 0.387 | 0.344 | 0.008 |
| F0-G | GT binary · 그룹 adv | 0.244 | 0.301 | 0.394 | 0.334 | — |
| F0-WA | WM · action-span credit | 0.244 | 0.325 | — | 0.346 | 0.010 |

네 arm 모두 base ±0.002. 보상 정의(밀도·형태·credit 위치)를 모두 바꿨는데 결과가 같다는 것은
**보상이 병목이 아니다**라는 뜻이다.

코드: `src/ego/step2_vlm_alignment/train_grpo_action.py` (reward mode 테이블, `JOINT_REWARD_MODES`),
`scripts/step2/pro_clean_chain.sh`, `pro_span_chain.sh`.

### 2.2 진단: 그룹-상대 advantage가 gradient를 지운다

GRPO의 advantage는 `r − 그룹 평균`이다. 한 프롬프트의 8개 롤아웃이 같은 보상을 받으면 advantage가 전부 0.
학습 로그의 `frac_reward_zero_std`가 이를 직접 보여준다.

| run | zero-std 스텝 비율(평균) | 후반 50스텝 |
|---|---|---|
| F0-W (WM, 그룹 adv) | 0.368 | 0.330 |
| F0-G (GT, 그룹 adv) | **0.528** | **0.560** |
| F0-WA (span credit) | 0.345 | 0.355 |

즉 GT 보상 run은 **스텝의 절반 이상이 무학습**이었다.

**결정적 관측.** reasoning을 빼고 action만 생성시키자 zero-std가 **1.00**이 되었다.
온도 1.0과 1.3 양쪽에서 8개 롤아웃이 글자까지 동일한 action을 냈다(스모크 2회 모두 grad 전부 0).
→ GRPO가 얻던 탐색 다양성은 행동 선택이 아니라 **reasoning 텍스트의 샘플링 노이즈**에서 왔다.
정작 배워야 할 자리에는 탐색이 없었다.

기록: `runs/f0_battery/f0smoke_ga.log`, `F0_GA_RESULTS.md`.

### 2.3 기각한 대안 — 후보 확률 직접 계산(exact-CE)

후보 5개 문자열을 teacher-forcing으로 채점하면 분산은 확보되지만,
(a) 실제 배치는 **생성**으로 행동을 고르므로 train/test가 어긋나고,
(b) VLM에게 확률을 언어로 계산시키는 구조적 모순이 남는다.
설계 단계에서 기각했고 작업물은 `scripts/step2/pro_gx_train.py` + `runs/f0_battery/GX_ABORT_NOTE.txt`에 기록만 남겼다.

### 2.4 처방: EMA baseline REINFORCE

`advantage = r − EMA(r)` (momentum 0.99), 프롬프트당 롤아웃 1개,
`loss = −adv · mean_logp(completion)`. 그룹이 사라지므로 정책이 결정적이어도 gradient가 산다.

코드: **`scripts/step2/pro_gr_train.py`** — 새 트레이너 전체. 주요 지점:
- `--reward {gt,wm}` / `--full_trace` / `--ema` 인자
- wm 모드는 GT 필터를 아예 타지 않음(**GT-free**), 보상은 후보 5개에 대해 재정규화된 likelihood
- Qwen3-VL M-RoPE 대응: 수동 forward에서 `mm_token_type_ids`를 completion 길이만큼 0으로 이어붙임
  (이 처리가 없으면 `ValueError: mm_token_type_ids is missing`)

체인: `scripts/step2/pro_ga_chain.sh`(진단) → `pro_we_chain.sh`(GT+EMA, 게이트 통과 시 자동 실행) →
`pro_wema_chain.sh`(WM+EMA, GT-free).

### 2.5 진단 실험 결과 — 병목 위치 확인

action-only 축소 문제에서 GT+EMA를 돌려 기준선 교체 효과만 분리했다.

| | acc | cond.acc | wm_follow |
|---|---|---|---|
| base (action-only 프롬프트) | 0.270 | 0.432 | 0.374 |
| F0-G (full-trace, 그룹 adv) | 0.244 | 0.394 | 0.334 |
| **F0-GR (action-only, GT+EMA)** | **0.338** | **0.542** | 0.440 |

사전 등록 게이트 +0.02 대비 **+0.094**. 단 이 수치는 **프롬프트가 다른 더 쉬운 평가 체제**에서 나온 것이다
(같은 조건의 base가 0.242가 아니라 0.270). 본 정책들과 같은 표에 놓고 순위를 매기면 안 된다 —
이것은 성능 주장이 아니라 **병목 위치의 진단**이다.

### 2.6 최종 비교 — GT 보상 vs WM 보상 (둘 다 EMA)

| 정책 | acc | G2 | cond.acc | wm_follow | ③ |
|---|---|---|---|---|---|
| base | 0.242 | 0.309 | — | 0.328 | 0.016 |
| F0-WE (GT binary + EMA) | 0.258 | 0.309 | 0.413 | 0.362 | 0.018 |
| **F0-W-EMA (WM likelihood + EMA, GT-free)** | **0.280** | **0.382** | **0.452** | 0.350 | 0.008 |

**왜 GT-free 쪽이 이겼나 — 학습 곡선이 설명한다.**

| run | reward_ma 시작→끝 | |adv| |
|---|---|---|
| F0-GR (GT) | 0.390 → 0.485 (7,000샘플) | 0.47–0.50 |
| F0-WE (GT) | 0.360 → 0.455 (5,000샘플) | 0.46–0.50 |
| F0-W-EMA (WM) | 0.285 → 0.313 (5,000샘플) | 0.19–0.22 |

- GT 이진 보상은 |adv|가 0.45–0.50에서 고정된다. 틀린 순간 **trace 전체가 강하게 눌리고**,
  후보 간 순위 정보는 버려진다. 학습 보상은 잘 오르지만(0.36→0.455) heldout은 덜 오른다
  → 상승분 상당수가 **학습 필터(GT∈top5) 안쪽에 대한 적합**이다.
- WM likelihood는 후보에 대해 재정규화된 연속값이라 |adv| ≈ 0.20으로 부드럽고 순위 정보가 보존된다.
  학습 보상은 거의 평평한데 heldout G2가 0.309 → **0.382**로 올랐다.

**G2 상승이 핵심이다.** 사전에 나는 "WM 보상 → top-1 복사 붕괴 → G2→0"을 예측했으나 틀렸다.
실제로는 wm_follow가 0.350에 머물고 G2가 올랐다. 이유: WM likelihood는 후보 5개에 대해
**재정규화**되므로, WM 분포가 평평한 곳(=바로 G2 구간)에서는 복사 압력 자체가 약해진다.
복사 압력이 정확히 필요 없는 곳에서 스스로 감쇠하는 구조다.

### 2.7 판정단(gemini-2.5-pro) — 그리고 이 프로젝트에서 가장 중요한 한 줄

4개 정책의 **동일한 heldout 40샘플**, 7항목 × 2점 = 14점.

| 항목 | base | F0-W | F0-WE | F0-W-EMA |
|---|---|---|---|---|
| 행동 이력 활용 | 2.00 | 1.95 | 1.97 | 1.97 |
| 후보 검토 | 0.56 | 0.78 | 0.63 | **0.78** |
| 시각 근거 | 1.69 | 1.76 | 1.63 | **1.81** |
| 결론의 논리성 | 1.95 | 2.00 | 1.90 | 1.92 |
| 환각 없음 | 1.92 | 1.68 | 1.76 | **1.95** |
| belief 전역성 | 2.00 | 2.00 | 2.00 | 2.00 |
| belief–action 연결 | 1.97 | 1.95 | 1.90 | **2.00** |
| **합계** | 12.10 | 12.11 | 11.79 | **12.43** |

> **F0-W-EMA의 belief_action_link는 2.00/2.00 만점인데, 같은 정책의 ③ 인과민감도는 0.008이다.**
> belief를 통째로 다른 것으로 바꿔치기해도 action은 사실상 그대로다.
> **LLM 판정단은 인과의 부재를 탐지하지 못한다.** 개입(intervention) 테스트가 대체 불가능한 이유이며,
> B0의 목표 지표를 판정단 점수가 아니라 ③으로 잡은 근거다.

퇴화 점검(500 completion): 앞 12단어 고유 개수 base 467 / F0-WE 462 / F0-W-EMA 462, 평균 길이
103.0 / 99.2 / 101.9 단어. 문장 외우기 붕괴는 없다.

코드: `src/ego/step2_vlm_alignment/judge_reasoning.py`,
어댑터 `scratchpad/judge_adapter.py`(heldout을 frame_id로 조인해 정책 간 동일 샘플셋 보장).

### 2.8 실제 추론 로그에서 관찰되는 차이

G2 구간에서 base는 틀리고 W-EMA는 맞힌 샘플들에서 반복되는 패턴은 하나다 —
**W-EMA는 후보 목록을 명시적으로 검토한 뒤 고른다.** 판정단의 후보 검토 점수(0.56→0.78)와 일치한다.

```
P01_15_189   GT: put container
base   → put plate      "…the most logical next step is to put the item I'm holding into the dish rack."
W-EMA  → put container  "…Among the candidates, 'put plate' or 'put bowl' are plausible next steps,
                          but 'put container' matches the recent pattern of putting down washed items."

P01_14_283   GT: eat pasta
base   → put spatula    "…the logical next step is to put the spatula down to free my hand."
W-EMA  → eat pasta      "…Among the candidates, 'eat pasta' is the most direct continuation…
                          The other options like putting down the spatula or plate are secondary."
```

원문: `runs/f0_battery/{base_1f_strict,f0wema_final}.records.jsonl`.

### 2.9 F0를 확정하기 전에 반드시 해야 할 3가지 (미해결)

1. **테스트셋 선택 편향** — 6개 arm을 같은 heldout에서 비교하고 그 heldout에서 승자를 보고했다.
   승자 재측정용 별도 split이 필요하다.
2. **단일 시드** — n=500에서 Δacc 0.038의 95% CI는 대략 ±0.037. 시드 2회 반복 없이는 순위를 단정할 수 없다.
3. **복사 붕괴 미반증** — 5,000샘플 시점에도 reward_ma가 아직 상승 중이다. 더 오래 돌렸을 때
   wm_follow가 0.374로 수렴하며 G2가 무너지는지 확인해야 한다.

2 GPU 기준 12–15시간. 현재 B0-P12가 GPU를 점유 중이라 대기 상태이며, **사용자 승인 대기**다.

---

## 3. intro(B0) — 사전검증 결과와 재설계

### 3.1 B0가 하는 일

F0로 얻은 FAA 어댑터를 얼린 뒤, offline teacher가 만든 full-trace를 **DPO 선호 증류**로 얹는다.
목표는 정확도가 아니라 **belief → action 인과**(③)를 만드는 것.

### 3.2 B0-R1(GT-hidden gated teacher) 결과 — 4개 중 1개 통과

| 지표 | MVP | 사전 등록 기준 | R1 | 판정 |
|---|---|---|---|---|
| ③ 인과민감도 | 0.006 | > 0.03 | 0.008 | ✗ |
| action-span margin | +0.014 | ≥ +0.023 | +0.007 | ✗ |
| 생성 정확도 | 0.248 | ≥ 0.248 | 0.238 | ✗ |
| G2/G1 retention | — | ≥ 0.5 | 0.702 | ✓ |

게이트 자체는 정상 동작했다: **통과 725 / 탈락 643**, goal 누출 드랍 2.
teacher가 GT를 못 맞히면 **action을 사후 수정하지 않고 그 샘플을 버린다**(drop-not-patch) —
49–53%의 탈락률이 그 정책이 실제로 작동함을 증명한다.

코드: `src/ego/step2_vlm_alignment/retro/teacher.py`의 B0-R1 섹션
(`goal_prompt`, `gated_trace_prompt`, `goal_leaks()` — 어형 변화까지 검사, `GatedTeacherMixin.generate_gated_trace()`),
오케스트레이션 `b0/build_dpo_dataset_r1.py`, 체인 `scripts/step2/retro_r1_chain.sh`.
전체 스케일 B0는 게이트 미통과로 **자동 차단**되어 약 20 GPU시간을 아꼈다
(`runs/f0_battery/B0_FULL_SKIPPED`, `NEEDS_DECISION_B0_FULL`).

### 3.3 실패 원인 — 쌍이 가르치려는 것 말고 다른 걸 가르쳤다

DPO는 chosen/rejected를 가르는 **가장 쉬운 차이**를 배운다.
MVP·R1의 쌍은 chosen = teacher(frozen base VLM), rejected = FAA — 즉 **모델 정체성이 다르다**.
가장 쉬운 차이는 문체였고, span 분해가 이를 그대로 보여준다:

```
belief 구간 margin  +0.802
action 구간 margin  +0.014     ← 가르치려던 곳
```

모델은 "teacher처럼 쓰기"를 배웠고 action 선택은 거의 건드려지지 않았다.

### 3.4 재설계 — 최소대조 쌍 (P1 + P2)

원칙: **쌍은 가르치려는 것만 달라야 한다.** teacher·DPO 파이프라인은 유지하고 쌍의 출처만 바꾼다.

| | 구성 | 상쇄되는 것 |
|---|---|---|
| **P1 자기대조** | 같은 FAA의 맞춘 trace ≻ 틀린 trace, 양쪽 다 **원문 verbatim** | 문체(같은 모델·같은 분포) |
| **P2 최소대조** | reasoning·belief 동일, **action만 상이** | 그 외 전부 — gradient가 action 토큰에만 걸림 |
| **S4 teacher** | FAA가 8번 다 틀린 구간에만 gated teacher 투입 | 희소 구간 보충 |
| **P3 (대기)** | belief 반사실 쌍 — ③을 직접 겨냥 | P1+P2 통과 후 착수 |

**착수 전에 검증한 지름길 차단 2가지** (구현 핵심):
- **포맷 지름길** — P2에서 한쪽만 재조립하면 모델이 action이 아니라 *포맷 차이*를 학습한다.
  그래서 chosen·rejected를 **둘 다** `build_full_trace`로 정규 직렬화했고,
  스모크에서 **97쌍 전부 action 앞부분이 글자 단위로 완전 일치(0/97 불일치)** 함을 확인했다.
- **후보-밖 지름길** — rejected의 오답 action은 반드시 **그 프롬프트의 후보 5개 중**에서 뽑는다.

코드: **`src/ego/step2_vlm_alignment/retro/build_pairs_contrastive.py`** (미커밋).
프롬프트당 상한 `max_p1=4` / `max_p2=2`, G1/G2 태깅, 전부-오답 프롬프트는 `--out_hard`로 분리,
emit 전 `check_prompt_leakage`.

스모크(기존 4롤아웃 데이터, 200샘플): mixed 57 / all_correct 61 / all_wrong 82 →
**286쌍(P1 189 / P2 97), 누설 드랍 0.** 본 실행은 롤아웃을 8개로 늘려 mixed 비율을 키운다.

### 3.5 현재 상태 — 무인 실행 중

`scripts/step2/retro_p12_chain.sh` (미커밋), 2026-07-20 **10:59 KST** 착수.

```
S1 추가 롤아웃 4×1,500 프롬프트 (GPU 2장)   ~4h   ← 진행 중
S2 기존 4 + 신규 4 = 8 롤아웃 병합
S3 P1+P2 쌍 생성                            수초
S4 gated teacher 보충 (all-wrong 구간만)     ~2.5h
S5 통합 DPO — 쌍 2,000개 미만이면 중단,
   max_length 4096, weight-diff 무학습 가드   ~2h
S6 평가: acc / action-span margin / ③(참고)  ~1h
S7 자동 판정 → B0_P12_PASSED
```

완료 예상 **20:00 KST 전후**. 마커 `B0_P12_DONE` / `B0_P12_FAILED` 워처 설정 완료.

**사전 등록 성공 기준(이 단계): `acc ≥ 0.26` AND `action-span margin ≥ +0.023`.**
인과(③)는 이 단계 기준이 아니라 P3의 몫이며 참고용으로만 측정한다.
span margin은 비교 가능성을 위해 R1과 **동일한 heldout 쌍**(`b0_r1_dpo_heldout.jsonl`)에서 잰다.

---

## 4. 결론

1. **병목은 보상 설계가 아니라 credit 경로였다.** 그룹-상대 advantage는 정책이 결정적일수록 무력해지고
   action-only에서는 완전히 0이 된다. 기준선을 EMA로 바꾸는 것만으로 같은 조건에서 정확도가 움직였다.
2. **보상 정의가 도달점을 정한다.** GT 이진 보상은 학습 보상은 잘 올리지만 heldout은 덜 오른다(0.258).
   WM likelihood는 연속값이라 스텝이 부드럽고 WM이 틀린 구간(G2)에서 더 나은 선택을 배운다(0.280 / G2 0.382).
   그리고 GT 라벨이 필요 없다.
3. **reasoning 품질과 인과는 다른 축이다.** 판정단 만점(2.00)과 ③ 0.008이 공존한다.
   B0의 성패는 판정단이 아니라 개입 테스트로만 판정한다.
4. **F0는 아직 확정이 아니다.** §2.9의 세 가지 확인(별도 split · 시드 반복 · 장기학습 붕괴 검사)이 남아 있다.

---

## 5. 방법론 → 코드 대응표

| 방법론 | 코드 |
|---|---|
| GRPO 3태그 학습 · 보상 모드 | `src/ego/step2_vlm_alignment/train_grpo_action.py` |
| action-only 진단 모드 | 같은 파일의 `ACTION_ONLY` / `JOINT_SYSTEM_PROMPT_ACTION_ONLY`, `scripts/step2/eval_battery.py --action_only` |
| EMA baseline REINFORCE (F0-GR/WE/W-EMA) | `scripts/step2/pro_gr_train.py` |
| 기각된 exact-CE | `scripts/step2/pro_gx_train.py` + `runs/f0_battery/GX_ABORT_NOTE.txt` |
| 무인 조건부 체인 | `scripts/step2/pro_ga_chain.sh`, `pro_we_chain.sh`, `pro_wema_chain.sh`, `retro_full_chain.sh` |
| 평가·G2·wm_follow·개입(swap) | `scripts/step2/eval_battery.py` |
| 판정단 | `src/ego/step2_vlm_alignment/judge_reasoning.py` |
| B0 gated teacher (GT 은닉) | `src/ego/step2_vlm_alignment/retro/teacher.py`, `b0/build_dpo_dataset_r1.py`, `scripts/step2/retro_r1_chain.sh` |
| B0 최소대조 쌍 재설계 | `b0/build_pairs_contrastive.py`, `scripts/step2/retro_p12_chain.sh` *(미커밋)* |

### 남아 있는 코드 리뷰 지적사항 (수용, 미구현)

- DPOConfig 인자 fail-fast 화이트리스트
- `"five"` 하드코딩 6곳 (후보 개수 상수화)
- 후보 접근자 통일 (`topk_actions` / `topk_actions_with_score`) — 현 train 4,998행은 두 필드가 동일함을 확인
- DDP 그룹 통계 경합, span mask 견고성, provenance 해시
