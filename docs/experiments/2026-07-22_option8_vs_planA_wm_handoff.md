# 옵션 8 vs Plan A — V-JEPA2(WM) 담당자용 기대 효과·근거 정리 Handoff

> 작성일: 2026-07-22 KST
> 수신: Step 1 (V-JEPA2 probe) 학습 담당자 — 현재 `z1_start_m1_lobs{8,16}_vna` (Plan A Step-1) 진행 중
> 발신 근거 실험: `EGO/runs/next_ce/` (옵션 8 Step-2 하이브리드, 2026-07-22 무인 완주) ·
> 핸드오프 `EGO_jihun/docs/experiments/2026-07-22_next_ce_hybrid_option8_step2_handoff.md`
> 선행 문서: `2026-07-22_vjepa2-time-contract-audit-handoff.md` · `2026-07-22_step1-step2-temporal-semantics-risk-report.md`

---

## 0. 세 줄 요약

1. **Plan A와 옵션 8은 배타가 아니다.** 지금 돌리는 Plan A 런은 그대로 두고, **옵션 8 probe(라벨만 j+1로 교체)를 병렬로 추가**하는 것을 제안한다 — 기존 end−1s feature 캐시를 그대로 쓰므로 probe 학습 비용만 든다.
2. 오늘 Step-2 실측으로 **시스템 전체의 병목이 probe coverage임이 확정**됐다 (VLM 선택 학습 폭은 GT 계약과 무관하게 +16.4~16.6pp로 동일 — 차이는 전부 coverage). **당신이 coverage를 1pp 올리면 시스템 acc가 거의 그대로 따라 오른다.**
3. 옵션 8의 기대 효과: **더 싼 비용으로, 더 풍부한 관찰(진행 중 action 전체)에서, leakage 0인 진짜 next-action coverage**를 확보. 아래 근거와 함께 판정 기준을 미리 걸어두었다.

---

## 1. 두 계약의 정의 (모두 진짜 anticipation — target 노출 0)

```text
Plan A (strict start−1s) — 현재 진행 중인 런:
  … a_{j−1} ──┤gap├── [a_j ██████████]
        관찰창 ──────┤
                     t = start_j − 1s        GT = a_j (1초 뒤 시작, 고정 지평)

옵션 8 (end−1s, GT=j+1) — 제안:
  … [a_j ██████████████]──┤gap├── [a_{j+1} ████]
        관찰창 ─────────┤
                        t = end_j − 1s       GT = a_{j+1} (~1s+gap 뒤 시작, 가변 지평)
```

| | Plan A | 옵션 8 |
|---|---|---|
| anchor | start_j − 1s | end_j − 1s (기존 end_m1 인덱스와 동일) |
| GT | a_j | a_{j+1} (`start_frame > anchor`인 첫 annotation) |
| 관찰창에 담기는 것 | 이전 action 끝 + gap | **진행 중 a_j 전체** |
| target 노출 | 0% | 0% (다음 annotation이 겹쳐 시작하는 코너케이스만 정의로 방어) |
| feature | **재추출 필요** (현재 캐시 빌드 중) | **기존 end_m1 캐시 재사용** — probe만 재학습 |
| 프로토콜 정합 | EK100 표준 anticipation (τ_a=1s) | Ego4D LTA류 boundary-anchored |

---

## 2. 옵션 8에서 기대할 수 있는 효과와 근거

### E1. 시스템 acc 직결 — coverage가 유일한 병목임이 실측됨 ★

오늘 Step-2 하이브리드 실험(후보는 recognition probe 그대로, GT만 next)에서:

| 과제 | VLM 조건부 정확도 (base→학습 후) | Δ | coverage@5 | 시스템 acc |
|---|---|---:|---:|---:|
| recognition (기존) | 0.451 → 0.617 | +16.6pp | 0.626 | 0.386 |
| next (오늘) | 0.351 → 0.516 | **+16.4pp** | **0.253** | 0.130 |

- **VLM의 후보 내 판별 학습은 GT 계약과 무관하게 같은 폭** → 시스템 acc ≈ coverage × 조건부(≈0.52).
- 즉 probe가 cov_next@5를 0.25 → 0.40으로 올리면 시스템 acc는 0.13 → **~0.21**로 거의 선형 상승.
- **coverage는 Step-2 학습으로는 못 올린다** (oracle-subset 구조상 불가능) — 오직 probe 재학습만이 올릴 수 있다.

### E2. 무학습 하한이 이미 0.2525 — 라벨을 겨냥하면 오를 여지가 크다

- 오늘 측정된 cov_next@5 = 0.2525 (EK100 heldout n=1,398)는 **next를 전혀 겨냥하지 않은 recognition probe의 부산물** 수치다. 현재 action을 맞히도록 학습된 분포의 top-5에 "우연히" 다음 action이 든 비율.
- 옵션 8 probe는 같은 입력(같은 feature)에서 **명시적으로 j+1을 라벨로** 학습한다 — 0.2525는 사실상 옵션 8의 하한 추정치.
- 참조점: V-JEPA2 공식 EK100 anticipation(경계 1s 전)이 R@5 39.7% — 경계 근처 1s 전 예측이 도달 가능한 수준의 존재 증명.

### E3. 관찰 정보의 질 — 특히 GoalStep에서 유리할 구조적 이유

- 옵션 8의 관찰창에는 **진행 중 action 전체**가 들어간다 (완결된 절차 단계 하나). Plan A의 관찰창은 이전 action의 끝부분 + gap.
- GoalStep은 action 중앙값 12.6s의 절차적 태스크라 "지금 단계를 다 본 뒤 다음 단계"를 예측하는 구조가 자연스럽다.
- 실측 방증: GoalStep에서 end−1s 계열(z1_end_m1_lobs8_vna, 47.44%)과 start−1s 계열(b3_d4, 24.54%)의 격차 중 상당 부분이 관찰 정보 차이다 — 물론 이 격차의 대부분은 recognition leakage였지만, **"현재 action을 본다"는 관찰 이점 자체는 GT를 j+1로 바꿔도 유지**되고 leakage만 제거된다.

### E4. 비용 — probe 학습만, feature 재추출 0

- GoalStep: `index_end_m1_lobs8` + `goalstep_feature_cache_end_m1_*`를 그대로 쓰고 **라벨 컬럼만 j+1로 remap한 인덱스**를 하나 더 만들면 된다. 재추출 대비 수십 분 vs 수 시간.
- EK100: 동일 원리 (anticipation head 라벨을 다음 annotation으로).
- Plan A는 feature 재추출이 필수라 지금 캐시 빌드가 도는 중 — **그 사이 옵션 8 probe를 먼저 완주할 수 있다.**

### E5. 논문 방어 — "NEXT" 프롬프트·leakage 문제가 동시에 해소

- GT가 결정 시점에 시작 전이므로 `assert max(observed_time) < target_start`가 성립 — 리스크 보고서 P0 조치와 정합.
- Step-2 프롬프트의 "NEXT action" instruction이 참이 된다 (instruction–supervision mismatch 해소).
- 프레이밍: "recognition-grounded anticipation" — 진행 중 action을 인식해 다음 action을 제약. Plan A(strict)와 함께 실으면 decision-time 축의 두 점이 된다.

---

## 3. 유의할 리스크 (정직한 반론)

1. **상승 폭은 미지수.** E2는 하한 논리일 뿐, j+1 라벨 학습이 실제로 coverage를 얼마나 올릴지는 실측 전까지 모른다. EK100은 다음 action이 본질적으로 다봉(persistence 0.0715 — 반복이 거의 없음)이라 상한이 낮을 수 있다.
2. **가변 예측 지평.** 옵션 8의 지평은 1s + gap (EK100 gap 중앙값 1.49s 실측). 결과 보고 시 gap 분포 병기 필요. Plan A는 고정 1s로 더 깨끗한 실험 변수.
3. **Plan A가 표준 프로토콜 정합에서는 우위.** headline을 어느 쪽으로 할지는 coverage 실측 후 결정하면 된다 — 지금은 둘 다 확보가 정답.

---

## 4. 제안 실행 순서와 판정 기준 (사전 등록)

| 순서 | 작업 | 비용 |
|---|---|---|
| 1 | GoalStep: `index_end_m1_lobs8`의 라벨을 j+1로 remap한 `index_end_m1_next` 생성 (겹침 방어: `next = start > anchor`인 첫 annotation, 마지막 segment drop) | CPU 분 단위 |
| 2 | 기존 end_m1 feature 캐시로 probe 재학습 (config는 `z1_end_m1_lobs8_vna` 복제 + 인덱스 교체) | GPU ~기존 probe 1회분 |
| 3 | 평가: **cov_next@{5,10,15}** + Recall@5, canonical(start−1s = Plan A 완료분)과 동일 표로 비교 | 추론만 |
| 4 | EK100도 동일 적용 (Step-2 공급용) | 〃 |

**판정 기준 (실행 전 고정)**
- 옵션 8 probe의 cov_next@5가 **Plan A probe의 cov@5를 유의하게 상회**하면 → Step-2 후보 공급자로 옵션 8 채택, Plan A는 strict 프로토콜 비교점으로 보고
- 동등 이하이면 → Plan A 단독 headline, 옵션 8은 decision-time 곡선의 한 점으로만 수록
- 어느 쪽이든 **두 계약 모두 leakage 0**이므로 결과는 전부 논문에 사용 가능

---

## 5. 참조 산출물

| 위치 | 내용 |
|---|---|
| `EGO/runs/next_ce/data/phase0.json` | cov_next@5 0.2525 · L0_next 0.0665 · persistence 0.0715 · gap 1.49s (EK100) |
| `EGO/runs/next_ce/RESULTS.md` | Step-2 하이브리드 결과 (paired Δ=+0.0415, CI [+0.027, +0.056]) |
| `EGO_jihun/scripts/step2/build_next_gt.py` | next 라벨 remap 로직 선례 (`start_frame > anchor` 첫 row, drop 정책) — 인덱스 빌더에 이식 가능 |
| 실험 리포트 UI | https://claude.ai/code/artifact/0aaf43ab-4a9a-4d12-84e1-009a30d4b4d0 |
| 전략 문서 UI | https://claude.ai/code/artifact/bb9ff827-3ff1-4920-b104-e6b8d60b7bd1 |
