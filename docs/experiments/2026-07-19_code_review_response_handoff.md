# 코드리뷰 회신 핸드오프 — 검증·정정·재측정 결과

- **작성:** 2026-07-19 (야간 F0/B0 실행 직후)
- **대상:** `EGO_step2_code_handoff_2026-07-19.md`(코드리뷰) 작성자
- **목적:** 리뷰의 각 지적을 **실제 코드·실측으로 검증**한 결과와, 우리 쪽에서 새로 발견한 사항, 그리고
  다음 단계(GT-only 재편) 사양을 회신한다. 파일:라인·수치는 모두 재현 가능한 근거다.
- **동반 UI:** 같은 내용 트리아지 — 코드리뷰 트리아지 아티팩트(별도 전달).

---

## 0. 총평

리뷰는 근거가 탄탄합니다. **지적한 P0 상당수를 우리가 같은 밤에 독립적으로 실제로 겪었고**, 더 중요하게는
F0/B0의 연구적 발견(③ belief 인과 무력 · ④ judge 표면 만점)을 **구조적으로 설명하는 항목**(P1-1·P1-6·B0-1)이
정확했습니다. 다만 두 곳은 **정정**이 필요합니다(우리 쪽 이전 보고 포함). 분류:

| 상태 | 항목 |
|---|---|
| **코드로 확정 (리뷰 정확)** | P0-6, P0-7, P1-1, P1-6, B0-1, B0-4, B0-5 |
| **우리가 이미 수정/진단 (리뷰의 보강안이 더 견고)** | P0-2, P0-3, P0-4, P0-1 |
| **부분 충족 / 정합성 하드닝** | P0-5, B0-6, B0-7, B0-8, B0-2, B0-3 |
| **정정 필요** | (a) B0-5의 원인은 길이편향 아님 → span 집중 (재측정) · (b) "WM-only"는 오염된 명칭 |

---

## 1. 코드로 확정된 지적 (근거 첨부)

### P0-6 · evaluate.py가 존재하지 않는 모듈을 import — **확정**
- `evaluate.py:41-44`: `EGO_ROOT = ~/work/jihun/EGO` → `sys.path.insert` → `import train_qwen25vl_grpo_ek100`.
- `find . -name train_qwen25vl_grpo_ek100*` → **저장소에 없음.** clean clone은 ImportError, dev 머신은 stale 파일로 조용히 구버전 parser 평가. (경로도 우리 서버(`/mnt/nvme/...`)와 달라 이 머신에서도 깨짐.)
- 우리는 세션 내 **모든** 평가를 신규 `eval_battery.py`(패키지 import + 384)로 우회했음 — 그래서 **이번 세션 수치는 이 버그에 영향 없음.** 다만 이 우회 자체가 리뷰가 지적한 "평가 경로 분기"의 증상.
- 권고 수용: `from ego.step2_vlm_alignment import train_grpo_action as T` + `assert "src/ego/step2_vlm_alignment" in T.__file__`.

### P0-7 · train/eval generation budget 불일치 — **확정**
- 학습: `--max_completion_length 384` (`train_pro_final.sh:20`, `train_pro_final_v2.sh:44`).
- 평가: `evaluate.py` 기본 `max_new_tokens=256` (`:84`, `:211`), `eval_pro_final.sh`는 미덮어씀.
- **리포 자체 기록**(`docs/experiments/2026-07-17_f0_final.md:153`)에 `384 | Qwen3-VL은 verbose → 256에서 clipped_ratio 0.25` 명시 — 팀이 학습 시점에 이미 알았으나 eval 기본값엔 미전파(doc–code drift).
- 우리 `eval_battery.py`는 384 고정 + parse_rate≈0.99~1.0 (잘림 없음 실측 확인). 레거시 evaluate.py 경로로 낸 과거 수치만 재검증 필요(잘림은 acc를 낮추기만 하므로 그 값들은 하한).

### P1-1 · "WM-only 보상"은 WM-only가 아니다 — **확정 + 정량화 (우리 추가 기여)**
- `build_reward_funcs:1311` — `wm_likelihood` 모드 = `[format_reward_think, candidate_gate_reward_think, wm_likelihood_reward, think_convergence_reward]`.
- 학습 로그 `reward_log.jsonl`(250스텝) 항별 평균:

| 항 | 평균 | std(step) | 성격 |
|---|---|---|---|
| format | 0.198 | 0.003 | 유효군 내 ~상수 → 사실상 게이트 |
| candidate_gate | −0.005 | 0.015 | ~상수 게이트 |
| **wm_likelihood** | 0.315 | 0.099 | 가변 → advantage 구동 |
| **think_convergence** | 0.180 | 0.040 | 가변 → advantage 구동 |

- **정량 결론**: think_convergence가 전체 보상의 **26%**, 그리고 GRPO advantage를 실제로 구동하는 두 가변 항 중 **36%**(0.180/(0.315+0.180)).
- **문제의 본질**: `think_convergence_reward_joint:1116`는 reasoning의 정확성·시각근거를 검사하지 않고,
  "**마지막 언급 후보 == 최종 선택**(+0.15), 선택쌍 언급(+0.10), 미언급(−0.10)"만 본다.
  = **이미 고른 액션을 사후 정당화하는 추론을 직접 보상.** 이것이 ④ 표면 만점 ↔ ③ 인과 0 괴리의 학습신호 수준 원인.
- 권고 수용: format·gate를 additive reward가 아닌 **validity 게이트**로, think_convergence는 학습 reward에서 제거하고 평가 metric으로만. → "진짜 WM-only"를 새로 정의해야 skyline 비교가 성립.

### P1-6 · action 스칼라 보상이 trace 전체 토큰에 적용 — **구조상 확정**
- GRPO advantage는 completion 전 토큰(reasoning·belief·action)에 동일하게 실림. 보상은 final action만 평가.
- 따라서 우연히 맞은 action의 사후 rationale까지 강화 → belief는 인과가 아니라 **공기(共起)**로 학습.
  ③ 인과 역전(0.006~0.008)의 동역학적 설명. P1-1과 같은 결을 강화.

### B0-1 · teacher가 GT를 보고 reasoning을 씀 — **개념 확정**
- `b0/teacher.py:project_full_trace`가 GT action을 입력받아 reasoning/belief 생성 후 action을 GT로 덮음.
  chosen 자체가 "정답을 아는 상태의 post-hoc rationale". 규모를 키워도 ③가 0 근처에 머물 구조적 원인.
- 리뷰의 2단계(future→상위목표 추출 · GT 숨김 · verifier 검사 후 GT append)를 **풀 스케일 전 최우선**으로 수용.

---

## 2. 정정 항목 (우리 이전 보고 포함)

### B0-4 / B0-5 · margin 재측정 — 리뷰의 원인 진단을 정정, 그러나 결론은 강화
- 리뷰: `evaluate_b0._seq_logprob`가 sum log-prob → 짧은 chosen 유리(길이 편향). full-trace가 action 아닌 문체 학습 위험.
- 우리 재측정(`remeasure_retro_margin.py`, heldout **906쌍**, 길이정규화 + span별):

| | mean-token 개선 vs FAA | reasoning span | task_belief span | **action span** |
|---|---|---|---|---|
| B0 (teacher full-trace) | **+0.287** | +0.336 | +0.802 | **+0.014** |
| A1 (action-patch, teacher 제거) | +0.130 | +0.129 | +0.543 | **+0.023** |

- **정정 (a)**: 토큰당 정규화 후에도 B0(+0.287)가 A1(+0.130)의 ~2.2배 유지 → **길이 편향이 gap을 만든 게 아니다.**
  (우리가 어제 "margin +55.8은 길이 아티팩트"라 되돌린 설명은 부정확했음.)
- **정정 (b) — 더 결정적**: B0의 우위는 **task_belief(+0.80)·reasoning(+0.34) span에 집중**, **action span은 +0.014로
  A1(+0.023)보다 오히려 낮다.** 즉 teacher 투영이 A1보다 더 하는 건 **belief/reasoning 텍스트 모방**이고,
  **action 정렬은 A1과 동등(근소 열위).** 이는 리뷰 B0-4를 **정량 확증**한다.
- 정합성: 생성 acc(B0 0.248 ≈ A1 0.254)·③(0.006)와 완전 일치. + `pref_acc≈0`(절대 선호는 아직 chosen<rejected — teacher trace를 실제로 선호하게 되진 않음, "덜 비선호"일 뿐).
- **논문 함의**: "projected full-trace hindsight > action-patch"는 **action 정렬 기준으로 지지되지 않음.**
- 권고 수용: eval을 DPOTrainer collator와 정렬(B0-6) + mean-token·span별 상시 보고.

---

## 3. 우리가 이미 수정/진단한 항목 (리뷰 보강안이 더 견고)

| ID | 우리가 이번 밤 실제로 겪은 것 | 리뷰의 보강안 (수용) |
|---|---|---|
| P0-3 | `max_length 1536`이 프롬프트(이미지 포함)로 completion을 전부 잘라 **무학습**(margin 항등 0). ~1.5h 손실. 4096으로 상향해 grad_norm 16~18 회복. | correctness run은 `max_length=None` + image/action/EOS **토큰 생존율 100% assertion** + truncation 통계 저장. |
| P0-4 | smoke가 trainer를 안 돌려 3h 데이터 구축 후에야 사망. DPO trainer 1-스텝 예행 + **무학습 가드**(FAA 대비 가중치 diff) 편입. | `.smoke_ok`에 git SHA·TRL·파일 해시 묶어 코드/환경 변하면 자동 재smoke. |
| P0-2 | 우리가 넣은 수정이 바로 리뷰가 비판하는 **silent drop**(signature에 없는 인자 전부 자동 제외). | `max_prompt_length`만 화이트리스트, 나머지 미지원 인자는 **RuntimeError**. |
| P0-1 | 무학습 가드는 "가중치=FAA"는 잡지만 "checkpoint-50에서 죽음+가중치 다름"은 못 잡음. | `TRAINING_DONE` 마커(train+final save 후에만) + `--resume_from_checkpoint`. **상보적** — 둘 다 채택. |

---

## 4. 정합성 하드닝 (스토리 불변, 수용)

- **P0-5**: 체인이 `set -uo pipefail`(−e 없음), S9는 bare `python -m accelerate` vs 데이터단 `$PY`. PATH 덕에 지금은 동일 env지만 취약 → `set -euo` + `$PY` 통일.
- **B0-6**: eval tokenization ≠ DPO collator → collator 재사용 (B0-5와 묶어 처리).
- **B0-7**: 우리는 `b0_audit.jsonl`에 belief/action relation·gt_in_candidates 보존 중(부분 충족) → sha·cutoff time 포함 별도 audit manifest로 정식화.
- **B0-8**: belief-swap(③)은 돌렸으나 필수 게이트 아님, image/history swap 미측정 → `TRAINING_DONE`과 별개 `B0_VALIDATED` 게이트(belief/image/history swap + candidate-order 불변 + length-norm margin) 신설. **우리 판정 프레임(③가 진짜 지표)을 코드로 강제.**
- **B0-2/3**: prompt별 rollout 수로 학습 weight 왜곡 + exact-dedup만 → prompt당 hardest-negative 1개 또는 `1/N` 정규화 + 의미 클러스터링.

---

## 5. 다음 단계 — GT-only 재편 (F0 skyline)

§1의 P1-1 발견이 이 실험의 동기를 강화한다: "WM-only"가 실제로는 WM 64% + 사후합리화 36%였으므로,
**GT를 유일 신호로 둔 깨끗한 대조**가 필요하다.

### 사양 (구현 변경은 사실상 보상 함수 한 곳)
```
reward_mode = "gt_only"
  reward = 1.0 if (verb,noun)==GT else 0.0        # 평가와 동일 파서
  format/candidate_gate = validity 게이트(위반 시 고정 음수, additive 아님)
  think_convergence = 미사용 (P1-1 반영 — 사후합리화 항 제거)
zero-advantage 필터 : min_wm_spread 대신 "그룹 전원 정답 또는 전원 오답 프롬프트 스킵"
                      (gen 8, p≈0.24 → 신호 생존 프롬프트 ~89%)
GT∉top5 (38%)       : 드랍(oracle-subset 학습) + 드랍률 로깅. 대안 format-only 보상.
프롬프트/출력/최적화 : 불변 (top-5 셔플·점수 은닉·3-태그 · dr_grpo·LoRA r16·gen8·T1.0·384tok)
```

### 판정
- **GT-GRPO acc ≫ base** → 병목은 WM 신호 품질(관측 병목 rank1|in5). WM=분포 인터페이스 서사 강화.
- **GT-GRPO acc ≈ base** → 병목은 최적화/용량 → WM 신호 개선 투자 불필요라는 진단.
- 어느 쪽이든 논문 4열 표를 채운다: `no-train / WM-GRPO(F0) / GT-GRPO(skyline) / +B0`.

### 방법론 정합
- GT-GRPO는 **본 방법이 아니라 명시적 skyline ablation**으로만. F0의 GT-free 정체성은 유지되고,
  GT는 여전히 B0의 hindsight에만 희소하게 쓰인다(7/17 회의 결정과 일치).
- 예상 공수: 보상/필터 ~1-2h 구현 + 300~500 step ~3h.

---

## 6. Acceptance Criteria 응답 (리뷰 §8 대응)

| 항목 | 현재 | 조치 |
|---|---|---|
| clean clone 실행 | ✗ (evaluate.py) | P0-6 수정 후 충족 |
| stale import 없음 | ✗ | P0-6 |
| smoke가 실제 optimizer step | △ (편입했으나 버전 미묶음) | P0-4 보강 |
| partial checkpoint 안전 resume | △ (무학습 가드만) | P0-1 TRAINING_DONE+resume |
| failed run을 완료로 오인 안 함 | △→○ (가드 도입) | P0-1과 병행 |
| 동일 Python/TRL 환경 | △ | P0-5 |
| image/action/EOS 100% 보존 | 미검증 | P0-3 assertion |
| strict history cutoff | 부분(audit 보존) | B0-7 manifest |
| candidate permutation 불변 | 미검증 | P1-5 unit test |
| F0 reward가 논문 설명과 일치 | **✗ (오염 확정)** | P1-1 재정의 + GT-only skyline |
| full-trace vs action/belief margin 분리 | **○ (재측정 완료)** | §2 결과 반영 |
| length bias 통제 | **○ (mean-token 산출)** | §2 |

---

*근거 파일: `train_grpo_action.py:1311/1116/1063`, `reward_log.jsonl`, `b0/evaluate_retro.py`, `b0/teacher.py`,
`evaluate.py:41/84`, `scripts/step2/eval_battery.py`, `scripts/step2/remeasure_retro_margin.py`,
`runs/f0_battery/remeasure_b0.json`·`remeasure_a1.json`, `docs/experiments/2026-07-17_f0_final.md:153`.
재측정 원자료·수치는 위 JSON에 보존.*
