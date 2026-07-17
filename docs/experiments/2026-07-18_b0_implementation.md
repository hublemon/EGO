# B0 — Full-Trace Projected-Hindsight DPO 구현 노트

2026-07-18 · Step 2 (VLM Alignment) · B0 backward 트랙
핸드오프: `EGO_STEP2_B0_FULL_TRACE_DPO_VALIDATION_HANDOFF` (B0 담당자, 2026-07-18)
구현: `src/ego/step2_vlm_alignment/b0/` · 설정: `configs/step2/b0_full_trace_dpo.yaml`

---

## 0. 한 줄 요약

FAA(F0)를 freeze한 뒤, GT trajectory에서 만든 hindsight trace를 시점 `t`로 projection한
**coherent full-trace를 chosen**, frozen FAA online trace를 **rejected**로 쓰는 sequence-level DPO.
reasoning/belief/action을 **분리하지 않는다** (심야 회의 "리저닝 통합 + 프로젝션 후 전체 통합 선택").

## 1. 이 설계가 이전 우려를 해소하는 지점

앞선 이중-pair(HP-DPA) 리뷰에서 지적한 두 교란이 full-trace 방식으로 구조적으로 사라진다:

- **C5 자기모순 교란 해소**: chosen/rejected가 각각 **완결된 원본 trace**다. FAA reasoning에
  GT action을 이어붙이는 splicing이 없으므로 "결론이 뒤바뀐 완성문"이 생기지 않는다.
  `validate_dpo_dataset.check_pair_invariants`가 chosen==projected 원본, rejected==FAA 원본을
  byte 수준으로 확인한다.
- **C4 문체 교란 완화**: `SAME/SAME` drop + `B0-Shuffled` control로 "projector 문체 학습"을
  직접 검사한다. equivalence judge가 belief 동치를 판정해, 의미가 같은데 문체만 다른 쌍을 제거한다.

## 2. 모듈 지도

| 파일 | 역할 | GPU |
|---|---|---|
| `b0/trace_utils.py` | full-trace 파싱/직렬화/canonical (dependency-free) | X |
| `b0/route_pairs.py` | routing table · SAME/SAME drop · candidate support (§8·§10·§13) | X |
| `b0/validate_dpo_dataset.py` | leakage / no-splicing / SAME-SAME assertions (§15) | X |
| `b0/teacher.py` | raw hindsight · projection · equivalence 프롬프트 + frozen base VLM 래퍼 (§5·§6·§9) | 래퍼만 |
| `b0/build_dpo_dataset.py` | offline pair 오케스트레이션 (§26). teacher 주입 가능 | teacher 통해 |
| `b0/merge_b0_samples.py` | faa_traces + b0meta 병합 | X |
| `b0/generate_faa_traces.py` | frozen FAA online full-trace rollout (§4) | O |
| `b0/train_b0_dpo.py` | TRL DPOTrainer, ref=frozen FAA (§11) | O |
| `b0/evaluate_b0.py` | preference margin + GT accuracy split + coherence (§16~21) | 계산부 X / 생성부 O |
| `b0/validate_cli.py` | 저장 데이터셋 재검증 CLI | X |

**순수 로직(GPU 무관)이 B0의 검증 가능한 핵심**이다: routing, SAME/SAME drop, candidate support,
leakage/no-splicing, eval 계산. `scripts/step2/smoke_b0.py`가 이 전부를 합성 데이터로 단언한다.

## 3. 파이프라인

```
① generate_faa_traces  : freeze FAA → prompt 당 full-trace 4개 (F0 프롬프트 빌더 재사용 = 분포 일치)
② merge_b0_samples     : faa_traces + b0meta(gt/future) → samples
③ build_dpo_dataset    : teacher.raw → teacher.project → equivalence → route → emit/audit
④ validate_cli         : 저장 데이터셋 leakage/splicing 재검증 (train 게이트)
⑤ train_b0_dpo         : full-trace DPO (init=FAA, ref=frozen FAA)
⑥ evaluate_b0          : held-out FAA vs B0 (§16~24 Go 판정)
```

## 4. 핵심 계약 (코드로 고정)

- **정보 경계 (§2·§15)**: policy prompt = `(x≤t, H<t^GT, D_t)`만. GT/future/raw/projected/FAA
  trace/equivalence label은 프롬프트에 절대 없음 — `check_prompt_leakage`가 substring으로 검사.
  `future_gt_actions`는 F0에서 이미 `*_b0meta.jsonl`로 물리 분리돼 넘어온다.
- **no-splicing (§0·§15)**: chosen = projected 원본, rejected = FAA 원본. 필드 조립 금지.
- **SAME/SAME drop (§10)**: `belief≡ ∧ action=GT`는 학습 제외, audit 보존. 저장 파일에
  `training_status=DROPPED_SAME_SAME` 태그.
- **candidate support (§13)**: `a_GT ∈ D_t` 아니면 drop + `num_gt_outside_candidates` 집계.
- **stop-gradient equivalence (§7)**: teacher는 `eval()` + `requires_grad_(False)` + `no_grad()`.
- **reference = frozen FAA (§11)**: ⚠ TRL PEFT + `ref_model=None`은 **base**를 reference로 쓴다.
  B0는 FAA adapter를 얹은 **별도 frozen 모델**을 `ref_model`로 명시 전달한다 (train_b0_dpo.py).
- **검사 전용 메타 물리 제거**: build 단계의 `_leak_check`(원본 GT/future 포함)는 저장 시 제거
  (`_strip_leakcheck`) — 학습/감사 파일에 정답이 남지 않는다.

## 5. 검증 (Go/No-Go, §16~24)

네 조건 동시 충족:
1. held-out preference margin(B0) > FAA (특히 DIFFERENT subset)
2. GT/conditional action accuracy > FAA, end-to-end → **62% 목표 근접**
   (candidate_recall / conditional / end-to-end **분리 보고** — filtered를 전체로 오인 금지, §18)
3. full-trace coherence 유지/개선, future leakage 비증가
4. SAME/SAME audit margin 과증가 없음 (projector 문체 학습 아님)

Control 4종 (§22): FAA / B0-Raw(projection 없이) / B0-Projected(MVP) / B0-Shuffled(문체 검사).
judge는 F0와 동일 정책 — **gemini-2.5-pro 단독**, blind, 학습 미사용. human eval blind A/B/TIE 50~100.

## 6. 스모크 (서버에서 실행)

```bash
PYTHONPATH=src python scripts/step2/smoke_b0.py
```
routing 6경로, action relation, candidate support, full-trace round-trip, leakage(GT/future),
splicing 검출, SAME/SAME 물리 분리, accuracy split, recovery/regression, coherence proxy를 단언.
GPU/모델/torch 불필요.

## 7. 알려진 한계 · 확인 필요

1. **로컬 스모크 통과 (2026-07-18)**: Python 3.14.6 + pandas 실설치 후 로컬 실행 —
   `smoke_f0_v2.py` 33/33, `smoke_b0.py` 40/40 PASS. torch 는 스텁이므로 GPU 경로(rollout/
   DPO/eval 생성부)는 스모크 범위 밖 — 서버에서 §6 재실행 후 본 실행으로 확인.
2. **teacher 품질이 상한을 결정**: projection이 미래 누설 없이 belief를 정확히 낮추는지가 핵심.
   `has_future_leak_language`는 스크리닝일 뿐 — judge/human으로 교차 검증.
3. **evaluate.py 레거시 import**: F0 eval이 `train_qwen25vl_grpo_ek100`을 참조하는 문제와 별개로,
   B0 rollout(generate_faa_traces)은 `train_grpo_action`을 직접 import한다 — 서버 경로 확인.
4. **DPO 하이퍼(beta 0.1, lr 5e-6)는 초기값** — margin이 안 벌어지면 beta 스윕.
