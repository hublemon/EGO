# GRPO WM-only 전환 — 실행 Handoff (학습 run 3회 예산)

> 📎 관련 문서: [GRPO_TRAINING_LOG.md](GRPO_TRAINING_LOG.md) (실험 1~14 이력, Exp.14=`grpo_final`이 현재 GT-primary 최고 성적) · [GRPO_TRAIN_SPEC.md](GRPO_TRAIN_SPEC.md) (실험 4~6 원 명세) · [GRPO_DATASET_SPEC.md](GRPO_DATASET_SPEC.md) (JSONL 필드 정의, `likelihood` 필드 위치)
>
> 2026-07 세션에서 도출. **다음 학습 run 3회 안에** GT-free 전환을 성립시키고 논문 수치까지 뽑는 게 목표. 항별 ablation은 포기하고, 컴포넌트별 reward 로깅(이미 구현됨) + 학습-외 사후 검증으로 원인 추적을 대체한다.

---

## 0. 목표와 판정 지표

**한 문장 목표**: GT(human label)를 학습 신호에서 제거하고, WM(V-JEPA2 probe)의 likelihood 분포 정합만으로 학습하면서, held-out에서 WM 자신의 top-1을 능가하는 VLM을 만든다. GT는 held-out 검증에만 등장.

| 코드 | 목표 | 판정 기준 |
|---|---|---|
| **G1** | GT-free 학습 성립 | WM-likelihood reward만으로 advantage 소실 없이 held-out 곡선 상승 (Exp.10 실패 극복) |
| **G2** | WM 능가 | WM top-1 오답 & top-5 정답 구간(disagreement 구간)에서 VLM 정답 선택률 > chance |
| **G3** | Reasoning 인과성 | think 조건부 선택 likelihood(lift)가 학습에 따라 상승, format 강제 없이 |

### 왜 방향을 바꾸는가

- **Exp.14**(`runs/grpo_final`, reward_mode `think_gt_final`, train-batch reward 0.578)는 **GT accuracy가 주 신호**다 (`docs/GRPO_TRAINING_LOG.md` §4). 논문 클레임("reward는 오직 WM 자신의 예측 분포에서 산출, human label 불요")과 정면 충돌 → **Exp.14는 메인 결과가 아니라 "GT-oracle 상한 참조선"으로 재배치**.
- **Exp.10**(`runs/grpo_think_wm_rank_fix`, WM rank 단독)은 실패했다 (reward≈0 수렴, `docs/GRPO_TRAINING_LOG.md` §3). 원인은 leakage가 아니라 **advantage 소실**: WM 분포가 flat한 샘플에서 그룹 내 생성들의 reward가 비슷해져 gradient가 생기지 않음.
- **기존 실험 전부 train-batch 측정**. held-out 검증은 한 번도 수행된 적이 없다.

### 지표 표기 주의 (논문 작성 시 혼용 금지)

- 자체 CSV 실측(Phase 3, `EPIC_100_validation.csv` 기반, sample-level top-5 hit rate, n=704): verb 90.1 / noun 76.1 / action 상한 69.9%. top-1: verb 54.6 / noun 47.9%.
- V-JEPA2 논문 공식(mean-class recall@5, ViT-g384): verb 63.6 / noun 57.1 / action 39.7.
- 정의가 달라 직접 비교 불가. 자체 수치는 반드시 "sample-level top-5 hit rate"로 명기.

---

## 0.5. Run 1 실행 준비 — ✅ **완료 (2026-07-16). 다음 세션은 `bash run_grpo_run1_wmonly.sh` 한 줄로 시작.**

이번 세션에서 사전 준비 전 항목을 구현하고 **2×H200에서 6-step 스모크로 전 경로를 실제 검증**했다. 재확인 불필요.

- **스모크 결과 (실측)**: 정적 필터 4,646/4,998 통과 → leak assertion 통과 → 6 step 학습(loss 정상, grad_norm 0.04~0.06 비영) → checkpoint-6 저장 → `eval_heldout.py`가 GT/G2/이탈률 전 지표 산출 → `eval_reasoning_trace.py` lift 산출까지 **엔드투엔드 무오류**.
- **컴포넌트별 reward 로그 실측** (`reward_log.jsonl`, 6 step): `reward_wm_likelihood_reward`(P1)가 0.01~0.24로 **변동**(=advantage 생성됨, Exp.10의 reward≈0과 대비), `reward_think_convergence_reward`(P4) 0.003~0.18, gate/format 분리 기록, `ds_frac_groups_filtered`·`frac_reward_zero_std` 동반 기록. → **분기 판정에 필요한 신호가 전부 로그로 나온다.**
- **주의**: 스모크는 6 step이라 수치 자체는 의미 없음(G1 판정 아님). 검증한 것은 "파이프라인이 도는가"와 "로그가 나오는가"뿐.

산출물: `run_grpo_run1_wmonly.sh`(학습), `eval_heldout.py`+`eval_checkpoints_run1.sh`(P0/G1/G2), `eval_reasoning_trace.py`(G3), `data/grpo_dataset/grpo_heldout.jsonl`(1,417).

---

## 1. 지금 코드베이스 상태 (이번 세션에 직접 확인함)

다음 사실은 `train_qwen25vl_grpo_ek100.py` 읽기 + `eve-cu124` 환경에서 `trl==1.5.1` 실제 조회 + 6-step 스모크 실행으로 확인했다. 다음 세션이 재확인할 필요 없도록 기록.

| 항목 | 상태 | 근거 |
|---|---|---|
| 컴포넌트별 reward 로깅 | **이미 구현됨** | `GRPOLogger` 클래스(`train_qwen25vl_grpo_ek100.py:931`)가 `reward_log.jsonl`(reward별 mean), `completion_samples.jsonl`(그룹별 breakdown), `think_analysis.jsonl`(think 단어수·후보언급률)을 매 step 기록 중. 새 reward fn을 `build_reward_funcs`에 등록만 하면 자동으로 로그에 잡힌다. **사전 준비 항목 중 "컴포넌트별 로깅 확장"은 사실상 완료 상태.** |
| `likelihood` 필드 | **존재 확인** | `docs/GRPO_DATASET_SPEC.md`의 JSONL 스펙에 `topk_actions_with_score` 각 항목이 `{"rank":, "action":, "likelihood":}` 형태로 이미 저장됨 (V-JEPA2 probe softmax). `normalize_score()`(`train_qwen25vl_grpo_ek100.py:172`)가 이미 `likelihood` 키를 fallback으로 읽고 있어 **데이터 재생성 불필요**. |
| `loss_type="dr_grpo"` (P5) | **trl 1.5.1에서 지원 확인** | `GRPOConfig.loss_type` 필드가 `'grpo'/'dapo'/'bnpo'/'dr_grpo'` 지원 (docstring 직접 확인). 현재 CLI 기본값은 `dapo`(trl 기본), `run_grpo_final.sh`는 별도 지정 안 함 → 명시적으로 `--loss_type dr_grpo` 추가 필요. |
| Clip-higher (P2 일부) | **trl 1.5.1에서 지원 확인** | `GRPOConfig.epsilon`(하한, 기본 0.2) / `epsilon_high`(상한, 기본 None=epsilon과 동일) 필드 존재. `--epsilon_high` CLI 인자는 현재 스크립트에 없음 → `argparse`에 추가 필요. |
| **Dynamic sampling (P2 핵심)** | ~~trl 미내장~~ → **구현 완료** | `DynamicSamplingGRPOTrainer`(`_generate_and_score_completions` 오버라이드)가 그룹 std ≤ 임계치인 그룹 advantage를 마스킹 + `--min_wm_spread` 정적 필터. `scale_rewards="none"` 강제 검증 포함. |
| Held-out 데이터셋 | ~~없음~~ → **1,417행 생성 완료** | `data/grpo_dataset/grpo_heldout.jsonl`. `EPIC_100_validation.csv` × 디스크 32개 비디오. train(4,998, P01~P06 80개 비디오)과 비디오 단위 완전 분리. |
| P4 관련 기존 로직 | **`think_convergence_reward` 신규 구현** | 기존 `think_quality_reward`(길이 보너스)는 원칙상 배제. P4는 후보 언급의 "마지막 위치 == 최종 선택"(수렴)을 채점 — 후보 집합+텍스트만의 결정론적 함수. |

---

## 2. Run 실행 계획 (3회 한도)

### 사전 준비 — ✅ **전부 구현 완료 (2026-07-16)**

구현 결과와 설계 결정 (원 계획과 달라진 점은 굵게):

1. **Held-out 셋 구축 완료 + 생성까지 실행**: `make_grpo_dataset/` 5개 스크립트에 `--split validation` / `--selected` / `--out` 인자 추가 (기본값 = 기존 train 동작 그대로 → 기존 커맨드 재현성 유지). `EPIC_100_validation.csv` × 디스크 보유 32개 비디오 → **1,417샘플** 실제 생성 완료 (`data/grpo_dataset/grpo_heldout.jsonl`, train과 비디오 완전 분리, 프레임·V-JEPA2 top-5 추론 전량 에러 0). **held-out WM 구조 실측**:
   - WM top-1 == GT (action, fuzzy) = **40.0%** — 모방 상한(참조선). VLM이 WM을 그대로 베끼면 이 값이 천장.
   - GT ∈ top-5 (action, fuzzy) = **62.7%** — VLM 이론적 상한.
   - **G2 구간(WM top-1 오답 & GT∈top5) = 321샘플(22.7%)** — "WM 능가"를 측정할 표본. `--limit 500` 평가 시 약 113개, `--limit 1417`(전량)이면 321개. **G2 곡선 통계력을 위해 Run 3 최종 평가는 전량 권장.**
2. **`wm_likelihood_reward` 구현** (`train_qwen25vl_grpo_ek100.py`): **raw softmax가 아니라 후보셋 내 재정규화(`likelihood/Σtop-5`)가 기본** — 4,998행 실측 결과 raw는 median std 0.015로 format(0.15)/gate(0.5) 항에 묻히고, 재정규화는 median std 0.147로 학습 가능 스케일 (후보 5개가 실제 선택지이므로 조건부 분포가 올바른 target이기도 함). `--wm_likelihood_norm {candidate,raw}`로 전환 가능. null likelihood → 0.0 (실측 null 0건). **`assert_no_score_leak()`**: 데이터셋 빌드 시 raw·재정규화 값의 다양한 자릿수 표기가 프롬프트에 없는지 검사 (GT-free 모드에서 자동 실행).
3. **reward_mode 등록**: `"wm_likelihood": [format_reward_think, candidate_gate_reward_think, wm_likelihood_reward, think_convergence_reward]`. **`think_quality_reward`(20단어 이상 보너스)는 계획과 달리 제외** — '길이 보너스 금지' 원칙과 충돌. 후보 언급 유인은 P4가 기능(수렴) 기준으로 담당. **P4 `think_convergence_reward`도 Run 1에 포함** (계획대로): 최종 선택 noun이 think에 언급 +0.10, 마지막 언급 후보 == 최종 선택(수렴) +0.15, 언급 0회(장식 think) −0.10. 후보 집합+텍스트만의 결정론적 함수.
4. **Dynamic sampling 2단 구현**:
   - **정적 절반 (`--min_wm_spread`)**: reward가 WM likelihood만의 함수이므로 프롬프트별 achievable reward spread는 학습 전에 결정된다 → 재정규화 std < 임계치인 샘플을 데이터셋에서 제거. 0.05로 하위 7%(352/4,998) 제거. **GT-free 모드에선 기존 GT-in-top5 필터를 쓰지 않음** (GT가 학습 분포에 새는 뒷문 차단) — 이 필터가 그 자리를 대체.
   - **런타임 절반 (`DynamicSamplingGRPOTrainer`)**: `_generate_and_score_completions` 오버라이드로 그룹 reward std ≤ `--dynamic_sampling_std_threshold`(0.02)인 그룹의 advantage를 0으로 마스킹 + 필터율을 `reward_log.jsonl`의 `ds_frac_groups_filtered`로 기록. **오버샘플링 재샘플 대신 마스킹** — 정적 필터가 태생적 flat 프롬프트를 이미 제거했으므로 남는 무신호 그룹은 '정책이 같은 답만 생성한 경우'이고, 마스킹으로 노이즈 gradient 차단이 목적. **`scale_rewards="none"` 필수** (std 정규화가 있으면 advantage에서 reward 분산 복원 불가 — 코드에서 강제 검증).
5. **CLI 인자 추가 완료**: `--loss_type`, `--epsilon_high`, `--scale_rewards`, `--dynamic_sampling_std_threshold`, `--min_wm_spread`, `--wm_likelihood_norm`. 전부 미지정 시 기존 동작 유지.
6. **평가 스크립트 2종 작성 완료**:
   - **`eval_heldout.py`** (P0): 체크포인트(LoRA adapter)를 held-out에서 평가 — GT 정확도(fuzzy 포함), **G2**(WM top-1 오답 & GT∈top-5 구간 정답률, chance=0.20), 후보 이탈률, WM-follow rate, WM top-1 참조선. 프롬프트 조립·파싱을 train 스크립트에서 import해 분포 불일치 차단. `eval_checkpoints_run1.sh`가 전 체크포인트 일괄 실행 + 곡선 요약 출력.
   - **`eval_reasoning_trace.py`** (G3): `--mode lift`(계층 1), `--mode shuffle/mask`(계층 2 반사실 — mask가 P3 hacking 검출기), `--mode judge`(계층 3, letsur 게이트웨이 Gemini, `LETSUR_API_KEY` 필요). 판정 모델은 동결 base — 학습 정책 아님.

### Run 1 — WM-only 성립 (G1)

일괄 적용 (묶어도 안전한 이유: P2/P5는 reward가 아니라 최적화 장치라 reward 항들과 간섭 축이 다르고, P4는 결정론적 규칙이라 컴포넌트별 로그에서 기여가 분리 관찰됨).

```bash
bash run_grpo_run1_wmonly.sh   # 작성 완료 — 스크립트 헤더에 Exp.14 대비 변경점 7개 주석
```

주요 설정: `--reward_mode wm_likelihood --loss_type dr_grpo --scale_rewards none --epsilon_high 0.28 --min_wm_spread 0.05 --dynamic_sampling_std_threshold 0.02 --num_generations 8 --beta 0.0 --max_steps 1250 --save_steps 125`. `--drop_unrewardable_samples`는 전달하지 않음(GT 필터 — GT-free 경로에서 무시됨). beta 0.0 = KL off (Dr. GRPO 정합 + ref 모델 메모리 절약). save 125 = 체크포인트 10개 (trace eval 곡선용).

**P3은 Run 1에서 제외** (유일한 신경망 판정 reward 항이라 hacking 위험 최고 — 나머지가 건강한지 먼저 확인).

학습 중/후 held-out 평가를 `grpo_heldout.jsonl`로 체크포인트마다 실행(GT정확도/G2 구간 정확도/후보 이탈률).

**Run 1 종료 후 분기 판정** (컴포넌트별 `reward_log.jsonl` + held-out 곡선으로 판단):

| 관찰 | 진단 | Run 2 방향 |
|---|---|---|
| held-out 곡선 상승 + 무붕괴 + think 단어수 정상 | G1 성립 | **분기 A**: P3 추가(결론 마스킹 필수) |
| reward 정체 + dynamic sampling 필터율 과다 | advantage 소실 잔존 | **분기 B**: 필터 임계치 완화 + 온도 상향. `ds_frac_groups_filtered` 후반 증가 추세가 확인된 경우에 한해 생성 8→12 (§3 고정 설정 참조) |
| `reward_wm_likelihood_reward` 정체인데 `reward_candidate_gate` 등 P4만 상승 | P4 hacking | **분기 C**: P4 가중치 하향, gate 강화 |
| think 단어수 5 이하 붕괴 / 그룹 내 completion 4개 동일 | format-only 붕괴 | **분기 C'**: epsilon_high 추가 상향 + P4 가중치 재조정 |

### Run 2/3 사전 구현 — ✅ 완료 (2026-07-16, Run 1 학습과 병행 준비)

Run 1 이 도는 동안 다음을 미리 구현·검증해 두었다. **Run 1 진단 후 해당 분기 스크립트 한 줄로 즉시 착수 가능.**

1. **P3 구현 완료** (`think_support_reward`, `train_qwen25vl_grpo_ek100.py`): reward_mode `wm_likelihood_p3` = Run 1 구성 + P3. 동결 base 모델(lazy 로드, 프로세스당 1회)로 **text-only** `p(<action> JSON | 후보 목록, 결론-마스킹된 think)` 를 배치 채점 — 이미지 forward 를 빼 오버헤드 최소화 (grounding 은 P1 담당, P3 은 추론→결론 지지만). `p3_mask_conclusion()` 이 think 말미 결론 선언 문장을 제거(답안 예고편 hacking 차단, eval_reasoning_trace 계층 2와 동일 규칙). reward = `--p3_weight`(권장 0.25) × exp(평균 token logp) ∈ [0, w]. 결론-only think·parse 실패는 0. CPU 단위 테스트 통과(마스킹 4케이스, weight=0/parse 가드 시 모델 미로드).
2. **`--reward_weights` CLI** (분기 C 용): reward fn 별 가중치를 GRPOConfig 로 전달 (예: `1,1.5,1,0.5` = gate 강화 + P4 하향). 로깅 sink 는 자동으로 가중치 0.
3. **분기별 launch 스크립트**: `run_grpo_run2_branchA.sh`(+P3 w0.25, ~5.5h) / `branchB.sh`(필터 완화+temp 1.0, `NUM_GEN=12` 는 ds 필터율 후반 증가 확인 시만 — §3 고정 설정) / `branchC.sh`(P4 w0.5·gate w1.5, C' 은 `EPS_HIGH=0.35`).
4. **GT-oracle 참조선 자동 큐잉** (`after_run1_gtoracle.sh`, detached 실행 중): Run 1 held-out 평가 종료를 감지하면 Exp.14(`runs/grpo_final/checkpoint-1250`)를 같은 held-out 500샘플에서 자동 평가 → Run 3 결과표 3번 항목이 Run 1 밤새 확보됨.
5. **곡선 figure 스크립트** (`plot_run1_curves.py`, `/opt/conda/bin/python3` 로 실행): G1 acc 곡선(WM 참조선·GT-oracle 상한 포함) + G2 곡선(chance 라인) + 이탈률/wm_follow 진단 3분할 PNG — Run 3 메인 figure 초안.

### Run 2 — 진단 기반 1회 수정

- **분기 A**: `p(선택|think)` 항(P3) 추가, reference 모델은 학습 전 base 체크포인트 사용. **안전장치 필수**: think 말미 결론 선언 문장 마스킹 후 측정(답안 예고편 hacking 차단), 가중치는 P1/P4보다 작게, 논문에서는 "coherence regularizer"로 WM reward와 분리 서술.
- **분기 B/C**: 해당 처방 1세트만 적용해 재실행. P3는 이 경우 최종 보류 (run 예산상 Run 3에 신규 항 실험 불가).
- P3 hacking 여부는 별도 run 없이 `eval_reasoning_trace.py` 계층 2의 결론-마스킹 테스트가 Run 2 체크포인트에서 "마스킹 후 lift 붕괴 여부"로 판별.

**Run 2 종료 후**: 분기 A였다면 P3 유지/제외 결정. 분기 B/C였다면 G1 성립 여부로 확정 — 불성립 시 **논문 주장 축소 분기**(GT를 보조 신호로 인정, Run 3은 Exp.14 방식 + held-out 검증으로 전환).

### Run 3 — 최종 확정 run (논문용 수치)

Run 2까지 결론으로 설정 고정, 신규 항 추가 금지. 체크포인트 촘촘히 저장(trace eval 곡선용). 종료 후:

1. **G2 곡선** (WM-disagreement 구간 정답률, step별) — 메인 figure 후보.
2. **G3 곡선** (계층 1 lift, step별) + 계층 2 반사실 테스트 + 계층 3 외부 judge 채점.
3. **GT-oracle 참조선**: Exp.14 체크포인트(`runs/grpo_final`)를 동일 held-out에서 평가 → 대비 위치 확인.
4. 인과 접속사 빈도 추이(진단 지표, 보상 아님).

---

## 3. 각 개선이 왜 필요한가 (핵심만)

- **Dynamic sampling** — GRPO는 그룹 내 점수 *차이*로 배운다. WM 분포가 flat한 샘플은 4~8개 생성의 reward가 다 비슷해 gradient가 0. Exp.10의 "reward≈0"이 이것. 동점 그룹을 배치에서 빼는 필터 없이는 재현될 가능성이 높음.
- **Clip-higher(`epsilon_high`)** — 대칭 클리핑은 새로운 시도(원래 확률 낮던 선택지)가 좋은 점수를 받아도 확률을 조금밖에 못 올리게 막아 모델이 하던 것만 하게 만든다. 상한만 풀면 다양성 붕괴를 막는다.
- **Dr. GRPO(`loss_type`)** — 표준 GRPO는 응답당 벌점/보상이 정액제라 길수록 토큰당 몫이 희석된다: 틀린 응답은 길게 쓸수록 저위험, **맞힌 풍부한 reasoning도 토큰당 보상이 희석**되는 거꾸로 된 유인. 상수 정규화로 바꾸면 성과 낸 풍부한 reasoning은 전액 강화, 장황한 오답은 길이 비례 벌점 — 길이 중립이라 "짧게 쓰기 강화" 우려는 없다. 풍부함의 유인 자체는 P4가 담당.
- **P4(후보 언급 일관성)** — 정답 추론일수록 중간 과정이 결론으로 일찍 수렴하고, 오답일수록 후보 사이를 표류한다. discrete 후보 5개라 문자열 매칭으로 구현 가능. "배제 논리가 옳았는가"는 판정하지 않음 — 옳음의 기준을 넣으면 GT가 뒷문으로 들어온다.
- **P3(조건부 likelihood)** — **결정: 학습 전략에서 배제하지 않되 Run 2 분기 A 한정 + 안전장치 필수** (사용자 확정). 동결 reference 모델(로컬 forward, 외부 API 아님)로 `p(선택|think)`를 측정해 reasoning이 결론을 지지할수록 보너스. 유일한 신경망 판정 항이라 hacking 위험이 가장 높으므로: ① think 말미 결론 선언 문장 마스킹 후 측정(답안 예고편 차단), ② 가중치는 P1·P4보다 작게, ③ 논문에서는 WM-유래 reward와 구분되는 **"coherence regularizer"로 분리 서술** ("reward는 오직 WM 분포에서" 문장과의 충돌 관리). 유지/제외는 Run 2 후 계층 2 결론-마스킹 테스트로 판정 — 마스킹 후 lift 붕괴 시 Run 3에서 제외. 매 step 생성 수만큼 추가 forward가 붙어 step당 시간 증가 — Run 2 예산에 반영.

### reward 채택 기준 (전 run 공통)

> 모든 reward 항은 (WM의 출력물, 생성된 텍스트)만의 결정론적 함수여야 한다. 신경망의 판단이 개입하는 항은 reward가 될 수 없다.

| 항 | 기준 통과 여부 | 처리 |
|---|---|---|
| P1 `wm_likelihood_reward` | ✅ WM 분포 그 자체 | 주 신호 |
| P4 후보 일관성 + hallucination gate | ✅ 후보 집합(=WM 출력) 대조 규칙 | 보조 신호 |
| P3 `p(선택\|think)` | ❌ VLM prior의 판단 | **Run 2 분기 A 한정, 별도 항(coherence regularizer)으로 관리 + 마스킹 안전장치** |
| P2, P5 | 해당 없음(reward 아님) | 최적화 장치로 병행 |

### 고정 설정 (사용자 확정, 2026-07-16)

- **생성 수(num_generations)는 8이 기본이며 8 미만으로 내리지 않는다** (하향은 Exp.10 실패 조건의 복원 — 정책 쏠림 p에서 그룹 전원 일치 확률 ≈ p^G, G=4는 학습 후반 50%+).
- **단 하나의 예외 (증거 기반 상향)**: Run 1 `reward_log.jsonl`에서 **`ds_frac_groups_filtered`가 학습 후반에 증가하는 추세가 관찰되면**(= G=8로도 무신호 그룹이 늘어난다는 직접 증거), 분기 B에서 **12로 상향한다** (원 계획 유지). 이 신호가 없으면 분기 B에서도 8을 유지하고 필터 임계 완화(`--dynamic_sampling_std_threshold`, `--min_wm_spread`)·온도 상향만 사용.
- step당 시간은 생성 수에 비례(8→12 시 ~1.5×, Run 2 학습 ~6h)하므로 상향은 위 증거가 있을 때만.

### 금지 목록

- format 세부 제약(길이 보너스/패널티, 특정 어휘 강제) — gate는 `<think></think><action></action>` 최소 구조 검증만.
- 인과 접속사 카운트 보상 — 스팸으로 뚫림. 진단 로깅만.
- 후보 점수/순위의 프롬프트 노출 — 5a에서 확인된 즉시 붕괴 원인 (`HIDE_SCORES` 유지).
- reasoning "옳음" 판정 — GT가 뒷문으로 들어옴. 일관성/hallucination까지만.
- Intuitor(self-certainty)·TTRL(다수결) — EGO가 비판하는 순환 grounding과 동형. 관련 연구 대비 인용용으로만.

---

## 4. 리스크 / 확인 필요 사항

- ~~Dynamic sampling 미구현~~ → **구현 완료** (§2 사전 준비 4). 마스킹 방식이라 오버샘플링 오버헤드 없음 — step당 시간은 Exp.14와 동급 예상.
- **Held-out 확보 완료**: train은 P01~P06 80개 비디오(4,998행)로 확장되어 있었고, held-out은 validation split의 별도 32개 비디오에서 1,417행 확보 (비디오 단위 완전 분리).
- **GPU 예산**: 2×H200 기준 Exp.14가 ~3.5h/1,250 step. Run 1은 동급 + held-out 평가(체크포인트 10개 × 500샘플) 별도 수 시간.
- **주시할 것**: 재정규화 P1은 WM top-1 쏠림(rank1 share median 0.47)이 커서 "항상 rank1 복사" collapse 압력이 있음 — `wm_follow_rate`(eval)와 `generation_diversity`(think_analysis.jsonl)로 감시, 대응은 clip-higher 강화/온도 상향(분기 C'). held-out 실측으로 WM top-1이 GT와 40%만 일치하므로, "항상 rank1"에 수렴하면 held-out action acc가 40% 부근에서 정체되고 G2 구간 정답률은 0에 수렴 — 이 패턴이 곧 collapse 신호다.
- **환경 의존성 (이번 세션에서 해결)**: `eve-cu124` 환경의 `cv2`(V-JEPA2 dataloader가 import)가 시스템 라이브러리 `libxcb.so.1`, `libGL.so.1`을 요구하는데 이 노드엔 없었음 → `apt-get install -y libxcb1 libgl1 libglib2.0-0`로 해결. 새 노드에서 V-JEPA2 추론(held-out 재생성 등) 돌릴 때 `ImportError: libGL.so.1` 나오면 이 패키지 설치. **학습(`train_qwen25vl_grpo_ek100.py`)은 cv2 불필요** — 이 이슈 무관.

---

## 한 장 요약

| Run | 적용 | 목적 | 종료 후 결정 |
|---|---|---|---|
| **사전 준비** | held-out 셋 구축, `wm_likelihood_reward` 구현, dynamic sampling 커스텀 구현, `eval_reasoning_trace.py` 골격 | 실행 가능한 상태 확보 | — |
| **1** | P1(`wm_likelihood`)+P2(dynamic sampling·`epsilon_high`·8생성)+P5(`dr_grpo`)+P4(후보 일관성+gate) 일괄 | **G1 성립** | 로그 진단 → 분기 A(P3 추가)/B(신호 소실 수정)/C(hacking 억제) |
| **2** | 분기별 1세트 수정 (A면 +P3, 결론 마스킹 필수) | 설정 확정 | P3 유지/제외, G1 불성립 시 주장 축소 분기 |
| **3** | 설정 고정, 신규 항 금지 | **논문 수치 생산** | G2·G3 곡선 + GT-oracle 참조선(Exp.14) + judge 채점 |

**관통 원칙**: 형식을 강제하는 규칙을 추가하지 않고, 잘못된 유인 구조를 제거한다. reward에 관여하는 것은 WM의 출력물뿐이며, ablation의 부재는 컴포넌트별 로깅(이미 구현됨) + 학습-외 반사실 테스트로 대체한다.
