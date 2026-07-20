# 명명 규칙 — prospection / retrospection

2026-07-20 확정. 두 학습 트랙의 이름을 **prospection**(약칭 `pro`, 구 `F0`)과
**retrospection**(약칭 `retro`, 구 `B0`)로 통일했다.

| | prospection (`pro`) | retrospection (`retro`) |
|---|---|---|
| 구 이름 | F0 / extro | B0 / intro |
| 성격 | 사전(prospective) — 지금 시점에서 다음 행동을 고른다 | 사후(retrospective) — 나중에 안 사실로 만든 trace에서 선호를 증류한다 |
| 학습 | GRPO → REINFORCE + EMA 기준선 (on-policy) | teacher full-trace → DPO (offline) |
| 코드 | `scripts/step2/pro_*.py|sh` | `src/ego/step2_vlm_alignment/retro/`, `scripts/step2/retro_*.sh` |
| 목표 | WM 후보 중 GT action 선택률 | belief → action 인과 + GT 일치율 |

## 파일 대응표

### 패키지

| 구 | 신 |
|---|---|
| `src/ego/step2_vlm_alignment/b0/` | `src/ego/step2_vlm_alignment/retro/` |
| `b0/train_b0_dpo.py` | `retro/train_retro_dpo.py` |
| `b0/evaluate_b0.py` | `retro/evaluate_retro.py` |
| `b0/merge_b0_samples.py` | `retro/merge_retro_samples.py` |

`teacher.py` · `trace_utils.py` · `route_pairs.py` · `build_dpo_dataset*.py` ·
`build_pairs_contrastive.py` · `generate_faa_traces.py` · `validate_*.py` 는 이름 유지
(트랙 접두어가 없어도 모호하지 않다).

### 스크립트 (`scripts/step2/`)

| 구 | 신 | | 구 | 신 |
|---|---|---|---|---|
| `f0_gr_train.py` | `pro_gr_train.py` | | `b0_p12_chain.sh` | `retro_p12_chain.sh` |
| `f0_gx_train.py` | `pro_gx_train.py` | | `b0_r1_chain.sh` | `retro_r1_chain.sh` |
| `f0_clean_chain.sh` | `pro_clean_chain.sh` | | `b0_full_chain.sh` | `retro_full_chain.sh` |
| `f0_ga_chain.sh` | `pro_ga_chain.sh` | | `b0_auto_chain.sh` | `retro_auto_chain.sh` |
| `f0_we_chain.sh` | `pro_we_chain.sh` | | `b0_ablation_chain.sh` | `retro_ablation_chain.sh` |
| `f0_wema_chain.sh` | `pro_wema_chain.sh` | | `build_b0_pairs.sh` | `build_retro_pairs.sh` |
| `f0_span_chain.sh` | `pro_span_chain.sh` | | `train_b0.sh` | `train_retro.sh` |
| `f0_auto_pipeline.sh` | `pro_auto_pipeline.sh` | | `smoke_b0.py` | `smoke_retro.py` |
| `f0_rerun_signal.sh` | `pro_rerun_signal.sh` | | `remeasure_b0_margin.py` | `remeasure_retro_margin.py` |
| `train_f0_final*.sh` | `train_pro_final*.sh` | | `eval_faa_vs_b0.sh` | `eval_faa_vs_retro.sh` |
| `eval_f0_final.sh` | `eval_pro_final.sh` | | | |
| `build_f0_v2_data.sh` | `build_pro_v2_data.sh` | | | |
| `smoke_f0_v2.py` | `smoke_pro_v2.py` | | | |
| `colab_*f0*` | `colab_*pro*` | | | |
| `f0b0_ab_chain.sh` | `pro_retro_ab_chain.sh` | | | |

`eval_battery.py` · `eval_belief_swap.py` · `check_leakage.py` 는 두 트랙 공용이라 유지.

## 바꾸지 않은 것 — 그리고 그 이유

**실행 산출물의 식별자는 그대로 둔다.** 구체적으로:

- 마커 파일: `F0_WEMA_DONE`, `B0_P12_DONE`, `B0_P12_PASSED`, `F0B0_AB_DONE` …
- 실행 디렉터리: `runs/f0_battery/`, `runs/f0_battery/b0_p12/`, `outputs/step2/f0_*`, `b0_*`
- 실험 ID: `F0-W-EMA`, `F0-WE`, `F0-GR`, `B0-R1`, `B0-P12`

이들은 **이미 쌓인 결과를 가리키는 키**다. 바꾸면 (1) 진행 중이거나 재개 가능한 무인 체인의
idempotency 가 깨지고, (2) 지난 핸드오프 문서·평가 JSON·판정 마커와의 대응이 끊어져
약 10 GPU시간 분량의 결과가 고아가 된다. 코드 이름만 통일하고 데이터 키는 보존한다.

문서에서 과거 실험을 지칭할 때도 `F0-W-EMA` 같은 ID 를 그대로 쓴다. 새로 만드는
실험 ID 부터 `PRO-` / `RETRO-` 접두어를 쓴다.
