# GoalStep action_end−1s / 8초 V/N/A 실험

- 마지막 자동 갱신: `2026-07-21T22:36:44+00:00`
- 상태: **completed**
- 실시간 UI: https://conventional-vol-storm-matters.trycloudflare.com
- tmux: `ego_goalstep_end_m1_lobs8_vna`

## 실험 정의

기존 `action_start−1s` GoalStep index의 행 순서와 V/N/A label을 고정하고, observation endpoint만 `action_end−1s`로 바꾼 공개 V-JEPA EK100 loader 진단 실험이다. 관측 길이는 최대 8초이며 32 frame을 균일 샘플링하므로 실효 4fps다.

| 항목 | 값 |
|---|---|
| train / val sample | 30,374 / 7,214 |
| label space | verb 81 / noun 140 / action 293 |
| endpoint | `target_end_sec - 1.0s` |
| observation | 최대 8초, 32 frames, 4fps |
| backbone | frozen V-JEPA2 ViT-L/16, 256 |
| probe | depth 4, 16 heads |
| supervision | verb + noun + action focal loss |
| sampler | random, 전체 sample 1회/epoch |
| precision | train BF16 autocast / eval FP32 |
| epochs | 15 |
| batch | 32 |
| LR / WD | 3e-4 / 1e-4 |

## Endpoint 변화의 실측 특성

- train target action 일부가 보이는 비율: `99.832%`
- val target action 일부가 보이는 비율: `99.917%`
- train target action 관측량 중앙값: `11.970s`
- val target action 관측량 중앙값: `11.569s`

이 수치는 진짜 anticipation 난도보다 recognition 성격이 강한 의도적인 대조군이다.

## Epoch별 validation 실측값

| Epoch | Loss | V CMR@5 | V Top-5 | N CMR@5 | N Top-5 | A CMR@5 | A Top-1 | A Top-5 | sec |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2.679 | 33.059 | 66.150 | 24.750 | 58.250 | 18.751 | 12.500 | 37.150 | 286.400 |
| 2 | 1.010 | 41.333 | 70.800 | 32.180 | 68.100 | 29.329 | 22.250 | 48.550 | 276.700 |
| 3 | 0.855 | 49.335 | 73.950 | 33.304 | 69.000 | 31.996 | 21.900 | 48.600 | 281.900 |
| 4 | 0.731 | 46.167 | 73.550 | 40.935 | 69.300 | 34.545 | 23.200 | 49.900 | 280.500 |
| 5 | 0.592 | 44.521 | 72.400 | 37.196 | 67.150 | 32.344 | 23.350 | 48.800 | 280.400 |
| 6 | 0.435 | 46.955 | 72.100 | 37.220 | 67.950 | 35.826 | 23.300 | 51.150 | 280.700 |
| 7 | 0.283 | 49.197 | 73.700 | 38.773 | 66.950 | 34.708 | 22.200 | 48.600 | 279.900 |
| 8 | 0.179 | 44.839 | 68.150 | 39.900 | 66.750 | 33.685 | 21.500 | 45.750 | 277.400 |
| 9 | 0.108 | 48.002 | 69.650 | 40.093 | 64.550 | 36.367 | 22.800 | 47.600 | 280.200 |
| 10 | 0.066 | 46.396 | 69.900 | 40.666 | 64.150 | 35.710 | 23.900 | 47.300 | 279.900 |
| 11 | 0.041 | 50.608 | 70.150 | 44.012 | 66.600 | 36.207 | 24.400 | 47.750 | 280.400 |
| 12 | 0.025 | 48.569 | 70.700 | 42.754 | 65.800 | 37.554 | 24.600 | 47.650 | 280.300 |
| 13 | 0.015 | 48.324 | 69.250 | 42.895 | 66.100 | 37.860 | 24.350 | 47.700 | 279.800 |
| 14 | 0.009 | 50.419 | 69.800 | 43.323 | 65.250 | 38.047 | 24.550 | 47.550 | 280.300 |
| 15 | 0.007 | 50.018 | 69.450 | 43.220 | 65.000 | 38.129 | 24.500 | 47.650 | 280.800 |

## 최종 full-validation 결과

- 이 완료 run 당시의 legacy `best.pt`: **epoch 15** (Action CMR@5 기준)
- 새 기준의 best: **epoch 6** (val-subset Action Top-5 **51.15%**)
- 새 기준 모델 경로: `best_action_top5.pt` → `checkpoints/epoch_06.pt`
- full-val size: **7214**

| Head | CMR@5 | Top-1 | Top-5 |
|---|---:|---:|---:|
| verb | 48.136 | 41.683 | 68.353 |
| noun | 40.750 | 39.798 | 65.484 |
| action | 36.224 | 24.771 | 47.436 |

### Epoch 6 Action Top-5 기준 재평가

고정 2,000개 subset에서 Action Top-5가 가장 높았던 epoch 6을 전체 val 7,214개에 별도 평가했다.

| Head | Epoch 6 CMR@5 | Epoch 6 Top-1 | Epoch 6 Top-5 | Epoch 15 Top-5 | Top-5 변화 |
|---|---:|---:|---:|---:|---:|
| verb | 44.856 | 40.643 | **71.236** | 68.353 | +2.883%p |
| noun | 35.993 | 38.190 | **68.811** | 65.484 | +3.327%p |
| action | 33.410 | 23.371 | **50.042** | 47.436 | **+2.605%p** |

Action Top-5를 primary metric으로 삼으면 epoch 6이 실제 full val에서도 더 낫다. 반대로 Action CMR@5는 epoch 15의 36.224가 epoch 6의 33.410보다 높으므로, checkpoint 선택 지표에 따른 trade-off가 확인됐다.

## 산출물

- config: `configs/step1/goalstep/z1_end_m1_lobs8_vna.yaml`
- index: `src/ego/step1_action_anticipation/goalstep/index_end_m1_lobs8/`
- feature cache: `../datasets/Ego4D/goalstep_feature_cache_end_m1_lobs8_vna/`
- run: `outputs/goalstep/runs/z1_end_m1_lobs8_vna/`
- epoch checkpoints: `outputs/goalstep/runs/z1_end_m1_lobs8_vna/checkpoints/epoch_XX.pt`
- legacy CMR selection: `best.pt` (epoch 15)
- Action Top-5 selection: `best_action_top5.pt` (epoch 6)
- epoch 6 full-val: `full_val_epoch_06.json`, `likelihood_entropy_full_val_epoch_06.jsonl`
- remaining artifacts: `latest.pt`, `final_metrics.json`

## 운영 명령

```bash
tmux list-windows -t ego_goalstep_end_m1_lobs8_vna
tmux attach -t ego_goalstep_end_m1_lobs8_vna
tail -f outputs/goalstep/runs/z1_end_m1_lobs8_vna/logs/pipeline.log
```

SSH/VS Code/GPT 세션 종료는 tmux 내부의 feature 추출, 학습, UI, tunnel, reporter에 영향을 주지 않는다. 서버 재부팅 또는 tmux server 종료는 별도 예외다. Cloudflare quick-tunnel URL은 프로세스 재시작 시 바뀔 수 있다.
