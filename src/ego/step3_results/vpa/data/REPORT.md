# Ego4D GoalStep VPA (Visual Planning for Assistance) — Baseline Report

- 작성일: 2026-07-19
- 목적: Ego4D GoalStep(요리) 데이터에 COIN 스타일 **VPA 챌린지**를 이식하고, 나중에 도착할
  강화학습 VLM을 채점할 수 있는 코드·평가·baseline 파이프라인을 구축·검증한다.
- 코드: `scripts/vpa/{check_overlap,build_goalstep_vpa,eval_vpa,run_frontier_baseline,run_qwen_baseline}.py`,
  공통모듈 `scripts/vpa/vpa_common.py`, 예시 config `configs/vpa/vpa_config.example.yaml`.

---

## 1. 태스크·데이터·지표 정의

**VPA (Patel et al., ICCV 2023).** "지금까지의 관찰 히스토리 + 명시적 목표(goal)"가 주어지면
"다음 T개 high-level step"을 예측하는 절차 계획 태스크. 표준 horizon **T=3, T=4**.
- 본 이식은 **text-conditioned**: 영상 프레임 대신 `goal_description` + `관찰된 step label 히스토리`를
  입력으로 준다(프레임 입력 훅은 인터페이스에 열어둠 — Qwen 스크립트의 frame hook 주석 참고).

**평가지표 (COIN/VPA 정의 그대로).**
- **SR (Success Rate)**: 예측 T-step 시퀀스가 정답과 **순서까지** 정확히 일치한 샘플 비율.
- **mAcc (mean Accuracy)**: 위치 i마다 `1[pred_i==GT_i]`를 보고 평균(순서 민감, 위치별 정확도).
- **mIoU (mean IoU)**: 예측 step 집합 vs 정답 집합의 IoU(순서 무시)를 샘플별로 구해 평균.
- 표본이 작으므로 **부트스트랩 95% 신뢰구간**(1000 resample)을 함께 보고.

**데이터 (Ego4D GoalStep).** 요리 영상에 goal→step→substep 계층 라벨. 각 영상 = 1개 goal.
평가는 **`goalstep_val.json`의 134개 영상만** 사용(학습 세트 미사용).

---

## 2. 데이터 통계

**오염 검사(작업 1).** train(583) ∩ val(134) video_uid 교집합 = **0** (disjoint). `overlap_report.json`.

**VPA 샘플(작업 2).** level=step, **label-mode=action** — 각 step의 라벨은 Phase-2 taxonomy의
`<verb> <noun>` **action 클래스**(예: `knead dough`, `cook bread`, `add oil`)로, **Step-1
anticipation 모델이 학습/채점하는 것과 동일한 label 공간**이다(`goalstep_parsed_segments.csv` 조인,
OTHER step은 드롭). essential-only, min_observed=1, dev/test = 영상 단위 50/50(seed=42).

| 항목 | T=3 | T=4 |
|---|---|---|
| 총 샘플 | 2,032 | 1,928 |
| dev / test 샘플 | 919 / 1,113 | 869 / 1,059 |
| 샘플 생성한 영상 수 | 104 / 134 | 100 / 134 |
| 영상당 평균 샘플 | 19.5 | 19.3 |
| 후보 어휘(action vocab) 크기 | 252 | 252 |
| dev / test 영상 수 | 67 / 67 | 67 / 67 |

(step_category 문자열(305종) 대비 verb+noun action은 252종 — verb/noun 병합 반영. `--label-mode
step_category`로 원본 문자열 모드도 가능.)

---

## 3. 결과 표 (test split, 값=%, [ ]=부트스트랩 95% CI)

### T = 3

| 모델 | SR | mAcc | mIoU |
|---|---|---|---|
| Most-Probable (train 빈도) | 0.0 | 3.5 [2.9, 4.2] | 5.6 [5.0, 6.2] |
| Most-Probable w/ Goal | 0.1 | 6.4 [5.6, 7.2] | 11.3 [10.5, 12.2] |
| Frontier VLM (API) | — | — | — |
| Qwen3-VL-7B | — | — | — |

### T = 4

| 모델 | SR | mAcc | mIoU |
|---|---|---|---|
| Most-Probable (train 빈도) | 0.0 | 3.5 [3.0, 4.1] | 6.6 [6.0, 7.3] |
| Most-Probable w/ Goal | 0.1 | 6.5 [5.7, 7.2] | 14.5 [13.5, 15.5] |
| Frontier VLM (API) | — | — | — |
| Qwen3-VL-7B | — | — | — |

**Frontier VLM 상태**: endpoint `https://gw.letsur.ai/v1`, model `claude-sonnet-4-6` **정상 작동**.
label 공간을 verb+noun action으로 전환한 뒤, **서로 다른 5개 test 영상에서 1샘플씩**(대표성) 스모크
실행(T=3, 5호출) → 10/10 in-vocab. per-sample CSV: `runs/frontier/frontier_5videos_action_T3.csv`.
집계 **SR=0.0 · mAcc=6.7 · mIoU=4.0** (n=5). 프롬프트는 "goal + 관찰 action(verb+object) + 후보
action 목록 → 다음 T개 action을 목록에서 순서대로 JSON 배열로" 지시. 예측은 절차적으로 타당하나
(bread: `cook bread`가 GT 위치3 적중) 특정 영상의 반복·비정규 실제 순서라 exact/집합 일치는 낮음.
전체 test 채점은 후속(비용). n=5는 baseline과 공정 비교 불가. (키는 환경변수 전용, 어디에도 미저장.)

**Qwen3-VL-7B 상태**: `--dry-run` 스모크 통과(더미 2샘플에서 프롬프트 생성·JSON 파싱 정상,
프롬프트 ~17K자). 가중치 미도착이라 실제 수치는 비움 — 가중치 도착 후 옵션 없이 전량 실행하면 채워짐.

**Sanity 해석**: Most-Probable-w-Goal이 Most-Probable보다 일관되게 높음(goal 조건부가 유효) →
채점 파이프라인이 신호에 반응함을 확인. SR=0은 빈도 baseline이 3~4개 시퀀스를 순서까지 맞히긴
어렵기 때문(정상).

---

## 4. 정성 예시 (test, T=3, 라벨은 축약 표기)

goal = "Makes the bread":

| observed(최근 2) | GT future (3) | Most-Probable w/ Goal |
|---|---|---|
| knead the dough … / make dough by mixing flour … | cook or prepare bread … / cut the dough … / cook or prepare bread … | put the dough on a baking tray / knead the dough … / make dough … |
| make dough … / cook or prepare bread … | cut the dough … / cook or prepare bread … / knead the dough … | put the dough on a baking tray / knead the dough … / make dough … |

Most-Probable-w-Goal은 goal 내 고빈도 step을 순서 무관하게 내뱉어 일부 겹치지만(예: knead the dough),
시간순·반복 구조를 못 맞혀 SR=0. 향후 VLM은 히스토리를 읽고 다음 순서를 추론해야 이김.

---

## 5. 한계·주의

- **자체 구성 VPA**: 라벨 공간이 GoalStep step 어휘(자체 구성, ~305 후보)이며 **표준 VPA
  리더보드(COIN/CrossTask)가 아님** → 외부 SOTA와 직접 수치 비교 불가.
- **bespoke·소표본**: 134영상 → test 67영상. 부트스트랩 CI로 불확실성 표기.
- **text-conditioned**: 이번 평가는 영상 프레임 미사용(관찰 step 텍스트+goal만). 프레임 변형은
  인터페이스만 열어둠(Qwen frame hook).
- **정답 라벨 정규화**: 예측이 후보 어휘 밖이면 (1)정확일치 → (2)대소문자·공백 정규화 →
  (3)문자유사도(difflib) 최근접으로 매핑하고, 각 방법 사용 횟수를 로깅.
- **essential-only 기본**: is_relevant=="essential" step만 사용(`--include-nonessential`로 토글).

---

## 6. 강화학습 VLM을 같은 표에 끼워 넣는 법

평가는 **모델과 완전히 분리**되어 있다. 어떤 모델이든 아래 포맷의 preds json만 만들면 된다:

```json
{ "<sample_id>": ["label1", "label2", "label3"], ... }
```

절차:
1. `goalstep_vpa_T{3,4}.json`의 각 샘플(goal_text + observed_steps + horizon)을 모델에 입력,
   후보 어휘(`candidate_vocab.json`)에서 T개 라벨을 순서대로 예측하게 한다.
   (frontier/Qwen 스크립트의 `build_prompt`를 그대로 재사용 가능.)
2. preds json 저장.
3. 채점 — 기존과 동일:
   ```bash
   python scripts/vpa/eval_vpa.py \
     --gt outputs/goalstep/vpa/goalstep_vpa_T3.json \
     --vocab outputs/goalstep/vpa/candidate_vocab.json \
     --pred <preds_rl_vlm_T3.json> --split test --run-name rl_vlm
   ```
4. 위 결과 표에 "RL-VLM" 한 줄 추가.

---

## 7. 재현 명령

```bash
source ~/ml_env/bin/activate
# (1) 오염 검사
python scripts/vpa/check_overlap.py
# (2) VPA 샘플 (val 134개, T3/T4)
python scripts/vpa/build_goalstep_vpa.py --output-dir outputs/goalstep/vpa
# (3) sanity baseline 채점 (test)
python scripts/vpa/eval_vpa.py --gt outputs/goalstep/vpa/goalstep_vpa_T3.json \
  --vocab outputs/goalstep/vpa/candidate_vocab.json --split test --make-baselines \
  --output-dir outputs/goalstep/vpa/runs/sanity
# (4) frontier VLM (환경변수로 키/endpoint 지정)
export FRONTIER_API_KEY=...            # 절대 하드코딩 금지
export FRONTIER_BASE_URL=<endpoint>/v1
export FRONTIER_MODEL=<vision-model>
python scripts/vpa/run_frontier_baseline.py --gt outputs/goalstep/vpa/goalstep_vpa_T3.json \
  --vocab outputs/goalstep/vpa/candidate_vocab.json \
  --out outputs/goalstep/vpa/runs/frontier/preds_frontier_T3.json --split test --limit 5
python scripts/vpa/eval_vpa.py --gt outputs/goalstep/vpa/goalstep_vpa_T3.json \
  --vocab outputs/goalstep/vpa/candidate_vocab.json \
  --pred outputs/goalstep/vpa/runs/frontier/preds_frontier_T3.json --split test --run-name frontier
# (5) Qwen3-VL (가중치 도착 후; 지금은 --dry-run)
python scripts/vpa/run_qwen_baseline.py --dry-run --gt outputs/goalstep/vpa/goalstep_vpa_T3.json \
  --vocab outputs/goalstep/vpa/candidate_vocab.json --out /tmp/preds_qwen_dry_T3.json
```
