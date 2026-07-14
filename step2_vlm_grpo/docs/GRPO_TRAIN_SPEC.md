# GRPO 학습 명세 — 실험 4 / 5 / 6 (think-format · WM ranking · 2-stage)

> 실험 3 분석 결과, **reasoning 이 reward 에 인과적으로 연결되지 않아** 개선되지 않았다
> (gt_accuracy 평탄 + reason 단어수 15→8 퇴화). 실험 4~6 은 이 구조적 한계를 직접 해결한다.
> 결과 분석은 [GRPO_TRAINING_LOG.md](GRPO_TRAINING_LOG.md) 에 누적한다.

- 학습 코드: [train_qwen25vl_grpo_ek100.py](../train_qwen25vl_grpo_ek100.py)
- 변환 코드: [make_grpo_dataset/convert_to_train_format.py](../make_grpo_dataset/convert_to_train_format.py)
- 자동 실행: [make_grpo_dataset/_bg/auto_all_experiments.sh](../make_grpo_dataset/_bg/auto_all_experiments.sh)

---

## 0. 실험 한눈에

| # | 실험 | `--reward_mode` | 입력(WM) | 출력 포맷 | reward 구성 | 정렬 타깃 |
|---|---|---|---|---|---|---|
| 4 | think-format | `think_format` | verb5 + noun5 분리(셔플) | `<think>…</think><action>{v,n}</action>` | format + **think_quality** + candidate + **gt_accuracy** | EK100 GT |
| 5a | WM ranking | `wm_ranking` | action5 (JSON) | `{action_index,verb,noun,reason}` | format + candidate + **wm_ranking** | WM rank(차등) |
| 5b | think + ranking | `think_ranking` | verb5 + noun5 분리(셔플) | think-format | format + think_quality + candidate + **wm_ranking** | WM rank(차등) |
| 6-S1 | noun ranking | `noun_ranking` | action5 (JSON) | JSON | format + **noun_ranking** | WM noun rank |
| 6-S2 | action ranking | `action_ranking_from_noun` | action5 (JSON) | JSON | format + candidate + **wm_ranking** | WM action rank |

- 6-S2 는 `--resume_lora runs/grpo_stage_noun` 로 **Stage1 LoRA 를 가중치에 병합(merge_and_unload) 후 새 LoRA** 를 학습한다.
- 공통: `num_generations=4`, `per_device_train_batch_size=4`, 2×H200 DDP, lr=1e-5, 1 epoch, temp=0.8/top_p=0.95.
- 실측 step 시간(2GPU): **think 계열 ≈ 10.1 s/step**, **JSON ranking 계열 ≈ 3.8 s/step** (실험3 기준). 3000샘플 = **1500 step/run** (2 prompts/step).

---

## 1. 실험 3 → 실험 4~6 의 핵심 변경

| 항목 | 실험 3 (완료, △) | 실험 4~6 |
|---|---|---|
| 출력 포맷 | `{verb,noun,reason}` — reason 이 답 **뒤** | think-format: `<think>` 가 답 **앞** (인과화) |
| WM 입력 | action top-5 (조합 완성형) | **verb5 + noun5 분리** (Step3 planning 과 일관) / 또는 action5(JSON) |
| reasoning reward | 없음 → 추론 미개선 | `think_quality_reward` 로 **직접 채점** |
| 정렬 신호 | top-1 단일(포화) 또는 GT(평탄) | **WM rank 차등**(rank1=1.0…rank5=0.1) 옵션 추가 |
| 사후 분석 | parquet 수동 | `reward_log` / `completion_samples` / `think_analysis` 자동 로깅 |

---

## 2. 데이터 변환 (`convert_to_train_format.py`)

`grpo_dataset.jsonl` → 학습용 superset JSONL. **모든 `--mode` 가 동일 superset 을 출력**하므로
단일 파일이 모든 reward_mode 에 호환된다(think 의 score 제거·셔플은 학습 프롬프트 생성 시 처리).

출력 필드 (라인당):

```
image_path, episode_id, frame_id, task_goal, memory_context,
topk_nouns[{noun,score}],                 # JSON 프롬프트 noun 후보
topk_actions[{verb,noun,score}],          # JSON 프롬프트 action 후보
topk_verbs[str,…],                        # think 입력 verb 후보 (이름)
topk_actions_with_score[{rank,verb,noun,likelihood}],  # wm_ranking 보상용 (rank 보존)
topk_nouns_with_score[{rank,noun,likelihood}],         # noun_ranking 보상용 (rank 보존)
gt_verb, gt_noun
```

likelihood 검증 완료: `wm_output.top5_verb/noun/action` 모두 **non-null** (verb top1≈0.66, noun≈0.59, action≈0.18).

실행:
```bash
D=data/grpo_dataset
python make_grpo_dataset/convert_to_train_format.py --input $D/grpo_dataset.jsonl --output $D/grpo_train.jsonl
python make_grpo_dataset/convert_to_train_format.py --input $D/grpo_dataset.jsonl --output $D/grpo_train_think.jsonl --mode think_format
# 둘 다 4,998 라인 (동일 superset)
```

---

## 3. 출력 포맷 & 파싱

### think-format (실험 4·5b)
```
<think>
Step 1. [frame 관찰]
Step 2. [목표까지 남은 것]
Step 3. [후보 평가 및 선택 이유]
</think>
<action>{"verb": "wash", "noun": "knife"}</action>
```
`parse_action_from_think_format()` 가 `<action>{…}</action>` 에서 verb/noun 추출, `<think>` 텍스트 별도 반환.
파싱은 `PARSE_FORMAT` 전역(main 에서 reward_mode 로 결정)에 따라 think/json 디스패치 (`parse_vn`).

### JSON-format (실험 5a·6)
```json
{"action_index": 3, "verb": "wash", "noun": "knife", "reason": "..."}
```
기존 `parse_action_json()` 사용. 후보는 stage 지시문(기본 `gt`)으로 제시되며 score 표시·rank 순.

---

## 4. Reward 함수 (구현 기준)

```
format_reward_think        <think>…</think> + <action>…</action> 모두 존재 → 0.15
think_quality_reward       think 20단어↑ & 후보단어 언급 → 0.20 / 10단어↑ → 0.08 / else 0
candidate_reward_think     verb∈topk_verbs & noun∈topk_nouns → 0.50 / 하나만 → 0.10 / 둘다 아님 → -0.20
gt_accuracy_reward_think   verb=GT +0.25, noun=GT +0.35, 둘다 +0.40 (최대 1.0)
wm_ranking_reward          선택(verb,noun) 의 WM action rank → {1:1.0,2:.7,3:.4,4:.2,5:.1}, 후보밖 -0.2
noun_ranking_reward        선택 noun 의 WM noun rank → 동일 테이블, 후보밖 -0.2
format_reward / action_candidate_consistency_reward  (기존 JSON 용)
```
`build_reward_funcs(reward_mode)` 가 위 조합을 반환. WM rank 보상은 `parse_vn` 으로 think/json 양 포맷 지원.

---

## 5. 상세 로깅 (rank0 전용, 사후 비교용)

GRPO sink reward(+`GRPOLogger` 콜백)가 다음을 `runs/{실험}/` 에 독립 기록 (TensorBoard 와 별개, pandas 직독 가능):

| 파일 | 주기 | 내용 |
|---|---|---|
| `reward_log.jsonl` | 매 `logging_steps`(=5) | step, reward_total, **구성 reward 별 mean**, loss, grad_norm, epoch |
| `completion_samples.jsonl` | 매 100 step | 그룹 2개 × 생성 4개 원문 + reward_breakdown, gt/wm_rank1 |
| `think_analysis.jsonl` | 매 100 step (think 한정) | think 단어수(mean/min/max), candidate_mention_rate, **generation_diversity**, think 원문 |
| `meta.json` | 시작 1회 | reward_mode, samples, model, lora_r, max_completion, max_pixels, git_hash, spec md5 |
| `summary.json` | 종료 1회 | total_steps, elapsed_hours, final/trend reward·gt_accuracy, checkpoint |

`generation_diversity` = 그룹 내 서로 다른 (verb,noun) 비율 → mode collapse 조기 탐지.
sink reward 는 항상 0.0 을 반환하므로 GRPO advantage(상대보상)에는 영향이 없다.

---

## 6. 실행

### 자동 (권장) — 5개 순차 + 스모크 게이트 + setsid 분리
```bash
setsid bash make_grpo_dataset/_bg/auto_all_experiments.sh \
  > make_grpo_dataset/_bg/all_experiments.out 2>&1 < /dev/null &
# 진행: cat make_grpo_dataset/_bg/all_experiments.status
```
각 실험: 24샘플 2-step 2GPU 스모크(실패 시 1GPU 폴백) → 통과 시 3000샘플 본학습. PPID=1 분리로 VSCode/세션 종료에도 생존.
순서: 실험4 → 5a → 5b → 6-S1 → 6-S2. 한 실험 실패해도 다음 진행(6-S2 는 6-S1 성공 시에만).

### 수동 (개별)
```bash
ACC="accelerate launch --num_processes 2 --multi_gpu --mixed_precision bf16 --main_process_port 29560"
CM="--train_samples 3000 --num_train_epochs 1 --num_generations 4 --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 1 --logging_steps 5 --save_steps 500 --attn_impl sdpa"
D=data/grpo_dataset

# 실험4 think_format
$ACC train_qwen25vl_grpo_ek100.py --reward_mode think_format \
  --train_jsonl $D/grpo_train_think.jsonl --output_dir runs/grpo_think $CM
# 5a wm_ranking
$ACC train_qwen25vl_grpo_ek100.py --reward_mode wm_ranking \
  --train_jsonl $D/grpo_train.jsonl --output_dir runs/grpo_ranking $CM
# 5b think_ranking
$ACC train_qwen25vl_grpo_ek100.py --reward_mode think_ranking \
  --train_jsonl $D/grpo_train_think.jsonl --output_dir runs/grpo_think_ranking $CM
# 6 Stage1 noun_ranking
$ACC train_qwen25vl_grpo_ek100.py --reward_mode noun_ranking \
  --train_jsonl $D/grpo_train.jsonl --output_dir runs/grpo_stage_noun $CM
# 6 Stage2 action_ranking_from_noun (Stage1 병합 후 새 LoRA)
$ACC train_qwen25vl_grpo_ek100.py --reward_mode action_ranking_from_noun \
  --train_jsonl $D/grpo_train.jsonl --output_dir runs/grpo_stage_action \
  --resume_lora runs/grpo_stage_noun $CM
```
`--max_completion_length` 미지정 시 자동: **think 계열 256, JSON 계열 64**.

---

## 7. 모니터링 포인트 (collapse 조기 경고)

| 지표 | 기대 | 경고 신호 |
|---|---|---|
| `think_quality_reward` | 유지/상승 | 초반부터 0.20 포화 — 형식만 학습 |
| think 단어수 | 20~80 유지 | 5단어 이하 수렴 (실험3 퇴화 재현) |
| `generation_diversity` | 초반 높음 유지 | 전부 동일 → mode collapse |
| `loss` | 양수 유지 | near-zero 수렴 = advantage 소멸(실험1·2 포화) |
| `gt_accuracy`(실험4) | 상승 | 평탄 0.7 = 후보정보만으론 한계 → 영상 reasoning 부족 |

> **WM ranking(5a·6) 주의**: 후보가 score·rank 순으로 보이므로 "rank1만 고르기" 자명해 위험이 있다.
> rank 차등 보상(rank2~5 부분점수)으로 실험1·2 보다 완화되지만, loss near-zero·diversity 붕괴를 반드시 확인.
> 필요 시 `--shuffle_candidates` 추가로 순서 단서를 제거할 수 있다.

---

## 8. 다음 단계 (학습 후)

- 5개 결과를 `reward_log`/`completion_samples`/`think_analysis` 로 비교 → `GRPO_TRAINING_LOG.md` 갱신.
- **EK100 held-out 평가**: 학습 전 Qwen / WM top-1 / 실험2·3 / 실험4·5·6 의 action 정확도 비교 (train reward ≠ 일반화).
- reasoning 정성 평가 (영상 단서 인용·길이·일관성) 자동화.
- memory_context on/off ablation.
