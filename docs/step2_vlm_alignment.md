# Step 2 VLM Alignment

Step 2 aligns a Qwen3-VL policy using the Step 1 action prior plus action history and visual context.

## 현재 상태

**F0 (WM-only, GT-free) 트랙 구현·검증 완료** — `docs/experiments/2026-07-17_f0_final.md`.
SFT / noun-stage GRPO 는 미구현(스캐폴드만).

| | |
|---|---|
| 코드 | `src/ego/step2_vlm_alignment/train_grpo_action.py` (학습) · `evaluate.py` (평가) · `judge_reasoning.py` (리즈닝 judge 곡선) |
| 설정 | `configs/step2/f0_final_wm_only.yaml` |
| 실행 | `scripts/step2/train_f0_final.sh` · `scripts/step2/eval_f0_final.sh` |

## F0 설계 요약

```
입력  : 앵커 프레임 + action history + WM joint action 후보 5개 (점수 숨김·순서 셔플)
출력  : <reasoning>…</reasoning> <task_belief>…</task_belief> <action>{"verb","noun"}</action>
리워드: WM likelihood 단독 (+ 구조 게이트). GT 신호 0.
```

**핵심 원칙: reward 는 world-model likelihood 단독.** history-consistency·belief-quality·
ref-model-score 같은 **WM 이 아닌 신호는 넣지 않는다** — 넣으면 "GT-free WM 신호만으로 학습이
성립한다"는 주장이 오염된다. Action History 는 **리워드가 아니라 입력**으로만 쓴다.

## 상한선 (held-out n=500, ViT-G/384 + EK100 probe)

| | |
|---|---|
| 논리적 상한 (GT ∈ joint top-5) | **0.620** |
| 실질적 상한 (WM top-1 = GT) | **0.374** |
| G2 구간 (WM top-1 오답 & GT ∈ top-5) | **0.246** |

## 결과 요약

| | acc | G2 |
|---|---|---|
| Qwen2.5 base @ 25조합 | 0.044 | — |
| Qwen2.5 base @ joint top-5 | 0.248 | 0.333 |
| Qwen3 base @ joint top-5 | 0.230 | 0.301 |
| **Qwen3 step500 @ joint top-5** | **0.258** | **0.309** |

**포맷 효과 +0.204 · 모델 효과 −0.018 · 학습 효과 +0.028.**
G2 는 chance(0.20)를 처음으로 넘었으나, **학습이 올린 것이 아니다**(base 0.301 → 0.309).

## 함정 (재현 시 반드시 확인)

- **`<think>` 를 쓰지 말 것.** Qwen3-VL 토크나이저에서 예약 단일토큰(151667/151668)이고 Instruct
  변형은 절대 생성하지 않는다 → think 파싱 실패 → 수렴 리워드(P4) 영구 0. `<reasoning>` 을 쓴다.
- **`--hide_scores` 와 후보 셔플은 필수.** 점수/순서가 노출되면 top-1 복사로 즉시 퇴화한다.
- **학습과 평가의 `--reward_mode` 를 일치**시킬 것. 프롬프트 포맷이 어긋나면 숫자가 무의미해진다.
- **`max_steps` 는 `save_steps` 의 배수**여야 최종 체크포인트가 생성된다.
