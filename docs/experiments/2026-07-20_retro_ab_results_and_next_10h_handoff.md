# RETRO (B0) 2026-07-20 실행 결과 · 근본 원인 · 다음 10시간 계획

> 작성: 2026-07-20 14:0x UTC (23:0x KST)
> 대상 독자: 다음 세션에서 retro 트랙을 이어받는 사람
> 선행 문서: [`2026-07-20_f0_results_b0_prevalidation_handoff.md`](2026-07-20_f0_results_b0_prevalidation_handoff.md)
>
> 명명: F0/B0 → prospection(pro) / retrospection(retro)로 통일됨(커밋 `225c699`).
> **단, 실행 산출물의 식별자는 바뀌지 않았다** — 디렉터리·마커는 여전히 `b0_p12`, `B0_P12_DONE` 등.
> 아래에서 경로는 실제 디스크 이름을 그대로 쓴다.

---

## 0. 한 줄 결론

**오늘 검증한 가설 "credit을 action span에 국소화하면 belief→action 인과가 생긴다"는 두 트랙 모두에서 기각됐다.**
그리고 실행 데이터는 그 가설이 왜 애초에 성립할 수 없었는지를 보여준다 — **action span은 결정이 내려지는 자리가 아니라 읽어내는 자리이고, belief→action 인과 경로는 강화할 것이 없을 만큼 존재하지 않는다.**

이 기각은 예고돼 있었다. `pro_retro_ab_chain.sh` 헤더가 직접 써 뒀다:

> `둘 다 무반응이면 credit 국소화 가설이 기각되고, 남는 후보는 belief 를 직접 조작하는 P3(반사실 쌍) 뿐이다.`

둘 다 무반응이었다. 따라서 **다음 단계는 P3이며, 아래 §6에서 이를 "쌍 만들기"가 아니라 "인과를 목적함수로 승격"으로 강화해 제안한다.**

---

## 1. 실행 인벤토리 (UTC 2026-07-20)

산출물은 전부 `/mnt/nvme/migration/jihun/EGO/runs/` (아티팩트) + `/mnt/nvme/migration/jihun/EGO_jihun/outputs/step2/` (어댑터).
`EGO_jihun`은 실행 당시 `EGO_repo_snapshot`이었고 12:07에 개명됐다 — 결과 JSON 안의 `/mnt/nvme/migration/jihun/EGO_repo_snapshot/...` 경로는 **깨진 경로가 아니라 개명 전 기록**이다.
`EGO_jihun2`에는 오늘 retro 실행이 **하나도 없다** (12:35 생성된 새 클론, Step-1 전용).

| # | 실행 | 아티팩트 | 시각 | 상태 |
|---|---|---|---|---|
| 1 | **B0-P12** — P1(자기대조)+P2(최소대조) DPO | `EGO/runs/f0_battery/b0_p12/` | 01:59 → 08:20 | 완료 · **사전등록 기준 미충족** (`B0_P12_PASSED` 미생성) |
| 2 | **ARM A** — P2-only DPO | `EGO/runs/f0_battery/ab_a/` | 08:20 → 08:59 | 완료 · 전 span 악화 |
| 3 | **ARM B** — PRO W-EMA + `--credit action` | `EGO/runs/f0_battery/train_ac.log` | 08:22 → 12:06 | 완료 · 학습 신호 소멸 |
| 4 | acc 재현 검증 ×2 | `f0_battery/retro_p12_acc_repro_{1,2}.*` | 08:59 → 09:09 | 완료 · **정보량 0** (§5-C) |
| 5 | subset 편향 검증 | `f0_battery/retro_p12_alt500.*` | 09:10 → 09:20 | 완료 · **유일한 깨끗한 양성** |

크래시·NaN·OOM 없음. `B0_P12_FAILED` / `F0B0_AB_FAILED` 마커 없음. 로그의 유일한 경고는 HF 토큰 및 `use_fast` deprecation.

**설정 파일이 없다.** 모든 하이퍼파라미터가 체인 스크립트의 CLI 플래그다(`retro_p12_chain.sh`, `pro_retro_ab_chain.sh`). `configs/step2/b0_full_trace_dpo.yaml`은 존재하지만 오늘 실행에 쓰이지 않았다. `.pid`·tmux 세션도 남지 않아 **실행 출처는 체인 로그가 전부다.** → §6 개선 0.

---

## 2. 사전등록 기준 대비 실측

`f0_battery/B0_P12_RESULTS.md` (자동 생성) + `remeasure_b0p12.json` 원본 대조:

| 지표 | MVP | R1 | **목표** | **P12 실측** | 판정 |
|---|---|---|---|---|---|
| 생성 acc | 0.248 | 0.238 | ≥ 0.26 | **0.264** | ✓ |
| action-span margin | +0.014 | +0.007 | ≥ +0.023 | **−0.0048** | ✗ **부호 반전** |
| G2 | 0.342 | 0.2764 | (참고) | 0.3577 | — |
| ③ 인과민감도 | 0.006 | 0.008 | (다음 단계) | 0.006 | 변화 없음 |

**핵심 이상**: `1152/1152` P2 쌍이 `<action>` 이전까지 **문자 단위로 완전히 동일**한데(직접 검증함, 불일치 0), action-span margin이 오히려 음수가 됐다.

span 분해 (`remeasure_b0p12.json`, n=403, ref=FAA):
```
improvement_vs_ref.span = { reasoning: +0.1204,  task_belief: +0.4035,  action: −0.0048 }
improvement_DIFF_DIFF   = { n: 115, mean_token: +0.1054, action_span: −0.0137 }
```
credit은 여전히 `task_belief`가 전부 가져갔다(+0.40). action은 부호가 뒤집혔다.

---

## 3. 학습 곡선 (실측 시계열)

오늘 실행에는 `training_history.csv`도, `metrics_per_epoch.json`도, wandb도 **없다.** HF `trainer_state.json` + TB event + `gr_log.jsonl`이 전부다. → §6 개선 0.

### 3a. B0-P12 DPO — 정상 학습, 그러나 전이 없음
`outputs/step2/b0_p12_1f/checkpoint-267/trainer_state.json` · 267 step = 1 epoch / 4267 쌍

| step | loss | acc | margin | grad_norm |
|---:|---:|---:|---:|---:|
| 2 | 0.699 | 0.188 | −0.011 | 4.40 |
| 62 | 0.595 | 0.719 | 0.251 | 3.74 |
| 122 | 0.547 | 0.750 | 0.509 | 3.90 |
| 182 | 0.487 | 0.781 | 0.653 | 3.13 |
| 262 | 0.448 | 0.813 | 0.816 | 3.21 |
| 266 | 0.545 | 0.750 | 0.483 | 4.10 |

loss 0.699→0.448 (mean 0.566), acc 0.19→0.81 (mean 0.731), margin −0.011→0.816 (mean 0.388). NaN 없음, grad 스파이크 없음. **학습 자체는 건강하다.**

### 3b. ARM A (P2-only) — 교과서적 shortcut learning
`outputs/step2/b0_p2only_1f/checkpoint-117/trainer_state.json` · 117 step / 1872 쌍

| step | loss | acc | margin |
|---:|---:|---:|---:|
| 2 | 0.693 | 0.250 | −0.000 |
| 22 | 0.518 | **1.000** | 0.390 |
| 62 | 0.298 | **1.000** | 1.067 |
| 102 | 0.202 | **1.000** | 1.544 |
| 116 | 0.238 | **1.000** | 1.356 |

**step 4 이후 train accuracy가 사실상 1.00 고정**(전체 평균 0.9855), margin은 P12의 3배(평균 0.902), entropy는 0.377→0.297로 단조 감소.
그런데 heldout은 **전 span에서 악화**(`remeasure_abA.json`):
```
p2only.improvement_vs_ref.span = { reasoning: −0.1611, task_belief: −0.5747, action: −0.0200 }
pref_acc_sum: 0.0422 (FAA) → 0.0248,  mean_token_margin: −0.627 → −0.785
```
`F0B0_AB_RESULTS.md`의 belief:action 비율 칸은 `—`로 출력된다 — 두 값이 모두 음수라 비율 포매터가 처리 못 한 것. **이 arm의 선언된 목적 지표가 계산 불가 상태다.**

> 판별이 쉬워질수록 전이가 사라진다. action 토큰만 다른 쌍은 "정답 토큰 맞히기"라는 지름길을 제공하고, 모델은 그것만 배운다.

### 3c. ARM B (credit=action) — 그래디언트 소멸
`outputs/step2/f0_wema_actioncredit_1f/gr_log.jsonl` · 200샘플마다

```
seen  200  loss -0.0002  reward_ma 0.2953  baseline 0.3057  mean_abs_adv 0.1981
seen 1000  loss -0.0000  reward_ma 0.2652  baseline 0.2665  mean_abs_adv 0.1797
seen 2400  loss -0.0001  reward_ma 0.3006  baseline 0.3070  mean_abs_adv 0.2094
seen 3600  loss -0.0000  reward_ma 0.2836  baseline 0.2877  mean_abs_adv 0.1941
seen 4800  loss -0.0000  reward_ma 0.3120  baseline 0.2870  mean_abs_adv 0.2137
```
24개 전 구간에서 loss ∈ [−0.0003, +0.0001], reward_ma는 0.264–0.318 밴드를 벗어나지 않음.

**`credit=all` 동일 트레이너 대조** (`outputs/step2/f0_wema_fulltrace_1f/gr_log.jsonl`, 25행 전량):
```
credit=all    -0.0032 -0.0012 -0.0031 -0.0047 -0.0058 -0.0026 +0.0013 -0.0030 -0.0027 -0.0057
              +0.0028 +0.0015 -0.0092 +0.0038 -0.0114 +0.0006 +0.0024 -0.0095 -0.0038 -0.0034
              -0.0033 +0.0027 -0.0042 -0.0080 -0.0041      range [-0.0114, +0.0038]
credit=action -0.0002 -0.0000 -0.0003 -0.0002 -0.0000 -0.0001 ... (24행 전부)
                                                              range [-0.0003, +0.0001]
```
| | mean&#124;loss&#124; | range |
|---|---:|---|
| credit=all | **0.004160** | −0.0114 … +0.0038 |
| credit=action | **0.000092** | −0.0003 … +0.0001 |
| 비 | **45.4×** | |

**loss 크기가 45배 작다.** (주의: `credit=all`은 단조 하강이 아니라 부호가 진동한다 — REINFORCE의 정상 거동이다.
"credit=action이 평평하다"의 대조군은 "하강하는 곡선"이 아니라 **"진폭이 45배 큰 진동"**이다.)

기계적 원인을 코드에서 확인함 — `EGO_jihun/scripts/step2/pro_gr_train.py:187-193`:
```python
if args.credit == "action":
    k = _action_token_start(processor.tokenizer, comp_ids)
    span_lp = tok_lp[k:] ...
    tok_lp = span_lp
loss = -(adv * tok_lp.mean()) / args.accum
```
`<action>` span은 `{"verb": "put", "noun": "lid"}` 수준의 **거의 결정적인 JSON 토큰 몇 개**다. 이들의 logp ≈ 0 이고, `.mean()`이 span 길이로 다시 나눈다 → advantage가 아무리 커도 gradient가 소멸한다.

결과: ③ 인과민감도가 **0.0041로 오히려 악화**(base 0.016, W-EMA credit=all 0.0081, 목표 >0.03). acc 0.256, G2 0.3415(0.3821에서 하락).

---

## 4. Reasoning trace — 실제 생성물

출처: `EGO/runs/f0_battery/b0p12_gen_1f.records.jsonl` (500건, 132 정답 = 0.264)

**정답 예 (`P01_11_37`, GT `put lid`)**
```
<reasoning>
I am currently holding a blue lid and appear to be in the process of closing a container.
Given the recent action history, which includes taking and closing a container, it is likely
that I am finalizing the storage of the pizza or another item. The most logical next step is
to complete the action of putting the lid on the container to seal it properly.
</reasoning>
<task_belief>store the pizza in a container</task_belief>
<action>{"verb": "put", "noun": "lid"}</action>
```

**오답 예 (`P03_24_126`, GT `scoop salmon` → pred `put food`)** — 그럴듯하지만 일반적. belief와 action이 서로 모순이 없어 인과 스트레스가 걸리지 않는다:
```
<task_belief>serve food onto a plate</task_belief>
<action>{"verb": "put", "noun": "food"}</action>
```

**퇴화 실패 (`P03_23_116`, GT `check heat`)** — P12 유일의 파스 실패(parse_rate 0.998). 후보 소거 루프에 빠져 `<action>`을 못 내고 `max_new_tokens=384`에서 절단:
```
... The only viable action is "turn-off oven" if the oven is being turned off, but that
contradicts the context of ongoing cooking. However, "turn-on heat" is not correct either.
The only remaining candidate that makes sense is "turn-off oven" — but that would be counter
to the cooking task. The only candidate that fits is "turn-off oven" — but that contradicts
the cooking context. The only candidate that fits is "
```
ARM B(`abB_gen_1f.records.jsonl`, `P01_14_323`)에도 동일 병리 + `Wait — I must have misread.` 백트래킹이 나타난다.

**퇴화 붕괴는 없다**: 500개 완성문 중 빈 문자열 0, 평균 684.3자, **중복 최대 1회**. reasoning 단어수 93.1(P12)/100.4(A)/86.5(B), base ~103 범위 내.

**학습 쌍 예** (`b0_p12/pairs_p12.jsonl`, P2 `P06_05_217:p2:0`) — 접두가 바이트 단위로 동일:
```
CHOSEN   … <task_belief>prepare and serve pizza</task_belief>
         <action>{"verb": "put",  "noun": "pizza"}</action>
REJECTED … <task_belief>prepare and serve pizza</task_belief>
         <action>{"verb": "turn", "noun": "pizza"}</action>
```
반면 **P1 쌍은 reasoning 텍스트가 서로 다르다** — 즉 P1은 action span을 격리하지 못한다. P12가 P1 2564 + P2 1152로 구성됐으므로 **쌍의 69%가 span 확산을 유발한다.**

---

## 5. VLM judge 품질 — 오늘은 실행 없음, 그러나 결정적 결함이 문서화됨

**오늘 judge 실행은 0건이다.** 최신 judge 산출물은 7/18–19자(`f0_final_v2_val_1f/judge_curve.jsonl` 등). 다만 그 결과가 오늘 핸드오프 §2.7에 정리됐다.

judge = **gemini-2.5-pro**, 동일 heldout 40샘플, 7항목 × 2점 = 14점:

| 항목 | base | F0-W | F0-WE | F0-W-EMA | 스프레드 |
|---|---|---|---|---|---|
| 행동 이력 활용 | 2.00 | 1.95 | 1.97 | 1.97 | 0.05 |
| **후보 검토** | 0.56 | 0.78 | 0.63 | 0.78 | **0.22** |
| 시각 근거 | 1.69 | 1.76 | 1.63 | 1.81 | 0.18 |
| 결론의 논리성 | 1.95 | 2.00 | 1.90 | 1.92 | 0.10 |
| 환각 없음 | 1.92 | 1.68 | 1.76 | 1.95 | 0.27 |
| **belief 전역성** | 2.00 | 2.00 | 2.00 | 2.00 | **0.00** |
| belief–action 연결 | 1.97 | 1.95 | 1.90 | 2.00 | 0.10 |
| **합계** | 12.10 | 12.11 | 11.79 | 12.43 | **0.64 / 14** |

### judge 신뢰성 실패 — 원문 인용
> **F0-W-EMA의 belief_action_link는 2.00/2.00 만점인데, 같은 정책의 ③ 인과민감도는 0.008이다.**
> belief를 통째로 다른 것으로 바꿔치기해도 action은 사실상 그대로다.
> **LLM 판정단은 인과의 부재를 탐지하지 못한다.**

**판정: judge를 목표 지표로 쓸 수 없다.**
- `belief 전역성`은 4개 정책 전부 정확히 2.00 → **판별력 0**
- 전혀 다른 4개 정책의 총점 차이가 14점 중 0.64점
- 실동적 범위가 있는 항목은 `후보 검토`(0.56–0.78) 하나뿐

### 오늘 실제로 돌아간 "judge"는 teacher gate다
`EGO_jihun/src/ego/step2_vlm_alignment/retro/teacher.py` · `GatedTeacherMixin.generate_gated_trace`
규칙: 시도 1 greedy, 2–4는 T=0.8. `canonical(pred) == canonical(GT)` **그리고** pred ∈ 후보 5개일 때만 PASS. drop-not-patch.

---

## 6. 근본 원인 — 실행으로 확인된 것

### RC1. action span은 credit을 실어 나를 수 있는 자리가 아니다
두 arm이 **같은 원인, 다른 증상**을 보였다.

| | 증상 | 증거 |
|---|---|---|
| ARM B (online) | gradient 소멸 | mean\|loss\| 0.0042 → 0.000092 (**45×** 축소), `tok_lp.mean()`이 거의 결정적인 JSON 토큰 몇 개 |
| ARM A (offline) | shortcut learning | train acc 4스텝 만에 1.00, heldout 전 span 악화 |

`<action>`은 **결정이 내려지는 곳이 아니라 이미 내려진 결정을 읽어내는 low-entropy readout**이다.
여기에 credit을 걸면 → online은 신호가 0이 되고, offline은 지름길이 된다.

### RC2. belief→action 인과 경로가 존재하지 않는다 (강화할 것이 없다)
belief-swap 개입 실측:
- `swap_b0p12.records.jsonl` (998행): control 변화 1/499, swap 변화 **4/499** → ③ = 0.006
- `swap_abB.records.jsonl` (997행): control 4/497, swap **6/497** → ③ = 0.0041

**belief를 통째로 다른 것으로 갈아끼워도 heldout 전체에서 action이 바뀐 건 4건, 6건뿐이다.**
belief와 action은 시각+이력 컨텍스트로부터 **각각 독립적으로** 생성된다. credit 국소화는 없는 경로를 강화하려던 시도였다.

### RC3. 측정 정밀도 < 주장하는 효과 크기 ← **가장 시급**
- **subset만 바꿔도 acc가 0.264 → 0.302로 움직인다 (Δ0.038).** 하루 종일 주장한 효과 크기는 0.02–0.03.
- n=500 이항 se = 0.020
- 재현 3회가 전부 정확히 0.264이고 `.records.jsonl` 3개가 **446,162바이트로 바이트 동일**. `eval_battery`가 `do_sample=False`(greedy)라 **평가 잡음이 애초에 0**이다. 문서가 스스로 정정함:
  > `3회 동일 0.264 는 "환경·파이프라인 재현성"의 확인이지, 표본 잡음 강건성의 증거가 아니다.`
  → 약 15분의 GPU 시간이 정보량 0으로 소모됐다.

### RC4. 학습 신호가 데이터 층위에서 희박하다
`b0_p12/stats_p12.json` (1500 프롬프트 × 8 롤아웃):
```
mixed 641 / all_correct 278 / all_wrong 581 / no_valid_parse 0
```
**57.3% (859/1500)가 롤아웃 분산 0** → P1 쌍을 만들 수 없다. OUT 그룹 114개는 쌍 0개.

teacher gate (`t_stats_{0,1}.json`): 통과 **71/463 = 15.3%** (R1의 ≈53%에서 급락). GT가 후보셋 밖이라 드롭된 샘플 114개.

**teacher mode collapse** (`t_audit_{0,1}.jsonl`, 392 실패행):
- **219/392 (55.9%)가 4회 시도 전부 동일 예측**. 분포 `{1:219, 2:133, 3:37, 4:3}`
- 반복 오답 상위: `turn-off|tap` ×47, `mix|meat` ×34, `adjust|heat` ×32
- **T=0.8이 탐색을 거의 만들지 못한다.**

### RC5. judge가 이 문제를 볼 수 없다
§5 참조. belief-action link 만점 ↔ 실제 인과 0.008.

### RC6. 실행 출처가 남지 않는다
설정 YAML 없음(전부 CLI 플래그), `.pid` 없음, tmux 없음, `training_history.csv`/wandb 없음. 재현 가능한 것은 체인 로그뿐이다.

---

## 7. 개선안 — 근거와 함께

### 개선 0 (전제) · 실행 기록을 산출물로 만든다
**근거**: RC6. 오늘 5개 실행의 하이퍼파라미터가 셸 스크립트에만 존재한다.
- 실행마다 `run_config.json`(전체 CLI + git SHA + 데이터 해시) + `training_history.csv` 강제 기록
- 비용 거의 0, 이후 모든 비교의 전제

### 개선 1 (최우선) · 방법보다 측정을 먼저 고친다
**근거**: RC3. 효과(0.02–0.03) < 측정 흔들림(0.038). **지금 상태로는 어떤 개선도 검증할 수 없다.**
- heldout **전량** 사용 + 최소 2개 disjoint subset 병기 + 부트스트랩 95% CI
- greedy 반복 금지. 재현성은 seed×샘플링으로 측정
- 사전등록 임계값을 se 기반으로 재설정 → **최소 검출 가능 효과 ≈ 2 × 0.032 ≈ 0.064**
  (오늘의 "목표 ≥0.26 vs 실측 0.264"는 검출 불가능한 크기였다)

### 개선 2 (주력) · 인과를 "측정"에서 "목적함수"로 승격한다
**근거**: RC2. ③은 측정만 해서는 0.006에서 움직이지 않는다. 문서가 지목한 P3(반사실 쌍)의 강화판.
- **belief-swap consistency loss**: 같은 프레임에 belief를 GT/반사실로 교체 주입하고, action이 그에 **맞게 바뀌도록** 직접 학습. 개입 테스트를 그대로 학습 신호로 만든다.
- belief dropout을 병용해 "belief를 무시하면 손해"인 상황을 강제
- ③이 곧 학습 목표이므로 **평가지표 유출에 주의** — 학습용 반사실 belief와 평가용 swap belief 풀을 분리할 것

### 개선 3 (즉효) · `mean` → `sum`, 그리고 credit을 belief로
**근거**: RC1. `loss = -(adv * tok_lp.mean())`가 span 길이로 나눠 신호를 소멸시킨다.
- 즉시 수정: `tok_lp.mean()` → `tok_lp.sum()` (또는 credit=all과 스케일 정합). 1줄, 30분 검증 가능
- 더 나은 방향: credit을 action이 아니라 **belief 토큰**에 건다. 어차피 span margin을 belief가 전부 가져가고 있고(+0.40), belief는 action과 달리 고엔트로피 자유 텍스트라 gradient가 살아 있다

### 개선 4 · 롤아웃 분산을 확보한다
**근거**: RC4. 57.3%가 분산 0, teacher 55.9%가 4/4 동일.
- `num_generations` 4 → 8, 온도 상향(T 0.8 → 1.0–1.2), nucleus 다양화
- mixed 프롬프트 우선 커리큘럼 (all_correct/all_wrong은 쌍 생성에 기여 0)
- teacher gate에 diverse decoding 적용 — 현재 4회 시도가 사실상 1회다

### 개선 5 · judge를 목표 지표에서 내린다
**근거**: RC5. 총 스프레드 0.64/14, 포화 항목 2개.
- `belief 전역성`(스프레드 0.00) 제거, `행동 이력 활용`(0.05) 제거
- judge는 **진단 보조**로만 유지, 사전등록 기준은 ③ 인과 + acc CI로
- judge를 계속 쓴다면 인간 라벨 소표본과의 일치도를 먼저 측정할 것

### 개선 6 · 후보 소거 루프를 차단한다
**근거**: 파스 실패가 전부 이 패턴 + `max_new_tokens=384` 절단.
- repetition penalty, `max_new_tokens` 상향, 또는 `<action>` 포맷 강제 디코딩
- 빈도는 낮지만(0.2–0.6%) reasoning 품질 저하의 가시적 신호

---

## 8. 다음 10시간 실행 계획

**설계 원칙**: 개선 1(측정)을 먼저 고정하지 않으면 나머지 9시간이 오늘처럼 해석 불가능해진다. 따라서 측정 하네스가 H0–H1을 차지한다.

| 구간 | 작업 | 산출물 / 게이트 |
|---|---|---|
| **H0–H0.5** | 개선 0+1: `run_config.json`·`training_history.csv` 기록, 평가 하네스를 heldout 전량 + 2 disjoint subset + bootstrap CI로 교체 | `eval_harness_v2.py`, FAA 기준선 재측정 (CI 포함) |
| **H0.5–H1** | 개선 3 즉효: `mean`→`sum` 수정 후 300샘플 스모크. loss 크기가 `credit=all` 수준으로 복귀하는지만 확인 | **게이트 A**: loss 크기 ≥ 10× 회복 실패 시 ARM B 계열 폐기하고 H4–H6을 개선 2에 재배정 |
| **H1–H4** | **개선 2 주력**: belief-swap consistency 학습 (반사실 belief 주입 + belief dropout) | 어댑터 `retro_p3cf_1f` |
| **H4–H6** | 개선 3 본실행: credit=**belief** REINFORCE (게이트 A 통과 시 credit=action-sum도 병행) | 어댑터 `pro_beliefcredit_1f` |
| **H6–H8** | 개선 4: `num_generations` 8 + T 1.0 + mixed 우선 커리큘럼으로 롤아웃 재생성, P3 쌍 재구축 후 재학습 | `stats_p3.json` — **분산 0 비율 57.3% → <35% 확인** |
| **H8–H9.5** | 전 arm 개입 기반 평가: ③ 인과, acc(CI), span 분해, 2 subset 병기 | `RETRO_P3_RESULTS.md` |
| **H9.5–H10** | 결과 문서화 + 다음 핸드오프 | 본 문서 갱신 |

### 사전등록 기준 (이번 라운드)
측정 하네스 교체 후 확정하되, 현재 근거로 제안하는 값:

| 지표 | 현재 | 기준 | 근거 |
|---|---|---|---|
| ③ 인과민감도 | 0.006 | **> 0.05** | 노이즈(개입 변화 4/499)를 확실히 초과. 오늘의 >0.03은 지나치게 촘촘했음 |
| 생성 acc | 0.264 | **CI 하한 > FAA CI 상한** | 절대값 기준은 subset 흔들림 0.038에 묻힌다 |
| 롤아웃 분산 0 비율 | 57.3% | **< 35%** | 쌍 생성 가능량이 학습 신호의 상한 |
| judge 총점 | 12.43 | **기준 아님(진단용)** | 스프레드 0.64/14 |

### 중단 조건
- 게이트 A 실패 → ARM B 계열 폐기, 자원을 개선 2로
- H6 시점에 ③ < 0.02 → belief 조작 경로도 기각. 남는 후보는 **아키텍처 변경**(belief를 생성물이 아니라 action 디코딩의 조건부 입력으로 강제)이며, 이는 10시간 밖 범위

---

## 9. 병행 중인 Step-1 (참고)

retro와 무관하게 `EGO_jihun2`에서 GoalStep Z=1 V-JEPA2 action probe 학습이 진행 중이다.

- 설정 `configs/step1/goalstep/z1_jihun2.yaml`, 실행 `scripts/step1/goalstep/run_full_jihun2.sh` (tmux `ego_goalstep_jihun2`)
- 오늘 적용한 패치: 학습 루프만 **bf16 autocast**(평가·확률 산출은 fp32 유지), `val_subset_size` 500 → **2000**, `run_metadata.json`에 precision 기록
- 실측 근거: 실제 캐시 40배치 동일 시드 비교에서 **6.4× 가속**, 최종 loss 차 0.02%, top-1 일치 100%, entropy 차 ≤5e-4 nats
- 총 소요 7.9h → **약 1.4h**로 단축
- ⚠️ `val_subset_size` 변경으로 **cmr@5 절대값이 `z1.yaml`/`smoke.yaml`(500) 수치와 비교 불가**하다. 500샘플은 293개 action 클래스 중 150개(51%)만 덮고, 2000은 240개(82%)를 덮는다 — tail 클래스가 평균에 들어와 숫자가 낮게 읽힌다

**Step-1과 retro의 접점**: Step-1이 내보내는 likelihood/entropy는 sigmoid focal loss로 학습되는데 softmax로 산출된다. Step-3의 불확실성 트리거가 이 값을 소비하므로, retro의 ③ 인과 논의와 별개로 **캘리브레이션 점검이 필요하다**(temperature scaling 권장).

---

## 10. 검증된 사실 / 검증 안 된 것

**직접 원본 파일로 확인함**: 실행 인벤토리·타임스탬프, `B0_P12_RESULTS.md` 전 수치, `remeasure_b0p12.json`/`remeasure_abA.json` span 분해, `stats_p12.json` 롤아웃 분포, `pro_gr_train.py:187-193` 코드, P2 쌍 1152/1152 접두 동일, repro 3파일 바이트 동일.

**미확인 / 주의**:
- judge 표(§5)는 오늘 실행이 아니라 7/18–19 산출물을 오늘 문서화한 것 — 재실행으로 확인되지 않았다
- ③ 인과민감도의 통계적 불확실성이 보고된 적 없다. 4/499 vs 6/497은 CI가 겹칠 가능성이 매우 높다 → 개선 1에 포함시킬 것
- ARM B의 `--credit action` 효과와 `mean` 소멸 문제가 **교란돼 있다**. 개선 3의 sum 수정 없이는 "credit=action이 틀렸다"와 "구현이 신호를 죽였다"를 분리할 수 없다
