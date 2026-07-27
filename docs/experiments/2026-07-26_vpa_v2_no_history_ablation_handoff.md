# [계획] VPA v2 — action history 제거 ablation Handoff

> **2026-07-27 업데이트:** 이 계획의 T3/T4 실행은 완료됐다. 네 조건의 최종 수치와 해석은
> [`2026-07-27_vpa_v2_action_history_ablation_final_report.md`](./2026-07-27_vpa_v2_action_history_ablation_final_report.md)를
> 기준으로 한다. 아래 내용은 실행 전 가설·설계 기록으로 보존한다.

> 작성 2026-07-26 KST · **상태: 미실행. 다음 세션/장비에서 수행할 것.**
> 배경 문서: `2026-07-26_vpa_v2_results_handoff.md` (현행 결과) ·
> `src/ego/step3_results/vpa/v2/{METHODS,PROMPTS}.md` (프롬프트·코드 구조)

---

## 1. 가설

> **action history(완료 행동 텍스트)를 입력에서 제거하면, WM 기반으로 학습된 우리 모델이
> frontier·백본 대비 상대적으로 덜 나빠질 것이다.**

현행 전 arm은 완료 행동 최근 15개를 **텍스트로** 받는다. 이 텍스트가 매우 강한 신호라는 정황이 있다:

- `most_probable_goal` baseline이 **영상을 전혀 안 보고** history 빈도만으로 mAcc **13.48**을 낸다.
- `qwen_blind`(프레임 없이 history+goal만)가 mAcc **9.18**. 즉 텍스트만으로도 상당 부분이 설명된다.

→ 현재 표의 성능 중 얼마가 **시각 이해**이고 얼마가 **텍스트 히스토리 활용**인지 분리되지 않았다.
history를 빼면 남는 것은 "goal + 4초 영상"뿐이므로, **시각 기반 예측 능력만** 남는다.

**왜 우리 모델이 유리할 것으로 보는가**: step2는 (프레임 + WM 후보 + history)로 학습됐지만,
그 학습 신호의 핵심은 **WM 후보 중 어느 것이 현재 시각 상태와 맞는지 판별하는 것**이었다
(candidate-CE의 GADR 지표가 정확히 이 능력을 잰다). 반면 frontier·백본은 도메인 학습이 없어
텍스트 히스토리라는 일반적 단서에 더 의존할 가능성이 크다.

### 1-1. 반대 가능성 (반드시 함께 검토)

- **우리 모델에게도 OOD다.** step2 학습 프롬프트는 `"Completed actions so far (oldest to newest):"`
  로 시작한다(`vlm.py:user_prompt`). history를 비우면 우리 모델도 학습 분포 밖 입력을 받는다.
  → 상대적 열화가 작을지 클지는 **선험적으로 알 수 없다.** 가설이지 결론이 아니다.
- history 제거는 **태스크 난이도를 크게 올린다.** 전 arm 성능이 바닥에 붙어 바닥효과(floor effect)로
  arm 간 차이가 안 보일 수 있다. mAcc가 5% 아래로 붕괴하면 비교 자체가 무의미해진다.
- `ours_wm1st`은 WM 후보를 받으므로 history 없이도 첫 스텝이 보호된다 — **arm마다 열화 폭이
  다른 이유가 "시각 능력"이 아니라 "받은 힌트의 양"일 수 있다.** 해석 시 분리할 것.

---

## 2. 실험 설계

### 2-1. 측정 방식 — **기존 실험과 동일하게 VPA 3지표가 1순위**

**주 산출물은 지금까지와 똑같은 형식의 결과표**다. history 없는 조건에서 각 arm의
**SR · mAcc · mIoU**를 video-cluster 부트스트랩 CI와 함께 낸다. 도구도 그대로
(`evaluate.py` → `metrics_T3_*.json`).

```
arm                    n     SR                mAcc              mIoU
ours_full_nohist      915   x.xx [..,..]      xx.xx [..,..]     xx.xx [..,..]
qwen_backbone_nohist  915   ...
frontier_nohist       915   ...
```

그 다음 **arm 간 짝비교**도 기존과 동일하게 `paired.py`로 낸다 —
`ours_full_nohist − qwen_backbone_nohist`, `ours_full_nohist − frontier_nohist`.
이것이 "history 없이 붙었을 때 누가 더 나은가"라는 **직접적인 질문에 대한 답**이며,
현행 표(history 있음)와 나란히 놓으면 그 자체로 읽힌다.

| 비교 | 기존(history 있음) | 신규(history 없음) |
|---|---|---|
| ours_full − qwen_backbone | mAcc **+3.13** [1.33, 5.11] ✅ | ← 이 값이 얼마나 커지는가 |
| ours_full − frontier | mAcc +6.93 [−3.74, +21.77] ✗ | ← 유의해지는가 |

### 2-1-1. (부가) DiD — 열화 폭 자체를 비교하고 싶을 때만

위 표만으로 "우리 모델이 history에 **덜 의존한다**"를 정식 주장하려면 열화 폭을 비교해야 한다.
**부가 분석**으로만 수행한다.

```
Δ_arm = mAcc(history 있음) − mAcc(history 없음)
검정  : (Δ_ours − Δ_frontier) 를 video-cluster paired bootstrap 으로 재표집
```

> arm별 Δ의 CI가 겹치는지 눈으로 보는 것으로 판정하지 말 것
> (difference of significance ≠ significance of difference).
> 필요해지면 `paired.py`를 확장한다. **1순위 아님 — 위 §2-1 결과표가 먼저다.**

### 2-2. arm 목록 (history 없는 조건)

| arm | 기존 조건 대비 | 목적 |
|---|---|---|
| `ours_full_nohist` | ours_full − history | 주 비교 대상 |
| `ours_wm1st_nohist` | ours_wm1st − history | WM 힌트가 history 부재를 보상하는가 |
| `qwen_backbone_nohist` | backbone − history | 무학습 대조 |
| `frontier_nohist` | frontier − history | 외부 기준점 |
| `qwen_blind_nohist` | blind − history | **goal만 남음** — 사실상 하한선 |
| `most_probable_goal_nohist` | (baseline 재계산) | history 없이도 계산 가능한지 확인 필요 |

**최소 구성**: `ours_full_nohist` + `qwen_backbone_nohist` + `frontier_nohist` 3개면 가설 검정 가능.
GPU 약 4h + API 915콜.

### 2-3. 표본

현행과 **완전히 동일**해야 기존 표와 나란히 읽힌다 — `runs/vpa_v2/frames_subset_T3.json` (915샘플 / 71영상),
동일 프레임 캐시(`frame_cache_w4_g1_n8_s336`), 동일 어휘 293.

---

## 3. 구현 (예상 변경 최소)

### 3-1. `common.py` — `history` 인자 추가

```python
def build_prompt(sample, vocab, horizon, *, with_frames, candidate_mode="vocab",
                 history="full"):        # ← 신규: "full" | "none"
    ...
    if history == "none":
        obs_txt = "  (not provided)"
        # SYSTEM 에서도 history 언급 구절을 제거해야 한다 —
        # "and the actions the camera-wearer has already COMPLETED" 를 빼고 나머지는 그대로.
    else:
        obs_txt = "\n".join(...)  # 현행
```

**주의**: blind arm 설계와 같은 원칙 — **history 관련 문장만** 바꾸고 나머지 텍스트는 한 글자도
건드리지 말 것. 그래야 Δ가 history 기여분만 의미한다.

### 3-2. 러너에 플래그 노출

- `run_local_vlm.py`: `--history {full,none}` 추가 → `build_prompt(history=args.history)`
- `run_frontier.py`: 동일

### 3-3. 검증 (실행 전 필수)

1. `history="full"` 프롬프트가 **바이트 단위로 현행과 동일**한지 확인
   (`ours_full`/`backbone` 결과를 재사용하려면 필수 — 2026-07-26에 같은 검증을 한 전례 있음).
2. `history="none"` 프롬프트에 완료 행동 문자열이 **하나도 남아 있지 않은지** grep 확인.
3. 3샘플 스모크로 출력이 여전히 JSON 배열로 파싱되는지 확인.

---

## 4. 실행 순서

```bash
export PYTHONPATH=src; P=~/ml_env/bin/python

# (0) 검증 — §3-3
# (1) 로컬 arm (GPU 순차, 각 ~2h)
$P -m ego.step3_results.vpa.v2.run_local_vlm --gt runs/vpa_v2/vpa_v2_T3.json \
   --mode frames --history none --adapter outputs/step2_retrospection/cesft_v2/sft_r15/adapter \
   --candidates vocab --out runs/vpa_v2/preds/ours_full_nohist_T3
$P -m ego.step3_results.vpa.v2.run_local_vlm --gt runs/vpa_v2/vpa_v2_T3.json \
   --mode frames --history none --out runs/vpa_v2/preds/qwen_backbone_nohist_T3

# (2) frontier (API, 병렬 가능 — GPU 미사용)
FRONTIER_API_KEY=... $P -m ego.step3_results.vpa.v2.run_frontier \
   --gt runs/vpa_v2/vpa_v2_T3.json --subset runs/vpa_v2/frames_subset_T3.json \
   --history none --out runs/vpa_v2/preds/frontier_nohist_T3 --max-calls 915

# (3) 채점 + arm 내 paired (history 유무)
for a in ours_full qwen_backbone frontier; do
  $P -m ego.step3_results.vpa.v2.evaluate --gt runs/vpa_v2/vpa_v2_T3.json \
     --pred runs/vpa_v2/preds/${a}_nohist_T3.json --run-name ${a}_nohist \
     --subset runs/vpa_v2/frames_subset_T3.json
done
# (4) arm 간 짝비교 — 기존과 동일
for b in qwen_backbone_nohist frontier_nohist; do
  $P -m ego.step3_results.vpa.v2.paired --gt runs/vpa_v2/vpa_v2_T3.json \
     --subset runs/vpa_v2/frames_subset_T3.json \
     --a runs/vpa_v2/preds/ours_full_nohist_T3.json --a-name ours_full_nohist \
     --b runs/vpa_v2/preds/${b}_T3.json --b-name $b
done
# (5) [부가] DiD — §2-1-1. paired.py 확장 필요. 1순위 아님.
```

**비용**: GPU 약 4h(로컬 2 arm) + frontier API 915콜. frontier는 GPU를 안 쓰므로 병렬 실행 가능.

---

## 5. 결과 해석 가이드 (미리 정해 둘 것)

| 관측 | 해석 |
|---|---|
| **(1순위)** history 없는 조건에서 `ours_full − backbone` 격차가 현행(+3.13pp)보다 **커짐** | **가설 지지** — 텍스트 단서가 사라질수록 우리 학습의 이점이 드러난다 |
| 격차가 비슷하거나 줄어듦 | 우리 학습의 이점은 history 유무와 무관 — 현행 주장 범위 유지 |
| `ours_full − frontier` 가 **유의해짐** | 순수 시각 조건에서 frontier 대비 우위 주장 가능 (현행은 비유의) |
| (부가) Δ_ours < Δ_frontier 이고 DiD CI가 0 미포함 | "우리 모델이 history에 덜 의존한다"를 정식 주장 가능 |
| (부가) Δ 들이 비슷 | 세 모델의 history 의존도가 유사 |
| 전 arm mAcc < 5% | **바닥효과** — 비교 불가. 이 경우 T=1(다음 1개만)로 낮춰 재시도 검토 |

**보고 시 주의**: `ours_wm1st_nohist`가 좋게 나와도 그것은 "시각 능력"이 아니라 **WM 후보라는 추가
정보** 덕일 수 있다. 시각 능력 주장은 `ours_full`(WM 힌트 없음) 기준으로만 할 것.

---

## 6. 이 실험이 논문에서 갖는 위치

현행 결과는 "프레임이 기여한다"(frames − blind, mAcc +8.31pp)까지 보였다. 그러나 그 비교는
**history가 양쪽에 다 있는 상태**에서의 시각 기여분이다.

history를 빼면 "**텍스트 단서 없이 순수 영상만으로 얼마나 계획하는가**"를 재게 되고,
여기서 우리 모델이 상대적으로 강하다면 **WM 기반 학습이 시각 접지(visual grounding)를
강화했다**는 더 강한 주장이 가능해진다. 반대로 차이가 없다면 현행 주장 범위를 유지하면 된다.

어느 쪽이든 **현행 결과를 약화시키지 않는다** — 추가 정보만 준다. 우선순위는 중간.

---

## 7. 선행 조건

- [ ] 현행 실행 완주 (`ours_wm1st` 915 · frontier 915) — `runs/vpa_v2/finish_all.sh`
- [ ] (부가 분석을 할 경우에만) `paired.py`에 DiD 지원 추가
- [ ] `--history` 플래그 구현 + §3-3 검증 3종
- [ ] 영상·프레임 캐시는 이미 로컬에 있음 (재다운로드 불필요, 262MB 캐시 재사용)
