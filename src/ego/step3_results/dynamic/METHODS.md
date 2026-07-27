# Closed-Loop Dynamic Planning — 설계·계약·재현

작성 2026-07-26 · 코드 `src/ego/step3_results/dynamic/` · 산출물 `runs/dynamic_v1/`
논문 대응 절: `main.tex` §Embodied Planning "Closed-Loop Dynamic Planning" (L612–613), 남은 실험 4번

---

## 1. 무엇을 보이려는 실험인가

VPA(§Goal-Conditioned VPA)는 **한 시점의 관찰로 다음 T개를 통째로 뱉는 open-loop**다.
여기서 보이려는 것은 다르다 — 프레임워크가 **한 영상을 처음부터 끝까지 따라가며**
매 결정지점에서 (월드 모델이 후보 제시 → 언어 모델이 선택)을 반복해도 goal 을 향해
진행할 수 있는가. 즉 **닫힌 루프**다.

결정적 차이는 히스토리의 출처다.

| | history 입력 | belief |
|---|---|---|
| VPA / step2 eval | **ground truth** 완료 action | 없음 |
| **여기 (ego_closed)** | **모델이 앞서 스스로 고른 action** | 자기 이전 `task_belief` 누적 |

GT 는 채점에만 쓰고 프롬프트에 절대 넣지 않는다. 유일한 예외인 `oracle_gt_hist` arm 은
"자기 히스토리 오염의 비용"을 재기 위한 **의도적 대조군**이다.

## 2. 계약 (전 arm 공통 — 여기서만 분기)

`common.py` 하나에 모아 두었고, 시간·프레임 계약은 **VPA v2 모듈을 그대로 임포트**한다
(같은 프레임 캐시를 공유하고 표 간 비교가 성립해야 하므로).

| 항목 | 값 | 근거 |
|---|---|---|
| 관측창 | `[t−5s, t−1s]` (4초) | 원본 δ=4s + 1초 안전 간격 — 미래 누출 차단 |
| 프레임 | 8장 @2fps, 짧은 변 336px | step2 학습 입력과 토큰 형상 동일 |
| 후보 | **WM top-10 만** (셔플 상태 그대로) | EGO 배포 형태 — "WM 이 경계를 긋고 LM 이 고른다" |
| 출력 | `<reasoning>` / `<task_belief>` / `<action>` | step2 학습 포맷 그대로 (LoRA 분포 유지) |
| belief 캐리 | 직전 3개 | 프롬프트 길이와 최근성의 절충 |
| history 표시 | 최근 15개 | step2 `fmt_history` 와 동일 |
| 디코딩 | greedy (`do_sample=False`), max_new_tokens 320 | 재개 시 궤적 일관성 |

> ⚠ step2 **학습** 관측창은 8s@1fps 였다. 여기서는 사용자 지정이자 VPA v2 와 동일한
> 4s@2fps 를 쓴다 — 토큰 형상은 같고 시간 범위만 좁다. 표 간 비교에는 유리하나
> 학습 분포와 미세하게 다르다는 점은 결과 해석에 명시할 것.

## 3. 에피소드 구성 (`build_episodes.py`)

원천은 step2 검증셋 `runs/cesft_v2/data/context_val.jsonl` (결정지점마다 WM top-10).
다음 필터를 통과한 사슬만 남긴다.

| 필터 | 이유 | 실측 |
|---|---|---|
| `split == heldout` | dev 는 step2 학습 루프의 probe 대상 | −1,634 |
| 로컬 mp4 존재 | 프레임 추출 필요 | −370 |
| **level == step** | GoalStep 은 step 안에 substep 이 중첩 — 두 레벨을 섞으면 구간이 겹쳐 "선형 plan" 이 성립하지 않는다 | −1,534 |
| `target_start − obs_end ≤ 1.5s` | WM 후보가 **직전 관찰**에서 나온 것이어야 한다. gap 이 크면 수십 초 전 관찰의 낡은 후보 | −2,356 |
| `target_start ≥ 5.5s` | 4초 창이 영상 시작에 잘리지 않게 | |
| 에피소드 길이 ≥ 8, ≤ 40 | 너무 짧으면 동역학이 안 보이고, 긴 꼬리는 비용 통제 | |

**결과: 39 에피소드 / 868 스텝 / 21 goal category / 에피소드 평균 22.3 스텝.**
WM top-10 커버리지 **35.9%** — 나머지 64%의 스텝은 GT 가 후보에 없어 **구조적으로 정답이 불가능**하다.
따라서 SelAcc 는 전체와 covered 부분집합을 **둘 다** 보고한다.

## 4. arm 3종

| arm | history | belief | 무엇을 재는가 |
|---|---|---|---|
| `ego_closed` | 자기 예측 | ✅ | 본 실험 — 닫힌 루프 |
| `ego_nobelief` | 자기 예측 | ❌ | `ego_closed − 이것` = **belief 캐리의 기여분** |
| `oracle_gt_hist` | **GT** | ❌ | `이것 − ego_nobelief` = **자기 히스토리 오염의 비용** |

프레임·후보·디코딩은 세 arm 이 완전히 동일하다. 모델도 동일(step2 `sft_r15` LoRA).

## 5. 실행 구조 (`run_closed_loop.py`)

- **스텝 동기 배치**: 에피소드 내부는 순차(닫힌 루프라 필연), 에피소드끼리는 독립이므로
  같은 `step_idx` 를 여러 에피소드에서 모아 한 번에 생성한다. 실측 batch 16 → **9.9s/step**
  (batch 4 는 17.5s/step). 868스텝 ≈ arm당 2.4h.
- **재개**: `preds/<arm>.records.jsonl` 을 되읽어 각 에피소드 상태(선택 열·belief 열)를 복원.
  greedy 디코딩이라 같은 접두사에서 같은 출력이 나오므로 이어 붙여도 궤적이 일관된다.
- **매칭 실패 처리**: 3태그 파싱 실패나 후보 밖 출력은 difflib 최근접으로 **강제 투영**하고
  `forced=True` 로 남긴다. **WM top-1 로 폴백하지 않는다** — 그러면 baseline 행동이
  ours 지표에 섞인다. (스모크 16스텝 실측: malformed 0, forced 0)
- **프레임 결손 에피소드는 통째로 제외**한다. 중간 스텝을 건너뛰고 이어 붙이면
  "연속된 관찰" 전제가 조용히 깨지기 때문. (현재 868/868 프레임 확보 → 제외 0)

## 6. 지표 (`evaluate.py`)

SR/mAcc/mIoU 는 "한 번에 T개"용이라 쓰지 않는다. 닫힌 루프에서 볼 것:

| 지표 | 정의 |
|---|---|
| `SelAcc` / `SelAcc_covered` | 스텝 정확도 (전체 / GT∈top10 부분집합) |
| `WM_top1` | 같은 스텝에서 월드 모델 단독 — 전 arm 공통 바닥선 |
| `G1` / `GADR` | WM 이 맞은 지점의 유지율 / 틀린 지점의 교정률 |
| `progress_curve` | 에피소드 진행 5분위별 정확도 — **논문 신호 (1)** |
| `recovery_steps` | 오답 이후 다시 맞히기까지의 스텝 수 — **논문 신호 (2)** |
| `hist_purity` | 그 시점 프롬프트에 실린 자기 히스토리 중 GT 와 일치한 비율(오염도) |

CI 는 **영상 클러스터 부트스트랩**(영상당 10~40 스텝이 상관), arm 비교는 **paired**
(같은 스텝에서 두 arm 을 나란히 재표집해 차이의 분포를 직접 구한다).

## 7. 정성 평가 (`build_review_site.py` → `merge_ratings.py`)

에피소드 하나를 위에서 아래로 훑으며 스텝 카드(입력 관측 8프레임 스트립 · 모델이 고른 행동 ·
그 시점 belief · reasoning · WM 후보 10)를 보고 3인이 독립 판정한다.

- 스텝별 **타당 / 애매 / 부적절**, 에피소드별 **1~5점**(goal 수행 여부) + 메모.
- **GT 는 기본으로 가려 둔다.** 정답을 먼저 보면 판단이 정답에 정박된다. 토글로 열 수 있고
  연 사실을 `gt_revealed` 로 기록해 사후 분리 집계가 가능하다. 토글을 열면 GT action과 함께
  실제 영상의 GT onset 이후 `[t,t+2s)`를 2fps로 뽑은 4프레임(`t,+0.5,+1.0,+1.5s`)이
  표시된다. 이 이미지는 사이트 빌드 시 원본 mp4에서 미리 추출되며 추론·정량 평가는 다시
  실행하지 않는다.
- 관측/GT 스트립의 프레임을 클릭하면 336px 높이의 확대 뷰어가 열리며, 화면의 좌우 화살표
  또는 키보드 `←`/`→`로 같은 스트립의 바로 이전·다음 프레임을 탐색한다. GT 프레임은
  기존처럼 정답 토글을 연 뒤에만 UI에서 접근한다.
- 저장은 localStorage 자동, 제출은 `ratings_<이름>.json` 다운로드.
- 집계는 다수결 + **Fleiss κ**(일치도) + **다수결 × 정량 정오 교차표**.
  교차표의 "타당 × GT불일치" 칸이 핵심 — GT 라벨과 다르지만 goal 을 향해 합리적인 경로.

## 8. 재현

```bash
PYTHONPATH=src python -m ego.step3_results.dynamic.build_episodes --out-dir runs/dynamic_v1
PYTHONPATH=src python -m ego.step3_results.dynamic.extract_frames          # vpa_v2 캐시 공유
bash src/ego/step3_results/dynamic/run_all.sh                              # 3 arm + 채점 (≈7h)
PYTHONPATH=src python -m ego.step3_results.dynamic.build_review_site --arm ego_closed \
  --video-root data/Ego4D/v2/goalstep_videos
python -m http.server 8899 --directory runs/dynamic_v1/site
PYTHONPATH=src python -m ego.step3_results.dynamic.merge_ratings runs/dynamic_v1/ratings/*.json
```
