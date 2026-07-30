# Closed-Loop Dynamic Planning — 구현 현황 & 다음 세션 handoff

작성 2026-07-26 22:40 · 코드 `src/ego/step3_results/dynamic/` · 산출물 `runs/dynamic_v1/`
설계 상세는 `src/ego/step3_results/dynamic/METHODS.md` (이 문서는 **상태와 인수인계**만 다룬다)

---

## 0. 한 줄 요약

논문 §Embodied Planning 의 마지막 항목(main.tex L612–613, 남은 실험 4번)인 **closed-loop
dynamic planning** 파이프라인을 새로 구축했다. 39 에피소드 / 868 스텝 / 3 arm 실행이
tmux 세션 `dynplan` 에서 **진행 중**이며(2026-07-26 22:25 시작, 약 7시간 예상),
정성 평가 사이트와 집계 도구까지 코드는 모두 준비되어 있다.

## 1. 지금 어떤 상태인가

| 단계 | 상태 |
|---|---|
| 에피소드 빌드 | ✅ `runs/dynamic_v1/episodes.json` — 39 ep / 868 step / 21 goal cat / cov 35.9% |
| 프레임 추출 | ✅ 868/868 (`runs/vpa_v2/frame_cache_w4_g1_n8_s336` 공유, 679장 신규 추출) |
| 스모크 | ✅ 16스텝 — 3태그 파싱 100%, forced 0, batch16 **9.9s/step** |
| **본 실행 (3 arm)** | 🔄 **진행 중** — tmux `dynplan`, `bash runs/dynamic_v1/run_all.sh` |
| 채점 | ⏳ run_all.sh 끝에서 자동 실행 |
| 정성 사이트 | ✅ 코드 완료, 실행은 예측 산출 후 |
| 3인 평가 집계 | ✅ 코드 완료 (`merge_ratings.py`) |
| REPORT.md | ⏳ 수치 나온 뒤 작성 |

### 진행 확인 방법

```bash
tmux attach -t dynplan                      # 세션 (분리: Ctrl-b d)
S=/tmp/claude-1002/-home-hogun/754f0145-ce4d-4e63-9ea1-a05977cb6dac/scratchpad
grep -a ETA $S/run_ego_closed.log | tail -3       # arm별 진행/ETA
wc -l runs/dynamic_v1/preds/*.records.jsonl       # 기록된 스텝 수 (arm당 868 이 목표)
tail -40 $S/evaluate.log                          # 채점 결과
```

**중단됐다면** 같은 명령을 다시 실행하면 된다 — `preds/<arm>.records.jsonl` 을 되읽어
에피소드별 (선택 열 + belief 열) 상태를 복원하고 남은 스텝부터 이어간다(greedy 디코딩이라
같은 접두사면 같은 출력 → 궤적 일관).

```bash
cd ~/Project/EGO && bash src/ego/step3_results/dynamic/run_all.sh
```

## 2. 디렉토리 지도

```
src/ego/step3_results/dynamic/          # 신규 패키지 (커밋 대상)
├── METHODS.md              설계·계약·지표 정의 — 먼저 읽을 것
├── common.py               ★ 계약 단일 출처: 창/프레임(vpa.v2.common 임포트), arm 정의,
│                             프롬프트 생성, 3태그 파싱, 후보 강제 투영, WM top-1 복원
├── build_episodes.py       context_val.jsonl → 영상별 연속 결정지점 사슬 (필터 6종)
├── extract_frames.py       vpa_v2 프레임 캐시 공유 추출 (계약 동일 → 재사용)
├── run_closed_loop.py      ★ 루프 러너 — 스텝 동기 배치 · 상태 복원 · 기록
├── evaluate.py             SelAcc/covered/G1/GADR/진행곡선/회복/오염도 + 클러스터 CI + paired
├── build_review_site.py    정성 평가 정적 사이트 (스트립 이미지 + 평가 UI) 생성
├── merge_ratings.py        3인 평가 병합 — 다수결 · Fleiss κ · 정량 교차표
└── run_all.sh              3 arm 순차 + 채점 (runs/dynamic_v1/ 에도 사본)

runs/dynamic_v1/                        # gitignore 대상 (산출물)
├── episodes.json           에피소드 정의 + contract + drop 통계
├── frames_manifest.json
├── preds/<arm>.records.jsonl           스텝마다 1행: 선택·belief·reasoning·GT·WM·프롬프트 입력
├── preds/<arm>.trajectories.json       에피소드별 선택 열만 뽑은 요약
├── metrics/metrics.json                채점 결과 (evaluate.py)
├── metrics/qualitative.json            3인 평가 집계 (merge_ratings.py)
├── ratings/                            평가자들이 내보낸 ratings_*.json 을 여기 모을 것
└── site/                               index.html + ep_*.html + strips/
```

## 3. 설계에서 반드시 지켜야 하는 것 (깨면 실험이 무의미해짐)

1. **GT 는 프롬프트에 넣지 않는다.** history 는 모델이 앞서 고른 action, belief 는 모델
   자신의 이전 `task_belief`. 유일한 예외 `oracle_gt_hist` 는 의도적 대조군이다.
2. **후보는 WM top-10 뿐이고 반드시 그 안에서 고른다.** 매칭 실패 시 difflib 최근접으로
   강제 투영하고 `forced=True` 로 기록 — **WM top-1 폴백 금지**(baseline 행동이 섞인다).
3. **관측은 target 시작 1초 전에 끝난다** (`[t−5s, t−1s]`). 프레임 캐시 디렉토리 이름에
   계약이 박혀 있어 계약을 바꾸면 자동으로 다른 캐시가 된다.
4. **level=step 만 사용.** GoalStep 은 step 안에 substep 이 중첩돼 두 레벨을 섞으면
   구간이 겹쳐 선형 plan 이 성립하지 않는다.
5. **gap ≤ 1.5s.** WM 후보는 그 시점 직전 관찰에서 나온 것이어야 한다.
6. **에피소드 중간을 건너뛰지 않는다.** 프레임 결손 에피소드는 통째로 제외.

## 4. 남은 작업 (다음 세션이 이어받을 순서)

1. **본 실행 완료 확인** — arm 3종 × 868 스텝. 실패 배치가 있으면 `error` 필드가 채워진
   행을 확인하고 재실행(재개 안전).
2. **채점 결과 읽기** — `runs/dynamic_v1/metrics/metrics.json`.
   해석 시 주의: 커버리지 35.9% 라 SelAcc 절대값은 낮게 나온다. 비교의 축은
   `WM_top1`(동일 스텝 바닥선)과 arm 간 paired Δ 이다.
3. **정성 사이트 생성 + 서빙**
   ```bash
   PYTHONPATH=src python -m ego.step3_results.dynamic.build_review_site --arm ego_closed
   python -m http.server 8899 --directory runs/dynamic_v1/site   # tmux 창 하나에 띄워둘 것
   ```
   3인에게 URL 을 주고 각자 이름 입력 → 판정 → **결과 내보내기** →
   받은 파일을 `runs/dynamic_v1/ratings/` 에 모은 뒤 `merge_ratings.py` 실행.
   ※ 868 스텝 전량을 3인이 보는 것은 과하다. 에피소드 단위로 나눠 배정하거나
   대표 10~15 에피소드만 평가하고 그 사실을 리포트에 명시할 것.
4. **REPORT.md 작성** — `runs/dynamic_v1/REPORT.md` 에 실측 수치·해석·한계.
5. **논문 반영** — main.tex L612–613 의 "이 평가는 아직 실행 전" 문장을 실측으로 교체.
   신호 (1) 진행 곡선, (2) 회복 스텝은 evaluate.py 가 직접 낸다. 신호 (3) **τ 스윕은
   미구현** — 필요하면 "K 스텝마다만 WM 후보를 갱신하고 그 사이는 자기 계획을 따른다"는
   변형을 `run_closed_loop.py` 에 추가해야 한다(현재 τ=1, 즉 매 스텝 재접지).

## 5. 알려진 한계 / 리포트에 반드시 적을 것

- **커버리지 35.9%** — GT 가 WM top-10 밖인 스텝은 구조적으로 정답 불가. 전체 SelAcc 를
  "GoalStep 계획 성능"으로 일반화하지 말 것. arm 간 비교(동일 표본, paired)는 유효.
- **학습 창과의 불일치** — step2 학습은 8s@1fps, 여기는 4s@2fps(사용자 지정 + VPA v2 정합).
  토큰 형상은 같으나 시간 범위가 좁다.
- **에피소드 = 결정지점 사슬**이지 영상의 모든 action 이 아니다. 필터로 빠진 지점이 있어
  모델의 history 에는 그 사이 실제 행동이 빠져 있다. `episodes.json.stats.drops` 참조.
- **평가 대상 arm 은 정성 사이트에서 ego_closed 하나**다. 대조군까지 사람이 보게 하면
  판정이 서로 오염된다(원하면 arm 별로 별도 사이트를 만들고 평가자를 분리할 것).
- 정성 판정의 **GT 노출 여부**가 기록되므로(`gt_revealed`), 노출 전/후 판정을 분리해
  보고할 수 있다.

## 6. 관련 문서

- `src/ego/step3_results/dynamic/METHODS.md` — 계약·지표 정의
- `src/ego/step3_results/vpa/v2/METHODS.md` — open-loop VPA (시간 계약의 원출처)
- `develop_report/2026-07-25_step2_cesft_sft_r15_model_usage_handoff.md` — 정책 모델 사용법
- `develop_report/2026-07-25_vpa_v2_frame_conditioned_plan.md` — 창 설계 근거
- `develop_report/paper/EGO_AAAI27_overleaf/main.tex` L612–613, L623
