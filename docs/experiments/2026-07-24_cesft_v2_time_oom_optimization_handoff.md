# cesft_v2 시간 단축 · OOM 방지 최적화 Handoff (목표·실행 불변)

> 작성: 2026-07-24 KST · EGO_jihun 세션(교차 검토). **대상: EGO_jihun3 cesft_v2 무인 체인.**
> **대원칙: 실험 목표(통합 유의성 = G-NH ∧ G-CC1/CC3)와 현재 실행은 그대로 유지한다.
> 지금 돌고 있는 theta_ce는 어떤 이유로도 중단·재시작하지 않는다** (스테이지 멱등이라 중단 = 처음부터).
> 관련: [[2026-07-24_evaluation_metrics_handoff]] · [[2026-07-24_interventional_belief_sensitivity_metric_handoff]] ·
> EGO_jihun `docs/experiments/2026-07-24_history_strip_ablation_results_handoff.md`(strip 3-PASS) ·
> `2026-07-24_reasoning_quality_quantitative_evidence_handoff.md`(성립부등식·DiD 확정치)

---

## 0. 결론 요약

| 항목 | 판정 |
|---|---|
| EGO_jihun 기존 CE 체크포인트(goalstep_v2_c)로 θ_CE 대체 | **불가 — 5중 불일치 (§1). theta_ce 완주가 맞음** |
| B_candfree(1.7h)+B_nohist(3.6h) | **EGO_jihun 확정 결과 + strip-eval(0.5h)로 대체** — −5.3h |
| C_r30(2.5h) | **r15 G-NH PASS 시 스킵** (조건부 게이트) |
| C_wise α 3점(1.7h) | **α=0.5 한 점으로 축소** — −1.1h |
| base 앵커 재측정 | **스킵** — E0 실측(SelAcc .200 · WM-top1 .242 · harden belief .058) 재인용 |
| OOM | 동시-arm(주범)은 **근본 해결됨**(MAX_PARALLEL=1). 디코드 진동·오류는 §3 가드로 **완화** — 근본 해결은 프레임 사전 추출 캐시(§3-1, 경보 시 응급/다음 사이클 표준) |
| 프레임 수 | **8프레임 유지** — retro4 계약의 구성 요소, 중도 변경 = 학습·평가 불일치 (§1-1) |
| 총 예산 | **25.3h → 약 14h** (부록A 게이트 통과 시 +3h) |

---

## 1. 왜 EGO_jihun CE 체크포인트를 θ_CE 자리에 못 쓰나 (5중 불일치)

같은 베이스 모델(Qwen3-VL-8B + LoRA)이라 **기술적으로 로드는 되지만**, 다음이 전부 어긋난다:

| 축 | EGO_jihun C (goalstep_v2_c) | cesft_v2 θ_CE (retro4) | 귀결 |
|---|---|---|---|
| **① 과제·시간 계약** | end−1s 관측 → **current action** GT (인식 성향) | end−1s 관측 → **strict-next A3**, horizon 가변 **평균 12.8s** | 배우는 과제 자체가 다름 — 최치명 |
| **② WM 공급자** | V-JEPA2 probe 계열 (cov@10 49.2%) | jihun2 Phase-1 HistoryContextResidualHead (Top-10 43.9%) | 후보 분포·"wm_cand" 의미 불일치 |
| **③ 관측 형식** | **1프레임** | **8프레임**@336 (관측창 8s) | vision 입력 OOD |
| **④ 프롬프트·채점** | JOINT 스키마 + sum-logp | cesft SYSTEM + **length-norm** logp, τ | CE 판별은 표면형에 민감 (실측: Q가 형식 바뀌자 presented 25.8→free 붕괴 사례) |
| **⑤ 스플릿** | v2_train 2,500 (video 스플릿 상이) | covered train ∩ gt_rank≤10 (video-disjoint 통제) | **eval 오염(leakage) 미검증** — 재사용하려면 video_uid 교집합 0 확인 필수 |

**논리적 문제가 하나 더**: G-NH는 "SFT가 **θ_CE 대비** CE 능력을 잃지 않는가"의 게이트다. θ_CE 자리에
미스매치 체크포인트를 넣으면 기준선 자체가 (불일치로) 낮아져 **게이트가 저절로 쉬워지고 통합 주장이 무효**가 된다.

**실용 판단**: theta_ce 잔여 ~3h vs 재사용으로 아끼는 최대 4h — 그 4h를 아끼려다 불일치가 본 실험 25h 전체를
무효화할 수 있다. **완주가 합리적.** (선택 부록감: 완주 후 EGO_jihun C를 cesft eval에 zero-shot으로 얹어
"다른 WM·다른 계약에서 배운 판별의 이전 가능성"을 0.5h로 재는 cross-pipeline robustness — 지금은 하지 않음.)

### 1-1. 프레임 수 판정 — 1 vs 8, 무엇이 연구 의의·CE+SFT·실측에 부합하나

**결론: cesft_v2는 8프레임 유지.** 프레임 수는 튜닝 노브가 아니라 **retro4 시간 계약의 구성 요소**
("관측 = 행동 종료 1s 전까지의 최근 ≤8s 비디오")이고, theta_ce가 이미 8프레임으로 학습 중이므로
**중도 변경 = 학습·평가 불일치**로 본 실험을 훼손한다. 근거 4개:

1. **정직한 anticipation 계약의 실현**: "다음 행동 시작 전까지 관측한 비디오"가 과제 정의다. 1프레임은
   관측을 인위로 축소한 것 — 파일럿(EGO_jihun)에서는 비용 때문에 정당했지만, 본 실험·논문 수치는
   계약 그대로의 관측이 맞다.
2. **WM과의 정보 대칭**: WM(비디오 기반 prior)과 LM(8프레임)이 같은 관측창을 본다 → "LM 선택 우위/열위가
   시각 정보 격차 때문"이라는 양방향 반론을 차단. 1프레임이면 LM이 불리한 조건이라 승리 주장은 강해지지만
   "왜 관측을 안 보여줬나"는 설계 반론이 생긴다. 대칭이 학술적으로 깔끔하다.
3. **실측 정합 — 8프레임은 history 서사를 위협하지 않는다**: 1프레임 파일럿에서 이미 history 인과
   +12.6pp·이중 해리 확정 = **시각 최소 조건에서 성립**. 8프레임 관측창은 8초(직전 1~2행동)만 커버하고
   history는 최대 8행동·수 분을 커버하므로 **관측창이 history를 대체하지 못한다**는 것이 예측이고,
   8프레임에서 strip Δ>0가 재현되면 "history 효과는 관측량과 무관"이라는 **강건성 근거가 추가**된다.
   (만약 8프레임에서 history 효과가 소멸하면 그 자체가 중요한 발견 — 어느 쪽이든 정보 이득.)
   보조 실측: WM-rescue 국소화에서 운동동사 집중 가설이 **기각**되고 상태동사에서 더 컸음(+3.1 vs +0.7)
   — "프레임을 늘리면 LM이 WM의 시간 지각 니치를 잠식한다"는 우려가 실측상 근거 약함.
4. **두 파이프라인 = 자연 프레임-대조**: 1프레임(EGO_jihun 파일럿)과 8프레임(cesft_v2 본 실험) 결과를 병기하면
   추가 학습 없이 frame-robustness 서사가 생긴다. 단 계약·WM·데이터가 함께 다르므로 엄밀 ablation이 아니라
   **시사 수준**으로 각주할 것.

**비용 각주**: 8프레임이 속도·RAM의 지배 요인이므로, 다음 사이클에서 축소가 필요하면 **4프레임**(관측창 8s를
0.5fps로 커버, 짝수 유지로 temporal_patch_size=2 정합)이 파레토 후보다. SFT·모든 eval은 CE와 동일 프레임 수를
쓴다(학습·평가 일치 원칙) — 현 체인에서는 절대 변경 금지.

## 2. 이미 학습·측정한 것으로 대체 가능한 스테이지 (시간 단축의 본체)

목표 기준(통합 유의성)으로 스테이지를 재판정한 결과:

| 스테이지 | 예산 | 처분 | 대체 근거 (전부 실측 확정) |
|---|---:|---|---|
| A_theta_ce + 배터리 | 4.5h | **유지 (진행 중)** | 통합의 재료 + G-ACC1 |
| A_sft_r15 + 배터리 + harden | 3.7h | **유지 — 본 실험** | G-NH ∧ G-CC1/CC3 판정 그 자체 |
| **B_candfree + 배터리** | 2.2h | **연기** | 성립부등식(WM-cand>cand-free)은 EGO_jihun에서 확정: full +2.4pp(C제시 23.2 vs Q자유 20.8) · covered 선택 +19.2pp · 학습효과 DiD +8.4pp[8.0,8.8]. 같은-파이프라인 수치가 논문에 꼭 필요해지면 그때 샘플 캡 2,500으로 |
| **B_nohist (no-history CE 학습)** | 3.6h | **strip-eval 0.5h로 대체** | 추론-시 history-strip이 같은 질문(G-ACC2)을 **paired 인과로 더 강하게** 답함 (EGO_jihun 3목표 PASS: 인과 +12.6pp[10.5,14.7]·용량-반응·무-hist WM>LM +4.2pp). 학습-arm 비교는 "다른 체크포인트" 교란이 남지만 strip은 같은 체크포인트에서 history만 조작 |
| C_r0 | 2.5h | **유지** | "잃지 않음"의 비자명성 대조 (r0 G-NH FAIL이어야 r15 의의 성립) — 재사용 불가(같은 θ_CE에서 시작해야 함) |
| **C_r30** | 2.5h | **게이트: r15 G-NH FAIL 시에만** | r15 PASS면 fallback 불필요 |
| **C_wise** | 1.7h | **α=0.5 한 점** | 학습 0·평가만이라 점 수가 곧 시간. frontier 곡선은 논문 필요 시 추가 |
| C_cstack/cctrl/eval | 3.0h | 유지 (P-UTIL 게이트 기존대로) | |
| **base 배터리/harden 재측정** | ~0.7h | **스킵** | E0 앵커 실측 존재: SelAcc 0.200 · WM-top1 0.242 · GADR 0.164 · cov@10 0.434 · harden belief 0.058[.043,.075] |
| harden IV_N | — | **headline(r15)만 800, 나머지 400** | retro3 n≈990에서 CI[.358,.421]로 충분히 좁았음 |

**θ_CE strip-eval 구현 노트**: 학습 스크립트의 `ARMS["no_history"]` 프리셋(use_history=False)이 이미 있으므로,
배터리 eval에 `--no_history` 플래그 하나 추가해 **같은 θ_CE 어댑터로 history만 제거한 재추론**을 돌리면 된다
(EGO_jihun `eval_candidate_scored.py --no_memory` 패턴과 동일 — `T.NO_MEMORY=True` 방식 참조).
paired 분석은 EGO_jihun `scripts/step2/v3_hstrip_analysis.py`를 sample_id 조인으로 이식.

## 3. OOM 방지 (현 실행은 감시만, 다음 스테이지부터 적용)

**원인 정리**: ① 240G peak의 주범이었던 **동시-arm**(theta_ce+cand_free 208G→SIGTERM)은 MAX_PARALLEL=1로
이미 해소. ② 잔여 리스크는 **decord 디코드 진동**: 단일 arm에서도 python RSS 92↔119G(±25G), cgroup cur
187~214G/240G — 원인은 prefetch 더블버퍼 × extract 워커 4 × 긴 Ego4D 영상의 VideoReader 버퍼.
③ 11:00 크래시는 OOM이 아니라 **decord 디코드 오류**(shape '[-1,3,2,16,16]' — vlm.py:56 주석의 기존 이슈 계열,
프레임 수/크기 불일치가 per-sample try/except **앞 단계**에서 터짐).

**가드 3개 (다음 스테이지부터, 코드 변경은 현 실행 종료 후 적용)**:
1. `extract_frames_parallel` workers **4→2**, `prefetch_chunks` batch **8→4** — 진동 상단 ~반감.
2. **프레임 검증 가드**: prefetch 직후 "프레임 개수 == n_frames(짝수) ∧ 모든 프레임 동일 크기" 검증, 불합격 시
   `skip_decode` 태깅 후 continue — 디코드 오류가 런 킬이 아니라 샘플 스킵으로 끝나게.
3. **경보 기준**: mem.log cur > **225G** 지속 2틱이면 알림 (watchdog에 조건 추가). RAM_FLOOR_GB=100 유지.

### 3-1. 정직한 구분 — 위 조치 중 무엇이 "근본 해결"이고 무엇이 "완화"인가

| 원인 | 현행/§3 조치 | 성격 | 진짜 근본 해결 |
|---|---|---|---|
| ① 동시 arm RAM 합산 (240G peak의 주범) | MAX_PARALLEL=1 + RAM_FLOOR admission | **근본 해결 ✓** (구조적으로 재발 불가) | — |
| ② decord 디코드 버퍼 진동 (±25G) | workers 4→2·chunk 8→4 | **완화** (상단을 낮출 뿐, 진동 자체는 남음) | **프레임 사전 추출 캐시** (아래) |
| ③ decord 디코드 오류 크래시 | 프레임 검증 가드 | **전환** (런 킬→샘플 스킵; 오류 발생 자체는 남음) | 사전 추출 캐시로 함께 소멸 |

**②③의 근본 해결 = 프레임 사전 추출 캐시**: 학습/평가 풀(≈6k 샘플)의 8프레임@336을 **오프라인에서 1회
추출·검증해 디스크에 저장**(JPEG, 수 GB)하고, 학습 루프는 이미지 파일만 읽는다. 효과: (a) 학습 중 RAM이
디코드와 무관해져 **상수화** — 진동 소멸, (b) decord 스레드 이슈가 학습 경로에서 **원천 제거** — 추출 단계에서
1회 검증으로 끝, (c) 재실행·재평가 시 디코드 0회로 **속도도 향상**. EGO_jihun 파이프라인이 정확히 이 방식
(`image_path` 사전 추출)으로 돌았고 동일 서버에서 RAM 문제가 없었다 — **실증된 해법**이다.

**도입 시점 판단 (정직한 2층 구분)**:
- **현재 theta_ce 스테이지**: 캐시 전환 = 입력 경로 변경 = **재시작**(진행분 손실). **불가.** on-the-fly 유지.
- **미래 스테이지(sft_r15·r0·eval 등)**: 지금(theta_ce가 GPU를 점유해 CPU/디스크가 노는 동안) **병렬 사전 추출
  가능하며, 그게 더 낫다.** 미룰 이유 없음 — 단 **RAM-경계 추출**로 해야 함(추출도 decord 사용 → 240G 캡 근처
  무경계 실행 시 OOM 유발). 경계 규약:
  - worker **1개**, 영상 **1개씩** 처리 → JPEG(@336, 8프레임) 저장 → 즉시 `del`·`gc` (per-video RAM 상수).
  - 매 영상 전 cgroup 여유 RAM 확인, **여유 < 60G면 일시정지**(theta_ce 우선). `RAM_FLOOR` 재사용.
  - 추출 시 §3 가드2 프레임 검증(개수 짝수·크기 동일)을 **1회 수행**해 불량 영상은 매니페스트에 기록.
  - 출력: `runs/cesft_v2/frame_cache/<video_uid>/<sample_id>.jpg` + `manifest.jsonl`(sample_id→경로·검증결과).
  - 학습/평가 로더는 캐시 히트 시 이미지만 읽고, 미스면 on-the-fly 폴백(무손실 점진 전환).

즉 정확한 상태 서술: **①은 근본 해결됨. ②③은 (a) 현재 스테이지는 완화+경보로 버티고, (b) 미래 스테이지는
지금 RAM-경계 사전 추출로 근본 해결에 착수**한다. "다음 사이클로 전면 미루기"는 과보수였고 철회한다.

## 4. 수정 후 체인 (직렬, 예상 ~14h)

```
[진행중] A_theta_ce(≈3h 잔여) → A_eval_theta(0.5h)
→ A_sft_r15(2.5h) → A_eval_r15(0.5h) → A_harden_r15(0.7h, IV_N=800)   ← 본 실험 판정
→ NEW: theta_ce strip-eval(0.5h)                                        ← B_nohist 대체
→ C_r0(2.5h) → C_eval_hard(r0만, IV_N=400, 1.0h)
→ [게이트] r15 G-NH FAIL 시에만 C_r30(2.5h)+eval
→ C_wise α=0.5 한 점(0.6h)
→ [게이트] P-UTIL PASS 시 부록A(3h)
→ report
```
마커 조작으로 구현: `B_candfree`·`B_nohist`는 **마커 선치기(touch)로 SKIP** 처리하고 사유를 chain.log에 기록
(chain.json 순서 재작성보다 안전 — 실행 중 supervisor가 파일을 다시 읽어도 무해).

## 5. 적용 절차 (순서 엄수)

1. **지금**: 아무것도 죽이지 않는다. `runs/cesft_v2/logs/mem.log` 감시 + 경보 기준만 추가.
1-b. **theta_ce가 도는 지금 (실측: cgroup 216G/240G, 여유 ~24G — 매우 빡빡)**:
   - **동시 사전 추출은 하지 않는다** — 여유 24G에 추출기(경계형이어도 decord 사용)를 얹으면 theta_ce에
     OOM 리스크 추가. 사전 추출은 **theta_ce 완료 갭(RAM 해제 후)**에 굽는다(§5-2c).
   - **지금 할 수 있는 건 "코드 준비"뿐 (GPU/RAM 미사용)**: 캐시 추출기·로더 캐시히트분기·vlm.py 가드(§3)를
     **작성만** 해두고 실행은 안 한다. theta_ce는 도는 코드를 다시 안 읽으므로 파일 수정 자체는 무해하나,
     혼선을 피해 **완료 후 커밋/적용**을 권장.
   - **theta_ce 재실행 금지**: 매몰 79분(35% 진행)이고 prefetch가 디코드를 GPU 뒤로 숨겨 캐시의 속도 이득이
     ~1.5%뿐 — 재실행은 손해. 재실행은 theta_ce가 **또 OOM으로 죽었을 때만**(그땐 0에서 재시작이니 캐시
     먼저 굽고 클린 재출발).
2. **theta_ce 완료 마커(S_CE_THETA_CE_DONE) 확인 후**:
   a. `touch runs/cesft_v2/markers/S_CE_CAND_FREE_DONE S7_EVAL_CAND_FREE_DONE S_CE_NO_HISTORY_DONE`
      + chain.log에 "EGO_jihun 확정 결과로 대체(핸드오프 §2)" 한 줄 기록.
   b. strip-eval 스테이지 추가: 배터리 스크립트에 `--no_history`(history만 공란, WM 후보 불변) 옵션 구현 →
      θ_CE 어댑터로 실행 → paired Δacc 산출(EGO_jihun v3_hstrip_analysis.py 이식).
   c. **프레임 캐시 빌드(이 갭에서, RAM 해제됨)**: 미래 스테이지(sft_r15·r0·eval) 풀의 8프레임@336을
      RAM-경계 추출(§3-1 규약) → `runs/cesft_v2/frame_cache/` + `manifest.jsonl`. 로더 캐시히트/폴백 분기 적용.
   d. vlm.py 가드 3개(§3) 적용 — theta_ce가 끝난 뒤이므로 안전.
3. **r15 harden 완료 후**: `G-NH PASS` → `touch S6_SFT_R30_DONE S3H_SFT_R30_DONE`(사유 기록) / FAIL → r30 진행.
4. **WiSE 단계**: 체인 스크립트의 α 루프를 `0.5` 한 점으로 (환경변수 또는 스크립트 수정).
5. **매 단계 보고**: 마커·소요·G-게이트 판정을 chain.log와 아티팩트에 남긴다.

## 6. 건드리면 안 되는 것 (명시적 금지)

- 실행 중 theta_ce 프로세스·supervisor·orchestrator·watchdog **중단 금지**.
- **A_sft_r15·C_r0의 학습 설정(샘플 수·epochs·replay ρ) 변경 금지** — 본 실험의 비교 가능성 훼손.
- harden **headline(r15)의 IV_N=800 유지** — 축소는 비-headline run만.
- EGO_jihun 체크포인트를 θ_CE 자리에 로드하는 시도 금지 (§1) — cross-pipeline 부록은 체인 완료 후 별도.
- 시간 계약(retro4)·WM prior·프롬프트 문구·**프레임 수(8)** 변경 금지 (§1-1 — 프레임 축소는 다음 사이클 결정).

## 7. 근거 좌표

| 무엇 | 위치 |
|---|---|
| 성립부등식·DiD·strip 확정치 | EGO_jihun `docs/experiments/2026-07-24_reasoning_quality_quantitative_evidence_handoff.md` §1-1·§3 |
| strip 구현 참조 | EGO_jihun `scripts/step2/eval_candidate_scored.py`(--no_memory) · `scripts/step2/v3_hstrip_analysis.py` |
| E0 base 앵커 | [[2026-07-24_evaluation_metrics_handoff]] §5 |
| OOM 이력·RAM 게이트 | `tools/parallel_orchestrator.py` 주석(2026-07-24) · `runs/cesft_v2/logs/mem.log` |
| 디코드 크래시 계열 | `src/ego/step2_retrospection/vlm.py:56` 주석 |
| 체인 예산 | `runs/cesft_v2/chain.json` |
