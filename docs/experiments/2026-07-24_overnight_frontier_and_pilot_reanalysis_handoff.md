# 야간 무인 실행 — Frontier(Gemini) 평가 + 파일럿 재분석 Handoff

> 작성: 2026-07-24 KST(UTC 18:57경) · EGO_jihun 세션. 사용자는 프롬프트 후 즉시 이탈 → 전부 무인 실행.
> **목적: (A) main.tex의 Frontier VLM 빈칸(빨강 $\dagger$)을 gemini-2.5-pro 실측으로 채우고,
> (B) reasoning_quality handoff §5의 "우리 쪽 검증 공백" 3건을 파일럿 실측으로 확정/정정.**
> 관련: [[2026-07-24_reasoning_quality_quantitative_evidence_handoff]] ·
> [[2026-07-24_cesft_v2_time_oom_optimization_handoff]] · main.tex §Results(sec:results)

---

## 0. 한 줄 요약 (다섯)

1. **[완료·정정] G-DELTA**: 동일 500-sample로 맞춰 재계산하니 C-presented vs Q-freegen **+1.2pp, CI[-5.2,+6.7] 비유의**.
   기존 handoff의 "full +2.4pp"는 C를 1520개·Q를 다른 500개로 잰 **표본 불일치 산물**이었음. covered-only +19.2pp는 편향치로 폐기.
2. **[완료·확정] GADR 귀인**: retention +25.5pp > GADR +18.5pp → **8.1(SelAcc 전반 동조)** 확정. 논문 초안의 "8.2 hard-case 교정 중심"은 실측과 불일치 → 서술 수정 필요.
3. **[완료·검증] trace 텍스트 지표**: 1인칭율 74.0→31.6(C)→21.2(Q)→7.4(P)% 등 논문 표(파랑) 재현. TRACE_TEXT_METRICS.json.
4. **[진행중·무인] Frontier gemini-2.5-pro**: VPA T3/T4 전량 + candidate-scored 선택(heldout 1520) 3워커 동시 백그라운드. 벽시계 ~3.5h.
5. **[대기·무인] cand_free strip**: GPU가 헤드라인 체인(cesft_v2)에 물려 있어, **CESFT_V2_CHAIN_DONE 마커 + GPU유휴** 시에만 발화. 오늘 체인 미완이면 dormant(무해) → 다음 세션 수동.

---

## 1. 완료된 GPU-free 분석 (파일럿 데이터, 즉시 확정)

### 1-1. G-DELTA full-set 재계산 — **정정 사항**
- 스크립트: `EGO_jihun/scripts/step2/g_delta_fullset_recompute.py`
- 결과: `EGO_jihun/runs/goalstep_v3_boundary/eval/G_DELTA_FULLSET_VERDICT.json` (사본: overnight_20260724/)
- 방법: C-freegen/Q-freegen이 쓴 **동일 500 sample_id**로 C-presented/Q-presented를 축소 → 완전 paired 4-arm, video-cluster(31개) paired bootstrap.
- **실측**: (matched 500) C-presented 22.0% · C-freegen 22.0% · Q-presented 13.6% · Q-freegen 20.8%
  - C-presented − Q-freegen(시스템 비교, 주 지표): **+1.2pp CI[-5.2,+6.7] — 0 포함, 비유의**
  - C-presented − C-freegen(동일 체크포인트 배포효과): **0.0pp** (후보 제시가 C에 이득 없음)
  - Q-presented − Q-freegen: **-7.2pp CI[-10.8,-3.8]** (후보 제시가 Q를 오히려 해침, 유의)
- **왜 기존 +2.4pp와 다른가**: 기존치는 C-presented를 전체 1520(23.2%)로, Q-freegen을 500(20.8%)로 잰 **다른 표본 비교**. 동일 표본이면 C가 22.0으로 내려가 델타 소멸.
- **함의**: 파일럿에서도 "WM-후보 제시 > 자유생성" 성립부등식은 **표본을 맞추면 유의하지 않다.** G-DELTA는 보조 게이트이니 헤드라인(G-NH)엔 영향 없으나, 논문에 성립부등식을 실측으로 주장하려면 표본-정합 재측정(가능하면 큰 n·video-cluster)이 필요. 단 cluster 31개라 검정력 자체가 약함(주의).

### 1-2. GADR 귀인 — **확정(8.1)**
- 스크립트: `scripts/step2/gadr_attribution.py` · 결과: `.../eval/GADR_ATTRIBUTION.json`
- 방법: WM 고정 cell(G1: wm1==gt, G2: gt∈top5 & wm1≠gt) 내 correct rate를 base→C 델타로 분해.
- **실측**: retention(G1) 28.6→54.1%(**+25.5pp**), GADR(G2) 18.5→37.0%(**+18.5pp**). GADR/retention 비 = 0.73(<1).
- **판정**: GADR 상승폭이 retention 상승폭보다 **작음** → "hard-case 교정만 크게 오른(8.2)" 게 아니라 **retention·GADR이 함께 오른 전반 상승(8.1)**. reasoning_quality handoff §5-3의 주장이 맞음.
- **함의**: main.tex §er-next/ablation에서 GADR을 "모방 불가한 hard-case 교정 능력"으로만 프레이밍하면 과장. "retention·correction 동반 상승"이 정직한 서술. (GADR>0·base 대비 상승 자체는 유효 — 방향만 조정.)

### 1-3. trace 텍스트 지표 재계산 — **검증(표 파랑 재현)**
- 스크립트: `scripts/step2/trace_text_metrics.py` · 결과: `.../eval/TRACE_TEXT_METRICS.json`
- candidate-free 완성문 파싱(GPU 불필요). 1인칭율 74.0/31.6/21.2/7.4%(base/C/Q/P), scene 54.4/47.0/26.4/29.4%, future 99/97.8/91.4/95%, elimination 88.4/80.2/84.2/85.8%, avg_words 80.8/69/64.2/65.3.
- main.tex Table~\ref{tab:trace_metrics}의 파랑 수치와 정합. **1인칭율은 candidate-FREE 조건** — 배포조건(후보 제시) 1인칭율(52.4→61.4%)은 별도이며 논문 주장은 반드시 후보-제시 trace 근거(handoff §2 [반증] 준수).

---

## 2. 무인 실행 중 — Frontier gemini-2.5-pro (API, GPU 미사용)

- 게이트웨이: `https://gw.letsur.ai/v1`, 모델 `gemini-2.5-pro`(사용자 제공 키). 키는 `EGO_jihun3/.env.local`(chmod 600, .gitignore 처리)에서만 로드 — **로그·코드에 하드코딩 없음**.
- 런처: `EGO_jihun3/runs/overnight_20260724/run_api_tracks.sh` (setsid 백그라운드, 세션 이탈 후 유지). 3워커 동시.
- **Track A — VPA(Table 4 tab:vpa 'Frontier VLM (full)')**: `run_frontier_baseline.py`로 T3(test 1042)·T4(test 988) 전량 생성 → `eval_vpa.py` 채점(SR/mAcc/mIoU + bootstrap CI). 기존 스크립트 재사용(신규 코드 0).
- **Track B — candidate-scored 선택(Table 3 tab:reasoning 'Frontier VLM' + Table 4 tab:trace 열)**: 신규 `EGO_jihun3/src/ego/step3_results/vpa/frontier_select_eval.py`. heldout 1520, WM Top-10 후보 텍스트 제시 → JSON {reasoning, choice} → SelAcc/G1/GADR/coverage 분해. **resume-safe**(records.jsonl에 있는 sample_id skip).
- **한계(반드시 논문에 명기)**: 게이트웨이 모델이 **text-only**(vision 미지원) → frontier는 이미지 미열람, base/C 정책은 프레임을 봄. **직접 비교 아님** — main.tex §er-next의 "text-conditioned, 별도 평가" 각주와 일치. reasoning 열에 각주 유지.
- 진행 스냅샷(18:57경): VPA T3 86/1042·T4 88/988, select 100/1520 acc≈0.23(apifail 0). select ETA ~3.5h.

### 결과 위치 (완료 시 자동 생성)
| 산출물 | 경로 |
|---|---|
| VPA 예측 | `runs/overnight_20260724/frontier/preds_frontier_T{3,4}.json` |
| VPA 채점 | `runs/overnight_20260724/frontier/frontier_T{3,4}*` (eval_vpa 출력) |
| 선택 records/집계 | `runs/overnight_20260724/frontier/frontier_select.{records.jsonl,json}` |
| 진행 로그 | `runs/overnight_20260724/logs/{master,vpa_T3,vpa_T4,select}.log` |
| 완료 마커 | `runs/overnight_20260724/markers/{VPA_T3_EVAL,VPA_T4_EVAL,SELECT}.DONE`, `API_TRACKS_ALL_DONE` |

### 상태 확인 방법 (다음 세션)
```
R=/mnt/nvme/migration/jihun/EGO_jihun3/runs/overnight_20260724
ls $R/markers/                         # *.DONE 확인
tail $R/logs/select.log                # 최종 acc/covered SelAcc
cat  $R/frontier/frontier_select.json  # Table 3 Frontier 행 수치
cat  $R/frontier/frontier_T3*/*.json   # Table 4 mAcc/mIoU
```
재시작 필요 시(중단됐다면): `bash $R/run_api_tracks.sh` — 완료분 자동 skip.

---

## 3. 무인 대기 — cand_free strip ablation (로컬 GPU 필요, 안전 게이트)

- **목표**: 후보를 아예 제시 안 하는 자유생성에서도 "무-hist WM>LM, hist LM≫WM" 크로스오버가 재현되는지 → **이중해리가 후보-리스트 구조의 인공물이 아님**을 입증(또는 결함 조기발견).
- 신규: `EGO_jihun/scripts/step2/v3_cf_freegen_eval.py`에 `--no_memory` 추가(기존 `T.NO_MEMORY` 재사용) + `scripts/step2/cf_freegen_strip_analysis.py`(paired Δacc·history-bin·DiD).
- 런처: `runs/overnight_20260724/run_gpu_strip_when_idle.sh` (setsid 백그라운드, PID 폴링).
- **안전 게이트(중요)**: 이 GPU에서 cesft_v2 헤드라인 체인(sft_r0→조건부 r30→wise→부록A, 수 시간)이 돈다. 동시 비디오-디코드 arm 2개 = OOM(cesft handoff 경고). 따라서 **발화 = `CESFT_V2_CHAIN_DONE` 마커 존재 AND 발화시점 GPU used<3GB**. 30분-유휴 자동발화는 **제거**(중간 갭 오발화 방지). 오늘 체인 미완이면 strip은 dormant로 남고 예산(10h) 소진 후 `STRIP_SKIPPED_NO_GPU_WINDOW` 남기고 무해 종료.
- **수동 실행(다음 세션, GPU 확실히 빈 뒤)**:
  ```
  # cesft 체인이 끝났거나 GPU가 확실히 idle 이면:
  cd /mnt/nvme/migration/jihun/EGO_jihun
  PY=/mnt/nvme/migration/jihun/envs/miniforge3/envs/eve-cu124/bin/python
  $PY scripts/step2/v3_cf_freegen_eval.py --jsonl runs/goalstep_v2/data/v2_heldout_eval.jsonl \
     --adapter outputs/step2/goalstep_v2_c/checkpoint-final --limit 500 --seed 42 --no_memory \
     --out runs/goalstep_v3_boundary/eval/cf_free_s_nohist.json   # C arm
  # Q arm: adapter를 goalstep_v3_q/checkpoint-final 로, out을 cf_free_q_nohist.json 로
  $PY scripts/step2/cf_freegen_strip_analysis.py
  ```
- 결과(완료 시): `runs/goalstep_v3_boundary/eval/CF_FREEGEN_STRIP_VERDICT.json` (C/Q history-bin Δacc + DiD + 무-hist WM vs LM).

---

## 4. 헤드라인(cesft_v2)과의 관계 — 이 세션이 건드리지 않은 것

- **G-ACC1 파일럿↔확정셋 불일치 신호(주의)**: 확정셋 candidate-CE>WM Top-1 = **+7.2pp CI[0.46,15.0]** (n=389/76클러스터) vs 파일럿 +19.7pp. 효과 1/3 축소·CI 하한 겨우 0 비켜감. 원인은 (1)과제 난이도(현재행동 인식→평균 12.8s 미래 예측), (2)WM coverage 51.4→41%, (3)표본 1/4. **깨진 것 아님(다른 파이프라인)이나 확정셋 증거는 약함** → harden(G-NH) 결과와 반드시 함께 해석하고, 여유되면 EVAL_N 키워 CI 재측정 권고.
- **cesft_v2 체인은 다른 세션 소관** — 이 세션은 프로세스/마커/스크립트 일절 미변경. sft_r0 진행중(스냅샷 19%). strip 런처만 read-only로 마커를 관찰.
- **strip-eval(θ_CE, cesft_v2 본셋)**: 다른 세션이 헤드라인 확보 직후 별도 런처로 실행 예정(이 세션의 cand_free strip과는 다른 것 — 저건 candidate-scored 본셋, 이건 파일럿 free-gen 크로스체크).

## 5. 다음 세션 체크리스트 (우선순위)

1. `runs/overnight_20260724/markers/API_TRACKS_ALL_DONE` 확인 → `frontier_select.json`·`frontier_T{3,4}` 수치를 main.tex Table 3/4의 Frontier 행에 기입(빨강$\dagger$→검정, **text-conditioned 각주 유지**).
2. **G-DELTA 정정 반영**: 성립부등식 주장 시 표본-정합 +1.2pp(비유의)를 정직히 반영하거나, 주장 강도 하향. covered-only +19.2pp 인용 금지.
3. **GADR 서술 수정**: "hard-case 교정 중심(8.2)"→"retention·GADR 동반 상승(8.1)".
4. cesft 체인 종료 확인 후 strip 결과(`CF_FREEGEN_STRIP_VERDICT.json`) 확인, 없으면 §3 수동 실행.
5. G-ACC1 확정셋 마진(§4) — harden 결과와 교차 해석.

## 6. 근거 좌표(이 세션 산출물)
| 무엇 | 위치 |
|---|---|
| G-DELTA 재계산 | `EGO_jihun/scripts/step2/g_delta_fullset_recompute.py` → `.../eval/G_DELTA_FULLSET_VERDICT.json` |
| GADR 귀인 | `scripts/step2/gadr_attribution.py` → `.../eval/GADR_ATTRIBUTION.json` |
| trace 텍스트 | `scripts/step2/trace_text_metrics.py` → `.../eval/TRACE_TEXT_METRICS.json` |
| Frontier VPA | `EGO_jihun3/src/ego/step3_results/vpa/run_frontier_baseline.py`(기존) |
| Frontier 선택 | `EGO_jihun3/src/ego/step3_results/vpa/frontier_select_eval.py`(신규) |
| free-gen strip | `scripts/step2/v3_cf_freegen_eval.py --no_memory`(수정) + `cf_freegen_strip_analysis.py`(신규) |
| 런처 | `EGO_jihun3/runs/overnight_20260724/run_api_tracks.sh`, `run_gpu_strip_when_idle.sh` |
| 키(로컬만) | `EGO_jihun3/.env.local` (chmod600, git 제외) |

---

## 7. [정정·중요] Frontier(gemini-2.5-pro) 실행 결과 — API COST 한도 소진으로 부분 완료

작성 후 확인(07-25 03:30경): 게이트웨이가 `{"type":"usage_limit_exceeded","detail":"COST limit exceeded"}`,
`retry-after: 3600` 반환. **순간 rate limit이 아니라 이 키의 비용 예산 소진**(하드 캡). 3워커 동시 실행이
비용을 빠르게 소진해 **세 트랙 모두 ~440~548번째에서 동시 중단**.

| 트랙 | 정상 응답 | 전체 | 상태 |
|---|---|---|---|
| VPA T3 | 440 | 1042 | **부분(42%)** — 나머지 빈 예측 |
| VPA T4 | 440 | 988 | **부분(45%)** |
| Track B select | 548 | 1520 | **부분(36%)** |
| 합계 호출 | ~1428 | — | 이 지점에서 COST 소진 |

- **VPA EVAL 마커(VPA_T{3,4}_EVAL.DONE)는 제거함**: 빈 예측(>55%) 위에서 채점돼 mAcc/mIoU가 무효였음.
  SELECT.DONE·API_TRACKS_ALL_DONE 도 제거(부분 데이터). 예산 복구 시 `run_api_tracks.sh` 재실행하면 깨끗이 resume.
- **정직한 부분 지표(정상 응답분만, 확정 아님·부분표본 편향 주의)**:
  Track B select covered SelAcc@5 = **0.312 (n=295)**, full acc 0.175, malformed 0.
  → 논문 Table 3 Frontier 추정치(SelAcc 24–28)와 비교해 부분표본상 상회하나, **표본 42% 편향**이라 확정 불가.
- **resume 조건**: 키 비용 예산 충전/리셋(retry-after 1h이나 하드 캡이면 리셋까지 대기) 후,
  `--sleep`으로 throttle + **단일 워커**로 재실행 권장(동시 3워커가 비용 급소진의 직접 원인).
  select는 `--retry_errors` 로 429 실패분만, VPA는 run_frontier_baseline.py 가 preds.json에 이어쓰기(부분 재개).
- **비용 절감 대안(다음 세션 판단)**: gemini-2.5-pro(reasoning, 고가) 대신 gemini-2.5-flash 로 전량 → 저비용,
  단 논문 'frontier flagship' 주장은 약화. 또는 대표 서브셋(n=200~300)만 pro로.
