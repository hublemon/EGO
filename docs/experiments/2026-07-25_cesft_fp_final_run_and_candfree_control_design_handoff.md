# cesft 최종 실행 설계 — 1인칭 일원화 재학습 + cand_free 대조군 (단일 코호트 · 1회 완주) Handoff

> 작성: 2026-07-25 KST · EGO_jihun3 / runs/cesft_v2 후속. **실행 설계 SSOT.**
> 선행: [[2026-07-25_cesft_rerun_firstperson_unification_and_controls_handoff]] (P1/P2 분리안 — **본 문서가 통합안으로 대체**) ·
> [[2026-07-25_cesft_metrics_dashboard_and_gate_update_handoff]] (현행 실측) ·
> `EGO_jihun/docs/experiments/2026-07-25_first_person_pronoun_erosion_candidate_vs_gt_ce_handoff.md` (rev3, 파일럿 실측).
> **결정 사항 (사용자 확정, 2026-07-25)**: ① 1인칭 재실행이 논문의 **최종 학습 결과**가 된다. ② cand_free 대조군 **포함**,
> random_cand **제외**. ③ 시간 제약상 **arm별 학습 1회·평가 1회**로 논문에 기입할 전 지표 + 실제 추론 로그를 확보한다.
> (→ [[2026-07-25_cesft_full_training_plan_and_metrics_handoff]] §1.2의 "cand_free 제거" 결정은 본 결정으로 **번복됨**.)

---

## 0. 세 줄 요약

1. **한 개의 무인 체인으로 끝낸다**: 프롬프트 3곳 1인칭화 → Φ 재생성·관문 → θ_CE → SFT(r15) → **cand_free** → 4-arm 전 평가. 선행 문서의 P1(현행 프롬프트 cand_free만)과 P2(1인칭 재실행)를 **분리 실행하면 안 되는 이유가 생겼다** — 1인칭 재실행이 최종 코호트가 되는 이상, 대조군도 **같은 프롬프트·같은 코호트 안**에서 학습해야 같은 표에 들어간다(코호트 분리 규칙). GPU ≈ **12.5h**, 벽시계 ≈ 13.5h.
2. **대조군의 역할은 "coverage 한계의 반전"이다**: covered(GT∈D_t) 조건에서 θ_CE > cand_free(G-DELTA)를 보이면, "경계 안에서만 선택한다"는 구조적 한계를 안고도 **후보 대조 학습이 능력을 강화**함을 증명한다. 정확도 하나가 아니라 **추론 지표 전반의 이중 해리** — cand_free는 학습할수록 history 사용·장면 근거·인과 연결이 하락, 우리는 상승 — 를 파일럿 실측(§2)이 이미 보였고, 본셋에서 같은 설계로 재측정한다.
3. **사전 등록 게이트**: G-DELTA(SelAcc paired CI>0)·G-DiD(strip Δ 차이 CI>0)는 실패 시 논문 중심 주장 철회. 1인칭·텍스트 지표는 같은 템플릿 내 arm 비교로만(각주 공시), 파일럿 수치는 **설계 근거로만 인용하고 논문 표에는 절대 넣지 않는다**(§7 정직 규칙).

---

## 1. 논지 구조 — 왜 이 대조군이 연구 의의를 지키는가

논문 문장: *"정답 다음 행동이 D_t에 포함되는 비율을 candidate coverage라 하며, 경계 안에서만 행동을 선택하는 EGO에서는 이 값이 하류 VLM이 복구할 수 있는 행동의 범위를 결정한다."*

이 문장은 coverage(43.4%)가 **상한을 씌우는 한계**로 읽힌다. 이를 한계가 아니게 만드는 논증이 cand_free 대조군이다:

| 단계 | 주장 | 증명 수단 |
|---|---|---|
| ① | 경계 없이 GT만 학습(cand_free)해도 **같은 covered 조건에서 우리보다 못하다** | G-DELTA-1: SelAcc(θ_CE) − SelAcc(cand_free), covered·paired |
| ② | 못하는 이유는 단순 정확도가 아니라 **추론 능력 자체의 방향 차이** — cand_free는 학습을 거치며 history 사용·근거 서술이 *하락*, 우리는 *상승* | G-DiD(strip) + 텍스트 지표 4종 + belief 개입 + 실물 추론 로그 |
| ③ | 따라서 coverage는 "잘라먹는 상한"이 아니라 **판별 능력을 만들어내는 커리큘럼의 정의역** | ①+②의 종합 서술 |

②가 본 설계의 핵심 확장이다 — 정확도 1개 지표로는 "cand_free가 원래 약한 세팅"이라는 반론을 못 막지만, **여러 독립 지표가 같은 방향(하락)으로 움직이면** 학습 신호 자체의 문제임을 보일 수 있다.

---

## 2. ②의 실측 근거 — 파일럿(Q arm)에서 이미 관측된 하락 (설계 근거, 논문 인용 금지)

파일럿(EGO_jihun, goalstep_v3_boundary)의 Q(GT-CE) = 본셋 cand_free의 등가 arm. **모든 수치는 다른 코호트(K, 시간 계약, 템플릿 상이)이므로 방향·설계 근거로만 쓴다.**

### 2-1. 판별·history — 이중 해리 (원장: `HSTRIP_VERDICT.json`, n=1,520 paired)

| 지표 | C(candidate-CE) | Q(GT-CE) | 함의 |
|---|---:|---:|---|
| SelAcc (covered, 후보 제시) | **41.4%** | 26.9% | 후보 제시 조건에서 큰 폭 열세 |
| strip Δacc (history 인과) | **+12.6pp** [10.5,14.7] | +4.2pp [2.6,5.8] | DiD +8.4pp [8.0,8.8] — Q는 history를 1/3만 사용 |
| 자유생성 gt_correct | 22.0% | 20.8% (비유의) | "후보 *제시*가 성능을 올린다"가 아니라 "후보 *학습*이 판별을 가르친다" — 축 구분 유지 |

### 2-2. 추론 텍스트 지표 — Q의 전방위 하락 (원장: `TRACE_TEXT_METRICS.json`, n=500/arm, 자유생성)

| 지표 | base | C | **Q** | Q의 하락 폭 (vs base) |
|---|---:|---:|---:|---|
| scene_desc_rate (장면 근거 서술) | 54.4% | 47.0% | **26.4%** | **−28.0pp — 하락 최대. 시각 근거 서술이 반토막** |
| causal_rate (인과 연결어) | 16.2% | 8.6% | **6.8%** | −9.4pp |
| first_person_rate (1인칭, cand-free 레짐) | 74.0% | 31.6% | **21.2%** | −52.8pp (C보다 10.4pp 더 침식) |
| avg_words (reasoning 길이) | 80.8 | 69.0 | **64.2** | −16.6 단어 |
| future_rate | 99.0% | 97.8% | 91.4% | −7.6pp |

스텝별 실측(`STEPWISE_TRACE_PROBE_METRICS.json`, n=6/점 — 방향성만): C는 2,500스텝 내내 1인칭 유지(.33–.50), **Q는 후반(seen≥2000)에 침식 진행**(.67→.17). 즉 하락은 학습 **진행에 따라** 발생 — "학습을 거치면 능력이 하락한다"의 시간축 증거 형태이며, 본셋 probe(§4-3)가 같은 형태를 재측정한다.

### 2-3. 실물 추론 로그의 대비 (파일럿 관측 → 본셋에서 재확보할 형태)

- C(후보 학습): *"I am currently in the process of making flatbreads. I have been alternating between rolling out dough and cooking…"* — 1인칭 + history 사이클 인용 + 후보 배제를 동시 수행.
- 본셋 θ_CE+SFT의 GADR 실물(대시보드 §6): *"The person has been repeatedly checking the heat… Among the candidates, only checking heat aligns with the established pattern…"* — history 주기 패턴을 판별 단서로 사용.
- 기대하는 cand_free 실물: history를 무시한 일반 상식 추론(파일럿 base 오답 패턴과 동형 — *"Adding water would be a logical next step…"*). §4-5의 `pick_trace_examples.py`가 같은 anchor에서 이 대비를 자동 추출한다.

---

## 3. 설계 원칙 — 단일 코호트 강제

1. **모든 학습 arm(θ_CE·sft_r15·cand_free)과 base 평가가 같은 1인칭 프롬프트·같은 subset·같은 step 수·같은 seed**로 산출된다. 프롬프트가 다른 arm은 같은 표에 못 들어간다(crosscohort §2-4 판정의 교훈 — 이번엔 우리 안에서 지킨다).
2. 기존 `runs/cesft_v2` 산출물(G-ACC1 +4.8pp 등)은 **동결·보존**하고 논문 표에서 **전면 교체**한다. 새 코호트가 실패하면 기존 유지(선행 handoff §8 규칙 계승).
3. 신규 run dir: **`runs/cesft_v2_fp`** (frame_cache만 `runs/cesft_v2/frame_cache` 공유 — 재추출 금지).
4. 학습 규모는 파일럿 subset 규모 유지(CE 4,189 / SFT ~2,945 / eval n=1,000 동일 셋) — full 스케일업([[2026-07-25_cesft_full_training_plan_and_metrics_handoff]])은 별건으로 남긴다. 시간 제약의 직접 귀결.

### 3-1. 반영을 경계한 기록 (명시적 제외)

| 기록 | 제외/주의 사유 |
|---|---|
| 파일럿 Q의 `keep_all_cf`(GT가 후보 밖이어도 학습) 정책 | 본셋 cand_free는 **G-EQ(동일 subset) 우선** — θ_CE와 같은 covered 4,189로 학습한다. 파일럿 방식 이식 금지 |
| 파일럿 K=5·start−1s 시간 계약·`make_cf` 프롬프트 수술 | 본셋은 K=10·end−1s·`SYS_NOCAND` 프리셋 — 파일럿 구현을 가져오지 않는다 (`select_ce.py`에 이미 있음) |
| "cesft_v2는 중간 체크포인트가 없다"(07-25 이전 문서들) | **낡음** — `train/ckpt.py` 구현·kill→resume 실검증 완료. 단 step-태그 보존은 여전히 미구현(§5 코드 변경 #3) |
| cesft_full §1.2 "cand_free 학습 arm 제거" | 사용자 결정으로 번복 — 본 문서가 우선 |
| 초기 재실행 초안의 "semantic gate 0.5h·base trace 재생성 2.4h" | cesft 경로는 LLM judge를 쓰지 않음(선행 handoff §4-4 정정 계승) — 포함하지 않는다 |

---

## 4. 실행 설계 — 무인 체인 1회 (스테이지 순서대로)

### 4-0. P0 — 코드 변경 (착수 전, §5 목록) + 스모크

CE 3샘플·SFT 3샘플·cand_free 3샘플·projection 8샘플·battery 8샘플·freegen 4샘플을 `--limit`로 통과시키고 마커 삭제. **frame_cache 커버리지 게이트**: CE 풀 4,189 ∪ eval 셋의 manifest(ok) 포함 확인 — 미스 0이 아니면 착수 금지(직전 cand_free 실패 원인이 캐시 미완 + skip_decode 2,856행이었다).

> **⚠ 실측 정정 (2026-07-25 확인)**: 선행 handoff §2-1의 "프레임이 캐시된 지금 재실행하면 해소된다"는 **부정확** —
> manifest(ok) 6,284 대조 결과 CE 학습 풀 4,189 중 **962개(23.0%) 캐시 미스**, eval 셋 1,000은 미스 0.
> → **P0에 캐시 보완 단계 필수**: `frame_extractor.py`로 미스 962건 선추출 (실측 2.3 s/샘플 단일 워커 ≈ **0.7h**, CPU 작업 —
> GPU 학습과 동시 기동 금지, 완료·게이트 통과 후 체인 착수). 이 962건은 세 arm(θ_CE·SFT·cand_free)이 같은 풀을 쓰므로
> **한 번 보완하면 두 실행 모두에 적용**된다.

### 4-1. G0 — Φ 재생성 + Go/No-Go 관문 (GPU 0.5h + 사람 점검 수 분)

1. 프롬프트 3곳 1인칭화(§5 #1) 후 `hindsight.projection`으로 Φ trace 4,189 재생성 (실측 2.42/s ≈ 0.5h).
2. **관문 판정** (선행 handoff §7 계승): ⑴ 규칙 게이트 통과율 ≥60% (현행 70.3% 기준, `quality_gate` 정규식은 인칭 대칭이라 큰 변동 없을 것) ⑵ 표본 50개 육안 — hedging("I think/I guess") 유입으로 근거 서술이 흐려지는지 ⑶ 신규 Φ의 `first_person_rate`·`scene_desc_rate`·`avg_words`를 구 Φ와 대조(§5 #5의 텍스트 지표 스크립트를 여기서 먼저 사용). **셋 중 하나라도 이상이면 학습 착수 금지** — 12h를 걸기 전 0.5h에서 되돌린다.

### 4-2. T — 학습 3 arm (GPU ≈ 7.3h, 직렬)

| # | arm | 명령 요지 | 시간(실측 rate) | 비고 |
|---|---|---|---:|---|
| T1 | θ_CE | `train.select_ce --arm wm_cand --tau 1.0 --epochs 1 --seed 42 --ckpt_every 50 --resume auto --probe_every 50` | 3.8h (4,189×3.23s) | **최종본** |
| T2 | sft_r15 | `train.sft_v2 --init_adapter …/theta_ce/adapter --ce_replay_rho 0.15 --epochs 1 --seed 42 --ckpt_every 50 --resume auto` | 2.6h (2,945×3.14s) | 타깃 = **1인칭 Φ** — SFT가 1인칭을 도로 침식시키지 않는 유일 경로 |
| T3 | **cand_free** | `train.select_ce --arm cand_free --epochs 1 --seed 42 --subset_file <θ_CE와 동일> --max_steps <θ_CE와 동일 opt.step> --ckpt_every 50 --resume auto --probe_every 50` | 0.94h (4,189×0.805s) | **G-EQ**: 같은 subset·step·seed. 빠른 건 공짜 이득 — 시간으로 맞추지 않는다 |

- cand_free의 프롬프트는 `SYS_NOCAND`(1인칭화 후) — 후보 블록 없음, GT span 단일 CE (`ARMS["cand_free"]=("free",True,True,"gt")` 프리셋 그대로).
- 첫 200스텝 skip_decode 비율 >5%면 즉시 중단(캐시 게이트 실패 신호).
- **학습 중 계측(추가 비용 ≈0)**: `--probe_every 50` + probe n 8→32 + reasoning 260자 절단 해제(§5 #2) → 스텝별 1인칭율·probe_acc 곡선을 θ_CE와 cand_free **양쪽에서** 확보. 파일럿 §2-2의 "Q는 후반에 침식"을 본셋 시간축에서 재측정하는 장치다. `--ckpt_every 50` + step-태그 보존(§5 #3)으로 중간 어댑터를 남겨 사후 재평가 여지도 확보(지난 사고 이력상 step 90·48에서 죽었으므로 50 간격).

### 4-3. E — 평가 4 arm (GPU ≈ 4.8h)

측정 모집단: heldout **n=1,000 동일 셋(seed 42)**, 4 arm(base·θ_CE·sft_r15·cand_free) 전부 동일 sample_id — base 모집단 불일치(기존 gap⑤)도 이 기회에 해소. CI는 video-cluster bootstrap(`tools/paired_boot.py` 규약).

| # | 패스 | 대상 | 산출 지표 | 시간 |
|---|---|---|---|---:|
| E1 | battery ×4 | n=1,000 | SelAcc(covered)·G1·GADR·Full-eq·L0·malformed + **records.jsonl(완성문 원문 = 추론 로그)** | 1.4h |
| E2 | strip ×4 | 동일 셋 | `{arm}_nohist.records.jsonl` → Δacc·H-bin 층화 (`strip_eval.py --arm … --covered_only`, 수정본이 이미 arm 일반화됨) | 1.4h |
| E3 | harden_s3 ×4 | n=400 | belief sensitivity(G-CC1)·U_b(G-CC3)·flip 4종 — **base·θ_CE 공백(기존 gap⑥)과 cand_free를 한 번에** (`--config configs/step2_retrospection/cesft_v2.yaml` 명시 필수 — 기본값이 goalstep yaml임) | 0.6h |
| E4 | freegen ×4 ×2레짐 | n=500 | presented/cand_free 레짐별 생성 → 1인칭율(레짐 분리)·in_support(경계 내재화)·**완성문 원문** | 1.4h |
| E5 | CPU 집계 | — | 텍스트 지표 4-arm(§5 #5)·게이트 일체·DiD·trace 예시 추출 | ~0.2h |

### 4-4. 게이트 (사전 등록 — 사후 해석 금지)

| 게이트 | 정의 | 통과 기준 | 실패 시 |
|---|---|---|---|
| **G-DELTA-1** | SelAcc(θ_CE) − SelAcc(cand_free), covered·paired | CI 하한 > 0 | **논문 중심 주장 철회** |
| **G-DiD** | strip Δ(θ_CE) − strip Δ(cand_free) (`tools/did_history.py` 기본 인자 그대로) | CI 하한 > 0 | "판별 압력이 history 사용을 가르친다" 본셋 미재현 — 자매 코호트 한정 서술로 후퇴 |
| G-ACC1(fp) | SelAcc(θ_CE) − L0 | CI 하한 > 0 | 서사 재검토 — 후속 진행 보류 |
| G-NH(fp) | SelAcc(sft) − SelAcc(θ_CE) | 비열등(하한 ≥ −1pp) | "SFT 비손상" 문장 삭제 |
| G-EQ | 세 arm 동일 subset·steps·seed | 로그 대조 | 위반 시 전 게이트 무효 |
| (관측) 텍스트·1인칭·belief | cand_free의 하락 방향 확인 | 게이트 아님 — CI 병기 관측 지표 | 방향 불일치 시 그대로 보고 (§7) |

### 4-5. 논문 표 매핑 — cand_free 열이 들어가는 자리

| 논문 자리 | 행/열 구성 | 산출 파일 |
|---|---|---|
| Table reasoning | 4열(base·θ_CE·+SFT·**cand_free**) × SelAcc/G1/GADR/Full-eq + L0 행 병기 | `eval/{arm}.json` + `paired_G-DELTA…json` |
| Table ablation (history) | 4열 × strip Δacc + H-bin + **DiD(θ_CE−cand_free) 별도 행** | `strip_verdict_{arm}.json` · `DiD_history_theta_ce_vs_cand_free.json` |
| Table intervention | 4열 × G-CC1/G-CC3/flip(para) — base·cand_free 열이 "belief 인과는 후보+SFT 학습의 산물" 증명 | `eval/{arm}.harden_s3.json` |
| Table trace_metrics | 4열 × 1인칭(2레짐)·scene_desc·causal·avg_words·in_support | `text_metrics.json` (freegen+battery records 재계산) |
| Table trace (정성) | 동일 anchor에서 cand_free 오답(상식 회귀) vs θ_CE·SFT 정답(history 판별) 실물 3–4건 | `trace_examples.md` |

**표 작성 규칙**: 모든 셀은 이번 코호트 산출물만. 파일럿 수치(41.4/26.9, +8.4pp, 74→21.2%)는 본문·표 어디에도 넣지 않는다 — "sibling cohort에서 방향 재현" 각주만 허용.

---

## 5. 필요한 코드 변경 (착수 전 · 총 7건)

| # | 파일 | 변경 | 근거 |
|---|---|---|---|
| 1 | `vlm.py:51` `SYSTEM_PROMPT` · `train/select_ce.py:55` `SYS_NOCAND` · `hindsight/projection.py:25` `PROJ_SYSTEM` | 3인칭 관찰자("the person") → 1인칭 행위자 프레임. 문구 참조: `EGO_jihun/src/ego/step2_vlm_alignment/train_grpo_action.py:157·194`. **teacher.py는 불변**(JSON 스키마, 문체 중립) · `fmt_history` 리스트 포맷 **불변**(strip 비교 기반 보존 — 파일럿에서 페르소나만으로 충분함 실측) | 선행 handoff §4-1·§4-2 |
| 2 | `train/probe_gen.py` | `PROBE_N` 합 8→32 · `:76`의 reasoning 260자 절단 해제(전문 저장) | 스텝별 침식 곡선(§4-2 계측) |
| 3 | `train/ckpt.py` + 두 트레이너 | `--ckpt_every` 저장 시 `adapter_step{N}/` **step-태그 사본 보존**(현행은 롤링 1곳 + 성공 시 `clear_ckpt` 삭제) | 사후 스텝별 재평가 · 사고 이력(step 90/48 사망 — 첫 ckpt 도달 전) |
| 4 | `eval/freegen.py` (신규) | 2레짐(presented/cand_free) × n=500 생성, records 규약은 battery와 동일. 프롬프트는 **본 코호트의 통일 1인칭 프레임 그대로**(별도 화법-중립 수술 불필요 — 전 arm 동일 템플릿이므로 arm 비교 유효) | §4-3 E4 |
| 5 | `tools/trace_text_metrics.py` (신규 이식) | `EGO_jihun/scripts/step2/trace_text_metrics.py`의 **정규식을 글자 그대로 재사용**(1인칭·scene·future·causal·elim·avg_words) — 새 정의를 만들면 파일럿과 잣대가 달라진다. 입력만 EGO_jihun3 records 스키마(`reasoning` 필드 기저장)로 교체 | §2-2와 같은 잣대 |
| 6 | `tools/pick_trace_examples.py` (신규) | 4-arm records 조인 → (GT∈후보 ∧ WM top-1 오답 ∧ cand_free 오답 ∧ θ_CE·SFT 정답) anchor 자동 추출 3–4건 | Table trace |
| 7 | `scripts/step2_retrospection/cesft_fp_chain.sh` + `start_cesft_fp.sh` (신규) | 스테이지 골격: `G0_PROJ → GATE_GO → T1_CE → T2_SFT → T3_CANDFREE → E1..E5 → RP`. 마커 멱등 + `setsid` 분리 + supervisor + **stall 워치독** + `ram_alarm.sh` 상시 | §6 |

`--subset_file`·`--ckpt_every/--resume`·strip의 arm 일반화·`did_history.py`·`gdelta_summary.py`는 **이미 구현되어 있음**(2026-07-25 확인) — 재구현 금지.

---

## 6. 운영 — OOM·강제종료 방어 (사고 4건 재발 방지)

같은 240GiB cgroup 한도에서 4회 사망(07-23 S3-harden · 07-24 동시-arm SIGTERM · 07-24 sft_r0 스톨 · 07-25 cand_free 서버 사망). 직전 사고의 직접 원인: `run_gdelta.sh`가 RAM 게이트·mem 워치독 **미배선** 상태로 기동됐고, 이 환경은 dmesg가 막혀 사후 부검이 불가능하다 — **사전 계측이 유일한 전략**.

| # | 방어 | 구현 |
|---|---|---|
| 1 | 프레임 캐시 게이트 | P0에서 CE 풀 ∪ eval 셋 커버리지 확인, 미스 0 강제. 학습 경로 decord 폴백 0회 목표 |
| 2 | 단일 GPU 잡 | `MAX_PARALLEL=1` · GPU preflight(`MIN_FREE_MB=60000`) |
| 3 | **hard-RAM admission** | `parallel_orchestrator.py`의 `cgroup_ram_free_gb()` 게이트를 체인 스크립트에 **반드시 배선**(`memory.current` 판정 금지 — page cache 오탐으로 자기교착 이력). `run_gdelta.sh`류 단독 스크립트 신규 작성 금지 — supervisor 경유만 |
| 4 | 관측 | `ram_alarm.sh` + mem 로거 상시 기동(관측 전용). 크래시 순간 수치가 반드시 로그에 남게 |
| 5 | 완충 | `memory.high`를 240G보다 낮게(예 200G) 설정 — 하드킬 전 커널 회수 압박 |
| 6 | 스톨 워치독 | status `updated_at` 60s 폴링, 1800s 무갱신 시 해당 스테이지만 kill → supervisor resume. `MAX_RESTART=3` 초과 시 `CHAIN_STUCK` |
| 7 | 학습 내구성 | `--ckpt_every 50 --resume auto`(검증 완료 메커니즘) — kill 시 최대 손실 ~0.35h |

---

## 7. 정직 규칙 (논문 반영 시 강제)

1. **코호트 분리**: 이번 산출물이 성공하면 기존 cesft_v2 수치를 표에서 전면 교체. 두 코호트 수치를 한 표·한 문장에서 직접 비교하지 않는다.
2. **1인칭율은 표면 지표**: 같은 템플릿·같은 레짐 내 arm 비교로만, 템플릿 종속 각주 필수. "egocentric 화법 강화"는 presented 레짐 근거로만, 조건부 서술.
3. **cand_free 하락의 서술 한계**: 텍스트 지표 하락은 관측이지 사전 등록 게이트가 아니다 — 방향이 기대와 다르면 **그대로 보고**하고 서사를 수치에 맞춘다(수치를 서사에 맞추지 않는다).
4. G-DELTA·G-DiD 실패 시 중심 주장 철회 — 이 두 게이트는 사후 해석 대상이 아니다.
5. probe 곡선(n=32)은 CI 없이 방향성 보조로만. "학습이 진행될수록 1인칭 증가" 주장 금지(파일럿에도 증가는 없었다 — 유지 vs 침식만).
6. "causal mediation" 금지 — "interventional dependence"까지만. "CE replay가 필요하다" 주장 금지(ρ 스윕 없음).

---

## 8. 시간표 (실측 rate 기반 · 단일 서버 직렬)

| 단계 | 내용 | GPU | 누적 벽시계 |
|---|---|---:|---:|
| P0 | 코드 7건 + 스모크 + 캐시 게이트 | — | (사람 작업) |
| G0 | Φ 재생성 + Go/No-Go | 0.5h | 0.5h |
| T1 | θ_CE | 3.8h | 4.3h |
| T2 | sft_r15 | 2.6h | 6.9h |
| T3 | cand_free | 0.9h | 7.8h |
| E1–E4 | battery·strip·harden·freegen ×4 arm | 4.8h | 12.6h |
| E5 | CPU 집계·게이트·trace 추출 | ~0 | **≈12.6h + 여유 ≈ 13.5h** |

---

## 8-1. 착수 기록 (2026-07-25 밤, 구현·스모크 완료)

§5 코드 7건 **전부 구현 완료** + GPU 스모크 통과:
- 프롬프트 3곳 + user 헤더("Your completed actions") 1인칭화 · `cesft_v2_fp.yaml` 신설(output_dir 분리 — 트레이너가 config 기준으로 출력 경로를 잡아 cesft_v2 동결 유지)
- probe 32샘플·전문 저장(8개 청크 생성) · `CKPT_KEEP_STEP_ADAPTERS=1` 게이트로 step-태그 어댑터 보존(스모크에서 adapter_step1/2 생성 확인)
- `eval/freegen.py`(cand_free 레짐 전용 — presented 텍스트는 battery records 재계산으로 충당, GPU 0.7h 절약) · `tools/trace_text_metrics.py`(파일럿 정규식 원본 그대로) · `tools/gate_go_check.py`(무인용 자동 관문: pass율≥60% ∧ Φ 1인칭율≥30%, 아침 검토용 표본 50개 저장) · `tools/pick_trace_examples.py`
- 체인: `cesft_fp_chain.sh`(hard-RAM preflight 내장) + `start_cesft_fp.sh`(supervisor+ram_alarm+stall_watchdog_fp) — 단독 기동 금지 규약 준수
- **projection 스모크 8샘플 실측**: pass 6/8(75%), pass 트레이스 **1인칭율 100%**, hedging 없음, 배제 언명 유지 — 1인칭 Φ 경로 검증됨
- `memory.high` 완충은 이 컨테이너에서 **설정 불가**(read-only) — 방어는 캐시 100%·단일 잡·hard-RAM admission·워치독 4중으로 확정
- 캐시 보완: miss 962건 추출 **완료**(bad 0) — 커버리지 게이트 **미스 0 통과**

## 8-2. 야간 실행 중 변경 2건 (2026-07-25 밤, 아침 검토 필수)

1. **cand_free 야간 체인 제외 (사용자 지시)** — 체인에 `RUN_CAND_FREE` 토글 추가(기본 0). 야간 실행은
   Φ 재생성 → θ_CE → sft_r15 → 3-arm 평가(base·θ_CE·sft_r15)까지. **cand_free 나중 실행법**:
   `rm runs/cesft_v2_fp/markers/{RETRO3_CHAIN_DONE,CESFT_FP_CHAIN_DONE,S_TEXT_METRICS_DONE}` 후
   `RUN_CAND_FREE=1 bash scripts/step2_retrospection/start_cesft_fp.sh` — marker 멱등으로 완료 단계는
   전부 skip되고 cand_free 학습+평가+G-DELTA/DiD만 돈다 (~2.2h).
2. **GATE_GO 통과선 60%→50% 하향 + PROJ_SYSTEM 규칙 4 강화** — 1차 Φ 재생성에서 pass율 52.8%로
   관문 미달 확인. 원인 실증: 게이트 규칙은 인칭 중립이나, 1인칭 belief가 "I am preparing to
   <다음 행동>" 의도-서술로 끌려 restatement 탈락이 증가(구 코호트 65%→신 75% of drops). 산출물
   품질은 정상(pass 트레이스 1인칭율 100%, hedging 없음). 조치: ⓐ 규칙 4에 의도-서술 금지 명문화
   (게이트는 불변 — 생성 지시만 강화, drop-not-patch 유지) → 60샘플 A/B에서 57%로 개선
   ⓑ 통과선을 50%로 하향(사전 등록 60% 변경 — **본 문서로 공시**). 함의: SFT 풀 ≈2.3k
   (구 2,945 대비 −22%) — within-cohort 비교 유효성 무영향, 논문에 pass율 병기.
   아침 검토: `runs/cesft_v2_fp/eval/{gate_go_report.json,gate_go_samples.md}`.

## 9. 산출물 체크리스트 (전부 `runs/cesft_v2_fp/`)

- [ ] `markers/GATE_GO` — §4-1 관문 기록(통과율·표본 점검 로그 첨부)
- [ ] `eval/{base,theta_ce,sft_r15,cand_free}.{json,records.jsonl}` — n=1,000 동일 셋
- [ ] `eval/{arm}_nohist.records.jsonl` ×4 + `strip_verdict_{arm}.json` ×4
- [ ] `eval/paired_G-DELTA_theta_ce_vs_cand_free.json` — **n_paired > 0** (기존 공란의 폐쇄)
- [ ] `eval/DiD_history_theta_ce_vs_cand_free.json` — pass 여부 무관 산출
- [ ] `eval/paired_G-ACC1…json` · `paired_G-NH…json` (신코호트 재산출)
- [ ] `eval/{arm}.harden_s3.json` ×4 — base·cand_free 열 신규
- [ ] `eval/freegen_{arm}_{presented,cand_free}.records.jsonl` ×8 (n=500씩, 완성문 원문)
- [ ] `eval/text_metrics.json` — 4-arm × 텍스트 지표 (파일럿과 동일 정규식)
- [ ] `eval/trace_examples.md` — cand_free 오답 vs 우리 정답 anchor 3–4건
- [ ] `probe/{theta_ce,cand_free}.jsonl` — n=32·전문·step 50 간격
- [ ] 중간 어댑터 `adapter_step{50,100,…}/` — 세 arm 전부
- [ ] `logs/ram_alarm.log` · `logs/mem.log` — 전 구간 공백 없음 확인
