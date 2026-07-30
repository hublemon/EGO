# cesft 재실행 계획 — 1인칭 프롬프트 일원화 + 대조군(G-DELTA) 공백 폐쇄 Handoff

> 작성: 2026-07-25 KST · EGO_jihun3 / runs/cesft_v2 후속.
> **성격**: 실행 계획 SSOT. 확정 결과는 [[2026-07-25_cesft_v2_quantitative_evidence_handoff]] ·
> [[2026-07-25_paper_capability_evidence_crosscohort_handoff]], 화법 분석은
> `EGO_jihun/docs/experiments/2026-07-25_first_person_pronoun_erosion_candidate_vs_gt_ce_handoff.md` (rev3).

---

## 0. 세 줄 요약

1. **재실행의 주목적은 1인칭이 아니라 G-DELTA다.** 본셋에는 candidate-free 대조군이 **없다** — `paired_G-DELTA_theta_ce_vs_cand_free.json`이 `"error": "no covered records for arm_b='cand_free'"`로 공란. 논문 중심 주장("WM 후보 경계로 학습해야 판별기가 생긴다")이 현재 **자매 코호트(EGO_jihun) 근거에만 의존**한다.
2. **1인칭 프롬프트 일원화는 그 재실행에 얹는 부수 이득**이고, 편집 대상은 3곳(정책·cand_free·투영)뿐이며 teacher는 손대지 않는다. 파일럿 선례상 판별 성능 손상 없이 1인칭 강화가 관측된 조합이다.
3. **두 목적은 분리 가능하고, G-DELTA 폐쇄는 단일 서버로 ~1.7h면 끝난다** (cand_free 학습 0.94h 실측 + 평가 0.7h). 프롬프트 일원화 재실행(~11.7h 직렬)은 그 다음 선택지다. — §6

> **주요 정정 (본 문서 내)**: ① cand_free 학습비용 3.7h → **0.94h** (실측 0.805 s/sample, §2-3) ② cesft 경로는 **LLM judge를 쓰지 않음** — semantic gate 0.5h·base trace 재생성 2.4h 항목 삭제 (§4-4).

---

## 1. 왜 재실행인가 — 현행 산출물의 공백 (대시보드 대조)

| # | 공백 | 현재 상태 | 재실행 시 확보 | 우선순위 |
|---|---|---|---|---|
| 1 | **G-DELTA (θ_CE vs cand_free)** | **공란** — arm 미존재 | 논문 중심 주장의 본셋 근거 | **최상** |
| 2 | **history 사용 이중 해리(DiD)** | 본셋은 θ_CE 단독 strip(+3.1pp [1.1, 5.2])만. **DiD 없음** | cand_free strip 추가 → 본셋 DiD 산출 | **최상** |
| 3 | boundary 효과 vs 객관식 효과 분리 | random_cand arm 미존재 (probe step 0만) | 사전 등록된 성립부등식 2개 완성 | 상 |
| 4 | **in_support (경계 내재화)** | 미측정 — freegen 패스 없음 | crosscohort가 egocentric 대신 쓰라고 권고한 **창발 서사 지표** | 상 |
| 5 | Base 모집단 불일치 | Base n=1,000 covered-only vs 학습 arm n=5,326 | 3 arm 동일 셋 재평가 | 상 |
| 6 | Base·θ_CE belief 개입 | 미측정 (SFT 3 arm만) | "belief 인과는 SFT 산물" 3점 곡선 완성 | 중 |
| 7 | **스텝별 곡선** | 중간 체크포인트 없음 | probe 확대 + ckpt 보존으로 시간축 확보 | 중 |
| 8 | 1인칭율 (2레짐) | 전 arm 0% (템플릿 함수) | 프롬프트 일원화로 측정 가능해짐 | 중 |
| 9 | malformed 상승 (1.1→4.1%) | 원인 미규명 | 프롬프트 통일 후 재확인 | 하 |

→ 재실행은 "1인칭 하나를 위한 10시간"이 아니라 **①②를 메우는 실행에 ③–⑨가 따라오는 구조**다.

---

## 2. 기존 candidate-free 실행이 왜 부족한가

### 2-1. 본셋(cesft_v2) 시도는 실패 후 취소됐다

- `outputs/step2_retrospection/cesft_v2/cand_free/`: **어댑터 없음**, `train_log.jsonl`만 존재.
- 로그 2,903행 중 **2,856행이 `{"skip_decode": ...}`** — 프레임 디코드 실패. 실제 학습 스텝은 **58**에서 멈춤.
- 원인: 프레임 추출(`FRAME_EXTRACT_RUNNING` 마커)과 동시 기동 → 캐시 미완 샘플 대량 skip. **설계 결함이 아니라 인프라 타이밍 문제** → 프레임이 캐시된 지금 재실행하면 해소된다.
- 이후 `runs/cesft_v2/markers/S_CE_CAND_FREE_DONE`이 hook에 의해 스킵 마커로 기록됨: `{"skipped_by":"post_theta_hook","reason":"EGO_jihun 성립부등식 확정결과로 대체(§2)"}` — 즉 **시간 절약을 위해 자매 코호트 결과로 대체하기로 한 의도적 결정**이었다.

### 2-2. 자매 코호트(EGO_jihun) 결과로 대체할 수 없는 이유

파일럿의 **Q arm이 곧 candidate-free GT-CE**이고 결과도 강력하다 (제시 평가에서 cand_q 13.3% vs cand_c 23.2%, DiD +8.4pp). 그럼에도 본셋 대체가 불가한 이유:

1. **크로스코호트 비교 금지 판정이 이미 있다** — [[2026-07-25_paper_capability_evidence_crosscohort_handoff]] §2-4가 표본 불일치 기반 비교(+2.4pp/+19.2pp)를 **인용 금지**로 강등했다. 같은 논리가 "본셋 θ_CE vs 파일럿 Q"에도 적용된다.
2. **조건이 다르다** — 시간 계약(start−1s vs end−1s), K(5 vs 10), 데이터 분할, 프롬프트 템플릿, 학습 예산이 모두 다르다.
3. **paired 검정이 불가능** — G-DELTA는 video-cluster paired bootstrap인데 두 코호트는 sample_id 교집합이 없다.
4. 논문 표에 "이 수치만 다른 실행에서"가 들어가면 리뷰어 표적이 된다.

**결론**: 대조군은 **같은 코호트·같은 샘플·같은 예산**으로 다시 학습해야 한다.

### 2-3. cand_free의 실제 학습 비용 — 0.9h (초안의 3.7h는 오류)

중단 전 47스텝의 실측 `sec` 평균 = **0.805 s/sample** (`cand_free/train_log.jsonl`). θ_CE의 3.23 s/sample 대비 **4배 빠르다** — cand_free는 후보 10개 span의 logprob을 채점하지 않고 GT span 하나만 forward하기 때문(`ARMS["cand_free"] = ("free", True, True, "gt")`).

→ 4,189 샘플 환산 **0.94h**. 본 문서 초안이 θ_CE 레이트를 잘못 적용해 3.7h로 적었던 것을 정정한다. (그 47스텝은 프레임 추출과 경합하던 구간이므로, 이 값은 오히려 보수적이다.)

**공정성**: equal-budget은 **시간이 아니라 step 수**로 맞춘다 (`--max_steps`). cand_free가 빠른 것은 손해가 아니라 공짜 이득이다.

---

## 3. 대조군 설계 — candidate-free GT-CE는 적절한가

### 3-1. 적절하다. 단 단독으로는 부족하다

`select_ce.py:46-52`의 ARMS 프리셋이 이미 사전 등록된 3분할을 구현하고 있다:

| arm | 구성 | 분리하는 효과 |
|---|---|---|
| `wm_cand` (θ_CE) | WM Top-K 후보 + selection CE | (본 방법) |
| `cand_free` | 후보 無 + **순수 GT-span CE** | **GT CE 자체 효과** |
| `random_cand` | GT + 무작위 K−1 후보 + selection CE | **객관식(multiple-choice) 효과** |

- `cand_free`만 두면 "θ_CE > cand_free"가 나와도 그것이 **WM 경계의 공이 아니라 단지 객관식 포맷의 공**일 가능성을 못 배제한다.
- `random_cand`가 있어야 **WM 경계 효과 = (θ_CE − random_cand)**로 분리된다. 이것이 [[2026-07-24_ce_sft_methodology_v2_handoff]] §3의 사전 등록 설계다.

### 3-2. 기대할 수 있는 것 (파일럿 실측 기반 예측)

| 지표 | 파일럿 Q(=cand_free 등가) 실측 | 본셋 기대 |
|---|---|---|
| 후보 **제시** 정확도 | 13.3% vs C 23.2% (n=1,520) | **cand_free 큰 폭 열세** — G-DELTA PASS 기대 |
| 후보 **비제시** 자유생성 정확도 | 20.8% vs C 22.0% (비유의) | **차이 없음** 기대 — "후보 제시 자체가 정확도를 올리진 않는다"의 본셋 재확인 |
| history 사용 (strip Δ) | +4.2pp vs C +12.6pp | **cand_free Δ가 작음** → 본셋 DiD 양수 기대 |
| 1인칭 (자유생성) | 21.2% vs C 31.6% | cand_free가 더 침식 |

### 3-3. 반드시 검증해야 하는 것 (게이트로 사전 등록)

| 게이트 | 정의 | 통과 기준 | 실패 시 함의 |
|---|---|---|---|
| **G-DELTA-1** | SelAcc(θ_CE) − SelAcc(cand_free), covered·paired | CI 하한 > 0 | 실패 시 "WM 후보 학습이 필요하다"는 **논문 중심 주장 철회** |
| **G-DELTA-2** | SelAcc(θ_CE) − SelAcc(random_cand) | CI 하한 > 0 | 실패 시 효과는 WM 경계가 아니라 **객관식 포맷**의 것 → 서사 전면 수정 |
| **G-DiD** | strip Δacc(θ_CE) − strip Δacc(cand_free) | CI 하한 > 0 | 실패 시 "판별 압력이 history 사용을 가르친다" 본셋 미재현 (자매 코호트 한정 서술로 후퇴) |
| G-EQ | 두 대조군의 학습 예산 동일성 | 같은 subset·steps·lr·seed | 위반 시 전 게이트 무효 |

**공정성 필수 조건**: `--subset_file`·`--max_steps`·`--seed 42`를 세 arm 동일하게 고정한다 (`--max_steps`가 equal-budget 통제용으로 이미 존재). cand_free는 후보 인코딩이 없어 샘플당 약간 빠르므로 **step 수로 맞추고 시간으로 맞추지 않는다.**

---

## 4. 프롬프트 1인칭 일원화 — 편집 대상과 손대지 말 것

### 4-1. 편집 대상 3곳

| # | 위치 | 현재(관찰자) | 변경 |
|---|---|---|---|
| 1 | `vlm.py:51` `SYSTEM_PROMPT` | "You are an egocentric activity **assistant** … **the person** already COMPLETED … what **the person** does next" | 행위자 프레임 ("You are … reasoning about **your own** ongoing activity … what **you** do next") |
| 2 | `train/select_ce.py:55` `SYS_NOCAND` | 동일 문구 (cand_free arm 전용) | 동일 규칙으로 통일 |
| 3 | `hindsight/projection.py:25` `PROJ_SYSTEM` | "Write what a careful **observer** could have concluded … **the person's** completed actions" | 1인칭 회고 서술로 변경 (**SFT 타깃 문체의 실제 출처**) |

참조 원본: `EGO_jihun/src/ego/step2_vlm_alignment/train_grpo_action.py:157` (파일럿 행위자 프레임 — 1인칭 52–74%가 나온 템플릿).

### 4-2. 손대지 않을 것 2곳 (근거 있음)

- **`hindsight/teacher.py:11` TEACHER_SYSTEM** — Ψ는 JSON 필드(`activity/stage/completed_subgoal/…`)만 반환하며 헤더 주석대로 "trace에 그대로 들어가지 않는다". 문체 중립 → 변경 불필요.
- **`vlm.fmt_history` (`- verb noun` 비인칭 리스트)** — 침식의 원인 물질이지만 **건드리지 않는다**. 이유: ① 파일럿도 항목은 동일한 비인칭 리스트였고 헤더만 행위자 프레임("Your recent action history")이었는데 1인칭 60.8%가 나왔다 → **페르소나 프롬프트만으로 충분함이 실측됨**. ② 이 포맷을 바꾸면 history 사용 지표(strip Δ, DiD)의 비교 기반이 흔들린다. 리스트 포맷의 인과 효과는 별도 ablation 과제(rev3 §9-4)로 남긴다.

### 4-3. 예상 문제와 사전 점검

| 리스크 | 점검 방법 | 중단 기준 |
|---|---|---|
| 규칙 게이트 통과율 변동 (현행 4,189→2,945 = 70.3%) | Φ 재생성 직후 통과율 계산 | **60% 미만이면 중단** |
| hedging("I think…") 유입으로 근거 서술 희석 | 표본 50개 육안 + `scene_desc_rate`·`avg_words` 대조 | 정성 판단 |
| CE arm 영향 | selection CE는 후보 span logprob — 페르소나 무관 | 저위험 |

**semantic judge 리스크는 해당 없음** — §4-4 참조. cesft 경로는 judge를 쓰지 않는다.

### 4-4. LLM judge(Gemini) 사용 이력 — cesft_v2는 **미사용**

| 실행 | judge 호출 | 상태 |
|---|---:|---|
| retro3 | **3,187건** | 완료 (`S4_semantic.json` done, `judge_errors: 0`, 1,788s @ 1.78/s) |
| retro4 | **747건** | 중단 (status는 418/2909에서 `running`으로 정지) |
| **cesft_v2** | **0건** | **S4_semantic 스테이지 자체가 없음** |

- judge 모델: `gemini-2.5-pro`, 엔드포인트 `https://gw.letsur.ai/v1` (`hindsight/semantic_gate.py:23-24`). 총 호출 ≈ **3,934건**.
- cesft_v2의 `data/chosen_train.jsonl`은 **retro4 산출물과 byte-identical**(md5 `b1e0851b…` 일치)이며, 게이트 값은 규칙 게이트의 `pass` 2,945 / `drop` 1,244뿐이다.
- `semantic_train.jsonl`의 유일한 소비처는 `pairs/build_pairs.py`(DPO 페어 구성)이고, cesft는 CE+SFT 경로라 이를 타지 않는다.
- **함의**: 재실행에서도 judge 스테이지는 불필요하다. 초안의 "semantic gate 0.5h"와 "base trace 재생성 +2.4h(judge 문체 정합용)"는 **모두 삭제**한다.

---

## 5. 이번 실행에서 함께 확보할 지표 (계측 설계)

### 5-1. 학습 중 (추가 비용 ≈ 0)

- **`--probe_every 50`** (현행 100) + **probe 샘플 8 → 32로 확대** + `probe_gen.py:76`의 `reasoning_head` 260자 절단 해제(전문 저장). → 스텝별 1인칭율·malformed·probe_acc 곡선을 n=32로 확보. 비용: 10회 × 32샘플 × 1.2s ≈ **7분**.
- **`--ckpt_every`로 저장되는 중간 체크포인트를 step 태그로 보존** (현행은 resume용 롤링 저장이라 덮어씀). step 100/200/300/400에 어댑터 사본 → 사후 strip·freegen 재평가 가능.

### 5-2. 학습 후 평가 패스

| 패스 | 내용 | 시간 | 메우는 공백 |
|---|---|---:|---|
| battery ×5 arm (n=1,000 동일 셋) | base·θ_CE·SFT·cand_free·random_cand | 1.7h | ①③⑤ |
| strip 평가 ×2 (θ_CE, cand_free) | no-history paired | 0.7h | ② |
| **freegen 2레짐 ×3 arm (n=500)** | 제시/비제시 × base·θ_CE·SFT | 1.0h | ④⑧ |
| harden ×3 (base·θ_CE·SFT, n=400) | belief 개입 | 0.5h | ⑥ |
| paired 게이트 일체 | G-DELTA-1/2, G-DiD, G-ACC1, G-NH | 수 분 | — |

합계 ≈ **3.9h** (2서버 분할 시 ~2h/서버).

---

## 6. 실행 계획 — 단일 서버 (2026-07-25 재작성)

### 6-0. 핵심 판단: 두 목적은 **분리 가능**하다

| 목적 | 필요한 것 | 프롬프트 변경 필요? |
|---|---|---|
| **G-DELTA 공백 폐쇄** (논문 중심 주장) | cand_free arm 1개 + 그 평가 | **불필요** — 기존 θ_CE(현행 프롬프트)와 비교하면 된다 |
| 1인칭 지표 확보 | 프롬프트 3곳 변경 + **전 arm 재학습** | 필요 |

기존 θ_CE·SFT·base 산출물이 모두 현행 프롬프트로 정합되어 있으므로, **대조군만 현행 프롬프트로 학습하면 G-DELTA는 즉시 닫힌다.** 여기에 프롬프트 변경을 섞으면 기존 산출물 전체를 재생산해야 한다.

### 6-1. 단계 P1 — G-DELTA 폐쇄 (권장, 최우선)

프롬프트 **변경 없이**, 현행 θ_CE와 짝을 이루는 대조군만 학습·평가한다.

| 단계 | 시간 | 산출 |
|---|---:|---|
| cand_free CE 학습 (4,189, equal steps) | **0.94h** | 대조군 어댑터 |
| battery cand_free (n=1,000, 후보 제시) | 0.35h | G-DELTA-1 입력 |
| battery cand_free no-history (n=1,000) | 0.35h | G-DiD 입력 (θ_CE strip은 기존 `strip_verdict.json` 재사용) |
| paired 게이트 2종 | 수 분 | G-DELTA-1, G-DiD |
| **소계** | **≈ 1.7h** | **논문 중심 주장의 본셋 근거** |

(+ `random_cand`를 추가하면 학습 3.8h + 평가 0.35h = **+4.2h**. 객관식 효과 분리가 필요할 때만.)

### 6-2. 단계 P2 — 1인칭 프롬프트 일원화 재실행 (선택, 단일 서버)

```
Φ 재생성·관문 0.5h → θ_CE 3.8h → SFT 2.6h → cand_free 0.9h → 평가 3.9h
                                                        합계 ≈ 11.7h (직렬)
```

평가 n을 1,000으로 고정하고 freegen을 2레짐×3arm으로 한정한 값. **P1과 P2는 산출물이 서로 다른 코호트가 되므로 한 표에 섞지 않는다** (§8).

### 6-3. 시간이 없을 때의 권고 순서

1. **P1 (1.7h)** — 이것만으로 논문의 가장 큰 공백이 닫힌다. 무조건 먼저.
2. random_cand (+4.2h) — G-DELTA-2가 필요하다고 판단되면.
3. P2 (11.7h) — 1인칭은 각주/appendix 소재이고 crosscohort가 이미 "본문 능력지표 사용 금지"로 판정했다. **논문 마감이 임박하면 생략하고 "future work"로 서술하는 것이 합리적이다.**

기동은 CLAUDE.md 규약대로 `setsid` 분리 + 마커 멱등 + PPID=1 확인.

---

## 7. 착수 전 1시간 관문 (Go/No-Go)

1. 프롬프트 3곳 수정 → `PROJ_SYSTEM`으로 Φ 재생성 (0.5h).
2. 규칙 게이트 통과율 확인 — **≥60%** (현행 70.3%).
3. 신규 trace 50개 육안 + `first_person_rate`·`avg_words`·`scene_desc_rate` 계산.
4. 세 값이 정상이면 학습 착수. 아니면 **원인 규명까지 중단** — 9.2h를 걸기 전 여기서 되돌린다.

프레임 캐시 상태도 함께 확인한다 (cand_free 실패 원인이었던 skip_decode 재발 방지 — 첫 200스텝에서 skip 비율 5% 초과 시 즉시 중단).

---

## 8. 정직 규칙 (논문 반영 시)

- **코호트 분리 유지** — 이번 실행은 프롬프트가 바뀌므로 **기존 cesft_v2 수치(G-ACC1 +4.8pp 등)와 한 표에 섞지 않는다**. 새 실행이 성공하면 전면 교체, 실패하면 기존 유지.
- 1인칭율은 여전히 **레짐·템플릿 종속 표면 지표** — 같은 템플릿 내 arm 비교로만, 각주 공시 필수 (crosscohort §2-6 판정 유효).
- G-DELTA 실패 시 논문 중심 주장을 **철회**한다. 이 게이트는 사후 해석 대상이 아니라 사전 등록 판정이다.
- 스텝별 곡선은 n=32 probe — CI 산출 불가, 방향성 서사 보조로만.

---

## 9. 산출물 체크리스트

- [ ] `runs/cesft_v2_fp/eval/paired_G-DELTA_theta_ce_vs_cand_free.json` — **n_paired > 0**
- [ ] `paired_G-DELTA_theta_ce_vs_random_cand.json`
- [ ] `strip_verdict_{theta_ce,cand_free}.json` + DiD 산출
- [ ] `freegen_{presented,candfree}_{base,theta_ce,sft_r15}.records.jsonl` (n=500씩)
- [ ] `harden_s3` base·θ_CE 추가분
- [ ] `probe/*.jsonl` — n=32, 전문 reasoning, step 50 간격
- [ ] 중간 어댑터 step 100/200/300/400
- [ ] 게이트 통과율·문체 점검 로그 (§7 관문 기록)
