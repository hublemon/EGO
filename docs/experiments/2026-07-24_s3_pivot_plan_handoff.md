# EGO Step-2 방향 전환 — S3(reasoning 인과성) spine & 굳히기 계획 Handoff

- 작성: 2026-07-24 KST · EGO_jihun3
- 결정: **acc 트랙(S1/S2·DPO) 종료, S3(reasoning 인과성)로 spine 전환.** 새 9h speculative
  학습 없이, 이미 확보된 개입③ 효과를 eval 위주로 **굳혀 살았나/죽었나만 본다.**
- 선행 SSOT: `2026-07-24_wm_boundary_precheck_results_handoff.md` (precheck 3종)
- 실행 체인: `scripts/step2_retrospection/s3_harden_chain.sh` (무인, ssh-safe)
- 아티팩트: precheck https://claude.ai/code/artifact/576d19f8-6830-4f46-a76b-4fbfef3e0e54
  · retro3 결과 https://claude.ai/code/artifact/85926412-4c24-4c75-bcec-84c6c77dc3bd

---

## 0. 왜 전환하나 (증거)

| 축 | 실측 | 판정 |
|---|---|---|
| SFT acc\|cov | 0.223→0.234 (+1.1pp, ±1.3pp) | 무의미 |
| DPO | 문체학습 → collapse(margin +70) → 클램핑 필요 | 실패 |
| precheck F | candidate-free LM 0.123 ≪ WM 0.246, 융합 이득 0 | VLM이 WM 못 넘음 |
| precheck M_K | base 이미 0.43 구별, 학습이 못 올림 | S2 헤드룸 작음 |
| **개입③ (S3)** | causal_sensitivity **0.073→0.387** (swap flip 0.44 vs para 0.05) | **유일한 큰 양성** |

결론: acc로는 9h를 태워도 유의 효과 난망(≈75% 음성 예측). 반면 S3 효과는 **크고(9배 분리)
대조(paraphrase)가 이미 통과** → replication 생존 확률 높음(≈60~70%). 그리고 그 효과는
**이미 돌린 r1_sft로 측정돼 있어 새 학습이 불필요**하다.

## 1. 주장의 두 층위

- **약한 층(faithfulness)**: 행동이 verbalized reasoning에 인과적으로 결합 — swap하면 따라감.
  회의론: "SFT가 action을 reasoning 뒤에 붙였으니 기계적."  방어: base 동일포맷인데 0.08뿐 → 학습된 성질.
- **강한 층(utility)**: 그 인과 채널이 **유용** — own reasoning > empty > contradictory (p_gt 순서).
  이게 유지되면 "검증가능한 reasoning에 정렬"이라는 spine 성립. 안 되면 기계적 faithfulness로 격하.
  ⚠️ retro3 n=300에서 own vs empty 마진 얇음(0.006) → **n=1000·CI가 이 층의 생사 판정.**

## 2. 굳히기 eval 팩 (`eval/harden_s3.py`)

기존 intervention.py 확장. 각 arm의 battery records(gt_in_support·belief·reasoning 有)에서:

1. **n=1000 + bootstrap CI(2000 resample)** — 모든 flip율·p_gt·causal에 95% CI.
2. **필드 분해** — swap을 belief-only / reasoning-only / both로 나눠 어느 필드가 인과를 나르나.
3. **acc 직교성** — 정답/오답 샘플별 flip율 (causal이 acc와 독립 축인지).
4. **유용성 CI** — own p_gt − swap_both p_gt 의 CI 하한.

**사전등록 게이트 (멈춤 기준):**
- **G-S3a** (인과 실재): `causal_sensitivity(both−para)` CI 하한 > 0
- **G-S3b** (유용성 실재): `own − swap_both` p_gt CI 하한 > 0
- 둘 다 PASS → **논문 spine 확정.**  하나라도 FAIL → **라인 종료, 더 안 태움.**

## 3. 무인 체인 단계 (`s3_harden_chain.sh`)

자기복구 supervisor + 메모리 워치독 내장. marker resume. `runs/s3harden/markers/`.

| # | 단계 | RUNS | 성격 | 예상 |
|---|---|---|---|---|
| 1 | 강화: base | retro3 | eval만 (records 존재) | ~25m |
| 2 | 강화: r1_sft | retro3 | eval만 | ~25m |
| — | **S3H_DECISIVE_DONE** — 여기서 게이트 판정 나옴 | | | ~50m |
| 3 | 배터리: base | retro4 | GPU | ~25m |
| 4 | SFT: r1_sft_r4 (Phase-1 prior 재현) | retro4 | GPU 학습 | ~2.2h |
| 5 | 배터리: r1_sft_r4 | retro4 | GPU | ~25m |
| 6 | 강화: base / r1_sft_r4 | retro4 | eval | ~50m |
| | **S3HARDEN_CHAIN_DONE** | | | ~4.5h |

1~2가 **결정적**(retro3 r1_sft의 게이트), 3~6은 **다른 prior(Phase-1 K8)에서 재현 확증**.

## 4. 분기 (게이트 후)

- **G-S3a ∧ G-S3b PASS (retro3) + retro4 재현**: spine 확정. method =
  "WM이 집합 정의 → VLM이 그 위에서 **인과적·유용한 reasoning** 생성(개입③)". acc는 WM/fusion 위임.
  full confirmatory(fresh heldout·untouched test·belief vs reasoning 기여)로 승격.
- **G-S3a PASS, G-S3b FAIL**: faithfulness는 있으나 유용성 미확정 → 격하 보고, spine 재고.
- **G-S3a FAIL**: 인과 효과가 n=1000에서 사라짐 → **이 라인 종료.** 더 태우지 않음(사용자 확정 원칙).

## 5. 재현 / 운영

```bash
# 기동 (ssh 끊겨도 진행)
bash scripts/step2_retrospection/start_s3harden.sh
# 상태
cat runs/s3harden/markers/*        # 단계 완료
tail -f runs/s3harden/logs/chain.log
python3 -c "import json;print(json.load(open('runs/retro3/eval/r1_sft.harden_s3.json'))['verdict'])"
# 개별 재실행
RETRO3_RUNS=runs/retro3 PYTHONPATH=src PY=.../eve-cu124/bin/python \
  $PY -m ego.step2_retrospection.eval.harden_s3 --arm r1_sft \
  --adapter outputs/step2_retrospection/r1_sft/adapter --n 1000
```

한계: 개입③은 텍스트(candidate-scoring) 기반 — precheck와 동일 계보. n=1000으로 ±CI 축소.
retro4 재현은 Phase-1 prior(다른 시간계약·다른 support)라 절대수치는 retro3와 다를 수 있음(방향 확인용).

## 6. 산출물 위치

- 게이트 결과: `runs/{retro3,retro4}/eval/{arm}.harden_s3.json` (verdict·CI·필드분해·직교성)
- SFT 재현: `outputs/step2_retrospection/retro4/r1_sft_r4/adapter`
- 이전 트랙 보존: `dpo_d1_g3abort`(문체학습) · `dpo_d1_fix`(collapse, margin +70) — 실패 사례 아카이브
