# projected-SFT의 History 사용 — 실측 검증 & 연구 의의 연결 Handoff

> 작성: 2026-07-24 KST · EGO_jihun3. **지위: 실측 검증 완료 (retro4 eval 재분석) + 의의 연결.**
> 검증 대상: dpo_free 확정 방법론의 ② projected-trace SFT.
> 근거 파일: `runs/retro4/eval/r1_sft_r4.records.jsonl`(SFT, n=480) · `runs/retro4/eval/base.records.jsonl`(base, n=998)
> 인과 지표 출처: `2026-07-24_dpo_free_ce_sft_methodology_handoff.md` §3(causal_belief·utility)
> 논문: `EGO_paper/EGO_AAAI27_EN/main.tex`(피벗본) — 제목 "The Limits of Your World Model Mean the Limits of Your Language Model"

---

## 0. 세 줄 요약

1. **검증 질문**: projected-SFT가 action history를 *유의미하게*, 그리고 *어떤 방식으로* 쓰는가?
2. **결과**: 쓴다. reasoning의 history 참조율이 base 72.3% → SFT **85.8%**(+13.5pp)로 오르고,
   사용 방식은 **"직전 완료행동 → 현재 국면 추정 → task_belief → 다음 행동"**의 progression 추론이다.
   실제 로그에서 이 추론이 WM top-1을 **이기고** 정답을 고른 사례도 확인된다.
3. **단, 정직하게**: history는 **belief를 잘 만드는 쪽**에 실리고 **action 전이로는 아직 덜** 실린다
   (belief 맞고 action 틀림 = 83/480, 17%). 그래서 acc는 거의 안 오른다(0.200→0.215).
   이 간극이 곧 논문의 "belief-conditioned selection" 주장을 **실증하면서 동시에 숙제로 남긴다**.

---

## 1. 무엇을 검증했나

confirmed 방법론 ② projected-SFT는 teacher가 만든 이상적 trace $(r_{\text{proj}}, g_{\text{proj}}, a_{\text{GT}})$를
재현하도록 학습한다. 이 trace는 설계상 **"시점 $t$에서 관찰 가능한 근거($x_{\leq t}$ 또는 $H_{<t}$)만 써야"**
한다(논문 projection 제약 1). 그래서 물었다: **모델이 실제로 $H_{<t}$(완료 행동 이력)를 읽고 쓰는가,
아니면 프레임만 보고 서술을 지어내는가?**

## 2. 실측 결과

### 2.1 History를 얼마나 참조하나 (reasoning 텍스트 기준)

| 측정 | projected-SFT | base | Δ |
|---|---|---|---|
| history 참조(넓은 cue: recent/previous/just/so far/continu…) | **412/480 = 85.8%** | 722/998 = 72.3% | **+13.5pp** |
| 강한 마커(just finished / recent history / transition from…) | 344/480 = 71.7% | 688/998 = 68.9% | +2.8pp |
| strict GT acc | 0.215 | 0.200 | +1.5pp |

→ SFT가 **명시적 history-grounding을 유의하게 늘린다**. 단, base(같은 VLM)도 원래 서술을 많이 해서
강한 마커 격차는 작다. 그리고 **참조율↑이 곧 acc↑는 아니다**(§4).

### 2.2 실제 추론 로그 — "어떻게" 쓰는지가 보인다

**(A) 정답 사례 — history가 WM top-1을 이김** `[..._53]` gt=stir dish, pred=**stir dish ✓**, wm_top1=check heat
> reasoning: *"The person has **just finished adding oil to the pan** and is now moving the pan to the burner… Among the candidates, only 'stir dish' directly aligns with the current state of having a pan on the heat."*
> belief: *"about to combine or mix ingredients."*

→ 직전 완료행동(add oil)을 읽어 "지금은 가열 시작 국면"이라 추론 → belief 고정 → **WM top-1(check heat)을 제치고** stir dish 선택. 모방을 넘어선 선택(GADR형 승리).

**(B) belief는 맞는데 action이 샌 사례** `[..._7]` gt=cut potato, pred=**wash knife ✗**, wm_top1=cut potato
> reasoning: *"The person has **just finished peeling the potato**… the next step involves preparing the potato for cooking."*
> belief: *"preparing to **cut** the potato after peeling it."* ← 국면 정확

→ history로 **올바른 국면(cut)**을 짚었는데 최종 action은 wash knife로 이탈. **belief→action 전이 실패.**

## 3. 어떻게 쓰는가 (메커니즘 한 줄)

**최근 완료행동($H_{<t}$) → 현재 task 국면 추정 → task_belief 형성 → 그 국면에 맞는 다음 행동 선택.**
이는 우연이 아니라 설계 의도의 재현이다 — projection 제약이 "reasoning의 evidence는 $x_{\leq t}$·$H_{<t}$에서
관찰 가능"을 강제하므로, SFT 타깃 자체가 history-근거 서술이고 모델이 이를 학습했다.

## 4. 한계 · 정직성 (보고 시 병기)

1. **belief는 잘, action은 덜**: `wm_top1=GT(정답이 후보에 있음)인데도 오답 = 83/480 (17%)`.
   history가 국면을 맞게 짚어도 최종 선택이 어긋나는 케이스. = dpo_free §3.4의 **"제어 강·유용 약"**
   (causal_belief 0.058→**0.390** 상승 / utility 0.108→0.067 하락)과 같은 현상. history는 belief 채널을
   살찌우지만 그 belief가 GT를 가리키는 힘(utility)은 아직 약하다.
2. **직접 history-ablation 미실행**: 위 "유의미"는 (a) 참조율 +13.5pp와 (b) history가 먹이는 belief 채널의
   인과 무게(0.39, harden_s3)로 **간접 입증**한 것. history를 끈 통제실험 수치는 아직 없다.
3. **경계 사례**: EK100(다른 셋업)에서는 명시적 task_history가 오히려 acc를 깎았다(goal 부재로 noun 과편향).
   goalstep은 goal이 있어 반대로 도움 되는 것으로 보이나, 이 대비는 각주로 남길 것.

## 5. 연구 의의와의 연결 — 이 검증이 논문의 무엇을 떠받치나

| 실측 | 뒷받침하는 논문 주장 | 위치 |
|---|---|---|
| history로 국면 추정 → belief → 선택 | **"belief-conditioned selection"**: 같은 화면·같은 후보라도 **다른 history가 다른 belief를 유도해 다른 선택**을 낳는다 | §Qualitative Examples(Table trace의 Retrospection 행: "직전에 채소를 꺼냈으므로 belief는 샐러드 준비") |
| "just finished peeling → preparing to cut" 류 추론 | **Egocentric Embodiment Grounding**: 완료 history를 주어 "이전 행동으로 인한 state transition·진행 중 task progression"을 포착 | §Methodology / Egocentric Embodiment Grounding |
| history 참조가 WM top-1을 이긴 사례(A) | **모방 초과(GADR)**: WM 경계 안에서 LM이 맥락으로 재선택 = 제목의 "경계 안 selection은 LM 몫" | §GADR |
| belief 강·action 약(B, 17%) | Retrospection(projected-SFT)이 **belief→action 인과 경로를 심는** 단계라는 정의, 그리고 그 경로가 아직 acc로 완전 전이 안 됨 | §Retrospection / Projected-Trace SFT |

**제목(coverage-cap)과의 관계**: history 사용은 **selection 축**의 능력이다(경계는 WM coverage가 정하고,
경계 *안에서* 무엇을 고를지를 history-기반 belief가 좌우). 즉 이 검증은 제목의 "경계 안에서는 LM의
과업 조건부 추론이 결과를 결정한다"는 절반을 실증하는 재료다.

## 6. 다음 — 이 주장을 논문에 걸려면 (제안, 미실행)

논문 §Ablation Studies에 이미 예고된 **"remove action history"** ablation을 실제로 돌려 못을 박자:
- **arm**: (i) full($x,H,D$) vs (ii) history-drop($x,D$ / $H$ 제거) — 동일 SFT·동일 eval.
- **측정**: history 제거 시 (a) causal_belief 하락, (b) belief-conditioned flip 감소, (c) acc 변화.
- **성공 서사**: "history 제거 → belief 채널·belief-conditioned selection 붕괴" 를 보이면 §5의 간접입증이
  직접입증으로 승격. (GPU: SFT 1회 + eval, v2 CE 급 ≈ 2h 내.)
- 이건 결과검증·비교군 담당(우리) 범위 — 바로 설계·실행 가능.

## 7. 근거 파일 좌표

| 무엇 | 위치 |
|---|---|
| projected-SFT eval records (n=480) | `EGO_jihun3/runs/retro4/eval/r1_sft_r4.records.jsonl` |
| base eval records (n=998) | `EGO_jihun3/runs/retro4/eval/base.records.jsonl` |
| 인과 지표(causal_belief·utility) 도구 | `EGO_jihun3/src/ego/step2_retrospection/eval/harden_s3.py` |
| 방법론 SSOT | `docs/experiments/2026-07-24_dpo_free_ce_sft_methodology_handoff.md` |
| 논문 반영 지점 | `EGO_paper/EGO_AAAI27_EN/main.tex` §Egocentric Embodiment Grounding · §Qualitative Examples · §Ablation Studies |
