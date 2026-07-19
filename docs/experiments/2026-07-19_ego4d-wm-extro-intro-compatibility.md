# Ego4D 학습 V-JEPA2 WM ↔ extro/intro 방법론 정합성 검토 Handoff

- 작성일: 2026-07-19 (KST)
- 목적: Ego4D LTA Z=1로 학습한 V-JEPA2 WM(frozen encoder+predictor + attentive
  probe, epoch 8 채택본)을 **extro(F0)/intro(B0) 트랙에 그대로 투입해도 되는지**
  계약(contract) 단위로 검토한 결과를 전달한다.
- 관련 문서:
  - `2026-07-19_extro_intro_implementation_review_handoff.md` — extro/intro 구현 상세
  - `2026-07-17_ego4d-lta-full-training-results.md` — Ego4D WM 학습 결과 (epoch 8 채택 경위)
  - `2026-07-17_ego4d-lta-model-release-handoff.md` — 모델 배포 패키지 (release zip 내)
  - `EGO/INTERFACE_FOR_WM.md` — EK100 시절 WM↔VLM 인터페이스 명세 (비교 기준)

---

## 0. 한 줄 결론

> **개념·인터페이스 층위에서는 정합하고, 드롭인 교체가 가능하다. 일부는 EK100보다
> 방법론 취지에 오히려 더 충실하다. 단, EK100 기준으로 캘리브레이션된 정량 전제
> 3가지(후보 커버리지 / spread 게이트 / 학습 split 위생)는 그대로 옮기면 어긋나므로
> 통합 전 재검증이 필수다.**

| 검토 항목 | 판정 |
|---|---|
| extro: GT-free WM likelihood reward 계약 | ✅ 정합 |
| extro: anticipation 문제 설정 | ✅ 정합 (EK100보다 개선) |
| intro: teacher hindsight용 미래 GT 시퀀스 | ✅ 정합 (native 제공) |
| 공통: 프롬프트 빌더 / 3태그 출력 / 파싱 / 라우팅 | ✅ 정합 (드롭인) |
| 후보 커버리지 (GT∈top-5) | ⚠️ 84% → ~19%, 상한·수율 재산정 필요 |
| `--min_wm_spread` / `frac_reward_zero_std` | ⚠️ epoch 8 기준 재측정 후 재캘리브레이션 |
| GRPO 학습 split 선택 | ⚠️ train은 WM 과적합 — dev 사용 권고 |
| canonical 정규화 / 체크포인트 버전 | ⚠️ taxonomy 형식 점검, epoch 8 확인 |

---

## 1. 용어 정리 (처음 읽는 사람용)

| 용어 | 의미 |
|---|---|
| **extro (F0)** | 외부(external) 세계 신호 기반 GRPO. WM의 joint top-5 likelihood 분포를 reward로 사용. GT-free. |
| **intro (B0)** | 내부(introspective) 신호 증류. teacher가 만든 full-trace를 preference pair로 DPO. GT는 오프라인 구성에만 사용. |
| **WM** | World Model = frozen V-JEPA2 + attentive probe. 정책(VLM)에게 (verb,noun) 후보와 likelihood를 공급. |
| **Ego4D WM** | 이번에 학습한 Ego4D FHO-LTA 버전 WM. verb 116 / noun 477 / action(등록 조합) 5,698 클래스. 최종 채택 checkpoint는 **epoch 8**. |
| **coverage@5** | 화면에 보인 joint top-5 후보 안에 GT가 들어있는 비율. 정책 정확도의 이론적 상한. |

두 트랙 모두에서 WM의 역할은 동일하다: **후보셋 공급 + (extro 한정) likelihood
reward 공급**. 따라서 정합성 검토는 "Ego4D WM이 이 두 역할을 EK100 WM과 같은
계약으로 수행할 수 있는가"로 환원된다.

---

## 2. 개념 층위 — 정합 (일부는 개선)

### 2.1 extro: "GT-free 외부 신호" 계약 유지

- extro의 reward는 **WM 예측 분포 그 자체의 함수**이고 GT를 참조하지 않는다.
  Ego4D WM은 EK100 때와 동일한 frozen V-JEPA2 위에 probe만 교체한 구조라,
  WM의 "외부 세계 신호" 지위가 그대로 유지된다.
- reward가 요구하는 것은 joint (verb,noun) likelihood인데, Ego4D WM의 action
  head(5,698 조합) softmax가 `topk_actions_with_score` 형식을 그대로 생성한다.
- probe 학습에 GT가 쓰였다는 타협은 EK100과 동일한 수준 — 계약 변화 없음.

### 2.2 extro: anticipation 설정은 오히려 개선

EK100 셋업은 ap=0.0(액션 종료 1초 전 관측)이라 `INTERFACE_FOR_WM.md` §5에서
스스로 "엄밀히는 anticipation이 아니라 late-action recognition"이라고 명시해
두었다. **Ego4D LTA는 정의상 관측 구간 종료 이후의 액션을 예측하는 진짜
anticipation**이므로, extrospection의 취지(외부 세계의 미래에 대한 WM 예측
신호)에 더 충실해진다. 외부 보고 시 달아두던 ap=0.0 단서도 불필요해진다.

### 2.3 intro: 미래 GT 시퀀스가 native로 제공

intro의 teacher는 hindsight 구성에 **미래 GT action 시퀀스**(`future_gt_actions`)
를 요구한다. EK100에서는 후속 segment를 이어붙여 구성해야 했지만, Ego4D LTA는
forecasting 벤치마크라 이 시퀀스가 주석에 원래부터 존재한다.
`memory_context`(과거 액션 히스토리)도 FHO 주석 시퀀스에서 자연스럽게 나온다.
**intro의 오프라인 pair 구성 요건과 자연 정합.**

---

## 3. 인터페이스/코드 층위 — 드롭인 가능

- `build_joint_conversation`이 소비하는 것은 ① joint top-5(+score) ② 프레임
  ③ memory_context 가 전부이고, 셋 모두 Ego4D WM + `action_registry.json` +
  `fho_lta_taxonomy.json` 으로 생성 가능하다.
- score 미노출·표시 순서 셔플·`assert_no_score_leak` 등 sycophancy 차단 장치는
  데이터셋 독립적 — 그대로 유지된다.
- intro 파이프라인에서 WM은 후보셋 공급자(`gt_in_candidates` 게이트 포함)로만
  관여하므로 교체가 파이프라인 구조를 건드리지 않는다.
- 3태그 출력 계약(`<reasoning>/<task_belief>/<action>`), 파싱, routing, DPO
  학습 코드 모두 WM 교체와 무관.

**유일한 코드 조정 지점**: `canonical_action`("verb|noun" 소문자 키) 정규화가
EK100 canonical key를 가정한다. Ego4D taxonomy 텍스트 형식(언더스코어, 괄호
병기 등)을 확인하고 정규화 함수를 맞출 것.

---

## 4. 정량 전제 — 어긋나는 부분 (통합 전 재검증 필수)

### 4.1 후보 커버리지 급락 — 가장 큰 리스크

epoch 8 heldout 기준 **GT∈joint top-5 ≈ 19.4%** (micro action top-5 acc).
EK100의 cross-product 기준 84%와 비교하면 차원이 다르다.

파급 효과:

1. **정책 acc 상한이 ~19%로 캡**된다 — 5지선다 밖의 GT는 절대 못 맞힌다.
2. extro `gt_only`(skyline)의 oracle-subset과 intro의 `gt_in_candidates`
   게이트 모두 **train 62,147개 중 ~1.2만 개만 생존**할 것으로 추정된다.
   학습 자체는 가능하지만 수율이 EK100과 전혀 다르다.
3. 다만 **G2 밴드(GT∈top5 ∧ GT≠top1) ≈ 19.4 − 6.1 ≈ 13pp**가 존재하므로,
   "정책이 WM을 이길 수 있는가"라는 핵심 신호 자체는 살아 있다.

**완화책**: EK100에서 이미 검증한 **verb top-5 × noun top-5 cross-product
(25 후보)** 전환. Ego4D도 verb 57.9% / noun 54.5% (micro top-5)이므로 joint
상한을 크게 끌어올릴 수 있다. 단, 현행 extro 빌더는 joint 5지선다를 쓰므로
이 경우 빌더 변경이 필요하다 (후보 수 5→25는 프롬프트·reward 매칭 로직에도
영향 — 별도 설계 필요).

### 4.2 `--min_wm_spread 0.05` 재캘리브레이션

이 게이트(후보셋 재정규화 likelihood 표준편차 임계)와 wm_clean의
`frac_reward_zero_std ≈ 0.5` 특성은 **EK100 action head 분포에 맞춰진
값**이다. 5,698-way 롱테일 head는 spread 분포가 다를 수밖에 없다 (top-1
쏠림 또는 과도한 평탄화 모두 가능).

- `likelihood_entropy.jsonl` 로 분포를 재측정한 뒤 임계를 다시 잡을 것.
- **주의**: 현재 이 파일은 **epoch 12 기준**이다. 최종 채택본인 epoch 8로
  재생성해야 한다 (`evaluate_heldout.py` 재실행).

### 4.3 reward의 split 위생

WM이 Ego4D **train split에 12 epoch(train loss 0.15)까지 적합**되었다.
GRPO를 같은 train split에서 돌리면 reward가 과적합된 likelihood가 되어
과신 편향이 생긴다.

- **권고**: 정책 학습 샘플은 **dev**(23,193개 — WM 가중치 학습에 미사용,
  checkpoint 선택에만 사용)에서 뽑을 것.
- heldout(6,233개)은 최종 평가 전용으로 계속 보존.

### 4.4 체크포인트 버전 주의

배포 zip(`ego4d-lta-model-release-0717`)에 든 `best_action.pt`는
**epoch 5**(dev action R@5 6.25)다. 최종 채택본은 **epoch 8**
(`outputs/ego4d_lta/runs/full/best_action.pt`, heldout action R@5 8.03)이므로
통합 시 어느 쪽을 받았는지 반드시 확인할 것.

---

## 5. 통합 전 체크리스트 (실행 순서 제안)

- [ ] epoch 8 checkpoint로 배포 패키지 갱신 (또는 수령 측에 epoch 확인)
- [ ] dev split에서 coverage@5 실측 (joint 5후보 기준 + cross-product 25후보 기준 비교)
- [ ] epoch 8 기준 `likelihood_entropy.jsonl` 재생성 → spread 분포 확인 →
      `--min_wm_spread` 재설정
- [ ] `canonical_action` 정규화 함수를 Ego4D taxonomy 형식에 맞춰 점검
- [ ] GRPO/DPO 학습 샘플 소스를 dev로 확정 (heldout은 최종 평가 전용 유지)
- [ ] (선택) cross-product 후보 전환 여부 결정 — 빌더·reward 매칭 변경 범위 산정

**참고**: coverage@5·spread 실측은 Ego4D feature cache가 있는 WM 저장소 환경에서
수행해야 한다 (이 머신에는 해당 산출물 없음).
