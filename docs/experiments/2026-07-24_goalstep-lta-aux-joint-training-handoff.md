# GoalStep×LTA 보조 감독 Joint Training Handoff (Step1, A3 next-action 계약)

- 작성일: 2026-07-24 KST
- 목적: GoalStep Step1 next-action(A3) 파이프라인의 visual foundation에 **LTA 세그먼트를
  부분 라벨(verb/noun) 보조 감독으로 주입**해, 과적합의 근본 원인(비디오 다양성 부족)을
  공략한다. **GoalStep action 293 클래스는 절대 늘리지 않는다.**
- 학습 방법론 기준 문서(계약·코드·평가 프로토콜을 그대로 따를 것):
  [2026-07-23_goalstep-history-p0a-phase1-phase2-final-report.md](2026-07-23_goalstep-history-p0a-phase1-phase2-final-report.md)
  - 주의: `2026-07-23_goalstep-history-context-implementation-handoff.md`는 **SUPERSEDED**
    (옛 P0-b hard-gate 시점 기록). 계약이 충돌하면 final report가 정본이다.
- 데이터 설계 근거: 2026-07-24 세션에서 실데이터로 계산한 LTA↔GoalStep 라벨 매칭
  실측치 (본 문서 §3에 수치와 재계산 코드 보존).

---

## 0. 결론 요약 (이것만 읽어도 되는 버전)

1. GoalStep Step1은 train 30,374 세그먼트가 **570개 비디오**에서 나와 (53.3개/비디오)
   probe가 조기 암기한다 (end−1s run 기준 train loss 0.007, val 피크 epoch 3-6).
   클래스당 샘플은 median 47개로 건강하므로, 처방은 세그먼트 증량이 아니라
   **비디오 다양성 증량**이다.
2. LTA(fho_lta train+val 97,105 세그먼트)에서 GoalStep 라벨 공간으로 매칭되는
   세그먼트를 실측한 결과: **action 완전 일치는 195개뿐(무의미)**, 그러나
   **verb+noun 동시 매칭 16,098개(994개 비디오), 한쪽 매칭 61,612개(1,296개 비디오)**.
   → 부분 라벨 마스킹으로 주입하면 train 비디오가 570 → 약 1,540개(2.7배)가 된다.
3. 주입 지점은 **direct next-action visual probe 학습(Stage 2)** 한 곳이다.
   P0-a ensemble → Phase 1 history → Phase 2 zoo는 final report의 방법론을
   **코드·게이트 규칙 변경 없이 재실행**한다 (frozen visual source가 바뀌므로
   전체 재실행 필수).
4. 누수: LTA 매칭 세그먼트 중 **730개(20개 비디오)가 GoalStep val 비디오와 겹친다.
   반드시 제외.** GoalStep train 비디오와 겹치는 3,711개는 문제없음.
5. 성공 판정은 final report와 동일한 paired 규칙(Δ>0 AND video-bootstrap CI 하한>0),
   비교 대상은 단계별로: direct ep3 **25.65** → P0-a **28.4052** → Phase 1 blend
   **30.3448** → Phase 2 champion **31.2356** (Action Top-5, val strict-next cohort 6,960).

---

## 1. 배경 진단 (왜 이걸 하는가)

| 근거 | 실측값 | 출처 |
|---|---|---|
| GoalStep train 규모 | 30,374 seg / **570 videos** (53.3 seg/video) | `index_end_m1_lobs8/train.parquet` |
| GoalStep val 규모 | 7,214 seg / 130 videos | `index_end_m1_lobs8/val.parquet` |
| action 클래스 분포 | 293 classes, median 47/class, <10샘플 클래스 1개, 최다 점유 2.8% | train.parquet 직접 집계 |
| 과적합 증거 | train loss 0.007 (epoch 15), val top-5 피크 epoch 3-6 후 하락 | [2026-07-21 VNA 결과](2026-07-21_goalstep-action-end-m1-lobs8-vna-results.md) |
| 대조 (LTA 학습) | 같은 구조로 train loss 0.25 (epoch 8) — 암기 안 됨 | [2026-07-17 LTA 결과](../../develop_report/2026-07-17_ego4d-lta-full-training-results.md) |

클래스당 샘플이 부족한 게 아니라 **비디오(환경) 다양성이 부족**해서 frozen feature 위의
probe가 570개 주방을 암기하는 구조. 같은 비디오에서 세그먼트를 더 잘라봐야 상관된
샘플만 늘어난다 → 외부 비디오 주입이 필요.

---

## 2. 계약 고정 (변경 금지 사항)

- **라벨 공간**: verb 81 / noun 140 / **action 293** (registry:
  `src/ego/step1_action_anticipation/goalstep/index_end_m1_lobs8/action_registry.json`).
  LTA 데이터는 이 공간으로 **매핑되어 들어오기만** 하고, 클래스 추가 없음.
- **예측 계약** (final report §1 그대로): 관찰 = A2.end−1s까지 8초(32프레임),
  target = strict-future same-level **A3**. `max(observed time) < A3.start` 불변식 유지.
- **평가**: GoalStep val strict-next cohort **6,960개** 고정. primary metric
  **Action Top-5** (Top-10을 부지표로 함께 보고 — 이 프로젝트의 실질 목표 지표).
  LTA aux 데이터는 어떤 형태로도 평가에 넣지 않는다.
- **모델 구조**: frozen V-JEPA2 ViT-L/16 256 + attentive probe (depth 4, 16 heads),
  V/N/A 3-head focal loss. history head 구조도 final report §3 그대로.

---

## 3. Stage 0 — LTA 보조 index 구축

### 3.1 이번 세션 실측 매칭 수치 (설계 근거)

매칭 방법: LTA 라벨 텍스트를 `split('_(')[0].lower().replace('_',' ')`로 정규화한 뒤,
GoalStep `taxonomy_original_k0`의 `verb_classes.csv`/`noun_classes.csv`의
class_key+members(동의어)와 exact match. registry(81/140)에 없는 클래스는 제외.

| 매칭 수준 (LTA train+val 97,105 seg 기준) | 세그먼트 | 비디오 | 비고 |
|---|---:|---:|---|
| verb+noun+action 모두 293개 안에 일치 | **195** (train 147 + val 48) | 89 | 13/293 action만 커버 — 이것만으론 무의미 |
| verb+noun 동시 매칭 (조합이 293에 없음) | **16,098** | 994 | **1차 주입 대상** |
| verb 또는 noun 한쪽 매칭 | **61,612** | 1,296 | 확장 옵션 (verb-only가 대부분) |
| — 세부: LTA train verb/noun/both | 31,817 / 20,702 / 11,332 | | train 63,956 중 |
| — 세부: LTA val both | 4,766 | | val 33,149 중 |

**누수 실측**: both-match 세그먼트 중 GoalStep **val** 비디오 소속 **730개(20 videos)
→ 제외 필수**. GoalStep train 비디오 소속 3,711개는 유지 가능.
LTA의 train/val split 구분은 무시해도 된다(우리 평가는 GoalStep val이므로 LTA val
세그먼트도 학습에 사용 가능) — 단 위 누수 규칙은 동일 적용.

### 3.2 주의: 위 수치는 "세그먼트 자체 라벨" 기준이다

A3 next-action 계약에서는 **target(다음 action)의 라벨**이 GoalStep 공간에 있어야
감독이 성립한다. LTA는 clip 내 `action_idx` 순서가 있으므로 A3 = 같은 clip의 다음
action으로 정의하면 되고, builder에서 **A3 라벨 기준으로 재집계**해야 한다
(모든 action이 어떤 행의 target이기도 하므로 분포는 위 수치와 유사할 것으로 예상;
±10% 수준 차이는 정상).

### 3.3 Builder 구현 스펙

새 파일: `src/ego/step1_action_anticipation/goalstep/build_lta_aux_index.py`
(기존 `build_goalstep_next_action_index.py`를 참고해 같은 스키마로 출력)

- 입력: `data/Ego4D/v2/annotations/fho_lta_train.json` + `fho_lta_val.json`,
  GoalStep taxonomy CSV 2개, registry JSON, GoalStep val video_uid 목록.
- 행 구성: LTA clip 내 연속 action 쌍 (A2, A3)마다
  - `obs_end_sec = A2.end − 1.0` (video 좌표계: `clip_parent_start_sec + action_clip_end_sec` 사용,
    `interval_*` 필드는 좌표계 미확정이므로 쓰지 말 것 —
    [2026-07-13 join 설계 문서](../../develop_report/2026-07-13_ego4d-lta-goalstep-join-method.md) §1.1 참고)
  - `obs_start_sec = max(clip 시작, obs_end_sec − 8.0)`
  - target 라벨: A3의 verb/noun을 §3.1 규칙으로 매핑 →
    `verb_label`(매칭 시, 아니면 −1), `noun_label`(동일), `action_label`
    (조합이 293 registry에 있을 때만, 아니면 −1)
  - `verb_mask`/`noun_mask`/`action_mask` boolean 컬럼 추가
  - verb·noun 둘 다 −1인 행은 버림
- 필터: `video_uid ∈ GoalStep val 130개` 제외 (fail-closed assert 포함).
- 출력: `src/ego/step1_action_anticipation/goalstep/index_lta_aux_end_m1_lobs8/`
  (`train.parquet` + `build_stats.json`; val 없음 — aux는 학습 전용)
- build_stats에 매칭 카운트/누수 제외 수/커버 action 수를 기록해 §3.1 수치와 대조.

### 3.4 재계산용 검증 코드 (이번 세션에서 실행 검증됨)

```python
import json, csv
def load_classes(path):
    out = {}
    for r in csv.DictReader(open(path)):
        members = set(m.strip() for m in r['members'].replace('|', ',').split(','))
        members.add(r['class_key'])
        out[r['class_id']] = members
    return out
gvs = load_classes('outputs/goalstep/taxonomy_original_k0/verb_classes.csv')
gns = load_classes('outputs/goalstep/taxonomy_original_k0/noun_classes.csv')
reg = json.load(open('src/ego/step1_action_anticipation/goalstep/'
                     'index_end_m1_lobs8/action_registry.json'))
def build_lookup(classes, keep):
    lk = {}
    for rid, members in classes.items():
        if rid not in keep: continue
        for m in members:
            lk.setdefault(m.lower().replace('_', ' '), rid)
    return lk
vlk = build_lookup(gvs, reg['verb_classes'])   # 85 surface words
nlk = build_lookup(gns, reg['noun_classes'])   # 140 surface words
def norm(w):
    return w.split('_(')[0].lower().replace('_', ' ').strip()
# LTA clip c에 대해: vlk.get(norm(c['verb'])), nlk.get(norm(c['noun']))
# action: (raw_v, raw_n) 조합이 reg['action_classes']의 f"{raw_v}|{raw_n}" 키에 있을 때만
```

---

## 4. Stage 1 — LTA aux feature 추출

- 프로토콜: GoalStep cache와 **완전 동일** — end−1s endpoint, 최대 8초 관찰,
  32프레임 균일 샘플링, 256px, frozen V-JEPA2 ViT-L/16 (`checkpoints/vjepa2/vitl.pt`).
- 비디오 소스: `data/Ego4D/v2/clip_256ss` (LTA clip 다운로드본, clip_uid 단위 파일).
  obs 좌표를 video→clip 좌표로 변환할 때 `clip_parent_start_sec`를 빼서 사용.
- 규모: 1차(both-match, 누수 제외) 약 **15.4k 세그먼트** 예상. 기존 GoalStep 추출
  실측(H200에서 30,374개)의 절반 규모이므로 수 시간 내 완료 예상.
- 출력: `../datasets/Ego4D/lta_aux_feature_cache_end_m1_lobs8/` (GoalStep cache와
  같은 `[temporal, spatial, 1024]` 포맷, cache ID = index 행과 1:1).
- 확장 옵션(any-match ~60k)은 Stage 2 1차 결과 확인 후에만 추출할 것 (비용 4배).

---

## 5. Stage 2 — Direct next-action probe 재학습 (aux 주입 지점)

기준 코드: `src/ego/step1_action_anticipation/goalstep/train_goalstep_z1.py` +
`configs/step1/goalstep/z1_end_m1_lobs8_next_action_vna_ep10.yaml`
(direct next ep3 Top-5 25.65를 만든 조합). 여기에 두 가지만 추가한다:

1. **Masked loss**: aux 행은 `*_mask`가 true인 head에만 focal loss를 흘린다.
   GoalStep 행은 기존과 동일하게 V/N/A 전부. action head gradient는 사실상
   GoalStep 행에서만 나온다(aux의 action_mask true는 ~150행뿐) → **293 클래스 불변**.
2. **혼합 샘플링**: epoch마다 GoalStep 29,293행 전부 + aux를 λ 비율로 샘플링.
   loss = `L_goalstep + λ·L_aux`, **λ=0.3, 배치 내 비율 약 7:3에서 시작**.

- 학습 설정은 기존과 동일: batch 32, BF16, 10 epochs, LR 3e-4, WD 1e-4, warmup 1.
- 산출: 새 run dir `outputs/goalstep/runs/z1_end_m1_lobs8_next_action_vna_ep10_ltaaux/`
  — **기존 run dir을 절대 덮어쓰지 말 것** (P0-a가 기존 epoch 1-8 확률을 참조함).
- 1차 체크(게이트 아님, 방향 확인용): full-val direct Top-5가 기존 25.65 대비
  올라가는지, val 피크 epoch이 3-6보다 늦춰지는지, train loss가 0.007까지
  떨어지지 않는지(암기 완화 신호).

## 6. Stage 3 — P0-a → Phase 1 → Phase 2 재실행 (방법론 변경 없음)

frozen visual source가 바뀌므로 final report §9의 파이프라인을 그대로 재실행:

| 순서 | 코드 (변경 없이 config/경로만 새 run으로) |
|---|---|
| P0-a ensemble | `scripts/step1/goalstep/run_history_phase0.py` — 새 probe의 epoch 1–8 확률로 video-disjoint 2-fold Caruana |
| derived store | `scripts/step1/goalstep/prepare_history_context_store.py` — 새 frozen logits로 재생성 (fingerprint가 바뀌므로 기존 store 재사용 불가, fail-closed가 막아줄 것) |
| Phase 1 | `train_goalstep_history_context.py` + `z1_history_context_k8_vna_ep10.yaml` |
| Phase 2 | `train_goalstep_history_probe_zoo.py` + evaluator (`evaluate_history_probe_zoo_vs_p0a.py`) |

- history index(`index_end_m1_lobs8_next_action_history_k8`)는 GoalStep 전용 그대로
  재사용 — **aux 데이터는 history 단계에 넣지 않는다** (LTA history chain 구축은
  이번 범위 밖의 후속 확장).
- 선택/승격 규칙, fold 구성(video-disjoint 2-fold, 동일 seed), bootstrap 방법 모두
  final report §7.3과 동일하게.

---

## 7. 평가와 성공 판정

| 비교 | 기존 값 (Action Top-5, val cohort 6,960) | 판정 규칙 |
|---|---:|---|
| 새 direct probe vs 기존 direct ep3 | 25.65 | 방향 확인 (게이트 아님) |
| 새 P0-a vs 기존 P0-a | 28.4052 | paired Δ>0 AND video-bootstrap CI 하한>0 |
| 새 Phase 1 blend vs 기존 30.3448 | 30.3448 | 동일 규칙 |
| **새 Phase 2 vs 기존 champion** | **31.2356** | 동일 규칙 — **이게 최종 판정** |

- Top-10도 모든 단계에서 함께 보고할 것 (기존 champion Top-10: 44.02).
- paired 비교는 같은 val cohort의 행 단위로 기존 OOF 확률
  (`phase2_vs_p0a_oof_scores.pt` 등 canonical artifact)과 직접 대조 가능.
- 기존 한계(final report §11 — 상속된 validation adaptivity, untouched test 부재)는
  이번에도 동일하게 적용되므로 결과 문서에 같은 캐비어트를 명시할 것.

### Ablation arms (최소 셋)

| Arm | 내용 | 목적 |
|---|---|---|
| A0 | λ=0 (aux 없음, 재학습만) | seed/재학습 분산 통제 |
| A1 | λ=0.3, both-match 15.4k | **본 실험** |
| A2 | λ=0.3, any-match ~60k (verb-only 포함) | 커버리지 확장 효과 (A1 성공 시에만) |

---

## 8. 주의사항 (실행 세션이 반드시 알아야 할 것)

1. **실행 환경**: 학습·feature cache(313 GB)·기존 run 산출물은 **jihun2 서버**에 있다
   (`EGO_jihun2` clone, interpreter **`eve-cu124`**, `PYTHONPATH=<repo>/src`).
   이 환경에는 pytest가 없으므로 smoke script / `python -m unittest`를 쓸 것
   (final report §10.2). 로컬(GB10) 저장소에는 annotation과 코드만 있다.
2. **좌표계 함정**: LTA의 `interval_*` 필드는 좌표계가 미확정이므로 사용 금지.
   `clip_parent_start_sec + action_clip_start/end_sec`로 video 좌표를 만들고,
   feature 추출 시 clip 파일 로컬 좌표로 되돌릴 것. 구현 전에 몇 개 샘플로
   `interval_start_sec ≈ clip_parent_start_sec + action_clip_start_sec` 여부를
   검증해 두면 좋다 (2026-07-13 join 문서 §3의 0단계).
3. **누수 fail-closed**: aux index builder와 trainer 양쪽에서
   `aux video_uid ∩ GoalStep val video_uid = ∅`를 assert. 20개 비디오/730행이
   걸러지는지 build_stats로 확인.
4. **기존 산출물 보존**: 기존 run dir(`z1_end_m1_lobs8_next_action*`,
   `history_context*`, `probe_zoo*`)과 canonical JSON/OOF artifact는 읽기 전용으로
   취급. 모든 신규 산출물은 `*_ltaaux` suffix run dir에 쓸 것.
5. **taxonomy 매핑의 한계**: exact word 매칭이라 보수적이다. 매칭률을 올리고
   싶으면 LTA taxonomy의 괄호 동의어 멤버를 lookup에 추가하는 확장이 있으나
   (§3.4 `norm`이 head token만 취하는 부분), 1차에서는 지금 규칙 그대로 갈 것
   (동의어 확장은 오매칭 검수 비용이 든다).
6. **adaptive MR24+8 run**: jihun2에서 재개되어 돌고 있을 수 있다
   (`ego_goalstep_adaptive_transition` tmux, final report §12). GPU 경합을
   확인하고 시작할 것.

---

## 9. 산출물 경로 계획

| 산출물 | 경로 |
|---|---|
| aux index builder | `src/ego/step1_action_anticipation/goalstep/build_lta_aux_index.py` |
| aux index | `src/ego/step1_action_anticipation/goalstep/index_lta_aux_end_m1_lobs8/` |
| aux feature cache | `../datasets/Ego4D/lta_aux_feature_cache_end_m1_lobs8/` |
| Stage 2 config | `configs/step1/goalstep/z1_end_m1_lobs8_next_action_vna_ep10_ltaaux.yaml` |
| Stage 2 run | `outputs/goalstep/runs/z1_end_m1_lobs8_next_action_vna_ep10_ltaaux/` |
| Stage 3 runs | `outputs/goalstep/runs/{history_context_phase0,z1_history_context_k8_vna_ep10,z1_history_context_probe_zoo_ep10}_ltaaux/` |
| 결과 문서 | `docs/experiments/2026-07-XX_goalstep-ltaaux-results.md` (신규 작성) |
