# GoalStep → verb/noun/action Taxonomy — Method (Phase 2)

Bespoke (verb, noun, action) taxonomy for the Ego4D **GoalStep** cooking step
set, built by applying the **EPIC-KITCHENS-100 (EK100)** verb/noun construction
methodology to GoalStep's `step_category` labels, and emitted in the **exact
schema of the FHO-LTA** artifacts so it plugs into the existing V-JEPA2
anticipation pipeline unchanged.

- Script: `scripts/step1/goalstep/parse_goalstep_to_verbnoun.py`
- Inputs: `data/Ego4D/v2/annotations/goalstep_{train,val}.json` (Ego4D v2.1)
- Outputs: `outputs/goalstep/taxonomy/`

## 1. Parsing source & level (Phase-1-confirmed)

| Decision | Choice | Rationale |
|---|---|---|
| Source field | `step_category`, **text after the colon** | 100% of `step_category` values are `"<coarse category>: <specific step>"`; the post-colon phrase is a clean imperative ("Add oil to a pan"). Only **501 distinct** labels → a controlled vocabulary, far cleaner than free-form `step_description` ("preheat pots", "add wood to fire"). |
| Level | **step + substep** | Both levels draw from the same 501-label `step_category` vocabulary, so parsing is level-independent; keeping both maximises coverage. |
| trainval goal rows | **excluded** | `goalstep_trainval.json` is goal-level only and multi-domain (EXERCISE, etc.) — out of scope for a cooking *step* taxonomy. |

## 2. EK100 method, step by step

1. **Extract (POS/dependency parse).** Each label is parsed with spaCy
   (`en_core_web_sm`). GoalStep labels are bare imperatives whose head verb
   spaCy systematically mis-tags. We therefore **coerce an imperative reading**:
   prepend a subject and lower-case the first letter (`"Store ingredients"` →
   `"I store ingredients"`), then take the ROOT verb and its object. Two
   POS-tolerant fallbacks recover the residual mis-tags:
   - object noun accepted from any `dobj`/`obj` relation **regardless of POS**
     (spaCy tags bare objects like *eggplant*, *radish* as ADJ);
   - verb falls back to the first content token (skipping determiners /
     adpositions / manner adverbs like *deep*, *evenly*) when no VERB/AUX is
     tagged — recovering *mash*, *toast*, *serve*, *fry*.
2. **Lemmatise.** verb/noun reduced to `lemma_` (plurals, tense collapsed:
   *ingredients*→*ingredient*, *chopped*→*chop*).
3. **Cluster into semantic classes.** Base class = lemma; a small **curated
   synonym→canonical merge map** collapses unambiguous synonyms. Each class is
   rendered FHO-style as `key_(member1,_member2)`.
   - Verbs merged: `put(place,set,lay,drop,position)`, `cut(chop,slice,dice,mince)`,
     `mix(combine,blend,whisk,beat)`, `get(grab,take,fetch,retrieve,collect,pick)`,
     `dispose(discard,throw,toss)`, `organize(arrange,tidy)`, `wash(rinse)`,
     `shape(form)`.
   - Nouns merged: `tool(utensil,cookware,equipment)`, `spice(seasoning)`,
     `vegetable(veggie,veg)`.
   - Deliberately **not** merged: `stir` vs `mix`, `wash` vs `clean` — frequent
     and semantically distinct; merging would erase real granularity.
4. **Action = (verb_class, noun_class).** Only combinations occurring in the
   **train** split are registered into a dense `action_label` index — identical
   to FHO `register_action_labels` (`build_lta_z1_index.py`).

## 3. Differences from EK100 / FHO

- **Input is a controlled label set, not free narration.** EK100 parses
  thousands of free-form narrations; GoalStep gives 501 curated `step_category`
  labels. Extraction is therefore far cleaner, but the (verb,noun) space is much
  smaller — **390 train-seen actions** vs FHO's 5,698. This is a property of the
  controlled vocabulary, not a defect.
- **Single domain.** GoalStep steps are entirely `COOKING:*`; the taxonomy is
  cooking-specific, whereas FHO/EK100 span many scenarios.
- **Automated clustering.** EK100's classes were partly hand-curated; here
  clustering is lemma + a small documented merge map (auditable in
  `verb_classes.csv` / `noun_classes.csv`).

## 4. Statistics

- Segments parsed (train+val step+substep): **39,262** (train 31,468 · val 7,637).
- Parse success **99.60%**; **OTHER 0.40%** (157 segments) — all "verb found but
  no object noun", dominated by **101× "Non cooking miscellaneous"** (genuinely
  non-actionable) plus a few single-word-object mis-tags (*Add coconut…*, *Cut
  orange*). Well under the 15% warning threshold.
- **N_verb = 100**, **N_noun = 190** (taxonomy vocabulary, train+val).
- **N_action = 293** (dense, train-seen; 이전 390).
- Dense registry (train-only, val-only classes dropped exactly as FHO 117→116):
  `num_verbs=98`, `num_nouns=188`, `num_actions=390`.
- Long tail: 6 verb classes and 5 noun classes with ≤2 segments
  (*baste, poach, bring…* / *dress, cooker, edamame…*).

**Top verb classes:** add(6120), stir(3379), wash(3054), cut(2800), check(1449),
cook(1420), organize(1372), put(1149), serve(1032), flip(1008).
**Top noun classes:** ingredient(7878), dough(4235), dish(1585), flatbread(1541),
tool(1323), heat(1115), hand(1030), onion(919), water(877), spice(801).

### Manual sample (original `step_category` → verb | noun)

| step_category | verb | noun |
|---|---|---|
| Make baked goods: Cut the dough into desired shapes | cut | dough |
| Make recipes: Add ingredients to oatmeal | add | ingredient |
| Peel and cut ingredients: Cut carrot | cut | carrot |
| Make recipes: Make sandwich | make | sandwich |
| Boil ingredients in water: Add water to the kettle to boil | add | water |
| General cooking activity: Steam ingredients over boiling water | steam | ingredient |
| General cooking activity: Dispose water in kitchen sink | dispose | water |
| General cooking activity: Fry eggs on a pan until the egg white… | fry | egg |
| Add ingredients to the recipe: Add broth or soup to recipe | add | broth |
| Make baked goods: Remove mixed dough from the dough mixer | remove | dough |
| General cooking activity: Wash spring onion or leek in water | wash | onion |

## 4b. 롱테일 가지치기 (k=10, 2026-07-19)

`action_classes.csv` 기준 누적 분석 결과, 클래스 수는 크게 줄어도 세그먼트 손실은 매우 작다
(`action_pruning_table.csv`, `action_pruning_tradeoff.png`). 이에 **총 지원(train+val) ≤ 10인
action 클래스를 학습 대상에서 제외**한다 (`--min-action-count 10`).

- 제외: **713 세그먼트(1.82%), 108개 action 클래스**. 유지: 38,392 세그먼트(97.78%).
- 연쇄 축소: **N_verb 100→81, N_noun 190→140, N_action 390→293**
  (해당 클래스에서만 쓰이던 verb/noun은 어휘에서 함께 사라짐).
- 규칙은 하나뿐이다: `--min-action-count 10 --prune-on train` — **train 지원 ≤10** 제거.
  train 지원 10 이하는 학습이 사실상 불가능하고 평가 분산만 키운다.
- 결과 지원: 세그먼트 기준 최소 train 11, 인덱스 기준 최소 train 샘플 9(중앙값 47).
- 구현: 제거 세그먼트는 `goalstep_parsed_segments.csv`에 `is_pruned=1`로 표시되고
  `pruned_segments.csv`에 원문과 함께 기록된다. taxonomy·registry·다운스트림 인덱스/VPA는
  `is_other==0 and is_pruned==0` 인 세그먼트만 사용한다. 원본(가지치기 전)은
  `outputs/goalstep/taxonomy_original_k0/`에 보존.
- ✅ **누출-free**: 가지치기 기준이 TRAIN 지원뿐이라 **val은 라벨 공간 결정에 전혀 관여하지 않는다.**
  부수 효과로 val 전용(train 미등장) 조합이 자동 제거되어, 인덱스 빌드의 "train-seen 제한"에서
  추가로 버려지는 val 샘플이 **0**이다.
  (val 지원 기준으로 클래스를 더 자르는 방식은 라벨 공간을 val에 의존시키므로 채택하지 않았다.)
- 알려진 한계: 293개 중 **23개 클래스는 val 평가 샘플이 0**이라 class-mean 지표에 잡히지 않는다
  (`metrics.py`가 support 0인 클래스를 NaN 처리 후 제외하므로 지표를 오염시키지는 않음).
  val 샘플 ≤2인 클래스도 34개라 매크로 지표의 분산 요인이다 → **평가 리포트에서 instance-level
  Recall@5를 함께 보고하고, 보조로 "val 지원 ≥5 클래스 한정" 매크로를 병기할 것**
  (보고 시점 계산이므로 라벨 공간을 바꾸지 않아 추가 누출이 없다).

## 5. Limitations & cautions

- This taxonomy is a **bespoke space** — it is neither FHO-LTA (117 verbs / 521
  nouns) nor GoalStep's official step taxonomy (514 steps). **Numbers here are
  NOT directly comparable to any external SOTA** on Ego4D LTA or GoalStep.
- Object selection takes the syntactic head noun, so a few multi-word objects
  collapse to a weaker head (*"a small amount of filling"* → `amount`, not
  `filling`). Verb selection takes the primary/root verb; compound-verb phrases
  ("check and adjust") keep the first verb (EK100's one-action-per-segment
  spirit) rather than being dropped as OTHER.
- The synonym merge maps are conservative and hand-authored; they are the main
  reviewable knob. Adjust `VERB_MERGE` / `NOUN_MERGE` in the script and re-run.
- val-only verb/noun classes exist in the taxonomy vocabulary but are absent
  from the train-seen action registry (same convention as FHO).
