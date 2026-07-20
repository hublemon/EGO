"""Phase 2: parse Ego4D GoalStep cooking step/substep labels into (verb, noun,
action) classes using the EPIC-KITCHENS-100 (EK100) methodology, emitting
FHO-LTA-schema-identical taxonomy + registry artifacts.

EK100 method, applied here:
  (1) EXTRACT  -- POS/dependency parse each label with spaCy, take the main
      (root) verb and its object noun. GoalStep labels are imperative phrases
      ("Add oil to a pan"); sentence-initial verbs are mis-tagged as nouns, so
      we coerce an imperative reading by prepending a subject ("I ") and
      lower-casing the first letter before parsing (standard EK100-style trick),
      plus POS-tolerant fallbacks for the many short imperatives spaCy still
      mis-tags ("mash"/"toast" -> NOUN, "serve" -> AUX, object "eggplant" -> ADJ).
  (2) LEMMATISE -- reduce verb/noun to lemma (spaCy lemma_).
  (3) CLUSTER  -- group lemmas into semantic classes: identity by lemma, plus a
      small curated synonym->canonical merge map (VERB_MERGE / NOUN_MERGE).
      Each class = canonical key + member list, rendered FHO-style as
      "key_(member1,_member2)".
  (4) ACTION   -- action = (verb_class, noun_class); dense-index only the
      combinations that occur in the TRAIN split (identical rule to FHO
      register_action_labels in build_lta_z1_index.py).

Parsing SOURCE (Phase-1-confirmed): step_category text after the colon
("Cook on a stovetop: Preheat a pan or pot" -> "Preheat a pan or pot").
100% of GoalStep step_category values carry the colon; there are only 501
distinct labels -- a controlled vocabulary, far cleaner than free-form
step_description. Parsing LEVEL: both step and substep (they share the same
step_category vocabulary). trainval goal-level rows are NOT parsed (goal-level,
multi-domain -- out of scope for a cooking step taxonomy).

Failure handling: a segment whose label yields no verb OR no object noun is
bucketed as OTHER, logged verbatim to other_segments.csv, and excluded from the
taxonomy/registry. If the OTHER rate exceeds --other-threshold (default 0.15) a
loud warning is printed.

Usage:
    python scripts/step1/goalstep/parse_goalstep_to_verbnoun.py \
        --annotations-dir data/Ego4D/v2/annotations \
        --output-dir outputs/goalstep/taxonomy
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

# --- EK100-style semantic clustering: curated synonym -> canonical merges. ---
# Kept deliberately conservative and unambiguous; every merge is documented in
# GOALSTEP_TAXONOMY_METHOD.md. Frequent, semantically distinct verbs (stir vs
# mix, wash vs clean) are intentionally NOT merged to preserve granularity.
VERB_MERGE = {
    "place": "put", "set": "put", "lay": "put", "drop": "put", "position": "put",
    "rinse": "wash",
    "chop": "cut", "slice": "cut", "dice": "cut", "mince": "cut",
    "combine": "mix", "blend": "mix", "whisk": "mix", "beat": "mix",
    "grab": "get", "take": "get", "fetch": "get", "retrieve": "get",
    "collect": "get", "pick": "get",
    "discard": "dispose", "throw": "dispose", "toss": "dispose",
    "arrange": "organize", "tidy": "organize",
    "form": "shape",
}
NOUN_MERGE = {
    "utensil": "tool", "cookware": "tool", "equipment": "tool",
    "seasoning": "spice",
    "veggie": "vegetable", "veg": "vegetable",
}

# Tokens never eligible to be the fallback verb (structural / manner words), so
# "Deep fry donuts" -> fry rather than deep.
_VERB_SKIP_POS = {"DET", "ADP", "ADV", "PUNCT", "PRON", "CCONJ", "PART", "SPACE", "NUM"}
_MANNER_WORDS = {
    "deep", "evenly", "gently", "carefully", "thoroughly", "finely", "roughly",
    "well", "lightly", "slightly", "quickly", "slowly", "properly", "neatly",
}


def build_extractor():
    import spacy

    nlp = spacy.load("en_core_web_sm", disable=["ner"])

    def extract(phrase: str):
        """Return (verb_lemma, noun_lemma), either possibly None.

        Primary: dependency parse of an imperative-coerced sentence. Fallbacks
        recover the many short imperatives spaCy mis-POS-tags: the object noun
        is accepted from a dobj/obj relation regardless of POS (spaCy tags bare
        objects like "eggplant" as ADJ), and the verb falls back to the first
        content token when no VERB/AUX is tagged."""
        phrase = (phrase or "").strip()
        if not phrase:
            return (None, None)
        # imperative coercion: "Store ingredients" -> "I store ingredients"
        doc = nlp("I " + phrase[0].lower() + phrase[1:])
        toks = [t for t in doc if t.i > 0]  # drop the injected "I"
        verb = (next((t for t in toks if t.pos_ in ("VERB", "AUX")
                      and t.dep_ in ("ROOT", "conj", "aux")), None)
                or next((t for t in toks if t.pos_ in ("VERB", "AUX")), None)
                or next((t for t in toks if t.pos_ not in _VERB_SKIP_POS
                         and t.lemma_.lower() not in _MANNER_WORDS), None))
        noun = None
        if verb is not None:
            noun = next((c for c in verb.children if c.dep_ in ("dobj", "obj")), None)
        for dep in ("dobj", "obj"):
            if noun is None:
                noun = next((t for t in toks if t.dep_ == dep), None)
        if noun is None:
            noun = next((t for t in toks if t.dep_ == "pobj" and t.pos_ in ("NOUN", "PROPN")), None)
        if noun is None:
            noun = next((t for t in toks if t.pos_ in ("NOUN", "PROPN") and t is not verb), None)
        vl = verb.lemma_.lower() if verb else None
        nl = noun.lemma_.lower() if noun else None
        if vl is not None and vl == nl:  # degenerate self-object
            nl = None
        return (vl, nl)

    return extract


def _post_colon(step_category: str) -> str:
    sc = step_category or ""
    return sc.partition(":")[2].strip() or sc.strip()


def iter_segments(annotations_dir: Path):
    """Yield (video_uid, split, level, start, end, step_category) for every
    dense step/substep segment in train+val."""
    for fname, split in [("goalstep_train.json", "train"), ("goalstep_val.json", "val")]:
        path = annotations_dir / fname
        if not path.exists():
            print(f"[warn] missing {path}")
            continue
        with open(path) as f:
            data = json.load(f)
        for v in data["videos"]:
            uid = v["video_uid"]
            for step in v.get("segments", []) or []:
                yield (uid, split, "step", step.get("start_time"), step.get("end_time"),
                       step.get("step_category", ""))
                for sub in step.get("segments", []) or []:
                    yield (uid, split, "substep", sub.get("start_time"), sub.get("end_time"),
                           sub.get("step_category", ""))


def canonical(lemma, merge):
    if lemma is None:
        return None
    return merge.get(lemma, lemma)


def class_display(key, members):
    """FHO-style display: 'put_(place,_set)'; bare 'wash' if no extra members."""
    extra = sorted(m for m in members if m != key)
    if not extra:
        return key
    return f"{key}_(" + ",_".join(extra) + ")"


def run(args):
    ann_dir = Path(args.annotations_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    extract = build_extractor()
    phrase_cache = {}

    segments = list(iter_segments(ann_dir))
    print(f"[info] dense step+substep segments (train+val): {len(segments)}")

    verb_members = {}
    noun_members = {}

    # --- pass 1: extract verb/noun per segment ---
    parsed = []
    other = []
    n_other = 0
    for (uid, split, level, start, end, sc) in segments:
        phrase = _post_colon(sc)
        if phrase not in phrase_cache:
            phrase_cache[phrase] = extract(phrase)
        vraw, nraw = phrase_cache[phrase]
        vclass = canonical(vraw, VERB_MERGE)
        nclass = canonical(nraw, NOUN_MERGE)
        is_other = vclass is None or nclass is None
        if is_other:
            n_other += 1
            other.append({"video_uid": uid, "split": split, "level": level,
                          "step_category": sc, "verb_raw": vraw or "", "noun_raw": nraw or ""})
        parsed.append({"video_uid": uid, "split": split, "level": level,
                       "start_time": start, "end_time": end, "step_category": sc,
                       "verb_raw": vraw or "", "noun_raw": nraw or "",
                       "verb_class": vclass or "OTHER", "noun_class": nclass or "OTHER",
                       "action_label": "" if is_other else f"{vclass} {nclass}",
                       "is_other": int(is_other), "is_pruned": 0})

    other_rate = n_other / max(1, len(segments))

    # --- pass 2: long-tail pruning. Action classes whose TOTAL support (train+val)
    # is <= --min-action-count are dropped entirely: their segments are flagged
    # is_pruned=1 and excluded from the taxonomy, the registry and any training
    # index built downstream. Rationale: with an ~8:2 train/val ratio a class with
    # <=10 total has ~8 train samples -- unlearnable, and its 1-2 val samples only
    # add evaluation variance. See outputs/goalstep/taxonomy/action_pruning_table.csv.
    # --prune-on train counts TRAIN support only, so val never influences which
    # classes exist (a clean protocol); --prune-on total uses train+val support.
    action_support = {}
    for r in parsed:
        if r["is_other"]:
            continue
        if args.prune_on == "train" and r["split"] != "train":
            continue
        k = (r["verb_class"], r["noun_class"])
        action_support[k] = action_support.get(k, 0) + 1
    if args.prune_on == "train":  # val-only combos have no train support -> always pruned
        for r in parsed:
            if not r["is_other"]:
                action_support.setdefault((r["verb_class"], r["noun_class"]), 0)
    pruned_actions = {k for k, c in action_support.items() if c <= args.min_action_count}
    n_pruned = 0
    pruned_log = []
    if args.min_action_count > 0:
        for r in parsed:
            if r["is_other"]:
                continue
            if (r["verb_class"], r["noun_class"]) in pruned_actions:
                r["is_pruned"] = 1
                n_pruned += 1
                pruned_log.append({"video_uid": r["video_uid"], "split": r["split"],
                                   "level": r["level"], "step_category": r["step_category"],
                                   "action_label": r["action_label"],
                                   "action_support": action_support[(r["verb_class"], r["noun_class"])]})

    # --- vocab is rebuilt from SURVIVING segments only (pruning can orphan a
    # verb/noun whose only actions were all in the tail) ---
    for r in parsed:
        if r["is_other"] or r["is_pruned"]:
            continue
        verb_members.setdefault(r["verb_class"], set()).add(r["verb_raw"])
        noun_members.setdefault(r["noun_class"], set()).add(r["noun_raw"])

    verb_keys = sorted(verb_members)
    noun_keys = sorted(noun_members)
    verb_display = [class_display(k, verb_members[k]) for k in verb_keys]
    noun_display = [class_display(k, noun_members[k]) for k in noun_keys]
    verb_id = {k: i for i, k in enumerate(verb_keys)}
    noun_id = {k: i for i, k in enumerate(noun_keys)}

    write_json(out_dir / "goalstep_verbnoun_taxonomy.json",
               {"verbs": verb_display, "nouns": noun_display})

    train_verbs, train_nouns, train_pairs = set(), set(), set()
    for r in parsed:
        if r["is_other"] or r["is_pruned"] or r["split"] != "train":
            continue
        v, n = verb_id[r["verb_class"]], noun_id[r["noun_class"]]
        train_verbs.add(v); train_nouns.add(n); train_pairs.add((v, n))
    verb_dense = {raw: d for d, raw in enumerate(sorted(train_verbs))}
    noun_dense = {raw: d for d, raw in enumerate(sorted(train_nouns))}
    action_classes = {f"{v}|{n}": a for a, (v, n) in enumerate(sorted(train_pairs))}
    write_json(out_dir / "action_registry.json", {
        "num_verbs": len(verb_dense),
        "num_nouns": len(noun_dense),
        "num_actions": len(action_classes),
        "verb_classes": {str(raw): d for raw, d in verb_dense.items()},
        "noun_classes": {str(raw): d for raw, d in noun_dense.items()},
        "action_classes": action_classes,
    })

    _write_actions_csv(out_dir / "action_classes.csv", action_classes, verb_keys, noun_keys, parsed)

    seg_verb_count, seg_noun_count = _class_freq(parsed)
    _write_classes_csv(out_dir / "verb_classes.csv", verb_keys, verb_id, verb_members, seg_verb_count)
    _write_classes_csv(out_dir / "noun_classes.csv", noun_keys, noun_id, noun_members, seg_noun_count)

    _write_csv(out_dir / "goalstep_parsed_segments.csv", parsed,
               ["video_uid", "split", "level", "start_time", "end_time",
                "step_category", "verb_raw", "noun_raw", "verb_class", "noun_class",
                "action_label", "is_other", "is_pruned"])
    _write_csv(out_dir / "other_segments.csv", other,
               ["video_uid", "split", "level", "step_category", "verb_raw", "noun_raw"])
    _write_csv(out_dir / "pruned_segments.csv", pruned_log,
               ["video_uid", "split", "level", "step_category", "action_label", "action_support"])

    _summary(out_dir, len(segments), n_other, other_rate, verb_keys, noun_keys,
             action_classes, seg_verb_count, seg_noun_count, args.other_threshold,
             args.min_action_count, n_pruned, len(pruned_actions), len(action_support))


def _class_freq(parsed):
    v, n = {}, {}
    for r in parsed:
        if r["is_other"] or r["is_pruned"]:
            continue
        v[r["verb_class"]] = v.get(r["verb_class"], 0) + 1
        n[r["noun_class"]] = n.get(r["noun_class"], 0) + 1
    return v, n


def _write_actions_csv(path, action_classes, verb_keys, noun_keys, parsed):
    """EK100/FHO-style ACTION table, human-readable: each row is one registered
    (verb_class, noun_class) combination with its dense action_id and support."""
    tr, va = {}, {}
    for r in parsed:
        if r["is_other"] or r["is_pruned"]:
            continue
        key = (r["verb_class"], r["noun_class"])
        (tr if r["split"] == "train" else va)[key] = (tr if r["split"] == "train" else va).get(key, 0) + 1
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["action_id", "verb_id", "noun_id", "verb_key", "noun_key",
                    "action_label", "train_count", "val_count", "total_count"])
        for vn, aid in sorted(action_classes.items(), key=lambda x: x[1]):
            v_id, n_id = (int(x) for x in vn.split("|"))
            vk, nk = verb_keys[v_id], noun_keys[n_id]
            t, x = tr.get((vk, nk), 0), va.get((vk, nk), 0)
            w.writerow([aid, v_id, n_id, vk, nk, f"{vk} {nk}", t, x, t + x])


def _write_classes_csv(path, keys, id_map, members, freq):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class_id", "class_key", "members", "segment_count"])
        for k in keys:
            w.writerow([id_map[k], k, ";".join(sorted(members[k])), freq.get(k, 0)])


def _write_csv(path, rows, cols):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _summary(out_dir, n_seg, n_other, other_rate, verb_keys, noun_keys,
             action_classes, vfreq, nfreq, threshold,
             min_action_count=0, n_pruned=0, n_pruned_actions=0, n_actions_before=0):
    print("\n" + "=" * 68)
    print("GoalStep -> verb/noun/action (EK100 method) -- summary")
    print("=" * 68)
    print(f"Segments parsed (train+val) : {n_seg}")
    print(f"OTHER (no verb or no noun)  : {n_other}  ({100*other_rate:.2f}%)")
    if min_action_count > 0:
        kept = n_seg - n_other - n_pruned
        print(f"PRUNED (action support <= {min_action_count}) : {n_pruned} segs "
              f"({100*n_pruned/max(1,n_seg):.2f}%) across {n_pruned_actions} action classes "
              f"(of {n_actions_before} pre-prune)")
        print(f"KEPT for training           : {kept} segs ({100*kept/max(1,n_seg):.2f}%)")
    print(f"N_verb  (taxonomy)          : {len(verb_keys)}")
    print(f"N_noun  (taxonomy)          : {len(noun_keys)}")
    print(f"N_action(train-seen combos) : {len(action_classes)}")
    top_v = sorted(vfreq.items(), key=lambda x: -x[1])[:10]
    top_n = sorted(nfreq.items(), key=lambda x: -x[1])[:10]
    print("\nTop verb classes :", ", ".join(f"{k}({c})" for k, c in top_v))
    print("Top noun classes :", ", ".join(f"{k}({c})" for k, c in top_n))
    print(f"\nArtifacts written to {out_dir}/:")
    for name in ["goalstep_verbnoun_taxonomy.json", "action_registry.json",
                 "verb_classes.csv", "noun_classes.csv", "action_classes.csv", "pruned_segments.csv",
                 "goalstep_parsed_segments.csv", "other_segments.csv"]:
        print(f"  - {name}")
    if other_rate > threshold:
        print("\n" + "!" * 68)
        print(f"!!! WARNING: OTHER rate {100*other_rate:.2f}% EXCEEDS threshold "
              f"{100*threshold:.0f}% -- inspect other_segments.csv and revisit parsing !!!")
        print("!" * 68)
    print("=" * 68)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--annotations-dir", default="data/Ego4D/v2/annotations")
    p.add_argument("--output-dir", default="outputs/goalstep/taxonomy")
    p.add_argument("--prune-on", choices=["total", "train"], default="total",
                   help="support basis for --min-action-count: 'train' keeps val out of the "
                        "label-space decision (cleaner protocol); 'total' uses train+val")
    p.add_argument("--min-action-count", type=int, default=0,
                   help="drop action classes whose total (train+val) support is <= this; 0 = keep all")
    p.add_argument("--other-threshold", type=float, default=0.15,
                   help="Warn loudly if OTHER fraction exceeds this")
    run(p.parse_args())


if __name__ == "__main__":
    main()
