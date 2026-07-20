"""Dataset and label sanity-check scaffold for Step 1 (Phase 1 of the baseline workflow).

Loads the train/val annotations, resolves video paths, fits the label
mapping, and reports counts + missing videos -- before any model or GPU time
is spent. This is what ``ego step1 prepare`` runs.
"""

from __future__ import annotations

from pathlib import Path

from ego.common.config import get, load_config
from ego.common.io import write_json
from ego.common.logging import step_log
from ego.common.paths import expand_path
from ego.step1_action_anticipation.data.build_samples import build_step1_datasets


def prepare(config_path: str) -> dict:
    config = load_config(config_path)
    step_log(1, "Prepare", "Config loaded")
    step_log(1, "Prepare", f"Train annotation: {get(config, 'dataset.annotation_train')}")
    step_log(1, "Prepare", f"Validation annotation: {get(config, 'dataset.annotation_val')}")
    step_log(1, "Prepare", f"Video root: {get(config, 'dataset.video_root')}")

    datasets = build_step1_datasets(config)
    mapping = datasets.label_mapping

    num_val_samples = len(datasets.val) if datasets.val is not None else 0
    step_log(1, "Prepare", f"Train samples: {len(datasets.train)}")
    step_log(1, "Prepare", f"Validation samples: {num_val_samples}")
    step_log(1, "Prepare", f"Verb classes: {mapping.num_verbs}")
    step_log(1, "Prepare", f"Noun classes: {mapping.num_nouns}")
    step_log(1, "Prepare", f"Action classes: {mapping.num_actions}")
    step_log(1, "Prepare", f"Train videos: {datasets.num_train_videos}")
    step_log(1, "Prepare", f"Validation videos: {datasets.num_val_videos}")
    step_log(1, "Prepare", f"Missing videos: {len(datasets.missing_videos)}")

    summary = {
        "config_path": str(Path(config_path).resolve()),
        "train_samples": len(datasets.train),
        "val_samples": num_val_samples,
        "train_videos": datasets.num_train_videos,
        "val_videos": datasets.num_val_videos,
        "num_verb_classes": mapping.num_verbs,
        "num_noun_classes": mapping.num_nouns,
        "num_action_classes": mapping.num_actions,
        "missing_videos": datasets.missing_videos,
    }

    output_dir = get(config, "experiment.output_dir")
    if output_dir:
        out_path = expand_path(output_dir) / "dataset_summary.json"
        write_json(out_path, summary)
        step_log(1, "Prepare", f"Dataset summary written: {out_path}")

    return summary
