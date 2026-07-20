"""Pipeline scaffold connecting Step 1 candidates to Step 2 training data.

Step 2 itself is out of scope here; this module is the seam Step 2's dataset
builder is expected to read through. It loads ``action_candidates.jsonl`` and
validates every record against ``schemas/step1_candidates.schema.json`` so a
malformed Step 1 export fails loudly at the boundary instead of silently
corrupting Step 2 training data.
"""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema

from ego.common.exceptions import EgoError
from ego.common.io import read_jsonl

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "schemas" / "step1_candidates.schema.json"


def _load_schema() -> dict:
    with open(SCHEMA_PATH, encoding="utf-8") as f:
        return json.load(f)


def load_step1_candidates(candidates_path: str | Path, validate: bool = True) -> list[dict]:
    """Load and (by default) schema-validate every record in a Step 1 candidates JSONL file."""
    records = list(read_jsonl(candidates_path))
    if validate:
        schema = _load_schema()
        validator = jsonschema.Draft202012Validator(schema)
        for i, record in enumerate(records):
            errors = sorted(validator.iter_errors(record), key=lambda e: e.path)
            if errors:
                raise EgoError(
                    f"{candidates_path}: record {i} (sample_id={record.get('sample_id')!r}) "
                    f"violates step1_candidates.schema.json: {errors[0].message}"
                )
    return records
