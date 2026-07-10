# Interfaces

Inter-stage interfaces are scaffolded in `schemas/`:

- `step1_candidates.schema.json`
- `step2_policy_output.schema.json`
- `memory_state.schema.json`
- `planning_trajectory.schema.json`

The schemas are intentionally minimal and should be expanded as implementation decisions become concrete.

## Step 1 candidates (implemented)

`ego.contracts.candidates.StepOneCandidateRecord` is the Python-side contract
for `step1_candidates.schema.json`. `to_dict()` emits the schema's required
`candidates` field as the ranked action-pair (verb, noun) list, plus
`verb_candidates` / `noun_candidates` (independent per-head Top-K) and `gt`
as additional properties the schema permits but doesn't require.
`pipelines/step1_to_step2.load_step1_candidates()` is the read-side seam:
it loads a `action_candidates.jsonl` file and validates every record against
the schema with `jsonschema`, so a malformed Step 1 export fails at the
boundary rather than silently corrupting Step 2 data.
