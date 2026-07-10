# EGO

EGO is a scaffold for a research project on egocentric action anticipation, VLM policy alignment, and memory-context dynamic planning. This repository currently defines the package layout, command-line interface, configuration templates, schemas, and documentation surface for the project.

## Project Overview

EGO is organized around three research stages:

1. Step 1: V-JEPA2 Action Anticipation
   - Estimate a probability distribution over likely immediate future actions from egocentric video.
   - Produce Top-K verb, noun, and action-pair candidates with likelihoods.
2. Step 2: VLM Alignment
   - Use the Step 1 action prior with task, memory, and visual context.
   - Align a Qwen3-VL-8B policy with SFT and GRPO.
3. Step 3: Memory-context Dynamic Planning
   - Track recent actions, completed actions, distribution shifts, and uncertainty.
   - Re-call the VLM only when trigger conditions indicate that replanning is needed.

## Pipeline

```text
Egocentric video
    |
    v
V-JEPA2 action anticipation
    |
    v
Top-K action probability distribution
    |
    v
Task-conditioned VLM policy
    |
    v
Memory and trigger-based dynamic planning
```

## Repository Structure

```text
requirements/      Dependency notes by project stage
configs/           YAML experiment and pipeline templates
src/ego/           Python package and CLI scaffold
scripts/           Shell wrappers around canonical CLI commands
pipelines/         Stage-connection entry points
schemas/           JSON Schema contracts for inter-stage artifacts
tests/             Placeholder test modules for future implementation
docs/              Architecture, interface, dataset, and evaluation notes
data/              Tracked manifest/demo placeholders only
outputs/           Generated outputs, ignored except .gitkeep
checkpoints/       Model artifacts, ignored except .gitkeep
third_party/       External model and dependency tracking templates
```

## Installation

```bash
git clone <repository-url>
cd EGO

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -e .
```

Install stage-specific dependency notes after reviewing the target environment:

```bash
pip install -r requirements/step1.txt
pip install -r requirements/step2.txt
pip install -r requirements/step3.txt
pip install -r requirements/dev.txt
```

## Environment Configuration

Copy `.env.example` to `.env` locally and fill in machine-specific paths and tokens. Do not commit `.env`, dataset paths, API tokens, Hugging Face tokens, checkpoints, or generated artifacts.

## External Model Management

The default VLM reference is:

```text
Qwen/Qwen3-VL-8B-Instruct
```

Original pretrained weights must not be copied into this source repository. EGO-trained artifacts may later be published separately, such as:

- SFT LoRA adapter
- GRPO noun-stage adapter
- GRPO action-stage adapter
- Merged model, if needed

V-JEPA2 or V-JEPA2.1 integration should be recorded in `third_party/versions.yaml` only after verification, including the official repository, installation method, tested commit hash, required checkpoint name, checkpoint source, and local checkpoint environment variable. Original V-JEPA checkpoints must not be committed.

TRL is an external Hugging Face dependency, not vendored source code. Planned usage:

```text
train_sft.py          -> TRL SFTTrainer
train_grpo_noun.py   -> TRL GRPOTrainer
train_grpo_action.py -> TRL GRPOTrainer
```

EGO-specific reward functions are planned under `src/ego/step2_vlm_alignment/rewards/`.

External model licenses and the EGO source repository license are separate. The EGO source license is currently to be determined.

## Canonical CLI Commands

Step 1:

```bash
ego step1 prepare \
  --config configs/step1/ek100_vjepa2.yaml

ego step1 train \
  --config configs/step1/ek100_vjepa2.yaml

ego step1 infer \
  --config configs/step1/inference.yaml

ego step1 evaluate \
  --config configs/step1/inference.yaml
```

Step 2:

```bash
ego step2 build-data \
  --config configs/step2/sft_qwen3vl.yaml

ego step2 sft \
  --config configs/step2/sft_qwen3vl.yaml

ego step2 grpo-noun \
  --config configs/step2/grpo_stage1_noun.yaml

ego step2 grpo-action \
  --config configs/step2/grpo_stage2_action.yaml

ego step2 evaluate \
  --config configs/step2/grpo_stage2_action.yaml
```

Step 3:

```bash
ego step3 run \
  --config configs/step3/planning_eval.yaml

ego step3 evaluate \
  --config configs/step3/planning_eval.yaml
```

End-to-end:

```bash
ego pipeline run \
  --config configs/pipeline/ego_end_to_end.yaml

ego pipeline smoke-test \
  --config configs/pipeline/ego_end_to_end.yaml
```

All commands currently parse arguments, print the config path, and report that implementation is pending.

## Shell Scripts

Shell wrappers in `scripts/` call the same CLI commands. They are convenience entry points for VS Code tasks or terminal workflows and do not contain research logic.

## Checkpoints And Outputs

`outputs/` and `checkpoints/` are ignored except for `.gitkeep` files. Do not commit generated metrics, logs, model weights, adapters, or dataset-derived artifacts.

## Dataset Policy

Raw datasets, processed datasets, frames, videos, feature caches, and local manifests with sensitive paths must not be committed. Track only public-safe templates or synthetic demo metadata.

## Testing

Current test files are placeholders that explicitly skip execution. Once implementation begins, replace the skips with focused tests for contracts, mappings, reward functions, memory updates, trigger policies, and stage interfaces.

```bash
pytest --collect-only
```

## Current Implementation Status

```text
Repository scaffold: complete
Step 1 implementation: pending
Step 2 implementation: pending
Step 3 implementation: pending
End-to-end pipeline: pending
```

## License

License to be determined. External model and dependency licenses must be reviewed separately from the EGO source repository license.
