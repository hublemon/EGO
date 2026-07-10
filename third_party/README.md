# Third-party Models And Dependencies

Original pretrained model weights are not stored in this repository.

## Qwen3-VL

The default VLM is referenced by Hugging Face model ID:

```text
Qwen/Qwen3-VL-8B-Instruct
```

Do not copy original weights into EGO. EGO-trained artifacts such as SFT LoRA adapters, GRPO noun-stage adapters, GRPO action-stage adapters, or merged models may be published later in a separate Hugging Face model repository.

## V-JEPA2 Or V-JEPA2.1

Record only verified integration metadata here:

- Official repository
- Installation method
- Tested commit hash
- Required checkpoint name
- Checkpoint download method
- Local checkpoint environment variable

Do not commit original checkpoints.

## TRL

TRL is a Hugging Face dependency, not vendored EGO source code. Planned trainer usage:

- `SFTTrainer`
- `GRPOTrainer`
- `DPOTrainer`
- `RewardTrainer`

EGO-specific reward functions are planned under `src/ego/step2_vlm_alignment/rewards/`.
