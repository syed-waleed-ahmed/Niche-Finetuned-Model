---
base_model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
library_name: peft
pipeline_tag: text-generation
tags:
- base_model:adapter:TinyLlama/TinyLlama-1.1B-Chat-v1.0
- lora
- transformers
---

# Training Checkpoint

This directory contains an intermediate LoRA checkpoint produced during training.

## Purpose

The checkpoint captures training state for resuming or inspecting the fine-tuning process.
It is not intended to be treated as a published model artifact.

## Contents

- Adapter weights
- Optimizer state
- Scheduler state
- Trainer state
- RNG state

## Notes

- Base model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
- Technique: LoRA fine-tuning via PEFT
- Intended use: local checkpointing during training
- Runtime and training overrides should be managed through the root `.env` file or environment-specific deployment variables.