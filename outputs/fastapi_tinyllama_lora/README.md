---
base_model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
library_name: peft
pipeline_tag: text-generation
tags:
- base_model:adapter:TinyLlama/TinyLlama-1.1B-Chat-v1.0
- lora
- transformers
---

# FastAPI TinyLlama LoRA Adapter

This directory contains the exported adapter artifacts produced by the training pipeline.

## Summary

The adapter is tuned to answer FastAPI questions in a concise, code-oriented style.
It is intended to be loaded by the serving layer in this repository and is not a standalone base model checkpoint.

## Included Artifacts

- `adapter_config.json`
- `adapter_model.safetensors`
- `chat_template.jinja`
- tokenizer files required for inference

## Usage

Train the adapter with `python -m src.train_lora`, then serve it with `python main.py --serve`.

## Notes

- Base model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
- Technique: LoRA fine-tuning via PEFT
- Intended use: internal inference and local experimentation
- Runtime configuration is loaded from the root `.env` file if present.