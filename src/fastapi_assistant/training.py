"""LoRA supervised fine-tuning entrypoint.

Loads the base model, attaches LoRA adapters, fine-tunes on the FastAPI Q&A dataset
with completion-only loss and dynamic padding, and saves the adapter artifacts to
``settings.output_dir``. Runs on CPU for smoke tests and on a single free GPU
(Kaggle T4/P100) for real runs, using fp16/bf16 mixed precision when available.

Run via ``fastapi-assistant train`` or ``make train``.
"""

from __future__ import annotations

import json
import logging
import shutil
from typing import Any

import torch
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    set_seed,
)

from ._compat import dtype_kwargs, warmup_kwargs
from .config import Settings, get_settings
from .data import load_raw_dataset, tokenize_dataset
from .logging_config import configure_logging

log = logging.getLogger(__name__)


def _precision() -> tuple[torch.dtype, bool, bool]:
    """Return (load dtype, fp16, bf16) for the available hardware."""
    if not torch.cuda.is_available():
        return torch.float32, False, False
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16, False, True
    return torch.float16, True, False  # e.g. T4 / P100


def prepare_model_and_tokenizer(settings: Settings):
    """Load the base model + tokenizer and wrap the model with LoRA adapters."""
    tokenizer = AutoTokenizer.from_pretrained(settings.base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_dtype, _, _ = _precision()
    model: Any = AutoModelForCausalLM.from_pretrained(
        settings.base_model_name,
        device_map="auto" if torch.cuda.is_available() else None,
        **dtype_kwargs(load_dtype),
    )
    # Disable KV cache during training (incompatible with gradient checkpointing
    # and unnecessary for the forward/backward pass).
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=settings.lora_r,
        lora_alpha=settings.lora_alpha,
        lora_dropout=settings.lora_dropout,
        target_modules=settings.lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    # PEFT keeps the trainable adapter weights in fp32 even when the frozen base is
    # fp16/bf16, which is what makes mixed-precision LoRA numerically stable.
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, tokenizer


def train(settings: Settings | None = None) -> dict:
    """Run the full fine-tuning pipeline, persist the adapter, return final metrics."""
    settings = settings or get_settings()
    set_seed(settings.seed)
    settings.output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading dataset")
    raw_ds = load_raw_dataset(settings)
    validation = raw_ds["validation"] if "validation" in raw_ds else None
    log.info(
        "Train rows: %d, validation rows (from the training file): %d",
        len(raw_ds["train"]),
        len(validation) if validation is not None else 0,
    )

    log.info("Loading base model %s and applying LoRA", settings.base_model_name)
    model, tokenizer = prepare_model_and_tokenizer(settings)

    log.info("Tokenizing dataset")
    tokenized_ds = tokenize_dataset(tokenizer, raw_ds, settings)

    _, fp16, bf16 = _precision()
    training_args = TrainingArguments(
        output_dir=str(settings.output_dir),
        per_device_train_batch_size=settings.batch_size,
        per_device_eval_batch_size=settings.batch_size,
        gradient_accumulation_steps=settings.gradient_accumulation_steps,
        num_train_epochs=settings.num_epochs,
        learning_rate=settings.learning_rate,
        lr_scheduler_type="cosine",
        **warmup_kwargs(0.03),
        weight_decay=0.0,
        fp16=fp16,
        bf16=bf16,
        logging_steps=5,
        save_strategy="epoch",
        # Checkpoint selection uses the validation slice of the training file, never
        # the held-out eval set (that one only scores finished models).
        eval_strategy="epoch" if validation is not None else "no",
        save_total_limit=1,
        load_best_model_at_end=validation is not None,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        seed=settings.seed,
    )

    # Pads input_ids/attention_mask and pads labels with -100 so masked prompt
    # tokens stay masked after batching.
    collator = DataCollatorForSeq2Seq(
        tokenizer,
        padding=True,
        label_pad_token_id=-100,
        return_tensors="pt",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["validation"] if validation is not None else None,
        data_collator=collator,
    )

    log.info("Starting training")
    trainer.train()
    final_metrics = trainer.evaluate() if validation is not None else {}

    log.info("Saving adapter to %s", settings.output_dir)
    trainer.save_model(str(settings.output_dir))
    tokenizer.save_pretrained(str(settings.output_dir))
    (settings.output_dir / "training_metrics.json").write_text(
        json.dumps(
            {
                "base_model": settings.base_model_name,
                "train_rows": len(raw_ds["train"]),
                "validation_rows": len(validation) if validation is not None else 0,
                **{k: float(v) for k, v in final_metrics.items() if isinstance(v, (int, float))},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    # Checkpoints carry optimizer state (2x the adapter size); the final adapter is
    # all that serving and export need.
    for checkpoint in settings.output_dir.glob("checkpoint-*"):
        shutil.rmtree(checkpoint, ignore_errors=True)
    log.info("Training complete. Adapter saved to %s", settings.output_dir)
    return final_metrics


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    train(settings)


if __name__ == "__main__":
    main()
