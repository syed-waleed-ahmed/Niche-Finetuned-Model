"""LoRA supervised fine-tuning entrypoint.

Loads TinyLlama, attaches LoRA adapters, fine-tunes on the FastAPI Q&A dataset with
completion-only loss and dynamic padding, and saves the adapter artifacts to
``settings.output_dir``. Run via ``python -m fastapi_assistant --train`` or
``make train``.
"""

from __future__ import annotations

import logging

import torch
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

from .config import Settings, get_settings
from .data import load_raw_dataset, tokenize_dataset
from .logging_config import configure_logging

log = logging.getLogger(__name__)


def prepare_model_and_tokenizer(settings: Settings):
    """Load the base model + tokenizer and wrap the model with LoRA adapters."""
    tokenizer = AutoTokenizer.from_pretrained(settings.base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    on_cuda = torch.cuda.is_available()
    model = AutoModelForCausalLM.from_pretrained(
        settings.base_model_name,
        torch_dtype=torch.float16 if on_cuda else torch.float32,
        device_map="auto" if on_cuda else None,
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
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, tokenizer


def train(settings: Settings | None = None) -> None:
    """Run the full fine-tuning pipeline and persist the adapter."""
    settings = settings or get_settings()
    settings.output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading dataset")
    raw_ds = load_raw_dataset(settings)

    log.info("Loading base model and applying LoRA")
    model, tokenizer = prepare_model_and_tokenizer(settings)

    log.info("Tokenizing dataset")
    tokenized_ds = tokenize_dataset(tokenizer, raw_ds, settings)

    training_args = TrainingArguments(
        output_dir=str(settings.output_dir),
        per_device_train_batch_size=settings.batch_size,
        per_device_eval_batch_size=settings.batch_size,
        gradient_accumulation_steps=settings.gradient_accumulation_steps,
        num_train_epochs=settings.num_epochs,
        learning_rate=settings.learning_rate,
        weight_decay=0.0,
        logging_dir=str(settings.output_dir / "logs"),
        logging_steps=1,
        save_strategy="epoch",
        eval_strategy="epoch",
        save_total_limit=1,
        report_to="none",
        load_best_model_at_end=False,
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
        eval_dataset=tokenized_ds["eval"],
        data_collator=collator,
    )

    log.info("Starting training")
    trainer.train()

    log.info("Saving adapter to %s", settings.output_dir)
    trainer.save_model(str(settings.output_dir))
    tokenizer.save_pretrained(str(settings.output_dir))
    log.info("Training complete. Adapter saved to %s", settings.output_dir)


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    train(settings)


if __name__ == "__main__":
    main()
