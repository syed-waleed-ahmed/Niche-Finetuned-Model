import os

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model

from .config import (
    BASE_MODEL_NAME,
    OUTPUT_DIR,
    BATCH_SIZE,
    GRADIENT_ACCUMULATION_STEPS,
    NUM_EPOCHS,
    LEARNING_RATE,
)
from .dataset import load_fastapi_dataset, tokenize_dataset


def prepare_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    # LoRA configuration
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, tokenizer


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading dataset...")
    raw_ds = load_fastapi_dataset()

    print("Loading model & tokenizer...")
    model, tokenizer = prepare_model_and_tokenizer()

    print("Tokenizing dataset...")
    tokenized_ds = tokenize_dataset(tokenizer, raw_ds)
    train_ds = tokenized_ds["train"]
    eval_ds = tokenized_ds["eval"]

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        evaluation_strategy="steps",
        eval_steps=50,
        save_steps=100,
        logging_steps=10,
        save_total_limit=2,
        weight_decay=0.0,
        bf16=torch.cuda.is_available(),  # use bf16 if possible
        fp16=False,  # we use bf16 or full precision
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
    )

    print("Starting training...")
    trainer.train()

    print("Saving model...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"Training complete. Model saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
