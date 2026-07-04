from typing import Dict
from datasets import load_dataset
from transformers import PreTrainedTokenizerBase

from .config import TRAIN_PATH, EVAL_PATH, MAX_SEQ_LENGTH


def load_fastapi_dataset():
    """
    Load the JSONL files as a Hugging Face DatasetDict.
    Each line has: instruction, input, output.
    """
    data_files = {
        "train": TRAIN_PATH,
        "eval": EVAL_PATH,
    }
    dataset = load_dataset("json", data_files=data_files)
    return dataset


SYSTEM_PROMPT = (
    "You are a helpful assistant specialized in the FastAPI Python framework. "
    "You answer questions concisely and include code examples where appropriate."
)


def build_prompt(instruction: str, input_text: str = "", output: str | None = None) -> str:
    """
    Build the chat-style prompt used for both training and inference.
    """
    if input_text:
        user_part = f"Instruction: {instruction}\nInput: {input_text}"
    else:
        user_part = f"Instruction: {instruction}"

    prompt = (
        f"<s>[SYSTEM] {SYSTEM_PROMPT}\n"
        f"[USER] {user_part}\n"
        f"[ASSISTANT]"
    )

    if output is not None:
        prompt = f"{prompt} {output}</s>"

    return prompt


def format_example(example: Dict) -> str:
    """
    Turn an instruction/input/output example into a single chat-style string.
    You can customize this to experiment with prompt formats.
    """
    instruction = example["instruction"]
    input_text = example.get("input", "")
    output = example["output"]
    return build_prompt(instruction, input_text=input_text, output=output)


def tokenize_dataset(tokenizer: PreTrainedTokenizerBase, dataset):
    """
    Tokenize the dataset for causal LM training.
    """
    def _tokenize(example):
        text = format_example(example)
        tokenized = tokenizer(
            text,
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding="max_length",
        )
        # Ignore padding tokens during loss computation.
        tokenized["labels"] = [
            token_id if attention_mask == 1 else -100
            for token_id, attention_mask in zip(tokenized["input_ids"], tokenized["attention_mask"])
        ]
        return tokenized

    tokenized_ds = dataset.map(
        _tokenize,
        batched=False,
        remove_columns=dataset["train"].column_names,
    )
    return tokenized_ds