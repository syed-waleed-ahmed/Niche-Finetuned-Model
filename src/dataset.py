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


def format_example(example: Dict) -> str:
    """
    Turn an instruction/input/output example into a single chat-style string.
    You can customize this to experiment with prompt formats.
    """
    instruction = example["instruction"]
    input_text = example.get("input", "")
    output = example["output"]

    if input_text:
        user_part = f"Instruction: {instruction}\nInput: {input_text}"
    else:
        user_part = f"Instruction: {instruction}"

    prompt = (
        f"<s>[SYSTEM] {SYSTEM_PROMPT}\n"
        f"[USER] {user_part}\n"
        f"[ASSISTANT] {output}</s>"
    )
    return prompt


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
        # For causal LM, labels are the same as input_ids
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    tokenized_ds = dataset.map(
        _tokenize,
        batched=False,
        remove_columns=dataset["train"].column_names,
    )
    return tokenized_ds