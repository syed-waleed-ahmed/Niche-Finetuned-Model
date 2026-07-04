from functools import lru_cache

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from .config import (
    MODEL_MAX_NEW_TOKENS,
    MODEL_REPETITION_PENALTY,
    MODEL_TEMPERATURE,
    MODEL_TOP_P,
    OUTPUT_DIR,
)
from .dataset import build_prompt


@lru_cache(maxsize=1)
def load_finetuned_model():
    tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        OUTPUT_DIR,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    return model, tokenizer


def generate_answer(question: str, max_new_tokens: int = 256) -> str:
    model, tokenizer = load_finetuned_model()
    model.eval()

    prompt = build_prompt(question)
    token_limit = max_new_tokens or MODEL_MAX_NEW_TOKENS

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=token_limit,
            do_sample=True,
            temperature=MODEL_TEMPERATURE,
            top_p=MODEL_TOP_P,
            repetition_penalty=MODEL_REPETITION_PENALTY,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "[ASSISTANT]" in full_text:
        answer = full_text.split("[ASSISTANT]", 1)[1].strip()
    else:
        answer = full_text
    return answer