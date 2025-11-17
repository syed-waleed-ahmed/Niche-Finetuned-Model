import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from .config import OUTPUT_DIR
from .dataset import SYSTEM_PROMPT


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

    user_prompt = f"Instruction: {question}"

    prompt = (
        f"<s>[SYSTEM] {SYSTEM_PROMPT}\n"
        f"[USER] {user_prompt}\n"
        f"[ASSISTANT]"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.4,
            top_p=0.9,
        )

    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # naive way: take everything after [ASSISTANT]
    if "[ASSISTANT]" in full_text:
        answer = full_text.split("[ASSISTANT]", 1)[1].strip()
    else:
        answer = full_text
    return answer
