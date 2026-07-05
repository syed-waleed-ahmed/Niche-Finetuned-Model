"""Model loading and text generation.

The :class:`AssistantEngine` owns the tokenizer and model for the lifetime of the
process. It is designed for a single-process serving model:

* **Load once** — weights are loaded lazily and cached; a lock prevents duplicate
  concurrent loads.
* **Correct adapter handling** — the fine-tuned artifact on disk is a LoRA adapter,
  not a full model, so the base model is loaded first and the adapter is applied via
  ``peft.PeftModel``. If no adapter is present the base model is served directly
  (graceful degradation) with a warning.
* **Thread-safe generation** — HTTP requests are served from a threadpool, so a lock
  serializes ``model.generate`` calls, which are not safe to run concurrently on a
  single model instance. See ``ARCHITECTURE.md`` for the horizontal-scaling story.
"""

from __future__ import annotations

import logging
import threading

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import Settings
from .prompts import build_messages, encode_chat

log = logging.getLogger(__name__)


class ModelNotReadyError(RuntimeError):
    """Raised when the model could not be loaded."""


class AssistantEngine:
    """Owns and serves a (base model + optional LoRA adapter) pair."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model = None
        self._tokenizer = None
        self._uses_adapter = False
        self._load_lock = threading.Lock()
        self._generate_lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def uses_adapter(self) -> bool:
        return self._uses_adapter

    # -- loading -----------------------------------------------------------
    def load(self) -> None:
        """Load the tokenizer and model exactly once (thread-safe, idempotent)."""
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:  # re-check under lock
                return
            try:
                self._load_unlocked()
            except Exception as exc:  # surface a clear, typed error
                log.exception("Failed to load model")
                raise ModelNotReadyError(str(exc)) from exc

    def _load_unlocked(self) -> None:
        s = self._settings
        has_adapter = s.has_trained_adapter
        on_cuda = torch.cuda.is_available()
        dtype = torch.float16 if on_cuda else torch.float32
        device_map = "auto" if on_cuda else None

        # Prefer the adapter directory's tokenizer if it was saved alongside the
        # adapter (it may carry added special tokens); otherwise use the base model.
        tokenizer_src = (
            str(s.output_dir)
            if has_adapter and (s.output_dir / "tokenizer_config.json").is_file()
            else s.base_model_name
        )
        log.info("Loading tokenizer from %s", tokenizer_src)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_src)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        log.info("Loading base model %s (dtype=%s, cuda=%s)", s.base_model_name, dtype, on_cuda)
        model = AutoModelForCausalLM.from_pretrained(
            s.base_model_name,
            torch_dtype=dtype,
            device_map=device_map,
        )

        if has_adapter:
            from peft import PeftModel

            log.info("Applying LoRA adapter from %s", s.output_dir)
            model = PeftModel.from_pretrained(model, str(s.output_dir))
            self._uses_adapter = True
        else:
            log.warning(
                "No fine-tuned adapter found at %s; serving the base model. "
                "Run `python -m fastapi_assistant --train` (or `make train`) to fine-tune.",
                s.output_dir,
            )

        if device_map is None:
            model = model.to("cpu")
        model.eval()

        self._tokenizer = tokenizer
        self._model = model
        log.info(
            "Model ready (adapter=%s, device=%s)",
            self._uses_adapter,
            next(model.parameters()).device,
        )

    # -- inference ---------------------------------------------------------
    def generate(
        self,
        question: str,
        *,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        repetition_penalty: float | None = None,
    ) -> str:
        """Generate an answer for ``question``. Loads the model on first use."""
        if self._model is None:
            self.load()

        s = self._settings
        temp = s.model_temperature if temperature is None else temperature
        do_sample = temp > 0.0

        messages = build_messages(question)
        token_ids = encode_chat(self._tokenizer, messages, add_generation_prompt=True)
        input_ids = torch.tensor([token_ids], device=self._model.device)
        attention_mask = torch.ones_like(input_ids)

        gen_kwargs: dict[str, object] = {
            "max_new_tokens": max_new_tokens or s.model_max_new_tokens,
            "do_sample": do_sample,
            "top_p": top_p or s.model_top_p,
            "repetition_penalty": repetition_penalty or s.model_repetition_penalty,
            "pad_token_id": self._tokenizer.pad_token_id or self._tokenizer.eos_token_id,
            "eos_token_id": self._tokenizer.eos_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = temp

        # Serialize generation: a single model instance is not concurrency-safe.
        with self._generate_lock, torch.no_grad():
            output = self._model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_kwargs,
            )

        # Decode only the newly generated tokens, dropping the prompt.
        new_tokens = output[0, input_ids.shape[1]:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
