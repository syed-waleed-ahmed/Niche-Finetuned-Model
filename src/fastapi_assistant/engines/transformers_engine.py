"""torch + transformers backend: base model with an optional LoRA adapter.

The fine-tuned artifact is a LoRA *adapter*, not a full model, so the base model is
loaded first and the adapter applied with ``peft.PeftModel``. With no adapter the
base model is served (graceful degradation). This backend is the reference
implementation used for training-time evaluation and GPU serving; CPU production
serving uses the quantized llama.cpp backend.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterator, Sequence
from typing import Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
)

from .._compat import dtype_kwargs
from ..config import Settings
from ..hub import resolve_adapter_dir
from ..prompts import encode_chat
from .base import (
    BaseEngine,
    Deadline,
    FinishReason,
    GenerationParams,
    GenerationResult,
    Message,
    StreamEvent,
)

log = logging.getLogger(__name__)


class _DeadlineCriteria(StoppingCriteria):
    """Stop generation when the wall-clock budget is spent or the client left."""

    def __init__(self, deadline: Deadline) -> None:
        self.deadline = deadline

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs):
        stop = self.deadline.expired()
        return torch.full((input_ids.shape[0],), stop, dtype=torch.bool, device=input_ids.device)


class TransformersEngine(BaseEngine):
    backend = "transformers"

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings.base_model_name)
        self._settings = settings
        # Typed as Any: PEFT and transformers return several model/tokenizer classes
        # with no common public base that declares generate()/apply_chat_template().
        self._model: Any = None
        self._tokenizer: Any = None
        # A single transformers model instance is not safe to drive from several
        # threads at once, and on CPU parallel generation only adds contention.
        self._generate_lock = threading.Lock()

    def _load_impl(self) -> None:
        s = self._settings
        adapter_dir = resolve_adapter_dir(s)
        on_cuda = torch.cuda.is_available()
        dtype = torch.float16 if on_cuda else torch.float32
        device_map = "auto" if on_cuda else None

        # Prefer the tokenizer saved with the adapter (it may carry added tokens).
        tokenizer_src = (
            str(adapter_dir)
            if adapter_dir is not None and (adapter_dir / "tokenizer_config.json").is_file()
            else s.base_model_name
        )
        log.info("Loading tokenizer from %s", tokenizer_src)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_src)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        log.info("Loading base model %s (dtype=%s, cuda=%s)", s.base_model_name, dtype, on_cuda)
        model: Any = AutoModelForCausalLM.from_pretrained(
            s.base_model_name, device_map=device_map, **dtype_kwargs(dtype)
        )

        if adapter_dir is not None:
            from peft import PeftModel

            log.info("Applying LoRA adapter from %s", adapter_dir)
            model = PeftModel.from_pretrained(model, str(adapter_dir))
            self._uses_adapter = True
        elif s.use_adapter:
            log.warning(
                "No fine-tuned adapter found (OUTPUT_DIR=%s, ADAPTER_REPO_ID=%s); serving the "
                "base model. Run `fastapi-assistant train` to fine-tune.",
                s.output_dir,
                s.adapter_repo_id,
            )

        if device_map is None:
            model = model.to("cpu")
        model.eval()
        self._tokenizer = tokenizer
        self._model = model

    # -- context budget ----------------------------------------------------
    def context_window(self) -> int | None:
        self.load()
        window = getattr(self._model.config, "max_position_embeddings", None)
        return int(window) if window else None

    def count_prompt_tokens(self, messages: Sequence[Message]) -> int:
        self.load()
        return len(encode_chat(self._tokenizer, list(messages), add_generation_prompt=True))

    # -- helpers -----------------------------------------------------------
    def _prepare(self, messages: Sequence[Message], params: GenerationParams, deadline: Deadline):
        self.load()
        tok = self._tokenizer
        token_ids = encode_chat(tok, list(messages), add_generation_prompt=True)
        input_ids = torch.tensor([token_ids], device=self._model.device)
        kwargs: dict[str, object] = {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "max_new_tokens": params.max_new_tokens,
            "do_sample": params.do_sample,
            "repetition_penalty": params.repetition_penalty,
            "pad_token_id": tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,
            "eos_token_id": tok.eos_token_id,
            "stopping_criteria": StoppingCriteriaList([_DeadlineCriteria(deadline)]),
        }
        if params.do_sample:
            kwargs["temperature"] = params.temperature
            kwargs["top_p"] = params.top_p
        return input_ids, kwargs

    @staticmethod
    def _finish_reason(deadline: Deadline, n_new: int, params: GenerationParams) -> FinishReason:
        if deadline.reason is not None:
            return deadline.reason
        return "length" if n_new >= params.max_new_tokens else "stop"

    # -- inference ---------------------------------------------------------
    def generate(
        self,
        messages: Sequence[Message],
        params: GenerationParams,
        cancel: threading.Event | None = None,
    ) -> GenerationResult:
        deadline = Deadline(params.timeout_s, cancel)
        input_ids, kwargs = self._prepare(messages, params, deadline)
        with self._generate_lock, torch.no_grad():
            output = self._model.generate(**kwargs)
        new_tokens = output[0, input_ids.shape[1] :]
        text = self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return GenerationResult(
            text=text,
            prompt_tokens=int(input_ids.shape[1]),
            completion_tokens=int(new_tokens.shape[0]),
            finish_reason=self._finish_reason(deadline, int(new_tokens.shape[0]), params),
        )

    def stream(
        self,
        messages: Sequence[Message],
        params: GenerationParams,
        cancel: threading.Event | None = None,
    ) -> Iterator[StreamEvent]:
        cancel = cancel or threading.Event()
        deadline = Deadline(params.timeout_s, cancel)
        input_ids, kwargs = self._prepare(messages, params, deadline)
        streamer = TextIteratorStreamer(
            self._tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=None
        )
        kwargs["streamer"] = streamer
        error: list[BaseException] = []

        def _run() -> None:
            try:
                with torch.no_grad():
                    self._model.generate(**kwargs)
            except BaseException as exc:  # surfaced to the consumer below
                error.append(exc)
                streamer.end()

        with self._generate_lock:
            worker = threading.Thread(target=_run, name="generate-stream", daemon=True)
            worker.start()
            parts: list[str] = []
            try:
                for text in streamer:
                    if text:
                        parts.append(text)
                        yield StreamEvent(delta=text)
            finally:
                # Stop the generation thread if the consumer went away early.
                cancel.set()
                worker.join()
        if error:
            raise error[0]
        n_new = len(self._tokenizer.encode("".join(parts), add_special_tokens=False))
        yield StreamEvent(
            finish_reason=self._finish_reason(deadline, n_new, params),
            prompt_tokens=int(input_ids.shape[1]),
            completion_tokens=n_new,
        )
