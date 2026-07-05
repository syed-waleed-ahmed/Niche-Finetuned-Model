# Graph Report - D:\Projects & Stuff\niche_finetuned_model  (2026-07-05)

## Corpus Check
- 18 files · ~8,518 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 123 nodes · 219 edges · 13 communities detected
- Extraction: 60% EXTRACTED · 40% INFERRED · 0% AMBIGUOUS · INFERRED: 87 edges (avg confidence: 0.63)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 11|Community 11]]
- [[_COMMUNITY_Community 12|Community 12]]

## God Nodes (most connected - your core abstractions)
1. `Settings` - 37 edges
2. `AssistantEngine` - 14 edges
3. `ModelNotReadyError` - 13 edges
4. `FakeEngine` - 11 edges
5. `create_app()` - 8 edges
6. `get_settings()` - 8 edges
7. `train()` - 8 edges
8. `FastAPI application factory and HTTP routes.  The app is built by :func:`create_` - 7 edges
9. `FastAPI dependency: resolve the engine from application state.` - 7 edges
10. `Build and configure a FastAPI application instance.` - 7 edges

## Surprising Connections (you probably didn't know these)
- `Model loading and text generation.  The :class:`AssistantEngine` owns the tokeni` --uses--> `Settings`  [INFERRED]
  D:\Projects & Stuff\niche_finetuned_model\src\fastapi_assistant\inference.py → D:\Projects & Stuff\niche_finetuned_model\src\fastapi_assistant\config.py
- `Raised when the model could not be loaded.` --uses--> `Settings`  [INFERRED]
  D:\Projects & Stuff\niche_finetuned_model\src\fastapi_assistant\inference.py → D:\Projects & Stuff\niche_finetuned_model\src\fastapi_assistant\config.py
- `Owns and serves a (base model + optional LoRA adapter) pair.` --uses--> `Settings`  [INFERRED]
  D:\Projects & Stuff\niche_finetuned_model\src\fastapi_assistant\inference.py → D:\Projects & Stuff\niche_finetuned_model\src\fastapi_assistant\config.py
- `Load the tokenizer and model exactly once (thread-safe, idempotent).` --uses--> `Settings`  [INFERRED]
  D:\Projects & Stuff\niche_finetuned_model\src\fastapi_assistant\inference.py → D:\Projects & Stuff\niche_finetuned_model\src\fastapi_assistant\config.py
- `Generate an answer for ``question``. Loads the model on first use.` --uses--> `Settings`  [INFERRED]
  D:\Projects & Stuff\niche_finetuned_model\src\fastapi_assistant\inference.py → D:\Projects & Stuff\niche_finetuned_model\src\fastapi_assistant\config.py

## Communities

### Community 0 - "Community 0"
Cohesion: 0.15
Nodes (20): BaseSettings, Strongly-typed application settings.      Field names map to upper-cased environ, Settings, build_tokenize_fn(), load_raw_dataset(), Dataset loading and tokenization for supervised fine-tuning.  The JSONL dataset, Load the train/eval JSONL splits as a :class:`~datasets.DatasetDict`., Return a per-example tokenization function with completion-only masking. (+12 more)

### Community 1 - "Community 1"
Cohesion: 0.14
Nodes (16): get_settings(), Return a process-wide cached :class:`Settings` instance., configure_logging(), JsonFormatter, Logging configuration.  Provides plain human-readable logs for local development, Render each log record as a single-line JSON object., Configure the root logger. Safe to call multiple times (idempotent handlers)., build_parser() (+8 more)

### Community 2 - "Community 2"
Cohesion: 0.21
Nodes (13): get_engine(), FastAPI application factory and HTTP routes.  The app is built by :func:`create_, FastAPI dependency: resolve the engine from application state., Build and configure a FastAPI application instance., BaseModel, FastAPI Niche Assistant.  A LoRA fine-tuned TinyLlama model specialized in the F, GenerateRequest, GenerateResponse (+5 more)

### Community 3 - "Community 3"
Cohesion: 0.18
Nodes (9): interactive_chat(), Run a simple REPL that answers FastAPI questions until the user exits., fake_engine(), FakeEngine, Shared pytest fixtures.  These tests never load a real model or hit the network, A stand-in engine that echoes input instead of running a model., ModelNotReadyError, Raised when the model could not be loaded. (+1 more)

### Community 4 - "Community 4"
Cohesion: 0.2
Nodes (8): create_app(), client(), make_settings(), Factory for isolated Settings that never read a real .env file., HTTP API tests using a fake engine (no model or network required)., test_api_key_required_when_configured(), test_generate_forwards_generation_params(), test_ready_reports_unready_model()

### Community 5 - "Community 5"
Cohesion: 0.2
Nodes (3): Interactive terminal chat for local experimentation., Centralized, validated application configuration.  All settings are read from th, Model loading and text generation.  The :class:`AssistantEngine` owns the tokeni

### Community 6 - "Community 6"
Cohesion: 0.28
Nodes (4): AssistantEngine, Generate an answer for ``question``. Loads the model on first use., Owns and serves a (base model + optional LoRA adapter) pair., Load the tokenizer and model exactly once (thread-safe, idempotent).

### Community 7 - "Community 7"
Cohesion: 0.28
Nodes (7): build_messages(), Shared prompt construction.  Training and inference MUST build prompts identical, Build the chat-format message list shared by training and inference.      Args:, Tests for shared prompt construction., test_build_messages_basic(), test_build_messages_strips_whitespace(), test_build_messages_with_context()

### Community 8 - "Community 8"
Cohesion: 1.0
Nodes (0): 

### Community 9 - "Community 9"
Cohesion: 1.0
Nodes (1): Load the JSONL files as a Hugging Face DatasetDict.     Each line has: instruct

### Community 10 - "Community 10"
Cohesion: 1.0
Nodes (1): Build the chat-style prompt used for both training and inference.

### Community 11 - "Community 11"
Cohesion: 1.0
Nodes (1): Turn an instruction/input/output example into a single chat-style string.     Y

### Community 12 - "Community 12"
Cohesion: 1.0
Nodes (1): Tokenize the dataset for causal LM training.

## Knowledge Gaps
- **18 isolated node(s):** `Centralized, validated application configuration.  All settings are read from th`, `Strongly-typed application settings.      Field names map to upper-cased environ`, `Return a process-wide cached :class:`Settings` instance.`, `Logging configuration.  Provides plain human-readable logs for local development`, `Render each log record as a single-line JSON object.` (+13 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 8`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 9`** (1 nodes): `Load the JSONL files as a Hugging Face DatasetDict.     Each line has: instruct`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 10`** (1 nodes): `Build the chat-style prompt used for both training and inference.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 11`** (1 nodes): `Turn an instruction/input/output example into a single chat-style string.     Y`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 12`** (1 nodes): `Tokenize the dataset for causal LM training.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Settings` connect `Community 0` to `Community 1`, `Community 2`, `Community 3`, `Community 4`, `Community 5`, `Community 6`?**
  _High betweenness centrality (0.445) - this node is a cross-community bridge._
- **Why does `FakeEngine` connect `Community 3` to `Community 0`, `Community 4`?**
  _High betweenness centrality (0.123) - this node is a cross-community bridge._
- **Why does `AssistantEngine` connect `Community 6` to `Community 0`, `Community 2`, `Community 3`, `Community 4`, `Community 5`?**
  _High betweenness centrality (0.118) - this node is a cross-community bridge._
- **Are the 33 inferred relationships involving `Settings` (e.g. with `FastAPI application factory and HTTP routes.  The app is built by :func:`create_` and `FastAPI dependency: resolve the engine from application state.`) actually correct?**
  _`Settings` has 33 INFERRED edges - model-reasoned connections that need verification._
- **Are the 8 inferred relationships involving `AssistantEngine` (e.g. with `FastAPI application factory and HTTP routes.  The app is built by :func:`create_` and `FastAPI dependency: resolve the engine from application state.`) actually correct?**
  _`AssistantEngine` has 8 INFERRED edges - model-reasoned connections that need verification._
- **Are the 9 inferred relationships involving `ModelNotReadyError` (e.g. with `FastAPI application factory and HTTP routes.  The app is built by :func:`create_` and `FastAPI dependency: resolve the engine from application state.`) actually correct?**
  _`ModelNotReadyError` has 9 INFERRED edges - model-reasoned connections that need verification._
- **Are the 5 inferred relationships involving `FakeEngine` (e.g. with `Settings` and `ModelNotReadyError`) actually correct?**
  _`FakeEngine` has 5 INFERRED edges - model-reasoned connections that need verification._