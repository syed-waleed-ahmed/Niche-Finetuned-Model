# Graph Report - D:\Projects & Stuff\niche_finetuned_model  (2026-07-05)

## Corpus Check
- 19 files · ~19,148 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 133 nodes · 242 edges · 8 communities detected
- Extraction: 62% EXTRACTED · 38% INFERRED · 0% AMBIGUOUS · INFERRED: 92 edges (avg confidence: 0.64)
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

## God Nodes (most connected - your core abstractions)
1. `Settings` - 37 edges
2. `AssistantEngine` - 14 edges
3. `ModelNotReadyError` - 13 edges
4. `FakeEngine` - 11 edges
5. `create_app()` - 8 edges
6. `get_settings()` - 8 edges
7. `train()` - 8 edges
8. `FastAPI application factory and HTTP routes.  The app is built by :func:`creat` - 7 edges
9. `FastAPI dependency: resolve the engine from application state.` - 7 edges
10. `Build and configure a FastAPI application instance.` - 7 edges

## Surprising Connections (you probably didn't know these)
- `Settings` --uses--> `Dataset loading and tokenization for supervised fine-tuning.  The JSONL datase`  [INFERRED]
  D:\Projects & Stuff\niche_finetuned_model\src\fastapi_assistant\config.py → D:\Projects & Stuff\niche_finetuned_model\src\fastapi_assistant\data.py
- `Settings` --uses--> `Load the train/eval JSONL splits as a :class:`~datasets.DatasetDict`.`  [INFERRED]
  D:\Projects & Stuff\niche_finetuned_model\src\fastapi_assistant\config.py → D:\Projects & Stuff\niche_finetuned_model\src\fastapi_assistant\data.py
- `Settings` --uses--> `Model loading and text generation.  The :class:`AssistantEngine` owns the toke`  [INFERRED]
  D:\Projects & Stuff\niche_finetuned_model\src\fastapi_assistant\config.py → D:\Projects & Stuff\niche_finetuned_model\src\fastapi_assistant\inference.py
- `Settings` --uses--> `Raised when the model could not be loaded.`  [INFERRED]
  D:\Projects & Stuff\niche_finetuned_model\src\fastapi_assistant\config.py → D:\Projects & Stuff\niche_finetuned_model\src\fastapi_assistant\inference.py
- `Settings` --uses--> `Owns and serves a (base model + optional LoRA adapter) pair.`  [INFERRED]
  D:\Projects & Stuff\niche_finetuned_model\src\fastapi_assistant\config.py → D:\Projects & Stuff\niche_finetuned_model\src\fastapi_assistant\inference.py

## Communities

### Community 0 - "Community 0"
Cohesion: 0.12
Nodes (15): create_app(), client(), fake_engine(), FakeEngine, make_settings(), Shared pytest fixtures.  These tests never load a real model or hit the networ, A stand-in engine that echoes input instead of running a model., Factory for isolated Settings that never read a real .env file. (+7 more)

### Community 1 - "Community 1"
Cohesion: 0.14
Nodes (16): get_settings(), Return a process-wide cached :class:`Settings` instance., configure_logging(), JsonFormatter, Logging configuration.  Provides plain human-readable logs for local developme, Render each log record as a single-line JSON object., Configure the root logger. Safe to call multiple times (idempotent handlers)., build_parser() (+8 more)

### Community 2 - "Community 2"
Cohesion: 0.17
Nodes (17): BaseSettings, Strongly-typed application settings.      Field names map to upper-cased envir, Settings, build_tokenize_fn(), Return a per-example tokenization function with completion-only masking., Tokenize every split, dropping the original text columns., tokenize_dataset(), Tests for typed configuration. (+9 more)

### Community 3 - "Community 3"
Cohesion: 0.15
Nodes (10): interactive_chat(), Run a simple REPL that answers FastAPI questions until the user exits., dtype_kwargs(), _major_version(), Small cross-version compatibility helpers for the transformers library., Return the correct ``from_pretrained`` dtype kwarg for the installed version., AssistantEngine, Generate an answer for ``question``. Loads the model on first use. (+2 more)

### Community 4 - "Community 4"
Cohesion: 0.13
Nodes (7): Interactive terminal chat for local experimentation., Centralized, validated application configuration.  All settings are read from, load_raw_dataset(), Dataset loading and tokenization for supervised fine-tuning.  The JSONL datase, Load the train/eval JSONL splits as a :class:`~datasets.DatasetDict`., Model loading and text generation.  The :class:`AssistantEngine` owns the toke, Shared prompt construction.  Training and inference MUST build prompts identic

### Community 5 - "Community 5"
Cohesion: 0.21
Nodes (13): get_engine(), FastAPI application factory and HTTP routes.  The app is built by :func:`creat, FastAPI dependency: resolve the engine from application state., Build and configure a FastAPI application instance., BaseModel, FastAPI Niche Assistant.  A LoRA fine-tuned TinyLlama model specialized in the, GenerateRequest, GenerateResponse (+5 more)

### Community 6 - "Community 6"
Cohesion: 0.17
Nodes (13): build_messages(), encode_chat(), Build the chat-format message list shared by training and inference.      Args, Tokenize a chat message list into a flat list of token ids.      Normalizes ac, _DictTokenizer, _ListTokenizer, Tests for shared prompt construction., Mimics transformers >= 5, which returns a BatchEncoding (dict). (+5 more)

### Community 7 - "Community 7"
Cohesion: 1.0
Nodes (0): 

## Knowledge Gaps
- **18 isolated node(s):** `Centralized, validated application configuration.  All settings are read from`, `Strongly-typed application settings.      Field names map to upper-cased envir`, `Return a process-wide cached :class:`Settings` instance.`, `Logging configuration.  Provides plain human-readable logs for local developme`, `Render each log record as a single-line JSON object.` (+13 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 7`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Settings` connect `Community 2` to `Community 0`, `Community 1`, `Community 3`, `Community 4`, `Community 5`?**
  _High betweenness centrality (0.424) - this node is a cross-community bridge._
- **Why does `AssistantEngine` connect `Community 3` to `Community 0`, `Community 2`, `Community 4`, `Community 5`?**
  _High betweenness centrality (0.163) - this node is a cross-community bridge._
- **Why does `encode_chat()` connect `Community 6` to `Community 3`, `Community 4`?**
  _High betweenness centrality (0.121) - this node is a cross-community bridge._
- **Are the 33 inferred relationships involving `Settings` (e.g. with `FastAPI application factory and HTTP routes.  The app is built by :func:`creat` and `FastAPI dependency: resolve the engine from application state.`) actually correct?**
  _`Settings` has 33 INFERRED edges - model-reasoned connections that need verification._
- **Are the 8 inferred relationships involving `AssistantEngine` (e.g. with `FastAPI application factory and HTTP routes.  The app is built by :func:`creat` and `FastAPI dependency: resolve the engine from application state.`) actually correct?**
  _`AssistantEngine` has 8 INFERRED edges - model-reasoned connections that need verification._
- **Are the 9 inferred relationships involving `ModelNotReadyError` (e.g. with `FastAPI application factory and HTTP routes.  The app is built by :func:`creat` and `FastAPI dependency: resolve the engine from application state.`) actually correct?**
  _`ModelNotReadyError` has 9 INFERRED edges - model-reasoned connections that need verification._
- **Are the 5 inferred relationships involving `FakeEngine` (e.g. with `Settings` and `ModelNotReadyError`) actually correct?**
  _`FakeEngine` has 5 INFERRED edges - model-reasoned connections that need verification._