# Graph Report - D:\Projects & Stuff\niche_finetuned_model  (2026-11-13)

## Corpus Check
- 97 files · ~205,908 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 880 nodes · 2078 edges · 48 communities detected
- Extraction: 53% EXTRACTED · 47% INFERRED · 0% AMBIGUOUS · INFERRED: 970 edges (avg confidence: 0.67)
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
- [[_COMMUNITY_Community 13|Community 13]]
- [[_COMMUNITY_Community 14|Community 14]]
- [[_COMMUNITY_Community 15|Community 15]]
- [[_COMMUNITY_Community 16|Community 16]]
- [[_COMMUNITY_Community 17|Community 17]]
- [[_COMMUNITY_Community 18|Community 18]]
- [[_COMMUNITY_Community 19|Community 19]]
- [[_COMMUNITY_Community 20|Community 20]]
- [[_COMMUNITY_Community 21|Community 21]]
- [[_COMMUNITY_Community 22|Community 22]]
- [[_COMMUNITY_Community 23|Community 23]]
- [[_COMMUNITY_Community 24|Community 24]]
- [[_COMMUNITY_Community 25|Community 25]]
- [[_COMMUNITY_Community 26|Community 26]]
- [[_COMMUNITY_Community 27|Community 27]]
- [[_COMMUNITY_Community 28|Community 28]]
- [[_COMMUNITY_Community 29|Community 29]]
- [[_COMMUNITY_Community 30|Community 30]]
- [[_COMMUNITY_Community 31|Community 31]]
- [[_COMMUNITY_Community 32|Community 32]]
- [[_COMMUNITY_Community 33|Community 33]]
- [[_COMMUNITY_Community 34|Community 34]]
- [[_COMMUNITY_Community 35|Community 35]]
- [[_COMMUNITY_Community 36|Community 36]]
- [[_COMMUNITY_Community 37|Community 37]]
- [[_COMMUNITY_Community 38|Community 38]]
- [[_COMMUNITY_Community 39|Community 39]]
- [[_COMMUNITY_Community 40|Community 40]]
- [[_COMMUNITY_Community 41|Community 41]]
- [[_COMMUNITY_Community 42|Community 42]]
- [[_COMMUNITY_Community 43|Community 43]]
- [[_COMMUNITY_Community 44|Community 44]]
- [[_COMMUNITY_Community 45|Community 45]]
- [[_COMMUNITY_Community 46|Community 46]]
- [[_COMMUNITY_Community 47|Community 47]]

## God Nodes (most connected - your core abstractions)
1. `Settings` - 100 edges
2. `GenerationParams` - 43 edges
3. `post()` - 42 edges
4. `create_app()` - 39 edges
5. `make_settings()` - 35 edges
6. `handle()` - 32 edges
7. `str()` - 30 edges
8. `DocChunk` - 30 edges
9. `StreamEvent` - 28 edges
10. `FakeEngine` - 28 edges

## Surprising Connections (you probably didn't know these)
- `Interactive terminal chat for local experimentation.` --uses--> `Settings`  [INFERRED]
  D:\Projects & Stuff\niche_finetuned_model\src\fastapi_assistant\cli.py → D:\Projects & Stuff\niche_finetuned_model\src\fastapi_assistant\config.py
- `Run a simple REPL that answers FastAPI questions until the user exits.` --uses--> `Settings`  [INFERRED]
  D:\Projects & Stuff\niche_finetuned_model\src\fastapi_assistant\cli.py → D:\Projects & Stuff\niche_finetuned_model\src\fastapi_assistant\config.py
- `Settings` --uses--> `Model release tooling: merge the adapter, write a model card, publish to the Hub`  [INFERRED]
  D:\Projects & Stuff\niche_finetuned_model\src\fastapi_assistant\config.py → D:\Projects & Stuff\niche_finetuned_model\src\fastapi_assistant\export.py
- `Settings` --uses--> `Load base + adapter, merge LoRA weights into the base, save a full model.`  [INFERRED]
  D:\Projects & Stuff\niche_finetuned_model\src\fastapi_assistant\config.py → D:\Projects & Stuff\niche_finetuned_model\src\fastapi_assistant\export.py
- `Settings` --uses--> `Upload ``folder`` to ``repo_id`` and tag the resulting commit. Returns the commi`  [INFERRED]
  D:\Projects & Stuff\niche_finetuned_model\src\fastapi_assistant\config.py → D:\Projects & Stuff\niche_finetuned_model\src\fastapi_assistant\export.py

## Communities

### Community 0 - "Community 0"
Cohesion: 0.04
Nodes (53): AdmissionController, AdmissionRejected, QueueFullError, QueueTimeoutError, Admission control: bounded concurrency plus a bounded wait queue.  A replica can, Base class for requests the controller refused to run., The wait queue is full; retry later or elsewhere (HTTP 429)., The request waited too long for a free slot (HTTP 503). (+45 more)

### Community 1 - "Community 1"
Cohesion: 0.06
Nodes (77): create_app(), json(), Check that ``messages`` fit the context window; clamp the answer budget., client(), make_settings(), write_model_card(), FakeEngine, post() (+69 more)

### Community 2 - "Community 2"
Cohesion: 0.04
Nodes (72): _init_sentry(), BaseSettings, Run a simple REPL that answers FastAPI questions until the user exits., get_settings(), Return a process-wide cached :class:`Settings` instance., Strongly-typed application settings.      Field names map to upper-cased environ, Settings, A stand-in engine that echoes input instead of running a model. (+64 more)

### Community 3 - "Community 3"
Cohesion: 0.05
Nodes (61): BaseModel, fake_engine(), FakeEngine, A stand-in engine that echoes the last user message instead of running a model., _api_errors(), check_draft(), DraftBatch, generate_dataset() (+53 more)

### Community 4 - "Community 4"
Cohesion: 0.04
Nodes (43): cacheKey(), cacheWriteCommand(), canonicalJson(), isJsonContentType(), keyMarker(), markerWriteCommand(), parseCacheableBody(), wantsStream() (+35 more)

### Community 5 - "Community 5"
Cohesion: 0.04
Nodes (59): enabled(), dtype_kwargs(), _major_version(), Small cross-version compatibility helpers for the transformers library., Return the correct ``from_pretrained`` dtype kwarg for the installed version., Return the ``TrainingArguments`` kwarg for a warmup *ratio*.      transformers 5, warmup_kwargs(), _api_key_not_placeholder() (+51 more)

### Community 6 - "Community 6"
Cohesion: 0.06
Nodes (35): ApiKeyRecord, display_prefix(), Feedback, generate_api_key(), hash_api_key(), ModelNotReadyError, Principal, Return a new random API key (about 256 bits of entropy). (+27 more)

### Community 7 - "Community 7"
Cohesion: 0.07
Nodes (41): __getattr__(), FastAPI application factory and HTTP routes.  The app is built by :func:`creat, Build and configure a FastAPI application instance., FastAPI dependency: resolve the engine from application state., Build and configure a FastAPI application instance., _sources(), Authenticator, AuthError (+33 more)

### Community 8 - "Community 8"
Cohesion: 0.08
Nodes (43): Engine, GenerationParams, Sampling and budget parameters for one generation., interactive_chat(), Run a simple REPL that answers FastAPI questions until the user exits., code_execution(), code_validity(), EvalItem (+35 more)

### Community 9 - "Community 9"
Cohesion: 0.09
Nodes (20): readBodyCapped(), collect(), conversationReducer(), historyFor(), newAssistantTurn(), newId(), patch(), started() (+12 more)

### Community 10 - "Community 10"
Cohesion: 0.14
Nodes (27): _check_row(), content_words(), jaccard(), merge_shards(), near_duplicate_pairs(), normalize(), Dataset quality tooling: validation, leakage checks, statistics, shard merging., Append rows from ``shards`` to ``target``, skipping exact duplicates. (+19 more)

### Community 11 - "Community 11"
Cohesion: 0.1
Nodes (20): Interactive terminal chat for local experimentation., build_messages(), encode_chat(), format_user_content(), normalize_chat(), Shared prompt construction.  Training and inference MUST build prompts identic, Render a user turn, optionally followed by a context block.      This is the sin, Build the chat-format message list shared by training and inference.      Args: (+12 more)

### Community 12 - "Community 12"
Cohesion: 0.2
Nodes (12): ApiError, errorFromResponse(), getHealth(), headersFor(), isEventStream(), joinUrl(), metaFromResponse(), numberHeader() (+4 more)

### Community 13 - "Community 13"
Cohesion: 0.24
Nodes (7): PromptTooLongError, The prompt does not fit in the model's context window (HTTP 400)., JsonFormatter, Render each log record as a single-line JSON object., Request limits, context-window budgeting, error typing, headers, and new metrics, FakeEngine with a context window; one token per whitespace-separated word., test_json_logs_carry_cloud_logging_severity()

### Community 14 - "Community 14"
Cohesion: 0.36
Nodes (4): clamp(), loadSettings(), normalizeBaseUrl(), sanitizeSettings()

### Community 15 - "Community 15"
Cohesion: 0.5
Nodes (0): 

### Community 16 - "Community 16"
Cohesion: 0.67
Nodes (0): 

### Community 17 - "Community 17"
Cohesion: 1.0
Nodes (2): onKeyDown(), submit()

### Community 18 - "Community 18"
Cohesion: 0.67
Nodes (0): 

### Community 19 - "Community 19"
Cohesion: 0.67
Nodes (0): 

### Community 20 - "Community 20"
Cohesion: 1.0
Nodes (0): 

### Community 21 - "Community 21"
Cohesion: 1.0
Nodes (0): 

### Community 22 - "Community 22"
Cohesion: 1.0
Nodes (0): 

### Community 23 - "Community 23"
Cohesion: 1.0
Nodes (0): 

### Community 24 - "Community 24"
Cohesion: 1.0
Nodes (0): 

### Community 25 - "Community 25"
Cohesion: 1.0
Nodes (0): 

### Community 26 - "Community 26"
Cohesion: 1.0
Nodes (0): 

### Community 27 - "Community 27"
Cohesion: 1.0
Nodes (0): 

### Community 28 - "Community 28"
Cohesion: 1.0
Nodes (0): 

### Community 29 - "Community 29"
Cohesion: 1.0
Nodes (1): An empty API_KEY means "no static key"; a placeholder word is a mistake.

### Community 30 - "Community 30"
Cohesion: 1.0
Nodes (1): Treat placeholder values as "not configured".          Cloud Run mounts every se

### Community 31 - "Community 31"
Cohesion: 1.0
Nodes (0): 

### Community 32 - "Community 32"
Cohesion: 1.0
Nodes (0): 

### Community 33 - "Community 33"
Cohesion: 1.0
Nodes (0): 

### Community 34 - "Community 34"
Cohesion: 1.0
Nodes (0): 

### Community 35 - "Community 35"
Cohesion: 1.0
Nodes (0): 

### Community 36 - "Community 36"
Cohesion: 1.0
Nodes (0): 

### Community 37 - "Community 37"
Cohesion: 1.0
Nodes (0): 

### Community 38 - "Community 38"
Cohesion: 1.0
Nodes (0): 

### Community 39 - "Community 39"
Cohesion: 1.0
Nodes (0): 

### Community 40 - "Community 40"
Cohesion: 1.0
Nodes (1): Strongly-typed application settings.      Field names map to upper-cased envir

### Community 41 - "Community 41"
Cohesion: 1.0
Nodes (1): Return a process-wide cached :class:`Settings` instance.

### Community 42 - "Community 42"
Cohesion: 1.0
Nodes (1): Configure the root logger. Safe to call multiple times (idempotent handlers).

### Community 43 - "Community 43"
Cohesion: 1.0
Nodes (1): Build the chat-format message list shared by training and inference.      Args

### Community 44 - "Community 44"
Cohesion: 1.0
Nodes (1): Tokenize a chat message list into a flat list of token ids.      Normalizes ac

### Community 45 - "Community 45"
Cohesion: 1.0
Nodes (1): Input payload for POST /generate.

### Community 46 - "Community 46"
Cohesion: 1.0
Nodes (1): Response payload for POST /generate.

### Community 47 - "Community 47"
Cohesion: 1.0
Nodes (1): Response payload for /health and /ready.

## Knowledge Gaps
- **79 isolated node(s):** `Call the native /generate endpoint with httpx, including server-sent events.`, `Call the assistant with the official OpenAI Python SDK (blocking and streaming).`, `Kaggle GPU training pipeline: train -> evaluate -> gate -> merge -> GGUF -> publ`, ``name==version` lines for each package already installed (constraints format).`, `Admission control: bounded concurrency plus a bounded wait queue.  A replica can` (+74 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 20`** (2 nodes): `units.test.ts`, `buf()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 21`** (2 nodes): `notices.mjs`, `listDependencies()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 22`** (2 nodes): `CopyButton()`, `CopyButton.tsx`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 23`** (2 nodes): `Message.tsx`, `Answer()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 24`** (2 nodes): `MessageList.tsx`, `MessageList()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 25`** (2 nodes): `RetryBanner.tsx`, `RetryBanner()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 26`** (2 nodes): `SidePanel.tsx`, `SidePanel()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 27`** (2 nodes): `useHealth.ts`, `useHealth()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 28`** (2 nodes): `useSettings.ts`, `useSettings()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 29`** (1 nodes): `An empty API_KEY means "no static key"; a placeholder word is a mistake.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 30`** (1 nodes): `Treat placeholder values as "not configured".          Cloud Run mounts every se`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 31`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 32`** (1 nodes): `k6.js`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 33`** (1 nodes): `vite.config.ts`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 34`** (1 nodes): `main.tsx`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 35`** (1 nodes): `types.ts`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 36`** (1 nodes): `public-paths.test.ts`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 37`** (1 nodes): `settings.test.ts`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 38`** (1 nodes): `setup.ts`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 39`** (1 nodes): `snippets.test.ts`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 40`** (1 nodes): `Strongly-typed application settings.      Field names map to upper-cased envir`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 41`** (1 nodes): `Return a process-wide cached :class:`Settings` instance.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 42`** (1 nodes): `Configure the root logger. Safe to call multiple times (idempotent handlers).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 43`** (1 nodes): `Build the chat-format message list shared by training and inference.      Args`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 44`** (1 nodes): `Tokenize a chat message list into a flat list of token ids.      Normalizes ac`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 45`** (1 nodes): `Input payload for POST /generate.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 46`** (1 nodes): `Response payload for POST /generate.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 47`** (1 nodes): `Response payload for /health and /ready.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Settings` connect `Community 2` to `Community 0`, `Community 1`, `Community 3`, `Community 5`, `Community 6`, `Community 7`, `Community 8`, `Community 10`, `Community 11`?**
  _High betweenness centrality (0.190) - this node is a cross-community bridge._
- **Why does `handle()` connect `Community 4` to `Community 9`, `Community 5`, `Community 1`?**
  _High betweenness centrality (0.115) - this node is a cross-community bridge._
- **Why does `create_app()` connect `Community 1` to `Community 0`, `Community 2`, `Community 3`, `Community 6`, `Community 7`, `Community 8`?**
  _High betweenness centrality (0.102) - this node is a cross-community bridge._
- **Are the 96 inferred relationships involving `Settings` (e.g. with `FastAPI application factory and HTTP routes.  The app is built by :func:`creat` and `Build and configure a FastAPI application instance.`) actually correct?**
  _`Settings` has 96 INFERRED edges - model-reasoned connections that need verification._
- **Are the 41 inferred relationships involving `GenerationParams` (e.g. with `EvalItem` and `ItemResult`) actually correct?**
  _`GenerationParams` has 41 INFERRED edges - model-reasoned connections that need verification._
- **Are the 41 inferred relationships involving `post()` (e.g. with `main()` and `create_app()`) actually correct?**
  _`post()` has 41 INFERRED edges - model-reasoned connections that need verification._
- **Are the 35 inferred relationships involving `create_app()` (e.g. with `get_settings()` and `create_engine()`) actually correct?**
  _`create_app()` has 35 INFERRED edges - model-reasoned connections that need verification._