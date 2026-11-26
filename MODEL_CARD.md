---
language:
  - en
license: apache-2.0
base_model: Qwen/Qwen2.5-Coder-1.5B-Instruct
pipeline_tag: text-generation
tags:
  - fastapi
  - python
  - code
  - gguf
  - llama-cpp
  - lora
---

# Model card: FastAPI Niche Assistant

A small language model that answers questions about the
[FastAPI](https://fastapi.tiangolo.com/) web framework, served by the
`fastapi-assistant` service in this repository.

**Status (2026-11-15): the deployed model is `v2026.11.14-72d6b37`.** It is a LoRA
fine-tune of Qwen2.5-Coder-1.5B-Instruct, merged and quantized to GGUF. It passed both
release gates and replaced the previous release, `v2026.11.13-2b04940`, a fine-tune of
TinyLlama-1.1B-Chat (see [Previous releases](#previous-releases)). In the production
eval on the held-out set it scores 0.6663 against the previous release's 0.5463. It is
still a 1.5B model and often wrong; see [Limitations and risks](#limitations-and-risks).

| Repository on the Hugging Face Hub | Contents |
| --- | --- |
| [syed-waleed-ahmed/fastapi-assistant-gguf](https://huggingface.co/syed-waleed-ahmed/fastapi-assistant-gguf) | The served weights: the merged model as a Q4_K_M GGUF |
| [syed-waleed-ahmed/fastapi-assistant-lora](https://huggingface.co/syed-waleed-ahmed/fastapi-assistant-lora) | The LoRA adapter, for use with `transformers` and `peft` |

Both are tagged `v2026.11.14-72d6b37`; the tag is the release.

## Model details

| Field | Value |
| --- | --- |
| Project | FastAPI Niche Assistant, by Syed Waleed Ahmed |
| Base model | [Qwen/Qwen2.5-Coder-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct), by the Qwen team, Alibaba Cloud |
| Architecture | Decoder-only transformer (Qwen2 architecture), about 1.5B parameters |
| Upstream training data | Described on the base model's card; not audited by this project |
| Language | English |
| Weights license | Apache-2.0 (see [License](#license)) |
| Code license | MIT ([LICENSE](LICENSE)) |

### Deployed artifact

The deployed weights are pinned in [`model-release.json`](model-release.json):

| Field | Value |
| --- | --- |
| `model_version` | `v2026.11.14-72d6b37` |
| `finetuned` | `true` |
| Adapter | [syed-waleed-ahmed/fastapi-assistant-lora](https://huggingface.co/syed-waleed-ahmed/fastapi-assistant-lora) at tag `v2026.11.14-72d6b37` (`adapter_model.safetensors`, 73,911,112 bytes) |
| Repository | [syed-waleed-ahmed/fastapi-assistant-gguf](https://huggingface.co/syed-waleed-ahmed/fastapi-assistant-gguf) |
| File | `model-Q4_K_M.gguf` (986,048,160 bytes, about 0.99 GB) |
| Revision | tag `v2026.11.14-72d6b37` |
| SHA-256 | `a35c951777b7b791d7d8f372a9ff664118b7c37e72625bee03a93fbd0e9d4697` |
| Quantization | Q4_K_M, 4-bit, made with llama.cpp's `llama-quantize` from an fp16 GGUF of the merged model |
| Chat format | The ChatML chat template embedded in the GGUF (the manifest sets no `chat_format`), the same template used in training; see the stop sequences below |
| Runtime | llama.cpp through `llama-cpp-python` on CPU; context window 2,048 tokens by default (`LLAMACPP_N_CTX`) |

The service downloads the file at the pinned revision and, with `GGUF_SHA256` set (as
`fastapi-assistant export env model-release.json` does), refuses to load it if its
SHA-256 differs from the value above. The llama.cpp and remote engines stop at
`prompts.STOP_SEQUENCES`, which lists both the ChatML role markers (`<|im_end|>`,
`<|im_start|>`) and the Zephyr-style ones of the earlier TinyLlama releases
(`<|user|>`, `<|system|>`, `<|assistant|>`), so either kind of release serves
correctly.

The release has been live since 2026-11-15: Release run 43696171917 deployed it after
release pull request #53 was merged, and `/health` reports
`"model_version":"v2026.11.14-72d6b37"` and `"uses_adapter":true`. Reverting #53
restores the previous release.

### How the release was made

The release pipeline (`kaggle/pipeline.py`, started by `.github/workflows/train.yml`)
trained, gated and published this model on 2026-11-14, in Train run 43622001406 from
commit `72d6b37` (#44):

- **Method:** LoRA on `Qwen/Qwen2.5-Coder-1.5B-Instruct` with completion-only loss
  (the loss is computed on answer tokens only), using the same chat template at
  training, evaluation and serving time (`src/fastapi_assistant/prompts.py`).
- **Hyperparameters:** rank 16, alpha 32, dropout 0.05, target modules `q_proj`,
  `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`; 3 epochs;
  learning rate 2e-4 with a cosine schedule and 3% warm-up; maximum sequence length
  512; batch size 4 with 4 gradient-accumulation steps (effective batch 16).
- **Run:** on a free Kaggle T4 GPU in fp16. The checkpoint is selected by loss on a
  seeded 10% validation slice of the 502-row training file.
- **Export:** the adapter was merged into the base weights, converted to GGUF,
  quantized to Q4_K_M, and published to the Hugging Face Hub under the immutable tag
  `v2026.11.14-72d6b37`. Both repositories are labelled Apache-2.0 and carry a NOTICE
  (written by `export.write_model_card`) with the attribution to the base model.
- **Gate 1 (Kaggle):** the fine-tuned model's composite score on the held-out set had
  to exceed the base model's by at least `min_improvement` (0.005; the eval uses
  greedy decoding, so a tie counts as no improvement). Scored with `transformers` in
  fp16 on the GPU, in the same run, it reached 0.6716 against 0.5576 for the base
  Qwen2.5-Coder-1.5B-Instruct.
- **Gate 2 (CI, `.github/workflows/eval.yml`):** the quantized GGUF, run on llama.cpp
  as in production, must not score more than 0.02 below the model currently on
  `main`. It scored 0.6663 against 0.5463 for the previous release,
  `v2026.11.13-2b04940`, both scored in the same run.
- **Release:** pull request #53 updated `model-release.json` and `eval/baseline.json`.
  Merging it deployed the model; reverting it rolls back.

### Previous releases

| `model_version` | Model | Served | Score (production eval) |
| --- | --- | --- | --- |
| `v2026.11.13-2b04940` | LoRA fine-tune of TinyLlama-1.1B-Chat, Q4_K_M GGUF | From #33 (2026-11-13) until #53 | 0.5463 against the base TinyLlama's 0.4023 |
| `base-tinyllama-q4km` | The base TinyLlama-1.1B-Chat, unchanged | Until #33 | 0.4023 (CI), 0.4101 (laptop) |

**`v2026.11.13-2b04940` (TinyLlama fine-tune).** Base model
[TinyLlama/TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0),
by the TinyLlama project: a decoder-only transformer with 1.1B parameters, whose card
states that it "adopted exactly the same architecture and tokenizer as Llama 2", and
lists `cerebras/SlimPajama-627B`, `bigcode/starcoderdata`,
`HuggingFaceH4/ultrachat_200k` and `HuggingFaceH4/ultrafeedback_binarized` as its
training data. The file was `model-Q4_K_M.gguf` (667,815,296 bytes, SHA-256
`6bc3c2c1a0e6b2d082ec01d14b57f056bc1e474bf9035b93cbe0aef165efe380`) in the same
repository, at tag `v2026.11.13-2b04940`, with the TinyLlama chat template embedded in
the GGUF. It was trained on 2026-11-13 with LoRA rank 16, alpha 32, dropout 0.05 on
`q_proj`, `k_proj`, `v_proj` and `o_proj` (4,505,600 trainable parameters, 0.41% of
the model); 3 epochs; learning rate 2e-4; maximum sequence length 512; batch size 8
with 2 gradient-accumulation steps; on 315 training rows and 36 validation rows of the
351-row training file then in use. Training took 17 minutes on a free Kaggle GPU. The
training loss ended at 1.143; the validation loss fell from 1.176 after epoch 1 to
1.119 after epoch 3. Gate 1 (Kaggle, `transformers`): 0.538 against the base model's
0.383. Gate 2 (CI, llama.cpp): 0.5463 against the base model's 0.4023. Its NOTICE
attributes TinyLlama.

**`base-tinyllama-q4km` (base model).** It served the base model unchanged:
`tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf` from
[TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF](https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF)
at revision `52e7645ba7c309695bec7ac98f4f005b139cf465` (SHA-256
`9fecc3b3cd76bba89d504f29b616eedf7da85b96540e490ca5824d3f7d2776a0`), with the `zephyr`
chat format. Finding F-04 in
[docs/audit/INITIAL_AUDIT.md](docs/audit/INITIAL_AUDIT.md) (the live model was not
fine-tuned) was resolved when #33 replaced it.

When no release is configured, the image falls back to the official Qwen Q4_K_M GGUF
(`Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF`, pinned in the `Dockerfile`), the base model
without a fine-tune.

## Intended use

- Answering developer questions about FastAPI: routing, path and query parameters,
  request bodies, dependencies, security, middleware, testing, WebSockets, streaming,
  deployment and related topics. Answers are meant to be short: a lead sentence, a
  minimal code example, and a short note.
- Use through the project's HTTP API (`POST /generate`, or the OpenAI-compatible
  `POST /v1/chat/completions`), optionally with retrieval over the FastAPI 0.162.5
  documentation (`use_retrieval: true`), which returns the cited documentation pages.
- A worked example of fine-tuning, evaluating and serving a small model under a
  release gate.

The intended users are developers who can read and check Python code. Treat every
answer as a draft to verify against the [FastAPI documentation](https://fastapi.tiangolo.com/).

## Out-of-scope use

- Questions outside FastAPI and its immediate ecosystem. The system prompt asks the
  model to say briefly when a question is out of scope, but it does not do so
  reliably.
- Generating security-sensitive code (authentication, authorisation, cryptography,
  secret handling) for use without expert review.
- Running generated code automatically outside a sandbox.
- Any decision about people, and any medical, legal, or financial use.
- Languages other than English.
- Use as an authoritative reference for FastAPI behaviour or versions.

## Training data

The fine-tuning data is in this repository:

| File | Rows | Fields | With a code context (`input`) | With code in the answer | Answer length (mean / max chars) |
| --- | --- | --- | --- | --- | --- |
| `data/fastapi_qa_train.jsonl` | 502 | `instruction`, `input`, `output` | 68 | 477 | 837 / 1,391 |
| `data/fastapi_qa_eval.jsonl` (held out) | 108 | `id`, `category`, `instruction`, `input`, `output`, `must_include` | 20 | 104 | 766 / 1,316 |

Figures are from `fastapi-assistant data stats`. The eval rows span 41 categories (the
largest are security with 8 items, dependencies with 6, and openapi, debugging,
websockets, streaming, database and testing with 5 each). The training rows carry no
category labels. The previous release was trained on the first 351 training rows; the
eval set is the same for every release.

**Provenance.**

- The question-answer pairs were written for this project. The commit that grew the
  set from 32/8 to 351/108 rows (`3092ad9`) describes it as curated, states that most
  Python blocks were executed against FastAPI with `TestClient` while authoring, and
  notes that a few out-of-scope questions teach polite refusal. That commit is
  co-authored with Claude (Anthropic), and `docs/ARCHITECTURE.md` describes the data
  as "hand-written and Claude-drafted pairs from the pinned docs", reviewed before
  merging.
- The 151 rows added in #44 (rows 352 to 502) cover the FastAPI tutorial and advanced
  guide. They were written without access to the eval set, their code was run against
  FastAPI 0.162.5, and `data validate` dropped four drafts that were near-duplicates
  of eval questions. The eval set was not changed.
- The repository includes a drafting tool, `fastapi-assistant data generate`
  (`src/fastapi_assistant/datagen.py`). It sends one section of the FastAPI 0.162.5
  documentation per request to Claude (default model id `claude-opus-5-5`), asks for
  question-answer pairs grounded only in that section, and constrains the response to
  a JSON schema. Drafts are filtered locally (every code block must parse, answers
  are at most 1,400 characters, each answer must contain its own `must_include`
  identifiers, duplicates of existing rows are dropped), written to a separate file
  for human review, and enter the dataset only through `fastapi-assistant data merge`.
- Merging into the training set keeps only `instruction`, `input` and `output`, so the
  data files do not record which rows were hand-written and which were drafted.
- The FastAPI documentation used for grounding is MIT-licensed; see
  [docs/legal/LICENSE_AUDIT.md](docs/legal/LICENSE_AUDIT.md).

**Quality and leakage checks.** `fastapi-assistant data validate` runs in CI. It
checks the schema and length limits, that every Python block parses, that each eval
reference contains its own `must_include` identifiers, exact duplicates, and
near-duplicate leakage between eval and train (question word overlap with a Jaccard
threshold of 0.7, answer shingle overlap of 0.6). These checks are lexical, not
semantic.

**Personal data.** A pattern scan of both files found no email addresses other than
`example.com` placeholders. The data is technical question-and-answer text.

## Evaluation

### Method

`fastapi-assistant eval` (`src/fastapi_assistant/evaluation.py`) answers every
held-out question with greedy decoding and scores each answer against the reference:

- **`keyword_recall`:** the fraction of the item's `must_include` identifiers (API
  names any correct answer has to mention) present in the answer, case-insensitive.
- **`code_valid`:** scored only when the reference answer contains code (104 of 108
  items): the fraction of the answer's fenced Python blocks that parse. An answer with
  no code block scores 0.
- **`code_runs`:** optional (`--execute-code`): the fraction of blocks that run without
  error in a subprocess. Not measured in the baselines below.
- **`rouge_l`:** ROUGE-L F1 against the reference over code-aware tokens (identifiers,
  numbers, punctuation).
- **`score`:** the composite. For each item, the mean of the metrics that apply to it;
  then the mean over items.

Latency is wall-clock time per item. The release gate compares composite scores.

### Results (deployed model)

Setup: 108 held-out items; backend `llamacpp` (Q4_K_M GGUF); greedy decoding; 256 new
tokens at most; no retrieval. Both models were scored in the same run of the CI eval
gate (`.github/workflows/eval.yml`, on a GitHub-hosted runner) for release pull
request #53. The report is `eval/baseline.md`, and the per-item answers are in
`eval/baseline.json`.

| Metric | `v2026.11.14-72d6b37` (Qwen fine-tune, deployed) | `v2026.11.13-2b04940` (TinyLlama fine-tune, previous release) |
| --- | --- | --- |
| `score` (composite) | 0.6663 | 0.5463 |
| `keyword_recall` | 0.6852 | 0.4259 |
| `code_valid` | 0.8846 | 0.8462 |
| `code_runs` | not measured | not measured |
| `rouge_l` | 0.4488 | 0.3866 |
| `latency_p50_s` | 5.725 | 4.426 |
| `latency_p95_s` | 8.887 | 6.793 |

The composite rises by 0.120, most of it from keyword recall (+0.259): the new model
names the APIs a question needs far more often, where the TinyLlama fine-tune had not
improved on its own base model. Code validity and ROUGE-L also rise. The larger model
is slower: the median item takes 1.3 s longer on the CI runner.

Composite score by category, for categories with at least 5 items. The previous
release's column is from its own `eval/baseline.json` (in the git history).

| Category | Items | `v2026.11.14-72d6b37` | `v2026.11.13-2b04940` |
| --- | --- | --- | --- |
| security | 8 | 0.6955 | 0.5169 |
| dependencies | 6 | 0.7731 | 0.6588 |
| database | 5 | 0.4791 | 0.4272 |
| debugging | 5 | 0.8361 | 0.6948 |
| openapi | 5 | 0.5299 | 0.4545 |
| streaming | 5 | 0.4685 | 0.4318 |
| testing | 5 | 0.6082 | 0.5704 |
| websockets | 5 | 0.6942 | 0.4728 |

The lowest-scoring categories for the new model are logging (0.0865, 1 item),
deployment (0.3155, 3 items), rate limiting (0.4375, 2 items) and request state
(0.4629, 2 items). Categories with one to three items give noisy estimates.

Gate 1 on the Kaggle GPU scores the unquantized models with `transformers` in fp16,
in the same run:

| Metric | Fine-tuned | Base Qwen2.5-Coder-1.5B-Instruct |
| --- | --- | --- |
| `score` (composite) | 0.6716 | 0.5576 |
| `keyword_recall` | 0.7114 | 0.6219 |
| `code_valid` | 0.8558 | 0.7212 |
| `rouge_l` | 0.4548 | 0.3450 |

**Speed in production.** On Cloud Run (2 vCPU) the model generates about 7.6 tokens/s
(Grafana, mean over an hour). Two live answers took 14.2 s (104 tokens) and 26.3 s
(192 tokens).

### Results (previous release, TinyLlama fine-tune)

Same setup, scored in the CI eval gate for release pull request #33:

| Metric | `v2026.11.13-2b04940` (TinyLlama fine-tune) | Base model `base-tinyllama-q4km` |
| --- | --- | --- |
| `score` (composite) | 0.5463 | 0.4023 |
| `keyword_recall` | 0.4259 | 0.4321 |
| `code_valid` | 0.8462 | 0.4904 |
| `code_runs` | not measured | not measured |
| `rouge_l` | 0.3866 | 0.2891 |
| `latency_p50_s` | 4.945 | 6.722 |
| `latency_p95_s` | 7.621 | 7.846 |

The composite rose by 0.144, from code validity and ROUGE-L; keyword recall was
unchanged (-0.006). That fine-tune put a code block in 105 of 108 answers (the base
model, in its laptop run: 90), and its answers were about a third shorter (571
characters on average against 831). In the gate for #53 its latency was 4.426 s p50
and 6.793 s p95.

### Results (base TinyLlama, first release)

Setup: 108 held-out items; backend `llamacpp`; model
`TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF` (Q4_K_M); no adapter; greedy decoding;
256 new tokens at most; `model_version` `base-tinyllama-q4km`; laptop CPU. The
first column was `eval/baseline.json` until release #33 replaced it (it is in the
git history); the second is `eval/baseline-rag.json`.

| Metric | Base model | Base model + retrieval (`eval/baseline-rag.json`) |
| --- | --- | --- |
| `score` (composite) | 0.4101 | 0.4001 |
| `keyword_recall` | 0.4182 | 0.4985 |
| `code_valid` | 0.5385 | 0.4006 |
| `code_runs` | not measured | not measured |
| `rouge_l` | 0.2827 | 0.3026 |
| `latency_p50_s` | 7.34 | 11.629 |
| `latency_p95_s` | 11.605 | 23.178 |

Retrieval raised keyword recall by 0.0803 and ROUGE-L by 0.0199 but lowered code
validity by 0.1379, so the composite fell by 0.0100, and median latency rose by
about 58%. Retrieval is therefore opt-in per request. Retrieval has not been
measured with a fine-tuned model yet.

### The fine-tune against its own base model, paired

The table above compares two fine-tunes of different base models, trained on
different amounts of data, so it cannot say how much the fine-tuning itself
added. Gate 1 can: it scores the fine-tune and the untuned
Qwen2.5-Coder-1.5B-Instruct on the same 108 items, on the same Kaggle T4
(`transformers`, fp16), in the training run for `v2026.11.14-72d6b37`. Both
per-item reports are committed in `eval/reference/`, and
`scripts/paired_bootstrap.py` computes the difference per item with a 95%
interval from 10,000 resamples:

| Metric | Fine-tune | Base | Difference (95% interval) | Items better / worse |
| --- | --- | --- | --- | --- |
| `score` (composite) | 0.6716 | 0.5576 | +0.114 (0.069 to 0.160) | 84 / 24 |
| `keyword_recall` | 0.7114 | 0.6219 | +0.090 (0.022 to 0.161) | 31 / 16 |
| `code_valid` | 0.8558 | 0.7212 | +0.135 (0.039 to 0.231) | 21 / 7 |
| `rouge_l` | 0.4548 | 0.3450 | +0.110 (0.091 to 0.129) | 94 / 14 |

Every interval excludes zero, so on these metrics the fine-tune is better
than its base on this set. The intervals cover the sampling of items only:
there is one training run with one seed, so the variation between runs is
unknown, and the caveats below apply to every row.

```bash
python scripts/paired_bootstrap.py eval/reference/gate1-finetune-qwen.json eval/reference/gate1-base-qwen.json
```

### Caveats

- All metrics are automatic and lexical. They reward naming the right APIs, parseable
  code, and overlap with a reference; they do not establish that an answer is
  correct. No human evaluation has been recorded.
- The eval set was written by the same process as the training set. The leakage
  checks are lexical, so paraphrased overlap could go undetected.
- Training never reads the held-out eval file. The best checkpoint is chosen by loss
  on a seeded 10% validation slice of the training file (`validation_fraction`), so
  the eval set scores finished models only. An earlier version of `training.py` used
  the eval file for checkpoint selection; this was fixed before any fine-tune was
  trained.

## Limitations and risks

- **Incorrect answers stated confidently.** The model invents parameters and APIs and
  misdescribes how FastAPI works, often in well-formatted, parseable code, which makes
  the errors easier to miss. Live, after the release, one answer described a header as
  a query parameter and another over-complicated returning a 201 response. The
  previous release made similar errors in its eval answers (in the git history): item
  `core_http-003` passed a nonexistent `item_field` argument to `Body`, and item
  `async_io-014` called `JSONResponse` with a `text` argument it does not take.
- **Small, quantized model.** At about 1.5B parameters and 4-bit quantization,
  quality is limited: the composite score is 0.6663, and keyword recall, the share of
  the required API names an answer mentions, is 0.6852, so about a third of them are
  still missing.
- **Slow on a CPU.** About 7.6 tokens/s on the 2-vCPU Cloud Run instance: two live
  answers took 14.2 s (104 tokens) and 26.3 s (192 tokens). In the CI eval its median
  answer took 5.725 s against 4.426 s for the previous, smaller release.
- **Outdated idioms.** Three of the 108 eval answers in `eval/baseline.json`
  (`async_io-002`, `async_io-015`, `async_io-022`) use `@app.on_event` instead of a
  lifespan handler, which the dataset teaches. The set is small, so other older idioms
  can appear.
- **Retrieval trade-off.** With the base TinyLlama, retrieved documentation in the
  prompt led the model to copy code fragments it could not adapt, which lowered code
  validity. Retrieval has not been measured with this model.
- **Short context.** The default context window is 2,048 tokens; longer prompts are
  rejected with HTTP 400 `context_length_exceeded`.
- **Generated code is untrusted.** It can be insecure or wrong. Review it before use
  and run it only in a sandbox.
- **Prompt injection.** User input can steer the model's output. The model has no
  tools and no data access, so the effect is limited to the text it returns.
- **Bias.** The base model was pre-trained on large web and code corpora and can
  reflect their biases. The narrow FastAPI domain reduces, but does not remove, this
  risk.
- **Coverage.** The training set has 502 examples in English, so many FastAPI topics
  are covered thinly or not at all.

## How to use

```bash
pip install -e ".[serve]"
fastapi-assistant export env model-release.json > .env   # pinned weights and settings
fastapi-assistant serve                                  # http://localhost:8000/docs
```

```bash
curl -s localhost:8000/generate -H 'X-API-Key: <key>' -H 'Content-Type: application/json' \
  -d '{"question": "How do I return a 201 from a POST endpoint?", "temperature": 0}'
```

See the [README](README.md) for the OpenAI-compatible API, streaming, and retrieval.

## Environmental impact

Not measured in energy or emissions. The fine-tune trained on one free Kaggle T4 GPU
(the previous release trained for 17 minutes on a free Kaggle GPU, and its whole
kernel run took about an hour). The deployed model runs on CPU.

## License

- **Weights:** the base model, Qwen2.5-Coder-1.5B-Instruct, is published by the Qwen
  team, Alibaba Cloud, under Apache-2.0, as declared on its Hugging Face card and in
  its `LICENSE` file (see [docs/legal/LICENSE_AUDIT.md](docs/legal/LICENSE_AUDIT.md)).
  The fine-tuned GGUF and adapter are published under Apache-2.0, with a NOTICE that
  attributes the base model and describes the modifications; redistributions must
  keep both. The earlier releases are derived from TinyLlama-1.1B-Chat (Apache-2.0),
  and their NOTICE attributes TinyLlama.
- **Code:** MIT ([LICENSE](LICENSE)).
- **Dataset:** distributed in this repository under its MIT license. Before
  publishing the dataset separately, read the open questions in
  [docs/legal/LICENSE_AUDIT.md](docs/legal/LICENSE_AUDIT.md).
- **Retrieval:** the embedding model `BAAI/bge-small-en-v1.5` (MIT) and the FastAPI
  documentation (MIT) are used for retrieval only and are not part of the generator's
  weights.

## Citation

To cite this project, use [CITATION.cff](CITATION.cff). To cite the base model, see
the citation on the
[Qwen2.5-Coder-1.5B-Instruct model card](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct).
The earlier releases' base model is described on the
[TinyLlama model card](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0).
