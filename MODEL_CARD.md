---
language:
  - en
license: apache-2.0
base_model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
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

**Status (2026-11-13): the deployed model is the first fine-tuned release,
`v2026.11.13-2b04940`.** It is a LoRA fine-tune of TinyLlama-1.1B-Chat, merged and
quantized to GGUF. It passed both release gates and replaced the base model, which
was served until then (finding F-04 in
[docs/audit/INITIAL_AUDIT.md](docs/audit/INITIAL_AUDIT.md), now resolved). On the
held-out set it scores 0.546 against the base model's 0.402. It is still a 1.1B model
and often wrong; see [Limitations and risks](#limitations-and-risks).

| Repository on the Hugging Face Hub | Contents |
| --- | --- |
| [syed-waleed-ahmed/fastapi-assistant-gguf](https://huggingface.co/syed-waleed-ahmed/fastapi-assistant-gguf) | The served weights: the merged model as a Q4_K_M GGUF |
| [syed-waleed-ahmed/fastapi-assistant-lora](https://huggingface.co/syed-waleed-ahmed/fastapi-assistant-lora) | The LoRA adapter, for use with `transformers` and `peft` |

Both are tagged `v2026.11.13-2b04940`; the tag is the release.

## Model details

| Field | Value |
| --- | --- |
| Project | FastAPI Niche Assistant, by Syed Waleed Ahmed |
| Base model | [TinyLlama/TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0), by the TinyLlama project |
| Architecture | Decoder-only transformer, 1.1B parameters. The TinyLlama card states that it "adopted exactly the same architecture and tokenizer as Llama 2". |
| Upstream training data | As listed on the base model's card: `cerebras/SlimPajama-627B`, `bigcode/starcoderdata`, `HuggingFaceH4/ultrachat_200k`, `HuggingFaceH4/ultrafeedback_binarized` |
| Language | English |
| Weights license | Apache-2.0 (see [License](#license)) |
| Code license | MIT ([LICENSE](LICENSE)) |

### Deployed artifact

The deployed weights are pinned in [`model-release.json`](model-release.json):

| Field | Value |
| --- | --- |
| `model_version` | `v2026.11.13-2b04940` |
| `finetuned` | `true` |
| Adapter | [syed-waleed-ahmed/fastapi-assistant-lora](https://huggingface.co/syed-waleed-ahmed/fastapi-assistant-lora) at tag `v2026.11.13-2b04940` |
| Repository | [syed-waleed-ahmed/fastapi-assistant-gguf](https://huggingface.co/syed-waleed-ahmed/fastapi-assistant-gguf) |
| File | `model-Q4_K_M.gguf` (667,815,296 bytes) |
| Revision | tag `v2026.11.13-2b04940` |
| SHA-256 | `6bc3c2c1a0e6b2d082ec01d14b57f056bc1e474bf9035b93cbe0aef165efe380` |
| Quantization | Q4_K_M, 4-bit, made with llama.cpp's `llama-quantize` from an fp16 GGUF of the merged model |
| Chat format | The chat template embedded in the GGUF (the manifest sets no `chat_format`), the same TinyLlama template used in training |
| Runtime | llama.cpp through `llama-cpp-python` on CPU; context window 2,048 tokens by default (`LLAMACPP_N_CTX`) |

The service downloads the file at the pinned revision and, with `GGUF_SHA256` set (as
`fastapi-assistant export env model-release.json` does), refuses to load it if its
SHA-256 differs from the value above. The hash also matches the Hugging Face Hub's
LFS metadata for that file at that tag.

The previous release, `base-tinyllama-q4km`, served the base model unchanged:
`tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf` from
[TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF](https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF)
at revision `52e7645ba7c309695bec7ac98f4f005b139cf465` (SHA-256
`9fecc3b3cd76bba89d504f29b616eedf7da85b96540e490ca5824d3f7d2776a0`), with the `zephyr`
chat format. Reverting release pull request #33 restores it.

### How the release was made

The release pipeline (`kaggle/pipeline.py`, started by `.github/workflows/train.yml`)
trained, gated and published this model on 2026-11-13:

- **Method:** LoRA on `TinyLlama/TinyLlama-1.1B-Chat-v1.0` with completion-only loss
  (the loss is computed on answer tokens only), using the same chat template at
  training, evaluation and serving time (`src/fastapi_assistant/prompts.py`).
- **Hyperparameters:** rank 16, alpha 32, dropout 0.05, target modules `q_proj`,
  `k_proj`, `v_proj`, `o_proj` (4,505,600 trainable parameters, 0.41% of the model);
  3 epochs; learning rate 2e-4 with a cosine schedule and 3% warm-up; maximum
  sequence length 512; batch size 8 with 2 gradient-accumulation steps.
- **Run:** 315 training rows and 36 validation rows (a seeded 10% slice of the
  training file). Training took 17 minutes on a free Kaggle GPU. The training loss
  ended at 1.143; the validation loss fell from 1.176 after epoch 1 to 1.119 after
  epoch 3, and the checkpoint with the lowest validation loss was kept.
- **Export:** the adapter was merged into the base weights, converted to GGUF,
  quantized to Q4_K_M, and published to the Hugging Face Hub under the immutable tag
  `v2026.11.13-2b04940`. Both repositories are labelled Apache-2.0 and carry a NOTICE
  with the attribution to TinyLlama.
- **Gate 1 (Kaggle):** the fine-tuned model's composite score on the held-out set had
  to exceed the base model's by at least `min_improvement` (0.005; the eval uses
  greedy decoding, so a tie counts as no improvement). Scored with `transformers` on
  the GPU, it reached 0.538 against 0.383.
- **Gate 2 (CI, `.github/workflows/eval.yml`):** the quantized GGUF, run on llama.cpp
  as in production, must not score more than 0.02 below the model currently on
  `main`. It scored 0.5463 against the base model's 0.4023, both scored in the same
  run.
- **Release:** pull request #33 updated `model-release.json` and `eval/baseline.json`.
  Merging it deployed the model; reverting it rolls back.

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
| `data/fastapi_qa_train.jsonl` | 351 | `instruction`, `input`, `output` | 49 | 328 | 735 / 1,386 |
| `data/fastapi_qa_eval.jsonl` (held out) | 108 | `id`, `category`, `instruction`, `input`, `output`, `must_include` | 20 | 104 | 766 / 1,316 |

Figures are from `fastapi-assistant data stats`. The eval rows span 41 categories (the
largest are security with 8 items, dependencies with 6, and openapi, debugging,
websockets, streaming, database and testing with 5 each). The training rows carry no
category labels.

**Provenance.**

- The question-answer pairs were written for this project. The commit that grew the
  set from 32/8 to 351/108 rows (`3092ad9`) describes it as curated, states that most
  Python blocks were executed against FastAPI with `TestClient` while authoring, and
  notes that a few out-of-scope questions teach polite refusal. That commit is
  co-authored with Claude (Anthropic), and `docs/ARCHITECTURE.md` describes the data
  as "hand-written and Claude-drafted pairs from the pinned docs", reviewed before
  merging.
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

### Results (fine-tuned model, as deployed)

Setup: 108 held-out items; backend `llamacpp`; greedy decoding; 256 new tokens at
most; no retrieval. Both models were scored in the same run of the CI eval gate
(`.github/workflows/eval.yml`, on a GitHub-hosted runner) for release pull request
#33. The fine-tuned values are in `eval/baseline.json`.

| Metric | Fine-tuned `v2026.11.13-2b04940` | Base model `base-tinyllama-q4km` |
| --- | --- | --- |
| `score` (composite) | 0.5463 | 0.4023 |
| `keyword_recall` | 0.4259 | 0.4321 |
| `code_valid` | 0.8462 | 0.4904 |
| `code_runs` | not measured | not measured |
| `rouge_l` | 0.3866 | 0.2891 |
| `latency_p50_s` | 4.945 | 6.722 |
| `latency_p95_s` | 7.621 | 7.846 |

The composite rises by 0.144. Code validity and ROUGE-L account for it; keyword
recall is unchanged (-0.006). The fine-tune puts a code block in 105 of 108 answers
(the base model, in its laptop run: 90), and its answers are about a third shorter
(571 characters on average against 831), which also lowers latency.

Composite score by category, for categories with at least 5 items. The base column is
the base model's previous `eval/baseline.json` (laptop CPU, composite 0.4101), the
only base run with per-item results in the repository; its overall score is within
0.008 of the CI run above.

| Category | Items | Fine-tuned | Base |
| --- | --- | --- | --- |
| security | 8 | 0.5169 | 0.2045 |
| dependencies | 6 | 0.6588 | 0.5916 |
| database | 5 | 0.4272 | 0.3079 |
| debugging | 5 | 0.6948 | 0.5665 |
| openapi | 5 | 0.4545 | 0.2694 |
| streaming | 5 | 0.4318 | 0.3184 |
| testing | 5 | 0.5704 | 0.3684 |
| websockets | 5 | 0.4728 | 0.3976 |

The lowest-scoring categories for the fine-tune were shutdown (0.2803, 1 item),
templates (0.2959, 1 item), rate limiting (0.3353, 2 items) and deployment (0.3481,
3 items). Categories with one to three items give noisy estimates.

Gate 1 on the Kaggle GPU scores the unquantized models with `transformers` instead:
0.538 for the fine-tune against 0.383 for the base model.

### Results (base model, previous release)

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

Retrieval raises keyword recall by 0.0803 and ROUGE-L by 0.0199 but lowers code
validity by 0.1379, so the composite falls by 0.0100, and median latency rises by
about 58%. Retrieval is therefore opt-in per request. Retrieval has not been
measured with the fine-tuned model yet.

The base model's lowest-scoring categories were shutdown (0.0558, 1 item), lifespan
(0.1358, 2 items) and deployment (0.1584, 3 items).

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

- **Incorrect answers stated confidently.** Both the base model and the fine-tune
  invent parameters and APIs, and the fine-tune does it in well-formatted, parseable
  code, which makes the errors easier to miss. In `eval/baseline.json` (the
  fine-tune), item `core_http-003` passes a nonexistent `item_field` argument to
  `Body`, item `core_http-001` annotates a path parameter as `str | r'/path'`, and
  item `async_io-014` calls `JSONResponse` with a `text` argument it does not take.
  The base model made similar errors: during the initial audit it claimed there is a
  "`path` argument in the `Path` class".
- **Small, quantized model.** At 1.1B parameters and 4-bit quantization, quality is
  limited: the composite score is 0.5463, and keyword recall, the share of the
  required API names an answer mentions, is 0.4259.
- **Outdated idioms.** The base model was not trained for current FastAPI and Pydantic
  versions and produced older patterns where the dataset teaches newer ones: in its
  eval answers, `@app.on_event("startup")` instead of a lifespan handler and the
  Pydantic v1 method `.dict()` instead of `model_dump()`. Neither appears in the
  fine-tune's 108 eval answers, but the set is small, so older idioms can still
  appear.
- **Retrieval trade-off.** With retrieved documentation in the prompt, the model tends
  to copy code fragments it cannot adapt, which lowers code validity.
- **Short context.** The default context window is 2,048 tokens; longer prompts are
  rejected with HTTP 400 `context_length_exceeded`.
- **Generated code is untrusted.** It can be insecure or wrong. Review it before use
  and run it only in a sandbox.
- **Prompt injection.** User input can steer the model's output. The model has no
  tools and no data access, so the effect is limited to the text it returns.
- **Bias.** The base model was pre-trained on large web and code corpora and can
  reflect their biases. The narrow FastAPI domain reduces, but does not remove, this
  risk.
- **Coverage.** The training set has 351 examples in English, so many FastAPI topics
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

Not measured in energy or emissions. The fine-tune trained for 17 minutes on one
free Kaggle GPU, and the whole kernel run (training, both evaluations, the merge and
the GGUF conversion) took about an hour on it. The deployed model runs on CPU.

## License

- **Weights:** the base model and the GGUF quantization are published under
  Apache-2.0, as declared on both Hugging Face model cards (checked through the
  Hugging Face API, the GGUF card at the pinned revision). The fine-tuned GGUF and
  adapter are published under Apache-2.0, with a NOTICE that attributes TinyLlama and
  describes the modifications; redistributions must keep both.
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
[TinyLlama model card](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0).
