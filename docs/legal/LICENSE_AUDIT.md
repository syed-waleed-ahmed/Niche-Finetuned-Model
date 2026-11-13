# Third-party license audit

Date: 2026-11-13. Branch: `feat/production-hardening` (commit `caad7bb`).

This audit lists every third-party component that the project distributes or uses,
the license each one declares, and how that was checked. It is an engineering
inventory, not legal advice. Anything that could not be checked from a primary source
is marked **UNVERIFIED** rather than guessed.

## 1. What the project distributes

| Artifact | Contents relevant to licensing |
| --- | --- |
| This git repository (public on GitHub) | Source code, the dataset (`data/*.jsonl`), eval baselines (model-generated text), docs. |
| Container image (`ghcr.io`, built by `Dockerfile`) | Python 3.11 on Debian (`python:3.11-slim`), the serving dependencies from `requirements/serve.txt` (including a compiled llama.cpp), the project wheel, and by default (`BAKE_RAG_INDEX=true`) the FastAPI documentation index and the embedding model under `/models`. Model weights are **not** included unless the image is built with `BAKE_MODEL=true` (default `false`). |
| Web UI bundle (`web/dist`, served by the Cloudflare Worker) | The React application and its production npm dependencies, bundled by Vite. |
| Cloudflare Worker | Only this project's TypeScript source; it has no runtime npm dependencies. |
| Model artifacts on the Hugging Face Hub (future) | A LoRA adapter and a merged Q4_K_M GGUF, labelled Apache-2.0 by `src/fastapi_assistant/export.py`. None has been published yet. |

## 2. Verification methods

| Method | Used for |
| --- | --- |
| Hugging Face API, unauthenticated: `HfApi().model_info(repo, revision=...).card_data` and `dataset_info(...)` | Model and dataset licenses (card metadata `license` and `license:*` tags). The GGUF repository was checked at the pinned revision. |
| `fastembed.TextEmbedding.list_supported_models()` (fastembed 0.9.1) | Which Hub repository the embedding model is actually downloaded from. |
| Installed package metadata, `importlib.metadata` in the project's `.venv` (Python 3.11, Windows) | Python packages: `License-Expression`, `License`, license classifiers, and license files in `*.dist-info`. |
| `web/package-lock.json` and `deploy/cloudflare-worker/package-lock.json` (`license` field of every locked package), cross-checked against `node_modules/<pkg>/package.json` | npm packages. |
| GitHub REST API `GET /repos/{owner}/{repo}/license`, and the `LICENSE` file at a specific tag | Upstream repositories (llama.cpp, FastAPI, pre-commit hooks). |

## 3. Models

| Component | Role in this project | Distributed by this project? | License | How verified |
| --- | --- | --- | --- | --- |
| [TinyLlama/TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) | Base model for fine-tuning; base of the deployed GGUF | No. Downloaded from the Hub at training time. A fine-tuned derivative would be. | Apache-2.0 | HF API: `card_data.license = apache-2.0`, tag `license:apache-2.0`, not gated (commit `fe8a4ea1`). The repository has no `LICENSE` file; the license is declared in the card metadata only. |
| [TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF](https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF), file `tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf` | Deployed weights (`model-release.json`) | No by default: downloaded at runtime at revision `52e7645b`. Yes if an image is built with `BAKE_MODEL=true`. | Apache-2.0 | HF API at revision `52e7645ba7c309695bec7ac98f4f005b139cf465`: `card_data.license = apache-2.0`, `base_model = TinyLlama/TinyLlama-1.1B-Chat-v1.0`, not gated. The Hub's LFS SHA-256 for the file equals the `sha256` in `model-release.json`. No `LICENSE` file in the repository. |
| [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) | Embedding model for retrieval (`EMBEDDING_MODEL` in `config.py`) | Yes, inside the image when the RAG index is baked (default). | MIT | HF API: `card_data.license = mit`. |
| [Qdrant/bge-small-en-v1.5-onnx-Q](https://huggingface.co/Qdrant/bge-small-en-v1.5-onnx-Q) | The quantized ONNX export of the model above that fastembed actually downloads | Yes, as above. | MIT | fastembed 0.9.1 registry maps `BAAI/bge-small-en-v1.5` to this repository with `license: mit`; HF API: `card_data.license = mit`. fastembed downloads only the model and tokenizer files, and the repository has no `LICENSE` file, so no license text reaches the image. |

Notes on the base model:

- The TinyLlama card states that it "adopted exactly the same architecture and
  tokenizer as Llama 2". This audit relies on the TinyLlama project's Apache-2.0
  declaration and did not assess the tokenizer's provenance separately.
- Upstream training data listed on the TinyLlama card, checked for information only:

  | Dataset | License (HF API) |
  | --- | --- |
  | `cerebras/SlimPajama-627B` | **UNVERIFIED**: the anonymous API request returned 401. |
  | `bigcode/starcoderdata` | `other`; gated (`auto`), with terms of use. |
  | `HuggingFaceH4/ultrachat_200k` | MIT |
  | `HuggingFaceH4/ultrafeedback_binarized` | MIT |

- The GGUF repository's card lists `OpenAssistant/oasst_top1_2023-08-25` (Apache-2.0)
  instead of the two HuggingFaceH4 datasets. This is a metadata difference between the
  two cards; it does not change the license of either model.

## 4. Documentation used for retrieval and data drafting

| Component | Role | Distributed? | License | How verified |
| --- | --- | --- | --- | --- |
| FastAPI documentation (`docs/en/docs` and `docs_src`) at tag `0.162.5`, downloaded from `codeload.github.com/fastapi/fastapi` (`src/fastapi_assistant/docs_corpus.py`) | Retrieval corpus; grounding for `fastapi-assistant data generate` | Yes. With `BAKE_RAG_INDEX=true` (default) the image contains the chunked documentation text (`/models/rag_index/chunks.jsonl`), and retrieved excerpts are placed in prompts. | MIT, "Copyright (c) 2018 Sebastián Ramírez" | `LICENSE` file at tag `0.162.5`; GitHub API `fastapi/fastapi`: MIT; installed `fastapi` 0.162.5 metadata: `License-Expression: MIT`. The repository license covers the docs directory; no separate docs license was found. |

## 5. Inference runtime

| Component | Version | License | How verified |
| --- | --- | --- | --- |
| llama-cpp-python | 0.3.36 | MIT | Installed metadata `License: MIT`; `LICENSE.md` ("Copyright (c) 2023 Andrei Betlen"); GitHub API `abetlen/llama-cpp-python`: MIT. |
| llama.cpp and ggml (compiled into the llama-cpp-python wheel) | as vendored by 0.3.36 | MIT | GitHub API `ggml-org/llama.cpp`: MIT. The installed package (checked on the local installation) ships only llama-cpp-python's own `LICENSE.md`; llama.cpp's license text is **not** included, so it is not expected in the image either. |
| onnxruntime | 1.31.1 | MIT | Installed metadata; ships `LICENSE` and `ThirdPartyNotices.txt`. |
| fastembed | 0.9.1 | Apache-2.0 | Installed metadata; ships `LICENSE` and `NOTICE` ("This product includes software developed by Qdrant"). |

## 6. Python dependencies in the container image (`requirements/serve.txt`)

Verified from installed metadata in the project's `.venv`. "Files" means a license
file is present in the installed `dist-info`, so it is carried into the image.

| Package | Pinned | License | Notes |
| --- | --- | --- | --- |
| annotated-doc | 0.0.6 | MIT | files |
| annotated-types | 0.8.0 | MIT | files |
| anyio | 4.16.0 | MIT | files |
| certifi | 2026.10.23 | MPL-2.0 | files; file-level copyleft (see section 10) |
| charset-normalizer | 3.5.2 | MIT | files |
| click | 8.5.1 | BSD-3-Clause | files |
| colorama | 0.4.6 | BSD (classifier) | files; Windows only, not in the Linux image |
| diskcache | 5.6.3 | Apache-2.0 | files; see SECURITY.md for advisory PYSEC-2026-2447 |
| fastapi | 0.162.5 | MIT | files |
| fastembed | 0.9.1 | Apache-2.0 | files, including `NOTICE` |
| filelock | 4.2.10 | MIT | Checked on the installed 4.2.4; the pinned 4.2.10 was not installed locally. |
| flatbuffers | 25.12.19 | Apache-2.0 | metadata only; no license file in `dist-info` |
| fsspec | 2026.9.0 | BSD-3-Clause | files |
| h11 | 0.16.0 | MIT | files |
| hf-xet | 1.8.0 | Apache-2.0 | files |
| httpcore | 1.0.9 | BSD-3-Clause | files |
| httptools | 0.9.0 | MIT | files, including vendored http-parser (MIT) and llhttp licenses |
| httpx | 0.28.1 | BSD-3-Clause | files |
| huggingface-hub | 1.33.0 | Apache-2.0 | files |
| idna | 3.20 | BSD-3-Clause | files |
| jinja2 | 3.1.6 | BSD (classifier) | files |
| llama-cpp-python | 0.3.36 | MIT | files; see section 5 |
| loguru | 0.7.3 | MIT (classifier) | no license file in `dist-info` |
| markupsafe | 3.0.4 | BSD-3-Clause | Checked on the installed 3.0.3; the pinned 3.0.4 was not installed locally. |
| mmh3 | 5.3.1 | MIT | files |
| numpy | 2.4.6 / 2.5.4 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 | files; checked on 2.4.6 (the Python 3.11 pin used by the image) |
| onnxruntime | 1.31.1 | MIT | files and third-party notices |
| opentelemetry-api | 1.45.2 | Apache-2.0 | files |
| packaging | 26.3 | Apache-2.0 OR BSD-2-Clause | files |
| pillow | 13.0.0 | MIT-CMU | files |
| prometheus-client | 0.26.0 | Apache-2.0 AND BSD-2-Clause | files, including `NOTICE` |
| protobuf | 7.36.2 | BSD-3-Clause | files |
| psycopg | 3.3.9 | LGPL-3.0-only | files; see section 10 |
| psycopg-binary | 3.3.9 | LGPL-3.0-only | files; bundles native libraries (see section 10) |
| psycopg-pool | 3.3.3 | LGPL-3.0-only | files |
| py-rust-stemmers | 0.1.8 | MIT | no license field in metadata; `LICENSE` file reads "MIT License, Copyright (c) 2024 qdrant" |
| pydantic | 2.14.3 | MIT | files |
| pydantic-core | 2.50.4 | MIT | files |
| pydantic-settings | 2.16.2 | MIT | files |
| python-dotenv | 1.2.4 | BSD-3-Clause | files |
| pyyaml | 6.0.3 | MIT | files |
| requests | 2.34.2 | Apache-2.0 | files, including `NOTICE` |
| sentry-sdk | 2.75.0 | MIT | files |
| starlette | 1.10.0 | BSD-3-Clause | files |
| tokenizers | 0.23.3 | Apache-2.0 (classifier) | no license file in `dist-info` |
| tqdm | 4.70.1 | MPL-2.0 AND MIT | files |
| typing-extensions | 4.16.0 | PSF-2.0 | files |
| typing-inspection | 0.4.5 | MIT | files |
| tzdata | 2026.6 | Apache-2.0 | files; Windows only, not in the Linux image |
| urllib3 | 2.8.0 | MIT | files |
| uvicorn | 0.54.0 | BSD-3-Clause | files |
| uvloop | 0.23.0 | **UNVERIFIED** | Linux and macOS only; not installable on the Windows machine used for this audit. |
| watchfiles | 1.3.0 | MIT | files |
| websockets | 17.4 | BSD-3-Clause | files |
| win32-setctime | 1.2.1 | MIT | files; Windows only, not in the Linux image |

## 7. Python dependencies used but not distributed

These are installed for training, development, or data drafting, not in the
production image.

| Package | Version checked | License | Extra |
| --- | --- | --- | --- |
| torch | 2.15.0+cpu | Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND ISC AND MIT AND Zlib | `torch`, `train` |
| transformers | 5.22.2 | Apache-2.0 | `torch`, `train` |
| peft | 0.21.2 | Apache-2.0 | `torch`, `train` |
| accelerate | 1.15.0 | Apache-2.0 | `torch`, `train` |
| sentencepiece | 0.2.3 | Apache-2.0 | `torch`, `train` |
| datasets | 5.2.0 | Apache-2.0 | `train` |
| anthropic | 2.8.0 | Apache-2.0 | `datagen` |
| openai | 3.50.0 | Apache-2.0 | `dev` (runs `examples/openai_sdk.py` in the tests) |
| ruff | 0.17.5 | MIT | `dev` |

Pre-commit hooks (`.pre-commit-config.yaml`), checked with the GitHub API:
`astral-sh/ruff-pre-commit` Apache-2.0, `pre-commit/pre-commit-hooks` MIT,
`gitleaks/gitleaks` MIT, and `pre-commit/pre-commit` MIT. Other development tools
(pytest, pytest-cov, mypy) were not audited; they are not distributed.

## 8. npm dependencies

### Web UI (`web/`): bundled into `web/dist` and served to browsers

Direct production dependencies:

| Package | Locked version | License |
| --- | --- | --- |
| react | 19.3.1 | MIT |
| react-dom | 19.3.1 | MIT |
| react-markdown | 10.1.0 | MIT |
| rehype-highlight | 7.0.2 | MIT |
| remark-gfm | 4.0.1 | MIT |
| highlight.js | 11.12.0 | BSD-3-Clause |

The full production tree in `web/package-lock.json` has 110 packages: 107 MIT, 2
BSD-3-Clause (`highlight.js` 11.12.0, and 11.11.2 nested under `lowlight`), and 1 ISC
(`@ungap/structured-clone` 1.4.0). No package lacks a license field. The installed
`node_modules` gave the same distribution (109 packages); it is slightly behind the
lockfile (`highlight.js` 11.11.2 at the top level), which does not change any license.

Development dependencies (Vite, Vitest, TypeScript, jsdom, Testing Library) are not
bundled into `web/dist`. Direct ones checked: vite MIT, vitest MIT, typescript
Apache-2.0, jsdom MIT, `@vitejs/plugin-react` MIT, `@testing-library/react` MIT.

### Edge Worker (`deploy/cloudflare-worker/`)

No production dependencies: every import in `src/` is a relative import of the
project's own modules, and the lockfile has no non-development packages. Development
tooling from the lockfile: wrangler 4.159.0 (MIT OR Apache-2.0),
`@cloudflare/workers-types` 5.20261110.1 (MIT OR Apache-2.0), vitest 5.1.0 (MIT),
typescript 7.0.2 (Apache-2.0).

## 9. Container base image and infrastructure images

| Component | Distributed? | License | Status |
| --- | --- | --- | --- |
| `python:3.11-slim` base image (CPython and Debian packages, plus `libgomp1` installed by the `Dockerfile`) | Yes, as the base layers of the published image | CPython: PSF-2.0. Debian packages: various. | **UNVERIFIED**: the Debian package licenses were not inventoried. |
| `pgvector/pgvector:pg16` (`docker-compose.yml`), `cloudflare/cloudflared:2026.10.7` and `grafana/alloy:v1.21.3` (`deploy/vm/docker-compose.yml`) | No. Users pull them from their registries. | Not audited | Not redistributed by this project. |

## 10. Obligations and open items

### Apache-2.0 (TinyLlama weights, GGUF quantization, fastembed, requests, and others)

- Redistribution must include a copy of the Apache-2.0 license, keep copyright,
  patent, trademark and attribution notices, carry any `NOTICE` file, and mark
  modified files as changed.
- **Deployed weights:** not redistributed in the default image, so nothing is owed for
  them today. An image built with `BAKE_MODEL=true` must add the Apache-2.0 text and
  attribution to TinyLlama and TheBloke, because neither Hub repository has a
  `LICENSE` file to copy.
- **Fine-tuned weights (future): resolved.** The merged GGUF and the adapter are
  derivatives of TinyLlama. `export.write_model_card` now adds a license and
  attribution section (Apache-2.0, the base model, and a statement of the
  modification) and writes a `NOTICE` file into the published folder; covered by
  `tests/test_hardening.py`.
- **Python packages:** their `LICENSE` and `NOTICE` files are carried in `dist-info`
  inside the image, except `flatbuffers` and `tokenizers` (metadata only).

### MIT, BSD and ISC (notice-preserving licenses)

- **Web UI bundle: resolved.** `web/scripts/notices.mjs` runs before every
  `npm run build` and writes `THIRD_PARTY_NOTICES.txt` (the license file of each of the
  110 production packages) into `web/dist`, served next to the app.
- **FastAPI documentation, embedding model and llama.cpp in the image: resolved.**
  `THIRD_PARTY_NOTICES.md` at the repository root carries the upstream license texts
  (llama.cpp and FastAPI fetched verbatim from their repositories; the standard MIT
  text for `bge-small-en-v1.5`, whose repositories have no license file), and the
  Dockerfile copies it with `LICENSE` to `/usr/share/doc/fastapi-assistant/`.
- **Project code:** MIT, "Copyright (c) 2026 Syed Waleed Ahmed". The `Dockerfile`
  copies `LICENSE` into the build stage, where setuptools includes it in the project
  wheel by default.

### LGPL-3.0 (psycopg, psycopg-binary, psycopg-pool)

The packages are distributed unmodified, as separately installed Python packages that
a user can replace, and their license texts are in `dist-info`. That is consistent
with the LGPL's conditions for unmodified use. `psycopg-binary` also bundles native
libraries (`libpq` and its dependencies); their individual licenses were **not**
verified.

### MPL-2.0 (certifi, tqdm)

File-level copyleft. Distributed unmodified with their license files; the source is
available from PyPI. No action is needed unless these files are modified.

### Dataset (`data/*.jsonl`)

- The dataset is already public in this repository under the repository's MIT
  license.
- **Provenance:** the rows were written for this project with AI assistance. The
  commit that added most of them (`3092ad9`) is co-authored with Claude, and
  `docs/ARCHITECTURE.md` describes the data as "hand-written and Claude-drafted pairs
  from the pinned docs". The data files do not record which rows were drafted.
- **UNVERIFIED: model-provider terms.** The terms of service that governed the use of
  Claude to draft rows were not reviewed for conditions on using generated output as
  training data for another model. The owner should confirm this before the dataset or
  fine-tuned weights are published as standalone artifacts.
- **UNVERIFIED: verbatim documentation content.** Drafting is grounded in the
  MIT-licensed FastAPI documentation, and rows were not checked for verbatim excerpts
  (for example code examples from `docs_src`). MIT permits this use with attribution,
  so the simplest remedy is to credit the FastAPI documentation and include its MIT
  notice wherever the dataset is published separately.
- The eval baselines (`eval/baseline*.json`) contain text generated by the
  Apache-2.0 TinyLlama model. No restriction on outputs was found in the Apache-2.0
  license.

## 11. Publication clearance

| Artifact | Status |
| --- | --- |
| Source code (MIT) | Cleared. |
| Container image | Cleared for the notices in section 10 (now shipped in `/usr/share/doc/fastapi-assistant/`). The Debian layer inventory and `psycopg-binary` native libraries remain UNVERIFIED. |
| Web UI bundle | Cleared: `THIRD_PARTY_NOTICES.txt` ships with `web/dist`. |
| Deployed GGUF | Not redistributed by default. Building with `BAKE_MODEL=true` requires the Apache-2.0 text and attribution. |
| Fine-tuned adapter and GGUF (not yet produced) | Permitted under Apache-2.0 with the obligations above (license text, attribution, statement of changes). Also depends on the dataset items below. |
| Dataset as a standalone release (for example on the Hugging Face Hub) | **Not cleared.** Open: model-provider terms for drafted rows (UNVERIFIED), and FastAPI documentation attribution. |
