# Contributing

Thank you for helping improve the FastAPI Niche Assistant. This guide explains how to
set up a development environment, which checks a change must pass, the rules that
protect the evaluation, and how to propose a new model release.

By taking part you agree to follow the [Code of Conduct](CODE_OF_CONDUCT.md). Report
security vulnerabilities privately as described in [SECURITY.md](SECURITY.md), never in
a public issue.

## Before you start

- For anything larger than a small fix, open an issue first so the approach can be
  agreed before you write code.
- Read [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the design and the
  [README](README.md) for the project layout.
- Keep each pull request to one logical change.

## Development setup

You need:

- Python 3.10, 3.11 or 3.12 (CI tests all three; the production image uses 3.11);
- Node.js 22 or later, for the edge Worker and the web UI;
- Git; Docker is optional (`docker compose up --build` runs the service with Postgres
  and pgvector).

```bash
git clone https://github.com/syed-waleed-ahmed/Niche-Finetuned-Model.git
cd Niche-Finetuned-Model
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip

# Optional but recommended without a GPU: the CPU-only torch build is much smaller
# than the default CUDA build. CI installs it this way.
pip install torch --index-url https://download.pytorch.org/whl/cpu

pip install -e ".[dev,train,postgres]"
```

The extras in `pyproject.toml`:

| Extra | Installs | Needed for |
| --- | --- | --- |
| `dev` | pytest, pytest-cov, ruff, mypy, openai | tests and checks (the OpenAI SDK runs `examples/openai_sdk.py` in the tests) |
| `train` | `torch` extra plus `datasets` | fine-tuning |
| `torch` | torch, transformers, peft, accelerate, sentencepiece, protobuf | the transformers backend, merging, export |
| `postgres` | psycopg and its pool | per-user keys, usage, pgvector |
| `llamacpp` | llama-cpp-python | the production GGUF backend (needs a C/C++ toolchain if no wheel is available) |
| `rag` | fastembed | retrieval over the FastAPI docs |
| `sentry` | sentry-sdk | error reporting |
| `datagen` | anthropic | drafting training data with Claude |
| `serve` | `llamacpp`, `postgres`, `rag`, `sentry` | everything the production image runs |

The test suite needs no model, network access, or database. The Makefile wraps the
common commands (`make test` runs pytest; `make lint` runs ruff and mypy); on Windows
without `make`, run the commands below directly.

## Checks

Every pull request must pass the same checks CI runs. Run them locally first:

```bash
# Python
pytest                                   # unit, API, and examples tests
ruff check src tests examples            # lint
ruff format --check src tests examples   # formatting (fix with: ruff format src tests examples)
mypy                                     # type check; configured in pyproject.toml [tool.mypy]
fastapi-assistant data validate          # dataset schema, code, duplicates, leakage

# Edge Worker
cd deploy/cloudflare-worker && npm ci && npm run typecheck && npm test && cd ../..

# Web UI
cd web && npm ci && npm run typecheck && npm test && npm run build && cd ..
```

Optional suites:

- `TEST_DATABASE_URL=postgresql://... pytest -m postgres` runs the Postgres and
  pgvector integration tests (CI runs them against `pgvector/pgvector:pg16`).
- `pytest -m slow` downloads a real model.

CI also builds the container image, boots it, answers a real question, scans the
image with Trivy, scans the full git history with gitleaks, and runs CodeQL. The
Security workflow runs `pip-audit` on `requirements/serve.txt`, `npm audit` for the
Worker and the web UI, and GitHub dependency review.

## Pre-commit hooks

The repository ships a [pre-commit](https://pre-commit.com) configuration with ruff
(lint and format), basic file hygiene checks, and gitleaks:

```bash
pip install pre-commit
pre-commit install            # run the hooks on every commit
pre-commit run --all-files    # run them once over the whole tree
```

The first run takes longer: pre-commit builds the hook environments, including
gitleaks. The whitespace fixers deliberately skip `data/*.jsonl`, `eval/` and
`graphify-out/`.

## Dependencies

- Python dependency ranges live in `pyproject.toml`. The production image installs
  from `requirements/serve.txt`, a hash-pinned lock. If you change the `serve` extras
  (or anything they pull in), regenerate the lock with `make lock` (needs
  [uv](https://docs.astral.sh/uv/)) and commit it in the same pull request.
- npm projects (`web/`, `deploy/cloudflare-worker/`) commit their `package-lock.json`;
  install with `npm ci`.
- Dependabot opens update pull requests weekly; prefer those to manual bumps.
- New dependencies need a reason in the pull request and a license compatible with
  MIT distribution. Record new third-party components in
  [docs/legal/LICENSE_AUDIT.md](docs/legal/LICENSE_AUDIT.md).

## Changing the dataset

Training rows (`data/fastapi_qa_train.jsonl`) have `instruction`, `input` and
`output`. Eval rows (`data/fastapi_qa_eval.jsonl`) also have `id`, `category` and
`must_include`. Answers follow one style: a one or two sentence lead, one minimal
runnable example in a `python` fence, then at most two short sentences; at most 1,400
characters; current idioms (Pydantic v2, `Annotated[...]` parameters, `lifespan`
instead of `on_event`).

To add training data:

1. Write rows by hand, or draft them with Claude from the official docs:
   `pip install -e ".[datagen]"`, then
   `fastapi-assistant data generate --out drafts.jsonl --pages 20`. This needs your
   own Anthropic credentials and makes one API request per documentation section.
2. Review every drafted row yourself. Fix or delete anything that is wrong, unclear,
   or not supported by the documentation.
3. Merge reviewed rows with `fastapi-assistant data merge drafts.jsonl --split train`.
4. Run `fastapi-assistant data validate` and `fastapi-assistant data stats`.
5. In the pull request, say how the rows were produced (hand-written, or drafted and
   reviewed) and what you checked.

Only add content you have the right to contribute. The FastAPI documentation is
MIT-licensed; do not paste text from sources with other terms.

## Evaluation integrity

The held-out eval set is the release gate: a model ships only if it scores well on
it. A gate is only meaningful if nobody optimises against it. These rules are not
negotiable:

1. **Never edit `data/fastapi_qa_eval.jsonl` to improve a score.** Do not rewrite,
   remove, or relabel items, references, or `must_include` identifiers because a model
   gets them wrong. A genuine error in an eval item (for example a wrong reference
   answer) is fixed in a separate pull request that changes nothing else, explains the
   error with evidence, and records the new baseline (see rule 2).
2. **Never edit the eval baselines by hand.** `eval/baseline.json`,
   `eval/baseline.md`, `eval/baseline-rag.json` and `eval/baseline-rag.md` are written
   only by `fastapi-assistant eval`. The training workflow's release pull request
   replaces `eval/baseline.json` and `eval/baseline.md` with the report produced by
   the eval workflow. When the eval set changes, the eval workflow fails unless
   `eval/baseline.json` changes in the same pull request; regenerate it with
   `fastapi-assistant eval` on the production backend (load the settings with
   `fastapi-assistant export env model-release.json`), never by hand.
3. **Never train or tune on the eval set.** Do not add eval items, or paraphrases of
   them, to the training data, and do not choose prompts, hyperparameters, or data by
   repeatedly scoring against the eval set. The leakage check in
   `fastapi-assistant data validate` catches near-duplicates, not paraphrases, so this
   depends on you. Training itself never reads the eval file: checkpoint selection
   uses a seeded validation slice of the *training* file (`VALIDATION_FRACTION`,
   default 0.1; see `src/fastapi_assistant/data.py`).
4. **Report results honestly.** Paste the eval report produced by the tool. Do not
   round in your favour, drop categories, or quote a subset of items.

Changes to `src/fastapi_assistant/prompts.py`, `src/fastapi_assistant/engines/` or
`src/fastapi_assistant/evaluation.py` trigger the eval workflow, which scores the
model against `eval/baseline.json` and fails on a regression of more than 0.02 in
the composite score.

## Proposing a model release

The deployed model is whatever `model-release.json` pins. A release changes that file
and nothing about the eval set.

**Maintainers** run the "Train on Kaggle" workflow (`.github/workflows/train.yml`).
It needs the `KAGGLE_API_TOKEN` secret (or the legacy `KAGGLE_KEY`), the Kaggle
username as the `KAGGLE_USERNAME` variable (a secret is also accepted), the
`HF_NAMESPACE` variable, and an `HF_TOKEN` secret attached to the Kaggle kernel. On
first setup, run the workflow once with `push_only: true` to create the kernel, attach
`HF_TOKEN` to it on Kaggle, then run the workflow normally. It trains on a free
Kaggle GPU, requires the fine-tune to beat the base model's composite score by at
least 0.005 (`min_improvement`), merges and quantizes it, publishes the adapter and the GGUF to the
Hugging Face Hub under a new tag, scores the GGUF with the production backend, and
opens a pull request that updates `model-release.json` and `eval/baseline.json`.
Review the eval table in that pull request; merging it deploys the model, and
reverting it rolls back.

**Contributors** without access to the repository secrets:

1. Open an issue describing the proposed change (data, hyperparameters, or base
   model) and the expected effect.
2. Run `kaggle/pipeline.py` in your own Kaggle account, with `repo` and `ref` in its
   `CONFIG` block pointing at your fork and branch, `hf_namespace` set to your Hugging
   Face namespace, and your own `HF_TOKEN` attached as a Kaggle secret. It writes
   `model-release.json`, `eval-base.json` and `eval-finetuned.json`.
3. Make sure the published weights are public, carry the Apache-2.0 license of the base
   model with attribution, and are pinned to the release tag with the GGUF's SHA-256
   in `model-release.json`.
4. Open a pull request that changes only `model-release.json`, and paste both eval
   reports into the description. The eval workflow scores your model and the current
   one on the same eval set with the production backend and uploads the report as a
   workflow artifact.
5. Do not edit `eval/baseline.json`: if the release is accepted, the maintainer
   replaces it with the report produced by the eval workflow.

A release must pass both gates, use the shared prompt format in `prompts.py`, and
come with an updated [MODEL_CARD.md](MODEL_CARD.md).

## Commits and pull requests

- Branch from `main` with a short descriptive name, for example `feat/...`, `fix/...`
  or `docs/...`, and target `main`.
- Write commit subjects in the imperative mood and sentence case, about 72 characters
  at most, with no trailing period (for example "Add a non-interactive mode to the
  secrets script"). Use the body to explain what changed and why.
- Fill in the pull request template. Link the issue (`Fixes #123`).
- Update the documentation the change affects (README, `docs/`, `.env.example`, the
  model card) and add a line to the `Unreleased` section of
  [CHANGELOG.md](CHANGELOG.md).
- Preview the docs site: `pip install -r requirements/docs.txt && mkdocs serve`.
- Never commit secrets, API keys, `.env` files, model weights, or generated
  `outputs/`. CI scans the full history with gitleaks; a leaked secret must be
  rotated, not only deleted.
- All CI checks must pass before review.

## License

By contributing, you agree that your contributions are licensed under the
[MIT License](LICENSE) that covers this project.
