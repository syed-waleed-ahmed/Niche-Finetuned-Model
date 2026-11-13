## Summary

<!-- What does this change do, and why? Link the issue, for example "Fixes #123". -->

## Type of change

- [ ] Bug fix
- [ ] New feature
- [ ] Model release (`model-release.json`)
- [ ] Dataset change (`data/fastapi_qa_train.jsonl`)
- [ ] Documentation
- [ ] Refactoring, CI, or dependencies

## How it was tested

<!-- Commands you ran and what you checked by hand. For model or prompt changes,
paste the eval report produced by `fastapi-assistant eval`, unedited. -->

## Checklist

- [ ] `pytest` passes.
- [ ] `ruff check src tests examples` and `ruff format --check src tests examples` pass.
- [ ] `mypy` passes.
- [ ] `fastapi-assistant data validate` passes (required for dataset changes).
- [ ] Worker checks pass if `deploy/cloudflare-worker/` changed: `npm run typecheck && npm test`.
- [ ] Web UI checks pass if `web/` changed: `npm run typecheck && npm test && npm run build`.
- [ ] No secrets, API keys, `.env` files, model weights, or generated `outputs/` are committed.
- [ ] `data/fastapi_qa_eval.jsonl` and the `eval/baseline*` files are not edited by hand or to improve a score, and no eval data is used for training or tuning (see "Evaluation integrity" in CONTRIBUTING.md).
- [ ] Documentation is updated where behaviour changed (README, `docs/`, `.env.example`, MODEL_CARD.md).
- [ ] A line is added under `Unreleased` in CHANGELOG.md.
- [ ] New third-party dependencies, if any, are recorded in `docs/legal/LICENSE_AUDIT.md`.
