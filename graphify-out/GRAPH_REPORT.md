# Graph Report

Generated on 2026-07-05 for the `niche_finetuned_model` repository.

## Scope

- 8 source and documentation files analyzed
- 33 nodes and 49 edges identified
- 9 communities detected
- 84% extracted relationships and 16% inferred relationships

## High-Level Structure

The repository separates into a small number of clear functional areas:

1. Model serving and request handling in `src/api.py` and `src/inference.py`.
2. Training and dataset preparation in `src/train_lora.py` and `src/dataset.py`.
3. CLI and application entrypoint logic in `main.py` and `src/cli.py`.
4. Supporting configuration in `src/config.py`.

## Primary Entry Points

- `generate_answer()`
- `load_finetuned_model()`
- `build_prompt()`
- `interactive_chat()`
- `main()`

## Architectural Observations

- The model-serving path is tightly centered on a reusable inference loader and a minimal HTTP API surface.
- Training and inference share the same prompt construction logic, which keeps the dataset format aligned with runtime behavior.
- The project has a small, well-defined dependency graph and is suitable for a standalone portfolio or reference implementation.

## Notes

- This report is generated output from graphify and should be treated as a derived artifact.
- The generated cache for graphify output is ignored by `.gitignore`; the HTML and markdown summaries remain available for review.