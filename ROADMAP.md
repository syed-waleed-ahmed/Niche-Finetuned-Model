# Roadmap

This is where the project is heading. It is a plan, not a promise: priorities change
with what contributors pick up and what the evaluation shows. Items marked
**good first issue** have a matching issue with that label and are small enough for a
first contribution.

To propose something new, open a [feature request](https://github.com/syed-waleed-ahmed/Niche-Finetuned-Model/issues/new/choose)
or start a thread in [Discussions](https://github.com/syed-waleed-ahmed/Niche-Finetuned-Model/discussions).

## Now

| Item | Why |
| --- | --- |
| Larger, more varied training set | Keyword recall did not move with the first fine-tune (0.426 against the base model's 0.432): it still names the wrong API about as often. More questions per topic, especially on dependencies, security and testing, should help. |
| Measure retrieval with the fine-tuned model | Retrieval was measured on the base model only, where it was a trade-off. The fine-tune trained on context examples, so `fastapi-assistant eval --rag` on it decides whether retrieval should become the default. |

## Next

| Item | Why |
| --- | --- |
| Cancel non-streaming generations when the caller disconnects | Today such a generation keeps the only slot busy for up to 60 seconds (audit item O-2). **Good first issue.** |
| Attributable edge logs | Log the truncated key identity the Worker already computes, so abuse can be traced to a client (audit item O-5). **Good first issue.** |
| Dashboards for the new metrics | Grafana panels for queue wait, time to first token and rejected requests. **Good first issue.** |
| Narrower deployer permissions | Grant the CI deployer access to individual secrets instead of the whole project (audit item O-4). |
| A generation check in the uptime probe | The probe covers `/health` and `/v1/models`; a tiny generation would also catch a broken model. |

## Later

| Item | Why |
| --- | --- |
| A stronger base model | A 1.1B model is fast on a free CPU but often wrong. A 1.5B to 3B model, quantized, may still fit the free tier. |
| Retrieval on by default | Answers grounded in the FastAPI documentation are more accurate; the index needs to fit the image size and cold-start budget. |
| Backup origin | A Hugging Face Space or a small VM behind the edge circuit breaker (configurations exist in `deploy/`). |
| Evaluation with an LLM judge | ROUGE-L and keyword recall miss answers that are right but worded differently. |

## Done

| Item | Result |
| --- | --- |
| The first fine-tuned release | `v2026.11.13-2b04940` is the served model: 0.546 on the held-out set against the base model's 0.402 (#33). |
| Public demo access | Visitors can try the live service without a key, behind the edge rate limits (#24). |
| Documentation website | The guides in `docs/`, published as a searchable site on GitHub Pages (#30). |
| Web UI refresh | A welcome screen, access onboarding, light and dark themes, and a mobile layout (#32). |

See [CHANGELOG.md](CHANGELOG.md) for everything already released.
