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
| Larger, more varied training set | Keyword recall did not move with the first fine-tune (0.426 against the base model's 0.432). With 151 more training rows (351 to 502) and the Qwen base model it rose to 0.685 (#44, #53), but about a third of the required API names are still missing, and live answers still get FastAPI wrong. More questions per topic, especially on the weakest categories (logging, deployment, rate limiting, streaming), should help. |
| Measure retrieval with the fine-tuned model | Retrieval was measured on the base model only, where it was a trade-off. The fine-tune trained on context examples, so `fastapi-assistant eval --rag` on it decides whether retrieval should become the default. |

## Next

| Item | Why |
| --- | --- |
| Narrower deployer permissions | Grant the CI deployer access to individual secrets instead of the whole project (audit item O-4). |
| A generation check in the uptime probe | The probe covers `/health` and `/v1/models`; a tiny generation would also catch a broken model. |

## Later

| Item | Why |
| --- | --- |
| Retrieval on by default | Answers grounded in the FastAPI documentation are more accurate; the index needs to fit the image size and cold-start budget. |
| Backup origin | A Hugging Face Space or a small VM behind the edge circuit breaker (configurations exist in `deploy/`). |
| Evaluation with an LLM judge | ROUGE-L and keyword recall miss answers that are right but worded differently. |

## Done

| Item | Result |
| --- | --- |
| A stronger base model | `v2026.11.14-72d6b37`, a fine-tune of Qwen2.5-Coder-1.5B-Instruct, replaced the TinyLlama-1.1B fine-tune: 0.666 on the held-out set against 0.546, served on the same free-tier CPU at about 7.6 tokens/s (#44, #53). It is still often wrong. |
| The first fine-tuned release | `v2026.11.13-2b04940`, a TinyLlama-1.1B fine-tune, served from #33 until #53: 0.546 on the held-out set against the base model's 0.402. |
| Public demo access | Visitors can try the live service without a key, behind the edge rate limits (#24). |
| Documentation website | The guides in `docs/`, published as a searchable site on GitHub Pages (#30). |
| Web UI refresh | A welcome screen, access onboarding, light and dark themes, and a mobile layout (#32). |
| The first four starter issues | Non-streaming generations stop when the caller disconnects (#26), edge logs identify the key (#27), Grafana panels for queue wait, first token and prompt rejections (#28), and a JavaScript streaming example (#29). |

See [CHANGELOG.md](CHANGELOG.md) for everything already released.
