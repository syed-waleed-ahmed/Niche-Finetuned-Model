---
title: FastAPI Assistant
sdk: docker
app_port: 8080
pinned: false
license: mit
short_description: Fine-tuned small model that answers FastAPI questions
---

# FastAPI Assistant (Hugging Face Space origin)

An optional third origin. This folder becomes the root of a Docker Space: it reuses
the published multi-arch image from GHCR, so nothing is built here.

Setup:

1. Create a new Space with the Docker SDK and push these two files to it.
2. In Space settings, add secrets `ORIGIN_SHARED_SECRET`, `DATABASE_URL`, and
   `METRICS_TOKEN` (same values as the other origins), and variables from
   `fastapi-assistant export env model-release.json`.
3. Add the Space URL (`https://<user>-<space>.hf.space`) to the Worker's `ORIGINS`.

Free CPU Spaces sleep when idle and wake on the next request (a slow first answer).
Hugging Face changed free-tier Space policies during 2026, so check that the free CPU
hardware is still available to your account before relying on this origin.
