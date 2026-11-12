# Hugging Face Docker Space: run the published image unchanged.
FROM ghcr.io/syed-waleed-ahmed/niche-finetuned-model:latest

# Spaces run the container as uid 1000; make the model caches writable for it.
USER root
RUN chmod -R a+rwX /models
USER 1000
ENV HOME=/tmp HF_HOME=/tmp/hf API_PORT=8080
