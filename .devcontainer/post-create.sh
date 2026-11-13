#!/usr/bin/env bash
# Runs once when the dev container is created. Installs what the test suites need,
# the same way CI does, and the git hooks.
set -euo pipefail

python -m pip install --upgrade pip
# The default torch wheel bundles CUDA (several GB); the tests need no GPU.
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev,train,postgres]"
pip install pre-commit
pre-commit install

(cd deploy/cloudflare-worker && npm ci)
(cd web && npm ci)

cat <<'EOF'

Dev container ready.

  pytest                          Python tests (no model, network or database needed)
  make lint                       ruff and mypy
  make worker-test web-test       Worker and web UI tests

To serve the released model (downloads a 0.7 GB GGUF and compiles llama.cpp once):

  pip install -e ".[serve]"
  fastapi-assistant export env model-release.json > .env
  fastapi-assistant serve         # http://localhost:8000/docs
  make web-dev                    # in a second terminal: http://localhost:5173

EOF
