#!/usr/bin/env bash
# Convert a merged Hugging Face model to a quantized GGUF for llama.cpp.
#
#   scripts/export_gguf.sh MERGED_DIR OUT_DIR [QUANT]
#
# MERGED_DIR comes from `fastapi-assistant export merge --out MERGED_DIR`.
# QUANT defaults to Q4_K_M (about 0.7 GB for a 1.1B model, small quality loss).
# Linux/macOS; needs git, cmake, a C++ compiler, and torch + transformers +
# sentencepiece already installed (true on Kaggle and in the training extra).
# The converter embeds the tokenizer's chat template in the GGUF, so the served
# prompt format matches training without extra configuration.
set -euo pipefail

MERGED_DIR=${1:?usage: export_gguf.sh MERGED_DIR OUT_DIR [QUANT]}
OUT_DIR=${2:?usage: export_gguf.sh MERGED_DIR OUT_DIR [QUANT]}
QUANT=${3:-Q4_K_M}
LLAMA_CPP_REF=${LLAMA_CPP_REF:-b11971}
WORK=${LLAMA_CPP_DIR:-${TMPDIR:-/tmp}/llama.cpp-${LLAMA_CPP_REF}}

mkdir -p "$OUT_DIR"
if [ ! -d "$WORK" ]; then
  git clone --depth 1 --branch "$LLAMA_CPP_REF" https://github.com/ggml-org/llama.cpp "$WORK"
fi

# The converter only needs the bundled gguf-py package on top of torch/transformers.
# Installing the full requirements file would replace a CUDA torch with a CPU build.
pip install --quiet "$WORK/gguf-py"

F16="$OUT_DIR/model-f16.gguf"
python "$WORK/convert_hf_to_gguf.py" "$MERGED_DIR" --outtype f16 --outfile "$F16"

cmake -S "$WORK" -B "$WORK/build" -DGGML_NATIVE=OFF -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build "$WORK/build" --target llama-quantize -j "$(nproc 2>/dev/null || echo 4)" >/dev/null
"$WORK/build/bin/llama-quantize" "$F16" "$OUT_DIR/model-$QUANT.gguf" "$QUANT"
rm -f "$F16"

ls -lh "$OUT_DIR"/*.gguf
