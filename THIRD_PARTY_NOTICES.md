# Third-party notices

The container image built from this repository includes the third-party material
below in addition to the Python packages installed from `requirements/serve.txt`
(whose license files are kept in each package's `dist-info` directory). The web UI
bundle ships its own notices file, `THIRD_PARTY_NOTICES.txt`, generated at build
time from its npm dependencies. See `docs/legal/LICENSE_AUDIT.md` for the full
inventory and how each license was verified.

## llama.cpp and ggml

Compiled into the `llama-cpp-python` wheel. Source: https://github.com/ggml-org/llama.cpp

```text
MIT License

Copyright (c) 2023-2026 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## FastAPI documentation

The retrieval index baked into the image (`/models/rag_index`) contains chunks of the
FastAPI documentation at tag 0.162.5. Source: https://github.com/fastapi/fastapi

```text
The MIT License (MIT)

Copyright (c) 2018 Sebastián Ramírez

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```

## bge-small-en-v1.5 embedding model

Baked into the image for retrieval (`/models/fastembed`): the quantized ONNX export
`Qdrant/bge-small-en-v1.5-onnx-Q` of `BAAI/bge-small-en-v1.5` by the Beijing Academy
of Artificial Intelligence. Both model repositories declare the MIT license in their
model cards and include no license file, so the standard MIT text is reproduced:

```text
MIT License

Copyright (c) BAAI (Beijing Academy of Artificial Intelligence)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Qwen model weights

Not included in the image by default: the GGUF is downloaded at start-up. The deployed
release, `v2026.11.14-72d6b37`, is `model-Q4_K_M.gguf` from
`syed-waleed-ahmed/fastapi-assistant-gguf`, a LoRA fine-tune of
`Qwen/Qwen2.5-Coder-1.5B-Instruct` by the Qwen team, Alibaba Cloud; its Hub
repositories carry a NOTICE that attributes the base model and describes the
modifications. Without a configured release, the image falls back to
`qwen2.5-coder-1.5b-instruct-q4_k_m.gguf` from `Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF`
(see the `Dockerfile`). All of these are licensed under the Apache License, Version 2.0
(https://www.apache.org/licenses/LICENSE-2.0). An image built with `BAKE_MODEL=true`
redistributes these weights under that license, and must carry the license text and
the attribution.

## TinyLlama model weights (earlier releases)

The releases up to `v2026.11.13-2b04940` were based on
`TinyLlama/TinyLlama-1.1B-Chat-v1.0`: the base release served
`TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF`, a quantization of it, and
`v2026.11.13-2b04940` a LoRA fine-tune of it. Both are licensed under the Apache
License, Version 2.0 (https://www.apache.org/licenses/LICENSE-2.0). They are not
included in the image by default; an image built with `BAKE_MODEL=true` for one of
those releases redistributes its weights under that license.
