# Graph Report - D:\Projects & Stuff\niche_finetuned_model  (2026-07-05)

## Corpus Check
- 8 files · ~2,052 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 33 nodes · 49 edges · 9 communities detected
- Extraction: 84% EXTRACTED · 16% INFERRED · 0% AMBIGUOUS · INFERRED: 8 edges (avg confidence: 0.8)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]

## God Nodes (most connected - your core abstractions)
1. `generate_answer()` - 5 edges
2. `HealthResponse` - 4 edges
3. `build_prompt()` - 4 edges
4. `load_finetuned_model()` - 4 edges
5. `main()` - 4 edges
6. `main()` - 3 edges
7. `GenerateResponse` - 3 edges
8. `ready()` - 3 edges
9. `generate()` - 3 edges
10. `interactive_chat()` - 3 edges

## Surprising Connections (you probably didn't know these)
- `main()` --calls--> `interactive_chat()`  [INFERRED]
  D:\Projects & Stuff\niche_finetuned_model\main.py → D:\Projects & Stuff\niche_finetuned_model\src\cli.py
- `ready()` --calls--> `load_finetuned_model()`  [INFERRED]
  D:\Projects & Stuff\niche_finetuned_model\src\api.py → D:\Projects & Stuff\niche_finetuned_model\src\inference.py
- `generate_answer()` --calls--> `generate()`  [INFERRED]
  D:\Projects & Stuff\niche_finetuned_model\src\inference.py → D:\Projects & Stuff\niche_finetuned_model\src\api.py
- `main()` --calls--> `load_fastapi_dataset()`  [INFERRED]
  D:\Projects & Stuff\niche_finetuned_model\src\train_lora.py → D:\Projects & Stuff\niche_finetuned_model\src\dataset.py
- `generate_answer()` --calls--> `build_prompt()`  [INFERRED]
  D:\Projects & Stuff\niche_finetuned_model\src\inference.py → D:\Projects & Stuff\niche_finetuned_model\src\dataset.py

## Communities

### Community 0 - "Community 0"
Cohesion: 0.43
Nodes (7): generate(), GenerateRequest, GenerateResponse, health(), HealthResponse, ready(), BaseModel

### Community 1 - "Community 1"
Cohesion: 0.47
Nodes (4): lifespan(), interactive_chat(), generate_answer(), load_finetuned_model()

### Community 2 - "Community 2"
Cohesion: 0.5
Nodes (4): build_prompt(), format_example(), Build the chat-style prompt used for both training and inference., Turn an instruction/input/output example into a single chat-style string.     Y

### Community 3 - "Community 3"
Cohesion: 1.0
Nodes (2): main(), serve_api()

### Community 4 - "Community 4"
Cohesion: 0.67
Nodes (2): load_fastapi_dataset(), Load the JSONL files as a Hugging Face DatasetDict.     Each line has: instruct

### Community 5 - "Community 5"
Cohesion: 1.0
Nodes (2): main(), prepare_model_and_tokenizer()

### Community 6 - "Community 6"
Cohesion: 0.67
Nodes (0): 

### Community 7 - "Community 7"
Cohesion: 1.0
Nodes (2): Tokenize the dataset for causal LM training., tokenize_dataset()

### Community 8 - "Community 8"
Cohesion: 1.0
Nodes (0): 

## Knowledge Gaps
- **4 isolated node(s):** `Load the JSONL files as a Hugging Face DatasetDict.     Each line has: instruct`, `Build the chat-style prompt used for both training and inference.`, `Turn an instruction/input/output example into a single chat-style string.     Y`, `Tokenize the dataset for causal LM training.`
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 7`** (2 nodes): `Tokenize the dataset for causal LM training.`, `tokenize_dataset()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 8`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `generate_answer()` connect `Community 1` to `Community 0`, `Community 2`?**
  _High betweenness centrality (0.231) - this node is a cross-community bridge._
- **Why does `interactive_chat()` connect `Community 1` to `Community 3`?**
  _High betweenness centrality (0.172) - this node is a cross-community bridge._
- **Why does `build_prompt()` connect `Community 2` to `Community 1`, `Community 4`?**
  _High betweenness centrality (0.123) - this node is a cross-community bridge._
- **Are the 3 inferred relationships involving `generate_answer()` (e.g. with `generate()` and `interactive_chat()`) actually correct?**
  _`generate_answer()` has 3 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `load_finetuned_model()` (e.g. with `lifespan()` and `ready()`) actually correct?**
  _`load_finetuned_model()` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `main()` (e.g. with `load_fastapi_dataset()` and `tokenize_dataset()`) actually correct?**
  _`main()` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Load the JSONL files as a Hugging Face DatasetDict.     Each line has: instruct`, `Build the chat-style prompt used for both training and inference.`, `Turn an instruction/input/output example into a single chat-style string.     Y` to the rest of the system?**
  _4 weakly-connected nodes found - possible documentation gaps or missing edges._