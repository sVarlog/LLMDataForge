# 🧠 LLM Fine-Tuning, Merging & GGUF Conversion

This repo shows how to:

1. **Fine-tune** a base LLM (e.g. Qwen/DeepSeek) with QLoRA
2. **Merge** the resulting LoRA adapter into the base model
3. **Convert** the merged model into GGUF (for `llama.cpp` / local inference)

---

## 📂 Project Layout

```
.
├── config/
│   ├── config.py             # Central paths/model identifiers
│   ├── training_config.py    # Training + prompt constants (DATA_PATH, epochs, etc.)
│   └── lora_config.json      # LoRA/QLoRA hyper-parameters consumed by src/train.py
├── datasets/                 # Legacy single-file builders kept for reference
│   ├── build_dataset.py
│   └── <domain>/             # Original category JSONL shards
├── datasets_new/             # Canonical hierarchical datasets (see datasets_new/README.md)
│   ├── build_train_jsonl.py  # Flattens topics → train_data.jsonl with stats
│   ├── create_dataset_by_structure.py
│   ├── schemas/              # reasoning + difficulty schemas
│   ├── scripts/              # helpers such as normalizeJson.py
│   ├── structure.enriched.json
│   └── topics/<category>/<subcategory>/<category>.<subcategory>.<content>.json
├── merged-models/deepseek-ai/  # LoRA merged HF folders + gguf-output
├── output/deepseek-ai/         # QLoRA checkpoints (training-*/checkpoint-*)
├── src/
│   ├── main.py                # RUN_MODE-aware entrypoint
│   ├── train.py               # QLoRA training & generation helpers
│   ├── helpers/               # prompt builders, loggers, template copier
│   ├── merge/merge_adapter.py # Adapter → base merger
│   ├── convert/convert_to_gguf.sh
│   └── test/test_model.py     # HF + GGUF smoke tests
├── templates/chat_template.jinja # Copied to checkpoints/merged outputs
├── tools/llama/               # transformers-to-gguf.py and friends (MIT-licensed)
├── requirements.txt / _requirements.txt
├── docker-compose.yml / Dockerfile
└── env/                       # Local virtual environment (not tracked)
```

Key notes:

-   `config/` centralizes all paths/hyper-parameters so scripts can stay argument-light.
-   `src/main.py` orchestrates training/tests through the `RUN_MODE` env (train, test-training, test-merging, test-gguf).
-   `templates/chat_template.jinja` ships with every checkpoint/merge to keep chat formatting consistent.
-   `tools/llama/` mirrors upstream llama.cpp conversion utilities that are invoked by `src/convert/convert_to_gguf.sh`.
-   `env/` is a convenience virtual environment for local runs; recreate it via `python -m venv env && env/Scripts/activate` if desired.

---

## 📖 Datasets

-   `datasets/` retains the first-generation domain JSON plus helper scripts (`build_dataset.py`, `combine_datasets.py`). These files are still handy for quick experiments but are no longer the source of truth.
-   `datasets_new/` is the canonical, metadata-rich dataset pipeline:
    -   `structure.enriched.json` defines every category, subcategory, description, tags, and content types.
    -   `topics/<category>/<subcategory>/<category>.<subcategory>.<content_type>.json` hosts the authored samples (now includes categories like `identity` that rely on placeholder substitution).
    -   `schemas/` houses both `schema_reasoning.json` (question/think/output layout) and `difficulty_schema.json` (allowed range + semantics for 1–6 difficulty levels).
    -   `scripts/normalizeJson.py` plus `scripts/placeholder.txt` make it easy to paste multiline text and emit JSON-safe strings before dropping them into topic files.
    -   `build_train_jsonl.py` consolidates everything into `train_data.jsonl`, normalizes difficulty values, injects tags/metadata, runs optional token statistics (`--tokenizer-path`, `--no-stats`), and replaces identity placeholders so the final samples are trainer-ready.
    -   `create_dataset_by_structure.py` can scaffold missing topic directories/files based on the enriched structure.
-   A dedicated `datasets_new/README.md` (kept in sync with the folder contents) documents contribution rules, schema expectations, and troubleshooting steps. Always update that file alongside new data drops.

`config/training_config.DATA_PATH` points to `datasets_new/train_data.jsonl` by default, so regenerating the file immediately feeds the latest data into training.

---

## ⚙️ Configuration & orchestration

-   Edit `config/config.py` to change base model IDs, repo names, and shared directories.
-   Tune `config/training_config.py` for dataset paths, training epochs, resume flags (`TRAINING_NEW`, `TRAINING_EXTRA_EPOCHS`), stopping delimiters, evaluation prompts, and logging destinations.
-   `config/lora_config.json` collects all LoRA ranks/alphas/dropouts that `src/train.py` loads dynamically.
-   `templates/chat_template.jinja` is persisted into every checkpoint/merge via `src/helpers/persist_chat_template.py`, ensuring downstream inference uses the same chat format.
-   `src/helpers/` bundles `build_messages.py` (prompt assembly), `loggers.py`, and other utilities shared between training, evaluation, and conversion.
-   `src/main.py` reads `RUN_MODE` (`train`, `test-training`, `test-merging`, `test-gguf`) to sequence training and smoke tests without juggling multiple entrypoints.

## 🚀 Quickstart

### 1. Build your dataset

```powershell
python datasets_new/build_train_jsonl.py --structure datasets_new/structure.enriched.json --topics-dir datasets_new/topics --output datasets_new/train_data.jsonl
```

Add `--tokenizer-path <local-model-or-HF-id>` to gather token statistics, or `--no-stats` if you only need the JSONL. Identity-focused categories automatically swap `${NAME}`/`${SPEC}` placeholders during this step.

### 2. Train with QLoRA

```powershell
# equivalent forms:
python src/train.py
# or:  set RUN_MODE=train; python -m src.main
```

Outputs land under `output/deepseek-ai/training-*/checkpoint-*`. The trainer copies `tokenizer.json`, `vocab.json`, `merges.txt`, `special_tokens_map.json`, and `chat_template.jinja` into each checkpoint and records the base-model path for offline evaluation.

Notes on resuming training

-   To continue the last training run instead of starting a new one, set `TRAINING_NEW = False` in `config/training_config.py`.
-   The resume logic prefers an epoch-based continuation: the trainer reads the epoch recorded in the checkpoint's `trainer_state.json` and will extend training by `TRAINING_EXTRA_EPOCHS` (see `TRAINING_EPOCHS` and `TRAINING_EXTRA_EPOCHS` in `config/training_config.py`). This avoids issues with absolute `max_steps` when resuming from checkpoints.
-   The codebase also includes small helpers under `src/helpers/` (for example `build_messages.py` and `loggers.py`) to keep prompt construction and logging consistent when resuming and running generations.

Generation stopping

-   The training/generation utilities include a decoding-based stopper that looks for output delimiters like `</output>` (or the model's end token) in decoded text rather than relying solely on exact token-id sequences. This is more robust across tokenizers and prevents the model from emitting unwanted extra tokens after the intended end marker.

### 3. Merge LoRA into the base

```powershell
python src/merge/merge_adapter.py
```

-   Picks the **last** `training-*` / `checkpoint-*`
-   Reads the adapter’s added embedding rows (via `adapter_model.safetensors`)
-   Resizes the HF base model to match
-   Merges & unloads LoRA weights
-   Saves under `merged-models/deepseek-ai/merging-K/`
-   Copies across your **full** trained-tokenizer artifacts:
    -   `tokenizer.json`
    -   `vocab.json`
    -   `merges.txt`
    -   `special_tokens_map.json`
    -   `chat_template.jinja`

### 4. Convert to GGUF

```bash
bash src/convert/convert_to_gguf.sh --outtype q8_0
```

-   Locates the latest `merged-models/.../merging-K/`
-   Runs `transformers-to-gguf.py` → emits `*.gguf` in `merging-K/gguf-output/`

---

## 🧪 Smoke tests & evaluations

`src/test/test_model.py` contains three helpful entry points that the main runner can call automatically (set `RUN_MODE` to `test-training`, `test-merging`, or `test-gguf`) or run ad hoc from Python:

-   `run_test_training()` attaches the latest adapter checkpoint to the base model and prints completions for curated prompts (no internet access required if you provide `BASE_MODEL_DIR`).
-   `run_test_merging()` validates the most recent merged HF model under `merged-models/deepseek-ai/`.
-   `run_test_gguf()` spins up `llama.cpp` via `llama-cpp-python` against the newest GGUF artifact and reuses the same tokenizer/template for apples-to-apples comparisons.

Each mode respects `TEST_MODE`, `TEST_SAMPLES`, and context/window env vars so you can gate deployments with quick, deterministic sanity checks.

---

## 📝 Why copy _all_ tokenizer files?

When you added custom special/chat tokens and a Jinja template:

-   **`tokenizer.json`** holds your merges + special tokens + chat_template
-   **`vocab.json`** + **`merges.txt`** define your BPE vocabulary
-   **`special_tokens_map.json`** maps names → IDs
-   **`chat_template.jinja`** is your prompt-format template

By shipping them alongside the merged model, you preserve _exactly_ the same tokenization and chat layout your fine-tune used.

---

## 🛠 Fine-Tuning Tips

-   Use small batches (2–4) with gradient accumulation 16–32
-   Train for 3–5 epochs on ~2–3K samples to start
-   Monitor loss & generations via the built-in eval callback

---

## 🎉 Results

-   Adapter merging “just worked” once we resized embeddings and carried over the custom tokenizer.
-   Downstream GGUF conversion now sees the proper `tokenizer.model` alongside JSON/BPE files.

---

<!-- Third-Party Code -->

## 🛠️ Third-Party Code

We include parts of the [llama.cpp](https://github.com/ggml-org/llama.cpp) project under its MIT license:

```bash
Copyright (c) 2023-2024 The ggml authors
Copyright (c) 2023 Georgi Gerganov
```

### Those files are included verbatim from llama.cpp and are subject to the same MIT terms:

-   `tools/llama/convert_hf_to_gguf.py`
-   `tools/llama/convert_hf_to_gguf_update.py`
-   `tools/llama/convert_llama_ggml_to_gguf.py`
-   `tools/llama/convert_lora_to_gguf.py`
-   `tools/llama/gguf-py/gguf/*`
