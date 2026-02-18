# Decisions — nouveau

Architecture Decision Records (ADRs). Each entry records what was decided, why, and what was ruled out.

---

## ADR-001: Model backend — `transformers` over `gpt_2_simple`

**Status:** Decided

**Context:**
The original prototype used `gpt_2_simple`, which wraps GPT-2 in a TensorFlow 1.x session-based API. This library is effectively unmaintained, requires legacy TF1 patterns (`tf.Session`), and is incompatible with modern Python/TF environments. The fine-tuned checkpoint is a standard GPT-2 model and is not locked to `gpt_2_simple`'s format.

**Decision:**
Use Hugging Face `transformers` with `torch` as the inference backend. Default model is `gpt2` (124M parameters), which is CPU-friendly, free, and directly compatible with any existing fine-tuned checkpoint. The model is configurable — larger models can be specified in config without code changes.

**Ruled out:**
- *LLM APIs (OpenAI, Anthropic, etc.)* — not free, requires internet, not appropriate for a hobby project in active development.
- *llama.cpp / GGUF* — better CPU performance but adds a non-Python binary dependency and complicates fine-tuning. Worth revisiting if CPU inference becomes a bottleneck.
- *Keeping `gpt_2_simple`* — too fragile, not worth the maintenance cost.

**Consequences:**
- `torch` and `transformers` are now core dependencies (large install, but standard).
- Fine-tuning is now done via the `transformers` Trainer API (see ADR-005).
- Inference code is more explicit than the old one-liner, but also more readable and debuggable.

---

## ADR-002: Packaging — `uv` + `pyproject.toml`

**Status:** Decided

**Context:**
The original project had no dependency management at all — no `requirements.txt`, no `setup.py`. Reproducing the environment required guessing library versions. The project needs a modern, reproducible build.

**Decision:**
Use `uv` as the package manager with a `pyproject.toml` (PEP 517/518). `uv` is fast, reproducible, and produces a lockfile. Entry points are declared in `[project.scripts]` so the CLI is installed as `nouveau`.

**Ruled out:**
- *`pip` + `requirements.txt`* — works but offers no lockfile, no editable install story, no entry point declarations.
- *`poetry` (the tool, not the concept)* — heavier, slower than `uv`, and the name collision with this project is confusing.
- *`conda`* — unnecessary complexity for a pure-Python project.

**Consequences:**
- `uv sync` sets up the environment. `uv run nouveau` runs the CLI.
- A `uv.lock` file is committed for reproducibility.

---

## ADR-003: Project layout — `src/nouveau/`

**Status:** Decided

**Context:**
The original code was two flat files in the root directory. As the project grows to include a CLI, model abstraction, generators, storage, and training, a flat layout becomes hard to navigate.

**Decision:**
Use the `src/` layout (`src/nouveau/`). This is the modern Python best practice: it prevents accidental imports of uninstalled code, plays well with `pyproject.toml`, and makes the package boundary explicit.

**Consequences:**
- `poetry.py` and `data.py` are retired. Their logic is distributed to `src/nouveau/`.
- All imports use the `nouveau` package name.

---

## ADR-004: Generator dispatch — dict registry over `globals()`

**Status:** Decided

**Context:**
The original CLI resolved generator functions via `globals()[sys.argv[2]]`, allowing any name passed on the command line to be resolved as a Python global. This is effectively arbitrary code execution from CLI input.

**Decision:**
Generators are registered in an explicit `GENERATORS: dict[str, Callable]` in `generators.py`. The CLI looks up the generator name in this dict and fails with a clear error message if the name is not found.

**Ruled out:**
- *`globals()`* — unsafe, no input validation, opaque to static analysis tools.
- *Class-based strategy pattern* — overly formal for functions this simple.

**Consequences:**
- Adding a new generator requires registering it in `GENERATORS`. This is a minor friction but enforces an explicit contract.
- The CLI can enumerate available generators from the dict (useful for help text).

---

## ADR-005: Fine-tuning — in-repo `train.py` using `transformers` Trainer

**Status:** Decided

**Context:**
Fine-tuning previously lived in an external Google Colab notebook linked from `data.py`. The notebook used `gpt_2_simple`'s fine-tuning wrapper. Links can break, Colab environments drift, and the workflow is not reproducible without manual steps.

**Decision:**
Fine-tuning lives in `train.py` at the repo root, using the `transformers` Trainer API with the Gutenberg Poetry Corpus (loaded via Hugging Face `datasets`). The script runs locally or can be pointed at a free Colab/Kaggle GPU by uploading the repo.

**Context on GPU access:**
GPT-2 (124M) can be fine-tuned on a free Colab GPU tier in reasonable time. The script should work locally on CPU (slower) and on any CUDA-capable GPU.

**Consequences:**
- Fine-tuning is now self-contained and reproducible.
- `datasets` and `transformers` are shared dependencies between training and inference.
- `data/prepare.py` handles dataset download and preprocessing; `train.py` handles the training loop.

---

## ADR-006: Poem storage — structured JSON

**Status:** Decided

**Context:**
The original code saved poems as JSON content inside files named with a `.txt` extension (e.g. `2024-01-01 12:00:00.txt`). The JSON schema was minimal: `{"generator": "...", "poem": ["line1", "line2"]}`. This loses authorship information (which lines were human vs. AI) and doesn't scale to more complex scenarios.

**Decision:**
Poems are saved as `.json` files with a richer schema:

```json
{
  "schema_version": 1,
  "created_at": "2024-01-01T12:00:00",
  "model": "gpt2",
  "generator": "gpt_last",
  "lines": [
    {"author": "human", "text": "..."},
    {"author": "ai",    "text": "..."}
  ]
}
```

The `lines` array uses `{"author", "text"}` objects to preserve authorship. `author` is `"human"` or `"ai"` now, but can extend to agent names for multi-agent scenarios.

**Ruled out:**
- *TOML* — awkward for lists of structured text, better suited to config files than data files.
- *Markdown with YAML frontmatter* — human-readable, but parsing frontmatter adds a dependency and the format is fiddly when text contains special characters.
- *Plain text* — loses all metadata; not extensible.
- *SQLite* — overkill for a hobby CLI tool saving a few files.

**Consequences:**
- Poem files are not human-readable at a glance (unlike plain text). Acceptable tradeoff.
- A simple pretty-printer or viewer command could be added later to the CLI.
- `schema_version` is included from the start so the format can evolve without breaking old files.

---

## ADR-007: CLI framework — `click`

**Status:** Decided

**Context:**
The original CLI used raw `sys.argv`. This provides no help text, no type coercion, and no error messages for bad input.

**Decision:**
Use `click` for the CLI. It is lightweight, well-documented, and produces good help text with minimal boilerplate.

**Ruled out:**
- *`argparse`* — fine but more verbose for the same result; no decorator-based API.
- *`typer`* — built on `click`, adds type-annotation magic. Slightly heavier, unnecessary for a simple CLI.

**Consequences:**
- `click` is a dependency (small and stable).
- The CLI gains `--help`, proper error messages, and type validation for free.
