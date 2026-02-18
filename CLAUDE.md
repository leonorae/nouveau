# CLAUDE.md — nouveau

Context for AI assistants working on this project.

## What this project is

**nouveau** is a tool for human-AI collaborative poetry. A human or AI agent types some unit of poetry, something else (another model or simple generator) is selected to respond, and on it goes. Finished poems are saved with metadata.

It started as a `gpt_2_simple` prototype and is being refactored into a proper, maintainable Python project. The core concept — including the generator strategy pattern and the `gpt_closure` idea — is worth preserving.

## Current state

The project is mid-refactor. The old prototype files (`poetry.py`, `data.py`) represent the original behavior. New code will go in `src/nouveau/`. Do not treat the old files as the source of truth for architecture — treat `Decisions.md` and this file as the source of truth.

## Architecture overview

```
src/nouveau/
    cli.py          # Click-based CLI entry point
    poem.py         # Poem data model and storage
    generators.py   # Generator strategy functions + registry
    model.py        # Model loading and inference abstraction
data/
    prepare.py      # Dataset prep (Gutenberg Poetry Corpus)
train.py            # Fine-tuning script (transformers Trainer)
poems/              # Output directory, gitignored
pyproject.toml      # uv project config and dependencies
```

## Key conventions

### Model
- Backend is Hugging Face `transformers`. Default model is `gpt2` (124M params, runs on CPU).
- Model loading lives in `model.py` and is abstracted behind a simple interface so the model can be swapped via config without touching generator logic.
- Fine-tuning is done locally via `train.py`, not in an external notebook.

### Generators
- Generator strategies are plain functions with the signature `(poem: Poem) -> str`.
- They are registered in a dict (`GENERATORS`) in `generators.py`, not resolved via `globals()`.
- New generators should be added to that dict — the CLI reads from it.

### Poem storage
- Poems are saved as `.json` files in `./poems/`.
- Each file has a metadata header and a `lines` list where each entry is `{"author": "human"|"ai", "text": "..."}`.
- This format is designed to extend cleanly to multi-agent scenarios.
- See `poem.py` for the schema.

### CLI
- Entry point is `nouveau` (via `pyproject.toml` script).
- `Click` is used for argument parsing.
- Keep the CLI simple. TUI (e.g. `rich`/`textual`) is on the roadmap but not now.

### Packaging
- `uv` is the package manager. Use `uv add` to add dependencies, not `pip install`.
- `uv run nouveau` to run the CLI during development.
- `uv run python train.py` for fine-tuning.

## What to be careful about

- The project is a hobby, not a product. This is serious business. SRS biznatch
- Do not add API-based model backends.

## Development workflow

```bash
uv sync                  # install dependencies
uv run nouveau 10 gpt_last   # run the CLI
uv run python train.py   # fine-tune the model
uv run pytest            # run tests (when they exist)
```

## Branch
Active development branch: `claude/review-repository-k33Uw`
