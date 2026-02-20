# CLAUDE.md — nouveau

Context for AI assistants working on this project.

## What this project is

**nouveau** is a collaborative poetry environment. The current form is a CLI where a human types a line and an AI generates the next, alternating until the poem reaches a target length. Finished poems are saved with metadata.

The direction it's heading: LLMs as first-class participants — not just backends, but poets, directors, and trainers. Humans can direct LLMs, LLMs can direct other LLMs, and poem sessions can feed back into model weights via RL. The codebase itself is fair game for AI collaborators to read, extend, and edit. Recursive and experimental is fine here.

The refactor from a `gpt_2_simple` prototype is complete. The architecture is stable.

## Architecture

```
src/nouveau/
    cli.py          # Click-based CLI entry point
    poem.py         # Poem data model and storage
    generators.py   # Generator strategies, combinators, score factories, registry
    model.py        # Model loading and inference (GPT-2 via transformers)
data/
    prepare.py      # Dataset prep (Gutenberg Poetry Corpus)
train.py            # Fine-tuning script (transformers Trainer API)
poems/              # Saved poems, gitignored
pyproject.toml      # uv project config
```

## Generator architecture

`generators.py` has three layers:

```
ContextFn    = (Poem) -> str              # selects text from the poem
GeneratorFn  = (Poem, Model) -> str       # calls the model, returns a line
ScoreFactory = (Poem) -> (str) -> float   # cost function for rejection sampling
```

**Combinators:**
- `make_generator(context_fn)` — bridges ContextFn → GeneratorFn
- `make_conditional(condition, if_true, if_false)` — state-dependent dispatch
- `make_constrained_generator(context_fn, make_score, n_candidates, max_new_tokens)` — generates N candidates, returns the one with lowest cost

**Context selectors** (factory functions, any granularity):
- Line: `last_lines(n)`, `first_lines(n)`, `line_window(int|list[int]|slice)`
- Word: `last_words(n)`, `first_words(n)`

**Score factories** (pluggable reward functions — same interface RL will use):
- `syllable_scorer(target)` — distance from target syllable count
- `rhyme_scorer(n_chars)` — end-sound match with previous line
- `sentiment_scorer(target)` — VADER compound distance from target

Named instances are registered in `GENERATORS`. The CLI reads from it; `click.Choice` is generated dynamically. Adding a generator is one line in that dict.

The score factories are already reward functions. The path from `make_constrained_generator` (inference-time rejection sampling) to GRPO (training-time group scoring with gradient updates) is short — they're the same operation, one updates weights and one doesn't.

## Model

- Backend: HuggingFace `transformers`, default `gpt2` (124M, CPU-friendly).
- Interface: `Model(model_name, temperature).generate(prefix, max_new_tokens)`.
- Swappable via CLI `--model` flag without touching generator logic.
- Fine-tuning: `train.py` using the Trainer API on the Gutenberg Poetry Corpus.
- Everything runs locally, offline, for free.

## Poem storage

JSON files in `./poems/`. Schema:
```json
{
  "schema_version": 1,
  "created_at": "...",
  "model": "gpt2",
  "generator": "last",
  "lines": [
    {"author": "human", "text": "..."},
    {"author": "ai",    "text": "..."}
  ]
}
```
`author` is `"human"` or `"ai"` now, but the field is designed to hold agent names for multi-agent scenarios.

## What's next

- **RL training loop** (`train_rl.py`): score factories → reward functions, GRPO via `trl`. Poem sessions log accepted/rejected lines; periodic updates personalise the model to the collaborator's aesthetic.
- **Multi-agent composition**: multiple `Model` + `GeneratorFn` pairs taking turns, with a director (human or LLM) choosing which voice speaks next.
- **Scoring improvements**: the current heuristics (vowel-cluster syllables, last-N-chars rhyme) are rough. `pronouncing`/cmudict for phonemes, `pyphen` for syllables are the next step up without heavy deps.

## Constraints

- No web server, no database, no async infrastructure — not now.
- No API-based model backends. Local and offline.
- `uv` for packaging. `uv add` to add deps, not `pip install`.
- Prefer simple and readable. This is experimental, not a product.

## Development workflow

```bash
uv sync                          # install dependencies
uv run nouveau compose 10 last   # run the CLI
uv run python train.py           # fine-tune the model
uv run pytest                    # run tests
```

## Branch

Active development branch: `claude/review-repository-k33Uw`
