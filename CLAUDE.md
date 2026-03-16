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

## Extended architecture: machines, corpora, directors, recursion

This section formalizes where the project is heading. None of this is implemented yet. It is written here so that AI collaborators working on the codebase have a shared model of the design intent.

### The score factory as unifying interface

Score factories currently do one thing: rejection sampling at inference time. But the interface `(Poem) -> (str) -> float` is general enough to unify four separate operations:

```
1. Inference-time selection     make_constrained_generator  ← implemented
2. Corpus filtering             keep poems below cost threshold
3. Training-time reward         GRPO via trl
4. Machine fitness              select/mutate generator configurations
```

Everything downstream — corpus curation, RL training, machine evolution — can use the same scoring vocabulary already defined. No new interface needed.

### Machine

A **machine** is a serializable poem-generation configuration. Currently implicit in CLI flags; should become an explicit, storable, shareable object.

```python
@dataclass
class Machine:
    generator: str        # name in GENERATORS or factory spec, e.g. "fugue" or "syllables:7"
    model: str            # HuggingFace model ID or local path
    temperature: float
    max_lines: int
    authors: list[str]    # ["human"] for compose, ["A", "B"] for duet, etc.
    seed_from: str | None # path to a poem whose last lines seed the context
    score: str | None     # score factory spec for corpus-level selection
```

A machine is a recipe. Running it produces a poem. A duet is two machines sharing a poem. The goal is to make machines serializable as JSON so they can be stored, shared, mutated, and passed between agents.

### Corpus

A **corpus** is a collection of poems with shared provenance. Operations:

- `run(machine, n)` — run a machine n times headlessly, collect poems
- `rank(score_factory)` — order poems by cost (lower = better fit for the score)
- `filter(score_factory, threshold)` — keep only poems below a cost threshold
- `sample(n, weighted=True)` — draw poems, optionally weighted by score
- `as_context(selector)` — extract lines from the corpus as a new ContextFn

The last one is the recursive step: a corpus becomes an input to a new machine. A poem produced by `fugue` can be fed as context to `trance`, which produces a poem that is then filtered by `novelty_scorer`, whose survivors become training data.

### Director

A **director** is an agent that does not write lines but shapes the session. Currently the human at the CLI is the only director (they choose when to stop, can Ctrl-C). The design is to make direction explicit as an event/command protocol:

```
Events (session emits):  line_generated(line, author, turn)
                         poem_complete(poem)

Commands (director sends): inject(text, author)
                           switch_generator(name)
                           set_temperature(t)
                           end()
```

A director can be: a human watching output, another LLM evaluating lines, a script watching a score threshold and switching generators when it's crossed, or a timer. Multiple directors can watch the same session. This is also the foundation for the human-intervenes-on-AI-duet workflow: run `fugue vs trance`, but the human can inject a line or switch one generator mid-poem when something interesting happens.

### The recursive loop

```
machines → run → corpus → score → select
   ↑                                 |
   └──── mutate ← analyze ←─────────┘
                     ↑
             score factories
```

And between levels:

- a poem is an output, and also a valid seed (via `seed_from`)
- a corpus is a collection, and also a training dataset (via `train.py`)
- a machine spec is just data — it can be generated by an LLM given a corpus analysis
- so: `LLM → machine spec → run → corpus → analyze → LLM → new machine spec → ...`

The human or an orchestrating agent sits outside this loop, occasionally intervening: changing the score function, injecting a seed poem, deciding which corpus to promote to training data.

### What gets built next, in order

1. **Headless batch runner** — `nouveau run machines/fugue.json 20` runs a machine 20 times, saves to a named corpus directory. (The `duet` command is almost this already.)
2. **Corpus tools** — `nouveau corpus rank poems/ --score novelty` prints ranked poems; `nouveau corpus filter` keeps the good ones.
3. **Machine files** — JSON schemas for machines; a `machines/` directory of named configurations the way `poems/` holds results.
4. **Director protocol** — event hooks in the session loop; a `--director` flag for `compose`/`duet` that loads a director script.
5. **RL training loop** (`train_rl.py`) — score factories → GRPO rewards via `trl`. Sessions log candidates; periodic updates personalise the model.

## What's next (near term)

- **Scoring improvements**: the current heuristics (vowel-cluster syllables, last-N-chars rhyme) are rough. `pronouncing`/cmudict for phonemes, `pyphen` for syllables are the next step up without heavy deps.
- **Machine serialization**: make `Machine` a dataclass, add `machines/` directory, wire up `nouveau run`.
- **Corpus CLI**: `nouveau corpus` subcommand with `run`, `rank`, `filter`, `show`.

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
