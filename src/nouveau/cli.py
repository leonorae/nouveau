import json
from pathlib import Path

import click

from nouveau.generators import GENERATOR_FACTORIES, GENERATORS, score_poem
from nouveau.model import DEFAULT_MODEL, Model
from nouveau.poem import POEM_DIR, Poem


def _parse_generator(spec: str):
    """Resolve a generator spec into a GeneratorFn.

    Accepts:
      name          — look up in GENERATORS (e.g. 'last', 'rhyme')
      name:arg      — call GENERATOR_FACTORIES[name](arg) (e.g. 'syllables:5')
    """
    if ":" in spec:
        name, _, arg = spec.partition(":")
        if name not in GENERATOR_FACTORIES:
            _bad_generator(spec)
        try:
            return GENERATOR_FACTORIES[name](arg)
        except (ValueError, TypeError) as exc:
            raise click.BadParameter(
                f"bad argument for '{name}': {exc}", param_hint="GENERATOR"
            ) from exc
    if spec not in GENERATORS:
        _bad_generator(spec)
    return GENERATORS[spec]


def _bad_generator(spec: str) -> None:
    fixed = ", ".join(sorted(GENERATORS))
    parameterized = ", ".join(f"{k}:<arg>" for k in sorted(GENERATOR_FACTORIES))
    raise click.BadParameter(
        f"'{spec}' is not a valid generator.\n"
        f"  fixed:         {fixed}\n"
        f"  parameterized: {parameterized}",
        param_hint="GENERATOR",
    )


@click.group()
def cli() -> None:
    """nouveau — human-AI collaborative poetry."""


@cli.command()
@click.argument("max_lines", type=int)
@click.argument("generator", type=str, metavar="GENERATOR")
@click.option(
    "--model",
    "model_name",
    default=DEFAULT_MODEL,
    show_default=True,
    help="HuggingFace model ID or path to a local checkpoint.",
)
@click.option(
    "--temperature",
    default=0.7,
    show_default=True,
    help="Sampling temperature for generation.",
)
def compose(max_lines: int, generator: str, model_name: str, temperature: float) -> None:
    """Compose a poem: you write a line, the model writes the next.

    MAX_LINES is the total number of lines before the poem ends.

    GENERATOR is the strategy for AI line generation. Pass a name
    (last, first, window, bookend, alternating, closure, rhyme, hopeful, somber,
    cutup, erased, folded, markov, oulipo, dissolve, vanish, drift)
    or a parameterized factory (syllables:5, rhyme:4, sentiment:0.8,
    erasure:0.3, nplus:7, markov:2, lipogram:e).
    """
    if max_lines < 2:
        raise click.BadParameter("must be at least 2", param_hint="MAX_LINES")

    generator_fn = _parse_generator(generator)

    click.echo(f"Loading {model_name}...")
    model = Model(model_name=model_name, temperature=temperature)

    poem = Poem(max_lines=max_lines, generator_name=generator, model_name=model_name)

    click.echo("Begin. Press Ctrl-C to quit without saving.\n")

    while not poem.is_full():
        line = click.prompt("", prompt_suffix="")
        poem.add_line(line, author="human")

        if not poem.is_full():
            ai_line = generator_fn(poem, model)
            poem.add_line(ai_line, author="ai")
            click.echo(ai_line)

    path = poem.save()
    click.echo(f"\nPoem saved to {path}")


@cli.command()
@click.argument("max_lines", type=int)
@click.argument("generator1", type=str, metavar="GENERATOR1")
@click.argument("generator2", type=str, metavar="GENERATOR2")
@click.option("--name1", default="A", show_default=True, help="Name for the first agent.")
@click.option("--name2", default="B", show_default=True, help="Name for the second agent.")
@click.option(
    "--model",
    "model_name",
    default=DEFAULT_MODEL,
    show_default=True,
    help="HuggingFace model ID or path to a local checkpoint.",
)
@click.option(
    "--temperature",
    default=0.7,
    show_default=True,
    help="Sampling temperature for generation.",
)
def duet(
    max_lines: int,
    generator1: str,
    generator2: str,
    name1: str,
    name2: str,
    model_name: str,
    temperature: float,
) -> None:
    """Two AI generators compose a poem together.

    MAX_LINES total lines, alternating between GENERATOR1 and GENERATOR2.
    After the poem, prints a brief influence report.
    """
    if max_lines < 2:
        raise click.BadParameter("must be at least 2", param_hint="MAX_LINES")

    gen_fn1 = _parse_generator(generator1)
    gen_fn2 = _parse_generator(generator2)

    click.echo(f"Loading {model_name}...")
    model = Model(model_name=model_name, temperature=temperature)

    poem = Poem(
        max_lines=max_lines,
        generator_name=f"{generator1}|{generator2}",
        model_name=model_name,
    )

    pad = max(len(name1), len(name2))
    click.echo(f"\n{name1:<{pad}}  ({generator1})")
    click.echo(f"{name2:<{pad}}  ({generator2})")
    click.echo()

    agents = [(name1, gen_fn1), (name2, gen_fn2)]
    turn = 0
    while not poem.is_full():
        name, gen_fn = agents[turn % 2]
        line = gen_fn(poem, model)
        poem.add_line(line, author=name)
        click.echo(f"{name:<{pad}}  {line}")
        turn += 1

    path = poem.save()
    click.echo(f"\nPoem saved to {path}")

    # influence report
    lines1 = [l for l in poem.lines if l.author == name1]
    lines2 = [l for l in poem.lines if l.author == name2]

    def vocab(lines):
        return {w.lower().strip(".,;:!?\"'") for l in lines for w in l.text.split() if w}

    words1, words2 = vocab(lines1), vocab(lines2)
    if words1 and words2:
        overlap = words1 & words2
        jaccard = len(overlap) / len(words1 | words2)
        shared = ", ".join(sorted(overlap)[:10]) + (" ..." if len(overlap) > 10 else "")
        avg1 = sum(len(l.text.split()) for l in lines1) / len(lines1)
        avg2 = sum(len(l.text.split()) for l in lines2) / len(lines2)
        click.echo(f"\n--- influence ---")
        click.echo(f"vocabulary overlap  {jaccard:.2f}  ({len(overlap)} shared words)")
        click.echo(f"shared: {shared}")
        click.echo(f"avg words/line  {name1}={avg1:.1f}  {name2}={avg2:.1f}")


@cli.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def show(path: Path) -> None:
    """Display a saved poem."""
    data = json.loads(path.read_text())
    click.echo(f"{data['generator']} · {data['model']} · {data['created_at']}")
    click.echo()
    for entry in data["lines"]:
        label = "you" if entry["author"] == "human" else entry["author"]
        click.echo(f"{label}  {entry['text']}")


@cli.command()
@click.argument("generator1", type=str, metavar="GENERATOR")
@click.argument("generator2", type=str, metavar="GENERATOR2", required=False, default=None)
@click.option("--n", default=1, show_default=True, help="Number of poems to generate.")
@click.option("--lines", default=10, show_default=True, help="Lines per poem.")
@click.option("--out", "out_dir", default=None, help="Output directory (default: poems/).")
@click.option("--name1", default="A", show_default=True, help="Author name for first generator.")
@click.option("--name2", default="B", show_default=True, help="Author name for second generator.")
@click.option("--model", "model_name", default=DEFAULT_MODEL, show_default=True)
@click.option("--temperature", default=0.9, show_default=True)
@click.option("--seed", "seed_path", default=None, type=click.Path(exists=True, dir_okay=False),
              help="Seed each poem with the last lines of an existing poem file.")
@click.option("--seed-lines", default=2, show_default=True,
              help="How many trailing lines to import from --seed.")
def run(
    generator1: str,
    generator2: str | None,
    n: int,
    lines: int,
    out_dir: str | None,
    name1: str,
    name2: str,
    model_name: str,
    temperature: float,
    seed_path: str | None,
    seed_lines: int,
) -> None:
    """Run a generator headlessly N times, saving poems to a corpus directory.

    With one GENERATOR, the model writes every line (monologue).
    With two GENERATORs, they alternate (duet). No human input.

    \b
    Examples:
      nouveau run fugue --n 20 --out corpora/dream
      nouveau run trance spine --n 10 --lines 14 --out corpora/cross
      nouveau run mirror --seed poems/old.json --seed-lines 2 --out corpora/seeded
    """
    if lines < 2:
        raise click.BadParameter("must be at least 2", param_hint="--lines")

    gen_fn1 = _parse_generator(generator1)
    gen_fn2 = _parse_generator(generator2) if generator2 else gen_fn1
    author2 = name2 if generator2 else name1

    out_path = Path(out_dir) if out_dir else POEM_DIR
    gen_label = f"{generator1}|{generator2}" if generator2 else generator1

    seed_poem: Poem | None = None
    if seed_path:
        seed_poem = Poem.load(Path(seed_path))

    click.echo(f"Loading {model_name}...")
    model = Model(model_name=model_name, temperature=temperature)
    click.echo(f"Running {n} × {lines}-line [{gen_label}] → {out_path}/\n")

    for i in range(n):
        seed_count = 0
        if seed_poem:
            seed_tail = seed_poem.lines[-seed_lines:]
            seed_count = len(seed_tail)
        poem = Poem(max_lines=lines + seed_count, generator_name=gen_label, model_name=model_name)
        if seed_poem:
            for sl in seed_tail:
                poem.lines.append(sl)
        agents = [(name1, gen_fn1), (author2, gen_fn2)]
        turn = 0
        while not poem.is_full():
            name, gen_fn = agents[turn % 2]
            line = gen_fn(poem, model)
            poem.add_line(line, author=name)
            turn += 1
        path = poem.save(out_path)
        # preview: first non-seed line
        first_new = poem.lines[seed_count] if seed_count < len(poem.lines) else poem.lines[0]
        preview = first_new.text[:60] + ("…" if len(first_new.text) > 60 else "")
        click.echo(f'  [{i+1:>3}/{n}] {path.name}  \u201c{preview}\u201d')

    click.echo(f"\nDone. {n} poems saved to {out_path}/")


# ---------------------------------------------------------------------------
# corpus subgroup
# ---------------------------------------------------------------------------

def _load_corpus(directory: Path) -> list[tuple[Path, Poem]]:
    """Load all poem JSON files from a directory."""
    paths = sorted(directory.glob("*.json"))
    if not paths:
        raise click.ClickException(f"No poems found in {directory}")
    poems = []
    for p in paths:
        try:
            poems.append((p, Poem.load(p)))
        except Exception as exc:
            click.echo(f"  warning: could not load {p.name}: {exc}", err=True)
    return poems


def _parse_scorer(spec: str):
    """Resolve a score factory spec string into a ScoreFactory callable.

    Supported specs:
      novelty           novelty_scorer()
      novelty:0.5       novelty_scorer(0.8) — first positional float arg
      syllables:7       syllable_scorer(7)
      rhyme             rhyme_scorer()
      sentiment:0.6     sentiment_scorer(0.6)
      divergence        divergence_scorer()
      length:6          length_scorer(6)
    """
    from nouveau.generators import (
        novelty_scorer, syllable_scorer, rhyme_scorer, sentiment_scorer,
        divergence_scorer, length_scorer, alliteration_scorer, consonance_scorer,
    )
    _factories = {
        "novelty":      lambda a: novelty_scorer(float(a)) if a else novelty_scorer(),
        "syllables":    lambda a: syllable_scorer(int(a)),
        "rhyme":        lambda a: rhyme_scorer(int(a)) if a else rhyme_scorer(),
        "sentiment":    lambda a: sentiment_scorer(float(a)),
        "divergence":   lambda a: divergence_scorer(),
        "length":       lambda a: length_scorer(int(a)),
        "alliteration": lambda a: alliteration_scorer(),
        "consonance":   lambda a: consonance_scorer(float(a)) if a else consonance_scorer(),
    }
    name, _, arg = spec.partition(":")
    if name not in _factories:
        raise click.BadParameter(
            f"unknown scorer '{name}'. choices: {', '.join(sorted(_factories))}",
            param_hint="--score",
        )
    return _factories[name](arg or None)


@cli.group()
def corpus() -> None:
    """Inspect and filter a corpus of saved poems."""


@corpus.command("rank")
@click.argument("directory", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--score", "scorer_spec", default="novelty", show_default=True,
              help="Score factory spec (novelty, syllables:7, rhyme, sentiment:0.5, …).")
@click.option("--n", default=10, show_default=True, help="How many poems to show.")
@click.option("--worst", is_flag=True, default=False,
              help="Show highest-scoring poems first (most repetitive, most stuck, etc.).")
@click.option("--full", is_flag=True, default=False, help="Print full poem text.")
def corpus_rank(directory: Path, scorer_spec: str, n: int, worst: bool, full: bool) -> None:
    """Rank poems in a directory by a score factory.

    By default shows lowest-cost poems first (best fit for the scorer).
    Use --worst to surface the highest-cost poems (most looping, most divergent, etc.).
    """
    make_score = _parse_scorer(scorer_spec)
    poems = _load_corpus(directory)

    scored = []
    with click.progressbar(poems, label="scoring", width=30) as bar:
        for path, poem in bar:
            cost = score_poem(poem, make_score)
            scored.append((cost, path, poem))
    scored.sort(key=lambda x: x[0], reverse=worst)

    direction = "highest" if worst else "lowest"
    click.echo(f"\n--- {min(n, len(scored))} {direction}-scoring by {scorer_spec} ---\n")
    for rank, (cost, path, poem) in enumerate(scored[:n], 1):
        click.echo(f"{rank:>3}.  {cost:.3f}  {path.name}  [{poem.generator_name}]")
        if full:
            for line in poem.lines:
                click.echo(f"       {line.author:<8} {line.text}")
            click.echo()
        else:
            preview = poem.lines[0].text[:70] if poem.lines else ""
            click.echo(f"       {preview}")


@corpus.command("filter")
@click.argument("directory", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--score", "scorer_spec", default="novelty", show_default=True)
@click.option("--threshold", default=1.0, show_default=True,
              help="Score threshold for filtering.")
@click.option("--above", is_flag=True, default=False,
              help="Keep poems ABOVE the threshold (select for repetition, loops, etc.).")
@click.option("--out", "out_dir", default=None,
              help="Copy survivors to this directory instead of just listing them.")
def corpus_filter(directory: Path, scorer_spec: str, threshold: float, above: bool,
                  out_dir: str | None) -> None:
    """Filter a corpus by a score threshold.

    By default keeps poems whose mean cost is BELOW the threshold (novel, divergent).
    Use --above to keep poems ABOVE it (repetitive, looping, stuck).

    \b
    Examples:
      nouveau corpus filter corpora/dream --score novelty --threshold 5.0
      nouveau corpus filter corpora/dream --score novelty --threshold 10.0 --above
    """
    make_score = _parse_scorer(scorer_spec)
    poems = _load_corpus(directory)

    survivors = []
    with click.progressbar(poems, label="scoring", width=30) as bar:
        for path, poem in bar:
            cost = score_poem(poem, make_score)
            if (cost > threshold) if above else (cost < threshold):
                survivors.append((cost, path, poem))
    survivors.sort(key=lambda x: x[0], reverse=above)

    direction = "above" if above else "below"
    click.echo(f"\n{len(survivors)}/{len(poems)} poems {direction} threshold {threshold} [{scorer_spec}]\n")
    for cost, path, poem in survivors:
        click.echo(f"  {cost:.3f}  {path.name}")

    if out_dir and survivors:
        import shutil
        dest = Path(out_dir)
        dest.mkdir(parents=True, exist_ok=True)
        for _, path, _ in survivors:
            shutil.copy(path, dest / path.name)
        click.echo(f"\nCopied {len(survivors)} poems to {dest}/")


@corpus.command("show")
@click.argument("path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def corpus_show(path: Path) -> None:
    """Print a single poem with metadata."""
    poem = Poem.load(path)
    click.echo(f"generator: {poem.generator_name}  model: {poem.model_name}")
    click.echo(f"created:   {poem.created_at}  lines: {len(poem.lines)}")
    click.echo()
    for line in poem.lines:
        click.echo(f"  {line.author:<10} {line.text}")


@cli.command("list")
def list_poems() -> None:
    """List saved poems."""
    if not POEM_DIR.exists():
        click.echo("No poems found in ./poems/")
        return
    files = sorted(POEM_DIR.glob("*.json"))
    if not files:
        click.echo("No poems found in ./poems/")
        return
    for f in files:
        try:
            data = json.loads(f.read_text())
            n = len(data.get("lines", []))
            gen = data.get("generator", "?")
            click.echo(f"{f}  {n} lines  {gen}")
        except Exception:
            click.echo(f"{f}  (unreadable)")
