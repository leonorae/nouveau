import json
from pathlib import Path

import click

from nouveau.generators import GENERATOR_FACTORIES, GENERATORS
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
