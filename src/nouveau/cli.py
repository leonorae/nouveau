import json
from pathlib import Path

import click

from nouveau.generators import GENERATORS
from nouveau.model import DEFAULT_MODEL, Model
from nouveau.poem import POEM_DIR, Poem


@click.group()
def cli() -> None:
    """nouveau — human-AI collaborative poetry."""


@cli.command()
@click.argument("max_lines", type=int)
@click.argument("generator", type=click.Choice(list(GENERATORS)))
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
    GENERATOR is the strategy used for AI line generation.
    """
    if max_lines < 2:
        raise click.BadParameter("must be at least 2", param_hint="MAX_LINES")

    generator_fn = GENERATORS[generator]

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
@click.argument("path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def show(path: Path) -> None:
    """Display a saved poem."""
    data = json.loads(path.read_text())
    click.echo(f"{data['generator']} · {data['model']} · {data['created_at']}")
    click.echo()
    for entry in data["lines"]:
        label = "you" if entry["author"] == "human" else " AI"
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
