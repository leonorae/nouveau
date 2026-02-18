"""
Generator strategies for poem line generation.

Each generator is a function with signature (poem: Poem, model: Model) -> str.
Register new generators in GENERATORS to make them available in the CLI.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from nouveau.model import Model
    from nouveau.poem import Poem

GeneratorFn = Callable[["Poem", "Model"], str]


def gpt_last(poem: "Poem", model: "Model") -> str:
    """Generate based on the most recent user line."""
    return model.generate(poem[-1])


def gpt_first(poem: "Poem", model: "Model") -> str:
    """Generate based on the very first line of the poem."""
    return model.generate(poem[0])


def gpt_closure(poem: "Poem", model: "Model") -> str:
    """Like gpt_last, but uses gpt_first for the final line to close the poem."""
    if len(poem) == poem.max_lines - 1:
        return gpt_first(poem, model)
    return gpt_last(poem, model)


# light on the floor of a room I have never been in
# someone's handwriting, slanted — nevertheless
# the last word arrives before its reason
# a door I know the sound of, not the house

GENERATORS: dict[str, GeneratorFn] = {
    "gpt_last": gpt_last,
    "gpt_first": gpt_first,
    "gpt_closure": gpt_closure,
}
