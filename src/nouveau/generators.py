"""
Generator strategies for poem line generation.

Two-layer architecture:
  ContextFn   = (Poem) -> str
                Extracts a prompt from the poem. Pure selection, no model call.
                Operates at any granularity: line, word, token.

  GeneratorFn = (Poem, Model) -> str
                Calls the model with a context prompt and returns a new line.

make_generator(context_fn) bridges the two layers.
make_conditional(condition, if_true, if_false) composes generators.

Every generator in GENERATORS is a GeneratorFn. Strategies are model-agnostic;
names reflect selection logic, not the backend.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Union

if TYPE_CHECKING:
    from nouveau.model import Model
    from nouveau.poem import Poem

ContextFn = Callable[["Poem"], str]
GeneratorFn = Callable[["Poem", "Model"], str]
LineSelector = Union[int, list[int], slice]


# ---------------------------------------------------------------------------
# Core combinators
# ---------------------------------------------------------------------------

def make_generator(context_fn: ContextFn) -> GeneratorFn:
    """Wrap a context selector into a full generator callable."""
    def _generator(poem: "Poem", model: "Model") -> str:
        return model.generate(context_fn(poem))
    return _generator


def make_conditional(
    condition: Callable[["Poem"], bool],
    if_true: GeneratorFn,
    if_false: GeneratorFn,
) -> GeneratorFn:
    """Return a generator that dispatches to if_true or if_false based on poem state."""
    def _generator(poem: "Poem", model: "Model") -> str:
        return (if_true if condition(poem) else if_false)(poem, model)
    return _generator


# ---------------------------------------------------------------------------
# Line-level context selectors
# ---------------------------------------------------------------------------

def last_lines(n: int = 1) -> ContextFn:
    """Select the last n lines of the poem."""
    return lambda poem: "\n".join(line.text for line in poem.lines[-n:])


def first_lines(n: int = 1) -> ContextFn:
    """Select the first n lines of the poem."""
    return lambda poem: "\n".join(line.text for line in poem.lines[:n])


def _resolve_line_selector(poem: "Poem", selector: LineSelector) -> list:
    if isinstance(selector, int):
        return poem.lines[-selector:]
    if isinstance(selector, slice):
        return poem.lines[selector]
    n = len(poem.lines)
    return [poem.lines[i] for i in selector if -n <= i < n]


def line_window(selector: LineSelector = 3) -> ContextFn:
    """Select lines by int (last N), list of indices, or slice."""
    def _selector(poem: "Poem") -> str:
        lines = _resolve_line_selector(poem, selector)
        return "\n".join(line.text for line in lines)
    return _selector


# ---------------------------------------------------------------------------
# Word-level context selectors
# ---------------------------------------------------------------------------

def last_words(n: int) -> ContextFn:
    """Select the last n words across all lines."""
    def _selector(poem: "Poem") -> str:
        words = " ".join(line.text for line in poem.lines).split()
        return " ".join(words[-n:])
    return _selector


def first_words(n: int) -> ContextFn:
    """Select the first n words across all lines."""
    def _selector(poem: "Poem") -> str:
        words = " ".join(line.text for line in poem.lines).split()
        return " ".join(words[:n])
    return _selector


# ---------------------------------------------------------------------------
# Named generator instances
# ---------------------------------------------------------------------------

last        = make_generator(last_lines(1))
first       = make_generator(first_lines(1))
window      = make_generator(line_window(3))
bookend     = make_generator(line_window([0, -1]))
alternating = make_generator(line_window(slice(None, None, 2)))

# closure: on the penultimate turn, reach back to the opening line to close the poem
closure = make_conditional(
    condition=lambda poem: len(poem) == poem.max_lines - 1,
    if_true=first,
    if_false=last,
)


# ---------------------------------------------------------------------------
# Future generators (stubs — not yet registered)
# ---------------------------------------------------------------------------
# Architecture notes for generators requiring model.py extensions:
#
# rhyme:
#   context_fn: last_lines(1) or last_words(n)
#   Extra: model.generate() needs `bias_toward` accepting rhyme candidates.
#   Compute candidates via pronouncing/cmudict from the previous AI line,
#   pass as logit biases or a constrained prompt suffix.
#
# syllable:
#   context_fn: any selector
#   Extra: post-process token stream to hit a target syllable count (pyphen/cmudict),
#   or wrap model.generate() in a rejection-sampler loop.
#
# sentiment_arc:
#   context_fn: any selector
#   Extra: `sentiment_target: float` on model.generate(). Score candidates with
#   vader/textblob; re-sample until within tolerance or rank beam candidates.
#   Natural arc: neutral → tension → quiet resolution.


GENERATORS: dict[str, GeneratorFn] = {
    "last":        last,
    "first":       first,
    "window":      window,
    "bookend":     bookend,
    "alternating": alternating,
    "closure":     closure,
}
