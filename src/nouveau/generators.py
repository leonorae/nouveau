"""
Generator strategies for poem line generation.

Each generator is a callable with signature (poem: Poem, model: Model) -> str.
Register new generators in GENERATORS to make them available in the CLI.

Design principle: prefer factory functions over plain functions wherever a
generator needs configuration. A factory accepts parameters and returns a
GeneratorFn — this keeps the GENERATORS dict uniform while allowing per-instance
tuning. Every generator added here should follow this pattern so that future
callers (CLI flags, config files, composition layers) can parameterise them
without touching generator internals.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Union

if TYPE_CHECKING:
    from nouveau.model import Model
    from nouveau.poem import Poem

GeneratorFn = Callable[["Poem", "Model"], str]
LineSelector = Union[int, list[int], slice]


# ---------------------------------------------------------------------------
# Primitive helpers
# ---------------------------------------------------------------------------

def _select_lines(poem: "Poem", selector: LineSelector) -> list:
    """Return a subset of poem lines according to selector.

    - int       → last N lines
    - slice     → poem.lines[selector]
    - list[int] → poem.lines[i] for each i (skips out-of-range indices)
    """
    if isinstance(selector, int):
        return poem.lines[-selector:]
    if isinstance(selector, slice):
        return poem.lines[selector]
    # list of indices
    n = len(poem.lines)
    return [poem.lines[i] for i in selector if -n <= i < n]


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def make_window_generator(selector: LineSelector = 3) -> GeneratorFn:
    """Return a generator that prompts the model with lines chosen by *selector*.

    selector:
      int        – last N lines (default 3, backward-compatible with gpt_window)
      list[int]  – specific indices, e.g. [0, -1] for first + last
      slice      – arbitrary slice, e.g. slice(None, None, 2) for every other line
    """
    def _generator(poem: "Poem", model: "Model") -> str:
        lines = _select_lines(poem, selector)
        context = "\n".join(line.text for line in lines)
        return model.generate(context)
    return _generator


# ---------------------------------------------------------------------------
# Named generator instances
# ---------------------------------------------------------------------------

def gpt_last(poem: "Poem", model: "Model") -> str:
    """Generate based on the most recent line."""
    return model.generate(poem[-1])


def gpt_first(poem: "Poem", model: "Model") -> str:
    """Generate based on the very first line of the poem."""
    return model.generate(poem[0])


def gpt_closure(poem: "Poem", model: "Model") -> str:
    """Like gpt_last, but uses gpt_first for the final line to close the poem."""
    if len(poem) == poem.max_lines - 1:
        return gpt_first(poem, model)
    return gpt_last(poem, model)


# last 3 lines — backward-compatible default
gpt_window = make_window_generator(3)

# first + last line — orients the closing line toward the opening image
gpt_bookend = make_window_generator([0, -1])

# every other line — biases toward a single voice (human or AI, depending on parity)
gpt_alternating = make_window_generator(slice(None, None, 2))


# ---------------------------------------------------------------------------
# Future generators (stubs — not yet registered)
# ---------------------------------------------------------------------------
# These require capabilities not yet in model.py. Architecture notes below.
#
# gpt_rhyme:
#   Architecture: model.generate() needs a `bias_toward` argument accepting
#   a list of candidate word suffixes. Pre-compute rhyme candidates from the
#   previous AI line using a phoneme dict (pronouncing / cmudict), then pass
#   them as soft logit biases or append them as a constrained prompt suffix.
#
# gpt_syllable:
#   Architecture: post-process the token stream, trimming or extending output
#   to hit a target syllable count. Needs a syllable counter (pyphen / cmudict).
#   Alternatively, wrap model.generate() in a rejection-sampler loop that
#   re-samples until the count is within tolerance.
#
# gpt_sentiment_arc:
#   Architecture: add a `sentiment_target: float` param to model.generate().
#   Score each candidate line with a lightweight model (vader / textblob),
#   re-sample until the score falls within tolerance, or use it as a ranking
#   signal over beam candidates. Natural arc: open neutral → build tension →
#   resolve quietly.


GENERATORS: dict[str, GeneratorFn] = {
    "gpt_last": gpt_last,
    "gpt_first": gpt_first,
    "gpt_closure": gpt_closure,
    "gpt_window": gpt_window,
    "gpt_bookend": gpt_bookend,
    "gpt_alternating": gpt_alternating,
}
