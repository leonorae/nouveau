"""
Generator strategies for poem line generation.

Two-layer architecture:
  ContextFn   = (Poem) -> str        # what text to extract from the poem
  GeneratorFn = (Poem, Model) -> str # model call that produces a new line

Combinators:
  make_generator(context_fn)                          # basic bridge
  make_conditional(condition, if_true, if_false)      # state-dependent dispatch
  make_constrained_generator(context_fn, make_score)  # rejection-sample by cost

Score factories produce a (Poem) -> (str) -> float cost function that
make_constrained_generator uses to pick the best of n_candidates outputs.

Every generator in GENERATORS is a GeneratorFn. Strategies are model-agnostic;
names reflect selection and constraint logic, not the backend.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING, Callable, Union

if TYPE_CHECKING:
    from nouveau.model import Model
    from nouveau.poem import Poem

ContextFn = Callable[["Poem"], str]
GeneratorFn = Callable[["Poem", "Model"], str]
ScoreFactory = Callable[["Poem"], Callable[[str], float]]
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


def make_constrained_generator(
    context_fn: ContextFn,
    make_score: ScoreFactory,
    n_candidates: int = 8,
    max_new_tokens: int = 20,
) -> GeneratorFn:
    """Generate n_candidates outputs and return the one with the lowest score.

    make_score(poem) is called once per turn and returns a cost function
    (str) -> float over candidate strings (lower = better fit).

    max_new_tokens is forwarded to model.generate() — reduce it for
    length-constrained generators (e.g. haiku lines).
    """
    def _generator(poem: "Poem", model: "Model") -> str:
        context = context_fn(poem)
        score = make_score(poem)
        candidates = [model.generate(context, max_new_tokens=max_new_tokens)
                      for _ in range(n_candidates)]
        return min(candidates, key=score)
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
# Utilities
# ---------------------------------------------------------------------------

def _end_sound(word: str, n: int = 3) -> str:
    """Return the last n characters of a word (stripped of punctuation).

    Used as a coarse phonetic key for rhyme matching. Replace with
    pronouncing/cmudict for proper phoneme-level accuracy.
    """
    word = word.lower().rstrip(".,!?;:'\"")
    return word[-n:] if len(word) >= n else word


def count_syllables(text: str) -> int:
    """Approximate syllable count by counting vowel-sound clusters.

    Rough heuristic (silent e, diphthongs, etc. are ignored). pyphen or
    cmudict give more accurate results when precision matters.
    """
    count = len(re.findall(r"[aeiouy]+", text.lower()))
    return max(1, count)


_sentiment_analyzer = None


def _get_sentiment_analyzer():
    """Lazily instantiate the VADER sentiment analyzer (cached singleton)."""
    global _sentiment_analyzer
    if _sentiment_analyzer is None:
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            _sentiment_analyzer = SentimentIntensityAnalyzer()
        except ImportError as exc:
            raise ImportError(
                "sentiment generators require vaderSentiment: uv add vaderSentiment"
            ) from exc
    return _sentiment_analyzer


# ---------------------------------------------------------------------------
# Score factories
# ---------------------------------------------------------------------------

def syllable_scorer(target: int) -> ScoreFactory:
    """Cost = absolute distance from target syllable count."""
    return lambda poem: lambda text: abs(count_syllables(text) - target)


def rhyme_scorer(n_chars: int = 3) -> ScoreFactory:
    """Cost = 0.0 if last word rhymes with last word of previous line, else 1.0.

    'Rhyme' is approximated by matching the last n_chars of each word.
    """
    def make_score(poem: "Poem") -> Callable[[str], float]:
        if not poem.lines:
            return lambda text: 0.0
        ref = _end_sound(poem[-1].split()[-1], n_chars)

        def score(text: str) -> float:
            words = text.strip().split()
            if not words:
                return 1.0
            return 0.0 if _end_sound(words[-1], n_chars) == ref else 1.0

        return score
    return make_score


def sentiment_scorer(target: float) -> ScoreFactory:
    """Cost = absolute distance from target VADER compound sentiment.

    target range: -1.0 (most negative) to 1.0 (most positive), 0.0 = neutral.
    """
    def make_score(poem: "Poem") -> Callable[[str], float]:
        analyzer = _get_sentiment_analyzer()
        return lambda text: abs(analyzer.polarity_scores(text)["compound"] - target)
    return make_score


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

# constrained: pick the best of n_candidates by the given cost function
# haiku lines are short — cap token budget so candidates are plausibly the right length
haiku_5 = make_constrained_generator(last_lines(1), syllable_scorer(5), max_new_tokens=8)
haiku_7 = make_constrained_generator(last_lines(1), syllable_scorer(7), max_new_tokens=12)
rhyme   = make_constrained_generator(last_lines(1), rhyme_scorer())
hopeful = make_constrained_generator(last_lines(1), sentiment_scorer(0.6))
somber  = make_constrained_generator(last_lines(1), sentiment_scorer(-0.6))


GENERATORS: dict[str, GeneratorFn] = {
    "last":        last,
    "first":       first,
    "window":      window,
    "bookend":     bookend,
    "alternating": alternating,
    "closure":     closure,
    "haiku_5":     haiku_5,
    "haiku_7":     haiku_7,
    "rhyme":       rhyme,
    "hopeful":     hopeful,
    "somber":      somber,
}
