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

import random
import re
from collections import defaultdict
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
# Context transformers — ContextFn -> ContextFn
#
# These compose with any context selector to transform text before the model
# sees it. Inspired by Oulipo constraints, Burroughs cut-up, and erasure poetry.
# ---------------------------------------------------------------------------

def cut_up(context_fn: ContextFn, seed: int | None = None) -> ContextFn:
    """Burroughs cut-up: shuffle words from the context into new order.

    The model receives a rearranged version of the selected text,
    producing output that diverges from the original while staying
    in the same semantic field.
    """
    def _selector(poem: "Poem") -> str:
        text = context_fn(poem)
        words = text.split()
        rng = random.Random(seed if seed is not None else len(poem))
        rng.shuffle(words)
        return " ".join(words)
    return _selector


def fold_in(fn_a: ContextFn, fn_b: ContextFn) -> ContextFn:
    """Burroughs fold-in: interleave words from two context sources.

    Takes alternating words from each source, producing a composite
    text that carries traces of both originals.
    """
    def _selector(poem: "Poem") -> str:
        words_a = fn_a(poem).split()
        words_b = fn_b(poem).split()
        result = []
        for i in range(max(len(words_a), len(words_b))):
            if i < len(words_a):
                result.append(words_a[i])
            if i < len(words_b):
                result.append(words_b[i])
        return " ".join(result)
    return _selector


def erasure(context_fn: ContextFn, keep_ratio: float = 0.5,
            seed: int | None = None) -> ContextFn:
    """Erasure poetry: randomly remove words, leaving gaps.

    keep_ratio controls density — 0.3 is sparse, 0.7 is dense.
    Gaps are marked with '...' to preserve the sense of absence.
    """
    def _selector(poem: "Poem") -> str:
        text = context_fn(poem)
        words = text.split()
        if not words:
            return text
        rng = random.Random(seed if seed is not None else len(poem))
        kept = [w if rng.random() < keep_ratio else "..." for w in words]
        # collapse consecutive gaps
        result = []
        for w in kept:
            if w == "..." and result and result[-1] == "...":
                continue
            result.append(w)
        return " ".join(result)
    return _selector


def n_plus_7(context_fn: ContextFn, wordlist: list[str] | None = None,
             offset: int = 7) -> ContextFn:
    """Oulipo N+7: replace each noun-like word with the word N positions
    later in a dictionary.

    Without NLTK or a POS tagger, we use a heuristic: replace words
    longer than 3 characters that aren't common function words.
    A custom wordlist can be provided; otherwise uses a built-in
    small poetic vocabulary.
    """
    _function_words = frozenset({
        "the", "a", "an", "and", "but", "or", "nor", "for", "yet", "so",
        "in", "on", "at", "to", "by", "of", "with", "from", "into", "onto",
        "is", "are", "was", "were", "be", "been", "being", "am",
        "has", "have", "had", "do", "does", "did", "will", "would",
        "shall", "should", "may", "might", "can", "could", "must",
        "not", "no", "if", "then", "than", "that", "this", "these", "those",
        "it", "its", "he", "she", "they", "we", "you", "me", "him", "her",
        "us", "them", "my", "your", "his", "our", "their",
        "who", "whom", "whose", "which", "what", "where", "when", "how", "why",
        "all", "each", "every", "both", "few", "more", "most", "some", "any",
        "very", "too", "also", "just", "only", "still", "never", "always",
    })
    words_db = wordlist if wordlist is not None else _default_wordlist()

    def _selector(poem: "Poem") -> str:
        text = context_fn(poem)
        result = []
        for word in text.split():
            clean = word.lower().strip(".,!?;:'\"")
            if len(clean) > 3 and clean not in _function_words and words_db:
                replacement = _lookup_offset(clean, words_db, offset)
                result.append(replacement)
            else:
                result.append(word)
        return " ".join(result)
    return _selector


def _default_wordlist() -> list[str]:
    """A small curated wordlist for N+7 — poetic nouns and adjectives."""
    return sorted([
        "river", "stone", "light", "shadow", "water", "fire", "wind", "rain",
        "bone", "glass", "silver", "golden", "iron", "copper", "dust", "ash",
        "moon", "star", "sun", "cloud", "storm", "thunder", "lightning", "snow",
        "rose", "thorn", "seed", "root", "branch", "leaf", "flower", "bloom",
        "bird", "crow", "swan", "moth", "wolf", "deer", "horse", "bear",
        "hand", "eye", "mouth", "heart", "blood", "breath", "voice", "song",
        "door", "window", "wall", "floor", "roof", "bridge", "road", "path",
        "night", "dawn", "dusk", "morning", "evening", "hour", "moment", "year",
        "dream", "sleep", "wake", "silence", "echo", "ghost", "memory", "name",
        "salt", "honey", "milk", "wine", "bread", "smoke", "flame", "ember",
        "ocean", "tide", "wave", "shore", "island", "mountain", "valley", "field",
        "thread", "needle", "cloth", "silk", "wool", "string", "knot", "ribbon",
        "bell", "drum", "flute", "harp", "choir", "hymn", "prayer", "psalm",
        "knife", "sword", "arrow", "shield", "crown", "throne", "tower", "ruin",
    ])


def _lookup_offset(word: str, wordlist: list[str], offset: int) -> str:
    """Find word's position in sorted wordlist and return the word N places later."""
    import bisect
    idx = bisect.bisect_left(wordlist, word)
    return wordlist[(idx + offset) % len(wordlist)]


def markov_chain(context_fn: ContextFn, order: int = 2,
                 seed: int | None = None) -> ContextFn:
    """Build a Markov chain from the poem text and generate from it.

    The chain is built from the context each turn, so it evolves as
    the poem grows. Short poems produce short chains; the output
    gets more interesting as material accumulates.
    """
    def _selector(poem: "Poem") -> str:
        text = context_fn(poem)
        words = text.split()
        if len(words) <= order:
            return text

        # build transition table
        transitions: dict[tuple[str, ...], list[str]] = defaultdict(list)
        for i in range(len(words) - order):
            key = tuple(words[i:i + order])
            transitions[key].append(words[i + order])

        if not transitions:
            return text

        rng = random.Random(seed if seed is not None else len(poem))
        # start from a random key
        keys = list(transitions.keys())
        state = rng.choice(keys)
        result = list(state)
        for _ in range(15):  # generate up to 15 words
            choices = transitions.get(state)
            if not choices:
                break
            next_word = rng.choice(choices)
            result.append(next_word)
            state = tuple(result[-order:])

        return " ".join(result)
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


def combine_scores(
    *factories: ScoreFactory,
    weights: list[float] | None = None,
) -> ScoreFactory:
    """Combine multiple score factories by weighted sum (all costs lower = better).

    Example:
        make_constrained_generator(
            last_lines(1),
            combine_scores(syllable_scorer(7), rhyme_scorer()),
        )
    """
    def make_score(poem: "Poem") -> Callable[[str], float]:
        scorers = [f(poem) for f in factories]
        ws = weights if weights is not None else [1.0] * len(factories)
        return lambda text: sum(s(text) * w for s, w in zip(scorers, ws))
    return make_score


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

# constrained named defaults — use GENERATOR_FACTORIES for parameterized variants
rhyme   = make_constrained_generator(last_lines(1), rhyme_scorer())
hopeful = make_constrained_generator(last_lines(1), sentiment_scorer(0.6))
somber  = make_constrained_generator(last_lines(1), sentiment_scorer(-0.6))

# context-transformed generators — experimental / Oulipo-inspired
cutup    = make_generator(cut_up(line_window(3)))
erased   = make_generator(erasure(last_lines(1), keep_ratio=0.4))
folded   = make_generator(fold_in(first_lines(1), last_lines(1)))
markov   = make_generator(markov_chain(line_window(5), order=1))
oulipo   = make_generator(n_plus_7(last_lines(1)))
dissolve = make_generator(erasure(n_plus_7(line_window(3)), keep_ratio=0.5))


# Zero-arg generators registered for the CLI.
GENERATORS: dict[str, GeneratorFn] = {
    "last":        last,
    "first":       first,
    "window":      window,
    "bookend":     bookend,
    "alternating": alternating,
    "closure":     closure,
    "rhyme":       rhyme,
    "hopeful":     hopeful,
    "somber":      somber,
    "cutup":       cutup,
    "erased":      erased,
    "folded":      folded,
    "markov":      markov,
    "oulipo":      oulipo,
    "dissolve":    dissolve,
}

# Parameterized factories: CLI calls these as  name:arg  (e.g. syllables:5).
# Each value is a callable (str) -> GeneratorFn that parses its own argument.
GENERATOR_FACTORIES: dict[str, Callable[[str], GeneratorFn]] = {
    "syllables": lambda arg: make_constrained_generator(
        last_lines(1),
        syllable_scorer(int(arg)),
        max_new_tokens=max(8, int(arg) * 2),
    ),
    "rhyme": lambda arg: make_constrained_generator(
        last_lines(1),
        rhyme_scorer(int(arg)),
    ),
    "sentiment": lambda arg: make_constrained_generator(
        last_lines(1),
        sentiment_scorer(float(arg)),
    ),
    "erasure": lambda arg: make_generator(
        erasure(last_lines(1), keep_ratio=float(arg)),
    ),
    "nplus": lambda arg: make_generator(
        n_plus_7(last_lines(1), offset=int(arg)),
    ),
    "markov": lambda arg: make_generator(
        markov_chain(line_window(5), order=int(arg)),
    ),
}
