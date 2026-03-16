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


def lipogram(context_fn: ContextFn, banned: str = "e") -> ContextFn:
    """Oulipo lipogram: remove all words containing banned letters.

    The default bans 'e' (the most common English letter), following
    Perec's 'La Disparition'. The surviving words form a skeletal text
    that the model completes.
    """
    banned_set = frozenset(banned.lower())

    def _selector(poem: "Poem") -> str:
        text = context_fn(poem)
        words = text.split()
        kept = [w for w in words if not (set(w.lower()) & banned_set)]
        return " ".join(kept) if kept else text
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


def reverse_lines(context_fn: ContextFn) -> ContextFn:
    """Time runs the other way: feed lines in reverse order.

    The model is always trying to arrive somewhere it has already been.
    Works best with window or line_window; single-line contexts return unchanged.
    """
    def _selector(poem: "Poem") -> str:
        text = context_fn(poem)
        lines = text.split("\n")
        return "\n".join(reversed(lines))
    return _selector


def column(context_fn: ContextFn, position: int = -1) -> ContextFn:
    """Extract one word per line — the vertical spine of the poem.

    position=0: first word of each line (how things begin)
    position=-1: last word of each line (how things end)

    The model receives a list of isolated words with no surrounding
    syntax — pure semantic residue, one step removed from language.
    """
    def _selector(poem: "Poem") -> str:
        text = context_fn(poem)
        spine = []
        for line in text.split("\n"):
            words = line.split()
            if words:
                spine.append(words[position])
        return " ".join(spine)
    return _selector


def spiral(context_fn: ContextFn, repeats: int = 3) -> ContextFn:
    """A thought that repeats and dissolves on each repetition.

    The first echo is intact; each subsequent one loses more words,
    separated by ' / ' as a breath mark. The model enters through
    a clear door and exits through fog.
    """
    def _selector(poem: "Poem") -> str:
        text = context_fn(poem)
        words = text.split()
        if not words:
            return text
        parts = []
        for i in range(repeats):
            keep = max(1, int(len(words) * (1.0 - i / repeats)))
            parts.append(" ".join(words[:keep]))
        return " / ".join(parts)
    return _selector


def hypnagogic(base_fn: ContextFn, min_keep: float = 0.1) -> ContextFn:
    """Erasure that deepens as the poem grows — fog accumulates.

    Early in the poem the context is nearly intact. By the final lines,
    only traces remain. The poem forgets itself as it continues;
    the model responds to a memory of a memory.
    """
    def _selector(poem: "Poem") -> str:
        progress = len(poem.lines) / max(poem.max_lines, 1)
        keep_ratio = max(min_keep, 1.0 - progress * 0.9)
        return erasure(base_fn, keep_ratio=keep_ratio)(poem)
    return _selector


def palimpsest(context_fn: ContextFn, layers: int = 3,
               seed: int | None = None) -> ContextFn:
    """Stack erased versions of the context on top of each other.

    Each layer is the same source text, erased to a different density.
    The model sees the same text through multiple transparencies at once —
    old writing showing through new, the way parchment holds its ghosts.
    """
    def _selector(poem: "Poem") -> str:
        text = context_fn(poem)
        words = text.split()
        if not words:
            return text
        rng = random.Random(seed if seed is not None else len(poem))
        layer_texts = []
        for i in range(layers):
            keep_ratio = (i + 1) / layers
            kept = [w if rng.random() < keep_ratio else "..." for w in words]
            # collapse gaps
            result = []
            for w in kept:
                if w == "..." and result and result[-1] == "...":
                    continue
                result.append(w)
            layer_texts.append(" ".join(result))
        return "\n".join(layer_texts)
    return _selector


def markov_chain(context_fn: ContextFn, order: int = 2,
                 seed: int | None = None) -> ContextFn:
    """Build a Markov chain from the poem text and generate from it.

    The chain is built from the context each turn, so it evolves as
    the poem grows. Short poems produce short chains; the output
    gets more interesting as material accumulates.

    Uses backoff: if the current state has no successors, drops to
    a shorter state (order-1, then order-0) before giving up.
    """
    def _selector(poem: "Poem") -> str:
        text = context_fn(poem)
        words = text.split()
        if len(words) <= order:
            return text

        # build transition tables at all orders for backoff
        tables: list[dict[tuple[str, ...], list[str]]] = []
        for o in range(1, order + 1):
            t: dict[tuple[str, ...], list[str]] = defaultdict(list)
            for i in range(len(words) - o):
                key = tuple(words[i:i + o])
                t[key].append(words[i + o])
            tables.append(t)

        if not tables[-1]:
            return text

        rng = random.Random(seed if seed is not None else len(poem))
        # start from a random key at the highest order
        keys = list(tables[-1].keys())
        state = rng.choice(keys)
        result = list(state)
        for _ in range(15):  # generate up to 15 words
            # try highest order first, back off to lower orders
            next_word = None
            for o in range(order, 0, -1):
                backoff_state = tuple(result[-o:])
                choices = tables[o - 1].get(backoff_state)
                if choices:
                    next_word = rng.choice(choices)
                    break
            if next_word is None:
                break
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


def _jaccard(a: set, b: set) -> float:
    """Jaccard similarity between two sets. Returns 0.0 if both empty."""
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def divergence_scorer(min_sim: float = 0.1, max_sim: float = 0.6) -> ScoreFactory:
    """Cost penalizes output that is too similar or too dissimilar to the last line.

    Uses Jaccard similarity on word sets. Output in the band
    [min_sim, max_sim] scores 0; outside it, cost increases linearly.
    This implements "related divergence" — output that is inspired by
    but different from the input.
    """
    def make_score(poem: "Poem") -> Callable[[str], float]:
        if not poem.lines:
            return lambda text: 0.0
        ref_words = set(poem[-1].lower().split())

        def score(text: str) -> float:
            candidate_words = set(text.lower().split())
            sim = _jaccard(ref_words, candidate_words)
            if sim < min_sim:
                return min_sim - sim
            if sim > max_sim:
                return sim - max_sim
            return 0.0

        return score
    return make_score


def lipogram_scorer(banned: str = "e", weight_per_char: float = 0.5) -> ScoreFactory:
    """Soft lipogram: penalize (don't remove) words containing banned letters.

    Cost = number of banned-letter occurrences in the text * weight_per_char.
    A hard lipogram deletes; a soft lipogram nudges. The model can still
    use banned letters if nothing better is available, but prefers to avoid them.
    """
    banned_set = frozenset(banned.lower())

    def make_score(poem: "Poem") -> Callable[[str], float]:
        def score(text: str) -> float:
            return sum(1 for c in text.lower() if c in banned_set) * weight_per_char
        return score
    return make_score


def novelty_scorer(decay: float = 0.8) -> ScoreFactory:
    """Penalize words that have already appeared in the poem.

    Each word's penalty decays exponentially by how many lines ago it
    appeared — recent repetition costs more than distant echoes.
    Cost = sum of decay^(distance) for each repeated word.
    """
    def make_score(poem: "Poem") -> Callable[[str], float]:
        if not poem.lines:
            return lambda text: 0.0
        # build word->most-recent-distance map (1 = last line, 2 = line before, etc.)
        word_dist: dict[str, int] = {}
        for i, line in enumerate(reversed(poem.lines)):
            for w in line.text.lower().split():
                if w not in word_dist:
                    word_dist[w] = i + 1

        def score(text: str) -> float:
            cost = 0.0
            for w in text.lower().split():
                if w in word_dist:
                    cost += decay ** word_dist[w]
            return cost

        return score
    return make_score


def length_scorer(target_words: int) -> ScoreFactory:
    """Cost = absolute distance from target word count."""
    return lambda poem: lambda text: abs(len(text.split()) - target_words)


def alliteration_scorer() -> ScoreFactory:
    """Reward (negative cost) for alliterative patterns.

    Cost = -count of adjacent word pairs sharing a first letter.
    Lower is better, so more alliteration = lower cost.
    """
    def make_score(poem: "Poem") -> Callable[[str], float]:
        def score(text: str) -> float:
            words = [w.lower().strip(".,!?;:'\"") for w in text.split() if w]
            if len(words) < 2:
                return 0.0
            pairs = sum(
                1 for a, b in zip(words, words[1:])
                if a and b and a[0] == b[0]
            )
            return -pairs  # negative = reward
        return score
    return make_score


def consonance_scorer(target_density: float = 0.6) -> ScoreFactory:
    """Penalize distance from a target consonant density.

    Consonant density = fraction of alphabetic characters that are consonants.
    High density (0.7+) creates percussive, clipped text.
    Low density (0.4-) creates open, vowel-heavy flow.
    """
    _vowels = frozenset("aeiouy")

    def make_score(poem: "Poem") -> Callable[[str], float]:
        def score(text: str) -> float:
            alpha = [c for c in text.lower() if c.isalpha()]
            if not alpha:
                return 0.0
            consonants = sum(1 for c in alpha if c not in _vowels)
            density = consonants / len(alpha)
            return abs(density - target_density)
        return score
    return make_score


def vocabulary_scorer(corpus_fn: ContextFn) -> ScoreFactory:
    """Reward candidates that introduce words not yet in the poem's vocabulary.

    Score = fraction of candidate words that already appear in the corpus
    context. Lower = more novel vocabulary. Uses the corpus_fn to define
    what counts as "already said."
    """
    def make_score(poem: "Poem") -> Callable[[str], float]:
        existing = set(corpus_fn(poem).lower().split())
        if not existing:
            return lambda text: 0.0

        def score(text: str) -> float:
            words = text.lower().split()
            if not words:
                return 0.0
            repeated = sum(1 for w in words if w in existing)
            return repeated / len(words)

        return score
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
dissolve  = make_generator(erasure(n_plus_7(line_window(3)), keep_ratio=0.5))
vanish    = make_generator(lipogram(last_lines(1), banned="e"))
drift     = make_constrained_generator(last_lines(1), divergence_scorer())

# soft-constrained aesthetic personalities — composed scorers
sparse = make_constrained_generator(
    last_lines(1),
    combine_scores(length_scorer(4), novelty_scorer(), consonance_scorer(0.4)),
)
dense = make_constrained_generator(
    line_window(3),
    combine_scores(length_scorer(8), alliteration_scorer(), consonance_scorer(0.7)),
)
echo = make_constrained_generator(
    last_lines(1),
    combine_scores(
        divergence_scorer(0.2, 0.5), rhyme_scorer(), novelty_scorer(decay=0.5),
        weights=[2.0, 3.0, 1.0],
    ),
)
ghost = make_constrained_generator(
    erasure(line_window(3), keep_ratio=0.4),
    combine_scores(
        lipogram_scorer("e"), novelty_scorer(), length_scorer(5),
        weights=[1.0, 0.5, 1.0],
    ),
)
strange = make_constrained_generator(
    n_plus_7(last_lines(1)),
    combine_scores(
        divergence_scorer(0.05, 0.4),
        alliteration_scorer(),
        vocabulary_scorer(line_window(5)),
        weights=[3.0, 1.0, 2.0],
    ),
)

# dream generators — temporal distortion, self-dissolution, recursive echo

# time runs backward — the model always approaches where it's already been
reverie  = make_generator(reverse_lines(line_window(3)))

# only the last word of each line — endpoints become the whole sentence
spine    = make_generator(column(line_window(5), position=-1))

# a phrase that repeats and loses words like a thought at the edge of sleep
trance   = make_generator(spiral(last_lines(1), repeats=4))

# fog accumulates — context dissolves as the poem grows longer
fugue    = make_generator(hypnagogic(line_window(5)))

# the same text seen through multiple transparencies at once
palimps  = make_generator(palimpsest(line_window(3), layers=3))

# spiral context + novelty pressure: the model escapes the loop or repeats forever
lucid    = make_constrained_generator(
    spiral(last_lines(1), repeats=3),
    combine_scores(novelty_scorer(decay=0.9), length_scorer(6)),
)


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
    "vanish":      vanish,
    "drift":       drift,
    "sparse":      sparse,
    "dense":       dense,
    "echo":        echo,
    "ghost":       ghost,
    "strange":     strange,
    "reverie":     reverie,
    "spine":       spine,
    "trance":      trance,
    "fugue":       fugue,
    "palimps":     palimps,
    "lucid":       lucid,
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
    "lipogram": lambda arg: make_generator(
        lipogram(last_lines(1), banned=arg),
    ),
}
