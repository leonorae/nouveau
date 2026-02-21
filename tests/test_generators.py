from nouveau.generators import (
    _end_sound,
    alternating,
    bookend,
    closure,
    combine_scores,
    count_syllables,
    first,
    first_lines,
    first_words,
    last,
    last_lines,
    last_words,
    line_window,
    make_conditional,
    make_constrained_generator,
    make_generator,
    rhyme_scorer,
    sentiment_scorer,
    syllable_scorer,
    window,
)
from nouveau.poem import Poem


def make_poem(*lines, max_lines=10):
    p = Poem(max_lines=max_lines, generator_name="last", model_name="fake")
    for i, text in enumerate(lines):
        p.add_line(text, author="human" if i % 2 == 0 else "ai")
    return p


# ---------------------------------------------------------------------------
# last / first (single-line selectors)
# ---------------------------------------------------------------------------

def test_last_uses_final_line(fake_model):
    poem = make_poem("first", "second", "third")
    last(poem, fake_model)
    assert fake_model.last_prefix == "third"


def test_first_uses_first_line(fake_model):
    poem = make_poem("first", "second", "third")
    first(poem, fake_model)
    assert fake_model.last_prefix == "first"


def test_first_returns_generated(fake_model):
    poem = make_poem("opening")
    result = first(poem, fake_model)
    assert result == "[opening]"


def test_last_returns_generated(fake_model):
    poem = make_poem("a", "b", "closing")
    result = last(poem, fake_model)
    assert result == "[closing]"


# ---------------------------------------------------------------------------
# closure (make_conditional)
# ---------------------------------------------------------------------------

def test_closure_uses_last_when_not_final(fake_model):
    # max_lines=6, poem has 4 lines — not the penultimate position yet
    poem = make_poem("a", "b", "c", "d", max_lines=6)
    closure(poem, fake_model)
    assert fake_model.last_prefix == "d"


def test_closure_uses_first_on_final_line(fake_model):
    # max_lines=6, poem has 5 lines — the next AI line closes the poem
    poem = make_poem("a", "b", "c", "d", "e", max_lines=6)
    closure(poem, fake_model)
    assert fake_model.last_prefix == "a"


def test_closure_two_line_poem(fake_model):
    # smallest possible: max_lines=2, 1 line in -> closure triggers immediately
    poem = make_poem("only", max_lines=2)
    closure(poem, fake_model)
    assert fake_model.last_prefix == "only"


# ---------------------------------------------------------------------------
# window (line_window int selector)
# ---------------------------------------------------------------------------

def test_window_uses_last_three_lines(fake_model):
    poem = make_poem("a", "b", "c", "d", "e")
    window(poem, fake_model)
    assert fake_model.last_prefix == "c\nd\ne"


def test_window_fewer_lines_than_window(fake_model):
    poem = make_poem("a", "b")
    window(poem, fake_model)
    assert fake_model.last_prefix == "a\nb"


def test_window_single_line(fake_model):
    poem = make_poem("alone")
    window(poem, fake_model)
    assert fake_model.last_prefix == "alone"


def test_window_returns_generated(fake_model):
    poem = make_poem("x", "y", "z")
    result = window(poem, fake_model)
    assert result == "[x\ny\nz]"


# ---------------------------------------------------------------------------
# bookend (line_window list selector)
# ---------------------------------------------------------------------------

def test_bookend_uses_first_and_last(fake_model):
    poem = make_poem("alpha", "b", "c", "omega")
    bookend(poem, fake_model)
    assert fake_model.last_prefix == "alpha\nomega"


def test_bookend_single_line(fake_model):
    # with one line, index 0 and -1 are the same; deduplication is not required
    poem = make_poem("solo")
    bookend(poem, fake_model)
    assert fake_model.last_prefix == "solo\nsolo"


# ---------------------------------------------------------------------------
# alternating (line_window slice selector)
# ---------------------------------------------------------------------------

def test_alternating_even_indexed_lines(fake_model):
    poem = make_poem("a", "b", "c", "d", "e")
    alternating(poem, fake_model)
    assert fake_model.last_prefix == "a\nc\ne"


def test_alternating_two_lines(fake_model):
    poem = make_poem("a", "b")
    alternating(poem, fake_model)
    assert fake_model.last_prefix == "a"


# ---------------------------------------------------------------------------
# line_window factory
# ---------------------------------------------------------------------------

def test_line_window_int(fake_model):
    gen = make_generator(line_window(2))
    poem = make_poem("a", "b", "c", "d")
    gen(poem, fake_model)
    assert fake_model.last_prefix == "c\nd"


def test_line_window_list(fake_model):
    gen = make_generator(line_window([0, 2]))
    poem = make_poem("a", "b", "c")
    gen(poem, fake_model)
    assert fake_model.last_prefix == "a\nc"


def test_line_window_slice(fake_model):
    gen = make_generator(line_window(slice(1, None)))
    poem = make_poem("a", "b", "c")
    gen(poem, fake_model)
    assert fake_model.last_prefix == "b\nc"


def test_line_window_out_of_range_indices_skipped(fake_model):
    gen = make_generator(line_window([0, 5]))
    poem = make_poem("a", "b", "c")
    gen(poem, fake_model)
    assert fake_model.last_prefix == "a"


# ---------------------------------------------------------------------------
# Word-level selectors
# ---------------------------------------------------------------------------

def test_last_words_spans_lines(fake_model):
    gen = make_generator(last_words(4))
    poem = make_poem("the rain falls", "soft on the ground")
    gen(poem, fake_model)
    assert fake_model.last_prefix == "soft on the ground"


def test_first_words_spans_lines(fake_model):
    gen = make_generator(first_words(3))
    poem = make_poem("the rain falls", "on the ground")
    gen(poem, fake_model)
    assert fake_model.last_prefix == "the rain falls"


def test_last_words_fewer_than_n(fake_model):
    gen = make_generator(last_words(10))
    poem = make_poem("just three words")
    gen(poem, fake_model)
    assert fake_model.last_prefix == "just three words"


# ---------------------------------------------------------------------------
# make_conditional
# ---------------------------------------------------------------------------

def test_make_conditional_dispatches_true(fake_model):
    gen = make_conditional(lambda p: True, first, last)
    poem = make_poem("a", "b", "c")
    gen(poem, fake_model)
    assert fake_model.last_prefix == "a"  # first was used


def test_make_conditional_dispatches_false(fake_model):
    gen = make_conditional(lambda p: False, first, last)
    poem = make_poem("a", "b", "c")
    gen(poem, fake_model)
    assert fake_model.last_prefix == "c"  # last was used


# ---------------------------------------------------------------------------
# Utilities: count_syllables, _end_sound
# ---------------------------------------------------------------------------

def test_count_syllables_basic():
    assert count_syllables("rain") == 1
    assert count_syllables("water") == 2
    assert count_syllables("beautiful") == 3  # beau-ti-ful (heuristic matches)


def test_count_syllables_minimum_is_one():
    # words with no clear vowels still return at least 1
    assert count_syllables("gym") >= 1


def test_count_syllables_multi_word():
    # counts vowel clusters across the whole string
    assert count_syllables("the rain falls") == 3


def test_end_sound_strips_punctuation():
    assert _end_sound("streets,") == _end_sound("streets")


def test_end_sound_returns_last_n_chars():
    assert _end_sound("stone", 3) == "one"
    assert _end_sound("tone", 3) == "one"


def test_end_sound_short_word():
    assert _end_sound("hi", 3) == "hi"


# ---------------------------------------------------------------------------
# Score factories
# ---------------------------------------------------------------------------

def test_syllable_scorer_exact_match():
    score = syllable_scorer(3)(None)  # poem arg unused
    assert score("the rain falls") == 0  # 3 syllables


def test_syllable_scorer_distance():
    score = syllable_scorer(5)(None)
    # "rain" = 1 syllable → cost = 4
    assert score("rain") == 4


def test_rhyme_scorer_match(fake_model):
    poem = make_poem("falling rain")
    score = rhyme_scorer()(poem)
    # "rain" ends "ain"; "train" also ends "ain" → rhymes
    assert score("the night train") == 0.0
    # "door" ends "oor" ≠ "ain" → no rhyme
    assert score("open door") == 1.0


def test_rhyme_scorer_no_prev_line():
    poem = make_poem()  # empty poem
    score = rhyme_scorer()(poem)
    assert score("anything") == 0.0  # no reference → always 0


def test_sentiment_scorer_positive():
    score = sentiment_scorer(1.0)(None)
    positive = score("I love this wonderful day")
    negative = score("I hate this terrible day")
    assert positive < negative  # closer to 1.0 = lower cost


def test_sentiment_scorer_negative():
    score = sentiment_scorer(-1.0)(None)
    negative = score("I hate this terrible day")
    neutral = score("the stone sits on the ground")
    assert negative < neutral  # closer to -1.0 = lower cost


# ---------------------------------------------------------------------------
# make_constrained_generator
# ---------------------------------------------------------------------------

class SequentialModel:
    """Returns outputs in order on successive generate() calls."""
    def __init__(self, outputs: list[str]):
        self._outputs = outputs
        self._idx = 0

    def generate(self, prefix: str, max_new_tokens: int = 20) -> str:
        result = self._outputs[self._idx % len(self._outputs)]
        self._idx += 1
        return result


def test_constrained_picks_lowest_cost():
    # syllable_scorer(1) → cost = |syllables - 1|
    # "hi" has 1 syllable (cost 0), "beautiful" has 4 (cost 3)
    model = SequentialModel(["beautiful afternoon sky", "hi", "open door"])
    gen = make_constrained_generator(last_lines(1), syllable_scorer(1), n_candidates=3)
    poem = make_poem("start")
    result = gen(poem, model)
    assert result == "hi"


# ---------------------------------------------------------------------------
# combine_scores
# ---------------------------------------------------------------------------

def test_combine_scores_sums_costs():
    # syllable_scorer(3) + syllable_scorer(1) with equal weights
    combined = combine_scores(syllable_scorer(3), syllable_scorer(1))
    score = combined(None)
    # "rain" = 1 syllable: |1-3| + |1-1| = 2 + 0 = 2
    assert score("rain") == 2.0
    # "the rain falls" = 3 syllables: |3-3| + |3-1| = 0 + 2 = 2
    assert score("the rain falls") == 2.0


def test_combine_scores_respects_weights():
    combined = combine_scores(syllable_scorer(5), syllable_scorer(1), weights=[2.0, 1.0])
    score = combined(None)
    # "rain" = 1 syllable: (|1-5| * 2) + (|1-1| * 1) = 8 + 0 = 8
    assert score("rain") == 8.0


def test_combine_scores_selects_best_candidate():
    # want 3 syllables AND rhyme with "rain"
    model = SequentialModel([
        "the night train",       # "ain" rhymes, 3 syllables → low cost
        "beautiful afternoon sky",  # no rhyme, 5 syllables → high cost
        "plain",                 # rhymes, 1 syllable → medium cost
    ])
    combined = combine_scores(syllable_scorer(3), rhyme_scorer())
    gen = make_constrained_generator(last_lines(1), combined, n_candidates=3)
    poem = make_poem("falling rain")
    result = gen(poem, model)
    assert result == "the night train"


def test_constrained_uses_context_fn():
    # verify the context fed to the model is what context_fn produces
    seen_prefixes = []

    class RecordingModel:
        def generate(self, prefix, max_new_tokens=20):
            seen_prefixes.append(prefix)
            return "output"

    gen = make_constrained_generator(last_lines(1), syllable_scorer(3), n_candidates=2)
    poem = make_poem("the rain falls")
    gen(poem, RecordingModel())
    assert all(p == "the rain falls" for p in seen_prefixes)
    assert len(seen_prefixes) == 2  # n_candidates calls
