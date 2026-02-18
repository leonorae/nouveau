from nouveau.generators import (
    alternating,
    bookend,
    closure,
    first,
    first_lines,
    first_words,
    last,
    last_lines,
    last_words,
    line_window,
    make_conditional,
    make_generator,
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
