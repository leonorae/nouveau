from nouveau.generators import (
    gpt_alternating,
    gpt_bookend,
    gpt_closure,
    gpt_first,
    gpt_last,
    gpt_window,
    make_window_generator,
)
from nouveau.poem import Poem


def make_poem(*lines, max_lines=10):
    p = Poem(max_lines=max_lines, generator_name="gpt_last", model_name="fake")
    for i, text in enumerate(lines):
        p.add_line(text, author="human" if i % 2 == 0 else "ai")
    return p


def test_gpt_last_uses_final_line(fake_model):
    poem = make_poem("first", "second", "third")
    gpt_last(poem, fake_model)
    assert fake_model.last_prefix == "third"


def test_gpt_first_uses_first_line(fake_model):
    poem = make_poem("first", "second", "third")
    gpt_first(poem, fake_model)
    assert fake_model.last_prefix == "first"


def test_gpt_first_returns_generated(fake_model):
    poem = make_poem("opening")
    result = gpt_first(poem, fake_model)
    assert result == "[opening]"


def test_gpt_last_returns_generated(fake_model):
    poem = make_poem("a", "b", "closing")
    result = gpt_last(poem, fake_model)
    assert result == "[closing]"


def test_gpt_closure_uses_last_when_not_final(fake_model):
    # max_lines=6, poem has 4 lines — not the penultimate position yet
    poem = make_poem("a", "b", "c", "d", max_lines=6)
    gpt_closure(poem, fake_model)
    assert fake_model.last_prefix == "d"


def test_gpt_closure_uses_first_on_final_line(fake_model):
    # max_lines=6, poem has 5 lines — the next AI line closes the poem
    poem = make_poem("a", "b", "c", "d", "e", max_lines=6)
    gpt_closure(poem, fake_model)
    assert fake_model.last_prefix == "a"


def test_gpt_closure_two_line_poem(fake_model):
    # smallest possible: max_lines=2, 1 line in -> closure triggers immediately
    poem = make_poem("only", max_lines=2)
    gpt_closure(poem, fake_model)
    assert fake_model.last_prefix == "only"


def test_gpt_window_uses_last_three_lines(fake_model):
    poem = make_poem("a", "b", "c", "d", "e")
    gpt_window(poem, fake_model)
    assert fake_model.last_prefix == "c\nd\ne"


def test_gpt_window_fewer_lines_than_window(fake_model):
    poem = make_poem("a", "b")
    gpt_window(poem, fake_model)
    assert fake_model.last_prefix == "a\nb"


def test_gpt_window_single_line(fake_model):
    poem = make_poem("alone")
    gpt_window(poem, fake_model)
    assert fake_model.last_prefix == "alone"


def test_gpt_window_returns_generated(fake_model):
    poem = make_poem("x", "y", "z")
    result = gpt_window(poem, fake_model)
    assert result == "[x\ny\nz]"


# ---------------------------------------------------------------------------
# gpt_bookend
# ---------------------------------------------------------------------------

def test_gpt_bookend_uses_first_and_last(fake_model):
    poem = make_poem("alpha", "b", "c", "omega")
    gpt_bookend(poem, fake_model)
    assert fake_model.last_prefix == "alpha\nomega"


def test_gpt_bookend_single_line(fake_model):
    # with one line, index 0 and -1 are the same; deduplication is not required
    poem = make_poem("solo")
    gpt_bookend(poem, fake_model)
    assert fake_model.last_prefix == "solo\nsolo"


# ---------------------------------------------------------------------------
# gpt_alternating
# ---------------------------------------------------------------------------

def test_gpt_alternating_even_indexed_lines(fake_model):
    poem = make_poem("a", "b", "c", "d", "e")
    gpt_alternating(poem, fake_model)
    assert fake_model.last_prefix == "a\nc\ne"


def test_gpt_alternating_two_lines(fake_model):
    poem = make_poem("a", "b")
    gpt_alternating(poem, fake_model)
    assert fake_model.last_prefix == "a"


# ---------------------------------------------------------------------------
# make_window_generator
# ---------------------------------------------------------------------------

def test_make_window_generator_int(fake_model):
    gen = make_window_generator(2)
    poem = make_poem("a", "b", "c", "d")
    gen(poem, fake_model)
    assert fake_model.last_prefix == "c\nd"


def test_make_window_generator_list(fake_model):
    gen = make_window_generator([0, 2])
    poem = make_poem("a", "b", "c")
    gen(poem, fake_model)
    assert fake_model.last_prefix == "a\nc"


def test_make_window_generator_slice(fake_model):
    gen = make_window_generator(slice(1, None))
    poem = make_poem("a", "b", "c")
    gen(poem, fake_model)
    assert fake_model.last_prefix == "b\nc"


def test_make_window_generator_out_of_range_indices_skipped(fake_model):
    # index 5 doesn't exist in a 3-line poem — should be silently skipped
    gen = make_window_generator([0, 5])
    poem = make_poem("a", "b", "c")
    gen(poem, fake_model)
    assert fake_model.last_prefix == "a"
