import pytest


class FakeModel:
    """Deterministic stand-in for Model. Records the last prefix passed to generate()."""

    def __init__(self):
        self.last_prefix: str | None = None

    def generate(self, prefix: str, max_new_tokens: int = 20) -> str:
        self.last_prefix = prefix
        return f"[{prefix[:16]}]"


@pytest.fixture
def fake_model():
    return FakeModel()


@pytest.fixture
def short_poem():
    from nouveau.poem import Poem

    p = Poem(max_lines=6, generator_name="gpt_last", model_name="fake")
    p.add_line("the river bends", author="human")
    p.add_line("light on the water", author="ai")
    p.add_line("no one remembers", author="human")
    p.add_line("only the grass", author="ai")
    return p
