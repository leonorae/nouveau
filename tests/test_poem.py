import json

import pytest

from nouveau.poem import Poem


def make_poem(**kwargs):
    defaults = dict(max_lines=4, generator_name="last", model_name="fake")
    return Poem(**{**defaults, **kwargs})


def test_add_line():
    p = make_poem()
    p.add_line("hello", author="human")
    assert len(p) == 1
    assert p[0] == "hello"


def test_add_multiple_lines():
    p = make_poem()
    p.add_line("a", author="human")
    p.add_line("b", author="ai")
    assert len(p) == 2
    assert p[1] == "b"


def test_is_full_false_when_empty():
    p = make_poem(max_lines=2)
    assert not p.is_full()


def test_is_full_true_when_at_capacity():
    p = make_poem(max_lines=2)
    p.add_line("a", author="human")
    p.add_line("b", author="ai")
    assert p.is_full()


def test_add_to_full_poem_raises():
    p = make_poem(max_lines=1)
    p.add_line("a", author="human")
    with pytest.raises(ValueError, match="poem is full"):
        p.add_line("b", author="human")


def test_to_dict_structure():
    p = make_poem(max_lines=2, generator_name="closure", model_name="gpt2-finetuned")
    p.add_line("a", author="human")
    p.add_line("b", author="ai")
    d = p.to_dict()
    assert d["schema_version"] == 1
    assert d["generator"] == "closure"
    assert d["model"] == "gpt2-finetuned"
    assert d["lines"] == [
        {"author": "human", "text": "a"},
        {"author": "ai", "text": "b"},
    ]
    assert "created_at" in d


def test_to_dict_preserves_author():
    p = make_poem()
    p.add_line("human line", author="human")
    p.add_line("ai line", author="ai")
    lines = p.to_dict()["lines"]
    assert lines[0]["author"] == "human"
    assert lines[1]["author"] == "ai"


def test_save_creates_json_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    p = make_poem(max_lines=2)
    p.add_line("a", author="human")
    p.add_line("b", author="ai")
    path = p.save()
    assert path.exists()
    assert path.suffix == ".json"
    data = json.loads(path.read_text())
    assert data["schema_version"] == 1
    assert len(data["lines"]) == 2


def test_save_content_roundtrip(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    p = make_poem(max_lines=2, generator_name="window")
    p.add_line("first line", author="human")
    p.add_line("second line", author="ai")
    path = p.save()
    data = json.loads(path.read_text())
    assert data["generator"] == "window"
    assert data["lines"][0]["text"] == "first line"
    assert data["lines"][1]["text"] == "second line"
