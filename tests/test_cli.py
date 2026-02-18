import json
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from nouveau.cli import cli


def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "compose" in result.output
    assert "show" in result.output
    assert "list" in result.output


def test_compose_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["compose", "--help"])
    assert result.exit_code == 0
    assert "MAX_LINES" in result.output
    assert "GENERATOR" in result.output


def test_compose_max_lines_too_small():
    runner = CliRunner()
    result = runner.invoke(cli, ["compose", "1", "last"])
    assert result.exit_code != 0


def test_compose_invalid_generator():
    runner = CliRunner()
    result = runner.invoke(cli, ["compose", "4", "nonexistent"])
    assert result.exit_code != 0


def test_compose_full_run(fake_model, tmp_path):
    runner = CliRunner()
    with patch("nouveau.cli.Model", return_value=fake_model):
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(
                cli,
                ["compose", "4", "last"],
                input="line one\nline two\n",
            )
    assert result.exit_code == 0, result.output
    assert "Poem saved to" in result.output


def test_compose_saves_valid_json(fake_model, tmp_path):
    runner = CliRunner()
    with patch("nouveau.cli.Model", return_value=fake_model):
        with runner.isolated_filesystem(temp_dir=tmp_path):
            runner.invoke(
                cli,
                ["compose", "4", "last"],
                input="first\nsecond\n",
            )
            poems = list(Path("poems").glob("*.json"))
            assert len(poems) == 1
            data = json.loads(poems[0].read_text())
    assert data["schema_version"] == 1
    assert data["generator"] == "last"
    assert len(data["lines"]) == 4
    assert data["lines"][0] == {"author": "human", "text": "first"}
    assert data["lines"][2] == {"author": "human", "text": "second"}


def test_compose_temperature_passed_to_model(tmp_path):
    captured = {}

    class CapturingModel:
        def __init__(self, model_name, temperature):
            captured["temperature"] = temperature

        def generate(self, prefix, max_new_tokens=20):
            return "line"

    runner = CliRunner()
    with patch("nouveau.cli.Model", CapturingModel):
        with runner.isolated_filesystem(temp_dir=tmp_path):
            runner.invoke(
                cli,
                ["compose", "2", "last", "--temperature", "1.2"],
                input="hello\n",
            )
    assert captured["temperature"] == 1.2


def test_show_displays_poem(tmp_path):
    poem_file = tmp_path / "poem.json"
    poem_file.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "created_at": "2024-01-01T12:00:00",
                "model": "gpt2",
                "generator": "last",
                "lines": [
                    {"author": "human", "text": "the river bends"},
                    {"author": "ai", "text": "light on the water"},
                ],
            }
        )
    )
    runner = CliRunner()
    result = runner.invoke(cli, ["show", str(poem_file)])
    assert result.exit_code == 0
    assert "last" in result.output
    assert "gpt2" in result.output
    assert "the river bends" in result.output
    assert "light on the water" in result.output


def test_show_labels_authors(tmp_path):
    poem_file = tmp_path / "poem.json"
    poem_file.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "created_at": "2024-01-01T12:00:00",
                "model": "gpt2",
                "generator": "last",
                "lines": [
                    {"author": "human", "text": "human line"},
                    {"author": "ai", "text": "ai line"},
                ],
            }
        )
    )
    runner = CliRunner()
    result = runner.invoke(cli, ["show", str(poem_file)])
    assert "you" in result.output
    assert " AI" in result.output


def test_show_missing_file():
    runner = CliRunner()
    result = runner.invoke(cli, ["show", "/nonexistent/path.json"])
    assert result.exit_code != 0


def test_list_no_poems(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(cli, ["list"])
    assert result.exit_code == 0
    assert "No poems" in result.output


def test_list_shows_poem_metadata(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        poems_dir = Path("poems")
        poems_dir.mkdir()
        (poems_dir / "2024-01-01.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "created_at": "2024-01-01T12:00:00",
                    "model": "gpt2",
                    "generator": "closure",
                    "lines": [
                        {"author": "human", "text": "a"},
                        {"author": "ai", "text": "b"},
                        {"author": "human", "text": "c"},
                        {"author": "ai", "text": "d"},
                    ],
                }
            )
        )
        result = runner.invoke(cli, ["list"])
    assert result.exit_code == 0
    assert "closure" in result.output
    assert "4 lines" in result.output
