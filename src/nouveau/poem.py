from __future__ import annotations

import datetime
import json
from dataclasses import dataclass, field
from pathlib import Path

POEM_DIR = Path("poems")
SCHEMA_VERSION = 1


@dataclass
class Line:
    author: str  # "human" | "ai"
    text: str


@dataclass
class Poem:
    max_lines: int
    generator_name: str
    model_name: str
    lines: list[Line] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.datetime.now().isoformat())
    schema_version: int = SCHEMA_VERSION

    def __len__(self) -> int:
        return len(self.lines)

    def __getitem__(self, index: int) -> str:
        return self.lines[index].text

    def is_full(self) -> bool:
        return len(self.lines) >= self.max_lines

    def add_line(self, text: str, author: str) -> None:
        if self.is_full():
            raise ValueError("poem is full")
        self.lines.append(Line(author=author, text=text))

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "created_at": self.created_at,
            "model": self.model_name,
            "generator": self.generator_name,
            "lines": [{"author": line.author, "text": line.text} for line in self.lines],
        }

    def save(self) -> Path:
        POEM_DIR.mkdir(exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        path = POEM_DIR / f"{timestamp}.json"
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path
