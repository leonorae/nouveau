"""
Prepare the Gutenberg Poetry Corpus for fine-tuning.

Outputs poetry.txt in the repo root, one line per sample.
Run with: uv run python data/prepare.py
"""
from datasets import load_dataset


def main() -> None:
    print("Loading biglam/gutenberg-poetry-corpus...")
    ds = load_dataset("biglam/gutenberg-poetry-corpus")["train"]

    # Sample every other line to reduce repetition within the same poem
    lines: list[str] = ds[::2]["line"]

    out_path = "poetry.txt"
    with open(out_path, "w") as f:
        f.write("\n".join(lines))

    print(f"Wrote {len(lines):,} lines to {out_path}")


if __name__ == "__main__":
    main()
