"""
Fine-tune GPT-2 on the prepared poetry corpus.

Prerequisites:
    uv run python data/prepare.py   # produces poetry.txt

Run:
    uv run python train.py

The fine-tuned checkpoint is saved to ./checkpoint/. Point the CLI at it with:
    uv run nouveau 10 gpt_last --model ./checkpoint
"""
from __future__ import annotations

from pathlib import Path

from datasets import Dataset
from transformers import (
    DataCollatorForLanguageModeling,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,
)

CORPUS = Path("poetry.txt")
CHECKPOINT_DIR = Path("checkpoint")
BASE_MODEL = "gpt2"
BLOCK_SIZE = 128


def load_corpus(tokenizer: GPT2Tokenizer) -> Dataset:
    text = CORPUS.read_text()
    lines = [l for l in text.splitlines() if l.strip()]

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=BLOCK_SIZE)

    ds = Dataset.from_dict({"text": lines})
    return ds.map(tokenize, batched=True, remove_columns=["text"])


def main() -> None:
    if not CORPUS.exists():
        raise FileNotFoundError(f"{CORPUS} not found. Run `uv run python data/prepare.py` first.")

    tokenizer = GPT2Tokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained(BASE_MODEL)

    dataset = load_corpus(tokenizer)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    args = TrainingArguments(
        output_dir=str(CHECKPOINT_DIR),
        num_train_epochs=3,
        per_device_train_batch_size=8,
        save_steps=500,
        save_total_limit=2,
        logging_steps=100,
        fp16=False,  # set True if your GPU supports it
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=collator,
    )

    print(f"Fine-tuning {BASE_MODEL} on {len(dataset):,} samples...")
    trainer.train()

    model.save_pretrained(CHECKPOINT_DIR)
    tokenizer.save_pretrained(CHECKPOINT_DIR)
    print(f"Checkpoint saved to {CHECKPOINT_DIR}/")


if __name__ == "__main__":
    main()
