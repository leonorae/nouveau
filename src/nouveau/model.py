from __future__ import annotations

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

DEFAULT_MODEL = "gpt2"


class Model:
    """Wraps a local HuggingFace GPT-2 model for single-line generation."""

    def __init__(self, model_name: str = DEFAULT_MODEL, checkpoint_path: str | None = None):
        path = checkpoint_path or model_name
        self.model_name = model_name
        self.tokenizer = GPT2Tokenizer.from_pretrained(path)
        self.llm = GPT2LMHeadModel.from_pretrained(path)
        self.llm.eval()

    def generate(self, prefix: str, max_new_tokens: int = 20, temperature: float = 0.7) -> str:
        inputs = self.tokenizer(prefix, return_tensors="pt")
        with torch.no_grad():
            output = self.llm.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        # decode only the newly generated tokens
        new_tokens = output[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        # trim to first newline so output stays on one line
        return text.split("\n")[0].strip()
