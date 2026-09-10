"""Standalone transformer package: tokenizer, model, optimizer, and training utilities."""

from artifacts.core.SGD.lr_schedule import lr_cosine_schedule
from artifacts.models.transformer.src.model import TransformerLM
from artifacts.models.transformer.src.optimizer import AdamW
from artifacts.models.transformer.src.tokenizer import train_bpe
from artifacts.tokenizers.bpe import Tokenizer

__all__ = [
    "AdamW",
    "Tokenizer",
    "TransformerLM",
    "lr_cosine_schedule",
    "train_bpe",
]
