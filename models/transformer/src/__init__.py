"""Standalone transformer package: tokenizer, model, optimizer, and training utilities."""

from models.transformer.src.modules import TransformerLM
from models.transformer.src.optimizer import AdamW, lr_cosine_schedule
from models.transformer.src.tokenizer import Tokenizer, train_bpe
from models.transformer.src.util import (
    configure_logging,
    download_and_concat,
    prepare_tokenizer,
    run_training,
    textfile_to_tokens_as_binary,
)

__all__ = [
    "AdamW",
    "Tokenizer",
    "TransformerLM",
    "configure_logging",
    "download_and_concat",
    "lr_cosine_schedule",
    "prepare_tokenizer",
    "run_training",
    "textfile_to_tokens_as_binary",
    "train_bpe",
]
