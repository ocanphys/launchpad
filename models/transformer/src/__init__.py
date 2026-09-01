"""Standalone transformer package: model, optimizer, and training utilities.
The tokenizer is the `tokenizers.bpe` artifact, re-exported here for the
training helpers that take one."""

from models.transformer.src.modules import TransformerLM
from models.transformer.src.optimizer import AdamW, lr_cosine_schedule
from models.transformer.src.util import (
    configure_logging,
    download_and_concat,
    run_training,
    textfile_to_tokens_as_binary,
)
from tokenizers.bpe import Tokenizer

__all__ = [
    "AdamW",
    "Tokenizer",
    "TransformerLM",
    "configure_logging",
    "download_and_concat",
    "lr_cosine_schedule",
    "run_training",
    "textfile_to_tokens_as_binary",
]
