"""Deterministic tokenization, vocabulary construction, and PyTorch batching for SentimentRNN."""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from cinemania.config import (
    MAX_SEQ_LENGTH,
    MAX_VOCAB_TOKENS,
    PAD_IDX,
    PAD_TOKEN,
    RNN_VOCAB_PATH,
    UNK_IDX,
    UNK_TOKEN,
)

logger = logging.getLogger(__name__)

TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")


def tokenize_text(text: str) -> list[str]:
    """Tokenize English review text deterministically into lowercase word tokens."""
    return TOKEN_PATTERN.findall(text.lower())


def build_vocab(
    texts: Iterable[str],
    max_tokens: int = MAX_VOCAB_TOKENS,
) -> dict[str, int]:
    """Build a token-to-index vocabulary from training texts only."""
    counter: Counter[str] = Counter()
    for text in texts:
        counter.update(tokenize_text(str(text)))

    vocab: dict[str, int] = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX}
    # Sort deterministically by (-frequency, token) so ties break consistently
    most_common = sorted(counter.items(), key=lambda item: (-item[1], item[0]))[:max_tokens]
    for idx, (token, _) in enumerate(most_common, start=2):
        vocab[token] = idx

    logger.info(
        "Built RNN vocabulary: %d total tokens (%d ordinary + 2 special tokens).",
        len(vocab),
        len(vocab) - 2,
    )
    return vocab


def save_vocab(vocab: Mapping[str, int], path: Path | str = RNN_VOCAB_PATH) -> Path:
    """Save vocabulary mapping to JSON."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(dict(vocab), indent=2), encoding="utf-8")
    logger.info("Saved RNN vocabulary to '%s'.", output_path)
    return output_path


def load_vocab(path: Path | str = RNN_VOCAB_PATH) -> dict[str, int]:
    """Load vocabulary mapping from JSON."""
    vocab_path = Path(path)
    if not vocab_path.exists():
        raise FileNotFoundError(f"RNN vocabulary file not found: '{vocab_path}'.")
    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    return {str(k): int(v) for k, v in vocab.items()}


def encode_review(
    text: str,
    vocab: Mapping[str, int],
    max_length: int = MAX_SEQ_LENGTH,
) -> list[int]:
    """Convert raw review text into a non-empty truncated sequence of vocabulary IDs."""
    unk_id = vocab.get(UNK_TOKEN, UNK_IDX)
    tokens = tokenize_text(text)[:max_length]
    if not tokens:
        return [unk_id]
    return [vocab.get(token, unk_id) for token in tokens]


def encode_dataframe(
    df: pd.DataFrame,
    vocab: Mapping[str, int],
    max_length: int = MAX_SEQ_LENGTH,
) -> pd.DataFrame:
    """Return a copy of df with an 'encoded' column without modifying 'review'."""
    encoded_df = df.copy()
    encoded_df["encoded"] = [encode_review(str(text), vocab=vocab, max_length=max_length) for text in encoded_df["review"]]
    return encoded_df


class ReviewRNNDataset(Dataset[tuple[list[int], int]]):
    """PyTorch Dataset holding encoded token-ID sequences and binary labels."""

    def __init__(self, encoded_sequences: Sequence[Sequence[int]], labels: Sequence[int]) -> None:
        if len(encoded_sequences) != len(labels):
            raise ValueError("encoded_sequences and labels must have the same length.")
        self.sequences = [list(seq) for seq in encoded_sequences]
        self.labels = [int(label) for label in labels]

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple[list[int], int]:
        return self.sequences[idx], self.labels[idx]


def collate_reviews(
    batch: Sequence[tuple[Sequence[int], int]],
    pad_idx: int = PAD_IDX,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dynamically pad a batch of variable-length encoded reviews to the batch max length."""
    if not batch:
        raise ValueError("Cannot collate an empty batch.")

    lengths = torch.tensor([len(seq) for seq, _ in batch], dtype=torch.int64)
    max_len = int(lengths.max().item())
    padded = torch.full((len(batch), max_len), fill_value=pad_idx, dtype=torch.long)
    labels = torch.tensor([label for _, label in batch], dtype=torch.float32)

    for row_idx, (seq, _) in enumerate(batch):
        seq_len = len(seq)
        padded[row_idx, :seq_len] = torch.tensor(seq, dtype=torch.long)

    return padded, lengths, labels
