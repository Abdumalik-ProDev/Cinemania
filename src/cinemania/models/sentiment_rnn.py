"""Custom PyTorch SentimentRNN architecture."""

from __future__ import annotations

import torch
from torch import nn

from cinemania.config import PAD_IDX, RNN_EMBEDDING_DIM, RNN_HIDDEN_DIM


class SentimentRNN(nn.Module):
    """Unidirectional vanilla RNN binary sentiment classifier."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = RNN_EMBEDDING_DIM,
        hidden_dim: int = RNN_HIDDEN_DIM,
        pad_idx: int = PAD_IDX,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.pad_idx = pad_idx

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=False,
        )
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        input_ids: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute sentiment probabilities of shape (batch_size, 1) ignoring trailing padding."""
        if lengths is None:
            lengths = (input_ids != self.pad_idx).sum(dim=1).clamp(min=1)

        embedded = self.embedding(input_ids)
        # pack_padded_sequence requires 1D CPU int64 lengths even when embeddings are on CUDA
        cpu_lengths = lengths.to(device="cpu", dtype=torch.int64).clamp(min=1)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            cpu_lengths,
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.rnn(packed)
        # hidden shape: (1, batch_size, hidden_dim) -> final valid state per sequence
        last_hidden = hidden[-1]
        logits = self.fc(last_hidden)
        return self.sigmoid(logits)
