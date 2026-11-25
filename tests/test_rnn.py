"""Unit tests for RNN vocabulary, encoding, dynamic batching, and SentimentRNN."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader

from cinemania.data.rnn_encoding import (
    ReviewRNNDataset,
    build_vocab,
    collate_reviews,
    encode_review,
)
from cinemania.evaluation import evaluate_rnn_model
from cinemania.models.sentiment_rnn import SentimentRNN


def test_vocab_special_tokens_and_max_size() -> None:
    texts = [f"word{i} common" for i in range(4500)]
    vocab = build_vocab(texts, max_tokens=4000)
    assert vocab["<pad>"] == 0
    assert vocab["<unk>"] == 1
    assert len(vocab) == 4002
    assert vocab["common"] == 2


def test_encode_review_unknown_truncation_and_empty() -> None:
    vocab = {"<pad>": 0, "<unk>": 1, "great": 2, "movie": 3}
    encoded = encode_review("Great unseenword movie!", vocab=vocab, max_length=256)
    assert encoded == [2, 1, 3]

    long_text = " ".join(["great"] * 300)
    assert len(encode_review(long_text, vocab=vocab, max_length=256)) == 256

    assert encode_review("!!! ???", vocab=vocab, max_length=256) == [1]


def test_collate_and_sentiment_rnn_padding_invariance() -> None:
    torch.manual_seed(42)
    batch = [([2, 3, 4], 1), ([2, 3, 4, 5, 6], 0)]
    padded, lengths, labels = collate_reviews(batch, pad_idx=0)
    assert padded.shape == (2, 5)
    assert padded[0, 3:].tolist() == [0, 0]
    assert lengths.tolist() == [3, 5]
    assert labels.tolist() == [1.0, 0.0]

    model = SentimentRNN(vocab_size=20, embedding_dim=128, hidden_dim=256, pad_idx=0)
    model.eval()
    with torch.no_grad():
        probs = model(padded, lengths)
        assert probs.shape == (2, 1)
        assert torch.all((probs >= 0.0) & (probs <= 1.0))

        # Trailing <pad> tokens must not change the prediction of the shorter sequence
        unpadded_short = torch.tensor([[2, 3, 4]], dtype=torch.long)
        unpadded_len = torch.tensor([3], dtype=torch.int64)
        prob_short = model(unpadded_short, unpadded_len)
        assert torch.allclose(probs[0:1], prob_short, atol=1e-6)

        # Inference without explicit lengths derives lengths from pad_idx cleanly
        probs_auto = model(padded)
        assert torch.allclose(probs, probs_auto, atol=1e-6)


def test_sample_weighted_evaluation_handles_uneven_batches() -> None:
    torch.manual_seed(0)
    dataset = ReviewRNNDataset([[2, 3], [4, 5], [2, 4]], [1, 0, 1])
    loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_reviews)
    model = SentimentRNN(vocab_size=10, embedding_dim=128, hidden_dim=256, pad_idx=0)
    criterion = nn.BCELoss()
    metrics = evaluate_rnn_model(model, loader, criterion, torch.device("cpu"))
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert metrics["loss"] > 0.0
