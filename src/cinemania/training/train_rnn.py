"""Training and evaluation pipeline for the custom PyTorch SentimentRNN."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from cinemania.config import (
    GLOBAL_SEED,
    MAX_SEQ_LENGTH,
    MAX_VOCAB_TOKENS,
    PAD_IDX,
    RNN_ARTIFACTS_DIR,
    RNN_BATCH_SIZE,
    RNN_EMBEDDING_DIM,
    RNN_EPOCHS,
    RNN_HIDDEN_DIM,
    RNN_LR,
    UNK_IDX,
    get_device,
    set_global_seeds,
)
from cinemania.data.rnn_encoding import (
    ReviewRNNDataset,
    build_vocab,
    collate_reviews,
    encode_dataframe,
    save_vocab,
)
from cinemania.evaluation import evaluate_rnn_model
from cinemania.models.sentiment_rnn import SentimentRNN

logger = logging.getLogger(__name__)


def train_and_evaluate_rnn(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    artifact_dir: Path | str = RNN_ARTIFACTS_DIR,
    batch_size: int = RNN_BATCH_SIZE,
    num_epochs: int = RNN_EPOCHS,
    lr: float = RNN_LR,
    max_length: int = MAX_SEQ_LENGTH,
    max_vocab_tokens: int = MAX_VOCAB_TOKENS,
    seed: int = GLOBAL_SEED,
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    """Train SentimentRNN on train_df, evaluate on test_df, and save artifacts."""
    set_global_seeds(seed)
    resolved_device = torch.device(device) if device is not None else get_device()
    output_dir = Path(artifact_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Starting SentimentRNN training on device=%s (train=%d, test=%d, epochs=%d, batch_size=%d, lr=%s)",
        resolved_device,
        len(train_df),
        len(test_df),
        num_epochs,
        batch_size,
        lr,
    )

    # Build vocabulary strictly on RNN training reviews (no test leakage)
    vocab = build_vocab(train_df["review"].astype(str).tolist(), max_tokens=max_vocab_tokens)
    vocab_path = save_vocab(vocab, output_dir / "vocab.json")

    train_encoded_df = encode_dataframe(train_df, vocab=vocab, max_length=max_length)
    test_encoded_df = encode_dataframe(test_df, vocab=vocab, max_length=max_length)

    train_dataset = ReviewRNNDataset(
        train_encoded_df["encoded"].tolist(),
        train_encoded_df["label"].tolist(),
    )
    test_dataset = ReviewRNNDataset(
        test_encoded_df["encoded"].tolist(),
        test_encoded_df["label"].tolist(),
    )

    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_reviews,
        generator=generator,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_reviews,
    )

    model = SentimentRNN(
        vocab_size=len(vocab),
        embedding_dim=RNN_EMBEDDING_DIM,
        hidden_dim=RNN_HIDDEN_DIM,
        pad_idx=PAD_IDX,
    ).to(resolved_device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    epoch_losses: list[dict[str, float | int]] = []
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_weighted_loss = 0.0
        total_samples = 0

        for input_ids, lengths, labels in train_loader:
            input_ids = input_ids.to(resolved_device)
            labels_col = labels.to(resolved_device).unsqueeze(1)

            optimizer.zero_grad()
            probs = model(input_ids, lengths)
            loss = criterion(probs, labels_col)
            loss.backward()
            optimizer.step()

            current_batch_size = input_ids.size(0)
            total_weighted_loss += float(loss.item()) * current_batch_size
            total_samples += current_batch_size

        avg_epoch_loss = total_weighted_loss / total_samples
        epoch_losses.append({"epoch": epoch, "train_loss": avg_epoch_loss})
        logger.info("RNN Epoch %d/%d | train_loss=%.6f", epoch, num_epochs, avg_epoch_loss)

    # Final evaluation on held-out shared test set
    test_metrics = evaluate_rnn_model(model, test_loader, criterion, resolved_device)
    logger.info("RNN Test Metrics (500 held-out reviews): %s", test_metrics)

    model_path = output_dir / "sentiment_rnn.pt"
    torch.save(model.state_dict(), model_path)

    config_payload = {
        "vocab_size": len(vocab),
        "embedding_dim": RNN_EMBEDDING_DIM,
        "hidden_dim": RNN_HIDDEN_DIM,
        "pad_idx": PAD_IDX,
        "unk_idx": UNK_IDX,
        "max_length": max_length,
        "batch_size": batch_size,
        "learning_rate": lr,
        "num_epochs": num_epochs,
        "seed": seed,
    }
    config_path = output_dir / "config.json"
    config_path.write_text(json.dumps(config_payload, indent=2), encoding="utf-8")

    logger.info("Saved RNN artifacts to '%s'.", output_dir)
    results = {
        "device": str(resolved_device),
        "vocab_size": len(vocab),
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "epoch_losses": epoch_losses,
        "test_metrics": test_metrics,
        "artifacts": {
            "model_path": str(model_path),
            "vocab_path": str(vocab_path),
            "config_path": str(config_path),
        },
    }
    (output_dir / "train_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results
