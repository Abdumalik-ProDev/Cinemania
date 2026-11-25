"""Reusable inference functions and CLI for SentimentRNN and DistilBERT."""

from __future__ import annotations

import argparse
import json
import threading
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from cinemania.config import (
    EXAMPLE_REVIEWS,
    ID2LABEL,
    LABEL2ID,
    MAX_SEQ_LENGTH,
    PAD_IDX,
    RNN_ARTIFACTS_DIR,
    RNN_EMBEDDING_DIM,
    RNN_HIDDEN_DIM,
    TRANSFORMER_CHECKPOINT_DIR,
    TRANSFORMER_MAX_LENGTH,
    get_device,
)
from cinemania.data.rnn_encoding import encode_review, load_vocab
from cinemania.evaluation import format_example_predictions_table
from cinemania.models.sentiment_rnn import SentimentRNN

_CACHE_LOCK = threading.Lock()
_RNN_CACHE: dict[tuple[str, str], tuple[SentimentRNN, dict[str, int], dict[str, Any]]] = {}
_TRANSFORMER_CACHE: dict[tuple[str, str], tuple[AutoModelForSequenceClassification, Any]] = {}


def clear_inference_cache() -> None:
    """Clear in-memory cached inference models (useful in tests or after retraining)."""
    with _CACHE_LOCK:
        _RNN_CACHE.clear()
        _TRANSFORMER_CACHE.clear()


def _validate_text(text: str) -> str:
    """Ensure input review is a non-empty string."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Input review must be a non-empty string.")
    return text


def _load_rnn_bundle(
    artifact_dir: Path | str,
    device: torch.device,
) -> tuple[SentimentRNN, dict[str, int], dict[str, Any]]:
    base_dir = Path(artifact_dir)
    cache_key = (str(base_dir.resolve()), str(device))
    with _CACHE_LOCK:
        if cache_key in _RNN_CACHE:
            return _RNN_CACHE[cache_key]

        model_path = base_dir / "sentiment_rnn.pt"
        vocab_path = base_dir / "vocab.json"
        config_path = base_dir / "config.json"

        for required_path in (model_path, vocab_path, config_path):
            if not required_path.exists():
                raise FileNotFoundError(
                    f"Required RNN artifact missing: '{required_path}'. Train the RNN before running inference."
                )

        vocab = load_vocab(vocab_path)
        config = json.loads(config_path.read_text(encoding="utf-8"))

        model = SentimentRNN(
            vocab_size=int(config.get("vocab_size", len(vocab))),
            embedding_dim=int(config.get("embedding_dim", RNN_EMBEDDING_DIM)),
            hidden_dim=int(config.get("hidden_dim", RNN_HIDDEN_DIM)),
            pad_idx=int(config.get("pad_idx", PAD_IDX)),
        ).to(device)

        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()

        bundle = (model, vocab, config)
        _RNN_CACHE[cache_key] = bundle
        return bundle


def _load_transformer_bundle(
    checkpoint_dir: Path | str,
    device: torch.device,
) -> tuple[AutoModelForSequenceClassification, Any]:
    ckpt_dir = Path(checkpoint_dir)
    if not ckpt_dir.exists() or not ckpt_dir.is_dir():
        raise FileNotFoundError(
            f"Transformer checkpoint directory not found: '{ckpt_dir}'. "
            "Train the Transformer before running inference."
        )
    if not any(ckpt_dir.iterdir()):
        raise FileNotFoundError(f"Transformer checkpoint directory is empty: '{ckpt_dir}'.")

    cache_key = (str(ckpt_dir.resolve()), str(device))
    with _CACHE_LOCK:
        if cache_key in _TRANSFORMER_CACHE:
            return _TRANSFORMER_CACHE[cache_key]

        tokenizer = AutoTokenizer.from_pretrained(str(ckpt_dir), local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(str(ckpt_dir), local_files_only=True).to(device)
        model.eval()

        bundle = (model, tokenizer)
        _TRANSFORMER_CACHE[cache_key] = bundle
        return bundle


def predict_sentiment_rnn(
    text: str,
    *,
    artifact_dir: Path | str = RNN_ARTIFACTS_DIR,
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    """Predict sentiment ('positive' or 'negative') for one review using the saved SentimentRNN."""
    validated = _validate_text(text)
    resolved_device = torch.device(device) if device is not None else get_device()
    model, vocab, config = _load_rnn_bundle(artifact_dir, resolved_device)

    max_length = int(config.get("max_length", MAX_SEQ_LENGTH))
    token_ids = encode_review(validated, vocab=vocab, max_length=max_length)
    input_tensor = torch.tensor([token_ids], dtype=torch.long, device=resolved_device)
    lengths_tensor = torch.tensor([len(token_ids)], dtype=torch.int64)

    model.eval()
    with torch.no_grad():
        prob_pos = float(model(input_tensor, lengths_tensor).squeeze().item())

    prob_neg = 1.0 - prob_pos
    pred_id = 1 if prob_pos >= 0.5 else 0
    label = ID2LABEL[pred_id]
    confidence = prob_pos if pred_id == 1 else prob_neg

    return {
        "model": "rnn",
        "label": label,
        "confidence": float(confidence),
        "probabilities": {
            "negative": float(prob_neg),
            "positive": float(prob_pos),
        },
    }


def predict_sentiment_transformer(
    text: str,
    *,
    checkpoint_dir: Path | str = TRANSFORMER_CHECKPOINT_DIR,
    device: torch.device | str | None = None,
    max_length: int = TRANSFORMER_MAX_LENGTH,
) -> dict[str, Any]:
    """Predict sentiment ('positive' or 'negative') for one review using the saved DistilBERT."""
    validated = _validate_text(text)
    resolved_device = torch.device(device) if device is not None else get_device()
    model, tokenizer = _load_transformer_bundle(checkpoint_dir, resolved_device)

    encoded = tokenizer(
        validated,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    encoded = {key: val.to(resolved_device) for key, val in encoded.items()}

    model.eval()
    with torch.no_grad():
        logits = model(**encoded).logits
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().tolist()

    prob_neg = float(probs[LABEL2ID["negative"]])
    prob_pos = float(probs[LABEL2ID["positive"]])
    pred_id = 1 if prob_pos >= prob_neg else 0
    label = ID2LABEL[pred_id]
    confidence = prob_pos if pred_id == 1 else prob_neg

    return {
        "model": "transformer",
        "label": label,
        "confidence": float(confidence),
        "probabilities": {
            "negative": prob_neg,
            "positive": prob_pos,
        },
    }


def evaluate_example_reviews(
    *,
    rnn_dir: Path | str = RNN_ARTIFACTS_DIR,
    transformer_dir: Path | str = TRANSFORMER_CHECKPOINT_DIR,
) -> list[dict[str, Any]]:
    """Run both saved models on the three required English example reviews."""
    rows: list[dict[str, Any]] = []
    for item in EXAMPLE_REVIEWS:
        rnn_out = predict_sentiment_rnn(item["review"], artifact_dir=rnn_dir)
        tf_out = predict_sentiment_transformer(item["review"], checkpoint_dir=transformer_dir)
        rows.append(
            {
                "category": item["category"],
                "review": item["review"],
                "rnn_prediction": rnn_out["label"],
                "rnn_confidence": rnn_out["confidence"],
                "rnn_prob_negative": rnn_out["probabilities"]["negative"],
                "rnn_prob_positive": rnn_out["probabilities"]["positive"],
                "transformer_prediction": tf_out["label"],
                "transformer_confidence": tf_out["confidence"],
                "transformer_prob_negative": tf_out["probabilities"]["negative"],
                "transformer_prob_positive": tf_out["probabilities"]["positive"],
            }
        )
    return rows


def main(argv: list[str] | None = None) -> None:
    """CLI entry point to classify reviews with both trained models."""
    parser = argparse.ArgumentParser(description="Run Cinemania sentiment inference on English movie reviews.")
    parser.add_argument("reviews", nargs="*", help="One or more English movie reviews to classify.")
    parser.add_argument(
        "--examples",
        action="store_true",
        help="Run inference on the three required assignment example reviews.",
    )
    args = parser.parse_args(argv)

    if args.examples or not args.reviews:
        rows = evaluate_example_reviews()
        print(format_example_predictions_table(rows))
        return

    for idx, review in enumerate(args.reviews, start=1):
        rnn_out = predict_sentiment_rnn(review)
        tf_out = predict_sentiment_transformer(review)
        print(f"Review {idx}: {review}")
        print(
            f"  RNN         -> {rnn_out['label']} (confidence={rnn_out['confidence']:.4f}, "
            f"probs={rnn_out['probabilities']})"
        )
        print(
            f"  Transformer -> {tf_out['label']} (confidence={tf_out['confidence']:.4f}, "
            f"probs={tf_out['probabilities']})"
        )


if __name__ == "__main__":
    main()
