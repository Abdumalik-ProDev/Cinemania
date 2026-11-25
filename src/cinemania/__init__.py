"""Cinemania IMDb review sentiment classification package."""

from __future__ import annotations

from typing import Any

from cinemania.config import ID2LABEL, LABEL2ID

__all__ = [
    "ID2LABEL",
    "LABEL2ID",
    "predict_sentiment_rnn",
    "predict_sentiment_transformer",
]


def __getattr__(name: str) -> Any:
    """Lazily expose inference helpers without importing torch/transformers on package import."""
    if name in ("predict_sentiment_rnn", "predict_sentiment_transformer"):
        from cinemania.inference import predict

        return getattr(predict, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
