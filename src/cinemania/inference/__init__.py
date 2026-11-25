"""Inference functions and CLI entry point for trained Cinemania models."""

from cinemania.inference.predict import (
    clear_inference_cache,
    predict_sentiment_rnn,
    predict_sentiment_transformer,
)

__all__ = [
    "clear_inference_cache",
    "predict_sentiment_rnn",
    "predict_sentiment_transformer",
]
