"""Unit tests for RNN and Transformer inference functions."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from cinemania.inference.predict import (
    clear_inference_cache,
    predict_sentiment_rnn,
    predict_sentiment_transformer,
)
from conftest import make_tiny_rnn_artifacts, make_tiny_transformer_checkpoint


def test_inference_validation_and_missing_checkpoints(tmp_path: Path) -> None:
    clear_inference_cache()
    for bad_input in ("   ", "", None, 123):
        with pytest.raises(ValueError, match="non-empty string"):
            predict_sentiment_rnn(bad_input, artifact_dir=tmp_path)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="non-empty string"):
            predict_sentiment_transformer(bad_input, checkpoint_dir=tmp_path)  # type: ignore[arg-type]

    with pytest.raises(FileNotFoundError):
        predict_sentiment_rnn("Great movie!", artifact_dir=tmp_path / "missing_rnn")
    with pytest.raises(FileNotFoundError):
        predict_sentiment_transformer("Great movie!", checkpoint_dir=tmp_path / "missing_tf")


def test_predict_rnn_and_transformer_from_saved_artifacts(tmp_path: Path) -> None:
    clear_inference_cache()
    rnn_dir = tmp_path / "rnn"
    tf_dir = tmp_path / "transformer"
    make_tiny_rnn_artifacts(rnn_dir)
    make_tiny_transformer_checkpoint(tf_dir, num_layers=1)

    rnn_res = predict_sentiment_rnn("Excellent film!", artifact_dir=rnn_dir, device="cpu")
    assert rnn_res["label"] in {"positive", "negative"}
    assert math.isfinite(rnn_res["confidence"])
    assert math.isclose(
        rnn_res["probabilities"]["positive"] + rnn_res["probabilities"]["negative"],
        1.0,
        rel_tol=1e-5,
    )

    tf_res = predict_sentiment_transformer("Boring film!", checkpoint_dir=tf_dir, device="cpu", max_length=32)
    assert tf_res["label"] in {"positive", "negative"}
    assert math.isfinite(tf_res["confidence"])
    assert math.isclose(
        tf_res["probabilities"]["positive"] + tf_res["probabilities"]["negative"],
        1.0,
        rel_tol=1e-5,
    )
