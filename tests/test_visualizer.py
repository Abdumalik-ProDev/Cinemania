"""Unit tests for the neural network visualizer inspector, HTTP server, and exporter."""

from __future__ import annotations

import json
import math
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from cinemania.inference.predict import clear_inference_cache, predict_sentiment_transformer
from cinemania.visualizer.inspector import (
    clear_visualizer_cache,
    inspect_rnn_review,
    inspect_transformer_review,
)
from cinemania.visualizer.server import create_server, export_static_visualizer
from conftest import make_tiny_rnn_artifacts, make_tiny_transformer_checkpoint


def test_visualizer_rejects_empty_text(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="non-empty string"):
        inspect_rnn_review("   ", artifact_dir=tmp_path)
    with pytest.raises(ValueError, match="non-empty string"):
        inspect_transformer_review("", checkpoint_dir=tmp_path)


def test_inspect_rnn_and_transformer_and_export(tmp_path: Path) -> None:
    clear_visualizer_cache()
    clear_inference_cache()
    rnn_dir = tmp_path / "rnn"
    tf_dir = tmp_path / "transformer"
    make_tiny_rnn_artifacts(rnn_dir)
    make_tiny_transformer_checkpoint(tf_dir, num_layers=2)

    rnn_trace = inspect_rnn_review("Excellent cast!", artifact_dir=rnn_dir, device="cpu")
    assert rnn_trace["tokens"] == ["excellent", "cast"]
    assert len(rnn_trace["steps"]) == 2
    assert len(rnn_trace["weights"]["embed_to_hidden"]) == 16
    assert len(rnn_trace["weights"]["embed_to_hidden"][0]) == 10
    assert rnn_trace["final_prediction"]["label"] in {"positive", "negative"}

    tf_trace = inspect_transformer_review("Excellent cast!", checkpoint_dir=tf_dir, device="cpu")
    seq_len = len(tf_trace["tokens"])
    assert tf_trace["truncated_for_display"] is False
    assert tf_trace["num_layers"] == 2
    assert tf_trace["num_heads"] == 2
    assert len(tf_trace["attentions"]) == 2
    assert len(tf_trace["attentions"][0]["heads"]) == 2
    assert len(tf_trace["attentions"][0]["avg"]) == seq_len
    assert len(tf_trace["attentions"][0]["avg"][0]) == seq_len
    assert len(tf_trace["layer_cls_activity"]) == 3  # embeddings + 2 blocks
    assert "cls_slice" not in tf_trace["layer_cls_activity"][0]
    probs = tf_trace["final_prediction"]["probabilities"]
    assert math.isclose(probs["positive"] + probs["negative"], 1.0, abs_tol=1e-3)

    # Long review (>64 display tokens) still matches full-length inference prediction
    long_review = " ".join(["excellent"] * 75 + ["boring"] * 10)
    long_tf_trace = inspect_transformer_review(long_review, checkpoint_dir=tf_dir, device="cpu")
    assert long_tf_trace["truncated_for_display"] is True
    assert len(long_tf_trace["tokens"]) == 64
    assert long_tf_trace["total_tokens"] > 64
    pred_full = predict_sentiment_transformer(long_review, checkpoint_dir=tf_dir, device="cpu")
    assert long_tf_trace["final_prediction"]["label"] == pred_full["label"]

    out_html = tmp_path / "nn_visualizer.html"
    exported = export_static_visualizer(
        out_html,
        presets=[{"category": "test", "review": "Excellent cast!"}],
        rnn_dir=rnn_dir,
        transformer_dir=tf_dir,
    )
    assert exported.exists()
    content = exported.read_text(encoding="utf-8")
    assert "window.__CINEMANIA_BOOTSTRAP__" in content
    assert "Excellent cast!" in content
    assert ".innerHTML =" not in content


def test_visualizer_http_server_routes_and_validation(tmp_path: Path) -> None:
    clear_visualizer_cache()
    clear_inference_cache()
    rnn_dir = tmp_path / "rnn"
    tf_dir = tmp_path / "transformer"
    make_tiny_rnn_artifacts(rnn_dir)
    make_tiny_transformer_checkpoint(tf_dir, num_layers=2)

    srv = create_server(host="127.0.0.1", port=0, rnn_dir=rnn_dir, transformer_dir=tf_dir)
    port = srv.server_address[1]
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/?v=1") as resp:
            assert resp.status == 200
            assert "Cinemania" in resp.read().decode("utf-8")

        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health") as resp:
            assert resp.status == 200
            assert json.loads(resp.read().decode("utf-8")) == {"status": "ok"}

        req_ok = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/inspect",
            data=json.dumps({"review": "Excellent cast!"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req_ok) as resp:
            assert resp.status == 200
            body = json.loads(resp.read().decode("utf-8"))
            assert len(body["rnn"]["steps"]) == 2

        for bad_payload in (b"not-json", b"[]", json.dumps({"review": "   "}).encode("utf-8")):
            req_bad = urllib.request.Request(
                f"http://127.0.0.1:{port}/api/inspect",
                data=bad_payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with pytest.raises(urllib.error.HTTPError) as exc_info:
                urllib.request.urlopen(req_bad)
            assert exc_info.value.code == 400

        req_big = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/inspect",
            data=b"x" * 70_000,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(req_big)
        assert exc_info.value.code == 413
    finally:
        srv.shutdown()
        srv.server_close()
