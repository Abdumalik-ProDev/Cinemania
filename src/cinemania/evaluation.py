"""Evaluation metrics, comparison reporting, and CSV/JSON output helpers."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch import nn
from torch.utils.data import DataLoader

from cinemania.config import (
    COMPARISON_REPORT_PATH,
    EXAMPLE_PREDICTIONS_PATH,
    METRICS_PATH,
)

logger = logging.getLogger(__name__)


def compute_binary_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    """Compute accuracy, precision, recall, and F1 for binary sentiment classification."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
    }


def evaluate_rnn_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate SentimentRNN in eval mode with gradients disabled and sample-weighted loss."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    y_true: list[int] = []
    y_pred: list[int] = []

    with torch.no_grad():
        for input_ids, lengths, labels in dataloader:
            input_ids = input_ids.to(device)
            labels_col = labels.to(device).unsqueeze(1)
            probs = model(input_ids, lengths)
            loss = criterion(probs, labels_col)

            batch_size = input_ids.size(0)
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size

            preds = (probs.squeeze(1) >= 0.5).to(torch.int64).cpu().tolist()
            y_pred.extend(preds)
            y_true.extend(labels.to(torch.int64).cpu().tolist())

    if total_samples == 0:
        raise ValueError("Evaluation dataloader is empty.")

    metrics = compute_binary_metrics(y_true, y_pred)
    metrics["loss"] = total_loss / total_samples
    return metrics


def save_metrics_json(metrics_payload: dict[str, Any], path: Path | str = METRICS_PATH) -> Path:
    """Save machine-readable metrics and experiment metadata to JSON."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    logger.info("Saved metrics JSON to '%s'.", output_path)
    return output_path


def save_example_predictions_csv(
    rows: list[dict[str, Any]],
    path: Path | str = EXAMPLE_PREDICTIONS_PATH,
) -> Path:
    """Save the example predictions table to CSV."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    logger.info("Saved example predictions CSV to '%s'.", output_path)
    return output_path


def format_example_predictions_table(rows: list[dict[str, Any]]) -> str:
    """Format example predictions as a readable Markdown table."""
    lines = [
        "| Category | Review | RNN Prediction | RNN Confidence | Transformer Prediction | Transformer Confidence |",
        "|---|---|---|---:|---|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['category']} | {row['review']} | {row['rnn_prediction']} | "
            f"{row['rnn_confidence']:.4f} | {row['transformer_prediction']} | {row['transformer_confidence']:.4f} |"
        )
    return "\n".join(lines)


def _build_conclusion(rnn_eval: dict[str, Any], transformer_eval: dict[str, Any]) -> str:
    """Generate a concise 2-3 sentence evidence-based conclusion from observed metrics."""
    rnn_acc = float(rnn_eval["accuracy"])
    rnn_f1 = float(rnn_eval["f1"])
    tf_acc = float(transformer_eval["accuracy"])
    tf_f1 = float(transformer_eval["f1"])

    if tf_acc > rnn_acc or (tf_acc == rnn_acc and tf_f1 >= rnn_f1):
        winner = "The fine-tuned DistilBERT Transformer"
        runner_up = "the custom SentimentRNN"
        w_acc, w_f1, r_acc, r_f1 = tf_acc, tf_f1, rnn_acc, rnn_f1
    else:
        winner = "The custom SentimentRNN"
        runner_up = "the fine-tuned DistilBERT Transformer"
        w_acc, w_f1, r_acc, r_f1 = rnn_acc, rnn_f1, tf_acc, tf_f1

    return (
        f"{winner} outperformed {runner_up} on the shared 500-review held-out test set, achieving "
        f"{w_acc:.4f} accuracy and {w_f1:.4f} F1 compared to {r_acc:.4f} accuracy and {r_f1:.4f} F1. "
        f"This advantage reflects both DistilBERT's pretrained contextual self-attention representations "
        f"(compared with a randomly initialized single-layer vanilla RNN over a 4,000-word vocabulary) "
        f"and the fact that DistilBERT was trained on 2,500 reviews rather than 1,500. "
        f"Because the training-set sizes differ per the assignment specification, the results demonstrate "
        f"overall pipeline performance on the shared benchmark rather than a purely controlled architectural isolation."
    )


def generate_comparison_markdown(
    metrics_payload: dict[str, Any],
    path: Path | str = COMPARISON_REPORT_PATH,
) -> Path:
    """Write a concise, evidence-based Markdown comparison report."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    splits = metrics_payload["splits"]
    cleaning = metrics_payload["cleaning"]
    rnn = metrics_payload["rnn"]
    transformer = metrics_payload["transformer"]
    examples = metrics_payload["example_predictions"]

    rnn_eval = rnn["test_metrics"]
    tf_eval = transformer["test_metrics"]
    rnn_final_train_loss = rnn["epoch_losses"][-1]["train_loss"]
    tf_train_loss = transformer.get("final_train_loss")
    tf_train_loss_str = f"{tf_train_loss:.4f}" if tf_train_loss is not None else "N/A"

    example_table = format_example_predictions_table(examples)
    conclusion = _build_conclusion(rnn_eval, tf_eval)

    pool_dist = splits.get("pool_distribution", {})
    rnn_dist = splits.get("rnn_train_distribution", {})
    tf_dist = splits.get("transformer_train_distribution", {})

    content = f"""# Cinemania Sentiment Classification — Model Comparison Report

## 1. Experiment Setup & Data Caveats

- **Source dataset:** `{metrics_payload['dataset_path']}` ({cleaning['raw_rows']:,} raw rows; {cleaning['duplicate_review_rows']} duplicate reviews removed; {cleaning['cleaned_rows']:,} unique cleaned reviews).
- **Balanced pool:** {splits['pool_total']:,} unique reviews ({pool_dist.get('positive', 0):,} positive, {pool_dist.get('negative', 0):,} negative; `random_state=101`).
- **Shared held-out test set:** {splits['shared_test_total']} stratified reviews (`seed=123`), never used during vocabulary construction, training, or checkpoint selection.
- **Training-size caveat:** Per specification, `SentimentRNN` trained on **{splits['rnn_train_total']:,} reviews** ({rnn_dist.get('positive', 0):,} positive, {rnn_dist.get('negative', 0):,} negative) while `distilbert-base-uncased` fine-tuned on **{splits['transformer_train_total']:,} reviews** ({tf_dist.get('positive', 0):,} positive, {tf_dist.get('negative', 0):,} negative). Because the training counts differ, this comparison evaluates the two specified workflows on an identical test set rather than isolating model architecture alone.

## 2. Held-Out Test Evaluation (500 Shared Reviews)

> Note: Training loss reflects optimization fit on each model's training partition and does not by itself prove generalization; model comparison is grounded in held-out test metrics below.

| Model | Train Samples | Test Samples | Final Train Loss | Test Loss | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `SentimentRNN` (10 epochs) | {splits['rnn_train_total']:,} | {splits['shared_test_total']} | {rnn_final_train_loss:.4f} | {rnn_eval['loss']:.4f} | {rnn_eval['accuracy']:.4f} | {rnn_eval['precision']:.4f} | {rnn_eval['recall']:.4f} | {rnn_eval['f1']:.4f} |
| `distilbert-base-uncased` (3 epochs) | {splits['transformer_train_total']:,} | {splits['shared_test_total']} | {tf_train_loss_str} | {tf_eval['loss']:.4f} | {tf_eval['accuracy']:.4f} | {tf_eval['precision']:.4f} | {tf_eval['recall']:.4f} | {tf_eval['f1']:.4f} |

## 3. Required Example Predictions

{example_table}

## 4. Conclusion

{conclusion}
"""
    output_path.write_text(content, encoding="utf-8")
    logger.info("Saved comparison report to '%s'.", output_path)
    return output_path
