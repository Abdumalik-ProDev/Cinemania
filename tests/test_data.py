"""Unit tests for data discovery, cleaning, persistence, and balanced splitting."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from cinemania.data.loading import discover_dataset_file, load_raw_dataset
from cinemania.data.preparation import clean_reviews, prepare_datasets, save_cleaned_dataset
from cinemania.evaluation import (
    generate_comparison_markdown,
    save_example_predictions_csv,
    save_metrics_json,
)


def _make_synthetic_df(n_per_class: int = 1600) -> pd.DataFrame:
    pos_reviews = [f"Great movie number {i} with awesome acting." for i in range(n_per_class)]
    neg_reviews = [f"Terrible film number {i} with boring pacing." for i in range(n_per_class)]
    return pd.DataFrame(
        {
            "review": pos_reviews + neg_reviews,
            "sentiment": ["positive"] * n_per_class + ["negative"] * n_per_class,
        }
    )


def test_discover_and_load_csv_and_xlsx(tmp_path: Path) -> None:
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        discover_dataset_file(empty_dir)

    csv_dir = tmp_path / "csv_dir"
    csv_dir.mkdir()
    csv_path = csv_dir / "train.csv"
    pd.DataFrame({" Reviews ": ["Loved it!"], " Sentiment ": ["positive"], "extra": [1]}).to_csv(
        csv_path, index=False
    )

    found = discover_dataset_file(csv_dir)
    assert found == csv_path

    df, loaded_path = load_raw_dataset(raw_dir=csv_dir)
    assert loaded_path == csv_path
    assert list(df.columns) == ["review", "sentiment"]
    assert df.loc[0, "review"] == "Loved it!"

    xlsx_dir = tmp_path / "xlsx_dir"
    xlsx_dir.mkdir()
    xlsx_path = xlsx_dir / "train.xlsx"
    pd.DataFrame({"Reviews": ["Hated it."], "Sentiment": ["neg"]}).to_excel(xlsx_path, index=False)
    df_xlsx, _ = load_raw_dataset(raw_dir=xlsx_dir)
    assert list(df_xlsx.columns) == ["review", "sentiment"]
    assert df_xlsx.loc[0, "sentiment"] == "neg"


def test_load_missing_columns_raises(tmp_path: Path) -> None:
    bad_csv = tmp_path / "bad.csv"
    pd.DataFrame({"text": ["hello"], "label": ["positive"]}).to_csv(bad_csv, index=False)
    with pytest.raises(ValueError, match="missing required column"):
        load_raw_dataset(path=bad_csv)


def test_clean_reviews_rules_and_reject_unknown_labels(tmp_path: Path) -> None:
    raw = pd.DataFrame(
        {
            "review": [
                "Original   Spacing & Punctuation!",
                None,
                "   ",
                "Another good one",
                "Original   Spacing & Punctuation!",
            ],
            "sentiment": [" POS ", "positive", "negative", "neg", "pos"],
        }
    )
    cleaned, stats = clean_reviews(raw, deduplicate=True)
    assert stats.raw_rows == 5
    assert stats.dropped_missing_or_empty_reviews == 2
    assert stats.duplicate_review_rows == 1
    assert len(cleaned) == 2
    # Original review text is preserved untouched
    assert cleaned.loc[0, "review"] == "Original   Spacing & Punctuation!"
    assert set(cleaned["sentiment"]) == {"positive", "negative"}

    saved_path = save_cleaned_dataset(cleaned, tmp_path / "preproccessed" / "cleaned_data.csv")
    assert saved_path.exists()
    reloaded = pd.read_csv(saved_path)
    assert list(reloaded.columns) == ["review", "sentiment"]
    assert len(reloaded) == 2

    invalid = pd.DataFrame({"review": ["Ok film"], "sentiment": ["neutral"]})
    with pytest.raises(ValueError, match="Unexpected sentiment"):
        clean_reviews(invalid)


def test_prepare_datasets_sizes_balance_and_no_leakage() -> None:
    df = _make_synthetic_df(1600)
    splits = prepare_datasets(df)

    assert len(splits.pool_df) == 3000
    assert splits.pool_df["review"].nunique() == 3000
    assert splits.pool_df["sentiment"].value_counts().to_dict() == {"positive": 1500, "negative": 1500}

    assert len(splits.transformer_train_df) == 2500
    assert len(splits.shared_test_df) == 500
    assert len(splits.rnn_train_df) == 1500
    assert splits.rnn_train_df["sentiment"].value_counts().to_dict() == {"positive": 750, "negative": 750}

    # Verify zero train/test overlap for both models
    test_ids = set(splits.shared_test_df["row_id"])
    test_reviews = set(splits.shared_test_df["review"])
    assert test_ids.isdisjoint(set(splits.transformer_train_df["row_id"]))
    assert test_ids.isdisjoint(set(splits.rnn_train_df["row_id"]))
    assert test_reviews.isdisjoint(set(splits.transformer_train_df["review"]))
    assert test_reviews.isdisjoint(set(splits.rnn_train_df["review"]))


def test_report_writers_generate_valid_outputs(tmp_path: Path) -> None:
    payload = {
        "dataset_path": "data/raw/train.xlsx",
        "cleaning": {"raw_rows": 100, "duplicate_review_rows": 2, "cleaned_rows": 98},
        "splits": {
            "pool_total": 30,
            "pool_distribution": {"positive": 15, "negative": 15},
            "rnn_train_total": 16,
            "rnn_train_distribution": {"positive": 8, "negative": 8},
            "transformer_train_total": 24,
            "transformer_train_distribution": {"positive": 12, "negative": 12},
            "shared_test_total": 6,
        },
        "rnn": {
            "epoch_losses": [{"epoch": 1, "train_loss": 0.5}],
            "test_metrics": {"loss": 0.6, "accuracy": 0.65, "precision": 0.66, "recall": 0.64, "f1": 0.65},
        },
        "transformer": {
            "final_train_loss": 0.2,
            "test_metrics": {"loss": 0.3, "accuracy": 0.9, "precision": 0.91, "recall": 0.89, "f1": 0.9},
        },
        "example_predictions": [
            {
                "category": "positive-oriented",
                "review": "Great cast.",
                "rnn_prediction": "positive",
                "rnn_confidence": 0.8,
                "transformer_prediction": "positive",
                "transformer_confidence": 0.99,
            }
        ],
    }
    metrics_path = save_metrics_json(payload, tmp_path / "metrics.json")
    csv_path = save_example_predictions_csv(payload["example_predictions"], tmp_path / "examples.csv")
    md_path = generate_comparison_markdown(payload, tmp_path / "comparison.md")

    assert metrics_path.exists()
    assert csv_path.exists()
    md_text = md_path.read_text(encoding="utf-8")
    assert "16 reviews** (8 positive, 8 negative)" in md_text
    assert "24 reviews** (12 positive, 12 negative)" in md_text
