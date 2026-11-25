"""Data cleaning, deduplication, persistence, and shared train/test splitting."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
from datasets import ClassLabel, Dataset

from cinemania.config import (
    CLEANED_DATA_PATH,
    KNOWN_SENTIMENT_ALIASES,
    LABEL2ID,
    POOL_PER_CLASS,
    POOL_SEED,
    RNN_SAMPLE_SEED,
    RNN_TRAIN_PER_CLASS,
    SPLIT_SEED,
    SPLITS_DIR,
    TEST_SIZE,
    TRANSFORMER_TRAIN_SIZE,
    VALID_SENTIMENTS,
)

logger = logging.getLogger(__name__)


@dataclass
class CleaningStats:
    """Audit trail of dataset cleaning and deduplication."""

    raw_rows: int
    raw_label_counts: dict[str, int]
    dropped_missing_or_empty_reviews: int
    dropped_missing_sentiments: int
    canonicalized_aliases: dict[str, int]
    duplicate_pair_rows: int
    duplicate_review_rows: int
    deduplicated: bool
    cleaned_rows: int
    cleaned_class_distribution: dict[str, int]


@dataclass
class PreparedSplits:
    """Reproducible datasets with a single shared 500-row held-out test set."""

    pool_df: pd.DataFrame
    transformer_train_df: pd.DataFrame
    rnn_train_df: pd.DataFrame
    shared_test_df: pd.DataFrame
    transformer_train_ds: Dataset
    shared_test_ds: Dataset

    def split_summary(self) -> dict[str, object]:
        """Return counts and class distributions for reporting."""
        return {
            "pool_total": len(self.pool_df),
            "pool_distribution": self.pool_df["sentiment"].value_counts().to_dict(),
            "transformer_train_total": len(self.transformer_train_df),
            "transformer_train_distribution": self.transformer_train_df["sentiment"].value_counts().to_dict(),
            "rnn_train_total": len(self.rnn_train_df),
            "rnn_train_distribution": self.rnn_train_df["sentiment"].value_counts().to_dict(),
            "shared_test_total": len(self.shared_test_df),
            "shared_test_distribution": self.shared_test_df["sentiment"].value_counts().to_dict(),
        }


def clean_reviews(df: pd.DataFrame, deduplicate: bool = True) -> tuple[pd.DataFrame, CleaningStats]:
    """Validate, clean, normalize labels, and optionally deduplicate reviews."""
    if not {"review", "sentiment"}.issubset(df.columns):
        raise ValueError(f"Expected columns ['review', 'sentiment'], got {list(df.columns)}.")

    working = df[["review", "sentiment"]].copy()
    raw_rows = len(working)
    raw_label_counts = {
        str(k): int(v) for k, v in working["sentiment"].fillna("<NA>").astype(str).value_counts().items()
    }
    logger.info("Before cleaning: %d rows; raw sentiment counts: %s", raw_rows, raw_label_counts)

    # Drop missing or empty/whitespace-only reviews without altering valid review text
    missing_review_mask = working["review"].isna() | (working["review"].astype(str).str.strip() == "")
    dropped_reviews = int(missing_review_mask.sum())
    working = working.loc[~missing_review_mask].copy()

    missing_sentiment_mask = working["sentiment"].isna()
    dropped_sentiments = int(missing_sentiment_mask.sum())
    if dropped_sentiments > 0:
        working = working.loc[~missing_sentiment_mask].copy()

    # Preserve original review text as string
    working["review"] = working["review"].astype(str)

    # Normalize sentiment strings
    working["sentiment"] = working["sentiment"].astype(str).str.strip().str.lower()

    # Explicitly canonicalize known IMDb short labels ('pos'/'neg') and record counts
    alias_counts: dict[str, int] = {}
    for alias, target in KNOWN_SENTIMENT_ALIASES.items():
        count = int((working["sentiment"] == alias).sum())
        if count > 0:
            alias_counts[f"{alias}->{target}"] = count
    if alias_counts:
        logger.info("Canonicalizing known sentiment aliases: %s", alias_counts)
        working["sentiment"] = working["sentiment"].replace(KNOWN_SENTIMENT_ALIASES)

    # Reject any unexpected sentiment labels loudly
    observed_labels = set(working["sentiment"].unique())
    unexpected = sorted(observed_labels - VALID_SENTIMENTS)
    if unexpected:
        raise ValueError(
            f"Unexpected sentiment label(s) found after normalization: {unexpected}. "
            f"Only {sorted(VALID_SENTIMENTS)} (and aliases {KNOWN_SENTIMENT_ALIASES}) are accepted."
        )

    duplicate_pair_rows = int(working.duplicated(subset=["review", "sentiment"]).sum())
    duplicate_review_rows = int(working.duplicated(subset=["review"]).sum())
    if duplicate_review_rows > duplicate_pair_rows:
        logger.warning(
            "Found %d conflicting-sentiment duplicate review(s); keeping first occurrence when deduplicating.",
            duplicate_review_rows - duplicate_pair_rows,
        )
    logger.info(
        "Duplicate check: %d duplicate (review, sentiment) rows, %d duplicate review texts.",
        duplicate_pair_rows,
        duplicate_review_rows,
    )

    if deduplicate and duplicate_review_rows > 0:
        working = working.drop_duplicates(subset=["review"], keep="first").reset_index(drop=True)
        logger.info("Deduplicated by review text (keep='first'): %d unique reviews remain.", len(working))
    else:
        working = working.reset_index(drop=True)

    cleaned_dist = {str(k): int(v) for k, v in working["sentiment"].value_counts().items()}
    logger.info("After cleaning: %d rows; class distribution: %s", len(working), cleaned_dist)

    stats = CleaningStats(
        raw_rows=raw_rows,
        raw_label_counts=raw_label_counts,
        dropped_missing_or_empty_reviews=dropped_reviews,
        dropped_missing_sentiments=dropped_sentiments,
        canonicalized_aliases=alias_counts,
        duplicate_pair_rows=duplicate_pair_rows,
        duplicate_review_rows=duplicate_review_rows,
        deduplicated=bool(deduplicate and duplicate_review_rows > 0),
        cleaned_rows=len(working),
        cleaned_class_distribution=cleaned_dist,
    )
    return working, stats


def save_cleaned_dataset(cleaned_df: pd.DataFrame, path: Path | str = CLEANED_DATA_PATH) -> Path:
    """Persist the cleaned dataset to CSV (e.g., data/preproccessed/cleaned_data.csv)."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df[["review", "sentiment"]].to_csv(output_path, index=False)
    logger.info("Saved cleaned dataset (%d rows) to '%s'.", len(cleaned_df), output_path)
    return output_path


def prepare_datasets(cleaned_df: pd.DataFrame) -> PreparedSplits:
    """Create the 3,000-row balanced pool, 2,500/500 Transformer split, and 1,500/500 RNN split."""
    if cleaned_df["review"].duplicated().any():
        raise ValueError(
            "cleaned_df contains duplicate reviews; deduplicate before creating the 3,000-review unique pool."
        )

    pos_df = cleaned_df[cleaned_df["sentiment"] == "positive"]
    neg_df = cleaned_df[cleaned_df["sentiment"] == "negative"]
    if len(pos_df) < POOL_PER_CLASS or len(neg_df) < POOL_PER_CLASS:
        raise ValueError(
            f"Insufficient unique reviews to sample {POOL_PER_CLASS} per class: "
            f"positive={len(pos_df)}, negative={len(neg_df)}."
        )

    pos_sample = pos_df.sample(n=POOL_PER_CLASS, random_state=POOL_SEED)
    neg_sample = neg_df.sample(n=POOL_PER_CLASS, random_state=POOL_SEED)
    pool_df = (
        pd.concat([pos_sample, neg_sample], ignore_index=True)
        .sample(frac=1.0, random_state=POOL_SEED)
        .reset_index(drop=True)
    )
    pool_df["row_id"] = range(len(pool_df))
    pool_df["label"] = pool_df["sentiment"].map(LABEL2ID).astype(int)

    # Convert to Hugging Face Dataset with ClassLabel for stratified split
    hf_dataset = Dataset.from_pandas(pool_df[["row_id", "review", "sentiment", "label"]], preserve_index=False)
    class_label = ClassLabel(names=["negative", "positive"])
    hf_dataset = hf_dataset.cast_column("label", class_label)

    split_dict = hf_dataset.train_test_split(
        test_size=TEST_SIZE,
        seed=SPLIT_SEED,
        stratify_by_column="label",
    )
    transformer_train_ds = split_dict["train"]
    shared_test_ds = split_dict["test"]

    transformer_train_df = transformer_train_ds.to_pandas()
    shared_test_df = shared_test_ds.to_pandas()

    # Sample 750 positive + 750 negative from Transformer training partition for RNN
    rnn_pos = transformer_train_df[transformer_train_df["sentiment"] == "positive"].sample(
        n=RNN_TRAIN_PER_CLASS, random_state=RNN_SAMPLE_SEED
    )
    rnn_neg = transformer_train_df[transformer_train_df["sentiment"] == "negative"].sample(
        n=RNN_TRAIN_PER_CLASS, random_state=RNN_SAMPLE_SEED
    )
    rnn_train_df = (
        pd.concat([rnn_pos, rnn_neg], ignore_index=True)
        .sample(frac=1.0, random_state=RNN_SAMPLE_SEED)
        .reset_index(drop=True)
    )

    # Strict validation of counts and train/test isolation
    assert len(pool_df) == POOL_PER_CLASS * 2
    assert pool_df["review"].nunique() == POOL_PER_CLASS * 2
    assert len(transformer_train_df) == TRANSFORMER_TRAIN_SIZE
    assert len(shared_test_df) == TEST_SIZE
    assert len(rnn_train_df) == RNN_TRAIN_PER_CLASS * 2

    test_ids = set(shared_test_df["row_id"])
    test_texts = set(shared_test_df["review"])
    assert test_ids.isdisjoint(set(transformer_train_df["row_id"]))
    assert test_ids.isdisjoint(set(rnn_train_df["row_id"]))
    assert test_texts.isdisjoint(set(transformer_train_df["review"]))
    assert test_texts.isdisjoint(set(rnn_train_df["review"]))

    splits = PreparedSplits(
        pool_df=pool_df,
        transformer_train_df=transformer_train_df,
        rnn_train_df=rnn_train_df,
        shared_test_df=shared_test_df,
        transformer_train_ds=transformer_train_ds,
        shared_test_ds=shared_test_ds,
    )
    logger.info("Prepared splits: %s", splits.split_summary())
    return splits


def save_prepared_splits(
    splits: PreparedSplits,
    cleaning_stats: CleaningStats | None = None,
    splits_dir: Path | str = SPLITS_DIR,
    dataset_path: str | None = None,
) -> Path:
    """Persist prepared splits to disk so standalone stages reuse identical partitions."""
    target_dir = Path(splits_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    splits.pool_df.to_parquet(target_dir / "pool.parquet", index=False)
    splits.transformer_train_df.to_parquet(target_dir / "transformer_train.parquet", index=False)
    splits.rnn_train_df.to_parquet(target_dir / "rnn_train.parquet", index=False)
    splits.shared_test_df.to_parquet(target_dir / "shared_test.parquet", index=False)

    metadata: dict[str, object] = {"split_summary": splits.split_summary()}
    if dataset_path is not None:
        metadata["dataset_path"] = dataset_path
    if cleaning_stats is not None:
        metadata["cleaning_stats"] = asdict(cleaning_stats)
    (target_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info("Saved prepared splits to '%s'.", target_dir)
    return target_dir


def load_prepared_splits(splits_dir: Path | str = SPLITS_DIR) -> PreparedSplits:
    """Reload persisted splits from disk."""
    target_dir = Path(splits_dir)
    pool_df = pd.read_parquet(target_dir / "pool.parquet")
    transformer_train_df = pd.read_parquet(target_dir / "transformer_train.parquet")
    rnn_train_df = pd.read_parquet(target_dir / "rnn_train.parquet")
    shared_test_df = pd.read_parquet(target_dir / "shared_test.parquet")

    class_label = ClassLabel(names=["negative", "positive"])
    transformer_train_ds = Dataset.from_pandas(
        transformer_train_df[["row_id", "review", "sentiment", "label"]], preserve_index=False
    ).cast_column("label", class_label)
    shared_test_ds = Dataset.from_pandas(
        shared_test_df[["row_id", "review", "sentiment", "label"]], preserve_index=False
    ).cast_column("label", class_label)

    return PreparedSplits(
        pool_df=pool_df,
        transformer_train_df=transformer_train_df,
        rnn_train_df=rnn_train_df,
        shared_test_df=shared_test_df,
        transformer_train_ds=transformer_train_ds,
        shared_test_ds=shared_test_ds,
    )
