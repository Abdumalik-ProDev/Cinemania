"""Centralized configuration, paths, hyperparameters, and reproducibility helpers."""

from __future__ import annotations

import logging
import random
from pathlib import Path

import numpy as np
import torch

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PREPROCESSED_DATA_DIR = DATA_DIR / "preproccessed"
CLEANED_DATA_PATH = PREPROCESSED_DATA_DIR / "cleaned_data.csv"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
SPLITS_DIR = ARTIFACTS_DIR / "splits"
RNN_ARTIFACTS_DIR = ARTIFACTS_DIR / "rnn"
RNN_VOCAB_PATH = RNN_ARTIFACTS_DIR / "vocab.json"

TRANSFORMER_ARTIFACTS_DIR = ARTIFACTS_DIR / "transformer"
TRANSFORMER_CHECKPOINT_DIR = TRANSFORMER_ARTIFACTS_DIR / "checkpoint"
TRANSFORMER_TRAINER_DIR = TRANSFORMER_ARTIFACTS_DIR / "trainer_output"

REPORTS_DIR = PROJECT_ROOT / "reports"
METRICS_PATH = REPORTS_DIR / "metrics.json"
COMPARISON_REPORT_PATH = REPORTS_DIR / "comparison.md"
EXAMPLE_PREDICTIONS_PATH = REPORTS_DIR / "example_predictions.csv"

# Label mapping (shared across RNN and Transformer)
LABEL2ID: dict[str, int] = {"negative": 0, "positive": 1}
ID2LABEL: dict[int, str] = {0: "negative", 1: "positive"}
KNOWN_SENTIMENT_ALIASES: dict[str, str] = {"pos": "positive", "neg": "negative"}
VALID_SENTIMENTS: set[str] = set(LABEL2ID.keys())

# Reproducibility seeds
GLOBAL_SEED = 42
POOL_SEED = 101
SPLIT_SEED = 123
RNN_SAMPLE_SEED = 101

# Dataset sizes
POOL_PER_CLASS = 1500
TEST_SIZE = 500
TRANSFORMER_TRAIN_SIZE = 2500
RNN_TRAIN_PER_CLASS = 750

# RNN vocabulary & architecture hyperparameters
MAX_VOCAB_TOKENS = 4000
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
PAD_IDX = 0
UNK_IDX = 1
MAX_SEQ_LENGTH = 256
RNN_EMBEDDING_DIM = 128
RNN_HIDDEN_DIM = 256
RNN_BATCH_SIZE = 32
RNN_LR = 0.001
RNN_EPOCHS = 10

# Transformer hyperparameters
TRANSFORMER_MODEL_NAME = "distilbert-base-uncased"
TRANSFORMER_MAX_LENGTH = 256
TRANSFORMER_EPOCHS = 3
TRANSFORMER_TRAIN_BATCH_SIZE = 8
TRANSFORMER_EVAL_BATCH_SIZE = 16
TRANSFORMER_LOGGING_STEPS = 100

# Required example reviews (Section 11)
EXAMPLE_REVIEWS: list[dict[str, str]] = [
    {
        "category": "negative-oriented",
        "review": "I had been looking forward to this film for a long time, but I did not like it at all.",
    },
    {
        "category": "positive-oriented",
        "review": "The cast is excellent, and the plot is very interesting.",
    },
    {
        "category": "mixed",
        "review": "The film's visual effects are good, but the story is weak.",
    },
]


def set_global_seeds(seed: int = GLOBAL_SEED) -> None:
    """Set deterministic seeds for Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Return CUDA device when available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def configure_logging(level: int = logging.INFO) -> None:
    """Configure consistent package-wide logging."""
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )
    else:
        root_logger.setLevel(level)
