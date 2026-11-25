"""Dataset discovery and raw file loading for Cinemania."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from cinemania.config import RAW_DATA_DIR

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = (".xlsx", ".csv")
COLUMN_ALIASES = {"reviews": "review"}


def discover_dataset_file(raw_dir: Path | str = RAW_DATA_DIR) -> Path:
    """Locate a supported dataset file (.xlsx or .csv) inside raw_dir."""
    directory = Path(raw_dir)
    if not directory.exists() or not directory.is_dir():
        raise FileNotFoundError(
            f"Raw data directory not found: '{directory}'. "
            f"Expected a directory containing a {', '.join(SUPPORTED_EXTENSIONS)} dataset."
        )

    candidates = sorted(
        [
            p
            for p in directory.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS and not p.name.startswith("~$")
        ],
        key=lambda p: p.name.lower(),
    )
    if not candidates:
        raise FileNotFoundError(
            f"No supported dataset file found in '{directory}'. "
            f"Supported formats: {', '.join(SUPPORTED_EXTENSIONS)}."
        )

    preferred_names = ("train.xlsx", "train.csv")
    for pref in preferred_names:
        for candidate in candidates:
            if candidate.name.lower() == pref:
                return candidate

    return candidates[0]


def load_raw_dataset(
    path: Path | str | None = None,
    raw_dir: Path | str = RAW_DATA_DIR,
) -> tuple[pd.DataFrame, Path]:
    """Load raw dataset from Excel or CSV and validate required columns."""
    dataset_path = Path(path) if path is not None else discover_dataset_file(raw_dir)
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset file does not exist: '{dataset_path}'. "
            f"Supported formats in '{RAW_DATA_DIR}': {', '.join(SUPPORTED_EXTENSIONS)}."
        )

    suffix = dataset_path.suffix.lower()
    if suffix == ".xlsx":
        df = pd.read_excel(dataset_path, engine="openpyxl")
    elif suffix == ".csv":
        df = pd.read_csv(dataset_path)
    else:
        raise ValueError(
            f"Unsupported dataset format '{suffix}' for '{dataset_path}'. "
            f"Supported formats: {', '.join(SUPPORTED_EXTENSIONS)}."
        )

    original_columns = list(df.columns)
    normalized_columns = [str(col).strip().lower() for col in df.columns]
    df.columns = normalized_columns

    # Explicitly normalize known column aliases (e.g. 'reviews' -> 'review') if needed
    for alias, canonical in COLUMN_ALIASES.items():
        if canonical not in df.columns and alias in df.columns:
            logger.info("Normalizing column name alias '%s' -> '%s' (source columns: %s)", alias, canonical, original_columns)
            df = df.rename(columns={alias: canonical})

    required = {"review", "sentiment"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Dataset '{dataset_path}' is missing required column(s): {sorted(missing)}. "
            f"Found columns (normalized): {list(df.columns)} (original: {original_columns})."
        )

    selected = df[["review", "sentiment"]].copy()
    logger.info("Loaded raw dataset from '%s' with %d rows.", dataset_path, len(selected))
    return selected, dataset_path
