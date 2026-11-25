"""End-to-end reproducible experiment runner for Cinemania."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

from cinemania.config import (
    CLEANED_DATA_PATH,
    COMPARISON_REPORT_PATH,
    EXAMPLE_PREDICTIONS_PATH,
    GLOBAL_SEED,
    METRICS_PATH,
    RAW_DATA_DIR,
    RNN_ARTIFACTS_DIR,
    SPLITS_DIR,
    TRANSFORMER_CHECKPOINT_DIR,
    TRANSFORMER_TRAINER_DIR,
    configure_logging,
    get_device,
    set_global_seeds,
)
from cinemania.data.loading import load_raw_dataset
from cinemania.data.preparation import (
    clean_reviews,
    load_prepared_splits,
    prepare_datasets,
    save_cleaned_dataset,
    save_prepared_splits,
)
from cinemania.evaluation import (
    format_example_predictions_table,
    generate_comparison_markdown,
    save_example_predictions_csv,
    save_metrics_json,
)
from cinemania.inference.predict import clear_inference_cache, evaluate_example_reviews
from cinemania.training.train_rnn import train_and_evaluate_rnn
from cinemania.training.train_transformer import train_and_evaluate_transformer

logger = logging.getLogger(__name__)


def run_prepare_stage() -> dict[str, Any]:
    """Load raw dataset, clean and persist cleaned_data.csv, and save shared splits."""
    raw_df, dataset_path = load_raw_dataset(raw_dir=RAW_DATA_DIR)
    cleaned_df, cleaning_stats = clean_reviews(raw_df, deduplicate=True)
    save_cleaned_dataset(cleaned_df, CLEANED_DATA_PATH)
    splits = prepare_datasets(cleaned_df)
    rel_dataset_path = (
        str(dataset_path.relative_to(dataset_path.parents[2]))
        if len(dataset_path.parents) > 2
        else str(dataset_path)
    )
    save_prepared_splits(
        splits,
        cleaning_stats=cleaning_stats,
        splits_dir=SPLITS_DIR,
        dataset_path=rel_dataset_path,
    )

    return {
        "dataset_path": rel_dataset_path,
        "cleaned_data_path": str(CLEANED_DATA_PATH),
        "cleaning": asdict(cleaning_stats),
        "splits": splits.split_summary(),
    }


def _load_split_metadata() -> dict[str, Any]:
    metadata_path = SPLITS_DIR / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Split metadata not found at '{metadata_path}'. Run prepare stage first."
        )
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def run_all(stage: str = "all") -> dict[str, Any]:
    """Execute the full Cinemania pipeline or an individual stage."""
    configure_logging()
    set_global_seeds(GLOBAL_SEED)
    device = get_device()
    logger.info("Running Cinemania experiment (stage=%s, device=%s, seed=%d)", stage, device, GLOBAL_SEED)

    state: dict[str, Any] = {}
    if stage in ("all", "prepare"):
        state.update(run_prepare_stage())
        if stage == "prepare":
            return state
    else:
        meta = _load_split_metadata()
        state["dataset_path"] = meta.get("dataset_path", str(RAW_DATA_DIR / "train.xlsx"))
        state["cleaning"] = meta.get("cleaning_stats", {})
        state["splits"] = meta["split_summary"]

    splits = load_prepared_splits(SPLITS_DIR)

    rnn_results: dict[str, Any] | None = None
    transformer_results: dict[str, Any] | None = None

    if stage in ("all", "rnn"):
        rnn_results = train_and_evaluate_rnn(
            splits.rnn_train_df,
            splits.shared_test_df,
            artifact_dir=RNN_ARTIFACTS_DIR,
            seed=GLOBAL_SEED,
        )
        if stage == "rnn":
            return rnn_results

    if stage in ("all", "transformer"):
        transformer_results = train_and_evaluate_transformer(
            splits.transformer_train_ds,
            splits.shared_test_ds,
            checkpoint_dir=TRANSFORMER_CHECKPOINT_DIR,
            trainer_output_dir=TRANSFORMER_TRAINER_DIR,
            seed=GLOBAL_SEED,
        )
        if stage == "transformer":
            return transformer_results

    # Load previous stage results (or existing metrics.json) when running evaluation stage alone
    if stage == "evaluate":
        rnn_stage_path = RNN_ARTIFACTS_DIR / "train_results.json"
        tf_stage_path = TRANSFORMER_TRAINER_DIR / "train_results.json"
        if rnn_stage_path.exists() and tf_stage_path.exists():
            rnn_results = json.loads(rnn_stage_path.read_text(encoding="utf-8"))
            transformer_results = json.loads(tf_stage_path.read_text(encoding="utf-8"))
        elif METRICS_PATH.exists():
            existing = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
            rnn_results = existing["rnn"]
            transformer_results = existing["transformer"]
            state["dataset_path"] = existing["dataset_path"]
            state["cleaning"] = existing["cleaning"]
            state["splits"] = existing["splits"]
        else:
            raise FileNotFoundError(
                "Training results not found in artifacts or reports/metrics.json. Run training stages first."
            )

    assert rnn_results is not None and transformer_results is not None

    # Always run example inference from saved disk checkpoints
    clear_inference_cache()
    example_rows = evaluate_example_reviews(
        rnn_dir=RNN_ARTIFACTS_DIR,
        transformer_dir=TRANSFORMER_CHECKPOINT_DIR,
    )
    save_example_predictions_csv(example_rows, EXAMPLE_PREDICTIONS_PATH)

    print("\n=== Required Example Review Predictions ===")
    print(format_example_predictions_table(example_rows))
    print()

    metrics_payload = {
        "seed": GLOBAL_SEED,
        "device": str(device),
        "dataset_path": state["dataset_path"],
        "cleaned_data_path": str(Path("data") / "preproccessed" / "cleaned_data.csv"),
        "cleaning": state["cleaning"],
        "splits": state["splits"],
        "rnn": rnn_results,
        "transformer": transformer_results,
        "example_predictions": example_rows,
    }
    save_metrics_json(metrics_payload, METRICS_PATH)
    generate_comparison_markdown(metrics_payload, COMPARISON_REPORT_PATH)
    logger.info("Experiment complete. Reports saved to '%s' and '%s'.", METRICS_PATH, COMPARISON_REPORT_PATH)
    return metrics_payload


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for running the Cinemania experiment."""
    parser = argparse.ArgumentParser(description="Run the Cinemania IMDb sentiment classification experiment.")
    parser.add_argument(
        "--stage",
        choices=["all", "prepare", "rnn", "transformer", "evaluate"],
        default="all",
        help="Pipeline stage to execute (default: all).",
    )
    args = parser.parse_args(argv)
    run_all(stage=args.stage)


if __name__ == "__main__":
    main()
