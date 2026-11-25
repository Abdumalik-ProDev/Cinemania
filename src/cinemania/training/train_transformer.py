"""Fine-tuning and evaluation pipeline for DistilBERT sequence classification."""

from __future__ import annotations

import inspect
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)

from cinemania.config import (
    GLOBAL_SEED,
    ID2LABEL,
    LABEL2ID,
    TRANSFORMER_CHECKPOINT_DIR,
    TRANSFORMER_EPOCHS,
    TRANSFORMER_EVAL_BATCH_SIZE,
    TRANSFORMER_LOGGING_STEPS,
    TRANSFORMER_MAX_LENGTH,
    TRANSFORMER_MODEL_NAME,
    TRANSFORMER_TRAINER_DIR,
    TRANSFORMER_TRAIN_BATCH_SIZE,
    get_device,
    set_global_seeds,
)
from cinemania.evaluation import compute_binary_metrics

logger = logging.getLogger(__name__)


def preprocess_function(
    examples: dict[str, list[Any]],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = TRANSFORMER_MAX_LENGTH,
) -> dict[str, Any]:
    """Tokenize batch of reviews and preserve numeric labels under 'labels'."""
    tokenized = tokenizer(
        examples["review"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )
    tokenized["labels"] = [int(label) for label in examples["label"]]
    return tokenized


def compute_metrics(eval_pred: Any) -> dict[str, float]:
    """Compute binary classification metrics for Hugging Face Trainer."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1).tolist()
    y_true = [int(x) for x in labels]
    return compute_binary_metrics(y_true, preds)


def _load_pretrained_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    """Load pretrained tokenizer, falling back to local cache if offline."""
    try:
        return AutoTokenizer.from_pretrained(model_name)
    except Exception:
        logger.warning("Online load failed for tokenizer '%s'; retrying with local_files_only=True.", model_name)
        return AutoTokenizer.from_pretrained(model_name, local_files_only=True)


def _load_pretrained_model(model_name: str) -> AutoModelForSequenceClassification:
    """Load pretrained sequence classification model, falling back to local cache if offline."""
    kwargs = {
        "num_labels": 2,
        "id2label": ID2LABEL,
        "label2id": LABEL2ID,
    }
    try:
        return AutoModelForSequenceClassification.from_pretrained(model_name, **kwargs)
    except Exception:
        logger.warning("Online load failed for model '%s'; retrying with local_files_only=True.", model_name)
        return AutoModelForSequenceClassification.from_pretrained(model_name, local_files_only=True, **kwargs)


def _build_training_arguments(
    trainer_output_dir: Path,
    num_epochs: int,
    train_batch_size: int,
    eval_batch_size: int,
    gradient_accumulation_steps: int,
    logging_steps: int,
    seed: int,
) -> TrainingArguments:
    """Construct TrainingArguments compatible across transformers 4.x and 5.x."""
    sig = inspect.signature(TrainingArguments.__init__).parameters
    kwargs: dict[str, Any] = {
        "output_dir": str(trainer_output_dir),
        "num_train_epochs": num_epochs,
        "per_device_train_batch_size": train_batch_size,
        "per_device_eval_batch_size": eval_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "logging_steps": logging_steps,
        "save_strategy": "epoch",
        "save_total_limit": 1,
        "load_best_model_at_end": False,
        "seed": seed,
    }

    # Keep test set strictly isolated during training (no evaluation/selection on test set)
    if "eval_strategy" in sig:
        kwargs["eval_strategy"] = "no"
    elif "evaluation_strategy" in sig:
        kwargs["evaluation_strategy"] = "no"

    if "report_to" in sig:
        kwargs["report_to"] = "none"

    if "use_cpu" in sig and not torch.cuda.is_available():
        kwargs["use_cpu"] = True
    elif "no_cuda" in sig and not torch.cuda.is_available():
        kwargs["no_cuda"] = True

    return TrainingArguments(**kwargs)


def _build_trainer(
    model: AutoModelForSequenceClassification,
    args: TrainingArguments,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
) -> Trainer:
    """Construct Trainer compatible across transformers 4.x and 5.x."""
    sig = inspect.signature(Trainer.__init__).parameters
    kwargs: dict[str, Any] = {
        "model": model,
        "args": args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "compute_metrics": compute_metrics,
    }
    if "processing_class" in sig:
        kwargs["processing_class"] = tokenizer
    elif "tokenizer" in sig:
        kwargs["tokenizer"] = tokenizer
    return Trainer(**kwargs)


def train_and_evaluate_transformer(
    train_ds: Dataset,
    test_ds: Dataset,
    *,
    model_name: str = TRANSFORMER_MODEL_NAME,
    checkpoint_dir: Path | str = TRANSFORMER_CHECKPOINT_DIR,
    trainer_output_dir: Path | str = TRANSFORMER_TRAINER_DIR,
    max_length: int = TRANSFORMER_MAX_LENGTH,
    num_epochs: int = TRANSFORMER_EPOCHS,
    train_batch_size: int = TRANSFORMER_TRAIN_BATCH_SIZE,
    eval_batch_size: int = TRANSFORMER_EVAL_BATCH_SIZE,
    logging_steps: int = TRANSFORMER_LOGGING_STEPS,
    seed: int = GLOBAL_SEED,
) -> dict[str, Any]:
    """Fine-tune distilbert-base-uncased on train_ds, evaluate once on test_ds, and save checkpoint."""
    set_global_seeds(seed)
    device = get_device()
    ckpt_dir = Path(checkpoint_dir)
    out_dir = Path(trainer_output_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Starting DistilBERT fine-tuning on device=%s (train=%d, test=%d, epochs=%d, batch_size=%d)",
        device,
        len(train_ds),
        len(test_ds),
        num_epochs,
        train_batch_size,
    )

    tokenizer = _load_pretrained_tokenizer(model_name)

    tokenized_train = train_ds.map(
        lambda batch: preprocess_function(batch, tokenizer=tokenizer, max_length=max_length),
        batched=True,
        remove_columns=train_ds.column_names,
    )
    tokenized_test = test_ds.map(
        lambda batch: preprocess_function(batch, tokenizer=tokenizer, max_length=max_length),
        batched=True,
        remove_columns=test_ds.column_names,
    )

    grad_accum = 1
    current_bs = train_batch_size
    while True:
        model = _load_pretrained_model(model_name)
        training_args = _build_training_arguments(
            trainer_output_dir=out_dir,
            num_epochs=num_epochs,
            train_batch_size=current_bs,
            eval_batch_size=eval_batch_size,
            gradient_accumulation_steps=grad_accum,
            logging_steps=logging_steps,
            seed=seed,
        )
        trainer = _build_trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_test,
            tokenizer=tokenizer,
        )
        try:
            train_result = trainer.train()
            break
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower() and current_bs > 1:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                current_bs = max(1, current_bs // 2)
                grad_accum *= 2
                logger.warning(
                    "CUDA OOM encountered; retrying with per_device_train_batch_size=%d, gradient_accumulation_steps=%d.",
                    current_bs,
                    grad_accum,
                )
            else:
                raise

    eval_metrics_raw = trainer.evaluate(eval_dataset=tokenized_test)
    test_metrics = {
        "loss": float(eval_metrics_raw["eval_loss"]),
        "accuracy": float(eval_metrics_raw["eval_accuracy"]),
        "precision": float(eval_metrics_raw["eval_precision"]),
        "recall": float(eval_metrics_raw["eval_recall"]),
        "f1": float(eval_metrics_raw["eval_f1"]),
    }
    logger.info("Transformer Test Metrics (500 held-out reviews): %s", test_metrics)

    trainer.save_model(str(ckpt_dir))
    tokenizer.save_pretrained(str(ckpt_dir))

    log_history = list(trainer.state.log_history)
    log_history_path = out_dir / "log_history.json"
    log_history_path.write_text(json.dumps(log_history, indent=2), encoding="utf-8")

    final_train_loss = None
    if hasattr(train_result, "training_loss") and train_result.training_loss is not None:
        final_train_loss = float(train_result.training_loss)

    logger.info("Saved Transformer checkpoint to '%s'.", ckpt_dir)
    results = {
        "device": str(device),
        "model_name": model_name,
        "train_samples": len(train_ds),
        "test_samples": len(test_ds),
        "effective_train_batch_size": current_bs,
        "gradient_accumulation_steps": grad_accum,
        "final_train_loss": final_train_loss,
        "log_history": log_history,
        "test_metrics": test_metrics,
        "artifacts": {
            "checkpoint_dir": str(ckpt_dir),
            "trainer_output_dir": str(out_dir),
            "log_history_path": str(log_history_path),
        },
    }
    (out_dir / "train_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results
