# Cinemania — IMDb Review Sentiment Classification

## 1. Project Overview

Build a reproducible sentiment-classification experiment for **Cinemania**, a hypothetical movie-streaming service. The system must classify an English movie review as either `positive` or `negative` using two independently trained models:

1. A custom PyTorch recurrent neural network (`SentimentRNN`) built with `nn.Embedding`, `nn.RNN`, `nn.Linear`, and `nn.Sigmoid`.
2. A Transformer classifier fine-tuned from `distilbert-base-uncased` using the Hugging Face `transformers` and `datasets` libraries.

The objective is to train both models, evaluate them on the same held-out reviews, expose reusable prediction functions, and produce an evidence-based comparison. This is a focused educational ML experiment, not a production streaming platform. Do not build a web application, REST API, authentication system, database, or deployment infrastructure unless explicitly requested.

## 2. Primary Requirements

The implementation is complete only when all of the following are true:

- The existing IMDb dataset in `data/raw/` is discovered and loaded successfully. **Do not download the dataset again.**
- The data is validated, cleaned, and transformed into reproducible, balanced datasets.
- The RNN vocabulary contains the 4,000 most frequent training tokens plus `<pad>` and `<unk>`.
- `SentimentRNN` uses the required dimensions and PyTorch layers, and is trained for exactly 10 epochs with `BCELoss` and Adam at learning rate `0.001`.
- A `distilbert-base-uncased` sequence-classification model is fine-tuned for 3 epochs using the Hugging Face `Trainer` API.
- Both models are evaluated against the **same 500 held-out reviews**.
- Separate inference functions return `positive` or `negative` for previously unseen English text.
- The three required example reviews are evaluated by both models.
- Training/evaluation metrics, predictions, model artifacts, and a concise comparison report are saved to disk.
- The full experiment can be reproduced with documented commands and fixed random seeds.

## 3. Data Location and Input Contract

### 3.1 Existing local dataset

The dataset has already been downloaded and is located under:

```text
data/raw/
```

The coding agent must inspect this directory and load the existing dataset file. The expected source is the IMDb 50K review spreadsheet, commonly named `train.xlsx`, but the implementation should identify the actual supported file present in `data/raw/` rather than assume a different location.

Supported input formats should include at least `.xlsx` and `.csv`. Use `pandas.read_excel()` for Excel files and `pandas.read_csv()` for CSV files. Do not access the internet, invoke `wget`, or download another copy. If no supported dataset file exists, stop with a clear error that identifies the expected directory and supported formats.

### 3.2 Required columns and cleaning

The source data must contain:

- `review`: the review text.
- `sentiment`: the class label.

Normalize column names by trimming whitespace and converting them to lowercase before validating them. Keep only these two columns for the classification workflow.

Apply the following cleaning rules:

1. Remove rows with a missing or empty review.
2. Normalize sentiment strings using `strip().lower()`.
3. Accept only `positive` and `negative`; fail with a useful error if unexpected labels remain. Do not silently map unknown values.
4. Preserve the original review text apart from handling missing/empty values. Do not aggressively rewrite or paraphrase reviews.
5. Log the number of rows before and after cleaning and the class distribution.
6. Do not silently remove duplicates. Report their count; only deduplicate if the source data or an explicit requirement makes that necessary.

Use this label mapping everywhere:

```python
LABEL2ID = {"negative": 0, "positive": 1}
ID2LABEL = {0: "negative", 1: "positive"}
```

## 4. Important Dataset-Size Inconsistency and Required Resolution

The original assignment contains two incompatible sample-size instructions:

- The data-preparation section requests 1,000 positive + 1,000 negative reviews: **2,000 total**.
- The Transformer section requests **2,500 training + 500 test reviews**: 3,000 total.

A 2,000-row DataFrame cannot be split into 2,500 training rows and 500 test rows. Do not silently change either requested count, attempt an impossible split, or duplicate records to reach 3,000.

Resolve this while preserving both requested counts and a common evaluation set:

1. From the cleaned source data, create a reproducible, balanced pool of **3,000 unique reviews**: 1,500 positive and 1,500 negative. Use `random_state=101`, then shuffle the combined DataFrame with `frac=1` and a fixed random state.
2. Create a Hugging Face `Dataset` from this pool. Use `Dataset.train_test_split()` with `test_size=500`, `seed=123`, and stratification by the numeric label. The resulting Transformer partitions must contain exactly **2,500 training** and **500 test** reviews, with both classes represented proportionally.
3. For the RNN, use the same 500-row test partition. From the 2,500-row training partition, sample 750 positive and 750 negative reviews using `random_state=101`. Combine these 1,500 training reviews with the shared 500-row test set. The RNN dataset therefore contains exactly **1,000 positive + 1,000 negative reviews**, as requested.
4. Train the RNN on its 1,500-row training partition and the Transformer on its 2,500-row training partition. Evaluate both against the exact same 500 test records. Never include a test record in either model's training data.

This is the required compatibility resolution. The two models intentionally have different training-set sizes because the source assignment specifies incompatible totals. The report must disclose this limitation: the shared test set makes the evaluation examples directly comparable, but the experiment does **not** isolate architecture as the only variable because the Transformer receives more training data.

If the installed Hugging Face `datasets` version requires a `ClassLabel` feature for stratification, define the label column with `ClassLabel(names=["negative", "positive"])` before calling `train_test_split()`.

## 5. RNN Data Encoding

### 5.1 Tokenization

Implement one deterministic tokenizer and use it consistently when building the vocabulary and encoding reviews. A lightweight lowercase English word tokenizer based on a regular expression is sufficient, for example:

```python
re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", text.lower())
```

Do not add an unrelated NLP framework or pretrained word embeddings. Keep tokenization simple, documented, and deterministic.

### 5.2 Vocabulary

Build the vocabulary **from the RNN training reviews only**. Do not use held-out test reviews to construct the vocabulary; this avoids using test data during training preparation.

Requirements:

- Count tokens with `collections.Counter`.
- Retain the 4,000 most frequent ordinary tokens.
- Reserve `<pad>` at index `0` and `<unk>` at index `1`.
- Store the vocabulary as a token-to-index mapping.
- The maximum vocabulary size is therefore 4,002, including the two special tokens.
- Save the vocabulary to `artifacts/rnn/vocab.json` so inference uses exactly the same token IDs as training.

### 5.3 Review encoding and batching

Implement `encode_review(text, vocab, max_length=256)`:

- Tokenize the input text.
- Convert known tokens to their vocabulary indices.
- Convert unknown tokens to index `1` (`<unk>`).
- Truncate sequences to 256 tokens.
- Return a non-empty sequence; use `<unk>` for an empty token sequence.

Add an encoded representation to the in-memory RNN data structure. Do not overwrite the original review text.

Create a PyTorch `Dataset` and `DataLoader`. The batch collator should pad sequences with index `0` and return sequence lengths as well as labels. Use dynamic batch padding rather than padding every review to the longest review in the entire dataset.

The RNN must not treat trailing padding as meaningful review content. Use `pack_padded_sequence()` with the true sequence lengths, or an equally correct method that selects each sequence's final valid hidden state.

## 6. SentimentRNN Architecture and Training

### 6.1 Required architecture

Implement a class named `SentimentRNN` that inherits from `torch.nn.Module`.

Required architecture:

| Component | Required configuration |
|---|---:|
| Vocabulary size | Actual vocabulary size, including special tokens |
| Embedding dimension | `128` |
| Hidden dimension | `256` |
| Embedding | `nn.Embedding`, with `padding_idx=0` |
| Recurrent layer | `nn.RNN`, unidirectional, `batch_first=True` |
| Classifier | `nn.Linear(256, 1)` |
| Output activation | `nn.Sigmoid` |

Use the final valid hidden state as the review representation. The model must return one probability per review, with a consistent output shape such as `(batch_size, 1)`.

Do not replace `nn.RNN` with LSTM, GRU, a pretrained encoder, or another architecture. Do not remove the sigmoid layer: the specified training loss is `nn.BCELoss`.

### 6.2 Training configuration

Use:

```python
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epochs = 10
```

Use a configurable batch size with a default of `32`. Set the model to training mode for each epoch, clear gradients before each optimization step, calculate the loss, backpropagate, and update parameters.

Print and record the **sample-weighted average training loss** for every epoch. Do not calculate the epoch average as an unweighted mean of batch means when the final batch may be smaller.

After the final epoch, evaluate the RNN on the shared 500-row test set in evaluation mode and with gradient calculation disabled. Record test loss and classification metrics. Do not use the test set for model selection or hyperparameter tuning.

### 6.3 Device and reproducibility

- Use CUDA when available; otherwise use CPU.
- Move model parameters and batch tensors to the selected device.
- Set fixed seeds for Python, NumPy, and PyTorch. Also use the specified sampling and split seeds.
- Log the selected device and relevant configuration.
- Keep the implementation runnable on a machine without a GPU.

## 7. Transformer Data Preparation

### 7.1 Hugging Face Dataset

Use the 3,000-row balanced pool and the 2,500/500 split defined in Section 4. Convert the cleaned DataFrame to a `datasets.Dataset` and use `train_test_split()` with `seed=123` and label stratification.

The final partitions must have these exact sizes:

| Partition | Number of reviews |
|---|---:|
| Transformer training | 2,500 |
| Shared test | 500 |

Verify and log the label distribution in each partition.

### 7.2 Tokenization

Load:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
```

Implement `preprocess_function` and apply it to both partitions with the Hugging Face `map()` method.

Requirements:

- Use `max_length=256`.
- Enable truncation and padding. Fixed-length padding is acceptable for this assignment; dynamic padding through a suitable data collator is also acceptable if batching remains correct.
- Preserve the numeric class labels as the model's `labels` field.
- Remove raw text and unused columns from the tokenized datasets before passing them to `Trainer`.
- Do not tokenize or train on the shared test reviews as part of the training data.

## 8. Transformer Model, Training, and Evaluation

### 8.1 Model

Load the sequence-classification model from the required checkpoint:

```python
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2,
    id2label=ID2LABEL,
    label2id=LABEL2ID,
)
```

Use the same label mapping as the RNN. Do not substitute another pretrained model.

### 8.2 Trainer configuration

Use the Hugging Face `Trainer` and `TrainingArguments` APIs. Required settings:

- `num_train_epochs=3`
- `logging_steps=100`
- `output_dir` under `artifacts/transformer/`

Also configure the evaluation dataset, batch sizes, checkpoint saving, and metric reporting. Evaluate at least once after training. A sensible default is evaluation and checkpoint saving at each epoch, with the best checkpoint selected by `eval_loss` when the installed Transformers version supports the corresponding settings.

Use a modest default per-device batch size (for example, `8`) so the experiment is practical on consumer GPUs. Keep CPU execution supported. Avoid external experiment-tracking services; configure reporting so the run does not require a third-party account.

Implement `compute_metrics` to calculate at least accuracy, precision, recall, and F1. Use the positive class (`1`) as the positive label and define zero-division behavior explicitly.

After training, call `trainer.evaluate()` on the shared 500-row test set. Record `eval_loss` and the classification metrics. Use the current API supported by the installed Transformers version; do not blindly use deprecated argument names or suppress compatibility errors.

## 9. Evaluation and Comparison

Evaluate both models on the **identical 500 held-out reviews**. At minimum, record:

- Training loss (RNN: per epoch; Transformer: Trainer training loss/log history).
- Test/evaluation loss.
- Accuracy.
- Precision.
- Recall.
- F1-score.

Also generate a comparison table containing each metric for both models. Do not claim that a lower training loss alone proves better generalization. Base the conclusion primarily on held-out evaluation metrics and the required example predictions.

Save raw metrics in machine-readable JSON and provide a readable Markdown report. Include the training sample counts for each model and disclose that they differ (1,500 RNN training reviews versus 2,500 Transformer training reviews).

## 10. Inference API

Implement two separate reusable functions:

```python
predict_sentiment_rnn(text: str) -> dict
predict_sentiment_transformer(text: str) -> dict
```

Each function must:

1. Accept one previously unseen English review as a string.
2. Validate that the input is a non-empty string.
3. Load or receive the already-trained model and its required tokenizer/vocabulary without retraining.
4. Apply the exact preprocessing used during training.
5. Run in evaluation mode with gradient calculation disabled.
6. Return the predicted label (`positive` or `negative`) and class confidence/probabilities in a clear, documented structure.

For the RNN, use the saved vocabulary and the same tokenizer, maximum length, padding index, and unknown-token behavior used during training. For the Transformer, use the matching saved tokenizer and model checkpoint. Do not load a newly initialized model for inference.

Provide a small inference entry point or script so a developer can run both predictors against arbitrary text from the command line or a Python session.

## 11. Required Example Predictions

Run both models on these English reviews. They are English translations of the three examples in the original assignment; using English keeps the examples consistent with the English IMDb corpus and `distilbert-base-uncased` checkpoint.

1. **Negative-oriented:** `I had been looking forward to this film for a long time, but I did not like it at all.`
2. **Positive-oriented:** `The cast is excellent, and the plot is very interesting.`
3. **Mixed:** `The film's visual effects are good, but the story is weak.`

Print a clear table with the input review, RNN prediction and confidence, and Transformer prediction and confidence. Preserve the actual model outputs; do not hardcode expected labels or alter predictions to make them agree with the sentiment suggested by the example text.

Save the predictions to `reports/example_predictions.csv`.

## 12. Required Outputs and Project Structure

Use a clean, modular Python project. A suitable structure is:

```text
.
├── data/
│   └── raw/                         # Existing dataset; do not download again
├── artifacts/
│   ├── rnn/
│   │   ├── sentiment_rnn.pt
│   │   ├── vocab.json
│   │   └── config.json
│   └── transformer/
│       ├── checkpoint/              # Hugging Face model/tokenizer files
│       └── trainer_output/           # Trainer checkpoints and logs
├── reports/
│   ├── metrics.json
│   ├── comparison.md
│   └── example_predictions.csv
├── src/
│   └── cinemania/
│       ├── __init__.py
│       ├── config.py
│       ├── data/
│       │   ├── __init__.py
│       │   ├── loading.py
│       │   ├── preparation.py
│       │   └── rnn_encoding.py
│       ├── models/
│       │   ├── __init__.py
│       │   └── sentiment_rnn.py
│       ├── training/
│       │   ├── __init__.py
│       │   ├── train_rnn.py
│       │   └── train_transformer.py
│       ├── inference/
│       │   ├── __init__.py
│       │   └── predict.py
│       ├── evaluation.py
│       └── run_experiment.py
├── tests/
│   ├── test_data.py
│   ├── test_rnn.py
│   └── test_inference.py
├── pyproject.toml
├── uv.lock
├── .gitignore
└── README.md
```

The exact module boundaries may be adjusted if the existing repository already has a sensible structure. Avoid unnecessary abstraction and duplicated preprocessing logic. Keep configuration values in one place rather than scattering magic numbers throughout the code.

Required generated artifacts:

- RNN state dictionary, vocabulary, and inference configuration.
- Transformer model and tokenizer checkpoint.
- Per-epoch RNN training losses and Transformer training logs.
- Test metrics for both models.
- A CSV containing the three example predictions.
- `reports/comparison.md` with a compact metric table, the sample-size caveat, and a **2–3 sentence conclusion** identifying which model performed better on this experiment and why.

Do not commit the raw dataset, large model checkpoints, virtual environments, caches, or generated temporary files to Git. Add appropriate entries to `.gitignore`. Keep reports and lightweight configuration files available in the repository.

## 13. Dependencies and Execution

Use `uv` for environment and dependency management. Define project dependencies in `pyproject.toml` and commit the generated `uv.lock`. Do not mix package managers or run ad-hoc global installations.

The implementation should use the relevant libraries:

- `pandas` and an Excel engine such as `openpyxl` for dataset loading.
- `numpy` and `scikit-learn` for reproducibility, splitting support where needed, and evaluation metrics.
- `torch` for the RNN.
- `datasets` and `transformers` for the Transformer workflow.

Document the exact commands in `README.md`. The expected workflow should be straightforward, for example:

```bash
uv sync
uv run python -m cinemania.run_experiment
```

Configure the package so the documented module command works from the repository root. The experiment entry point must run the data preparation, RNN training, Transformer training, evaluation, inference examples, and report generation in a predictable order. It should reuse prepared splits rather than sampling different data independently in different modules.

If training both models in one command is impractical, provide separate documented commands for each stage plus one command that runs the complete workflow. Do not claim a stage succeeded unless its artifacts and metrics were actually produced.

## 14. Testing and Quality Requirements

Add lightweight tests for important correctness conditions:

- The local dataset loader finds a supported file and validates required columns.
- Cleaning removes missing/empty reviews and rejects unknown sentiment labels.
- The balanced pool has 3,000 unique rows, with 1,500 examples per class.
- The Transformer split has exactly 2,500 train rows and 500 test rows.
- The RNN split has exactly 1,500 train rows and uses the same 500 test rows.
- No test review appears in either model's training partition.
- The vocabulary contains `<pad>` at index `0` and `<unk>` at index `1`.
- Unknown words map to `<unk>` and padding maps to `<pad>`.
- `SentimentRNN` returns one probability per input review and its output is in the `[0, 1]` range.
- Both inference functions return one of the two valid labels and finite probabilities.

Use small synthetic fixtures for unit tests where possible. Do not require full model training to run the ordinary unit-test suite. Add clear error handling for missing files, invalid labels, empty input, missing checkpoints, and device-related failures.

## 15. Coding-Agent Working Rules

Follow these rules throughout implementation:

1. **Inspect before changing.** Inspect the existing repository, `data/raw/`, and current project configuration before creating files. Preserve useful existing code and conventions.
2. **Do not redownload data.** The dataset already exists locally. Never invoke the source URL or `wget`.
3. **Do not fabricate results.** Run the actual training and inference workflows before writing metrics or the model-comparison conclusion. If hardware or dependency issues prevent a run, report the exact blocker and leave unavailable metrics clearly marked as not produced.
4. **Keep train/test separation strict.** The shared test set is for final evaluation only. Do not use its text or labels for vocabulary construction, optimization, checkpoint selection, or hyperparameter tuning.
5. **Reuse a single split definition.** Persist or deterministically regenerate the same row identities for both models. Avoid independently sampling test sets in separate training scripts.
6. **Keep preprocessing consistent.** Training and inference must call the same preprocessing implementation for each model.
7. **Make runs reproducible.** Centralize seeds, model settings, paths, and batch sizes in configuration.
8. **Prefer simple, readable code.** Implement the assignment directly. Do not introduce a database, microservices, a frontend, external AI APIs, or unrelated infrastructure.
9. **Handle resource constraints.** Detect CUDA availability, support CPU, and use conservative Transformer batch sizes. If out-of-memory occurs, reduce batch size or use gradient accumulation without changing the required model or epoch count.
10. **Verify before declaring completion.** Run unit tests, execute the available end-to-end workflow, inspect generated artifacts, and confirm both prediction functions work from saved checkpoints.
11. **Document limitations honestly.** In particular, disclose the different training-set sizes when interpreting the comparison.
12. **Summarize delivery.** At completion, state what was implemented, which commands were run, where artifacts and reports were saved, the actual evaluation metrics, and any remaining blockers.

## 16. Definition of Done

The task is done when a fresh environment can be initialized using the documented `uv` commands; the existing local dataset can be processed without downloading anything; both required models can be trained with their specified architectures and epoch counts; both models can produce predictions from saved artifacts; both models are evaluated on the same 500 unseen reviews; the three example reviews have recorded predictions; tests pass; and the comparison report contains real metrics plus a concise, appropriately qualified conclusion.
