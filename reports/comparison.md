# Cinemania Sentiment Classification — Model Comparison Report

## 1. Experiment Setup & Data Caveats

- **Source dataset:** `data/raw/train.xlsx` (25,000 raw rows; 96 duplicate reviews removed; 24,904 unique cleaned reviews).
- **Balanced pool:** 3,000 unique reviews (1,500 positive, 1,500 negative; `random_state=101`).
- **Shared held-out test set:** 500 stratified reviews (`seed=123`), never used during vocabulary construction, training, or checkpoint selection.
- **Training-size caveat:** Per specification, `SentimentRNN` trained on **1,500 reviews** (750 positive, 750 negative) while `distilbert-base-uncased` fine-tuned on **2,500 reviews** (1,250 positive, 1,250 negative). Because the training counts differ, this comparison evaluates the two specified workflows on an identical test set rather than isolating model architecture alone.

## 2. Held-Out Test Evaluation (500 Shared Reviews)

> Note: Training loss reflects optimization fit on each model's training partition and does not by itself prove generalization; model comparison is grounded in held-out test metrics below.

| Model | Train Samples | Test Samples | Final Train Loss | Test Loss | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `SentimentRNN` (10 epochs) | 1,500 | 500 | 0.0317 | 2.0847 | 0.5240 | 0.5234 | 0.5360 | 0.5296 |
| `distilbert-base-uncased` (3 epochs) | 2,500 | 500 | 0.2538 | 0.5397 | 0.8800 | 0.8862 | 0.8720 | 0.8790 |

## 3. Required Example Predictions

| Category | Review | RNN Prediction | RNN Confidence | Transformer Prediction | Transformer Confidence |
|---|---|---|---:|---|---:|
| negative-oriented | I had been looking forward to this film for a long time, but I did not like it at all. | negative | 0.9996 | negative | 0.8308 |
| positive-oriented | The cast is excellent, and the plot is very interesting. | positive | 0.9994 | positive | 0.9990 |
| mixed | The film's visual effects are good, but the story is weak. | negative | 0.9920 | negative | 0.9975 |

## 4. Conclusion

The fine-tuned DistilBERT Transformer outperformed the custom SentimentRNN on the shared 500-review held-out test set, achieving 0.8800 accuracy and 0.8790 F1 compared to 0.5240 accuracy and 0.5296 F1. This advantage reflects both DistilBERT's pretrained contextual self-attention representations (compared with a randomly initialized single-layer vanilla RNN over a 4,000-word vocabulary) and the fact that DistilBERT was trained on 2,500 reviews rather than 1,500. Because the training-set sizes differ per the assignment specification, the results demonstrate overall pipeline performance on the shared benchmark rather than a purely controlled architectural isolation.
