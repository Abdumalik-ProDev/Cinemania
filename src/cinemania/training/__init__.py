"""Training modules for SentimentRNN and DistilBERT."""

from cinemania.training.train_rnn import train_and_evaluate_rnn
from cinemania.training.train_transformer import train_and_evaluate_transformer

__all__ = ["train_and_evaluate_rnn", "train_and_evaluate_transformer"]
