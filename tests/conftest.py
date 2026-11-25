"""Shared lightweight model fixture builders for unit tests."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.pre_tokenizers import Whitespace
from transformers import DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizerFast

from cinemania.config import ID2LABEL, LABEL2ID
from cinemania.data.rnn_encoding import save_vocab
from cinemania.models.sentiment_rnn import SentimentRNN


def make_tiny_rnn_artifacts(artifact_dir: Path) -> None:
    """Create a tiny saved SentimentRNN checkpoint, vocab, and config in artifact_dir."""
    artifact_dir.mkdir(parents=True, exist_ok=True)
    vocab = {"<pad>": 0, "<unk>": 1, "excellent": 2, "cast": 3, "boring": 4, "film": 5, "story": 6}
    save_vocab(vocab, artifact_dir / "vocab.json")
    model = SentimentRNN(vocab_size=len(vocab), embedding_dim=128, hidden_dim=256, pad_idx=0)
    torch.save(model.state_dict(), artifact_dir / "sentiment_rnn.pt")
    config = {
        "vocab_size": len(vocab),
        "embedding_dim": 128,
        "hidden_dim": 256,
        "pad_idx": 0,
        "unk_idx": 1,
        "max_length": 256,
    }
    (artifact_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")


def make_tiny_transformer_checkpoint(ckpt_dir: Path, *, num_layers: int = 2) -> None:
    """Create a tiny offline DistilBERT checkpoint and WordPiece tokenizer in ckpt_dir."""
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    vocab_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "excellent", "cast", "boring", "film", "story"]
    vocab_map = {tok: idx for idx, tok in enumerate(vocab_tokens)}
    tok_backend = Tokenizer(WordPiece(vocab=vocab_map, unk_token="[UNK]"))
    tok_backend.pre_tokenizer = Whitespace()
    fast_tok = DistilBertTokenizerFast(
        tokenizer_object=tok_backend,
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
    )
    fast_tok.save_pretrained(str(ckpt_dir))

    cfg = DistilBertConfig(
        vocab_size=len(vocab_tokens),
        dim=32,
        n_layers=num_layers,
        n_heads=2,
        hidden_dim=64,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    model = DistilBertForSequenceClassification(cfg)
    model.save_pretrained(str(ckpt_dir))
