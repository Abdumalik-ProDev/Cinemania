"""Extract real PyTorch activations, weights, and attention maps for visualization."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from cinemania.config import (
    ID2LABEL,
    LABEL2ID,
    MAX_SEQ_LENGTH,
    RNN_ARTIFACTS_DIR,
    TRANSFORMER_CHECKPOINT_DIR,
    TRANSFORMER_MAX_LENGTH,
    get_device,
)
from cinemania.data.rnn_encoding import encode_review, tokenize_text
from cinemania.inference.predict import _load_rnn_bundle

_EAGER_LOCK = threading.Lock()
_EAGER_TRANSFORMER_CACHE: dict[tuple[str, str], tuple[AutoModelForSequenceClassification, Any]] = {}


def clear_visualizer_cache() -> None:
    """Clear cached eager Transformer models used for attention extraction."""
    with _EAGER_LOCK:
        _EAGER_TRANSFORMER_CACHE.clear()


def _load_eager_transformer(
    checkpoint_dir: Path | str,
    device: torch.device,
) -> tuple[AutoModelForSequenceClassification, Any]:
    ckpt_dir = Path(checkpoint_dir)
    if not ckpt_dir.exists() or not ckpt_dir.is_dir():
        raise FileNotFoundError(
            f"Transformer checkpoint directory not found: '{ckpt_dir}'. "
            "Train the Transformer before running the visualizer."
        )

    cache_key = (str(ckpt_dir.resolve()), str(device))
    with _EAGER_LOCK:
        if cache_key in _EAGER_TRANSFORMER_CACHE:
            return _EAGER_TRANSFORMER_CACHE[cache_key]

        tokenizer = AutoTokenizer.from_pretrained(str(ckpt_dir), local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            str(ckpt_dir),
            local_files_only=True,
            attn_implementation="eager",
        ).to(device)
        model.eval()

        bundle = (model, tokenizer)
        _EAGER_TRANSFORMER_CACHE[cache_key] = bundle
        return bundle


def _round_list(values: list[float], digits: int = 4) -> list[float]:
    return [round(float(v), digits) for v in values]


def _round_matrix(matrix: list[list[float]], digits: int = 4) -> list[list[float]]:
    return [[round(float(val), digits) for val in row] for row in matrix]


def inspect_rnn_review(
    text: str,
    *,
    artifact_dir: Path | str = RNN_ARTIFACTS_DIR,
    device: torch.device | str | None = None,
    top_embed: int = 10,
    top_hidden: int = 16,
    max_display_steps: int = 64,
) -> dict[str, Any]:
    """Trace SentimentRNN per-token embeddings, hidden states, and learned weights."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Input review must be a non-empty string.")

    resolved_device = torch.device(device) if device is not None else get_device()
    model, vocab, config = _load_rnn_bundle(artifact_dir, resolved_device)

    max_length = int(config.get("max_length", MAX_SEQ_LENGTH))
    raw_tokens = tokenize_text(text)[:max_length]
    token_ids = encode_review(text, vocab=vocab, max_length=max_length)
    display_tokens = raw_tokens if raw_tokens else ["<unk>"]

    input_tensor = torch.tensor([token_ids], dtype=torch.long, device=resolved_device)

    model.eval()
    with torch.no_grad():
        embedded = model.embedding(input_tensor)  # (1, T, E)
        all_hidden, _ = model.rnn(embedded)  # (1, T, H)
        final_logits = model.fc(all_hidden[:, -1, :])
        final_prob_pos = float(model.sigmoid(final_logits).squeeze().item())

        fc_weight = model.fc.weight[0]  # (H,)
        fc_bias = float(model.fc.bias[0].item())
        w_ih = model.rnn.weight_ih_l0  # (H, E)
        w_hh = model.rnn.weight_hh_l0  # (H, H)

        # Rank hidden neurons by absolute contribution at the final timestep
        final_contrib = (all_hidden[0, -1, :] * fc_weight).abs()
        k_hidden = min(top_hidden, all_hidden.size(-1))
        hidden_indices = torch.topk(final_contrib, k=k_hidden).indices.tolist()

        # Rank embedding dims by mean absolute activation across the review
        mean_embed_mag = embedded[0].abs().mean(dim=0)
        k_embed = min(top_embed, embedded.size(-1))
        embed_indices = torch.topk(mean_embed_mag, k=k_embed).indices.tolist()

        h_idx_t = torch.tensor(hidden_indices, device=resolved_device, dtype=torch.long)
        e_idx_t = torch.tensor(embed_indices, device=resolved_device, dtype=torch.long)

        embed_to_hidden = w_ih.index_select(0, h_idx_t).index_select(1, e_idx_t).cpu().tolist()
        hidden_to_hidden = w_hh.index_select(0, h_idx_t).index_select(1, h_idx_t).cpu().tolist()
        hidden_to_output = fc_weight.index_select(0, h_idx_t).cpu().tolist()

        # Compute step-by-step sentiment trajectory across timesteps
        step_logits = model.fc(all_hidden[0]).squeeze(-1)  # (T,)
        step_probs_pos = torch.sigmoid(step_logits).cpu().tolist()

        # Sample 64 neurons across the full 256-dim hidden layer for the strip heatmap
        strip_count = min(64, all_hidden.size(-1))
        strip_indices = torch.linspace(0, all_hidden.size(-1) - 1, steps=strip_count).long().to(resolved_device)

        emb_cpu = embedded[0].index_select(1, e_idx_t).cpu().tolist()
        hid_cpu = all_hidden[0].index_select(1, h_idx_t).cpu().tolist()
        strip_cpu = all_hidden[0].index_select(1, strip_indices).cpu().tolist()
        fc_sel_cpu = fc_weight.index_select(0, h_idx_t).cpu().tolist()

    steps: list[dict[str, Any]] = []
    limit = min(len(token_ids), max_display_steps)
    for t in range(limit):
        tok = display_tokens[t] if t < len(display_tokens) else "<unk>"
        tid = int(token_ids[t])
        h_vals = hid_cpu[t]
        contribs = [h_vals[i] * fc_sel_cpu[i] for i in range(len(h_vals))]
        p_pos = float(step_probs_pos[t])
        steps.append(
            {
                "step": t,
                "token": tok,
                "token_id": tid,
                "in_vocab": tok in vocab,
                "embed_values": _round_list(emb_cpu[t]),
                "hidden_values": _round_list(h_vals),
                "hidden_strip": _round_list(strip_cpu[t], digits=3),
                "contributions": _round_list(contribs),
                "logit": round(float(step_logits[t].item()), 4),
                "prob_positive": round(p_pos, 4),
                "prob_negative": round(1.0 - p_pos, 4),
            }
        )

    final_prob_neg = 1.0 - final_prob_pos
    pred_id = 1 if final_prob_pos >= 0.5 else 0
    return {
        "model": "rnn",
        "review": text,
        "truncated_for_display": len(token_ids) > max_display_steps,
        "total_tokens": len(token_ids),
        "tokens": [s["token"] for s in steps],
        "architecture": {
            "vocab_size": int(config.get("vocab_size", len(vocab))),
            "embedding_dim": int(config.get("embedding_dim", embedded.size(-1))),
            "hidden_dim": int(config.get("hidden_dim", all_hidden.size(-1))),
            "displayed_embed_dim": k_embed,
            "displayed_hidden_dim": k_hidden,
        },
        "selected_embed_indices": [int(i) for i in embed_indices],
        "selected_hidden_indices": [int(i) for i in hidden_indices],
        "weights": {
            "fc_bias": round(fc_bias, 4),
            "embed_to_hidden": _round_matrix(embed_to_hidden),
            "hidden_to_hidden": _round_matrix(hidden_to_hidden),
            "hidden_to_output": _round_list(hidden_to_output),
        },
        "steps": steps,
        "final_prediction": {
            "label": ID2LABEL[pred_id],
            "confidence": round(final_prob_pos if pred_id == 1 else final_prob_neg, 4),
            "probabilities": {
                "negative": round(final_prob_neg, 4),
                "positive": round(final_prob_pos, 4),
            },
        },
    }


def inspect_transformer_review(
    text: str,
    *,
    checkpoint_dir: Path | str = TRANSFORMER_CHECKPOINT_DIR,
    device: torch.device | str | None = None,
    max_display_tokens: int = 64,
    max_length: int = TRANSFORMER_MAX_LENGTH,
) -> dict[str, Any]:
    """Extract DistilBERT multi-head attention maps and [CLS] layer representations."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Input review must be a non-empty string.")

    resolved_device = torch.device(device) if device is not None else get_device()
    model, tokenizer = _load_eager_transformer(checkpoint_dir, resolved_device)

    full_encoded = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    total_tokens = int(full_encoded["input_ids"].size(1))
    truncated_for_display = total_tokens > max_display_tokens

    display_encoded = {
        k: v[:, :max_display_tokens].to(resolved_device) for k, v in full_encoded.items()
    }
    tokens = tokenizer.convert_ids_to_tokens(display_encoded["input_ids"][0].tolist())

    model.eval()
    with torch.no_grad():
        outputs = model(**display_encoded, output_attentions=True, output_hidden_states=True)
        if truncated_for_display:
            # Compute final prediction on full training-length sequence (up to max_length=256)
            full_device = {k: v.to(resolved_device) for k, v in full_encoded.items()}
            probs = torch.softmax(model(**full_device).logits, dim=-1).squeeze(0).cpu().tolist()
        else:
            probs = torch.softmax(outputs.logits, dim=-1).squeeze(0).cpu().tolist()

    attentions_by_layer: list[dict[str, Any]] = []
    for layer_idx, attn_tensor in enumerate(outputs.attentions):
        heads_tensor = attn_tensor.squeeze(0).cpu()
        avg_matrix = heads_tensor.mean(dim=0).tolist()
        head_matrices = [_round_matrix(heads_tensor[h].tolist(), digits=4) for h in range(heads_tensor.size(0))]
        attentions_by_layer.append(
            {
                "layer": layer_idx + 1,
                "avg": _round_matrix(avg_matrix, digits=4),
                "heads": head_matrices,
            }
        )

    layer_cls_activity: list[dict[str, Any]] = []
    for idx, hidden_tensor in enumerate(outputs.hidden_states):
        cls_vec = hidden_tensor[0, 0, :].cpu()
        l2_norm = float(torch.linalg.vector_norm(cls_vec).item())
        layer_name = "Embeddings" if idx == 0 else f"Block {idx}"
        layer_cls_activity.append(
            {
                "index": idx,
                "name": layer_name,
                "cls_norm": round(l2_norm, 4),
            }
        )

    prob_neg = float(probs[LABEL2ID["negative"]])
    prob_pos = float(probs[LABEL2ID["positive"]])
    pred_id = 1 if prob_pos >= prob_neg else 0

    return {
        "model": "transformer",
        "review": text,
        "truncated_for_display": truncated_for_display,
        "total_tokens": total_tokens,
        "tokens": tokens,
        "num_layers": len(outputs.attentions),
        "num_heads": int(outputs.attentions[0].size(1)) if outputs.attentions else 0,
        "hidden_size": int(outputs.hidden_states[0].size(-1)) if outputs.hidden_states else 0,
        "attentions": attentions_by_layer,
        "layer_cls_activity": layer_cls_activity,
        "final_prediction": {
            "label": ID2LABEL[pred_id],
            "confidence": round(prob_pos if pred_id == 1 else prob_neg, 4),
            "probabilities": {
                "negative": round(prob_neg, 4),
                "positive": round(prob_pos, 4),
            },
        },
    }


def inspect_review_both(
    text: str,
    *,
    rnn_dir: Path | str = RNN_ARTIFACTS_DIR,
    transformer_dir: Path | str = TRANSFORMER_CHECKPOINT_DIR,
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    """Run both SentimentRNN and DistilBERT introspection on one review."""
    rnn_trace = inspect_rnn_review(text, artifact_dir=rnn_dir, device=device)
    transformer_trace = inspect_transformer_review(text, checkpoint_dir=transformer_dir, device=device)
    return {
        "review": text,
        "rnn": rnn_trace,
        "transformer": transformer_trace,
    }
