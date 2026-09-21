from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from imessage_mlx.checkpoint import load_model
from imessage_mlx.data.redact import contains_obvious_pii
from imessage_mlx.dataset import load_tokens, materialize_batch, window_count
from imessage_mlx.generate import generate_ids
from imessage_mlx.model.transformer import causal_lm_loss, perplexity
from imessage_mlx.tokenizer.train import load_tokenizer
from imessage_mlx.utils import sha256_file, write_json


def _validate_evaluation_data_hashes(
    checkpoint_hashes: dict[str, str] | None, evaluation_hashes: dict[str, str]
) -> None:
    if checkpoint_hashes is None:
        raise ValueError(
            "Checkpoint does not record dataset hashes; start a new run before evaluation"
        )
    if checkpoint_hashes != evaluation_hashes:
        raise ValueError("Evaluation data does not match the checkpoint dataset")


def _validate_checkpoint_tokenizer_hash(recorded_hash: str | None, actual_hash: str) -> None:
    if recorded_hash is None:
        raise ValueError("Checkpoint does not record its tokenizer hash")
    if recorded_hash != actual_hash:
        raise ValueError("Checkpoint tokenizer does not match its recorded hash")


def _cross_entropy(model, tokens: np.ndarray, context: int, batch_size: int) -> float:
    count = window_count(tokens, context)
    if not count:
        raise ValueError("Split has no complete evaluation windows")
    total = 0.0
    examples = 0
    model.eval()
    for start in range(0, count, batch_size):
        indices = np.arange(start, min(start + batch_size, count))
        inputs, targets = materialize_batch(tokens, indices, context)
        loss = causal_lm_loss(model, mx.array(inputs), mx.array(targets))
        mx.eval(loss)
        total += float(loss.item()) * len(indices)
        examples += len(indices)
    return total / examples


def _unigram_loss(train: np.ndarray, test: np.ndarray, vocab_size: int) -> float:
    counts = np.bincount(np.asarray(train, dtype=np.int64), minlength=vocab_size).astype(np.float64)
    probabilities = (counts + 1.0) / (counts.sum() + vocab_size)
    return float(-np.log(probabilities[np.asarray(test, dtype=np.int64)]).mean())


def _me_target_mask(
    tokens: np.ndarray, *, me_id: int, other_id: int, turn_end_id: int
) -> np.ndarray:
    target_is_me = np.zeros(tokens.size, dtype=bool)
    inside_me_turn = False
    for index, token_id in enumerate(tokens):
        target_is_me[index] = inside_me_turn
        if int(token_id) == me_id:
            inside_me_turn = True
        elif int(token_id) in (other_id, turn_end_id):
            inside_me_turn = False
    return target_is_me


def _me_role_loss(
    model,
    tokens: np.ndarray,
    tokenizer,
    context: int,
    batch_size: int,
) -> tuple[float | None, int]:
    me_id = tokenizer.token_to_id("<|me|>")
    other_id = tokenizer.token_to_id("<|other|>")
    turn_end_id = tokenizer.token_to_id("<|turn_end|>")
    if None in (me_id, other_id, turn_end_id):
        return None, 0
    total_loss = 0.0
    total_tokens = 0
    count = window_count(tokens, context)
    target_is_me = _me_target_mask(tokens, me_id=me_id, other_id=other_id, turn_end_id=turn_end_id)
    model.eval()
    for start in range(0, count, batch_size):
        indices = np.arange(start, min(start + batch_size, count))
        inputs_np, targets_np = materialize_batch(tokens, indices, context)
        logits = model(mx.array(inputs_np))
        losses = nn.losses.cross_entropy(logits, mx.array(targets_np), reduction="none")
        mx.eval(losses)
        mask = np.stack(
            [
                target_is_me[int(index) * (context + 1) + 1 : (int(index) + 1) * (context + 1)]
                for index in indices
            ]
        )
        batch_losses = np.asarray(losses)
        total_loss += float(batch_losses[mask].sum())
        total_tokens += int(mask.sum())
    return (total_loss / total_tokens if total_tokens else None), total_tokens


def _matching_ngrams(
    tokens: np.ndarray, generated_arrays: list[np.ndarray], size: int
) -> set[bytes]:
    candidates = {
        array[index : index + size].tobytes()
        for array in generated_arrays
        for index in range(max(0, len(array) - size + 1))
    }
    if not candidates:
        return set()
    values = np.asarray(tokens, dtype=np.uint32)
    matches: set[bytes] = set()
    for index in range(max(0, len(values) - size + 1)):
        candidate = values[index : index + size].tobytes()
        if candidate in candidates:
            matches.add(candidate)
            if matches == candidates:
                break
    return matches


def evaluate_checkpoint(
    checkpoint: str | Path,
    data_dir: str | Path,
    output_path: str | Path,
    *,
    batch_size: int = 16,
    generation_samples: int = 20,
) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint)
    data_path = Path(data_dir)
    data_hashes = {
        name: sha256_file(data_path / f"{name}.npy") for name in ("train", "validation", "test")
    }
    trainer_state = json.loads((checkpoint_path / "trainer-state.json").read_text(encoding="utf-8"))
    _validate_evaluation_data_hashes(trainer_state.get("dataset_hashes"), data_hashes)
    _validate_checkpoint_tokenizer_hash(
        trainer_state.get("tokenizer_sha256"),
        sha256_file(checkpoint_path / "tokenizer/tokenizer.json"),
    )
    model = load_model(checkpoint_path)
    tokenizer = load_tokenizer(checkpoint_path / "tokenizer")
    train = load_tokens(data_path / "train.npy")
    validation = load_tokens(data_path / "validation.npy")
    test = load_tokens(data_path / "test.npy")
    context = model.config.max_sequence_length
    validation_loss = _cross_entropy(model, validation, context, batch_size)
    test_loss = _cross_entropy(model, test, context, batch_size)
    me_loss, me_tokens = _me_role_loss(model, test, tokenizer, context, batch_size)
    unigram_loss = _unigram_loss(train, test, model.config.vocab_size)

    turn_end = tokenizer.token_to_id("<|turn_end|>")
    eos = tokenizer.token_to_id("<|eos|>")
    eos_ids = {token_id for token_id in (turn_end, eos) if token_id is not None}
    test_windows = window_count(test, context)
    sample_count = min(generation_samples, test_windows)
    pii_generations = 0
    generated_arrays: list[np.ndarray] = []
    for index in range(sample_count):
        start = index * (context + 1)
        prompt = np.asarray(test[start : start + max(8, context // 2)], dtype=np.int64).tolist()
        generated = generate_ids(
            model,
            prompt,
            eos_ids=eos_ids,
            max_new_tokens=32,
            temperature=0.8,
            top_p=0.9,
            repetition_penalty=1.1,
            seed=42 + index,
        )
        decoded = tokenizer.decode(generated, skip_special_tokens=True)
        pii_generations += int(contains_obvious_pii(decoded))
        array = np.asarray(generated, dtype=np.uint32)
        generated_arrays.append(array)

    matching_by_size = {size: _matching_ngrams(train, generated_arrays, size) for size in (8, 16)}
    generated_by_size = {
        size: sum(max(0, len(array) - size + 1) for array in generated_arrays) for size in (8, 16)
    }
    overlap_by_size = {
        size: sum(
            array[index : index + size].tobytes() in matching_by_size[size]
            for array in generated_arrays
            for index in range(max(0, len(array) - size + 1))
        )
        for size in (8, 16)
    }
    longest_match = max((size for size, matches in overlap_by_size.items() if matches), default=0)

    report = {
        "checkpoint_model_sha256": sha256_file(checkpoint_path / "model.safetensors"),
        "parameter_count": model.parameter_count,
        "context_length": context,
        "train_tokens": int(train.size),
        "validation_tokens": int(validation.size),
        "test_tokens": int(test.size),
        "validation_loss": validation_loss,
        "validation_perplexity": perplexity(validation_loss),
        "test_loss": test_loss,
        "test_perplexity": perplexity(test_loss),
        "me_turn_test_loss": me_loss,
        "me_turn_test_perplexity": perplexity(me_loss) if me_loss is not None else None,
        "me_turn_test_tokens": me_tokens,
        "unigram_test_loss": unigram_loss,
        "unigram_test_perplexity": math.exp(min(unigram_loss, 50)),
        "beats_unigram_baseline": test_loss < unigram_loss,
        "data_hashes": data_hashes,
        "memorization": {
            "generation_samples": sample_count,
            "generated_8grams": generated_by_size[8],
            "matching_train_8grams": overlap_by_size[8],
            "generated_16grams": generated_by_size[16],
            "matching_train_16grams": overlap_by_size[16],
            "longest_detected_exact_match_tokens": longest_match,
            "training_tokens_scanned": int(train.size),
            "generations_with_obvious_pii_pattern": pii_generations,
            "matched_text_persisted": False,
        },
    }
    write_json(output_path, report)
    return report
