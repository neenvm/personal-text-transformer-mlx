import numpy as np
import pytest

from imessage_mlx.evaluate import (
    _matching_ngrams,
    _me_target_mask,
    _validate_checkpoint_tokenizer_hash,
    _validate_evaluation_data_hashes,
)


def test_me_target_mask_tracks_roles_across_windows() -> None:
    # Tokens: other, text, end, me, text, text, end, other, text.
    tokens = np.array([2, 10, 4, 1, 11, 12, 4, 2, 13], dtype=np.uint32)

    mask = _me_target_mask(tokens, me_id=1, other_id=2, turn_end_id=4)

    assert mask.tolist() == [False, False, False, False, True, True, True, False, False]
    # The second me-token target remains selected even when a model window starts at index 5.
    assert mask[5]
    # The token after turn_end is not attributed to the completed me turn.
    assert not mask[7]


def test_matching_ngrams_scans_beyond_two_million_tokens() -> None:
    train = np.zeros(2_000_020, dtype=np.uint32)
    planted = np.arange(1, 17, dtype=np.uint32)
    train[-len(planted) :] = planted

    matches = _matching_ngrams(train, [planted], 16)

    assert planted.tobytes() in matches


def test_evaluation_requires_the_checkpoint_training_data() -> None:
    checkpoint = {"train": "train-a", "validation": "validation-a", "test": "test-b"}
    current = {"train": "train-a", "validation": "validation-a", "test": "test-b"}
    _validate_evaluation_data_hashes(checkpoint, current)

    with pytest.raises(ValueError, match="does not record"):
        _validate_evaluation_data_hashes(None, current)
    with pytest.raises(ValueError, match="does not match"):
        _validate_evaluation_data_hashes(
            checkpoint,
            {"train": "train-b", "validation": "validation-a", "test": "test-b"},
        )
    with pytest.raises(ValueError, match="does not match"):
        _validate_evaluation_data_hashes(
            checkpoint,
            {"train": "train-a", "validation": "validation-a", "test": "foreign-test"},
        )


def test_evaluation_requires_the_recorded_checkpoint_tokenizer() -> None:
    _validate_checkpoint_tokenizer_hash("same", "same")
    with pytest.raises(ValueError, match="does not record"):
        _validate_checkpoint_tokenizer_hash(None, "current")
    with pytest.raises(ValueError, match="does not match"):
        _validate_checkpoint_tokenizer_hash("recorded", "tampered")
