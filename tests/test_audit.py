from imessage_mlx.audit import (
    _artifact_provenance_checks,
    _selected_model_matches,
    _training_completed,
)


def test_training_completion_uses_saved_configuration() -> None:
    assert _training_completed(
        {"global_step": 10, "epoch": 3, "early_stopped": False}, {"epochs": 3}
    )
    assert not _training_completed(
        {"global_step": 10, "epoch": 3, "early_stopped": False}, {"epochs": 5}
    )
    assert _training_completed(
        {"global_step": 10, "epoch": 2, "early_stopped": True}, {"epochs": 5}
    )


def test_selected_model_accepts_every_supported_matching_preset() -> None:
    assert _selected_model_matches({"selected": "model-1m"}, {"name": "model-1m"})
    assert _selected_model_matches({"selected": "model-7m"}, {"name": "model-7m"})
    assert not _selected_model_matches({"selected": "model-7m"}, {"name": "model-1m"})


def test_artifact_provenance_binds_evaluation_model_data_and_tokenizer() -> None:
    evaluation = {
        "checkpoint_model_sha256": "model",
        "data_hashes": {"train": "train", "validation": "validation", "test": "test"},
    }
    manifest = {
        "model_sha256": "model",
        "evaluation_model_sha256": "model",
        "checkpoint_data_hashes": evaluation["data_hashes"],
        "evaluation_data_hashes": evaluation["data_hashes"],
        "tokenizer_sha256": "tokenizer",
    }
    best_state = {
        "dataset_hashes": manifest["checkpoint_data_hashes"],
        "tokenizer_sha256": "tokenizer",
    }
    assert all(
        _artifact_provenance_checks(
            evaluation,
            evaluation,
            manifest,
            "tokenizer",
            best_state,
            "model",
            "tokenizer",
        ).values()
    )

    foreign_evaluation = {**evaluation, "checkpoint_model_sha256": "foreign"}
    checks = _artifact_provenance_checks(
        foreign_evaluation,
        evaluation,
        manifest,
        "tokenizer",
        best_state,
        "model",
        "tokenizer",
    )
    assert not checks["current_evaluation_matches_manifest"]
    assert not checks["exported_metrics_match_current_evaluation"]
    tampered_checks = _artifact_provenance_checks(
        evaluation,
        evaluation,
        manifest,
        "tampered",
        best_state,
        "stale-model",
        "tampered",
    )
    assert not tampered_checks["final_tokenizer_hash_matches"]
    assert not tampered_checks["best_checkpoint_model_matches"]
    assert not tampered_checks["best_checkpoint_tokenizer_matches"]
