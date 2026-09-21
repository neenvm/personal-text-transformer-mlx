from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from imessage_mlx.utils import ensure_private_dir, replace_directory, sha256_file, write_json


def export_model(
    checkpoint: str | Path,
    output_dir: str | Path,
    *,
    metrics_path: str | Path | None = None,
    split_report_path: str | Path | None = None,
    split_dir: str | Path | None = None,
) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint)
    destination = Path(output_dir)
    ensure_private_dir(destination.parent)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    temporary.chmod(0o700)
    try:
        for name in ("model.safetensors", "model-config.json"):
            shutil.copy2(checkpoint_path / name, temporary / name)
        shutil.copytree(checkpoint_path / "tokenizer", temporary / "tokenizer")
        generation_config = {
            "max_new_tokens": 64,
            "temperature": 0.8,
            "top_p": 0.9,
            "repetition_penalty": 1.1,
            "stop_tokens": ["<|turn_end|>", "<|eos|>"],
        }
        write_json(temporary / "generation-config.json", generation_config)
        if metrics_path is None or not Path(metrics_path).exists():
            raise FileNotFoundError("A completed evaluation metrics file is required for export")
        metrics: dict[str, Any] = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
        model_sha256 = sha256_file(temporary / "model.safetensors")
        evaluated_model_sha256 = metrics.get("checkpoint_model_sha256")
        if evaluated_model_sha256 is None:
            raise ValueError("Evaluation metrics do not record the evaluated checkpoint model")
        if evaluated_model_sha256 != model_sha256:
            raise ValueError("Evaluation metrics were produced from a different checkpoint model")
        write_json(temporary / "metrics.json", metrics)
        manifest: dict[str, Any] = {
            "model_sha256": model_sha256,
            "private_local_artifact": True,
            "pretrained_weights_used": False,
            "pretrained_tokenizer_used": False,
        }
        trainer_state_path = checkpoint_path / "trainer-state.json"
        if trainer_state_path.exists():
            trainer_state = json.loads(trainer_state_path.read_text(encoding="utf-8"))
            if "dataset_hashes" in trainer_state:
                manifest["checkpoint_data_hashes"] = trainer_state["dataset_hashes"]
            recorded_tokenizer_hash = trainer_state.get("tokenizer_sha256")
            actual_tokenizer_hash = sha256_file(checkpoint_path / "tokenizer/tokenizer.json")
            if recorded_tokenizer_hash is None or recorded_tokenizer_hash != actual_tokenizer_hash:
                raise ValueError("Checkpoint tokenizer does not match its recorded hash")
            manifest["tokenizer_sha256"] = actual_tokenizer_hash
        if "data_hashes" in metrics:
            manifest["evaluation_data_hashes"] = metrics["data_hashes"]
        manifest["evaluation_model_sha256"] = evaluated_model_sha256
        checkpoint_hashes = manifest.get("checkpoint_data_hashes")
        evaluation_hashes = manifest.get("evaluation_data_hashes")
        if checkpoint_hashes is None or evaluation_hashes is None:
            raise ValueError("Training and evaluation data provenance is required for export")
        if checkpoint_hashes != evaluation_hashes:
            raise ValueError("Evaluation data does not match the checkpoint dataset")
        if split_report_path is not None and Path(split_report_path).exists():
            manifest["split_report"] = json.loads(
                Path(split_report_path).read_text(encoding="utf-8")
            )
        if split_dir is not None:
            manifest["split_hashes"] = {
                name: sha256_file(Path(split_dir) / f"{name}.jsonl")
                for name in ("train", "validation", "test")
            }
        write_json(temporary / "data-manifest.json", manifest)
        readme = (
            "# Private local iMessage model\n\n"
            "This artifact was trained from random initialization and is sensitive. "
            "Keep it local.\n\n"
            "Run from the project root:\n\n"
            "```bash\nuv run imessage-mlx chat --model outputs/final\n```\n"
        )
        (temporary / "README.md").write_text(readme, encoding="utf-8")
        (temporary / "README.md").chmod(0o600)
        for path in temporary.rglob("*"):
            path.chmod(0o700 if path.is_dir() else 0o600)
        replace_directory(temporary, destination)
        destination.chmod(0o700)
        return manifest
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
