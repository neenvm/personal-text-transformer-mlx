from pathlib import Path

import pytest

from imessage_mlx import utils
from imessage_mlx.utils import ensure_private_dir, replace_directory


def test_ensure_private_dir_secures_intermediate_directories(tmp_path: Path) -> None:
    nested = ensure_private_dir(tmp_path / "private/intermediate/final")
    assert nested.stat().st_mode & 0o777 == 0o700
    assert (tmp_path / "private").stat().st_mode & 0o777 == 0o700
    assert (tmp_path / "private/intermediate").stat().st_mode & 0o777 == 0o700


def test_ensure_private_dir_does_not_chmod_existing_shared_directory(tmp_path: Path) -> None:
    shared = tmp_path / "shared"
    shared.mkdir(mode=0o755)
    shared.chmod(0o755)

    with pytest.raises(PermissionError, match="must not be accessible"):
        ensure_private_dir(shared)

    assert shared.stat().st_mode & 0o777 == 0o755


def test_replace_directory_restores_previous_artifact_on_failure(
    monkeypatch, tmp_path: Path
) -> None:
    destination = tmp_path / "model"
    destination.mkdir()
    (destination / "old").write_text("complete", encoding="utf-8")
    staged = tmp_path / "staged"
    staged.mkdir()
    (staged / "new").write_text("complete", encoding="utf-8")
    real_replace = utils.os.replace

    def fail_install(source, target):
        if Path(source) == staged and Path(target) == destination:
            raise OSError("injected install failure")
        return real_replace(source, target)

    monkeypatch.setattr(utils.os, "replace", fail_install)
    with pytest.raises(OSError, match="injected"):
        replace_directory(staged, destination)

    assert (destination / "old").read_text(encoding="utf-8") == "complete"
    assert not (tmp_path / ".model.backup").exists()
