from pathlib import Path

from typer.testing import CliRunner

from imessage_mlx import cli
from imessage_mlx.utils import write_jsonl


def test_prepare_stops_before_sessionization_when_privacy_audit_fails(
    monkeypatch, tmp_path: Path
) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        f"work_dir: {tmp_path / 'work'}\nsnapshot_db: {tmp_path / 'snapshot.db'}\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(cli, "inspect_schema", lambda *_args, **_kwargs: {})

    def unsafe_extract(_database, output, *_args, **_kwargs):
        write_jsonl(
            output,
            [
                {
                    "message_id": "a" * 24,
                    "chat_id": "b" * 24,
                    "timestamp_ns": 1,
                    "sender_role": "other",
                    "participant_id": "c" * 24,
                    "text": "contact me at private@example.com",
                    "has_attachment": False,
                    "is_group": False,
                    "service": "iMessage",
                }
            ],
        )
        return {"written_rows": 1}

    sessionized = False

    def mark_sessionized(*_args, **_kwargs):
        nonlocal sessionized
        sessionized = True
        return {}

    monkeypatch.setattr(cli, "extract_messages", unsafe_extract)
    monkeypatch.setattr(cli, "build_sessions", mark_sessionized)

    result = CliRunner().invoke(cli.app, ["prepare", "--config", str(config)])

    assert result.exit_code != 0
    assert "failed the privacy audit" in result.output
    assert not sessionized
