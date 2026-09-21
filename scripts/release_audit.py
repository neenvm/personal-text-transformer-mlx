"""Fail closed when a candidate public tree contains unsafe release material."""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
MAX_FILE_BYTES = 2_000_000
ALLOWED_ROOT_FILES = {
    ".gitignore",
    ".python-version",
    "AGENTS.md",
    "CONTRIBUTING.md",
    "LICENSE",
    "Makefile",
    "NOTICE.md",
    "README.md",
    "SECURITY.md",
    "pyproject.toml",
    "uv.lock",
}
ALLOWED_ROOT_DIRS = {".github", "configs", "docs", "scripts", "src", "tests"}
FORBIDDEN_PARTS = {
    "artifacts",
    "checkpoints",
    "corpora",
    "corpus",
    "datasets",
    "exports",
    "logs",
    "messages",
    "models",
    "outputs",
    "private",
    "runs",
    "snapshots",
    "tasks",
    "tokenizers",
    "weights",
    "work",
}
FORBIDDEN_SUFFIXES = {
    ".7z",
    ".arrow",
    ".ckpt",
    ".csv",
    ".db",
    ".feather",
    ".gz",
    ".jsonl",
    ".key",
    ".log",
    ".mlx",
    ".npy",
    ".npz",
    ".parquet",
    ".pem",
    ".pickle",
    ".pkl",
    ".pt",
    ".pth",
    ".rar",
    ".safetensors",
    ".sqlite",
    ".sqlite3",
    ".tar",
    ".tgz",
    ".txt",
    ".whl",
    ".zip",
}
ALLOWED_SUFFIX_PATHS = {PurePosixPath("tests/fixtures/synthetic_messages.jsonl")}
SUSPICIOUS_NAME_PARTS = {
    "chat-history",
    "conversation-export",
    "generated-output",
    "message-export",
    "prompt-dump",
    "transcript",
}
TEXT_PATTERNS = {
    "absolute user path": re.compile(r"/(?:Users|home)/[^/\s]+/"),
    "Codex task link": re.compile(r"codex://" r"threads/", re.IGNORECASE),
    "private key": re.compile(r"BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY"),
    "GitHub token": re.compile(r"\b(?:ghp|github_pat)_[A-Za-z0-9_]{20,}\b"),
    "OpenAI key": re.compile(r"\bsk-" r"[A-Za-z0-9_-]{20,}\b"),
    "AWS access key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "Google API key": re.compile(r"\bAIza[0-9A-Za-z_-]{30,}\b"),
    "Slack token": re.compile(r"\bxox[baprs]-[0-9A-Za-z-]{20,}\b"),
    "Stripe live key": re.compile(r"\b(?:sk|rk)_live_[0-9A-Za-z]{16,}\b"),
    "assigned secret": re.compile(
        r"(?i)(?:api[_-]?key|access[_-]?token|auth[_-]?token|client[_-]?secret|password)"
        r"\s*[:=]\s*[\"'][^\"'${}<\s]{8,}[\"']"
    ),
}


def candidate_paths() -> list[Path]:
    git = shutil.which("git")
    if git is None:
        raise RuntimeError("git is required for a fail-closed release audit")
    result = subprocess.run(
        [git, "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    relative_paths = [Path(raw.decode()) for raw in result.stdout.split(b"\0") if raw]
    return sorted(ROOT / path for path in relative_paths)


def path_findings(relative: PurePosixPath) -> list[str]:
    findings: list[str] = []
    if not relative.parts:
        return ["empty candidate path"]
    if len(relative.parts) == 1:
        if relative.name not in ALLOWED_ROOT_FILES:
            findings.append(f"unexpected root file: {relative}")
    elif relative.parts[0] not in ALLOWED_ROOT_DIRS:
        findings.append(f"unexpected root directory: {relative}")

    lower_parts = {part.lower() for part in relative.parts}
    if lower_parts & FORBIDDEN_PARTS:
        findings.append(f"forbidden path: {relative}")
    if relative.suffix.lower() in FORBIDDEN_SUFFIXES and relative not in ALLOWED_SUFFIX_PATHS:
        findings.append(f"forbidden artifact type: {relative}")

    normalized_name = relative.name.lower().replace("_", "-")
    if relative.parts[0] not in {"src", "tests"} and any(
        marker in normalized_name for marker in SUSPICIOUS_NAME_PARTS
    ):
        findings.append(f"suspicious generated-content filename: {relative}")
    return findings


def content_findings(relative: PurePosixPath, content: bytes) -> list[str]:
    if len(content) > MAX_FILE_BYTES:
        return [f"unexpected large file: {relative}"]
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        return [f"unexpected binary file: {relative}"]
    return [
        f"{label}: {relative}" for label, pattern in TEXT_PATTERNS.items() if pattern.search(text)
    ]


def main() -> int:
    findings: list[str] = []
    try:
        files = candidate_paths()
    except (OSError, subprocess.CalledProcessError, RuntimeError) as error:
        print(f"release audit could not enumerate the candidate Git tree: {error}")
        return 1

    if not files:
        print("release audit failed: candidate Git tree is empty")
        return 1

    for path in files:
        try:
            relative_path = path.relative_to(ROOT)
            relative = PurePosixPath(relative_path.as_posix())
        except ValueError:
            findings.append(f"candidate escapes repository root: {path}")
            continue
        if path.is_symlink():
            findings.append(f"symbolic link is not allowed: {relative}")
            continue
        if not path.is_file():
            findings.append(f"candidate is not a regular file: {relative}")
            continue
        findings.extend(path_findings(relative))
        try:
            findings.extend(content_findings(relative, path.read_bytes()))
        except OSError as error:
            findings.append(f"could not inspect {relative}: {error}")

    if findings:
        print("release audit failed:")
        for finding in sorted(set(findings)):
            print(f"- {finding}")
        return 1
    print(f"release audit passed: {len(files)} Git candidate files checked")
    return 0


if __name__ == "__main__":
    sys.exit(main())
