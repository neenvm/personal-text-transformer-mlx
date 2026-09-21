# Personal Text Transformer for MLX

[![CI](https://github.com/neenvm/personal-text-transformer-mlx/actions/workflows/ci.yml/badge.svg)](https://github.com/neenvm/personal-text-transformer-mlx/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A privacy-first research pipeline for training and evaluating a small decoder-only Transformer from
scratch on your own Messages history, entirely on an Apple-silicon Mac.

The project snapshots the Messages database read-only, pseudonymizes database identifiers, redacts
obvious contact strings, creates leakage-resistant chronological splits, trains a tokenizer only on
the training split, and keeps every dataset and model artifact local.

> [!WARNING]
> Pseudonymized conversations and trained models are still sensitive. They can contain or memorize
> personal information. Never publish datasets, tokenizers, checkpoints, weights, prompts, or
> generations produced by this project.

## What is included

- Read-only SQLite snapshotting with consistency checks
- HMAC-based pseudonymization of message, chat, and participant identifiers
- Default redaction for URLs, email addresses, and common phone-number-shaped strings
- An automatic privacy gate before sessionization or dataset splitting
- Chronological train, validation, and test splits with guard bands
- A byte-level BPE tokenizer fitted on the training split only
- A compact causal Transformer implemented with MLX
- Resumable training, held-out evaluation, provenance hashes, and completion auditing
- Synthetic fixtures and an end-to-end test suite

No command sends an iMessage, uploads data, invokes hosted inference, or enables telemetry.

## Requirements

- Apple silicon Mac
- macOS with Python 3.11
- [`uv`](https://docs.astral.sh/uv/)
- Full Disk Access for the terminal or application running the snapshot command

## Install

```bash
git clone https://github.com/neenvm/personal-text-transformer-mlx.git
cd personal-text-transformer-mlx
uv sync --frozen
make check
```

## Local workflow

First inspect the safety prerequisites:

```bash
uv run imessage-mlx doctor
```

When `safe_to_snapshot_real_data` is `true`, create a private local snapshot and prepare the
pseudonymized splits:

```bash
uv run imessage-mlx snapshot
uv run imessage-mlx prepare
```

The `prepare` command stops automatically if the extracted corpus fails the privacy audit. Continue
through tokenizer training, corpus sizing, model training, evaluation, and export only after that
gate passes. See [the architecture](docs/architecture.md) and
[troubleshooting guide](docs/troubleshooting.md) for the complete sequence and boundaries.

## Privacy model

- The live Messages database is opened using SQLite read-only mode and `PRAGMA query_only`.
- Attachments are never opened or copied.
- Newly created private output directories use owner-only permissions and are ignored by Git.
- Reports contain aggregate counts and hashes, not message text.
- Tests and public documentation use synthetic data only.
- The project has no application telemetry or automatic upload path.

The tool refuses to write through an existing output directory that grants group or other access;
it never silently changes that directory's permissions. The bundled defaults are installed with
the package, while the editable examples under `configs/` remain available for customization.

These controls reduce accidental publication during ordinary local use. Redaction is deliberately
conservative but not exhaustive, and neither pseudonymization nor the privacy audit makes text
anonymous. The project cannot protect a compromised macOS account, malicious dependencies,
untrusted model artifacts, or deliberately unsafe handling outside the pipeline. Review
[the privacy boundary](docs/privacy.md) before using real data.

## Research limitations

Small models trained from scratch on personal corpora often produce incoherent text or memorize
training material. Passing the pipeline's integrity and privacy gates does not prove conversational
quality, anonymity, or safety for sharing a trained model.

## Attribution

This project is derived from Pietro Schirano's
[`texts-to-transformer`](https://github.com/Doriandarko/texts-to-transformer). The original MIT
license is preserved, and the public changes are summarized in [NOTICE.md](NOTICE.md).

## Security

Do not disclose private data in a public issue. Use GitHub private vulnerability reporting as
described in [SECURITY.md](SECURITY.md).
