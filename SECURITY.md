# Security policy

This project operates on private communications. Treat every generated dataset, tokenizer,
checkpoint, evaluation prompt, and model as sensitive—even after pseudonymization.

## Never include private artifacts in a report

When opening an issue or security report, do not attach:

- `chat.db`, its WAL/SHM files, or any database snapshot
- Files from `work/` or `outputs/`
- Raw or processed message text
- Contact identifiers, pseudonym keys, prompts, or generated private replies
- Tokenizers, checkpoints, or final model weights trained on private messages

Use synthetic reproductions and aggregate counts only.

## Reporting a vulnerability

Report vulnerabilities through GitHub's private vulnerability reporting feature. Do not open a
public issue for a suspected data-exposure or code-execution vulnerability.

If you accidentally commit private data, stop immediately. Do not merely add it to `.gitignore` in a
later commit—the data remains in Git history. Rotate any exposed secrets, remove the data from Git
history, and replace the remote repository before publication.

## Supported version

Only the latest commit on `main` is supported during the alpha period.

## Scope

High-priority reports include accidental message disclosure, writes to the live Messages database,
unsafe handling of model artifacts, path traversal, command execution, and privacy-gate bypasses.
