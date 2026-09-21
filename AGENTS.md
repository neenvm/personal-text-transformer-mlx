# Repository Instructions

## Purpose

Maintain a privacy-first, local-only MLX pipeline for training and evaluating a small text model on
Apple silicon.

## Non-negotiable privacy boundary

- Never commit real messages, contact data, database snapshots, processed corpora, tokenizers,
  checkpoints, weights, prompts, or generations.
- Use synthetic fixtures in tests and documentation.
- Do not add telemetry, hosted training, remote inference, or automatic uploads.
- Keep the live Messages database read-only and process only a local snapshot.
- Run `make check` and inspect the staged diff before every push.

## Engineering boundary

- Preserve the chronological split, train-only tokenizer fitting, aggregate-only reporting, and
  privacy audit.
- Treat pseudonymized data as sensitive, not anonymous.
- Keep changes narrowly scoped and document any new external dependency or network behavior.
