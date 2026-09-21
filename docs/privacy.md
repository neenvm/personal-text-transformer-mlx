# Privacy boundary

This project processes private communications entirely on the local Mac.

- The source under `~/Library/Messages` is opened read-only and is never modified.
- All processing uses a SQLite backup under `work/`, never the live database.
- Attachments are not opened or copied. Only an attachment-presence marker may be used.
- Raw handles and chat identifiers are replaced with keyed HMAC identifiers before JSONL is
  written.
- URLs, email addresses, and common phone-number-shaped strings are redacted by default.
- Ordinary commands log aggregate counts, not message bodies.
- `prepare` runs the privacy audit automatically and stops before sessionization if it fails.
- `work/`, `outputs/`, databases, arrays, and model weights are ignored by Git.
- No data, tokenizer, checkpoint, generated sample, or model is uploaded.
- Tests use synthetic fixtures only.
- The inference command generates terminal text and never sends a message.

New output directories are created with owner-only permissions. If an existing target directory
grants group or other access, the tool refuses to use it instead of silently changing permissions.

Pseudonymized messages and trained weights are still sensitive and may memorize text. The
redaction patterns are not an exhaustive PII detector. Outputs are not anonymous and must remain
local.

## Threat boundary

This project protects against accidental publication and ordinary local mishandling. It does not
protect a compromised macOS account, a malicious dependency, an untrusted model artifact, or a
user who deliberately points output paths at a shared location. Run it only in a trusted local
environment, review dependency updates, and never open model or tokenizer artifacts from unknown
sources.
