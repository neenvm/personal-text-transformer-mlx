# Local iMessage model data card

## Scope

Private conversational text extracted from a read-only local Messages database snapshot.
Attachments are excluded. This data card must remain local unless every field has been reviewed
for disclosure risk.

## Required run-specific facts

- Snapshot and split hashes
- Retained and excluded row counts
- Body-recovery rate
- Train, validation, and test date ranges
- Token counts and model-selection calculation
- Model parameter count
- Validation and test perplexity
- Memorization-probe aggregates

Do not include message samples, names, handles, chat identifiers, contact counts, attachment names,
exact timestamps, or generated private text in any public data card.
