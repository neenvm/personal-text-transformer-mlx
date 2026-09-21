.PHONY: sync lint format test security-check check

sync:
	uv sync

lint:
	uv run ruff check .

format:
	uv run ruff format .

test:
	uv run pytest -q

security-check:
	uv run python scripts/release_audit.py

check: lint test security-check
	uv run ruff format --check .
	git diff --check
