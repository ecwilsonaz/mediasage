"""Every documented environment variable must reach the container through docker-compose.yml."""

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def documented_variables() -> set[str]:
    """Variables in the first column of the README's Environment Variables table."""
    section = (ROOT / "README.md").read_text().split("### Environment Variables")[1].split("\n### ")[0]
    rows = [line.split("|")[1] for line in section.splitlines() if line.startswith("| `")]
    return {name for row in rows for name in re.findall(r"`([A-Z][A-Z0-9_]+)`", row)}


def compose_environment() -> dict[str, str]:
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
    return dict(entry.split("=", 1) for entry in compose["services"]["mediasage"]["environment"])


def test_readme_table_is_found() -> None:
    assert {"PLEX_URL", "LLM_MODEL_GENERATION", "LIBRARY_RETRY_MINUTES"} <= documented_variables()


def test_compose_passes_every_documented_variable() -> None:
    assert documented_variables() - compose_environment().keys() == set()


def test_compose_takes_values_from_env_file() -> None:
    for name, value in compose_environment().items():
        assert value.startswith(f"${{{name}:-"), f"{name} should be ${{{name}:-}}"


def test_env_example_lists_every_documented_variable() -> None:
    example = (ROOT / ".env.example").read_text()
    listed = set(re.findall(r"^#?([A-Z][A-Z0-9_]+)=", example, flags=re.MULTILINE))
    assert documented_variables() - listed == set()


def test_empty_variable_counts_as_unset(monkeypatch) -> None:
    """Compose passes unset variables as empty strings."""
    from backend.plex_client import _env_int

    monkeypatch.setenv("PLEX_TIMEOUT", "")
    assert _env_int("PLEX_TIMEOUT", 300) == 300
    monkeypatch.setenv("PLEX_TIMEOUT", "45")
    assert _env_int("PLEX_TIMEOUT", 300) == 45
