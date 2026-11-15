"""Cloud model catalog, loaded from model_catalog.yaml at the project root."""

from dataclasses import dataclass
from pathlib import Path

import yaml

CATALOG_PATH = Path(__file__).resolve().parent.parent / "model_catalog.yaml"


@dataclass(frozen=True)
class CatalogModel:
    """A cloud model with known pricing (USD per million tokens) and context window."""

    id: str
    provider: str
    label: str
    context: int
    input_cost: float
    output_cost: float
    long_context_threshold: int | None = None
    long_input_cost: float | None = None
    long_output_cost: float | None = None
    effort: str | None = None
    legacy: bool = False


def load_catalog(path: Path = CATALOG_PATH) -> tuple[CatalogModel, ...]:
    """Load the model catalog from YAML, keyed by provider then model ID."""
    with open(path) as f:
        data = yaml.safe_load(f)

    return tuple(
        CatalogModel(id=model_id, provider=provider, **fields)
        for provider, models in data.items()
        for model_id, fields in models.items()
    )


MODEL_CATALOG = load_catalog()

CATALOG_BY_ID = {m.id: m for m in MODEL_CATALOG}
