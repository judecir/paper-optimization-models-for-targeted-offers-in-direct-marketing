"""Instance data and JSON serialization for direct-marketing offer models."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Instance:
    """Data for a direct-marketing offer-selection instance.

    Customer and product indices are zero-based in Python. ``p`` and ``c`` are
    customer-by-product matrices; budgets, fixed costs and MQCs are product-wise.
    """

    p: tuple[tuple[float, ...], ...]
    c: tuple[tuple[float, ...], ...]
    fixed_costs: tuple[float, ...]
    budgets: tuple[float, ...]
    minimum_offers: tuple[int, ...]
    max_offers: tuple[int, ...]
    hurdle_rate: float
    product_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        m = len(self.p)
        if m == 0:
            raise ValueError("An instance must contain at least one customer.")
        n = len(self.fixed_costs)
        if n == 0:
            raise ValueError("An instance must contain at least one product.")
        if len(self.c) != m or any(len(row) != n for row in self.p + self.c):
            raise ValueError("p and c must be customer-by-product matrices of equal shape.")
        if any(len(values) != n for values in (self.budgets, self.minimum_offers)):
            raise ValueError("Product-wise data must have one value per product.")
        if len(self.max_offers) != m:
            raise ValueError("max_offers must have one value per customer.")
        if self.product_names and len(self.product_names) != n:
            raise ValueError("product_names must have one name per product.")
        if self.hurdle_rate < 0:
            raise ValueError("hurdle_rate must be nonnegative.")
        if any(value < 0 for row in self.p for value in row):
            raise ValueError("Expected returns p_ij must be nonnegative.")
        if any(value <= 0 for row in self.c for value in row):
            raise ValueError("Variable costs c_ij must be positive.")
        if any(value < 0 for value in self.fixed_costs + self.budgets):
            raise ValueError("Fixed costs and budgets must be nonnegative.")
        if any(value < 0 for value in self.minimum_offers):
            raise ValueError("Minimum offer commitments must be nonnegative.")
        if any(value < 0 or value > n for value in self.max_offers):
            raise ValueError("Each customer offer limit must be between zero and n.")

    @property
    def m(self) -> int:
        """Number of customers."""
        return len(self.p)

    @property
    def n(self) -> int:
        """Number of products."""
        return len(self.fixed_costs)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return {
            "p": [list(row) for row in self.p],
            "c": [list(row) for row in self.c],
            "fixed_costs": list(self.fixed_costs),
            "budgets": list(self.budgets),
            "minimum_offers": list(self.minimum_offers),
            "max_offers": list(self.max_offers),
            "hurdle_rate": self.hurdle_rate,
            "product_names": list(self.product_names),
        }

    def to_json(self, path: str | Path) -> None:
        """Write this instance to a UTF-8 JSON file."""
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Instance:
        """Create an instance from a JSON-compatible mapping."""
        return cls(
            p=tuple(tuple(float(value) for value in row) for row in data["p"]),
            c=tuple(tuple(float(value) for value in row) for row in data["c"]),
            fixed_costs=tuple(float(value) for value in data["fixed_costs"]),
            budgets=tuple(float(value) for value in data["budgets"]),
            minimum_offers=tuple(int(value) for value in data["minimum_offers"]),
            max_offers=tuple(int(value) for value in data["max_offers"]),
            hurdle_rate=float(data["hurdle_rate"]),
            product_names=tuple(data.get("product_names", ())),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> Instance:
        """Read an instance from a UTF-8 JSON file."""
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def validation_instance() -> Instance:
    """Return the three-customer example specified for implementation validation."""
    return Instance(
        p=((6, 4), (8, 3), (10, 4)),
        c=((1, 1), (2, 1), (3, 2)),
        fixed_costs=(2, 1),
        budgets=(4, 3),
        minimum_offers=(2, 1),
        max_offers=(1, 1, 1),
        hurdle_rate=0.10,
        product_names=("A", "B"),
    )
