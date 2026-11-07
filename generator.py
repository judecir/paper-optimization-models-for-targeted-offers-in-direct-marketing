"""Deterministic instance generator following Section 5.1 of the paper."""

from __future__ import annotations

import argparse
import math
import warnings
from pathlib import Path
from random import Random

from instance import Instance

BUDGET_VARIANTS = ("minimum", "maximum", "random")
CAPACITY_VARIANTS = ("s", "l")
DEFAULT_SEED = 1
M_VALUES = {"s": (100, 200, 300), "M": (1000, 2000), "l": (10000,)}


def generate_budget_variants(
    m: int,
    n: int,
    hurdle_rate: float,
    capacity_variant: str,
    seed: int = DEFAULT_SEED,
) -> dict[str, Instance]:
    """Generate three instances sharing all data except product budgets."""
    if m <= 0 or n <= 0:
        raise ValueError("m and n must be positive.")
    if hurdle_rate < 0:
        raise ValueError("hurdle_rate must be nonnegative.")
    if capacity_variant not in CAPACITY_VARIANTS:
        raise ValueError("capacity_variant must be 's' or 'l'.")
    rng = Random(seed)
    costs = [[rng.randint(1, 3) for _ in range(n)] for _ in range(m)]
    returns = [[rng.randint(0, 16) for _ in range(n)] for _ in range(m)]

    if capacity_variant == "s":
        upper = max(1, math.floor(n / 5))
        max_offers = [rng.randint(1, upper) for _ in range(m)]
    else:
        max_offers = [
            rng.randint(math.ceil(n / 3), math.ceil(2 * n / 3))
            for _ in range(m)
        ]
    sum_capacity = sum(max_offers)
    average_capacity = sum_capacity / n
    lower_mqc = math.ceil(average_capacity)
    upper_mqc = math.floor(2 * average_capacity)
    if upper_mqc < lower_mqc:
        raise ValueError(
            "The MQC interval is empty after capping its upper bound at "
            "floor(2 * sum_i M_i / n)."
        )
    minimum_offers = [rng.randint(lower_mqc, upper_mqc) for _ in range(n)]

    budget_limits: dict[str, list[int]] = {
        name: [] for name in BUDGET_VARIANTS
    }
    for j in range(n):
        total_cost = sum(costs[i][j] for i in range(m))
        minimum_budget = math.floor(minimum_offers[j] * total_cost / m)
        maximum_budget = math.ceil(
            2 * (sum_capacity / n) * total_cost / m
        )
        if maximum_budget < minimum_budget:
            raise ValueError(
                "The Section 5.1 budget bounds are inconsistent for this instance."
            )
        budget_limits["minimum"].append(minimum_budget)
        budget_limits["maximum"].append(maximum_budget)
        budget_limits["random"].append(rng.randint(minimum_budget, maximum_budget))

    fixed_costs: list[int] = []
    for j in range(n):
        weighted_return = sum(
            returns[i][j] - (1 + hurdle_rate) * costs[i][j] for i in range(m)
        )
        scale = minimum_offers[j] / (m * (1 + hurdle_rate))
        raw_lower = scale * weighted_return / 2
        raw_upper = scale * weighted_return
        integer_lower = max(0, math.ceil(raw_lower))
        integer_upper = math.floor(raw_upper)
        if integer_upper < integer_lower:
            fallback = max(0, math.ceil(raw_lower))
            fixed_costs.append(fallback)
            warnings.warn(
                f"Product {j}: empty or negative fixed-cost interval "
                f"[{raw_lower:.6g}, {raw_upper:.6g}]; using fallback f_j={fallback}.",
                RuntimeWarning,
                stacklevel=2,
            )
        else:
            fixed_costs.append(rng.randint(integer_lower, integer_upper))

    common = {
        "p": tuple(tuple(float(value) for value in row) for row in returns),
        "c": tuple(tuple(float(value) for value in row) for row in costs),
        "fixed_costs": tuple(float(value) for value in fixed_costs),
        "minimum_offers": tuple(minimum_offers),
        "max_offers": tuple(max_offers),
        "hurdle_rate": hurdle_rate,
        "product_names": tuple(f"P{j + 1}" for j in range(n)),
    }
    return {
        name: Instance(budgets=tuple(float(value) for value in budget_limits[name]), **common)
        for name in BUDGET_VARIANTS
    }


def experiment_groups(include_large: bool = False) -> tuple[tuple[str, int], ...]:
    """Return group labels and customer counts from the experimental plan."""
    groups = [(f"S{i + 1}", m) for i, m in enumerate(M_VALUES["s"])]
    if include_large:
        groups.extend((f"M{i + 1}", m) for i, m in enumerate(M_VALUES["M"]))
        groups.extend((f"L{i + 1}", m) for i, m in enumerate(M_VALUES["l"]))
    return tuple(groups)


def generate_experiment_instances(
    output_dir: str | Path,
    *,
    seed: int = DEFAULT_SEED,
    include_large: bool = False,
) -> list[Path]:
    """Write all instances for selected groups and return their file paths."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []
    for group, m in experiment_groups(include_large):
        for n in (5, 10, 15):
            for hurdle_rate in (0.05, 0.10, 0.15):
                for capacity_variant in CAPACITY_VARIANTS:
                    variants = generate_budget_variants(
                        m, n, hurdle_rate, capacity_variant, seed
                    )
                    for budget_variant, instance in variants.items():
                        filename = (
                            f"{group}_n{n}_R{hurdle_rate * 100:.0f}_"
                            f"{capacity_variant}_{budget_variant}_seed{seed}.json"
                        )
                        path = directory / filename
                        instance.to_json(path)
                        generated.append(path)
    return generated


def main() -> None:
    """Generate the default 162-instance S1-S3 set."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("instances"))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--include-large", action="store_true")
    args = parser.parse_args()
    paths = generate_experiment_instances(
        args.output_dir, seed=args.seed, include_large=args.include_large
    )
    print(f"Generated {len(paths)} instances in {args.output_dir}.")


if __name__ == "__main__":
    main()
