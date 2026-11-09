from dataclasses import replace
from math import isclose

import pytest

from generator import generate_budget_variants
from instance import Instance, validation_instance
from models import brute_force_optimum, solve_model, validate_solution
from setcover import enumerate_columns, solve_setcover


def test_m3_columns_match_hand_calculation() -> None:
    columns = enumerate_columns(validation_instance())
    assert len(columns) == 8
    assert [
        (column.product, column.customers, column.objective_coefficient)
        for column in columns
    ] == [
        (0, (0, 1), 9),
        (0, (0, 2), 10),
        (1, (0,), 2),
        (1, (1,), 1),
        (1, (2,), 1),
        (1, (0, 1), 4),
        (1, (0, 2), 4),
        (1, (1, 2), 3),
    ]
    assert [round(column.hurdle_coefficient, 1) for column in columns] == [
        8.5, 9.4, 1.8, 0.8, 0.7, 3.7, 3.6, 2.6
    ]


def test_brute_force_finds_unique_expected_optimum() -> None:
    instance = validation_instance()
    objective, x, y, count = brute_force_optimum(instance)
    assert objective == 11
    assert count == 1
    assert x == ((1.0, 0.0), (0.0, 1.0), (1.0, 0.0))
    assert y == (1.0, 1.0)
    hurdle_lhs = sum(
        column.hurdle_coefficient
        for column in enumerate_columns(instance)
        if (column.product == 0 and column.customers == (0, 2))
        or (column.product == 1 and column.customers == (1,))
    )
    assert isclose(hurdle_lhs, 10.2)


def test_integer_models_and_lp_bound_order() -> None:
    instance = validation_instance()
    integer_results = [
        solve_model(instance, formulation, time_limit=60)
        for formulation in ("M1", "M2")
    ]
    setcover_integer = solve_setcover(instance, time_limit=60)
    assert [result.objective for result in integer_results] == [11, 11]
    assert setcover_integer.objective == 11
    assert all(result.is_optimal for result in (*integer_results, setcover_integer))
    assert integer_results[1].x == ((1.0, 0.0), (0.0, 1.0), (1.0, 0.0))
    assert integer_results[1].y == (1.0, 1.0)

    lp_m1 = solve_model(instance, "M1", relax=True, time_limit=60).objective
    lp_m2 = solve_model(instance, "M2", relax=True, time_limit=60).objective
    lp_m3 = solve_setcover(instance, relax=True, time_limit=60).objective
    assert lp_m1 is not None and lp_m2 is not None and lp_m3 is not None
    print(f"LP bounds: M1={lp_m1:.6f}, M2={lp_m2:.6f}, M3={lp_m3:.6f}")
    assert 11 <= lp_m3 + 1e-7 <= lp_m2 + 1e-7 <= lp_m1 + 1e-7


def test_validator_rejects_hurdle_violation() -> None:
    instance = replace(validation_instance(), hurdle_rate=10)
    with pytest.raises(ValueError, match="hurdle"):
        validate_solution(
            instance,
            ((1.0, 0.0), (0.0, 0.0), (0.0, 0.0)),
            (1.0, 0.0),
        )


def test_setcover_rejects_large_instances() -> None:
    instance = Instance(
        p=tuple((1.0,) for _ in range(13)),
        c=tuple((1.0,) for _ in range(13)),
        fixed_costs=(0.0,),
        budgets=(13.0,),
        minimum_offers=(1,),
        max_offers=tuple(1 for _ in range(13)),
        hurdle_rate=0,
    )
    with pytest.raises(ValueError, match="m <= 12"):
        enumerate_columns(instance)


def test_setcover_handles_products_without_admissible_columns() -> None:
    instance = generate_budget_variants(
        m=8,
        n=2,
        hurdle_rate=0.10,
        capacity_variant="l",
        seed=4004,
    )["maximum"]
    assert not enumerate_columns(instance)
    result = solve_setcover(instance)
    assert result.is_optimal
    assert result.objective == 0
