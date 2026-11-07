from generator import experiment_groups, generate_budget_variants
from instance import Instance


def test_generation_is_deterministic_and_budget_variants_share_core() -> None:
    first = generate_budget_variants(100, 5, 0.10, "s", seed=2026)
    second = generate_budget_variants(100, 5, 0.10, "s", seed=2026)
    assert first == second
    assert set(first) == {"minimum", "maximum", "random"}
    instances = list(first.values())
    assert all(instance.p == instances[0].p for instance in instances)
    assert all(instance.c == instances[0].c for instance in instances)
    assert all(instance.fixed_costs == instances[0].fixed_costs for instance in instances)
    assert all(instance.minimum_offers == instances[0].minimum_offers for instance in instances)
    assert all(instance.max_offers == instances[0].max_offers for instance in instances)
    assert all(instance.hurdle_rate == instances[0].hurdle_rate for instance in instances)
    assert first["minimum"].budgets != first["maximum"].budgets


def test_json_round_trip(tmp_path) -> None:
    instance = generate_budget_variants(100, 5, 0.10, "l", seed=3)["random"]
    path = tmp_path / "instance.json"
    instance.to_json(path)
    assert Instance.from_json(path) == instance


def test_default_and_full_experiment_group_counts() -> None:
    assert len(experiment_groups()) == 3
    assert len(experiment_groups(include_large=True)) == 6
