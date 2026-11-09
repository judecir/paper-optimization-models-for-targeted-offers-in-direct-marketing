import csv

from run_experiments import write_aggregate_report


def test_aggregate_keeps_solvers_separate(tmp_path) -> None:
    rows = [
        {
            "instance": "S1_n5_R5_s_minimum_seed1.json",
            "n": 5,
            "model": "M1",
            "solver": solver,
            "lp_gap_percent": 10.0,
            "mip_gap_percent": 0.0,
            "elapsed_seconds": elapsed,
            "termination": "optimal",
        }
        for solver, elapsed in (("appsi_highs", 3.0), ("gurobi", 1.0))
    ]
    output = tmp_path / "aggregate.csv"

    aggregate = write_aggregate_report(rows, output)

    assert len(aggregate) == 2
    assert {row["solver"] for row in aggregate} == {"appsi_highs", "gurobi"}
    assert {row["instances_solved_to_optimality"] for row in aggregate} == {1}
    with output.open(encoding="utf-8", newline="") as file:
        saved = list(csv.DictReader(file))
    assert len(saved) == 2
    assert {row["solver"] for row in saved} == {"appsi_highs", "gurobi"}
