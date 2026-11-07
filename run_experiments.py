"""Run M1 and M2 experiments and write detailed and aggregated CSV reports."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from generator import DEFAULT_SEED, generate_experiment_instances
from instance import Instance
from models import DEFAULT_SOLVER, SolveResult, solve_model

MODELS = ("M1", "M2")


def _relative_gap(lp: float | None, integer: float | None) -> float | None:
    if lp is None or integer is None or math.isclose(integer, 0.0, abs_tol=1e-12):
        return None
    return (lp - integer) / integer * 100


def _mip_gap(bound: float | None, incumbent: float | None) -> float | None:
    if bound is None or incumbent is None or math.isclose(incumbent, 0.0, abs_tol=1e-12):
        return None
    return (bound - incumbent) / abs(incumbent) * 100


def _result_row(
    path: Path,
    instance: Instance,
    formulation: str,
    solver: str,
    integer_result: SolveResult,
    lp_result: SolveResult,
) -> dict[str, object]:
    return {
        "instance": path.name,
        "m": instance.m,
        "n": instance.n,
        "hurdle_rate": instance.hurdle_rate,
        "model": formulation,
        "solver": solver,
        "objective": integer_result.objective,
        "best_bound": integer_result.best_bound,
        "mip_gap_percent": _mip_gap(integer_result.best_bound, integer_result.objective),
        "lp_objective": lp_result.objective,
        "lp_gap_percent": _relative_gap(lp_result.objective, integer_result.objective),
        "elapsed_seconds": integer_result.elapsed_seconds,
        "branch_and_bound_nodes": integer_result.branch_and_bound_nodes,
        "status": integer_result.status,
        "termination": integer_result.termination,
        "lp_status": lp_result.status,
        "lp_termination": lp_result.termination,
    }


def run_experiments(
    instance_paths: list[Path],
    output_csv: str | Path,
    *,
    solver: str = DEFAULT_SOLVER,
    time_limit: float = 3600,
) -> list[dict[str, object]]:
    """Solve both formulations and save one row per instance and model."""
    rows: list[dict[str, object]] = []
    for path in instance_paths:
        instance = Instance.from_json(path)
        for formulation in MODELS:
            integer_result = solve_model(
                instance, formulation, solver=solver, time_limit=time_limit
            )
            lp_result = solve_model(
                instance, formulation, solver=solver, relax=True, time_limit=time_limit
            )
            rows.append(
                _result_row(path, instance, formulation, solver, integer_result, lp_result)
            )
    if rows:
        write_results_csv(rows, output_csv)
    return rows


def write_results_csv(
    rows: list[dict[str, object]],
    output_csv: str | Path,
) -> None:
    """Write detailed per-instance and per-model solve results."""
    if not rows:
        return
    with Path(output_csv).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_aggregate_report(
    rows: list[dict[str, object]],
    output_csv: str | Path,
) -> list[dict[str, object]]:
    """Write mean LP gaps and optimal solve counts by group, n, and model."""
    grouped: dict[tuple[str, int, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        name = str(row["instance"])
        group = name.split("_", maxsplit=1)[0]
        grouped[
            (group, int(row["n"]), str(row["model"]), str(row.get("solver", DEFAULT_SOLVER)))
        ].append(row)
    aggregate: list[dict[str, object]] = []
    for (group, n, formulation, solver), items in sorted(grouped.items()):
        gaps = [float(row["lp_gap_percent"]) for row in items if row["lp_gap_percent"] is not None]
        mip_gaps = [
            float(row["mip_gap_percent"])
            for row in items
            if row.get("mip_gap_percent") is not None
        ]
        times = [float(row["elapsed_seconds"]) for row in items if row["elapsed_seconds"] is not None]
        aggregate.append(
            {
                "group": group,
                "n": n,
                "model": formulation,
                "solver": solver,
                "mean_lp_gap_percent": sum(gaps) / len(gaps) if gaps else None,
                "mean_mip_gap_percent": sum(mip_gaps) / len(mip_gaps) if mip_gaps else None,
                "mean_elapsed_seconds": sum(times) / len(times) if times else None,
                "instances_solved_to_optimality": sum(
                    str(row["termination"]).lower() == "optimal" for row in items
                ),
                "instance_count": len(items),
            }
        )
    if aggregate:
        write_results_csv(aggregate, output_csv)
    return aggregate


def main() -> None:
    """Generate (unless provided), solve, and summarize the experiment set."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances-dir", type=Path, default=Path("instances"))
    parser.add_argument("--results", type=Path, default=Path("results.csv"))
    parser.add_argument("--aggregate", type=Path, default=Path("aggregate.csv"))
    parser.add_argument("--solver", default=DEFAULT_SOLVER)
    parser.add_argument("--time-limit", type=float, default=3600)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--include-large", action="store_true")
    parser.add_argument("--generate", action="store_true")
    args = parser.parse_args()
    if args.generate or not args.instances_dir.exists():
        generate_experiment_instances(
            args.instances_dir, seed=args.seed, include_large=args.include_large
        )
    paths = sorted(args.instances_dir.glob("*.json"))
    rows = run_experiments(paths, args.results, solver=args.solver, time_limit=args.time_limit)
    aggregate = write_aggregate_report(rows, args.aggregate)
    print(json.dumps(aggregate, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
