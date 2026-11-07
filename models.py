"""Pyomo formulations and solve helpers for models M1 and M2."""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose
from time import perf_counter
from typing import Literal

import pyomo.environ as pyo
from pyomo.opt import SolverStatus, TerminationCondition

from instance import Instance

Formulation = Literal["M1", "M2"]
DEFAULT_SOLVER = "appsi_highs"
VALID_SOLVERS = (DEFAULT_SOLVER, "gurobi", "cplex")


@dataclass(frozen=True)
class SolveResult:
    """Portable summary of a Pyomo optimization run."""

    formulation: str
    solver: str
    objective: float | None
    best_bound: float | None
    status: str
    termination: str
    elapsed_seconds: float
    branch_and_bound_nodes: int | None
    x: tuple[tuple[float, ...], ...] | None
    y: tuple[float, ...] | None
    relaxed: bool

    @property
    def is_optimal(self) -> bool:
        """Whether the solver proved optimality."""
        return self.termination.lower() == "optimal"


def build_model(
    instance: Instance,
    formulation: Formulation = "M2",
    relax: bool = False,
) -> pyo.ConcreteModel:
    """Build M1 or M2, optionally with continuous [0, 1] decision variables."""
    if formulation not in ("M1", "M2"):
        raise ValueError(f"Unknown formulation {formulation!r}; expected 'M1' or 'M2'.")
    model = pyo.ConcreteModel(name=formulation)
    model.I = pyo.RangeSet(0, instance.m - 1)
    model.J = pyo.RangeSet(0, instance.n - 1)
    domain = pyo.UnitInterval if relax else pyo.Binary
    model.x = pyo.Var(model.I, model.J, domain=domain)
    model.y = pyo.Var(model.J, domain=domain)
    model.objective = pyo.Objective(
        expr=sum(
            (instance.p[i][j] - instance.c[i][j]) * model.x[i, j]
            for i in range(instance.m)
            for j in range(instance.n)
        )
        - sum(instance.fixed_costs[j] * model.y[j] for j in range(instance.n)),
        sense=pyo.maximize,
    )
    model.hurdle = pyo.Constraint(
        expr=sum(
            instance.p[i][j] * model.x[i, j]
            for i in range(instance.m)
            for j in range(instance.n)
        )
        >= (1 + instance.hurdle_rate)
        * (
            sum(
                instance.c[i][j] * model.x[i, j]
                for i in range(instance.m)
                for j in range(instance.n)
            )
            + sum(instance.fixed_costs[j] * model.y[j] for j in range(instance.n))
        )
    )
    model.budget = pyo.Constraint(
        model.J,
        rule=lambda m, j: sum(
            instance.c[i][j] * m.x[i, j] for i in range(instance.m)
        )
        <= instance.budgets[j],
    )
    model.customer_limit = pyo.Constraint(
        model.I,
        rule=lambda m, i: sum(m.x[i, j] for j in range(instance.n))
        <= instance.max_offers[i],
    )
    if formulation == "M1":
        model.link = pyo.Constraint(
            model.J,
            rule=lambda m, j: sum(m.x[i, j] for i in range(instance.m))
            <= instance.m * m.y[j],
        )
    else:
        model.link = pyo.Constraint(
            model.I,
            model.J,
            rule=lambda m, i, j: m.x[i, j] <= m.y[j],
        )
    model.minimum_commitment = pyo.Constraint(
        model.J,
        rule=lambda m, j: sum(m.x[i, j] for i in range(instance.m))
        >= instance.minimum_offers[j] * m.y[j],
    )
    return model


def _solver_options(
    solver_name: str,
    time_limit: float | None,
) -> dict[str, float]:
    if time_limit is not None and time_limit <= 0:
        raise ValueError("time_limit must be positive.")
    if solver_name not in VALID_SOLVERS:
        raise ValueError(f"Unsupported solver {solver_name!r}; choose from {VALID_SOLVERS}.")
    if time_limit is None:
        return {}
    option_name = {
        "appsi_highs": "time_limit",
        "gurobi": "TimeLimit",
        "cplex": "timelimit",
    }[solver_name]
    return {option_name: time_limit}


def _bound(results: object, objective: float | None, optimal: bool) -> float | None:
    problem = getattr(results, "problem", None)
    value = getattr(problem, "upper_bound", None)
    try:
        bound = float(value)
    except (TypeError, ValueError):
        bound = None
    if bound is not None and (bound == float("inf") or bound == float("-inf")):
        bound = None
    if bound is None and optimal:
        return objective
    return bound


def _node_count(results: object) -> int | None:
    statistics = getattr(results, "solver", None)
    branch = getattr(getattr(statistics, "statistics", None), "branch_and_bound", None)
    value = getattr(branch, "number_of_created_subproblems", None)
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def solve_model(
    instance: Instance,
    formulation: Formulation = "M2",
    *,
    solver: str = DEFAULT_SOLVER,
    relax: bool = False,
    time_limit: float | None = 3600,
) -> SolveResult:
    """Solve M1 or M2 and independently validate any integer incumbent."""
    model = build_model(instance, formulation, relax)
    optimizer = pyo.SolverFactory(solver)
    if optimizer is None or not optimizer.available(exception_flag=False):
        raise RuntimeError(
            f"Solver {solver!r} is unavailable. Install/configure it before solving."
        )
    optimizer.options.update(_solver_options(solver, time_limit))
    start = perf_counter()
    results = optimizer.solve(model, tee=False)
    elapsed = perf_counter() - start

    termination = str(results.solver.termination_condition)
    solver_status = str(results.solver.status)
    feasible = solver_status.lower() in (
        str(SolverStatus.ok).lower(),
        str(SolverStatus.warning).lower(),
    ) and termination.lower() not in (
        str(TerminationCondition.infeasible).lower(),
        str(TerminationCondition.infeasibleOrUnbounded).lower(),
        str(TerminationCondition.unbounded).lower(),
        str(TerminationCondition.error).lower(),
    )
    objective = pyo.value(model.objective, exception=False) if feasible else None
    x_values: tuple[tuple[float, ...], ...] | None = None
    y_values: tuple[float, ...] | None = None
    if objective is not None:
        x_values = tuple(
            tuple(float(pyo.value(model.x[i, j])) for j in range(instance.n))
            for i in range(instance.m)
        )
        y_values = tuple(float(pyo.value(model.y[j])) for j in range(instance.n))
        if not relax:
            validate_solution(instance, x_values, y_values)
    optimal = termination.lower() == "optimal"
    return SolveResult(
        formulation=formulation,
        solver=solver,
        objective=float(objective) if objective is not None else None,
        best_bound=_bound(results, float(objective) if objective is not None else None, optimal),
        status=solver_status,
        termination=termination,
        elapsed_seconds=elapsed,
        branch_and_bound_nodes=_node_count(results),
        x=x_values,
        y=y_values,
        relaxed=relax,
    )


def validate_solution(
    instance: Instance,
    x: tuple[tuple[float, ...], ...],
    y: tuple[float, ...],
    *,
    tolerance: float = 1e-6,
) -> None:
    """Raise ValueError unless a binary solution satisfies constraints (2)-(7)."""
    if len(x) != instance.m or any(len(row) != instance.n for row in x):
        raise ValueError("x has an invalid shape.")
    if len(y) != instance.n:
        raise ValueError("y has an invalid shape.")
    if any(not (isclose(value, 0, abs_tol=tolerance) or isclose(value, 1, abs_tol=tolerance))
           for row in x for value in row):
        raise ValueError("x must be binary.")
    if any(not (isclose(value, 0, abs_tol=tolerance) or isclose(value, 1, abs_tol=tolerance))
           for value in y):
        raise ValueError("y must be binary.")
    if sum(
        instance.p[i][j] * x[i][j]
        for i in range(instance.m)
        for j in range(instance.n)
    ) + tolerance < (1 + instance.hurdle_rate) * (
        sum(
            instance.c[i][j] * x[i][j]
            for i in range(instance.m)
            for j in range(instance.n)
        )
        + sum(instance.fixed_costs[j] * y[j] for j in range(instance.n))
    ):
        raise ValueError("The hurdle-rate constraint is violated.")
    for j in range(instance.n):
        if sum(instance.c[i][j] * x[i][j] for i in range(instance.m)) > instance.budgets[j] + tolerance:
            raise ValueError(f"The budget constraint is violated for product {j}.")
    for i in range(instance.m):
        if sum(x[i][j] for j in range(instance.n)) > instance.max_offers[i] + tolerance:
            raise ValueError(f"The customer offer limit is violated for customer {i}.")
    for j in range(instance.n):
        total_offers = sum(x[i][j] for i in range(instance.m))
        if total_offers > instance.m * y[j] + tolerance:
            raise ValueError(f"The product-linking constraint is violated for product {j}.")
        if total_offers + tolerance < instance.minimum_offers[j] * y[j]:
            raise ValueError(f"The minimum-commitment constraint is violated for product {j}.")


def brute_force_optimum(
    instance: Instance,
    *,
    tolerance: float = 1e-9,
) -> tuple[float, tuple[tuple[float, ...], ...], tuple[float, ...], int]:
    """Enumerate all offer matrices; return objective, x, y, and number of optima."""
    if instance.m * instance.n > 24:
        raise ValueError("Brute-force validation is limited to at most 24 offer decisions.")
    best_value = float("-inf")
    best_x: tuple[tuple[float, ...], ...] | None = None
    best_y: tuple[float, ...] | None = None
    optimum_count = 0
    for mask in range(1 << (instance.m * instance.n)):
        x = tuple(
            tuple(float((mask >> (i * instance.n + j)) & 1) for j in range(instance.n))
            for i in range(instance.m)
        )
        y = tuple(float(any(x[i][j] for i in range(instance.m))) for j in range(instance.n))
        try:
            validate_solution(instance, x, y, tolerance=tolerance)
        except ValueError:
            continue
        objective = sum(
            (instance.p[i][j] - instance.c[i][j]) * x[i][j]
            for i in range(instance.m)
            for j in range(instance.n)
        ) - sum(instance.fixed_costs[j] * y[j] for j in range(instance.n))
        if objective > best_value + tolerance:
            best_value, best_x, best_y, optimum_count = objective, x, y, 1
        elif isclose(objective, best_value, abs_tol=tolerance):
            optimum_count += 1
    if best_x is None or best_y is None:
        raise ValueError("The instance has no feasible solution.")
    return best_value, best_x, best_y, optimum_count
