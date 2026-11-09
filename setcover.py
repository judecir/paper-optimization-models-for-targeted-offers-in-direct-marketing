"""Explicit-column set-covering formulation M3 for small instances."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from time import perf_counter

import pyomo.environ as pyo

from instance import Instance
from models import DEFAULT_SOLVER, SolveResult, _bound, _node_count, _solver_options, validate_solution

MAX_CUSTOMERS_FOR_ENUMERATION = 12


@dataclass(frozen=True)
class Column:
    """One admissible campaign and its objective and hurdle coefficients."""

    product: int
    customers: tuple[int, ...]
    objective_coefficient: float
    hurdle_coefficient: float


def enumerate_columns(
    instance: Instance,
    *,
    max_customers: int = MAX_CUSTOMERS_FOR_ENUMERATION,
) -> tuple[Column, ...]:
    """Enumerate nonempty subsets meeting each product's budget and MQC."""
    if instance.m > max_customers:
        raise ValueError(
            f"M3 column enumeration is limited to m <= {max_customers}; got m={instance.m}."
        )
    columns: list[Column] = []
    for j in range(instance.n):
        for size in range(1, instance.m + 1):
            for subset in combinations(range(instance.m), size):
                if size < instance.minimum_offers[j]:
                    continue
                variable_cost = sum(instance.c[i][j] for i in subset)
                if variable_cost > instance.budgets[j] + 1e-9:
                    continue
                objective = sum(instance.p[i][j] - instance.c[i][j] for i in subset)
                objective -= instance.fixed_costs[j]
                hurdle = sum(
                    instance.p[i][j] - (1 + instance.hurdle_rate) * instance.c[i][j]
                    for i in subset
                )
                hurdle -= (1 + instance.hurdle_rate) * instance.fixed_costs[j]
                columns.append(Column(j, subset, objective, hurdle))
    return tuple(columns)


def build_setcover_model(
    instance: Instance,
    *,
    relax: bool = False,
) -> tuple[pyo.ConcreteModel, tuple[Column, ...]]:
    """Build M3 and return its explicit columns."""
    columns = enumerate_columns(instance)
    model = pyo.ConcreteModel(name="M3")
    model.K = pyo.RangeSet(0, len(columns) - 1) if columns else pyo.Set(initialize=[])
    model.I = pyo.RangeSet(0, instance.m - 1)
    model.J = pyo.RangeSet(0, instance.n - 1)
    model.z = pyo.Var(model.K, domain=pyo.UnitInterval if relax else pyo.Binary)
    model.objective = pyo.Objective(
        expr=sum(columns[k].objective_coefficient * model.z[k] for k in model.K),
        sense=pyo.maximize,
    )
    model.hurdle = pyo.Constraint(
        rule=lambda m: (
            sum(columns[k].hurdle_coefficient * m.z[k] for k in m.K) >= 0
            if columns
            else pyo.Constraint.Feasible
        )
    )

    def customer_limit_rule(m: pyo.ConcreteModel, i: int) -> object:
        customer_columns = [k for k in m.K if i in columns[k].customers]
        if not customer_columns:
            return pyo.Constraint.Feasible
        return sum(m.z[k] for k in customer_columns) <= instance.max_offers[i]

    def product_limit_rule(m: pyo.ConcreteModel, j: int) -> object:
        product_columns = [k for k in m.K if columns[k].product == j]
        if not product_columns:
            return pyo.Constraint.Feasible
        return sum(m.z[k] for k in product_columns) <= 1

    model.customer_limit = pyo.Constraint(model.I, rule=customer_limit_rule)
    model.product_limit = pyo.Constraint(model.J, rule=product_limit_rule)
    return model, columns


def solve_setcover(
    instance: Instance,
    *,
    solver: str = DEFAULT_SOLVER,
    relax: bool = False,
    time_limit: float | None = 3600,
) -> SolveResult:
    """Solve M3 and validate the projected integer solution independently."""
    model, columns = build_setcover_model(instance, relax=relax)
    optimizer = pyo.SolverFactory(solver)
    if optimizer is None or not optimizer.available(exception_flag=False):
        raise RuntimeError(f"Solver {solver!r} is unavailable.")
    if not columns:
        x = tuple(tuple(0.0 for _ in range(instance.n)) for _ in range(instance.m))
        y = tuple(0.0 for _ in range(instance.n))
        if not relax:
            validate_solution(instance, x, y)
        return SolveResult(
            formulation="M3",
            solver=solver,
            objective=0.0,
            best_bound=0.0,
            status="ok",
            termination="optimal",
            elapsed_seconds=0.0,
            branch_and_bound_nodes=0,
            x=x,
            y=y,
            relaxed=relax,
        )
    optimizer.options.update(_solver_options(solver, time_limit))
    start = perf_counter()
    results = optimizer.solve(model, tee=False)
    elapsed = perf_counter() - start
    termination = str(results.solver.termination_condition)
    solver_status = str(results.solver.status)
    objective_value = pyo.value(model.objective, exception=False)
    objective = float(objective_value) if objective_value is not None else None
    x: tuple[tuple[float, ...], ...] | None = None
    y: tuple[float, ...] | None = None
    if objective is not None:
        x_matrix = [[0.0] * instance.n for _ in range(instance.m)]
        y_vector = [0.0] * instance.n
        for k in model.K:
            weight = float(pyo.value(model.z[k]))
            col = columns[k]
            y_vector[col.product] += weight
            for i in col.customers:
                x_matrix[i][col.product] += weight
        x = tuple(tuple(row) for row in x_matrix)
        y = tuple(y_vector)
        if not relax:
            validate_solution(instance, x, y)
    optimal = termination.lower() == "optimal"
    return SolveResult(
        formulation="M3",
        solver=solver,
        objective=objective,
        best_bound=_bound(results, objective, optimal),
        status=solver_status,
        termination=termination,
        elapsed_seconds=elapsed,
        branch_and_bound_nodes=_node_count(results),
        x=x,
        y=y,
        relaxed=relax,
    )
