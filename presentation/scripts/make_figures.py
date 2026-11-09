"""Build every figure and summary table used in the presentation.

Reads only the CSV files and instance files already stored in the repository:
  results/M3_small_extra_results.csv   (experiment 1, 50 instances, m <= 12)
  results/S{1,2,3}_results.csv         (experiment 2, groups S1-S3, m = 100, 200, 300)
  instances/M3_small_extra/*.json      (to count the admissible columns of M3)

Run from the repository root:
  python presentation/scripts/make_figures.py
Outputs go to presentation/figures/ (PDF figures) and presentation/figures/summary.txt.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from instance import Instance  # noqa: E402
from setcover import enumerate_columns  # noqa: E402

OUT = ROOT / "presentation" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Tecnico visual identity
BLUE, SLATE, GREY = "#009DE0", "#46555F", "#D2D3D4"
SOLVER_ORDER = ("appsi_highs", "gurobi", "cplex")
SOLVER_LABEL = {"appsi_highs": "HiGHS", "gurobi": "Gurobi", "cplex": "CPLEX"}
SOLVER_COLOR = {"appsi_highs": BLUE, "gurobi": SLATE, "cplex": "#9AA7AF"}
MODEL_COLOR = {"M1": "#9AA7AF", "M2": SLATE, "M3": BLUE}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Source Sans 3", "Source Sans Pro", "DejaVu Sans"],
    "font.size": 10,
    "axes.edgecolor": SLATE,
    "axes.labelcolor": SLATE,
    "xtick.color": SLATE,
    "ytick.color": SLATE,
    "text.color": SLATE,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GREY,
    "grid.linewidth": 0.6,
    "legend.frameon": False,
    "savefig.bbox": "tight",
})

lines: list[str] = []


def log(text: str = "") -> None:
    lines.append(text)


# ---------------------------------------------------------------- experiment 1
e1 = pd.read_csv(ROOT / "results" / "M3_small_extra_results.csv")
columns = {
    path.name: len(enumerate_columns(Instance.from_json(path)))
    for path in (ROOT / "instances" / "M3_small_extra").glob("*.json")
}
e1["columns"] = e1["instance"].map(columns)
inst1 = e1.drop_duplicates("instance")

log("EXPERIMENT 1 (M3_small_extra)")
log(f"instances={inst1.shape[0]} rows={e1.shape[0]}")
log(f"m counts={inst1.m.value_counts().sort_index().to_dict()}")
log(f"n counts={inst1.n.value_counts().sort_index().to_dict()}")
log(f"integer termination={e1.integer_termination.value_counts().to_dict()}")
log(f"LP termination={e1.lp_termination.value_counts().to_dict()}")
spread = e1.groupby("instance").integer_objective.agg(lambda s: s.max() - s.min()).max()
log(f"max spread of integer optimum over solvers and models={spread:.3e}")
lp_spread = e1.groupby(["instance", "formulation"]).lp_gap_percent.agg(lambda s: s.max() - s.min()).max()
log(f"max spread of LP gap over solvers={lp_spread:.3e}")
ref = e1[e1.solver == "gurobi"]
gap1 = ref.pivot(index="instance", columns="formulation", values="lp_gap_percent")
lp1 = ref.pivot(index="instance", columns="formulation", values="lp_objective")
opt1 = ref.groupby("instance").integer_objective.first()
order_ok = ((opt1 <= lp1.M3 + 1e-6) & (lp1.M3 <= lp1.M2 + 1e-6) & (lp1.M2 <= lp1.M1 + 1e-6)).all()
log(f"z_IP <= z_M3 <= z_M2 <= z_M1 for all instances: {order_ok}")
log(f"strict M3<M2: {(lp1.M3 < lp1.M2 - 1e-6).sum()}  strict M2<M1: {(lp1.M2 < lp1.M1 - 1e-6).sum()}")
for f in ("M1", "M2", "M3"):
    g = gap1[f]
    log(f"{f}: mean={g.mean():.2f} median={g.median():.2f} max={g.max():.2f} zero-gap={int((g < 1e-6).sum())}")
log(f"columns: min={inst1['columns'].min()} median={inst1['columns'].median():.0f} max={inst1['columns'].max()}")
t1 = e1.groupby(["solver", "formulation"]).integer_elapsed_seconds.agg(["median", "mean", "max"])
log("integer time (s) by solver and formulation:\n" + t1.round(3).to_string())
lpt1 = e1.groupby(["solver", "formulation"]).lp_elapsed_seconds.median()
log("LP time median (s):\n" + lpt1.round(3).to_string())
m3 = e1[e1.formulation == "M3"]
for s, g in m3.groupby("solver"):
    rho = g[["columns", "integer_elapsed_seconds"]].corr("spearman").iloc[0, 1]
    log(f"Spearman(columns, M3 integer time) {s}: {rho:.3f}")
wide1 = e1.pivot_table(index=["instance", "solver"], columns="formulation",
                       values="integer_elapsed_seconds").reset_index()
wide1 = wide1.merge(inst1[["instance", "columns"]], on="instance")
big = wide1[wide1["columns"] >= 1000]
log(f"instances with >=1000 columns: {big.instance.nunique()}")
log("median integer time (s), instances with >=1000 columns:\n"
    + big.groupby("solver")[["M1", "M2", "M3"]].median().round(3).to_string())

# Figure 1: LP gap distribution per formulation (solver independent)
fig, ax = plt.subplots(figsize=(6.2, 3.0))
rng = np.random.default_rng(0)
for k, f in enumerate(("M1", "M2", "M3")):
    values = gap1[f].to_numpy()
    jitter = rng.uniform(-0.18, 0.18, size=values.size)
    ax.scatter(np.full(values.size, k) + jitter, values, s=16, color=MODEL_COLOR[f],
               alpha=0.85, edgecolor="white", linewidth=0.4, zorder=3)
    ax.hlines(np.median(values), k - 0.3, k + 0.3, color="black", linewidth=1.4, zorder=4)
    ax.hlines(values.mean(), k - 0.3, k + 0.3, color="black", linewidth=1.0,
              linestyle=(0, (3, 2)), zorder=4)
    zero = int((values < 1e-6).sum())
    ax.text(k, 285, f"LP gap = 0: {zero}/50", ha="center", fontsize=9)
ax.set_xticks([0, 1, 2], ["M1", "M2", "M3"])
ax.set_xlim(-0.6, 2.6)
ax.set_ylim(-8, 305)
ax.set_ylabel("LP gap (%)")
ax.plot([], [], color="black", linewidth=1.4, label="median")
ax.plot([], [], color="black", linewidth=1.0, linestyle=(0, (3, 2)), label="mean")
ax.legend(loc="center right")
fig.savefig(OUT / "exp1_lp_gap.pdf")
plt.close(fig)

# Figure 2: (a) admissible columns vs m, (b) M3 integer time vs columns
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.1))
ax = axes[0]
ax.scatter(inst1.m, inst1["columns"], s=18, color=BLUE, edgecolor="white", linewidth=0.4, zorder=3,
           label="admissible columns")
ms = np.arange(4, 13)
ax.plot(ms, 2 ** ms - 1, color=SLATE, linewidth=1.0, linestyle="--", label=r"$2^m-1$ subsets per product")
ax.set_yscale("log")
ax.set_xlabel("number of customers $m$")
ax.set_ylabel("columns of M3")
ax.set_title("(a) size of the explicit M3", fontsize=10, loc="left")
ax.legend(loc="upper left", fontsize=8)
ax = axes[1]
for s in SOLVER_ORDER:
    g = m3[m3.solver == s]
    ax.scatter(g["columns"], g.integer_elapsed_seconds, s=16, color=SOLVER_COLOR[s],
               edgecolor="white", linewidth=0.4, label=SOLVER_LABEL[s], zorder=3)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("columns of M3")
ax.set_ylabel("integer solve time (s)")
ax.set_title("(b) M3 integer solve time", fontsize=10, loc="left")
ax.legend(loc="upper left", fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "exp1_columns_time.pdf")
plt.close(fig)

# Figure 3: median integer time per solver and formulation
fig, ax = plt.subplots(figsize=(5.4, 2.8))
width = 0.25
for k, s in enumerate(SOLVER_ORDER):
    med = [t1.loc[(s, f), "median"] for f in ("M1", "M2", "M3")]
    mx = [t1.loc[(s, f), "max"] for f in ("M1", "M2", "M3")]
    pos = np.arange(3) + (k - 1) * width
    ax.bar(pos, med, width=width, color=SOLVER_COLOR[s], label=SOLVER_LABEL[s], zorder=3)
    ax.scatter(pos, mx, marker="_", s=120, color="black", zorder=4)
ax.scatter([], [], marker="_", s=120, color="black", label="maximum")
ax.set_xticks(range(3), ["M1", "M2", "M3"])
ax.set_yscale("log")
ax.set_ylabel("integer solve time (s)")
ax.legend(ncol=4, fontsize=8, loc="upper left")
ax.set_ylim(top=12)
fig.savefig(OUT / "exp1_time_by_solver.pdf")
plt.close(fig)

# ---------------------------------------------------------------- experiment 2
GROUPS = ("S1", "S2", "S3")
GROUP_M = {"S1": 100, "S2": 200, "S3": 300}
GROUP_COLOR = {"S1": "#B8C2C8", "S2": SLATE, "S3": BLUE}
frames = []
for grp in GROUPS:
    frame = pd.read_csv(ROOT / "results" / f"{grp}_results.csv")
    frame["group"] = grp
    frames.append(frame)
e2 = pd.concat(frames, ignore_index=True)
e2["cap"] = e2.instance.str.extract(r"_(s|l)_")[0]
# Reference optimum of an instance: best objective over the 6 runs (2 models x 3 solvers).
e2["zstar"] = e2.groupby("instance").objective.transform("max")
e2["gap"] = 100 * (e2.lp_objective - e2.zstar) / e2.zstar

log("")
log("EXPERIMENT 2 (S1, S2, S3)")
log(f"rows={e2.shape[0]} instances={e2.instance.nunique()}")
log("rows per group/solver/model:\n" + e2.groupby(["group", "solver", "model"]).size().unstack().to_string())
log(f"termination={e2.termination.value_counts().to_dict()} LP={e2.lp_termination.value_counts().to_dict()}")
log(f"max time (s)={e2.elapsed_seconds.max():.2f}")
spread = e2.groupby("instance").objective.agg(lambda s: s.max() - s.min())
log(f"instances with objective spread > 1e-6: {int((spread > 1e-6).sum())}")
tol = e2[e2.mip_gap_percent.abs() > 1e-6]
log("runs with positive final MIP gap:\n" + tol[["instance", "solver", "model", "objective", "best_bound", "mip_gap_percent"]].to_string())
log("objective range by group:\n" + e2.groupby("group").objective.agg(["min", "max"]).to_string())
lpo = e2.pivot_table(index=["instance", "solver"], columns="model", values="lp_objective")
log(f"z_LP(M2) <= z_LP(M1) in all runs: {bool((lpo.M2 <= lpo.M1 + 1e-6).all())}")
log(f"z_LP >= z* in all runs: {bool((e2.lp_objective >= e2.zstar - 1e-6).all())}")
log(f"max spread of LP objective over solvers: {e2.groupby(['instance','model']).lp_objective.agg(lambda s: s.max()-s.min()).max():.3e}")
ref2 = e2[e2.solver == "gurobi"]
log("LP gap (vs z*) by group and model:\n"
    + ref2.groupby(["group", "model"]).gap.agg(["mean", "median", "min", "max"]).round(2).to_string())
log("zero LP gap counts:\n" + ref2.groupby(["group", "model"]).gap.agg(lambda s: int((s.abs() < 1e-6).sum())).to_string())
log("LP gap mean by group, n, M variant:\n"
    + ref2.groupby(["group", "n", "cap", "model"]).gap.mean().unstack().round(2).to_string())
log("LP gap mean by group and M variant:\n"
    + ref2.groupby(["group", "cap", "model"]).gap.mean().unstack().round(2).to_string())
t2 = e2.groupby(["group", "solver", "model"]).elapsed_seconds.agg(["median", "mean", "max", "sum"])
log("integer time (s):\n" + t2.round(2).to_string())
log("M2 mean time by group, M variant, solver:\n"
    + e2[e2.model == "M2"].groupby(["group", "cap", "solver"]).elapsed_seconds.mean().unstack().round(2).to_string())
w2 = e2.pivot_table(index=["group", "instance", "solver"], columns="model", values="elapsed_seconds").reset_index()
paired = w2.groupby(["group", "solver"]).apply(lambda x: pd.Series({
    "M2_faster": int((x.M2 < x.M1).sum()),
    "geomean_ratio": float(np.exp(np.log(x.M2 / x.M1).mean())),
}))
log("paired M2 vs M1:\n" + paired.round(2).to_string())
e2r = e2.reset_index(drop=True)
fastest = e2r.loc[e2r.groupby(["group", "instance", "model"]).elapsed_seconds.idxmin()]
log("fastest solver counts:\n" + fastest.groupby(["group", "model", "solver"]).size().unstack(fill_value=0).to_string())
pv = e2.pivot_table(index=["group", "instance", "model"], columns="solver", values="elapsed_seconds")
for s in ("appsi_highs", "cplex"):
    r = np.log(pv[s] / pv.gurobi)
    log(f"{s}/gurobi geometric mean time ratio by group and model:\n"
        + np.exp(r.groupby(level=[0, 2]).mean()).unstack().round(2).to_string())
log("slowest runs:\n" + e2.sort_values("elapsed_seconds", ascending=False)
    .head(6)[["instance", "solver", "model", "elapsed_seconds"]].to_string())

# Figure 4: LP gap of M1 and M2 per instance, by group
fig, axes = plt.subplots(1, 2, figsize=(9.2, 2.9))
for ax, model in zip(axes, ("M1", "M2")):
    for k, grp in enumerate(GROUPS):
        for cap, marker, dx in (("s", "o", -0.14), ("l", "^", 0.14)):
            g = ref2[(ref2.model == model) & (ref2.group == grp) & (ref2.cap == cap)]
            jitter = rng.uniform(-0.1, 0.1, size=len(g))
            ax.scatter(np.full(len(g), k) + jitter + dx, g.gap.clip(lower=0), s=14, marker=marker,
                       color=MODEL_COLOR[model], edgecolor="white", linewidth=0.4, zorder=3,
                       label=(f"$M_i$ variant {cap}" if k == 0 else None))
    ax.set_xticks(range(3), [f"{g} ($m={GROUP_M[g]}$)" for g in GROUPS])
    ax.set_title(model, fontsize=10, loc="left")
    ax.set_ylabel("LP gap (%)")
    ax.legend(fontsize=8, loc="upper left")
axes[0].set_ylim(0, 100)
axes[1].set_ylim(-0.4, 14)
fig.tight_layout()
fig.savefig(OUT / "exp2_lp_gap.pdf")
plt.close(fig)

# Figure 5: integer time distribution per solver and model, one panel per group
fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.9), sharey=True)
for ax, grp in zip(axes, GROUPS):
    positions, data, colors = [], [], []
    for k, s in enumerate(SOLVER_ORDER):
        for j, model in enumerate(("M1", "M2")):
            positions.append(k * 2.6 + j)
            data.append(e2[(e2.group == grp) & (e2.solver == s) & (e2.model == model)].elapsed_seconds.to_numpy())
            colors.append(MODEL_COLOR[model])
    bp = ax.boxplot(data, positions=positions, widths=0.7, patch_artist=True, showfliers=True,
                    medianprops={"color": "black", "linewidth": 1.1},
                    flierprops={"marker": "o", "markersize": 2.5, "markerfacecolor": SLATE, "markeredgecolor": SLATE})
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor(SLATE)
    ax.set_xticks([k * 2.6 + 0.5 for k in range(3)], [SOLVER_LABEL[s] for s in SOLVER_ORDER])
    ax.set_yscale("log")
    ax.set_title(f"{grp} ($m={GROUP_M[grp]}$)", fontsize=10, loc="left")
axes[0].set_ylabel("integer solve time (s)")
axes[0].bar([0], [0], color=MODEL_COLOR["M1"], label="M1")
axes[0].bar([0], [0], color=MODEL_COLOR["M2"], label="M2")
axes[0].legend(loc="upper right", ncol=2, fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "exp2_time_boxplot.pdf")
plt.close(fig)

# Figure 6: growth of the mean and maximum time with m
fig, axes = plt.subplots(1, 2, figsize=(9.2, 2.9), sharey=True)
ms = [GROUP_M[g] for g in GROUPS]
for ax, stat, title in ((axes[0], "mean", "(a) mean time"), (axes[1], "max", "(b) maximum time")):
    for s in SOLVER_ORDER:
        for model, style in (("M1", "--"), ("M2", "-")):
            vals = [t2.loc[(g, s, model), stat] for g in GROUPS]
            ax.plot(ms, vals, linestyle=style, marker="o", markersize=4, color=SOLVER_COLOR[s],
                    label=f"{SOLVER_LABEL[s]}, {model}")
    ax.set_yscale("log")
    ax.set_xticks(ms, [f"{m}\n{g}" for m, g in zip(ms, GROUPS)])
    ax.set_xlabel("number of customers $m$")
    ax.set_title(title, fontsize=10, loc="left")
axes[0].set_ylabel("integer solve time (s)")
handles, labels = axes[0].get_legend_handles_labels()
fig.tight_layout(rect=(0, 0.1, 1, 1))
fig.legend(handles, labels, loc="lower center", ncol=6, fontsize=7.5)
fig.savefig(OUT / "exp2_time_growth.pdf")
plt.close(fig)

# Figure 7: paired M1 vs M2 time per instance, one panel per solver, coloured by group
fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.0))
for ax, s in zip(axes, SOLVER_ORDER):
    g = w2[w2.solver == s]
    lo = min(g.M1.min(), g.M2.min()) * 0.7
    hi = max(g.M1.max(), g.M2.max()) * 1.4
    ax.plot([lo, hi], [lo, hi], color=SLATE, linewidth=0.8, linestyle="--", zorder=2)
    for grp in GROUPS:
        h = g[g.group == grp]
        ax.scatter(h.M1, h.M2, s=12, color=GROUP_COLOR[grp], edgecolor="white", linewidth=0.3,
                   zorder=3, label=grp)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("M1 time (s)")
    ax.set_ylabel("M2 time (s)")
    faster = int((g.M2 < g.M1).sum())
    ax.set_title(f"{SOLVER_LABEL[s]}: M2 faster in {faster}/162", fontsize=10, loc="left")
    ax.legend(fontsize=7.5, loc="upper left")
fig.tight_layout()
fig.savefig(OUT / "exp2_paired_time.pdf")
plt.close(fig)

(OUT / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
