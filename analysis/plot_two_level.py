"""Phase 0b / 2 figures: two-level solution of the step system (results/phase0b/two-level)."""
import json
import os

import numpy as np
import pandas as pd

from common import RES, plt, save

D = os.path.join(RES, "phase0b", "two-level")
S = json.load(open(os.path.join(D, "summary.json")))
R = S["results"]
rows = []
for r in R:
    for ro in r["runs"]:
        rows.append(dict(case=r["case"], system=r["system"], rhs=r["rhs"], cells=r["cells"], method=ro["method"], it=ro["iters_to_target"],
                         cens=ro["censored"], aeq=ro["work_at_target"]["A_equivalents"], phases=ro["work_at_target"]["phases"]))
T = pd.DataFrame(rows)
METH = [("PCG-J", "k", "o"), ("TL-PR-s8-GS", "tab:orange", "s"), ("TL-SA-s8-CH", "tab:green", "^"), ("PTL-PR-s8-GS", "tab:purple", "D"), ("PTL-SA-s8-CH", "tab:red", "v")]

# ---------------------------------------------------------------- scaling with height / refinement
fig, ax = plt.subplots(1, 3, figsize=(14, 3.9))
for m, col, mk in METH:
    for k, (sel, xs, xl) in enumerate([
        (lambda h: f"E3 wall 1x{int(h)} d0.1 ", [1.0, 2.0, 4.0, 8.0], "wall height [m] (width 1 m, d_min 0.1)"),
        (lambda d: f"E3 wall 1x4 d{d} ", [0.2, 0.1, 0.05], "d_min [m] (wall 1 × 4 m)"),
    ]):
        its, aeq = [], []
        for x in xs:
            q = T[T.case.str.startswith(sel(x)) & (T.system == "IE h=1/480") & (T.rhs == "cold") & (T.method == m)]
            its.append(q.it.mean() if q.cens.sum() == 0 else np.nan)
            aeq.append(q.aeq.mean() if q.cens.sum() == 0 else np.nan)
        if k == 0:
            ax[0].plot(xs, its, marker=mk, color=col, label=m)
            ax[2].plot(xs, aeq, marker=mk, color=col, label=m)
        else:
            ax[1].plot(xs, its, marker=mk, color=col, label=m)
ax[0].set(xscale="log", yscale="log", xlabel="wall height [m] (width 1 m, d_min 0.1)", ylabel="iterations to all targets", title="Cold start, IE h = 1/480 (mean of 2 seeds)")
ax[1].set(xscale="log", yscale="log", xlabel="d_min [m] (wall 1 × 4 m)", title="Refinement at fixed aggregate size")
ax[2].set(xscale="log", yscale="log", xlabel="wall height [m]", ylabel="A-equivalents to target", title="Work incl. coarse solves (A-eq)")
for a in ax:
    a.legend(fontsize=7)
save(fig, "P0b2_scaling")

# ---------------------------------------------------------------- closed loop
c = pd.read_csv(os.path.join(D, "dyn_closed.csv"))
fig, ax = plt.subplots(1, 3, figsize=(14, 3.9), sharey=True)
for a, traj in zip(ax, ["BDF2 h=1/480", "BDF2 h=1/60", "Newmark h=1/480"]):
    for (m, col) in [("TL-PR-s8-GS", "tab:orange"), ("TL-SA-s8-CH", "tab:green"), ("PTL-SA-s8-CH", "tab:red")]:
        for guess, ls in [("extrap", "-"), ("prev", ":")]:
            ks, vals = [], []
            for k in [1, 2, 4, 8, 16, 32]:
                q = c[(c.traj == traj) & (c.method == f"{m} {guess}") & (c.k == k) & (c.step >= 3)]
                if len(q):
                    ks.append(k)
                    vals.append(q.F_traj.max())
            if ks:
                a.plot(ks, vals, ls, marker="o", ms=3, color=col, label=f"{m}, {guess}")
    a.axhline(0.01, color="k", lw=0.6, ls="--")
    a.set(xscale="log", yscale="log", xlabel="cycles / PCG iterations per step (k)", title=f"Closed loop, {traj}: routine steps ≥ 3", ylim=(1e-13, 1e13))
ax[0].set(ylabel="max force error / trajectory max force")
ax[0].legend(fontsize=6)
save(fig, "P0b2_closed_loop")

# ---------------------------------------------------------------- cost comparison per solve
fig, ax = plt.subplots(figsize=(10, 3.8))
labels, vals = [], []
for r in R:
    if r["rhs"] != "cold" or r["system"] != "IE h=1/480":
        continue
    if not (r["case"] in ("E3 wall 1x2 d0.1 s1", "E3 wall 1x8 d0.1 s1", "E3 wall 1x4 d0.05 s1") or (r["case"].startswith("3D") and r["case"].endswith("s1"))):
        continue
    nA = r["nnz_A"]
    ref = r["reference"]
    ch = [x for x in r["runs"] if x["method"] == "PTL-SA-s8-CH"][0]
    lv = [l for l in r["levels"] if l["s"] == 8 and l["kind"] == "Sa"][0]
    labels.append(r["case"].replace(" s1", "").replace("E3 wall ", "2D ").replace("3D ", "3D "))
    vals.append([ref["flops"] / (2 * nA), 4 * ref["nnz_L"] / (2 * nA), lv["flops_factor"] / (2 * nA), ch["work_at_target"]["A_equivalents"]])
vals = np.array(vals)
x = np.arange(len(labels))
for k, (lab, col) in enumerate([("direct: factorisation", "0.6"), ("direct: solve (factor reused)", "tab:blue"), ("two-level SA-s8: coarse factorisation", "tab:orange"), ("PTL-SA-s8-CH: solve to all targets", "tab:red")]):
    ax.bar(x + (k - 1.5) * 0.2, vals[:, k], 0.2, color=col, label=lab)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=7)
ax.set(yscale="log", ylabel="A-equivalents (2 nnz(A) flops each)", title="Cost per event / per solve, IE h = 1/480, cold RHS")
ax.legend(fontsize=7)
save(fig, "P0b2_cost")

# ---------------------------------------------------------------- open loop per step
o = pd.read_csv(os.path.join(D, "dyn_open.csv"))
o["iters"] = pd.to_numeric(o.iters, errors="coerce")
fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
q = o[(o.traj == "BDF2 h=1/480") & (o.method == "PTL-SA-s8-CH")]
for guess, col in [("extrap", "tab:red"), ("qhat", "tab:green"), ("prev", "0.5")]:
    g = q[q.predictor == guess]
    ax[0].plot(g.t, g.F0, color=col, lw=0.9, label=f"{guess}: guess error (step-relative)")
    ax[1].plot(g.t, g.iters, color=col, lw=0.9, label=guess)
ax[0].axhline(0.01, color="k", ls="--", lw=0.6)
ax[0].set(yscale="log", xlabel="t [s]", ylabel="force error of the initial guess", title="Open loop, BDF2 h = 1/480, push at t = 0")
ax[1].set(xlabel="t [s]", ylabel="PTL-SA-s8-CH iterations to all targets", title="Iterations per routine step (exact history)")
ax[0].legend(fontsize=7)
ax[1].legend(fontsize=7)
save(fig, "P0b2_open_loop")
