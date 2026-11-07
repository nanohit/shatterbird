"""Phase 0b / 1 figures: geometry contract and spectrum (results/phase0b/geometry-spectrum)."""
import json
import os

import numpy as np
import pandas as pd
from matplotlib.collections import PolyCollection

from common import RES, plt, save

D = os.path.join(RES, "phase0b", "geometry-spectrum")
S = json.load(open(os.path.join(D, "summary.json")))["results"]
cells = pd.read_csv(os.path.join(D, "cells.csv"))
cases = pd.read_csv(os.path.join(D, "cases.csv"))
polys = pd.read_csv(os.path.join(D, "polys2d.csv"))

# ---------------------------------------------------------------- cause
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
v0 = cells[(cells.variant == "V0") & (cells.cell == 1)]
is3 = v0.case.str.startswith("E10") | v0.case.str.startswith("cube2")
for sel, col, lab in [(is3, "tab:blue", "3D support cells"), (~is3, "tab:orange", "2D support cells")]:
    s = v0[sel & (v0.support == 1)]
    ax[0].scatter(s.d_sup, s.L_hat, s=6, color=col, alpha=0.5, label=lab)
inter = v0[v0.support == 0]
ax[0].axhspan(inter.L_hat.min(), inter.L_hat.max(), color="0.85", label="interior cells (range)")
d = np.logspace(-4, 0.3, 50)
ax[0].plot(d, 2.0 / (d * (d + 0.6)), "k--", lw=0.8, label=r"$2/(d(d+0.6))$ (rocking estimate)")
top = cells[(cells.variant == "V0") & (cells.kin_top > 0.9)]
ax[0].scatter(top.d_sup, top.L_hat, s=40, facecolors="none", edgecolors="r", label="top-mode cell")
ax[0].set(xscale="log", yscale="log", xlabel=r"generator–support distance $d/d_{\min}$", ylabel=r"cell block $\hat L_i=\lambda_{\max}(B_{ii})\,d_{\min}^2/c^2$",
          title="Control meshes: local frequency of support cells", xlim=(5e-5, 2))
ax[0].legend(fontsize=7)
order = ["V0", "V1 a0.25", "V1 a0.35", "V1 a0.5", "V0-free"]
dev = cases[cases.variant.isin(order) & ~cases.case.str.startswith("cube2") & ~cases.case.str.startswith("E9 contact") & ~cases.case.str.contains("4.0x8.0")]
names = list(dict.fromkeys(dev.case))
cols = {"V0": "tab:red", "V1 a0.25": "tab:orange", "V1 a0.35": "gold", "V1 a0.5": "tab:green", "V0-free": "k"}
for k, v in enumerate(order):
    s = dev[dev.variant == v].set_index("case").reindex(names)
    ax[1].scatter(np.arange(len(names)) + (k - 2) * 0.1, s.dt_crit_hat, s=18, color=cols[v], label=v, marker="x" if v == "V0-free" else "o")
ax[1].set_xticks(range(len(names)))
ax[1].set_xticklabels(names, rotation=45, ha="right", fontsize=7)
ax[1].set(yscale="log", ylabel=r"$\widehat{\Delta t}_{crit}=\Delta t\,c/d_{\min}$ (Ritz)", title="Critical step: control, support exclusion α, supports removed")
ax[1].legend(fontsize=7, ncol=3)
save(fig, "P0b_cause")

# ---------------------------------------------------------------- 2D mode maps
pairs = [("E2 block s1", "V0"), ("E2 block s1", "V1 a0.5"), ("E7 cantilever", "V0"), ("E7 cantilever", "V1 a0.5")]
fig, axs = plt.subplots(2, 2, figsize=(12, 5.5), gridspec_kw={"height_ratios": [1, 0.7]})
for a, (cs, vr) in zip(axs.flat, pairs):
    p = polys[(polys.case == cs) & (polys.variant == vr)]
    verts = [np.array([[float(t) for t in q.split()] for q in v.split(";")]) for v in p.verts]
    pc = PolyCollection(verts, array=p.kin_top.values, cmap="magma_r", edgecolors="0.6", linewidths=0.3)
    pc.set_clim(0, max(0.3, p.kin_top.max()))
    a.add_collection(pc)
    a.autoscale_view()
    a.set_aspect("equal")
    a.grid(False)
    row = cases[(cases.case == cs) & (cases.variant == vr)].iloc[0]
    a.set_title(f"{cs}, {vr}: top mode, Δt_crit = {row.dt_crit_us:.2f} µs (ĉ {row.dt_crit_hat:.3f})", fontsize=8)
    fig.colorbar(pc, ax=a, shrink=0.8, label="kinetic share")
save(fig, "P0b_modes2d")

# ---------------------------------------------------------------- bounds
runs = []
for c in S["dev3d"] + S["dev2d"]:
    runs += c["runs"]
for h in S["held_out"]:
    for s in h["seeds"]:
        runs += [s["control"], s["variant"]]
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
keys = [("gershgorin_over_theta", "Gershgorin"), ("cw_over_theta", "Collatz–Wielandt"), ("split_t1_over_theta", "split, t=1"), ("split_over_theta", "split, balanced")]
for k, (key, lab) in enumerate(keys):
    v = np.array([r["gap_ratios"][key] for r in runs])
    ax[0].scatter(np.full(len(v), k) + np.random.default_rng(k).uniform(-0.15, 0.15, len(v)), v, s=8, alpha=0.6)
syl = np.array([r["bounds_hat"]["sylvester_upper"] / r["theta1_hat"] for r in runs])
ax[0].scatter(np.full(len(syl), 4), syl, s=8)
ax[0].set_xticks(range(5))
ax[0].set_xticklabels([k[1] for k in keys] + ["Sylvester"], fontsize=8)
ax[0].set(ylabel=r"$U/\theta_1$", title="Upper bounds over the Ritz value (all runs)")
ax[0].axhline(2, color="k", lw=0.6, ls=":")
bl = np.array([r["gap_ratios"]["theta_over_block_lower"] for r in runs])
loc = np.array([r["top_mode"]["kin_top1"] for r in runs])
ax[1].scatter(loc, bl, s=10)
ax[1].set(xlabel="kinetic share of the most-participating cell (top mode)", ylabel=r"$\theta_1/\max_i L_i$", title="Per-cell lower bound is tight for single-cell modes")
save(fig, "P0b_bounds")

# ---------------------------------------------------------------- held-out
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
kappa = S["contract"]["kappa"]
xs, lab = [], []
for i, h in enumerate(S["held_out"]):
    for j, s in enumerate(h["seeds"]):
        x = 3 * i + j
        xs.append(x)
        lab.append(f"{h['dmin']} / {s['seed']}")
        ax[0].scatter(x, s["control"]["dt_hat"]["crit_ritz"], color="tab:red", s=18, label="control, Ritz" if x == 0 else None)
        ax[0].scatter(x, s["variant"]["dt_hat"]["crit_ritz"], color="tab:green", s=18, label="contract, Ritz" if x == 0 else None)
        ax[0].scatter(x, s["variant"]["dt_hat"]["safe_sylvester"], color="tab:green", marker="_", s=80, label="contract, 0.8·Sylvester" if x == 0 else None)
        ax[0].scatter(x, s["variant"]["dt_hat"]["safe_split"], color="tab:blue", marker="_", s=80, label="contract, 0.8·split" if x == 0 else None)
ax[0].axhline(0.8 * 2 / kappa, color="tab:blue", ls=":", lw=0.8, label=f"guaranteed by κ = {kappa}")
ax[0].set_xticks(xs)
ax[0].set_xticklabels(lab, rotation=45, fontsize=7)
ax[0].set(yscale="log", ylabel=r"$\Delta t\,c/d_{\min}$", xlabel=r"$d_{\min}$ [m] / seed (cube 2 m, held out)", title="Held-out: critical and safe steps")
ax[0].legend(fontsize=7)
hc = cells[cells.case.str.startswith("cube2") & (cells.cell == 1)]
bins = np.logspace(0, 5, 60)
ax[1].hist(hc[hc.variant == "V0"].U_hat, bins=bins, alpha=0.5, color="tab:red", label="control cells")
ax[1].hist(hc[hc.variant == "V-final"].U_hat, bins=bins, alpha=0.6, color="tab:green", label="contract cells")
ax[1].axvline(kappa**2, color="k", lw=0.8, ls="--", label=r"$\kappa^2$")
ax[1].set(xscale="log", yscale="log", xlabel=r"per-cell certificate $\hat U_i$", ylabel="cells", title="Held-out: local certificates")
ax[1].legend(fontsize=7)
save(fig, "P0b_heldout")

# ---------------------------------------------------------------- refinement (post-hoc)
p = os.path.join(D, "posthoc_refinement.csv")
if os.path.exists(p):
    r = pd.read_csv(p)
    gname = {0: "control (plain RSA)", 1: "α = 0.5 (contract)", 2: "α = 0.6"}
    gcol = {0: "tab:red", 1: "tab:green", 2: "tab:purple"}
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    for dim, a_f, a_c, L, title in [(3, ax[0], ax[1], 2.0, "cube 2 m, clamped base"), (2, ax[2], None, 1.2, "2D cantilever 8×1.2 m")]:
        q = r[r.dim == dim]
        for g in sorted(q.gen.unique()):
            s = q[q.gen == g].groupby("dmin")
            x = np.array(sorted(q.dmin.unique())) / L
            for col, axx in [("f0", a_f), ("compliance", a_c)]:
                if axx is None:
                    continue
                m = s[col].mean().values
                lo, hi = s[col].min().values, s[col].max().values
                axx.errorbar(x, m, yerr=[m - lo, hi - m], color=gcol[g], marker="o", ms=3, capsize=2, label=gname[g])
        a_f.set(xlabel=r"$d_{\min}/L$", ylabel="first frequency [Hz]", title=f"{title}: mode 1 (3 seeds)")
        a_f.legend(fontsize=7)
        if a_c is not None:
            a_c.set(xlabel=r"$d_{\min}/L$", ylabel="compliance $f^\\top u$ [J]", title=f"{title}: compliance")
    save(fig, "P0b_refinement")
