"""E6 figures: P–CMOD / P–δ for 3 resolutions × 3 tessellations, damage maps, budgets."""
import os

import numpy as np

from common import RES, load_csv, load_json, plt, save

S = load_json("E6", "summary")
runs = S["s0_runs"]
c = load_csv("E6", "s0_curves")
cols = {0.008: "tab:blue", 0.0056: "tab:orange", 0.004: "tab:green"}
fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
for r in runs:
    m = c["run"] == r["run"]
    lab = f"d_min = {r['dmin']*1e3:.1f} mm" if r["seed"] == 1 else None
    ax[0].plot(c["cmod"][m] * 1e3, c["load"][m] / 1e3, color=cols[r["dmin"]], lw=0.9, label=lab)
    ax[1].plot(c["disp"][m] * 1e3, c["load"][m] / 1e3, color=cols[r["dmin"]], lw=0.9, label=lab)
for r in S["step_refinement"]:
    m = c["run"] == r["run"]
    ax[0].plot(c["cmod"][m] * 1e3, c["load"][m] / 1e3, "k:", lw=0.8, label=f"Δd_max = {r['dd_max']} (5.6 mm, seed 1)")
ax[0].set(xlabel="CMOD [mm]", ylabel="P [kN]", title="Notched beam, S0, 3 resolutions × 3 tessellations")
ax[0].legend(fontsize=7)
ax[1].set(xlabel="load-point displacement [mm]", ylabel="P [kN]", title="P–δ", xlim=(0, 0.5))
d = np.array([r["dmin"] for r in runs]) * 1e3
ax[2].plot(d, [r["peak_load_N"] / 1e3 for r in runs], "o", label="peak load [kN]")
ax[2].plot(d, [r["D_over_Gc_Alig"] for r in runs], "s", label="D / (G_c A_lig)")
ax[2].plot(d, [r["W_over_Gc_Alig"] for r in runs], "^", label="W_ext / (G_c A_lig)")
ax[2].set(xlabel="d_min [mm]", title="Objectivity across resolutions", ylim=(0, 3.5))
ax[2].legend(fontsize=7)
save(fig, "E6_beam")

# damage maps: seed 1 of each resolution
fig, ax = plt.subplots(1, 3, figsize=(11, 4.6))
for k, r in enumerate([r for r in runs if r["seed"] == 1]):
    p = os.path.join(RES, "E6", f"damage_run{int(r['run'])}.csv")
    if not os.path.exists(p):
        continue
    dm = load_csv("E6", f"damage_run{int(r['run'])}")
    for i in range(len(dm["x0"])):
        dd = min(1.0, dm["d_max"][i])
        ax[k].plot([dm["x0"][i], dm["x1"][i]], [dm["y0"][i], dm["y1"][i]], color=plt.cm.magma_r(0.15 + 0.85 * dd), lw=0.6 + 2.0 * dd)
    ax[k].axvline(0.25, ymax=0.5, color="b", lw=1.0)
    ax[k].set(xlim=(0.17, 0.33), ylim=(0, 0.1), aspect="equal", title=f"d_min = {r['dmin']*1e3:.1f} mm (dark: d→1; blue: notch)")
save(fig, "E6_damage")

if S.get("budget"):
    b = load_csv("E6", "disp_control")
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    for k, r in enumerate(S["budget"]):
        m = b["run"] == k
        ax.plot(b["disp"][m] * 1e3, b["load"][m] / 1e3, label=f"{r['solver']}: peak {r['peak_load_N']/1e3:.2f} kN, D {r['D_J']:.2f} J")
    ax.set(xlabel="load-point displacement [mm]", ylabel="P [kN]", title="Displacement control, 8 mm mesh: S0 vs S1 budgets")
    ax.legend(fontsize=6)
    save(fig, "E6_budget")
