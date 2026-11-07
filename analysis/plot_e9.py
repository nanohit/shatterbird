"""E9 figures: impact damage vs time step; contact force histories."""
import numpy as np

from common import load_csv, load_json, plt, save

R = load_json("E9", "summary")["runs"]
s = load_csv("E9", "series")
fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
groups = sorted({(r["v0"], r["kc"]) for r in R})
for v0, kc in groups:
    rs = [r for r in R if r["v0"] == v0 and r["kc"] == kc]
    ref = [r for r in rs if r["label"] == "reference"][0]
    co = [r for r in rs if r["label"] == "coarse"]
    ev = [r for r in rs if r["label"].startswith("event")]
    ln, = ax[0].semilogx([r["h"] for r in co], [r["D_material_J"] / ref["D_material_J"] for r in co], "o-", label=f"v0={v0:g} m/s, k_c={kc:.0e}")
    ax[0].semilogx([1 / 480] * len(ev), [r["D_material_J"] / ref["D_material_J"] for r in ev], "*", ms=10, color=ln.get_color())
    ax[1].loglog([r["h"] for r in co], [r["peak_contact_force_N"] / ref["peak_contact_force_N"] for r in co], "o-", color=ln.get_color())
ax[0].axhline(1, color="k", lw=0.6)
ax[0].set(xlabel="time step h [s] (★: 1/480 with event-triggered substeps t_c/20, t_c/5)", ylabel="fracture dissipation / resolved reference",
          title="Impact damage vs time step (implicit Euler)")
ax[0].legend(fontsize=6)
ax[1].set(xlabel="time step h [s]", ylabel="peak contact force / reference", title="Peak contact force")
for r in R:
    if r["v0"] == 10 and r["kc"] == 1e8:
        m = s["run"] == r["run"]
        lab = r["label"] if r["label"] != "coarse" else f"coarse h=1/{round(1 / r['h'])}"
        ax[2].plot(s["t"][m] * 1e3, s["contact_force"][m] / 1e3, ".-" if r["label"] != "reference" else "-", ms=3, lw=0.8, label=lab)
ax[2].set(xlim=(0, 4), xlabel="t [ms]", ylabel="contact force [kN]", title="v0 = 10 m/s, k_c = 1e8 N/m")
ax[2].legend(fontsize=6)
save(fig, "E9_impact")
