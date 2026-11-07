"""E5 figures: facet law paths, step refinement, chains (snap-back)."""
import numpy as np

from common import load_csv, load_json, plt, save

R = load_json("E5", "summary")
cfg = R["meta"]["config"]
wc, gca = cfg["wc"], cfg["GcA"]
d = load_csv("E5", "facet_paths")
fig, ax = plt.subplots(1, 3, figsize=(13, 3.7))
for cid, lab in [(0, "monotonic"), (1, "cyclic + compression")]:
    m = d["case"] == cid
    ax[0].plot(d["dn"][m] / wc, d["force_n"][m] / 1e3, label=lab)
ax[0].set(xlabel="normal jump δn / w_c", ylabel="normal force [kN]", title="SB-IL1, single facet (ℓ = l = 20 mm)")
ax[0].legend()
for cid, lab in [(2, "φ=0°"), (3, "φ=30°"), (4, "φ=60°"), (5, "φ=90°"), (6, "non-proportional")]:
    m = d["case"] == cid
    ax[1].plot(np.arange(m.sum()) / m.sum(), d["D"][m] / gca, label=lab)
ax[1].set(xlabel="path parameter", ylabel="D / (G_c A)", title="Dissipation along mixed-mode paths → 1")
ax[1].legend(fontsize=7)
mono = R["results"]["monotonic_refinement"]
n = [r["steps"] for r in mono]
ax[2].loglog(n, [abs(r["max_abs_balance_residual_over_GcA"]) for r in mono], "o-", label="max |W_ext − U − D| / G_cA")
ax[2].loglog(n, [abs(1 - r["W_ext"] / gca) for r in mono], "s--", label="|1 − W_ext/G_cA| at separation")
ax[2].set(xlabel="load steps to 1.2 w_c", ylabel="relative", title="Energy ledger vs step refinement")
ax[2].legend(fontsize=7)
save(fig, "E5_facet")

c = load_csv("E5", "chains")
fig, ax = plt.subplots(1, 2, figsize=(10, 3.7))
labels = {0: "N=3, end disp.", 1: "N=3, opening", 2: "N=9, end disp.", 3: "N=9, opening", 4: "N=31, end disp. (fails)", 5: "N=31, opening"}
for cid, lab in labels.items():
    m = c["case"] == cid
    if not m.any():
        continue
    ls = "-" if cid % 2 else "--"
    ax[0].plot(c["end_disp"][m] * 1e6, c["load"][m] / 1e3, ls, label=lab)
    ax[1].plot(c["weak_opening"][m] * 1e6, c["load"][m] / 1e3, ls, label=lab)
ax[0].set(xlabel="end displacement [µm]", ylabel="load [kN]", title="Series bars: structural snap-back for L > 2 l_ch")
ax[1].set(xlabel="weak-facet opening [µm]", ylabel="load [kN]", title="Same runs vs local opening")
ax[0].legend(fontsize=7)
save(fig, "E5_chains")
