"""E2 figures: force error vs iterations (cold start, support removal)."""
import numpy as np

from common import load_csv, load_json, plt, save

S = load_json("E2", "summary")
names = S["methods"]
show = ["shared c=0.01", "shared c=0.1", "shared c=1", "perspring c=0.01", "perspring c=0.1", "perspring c=1", "S3 penalty", "PCG block-Jacobi"]

for name, title in [("cold", "cold start"), ("support_removal", "after removing 40 % of the support facets")]:
    d = load_csv("E2", name)
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    for i, n in enumerate(names):
        if n not in show:
            continue
        m = d["method"] == i
        if not m.any():
            continue
        ls = "-" if "shared" in n else ("--" if "perspring" in n else ":")
        if n.startswith("PCG"):
            ls = "-."
        ax[0].loglog(d["iter"][m], d["f_dual_l2"][m], ls, label=n)
        if not n.startswith("PCG"):
            ax[1].loglog(d["iter"][m], d["f_const_l2"][m], ls, label=n)
            ax[2].loglog(d["iter"][m], np.maximum(d["reaction"][m], 1e-12), ls, label=n)
    for a in ax:
        a.axhline(0.01, color="r", lw=0.8, ls="--")
        a.axvspan(8, 16, color="g", alpha=0.08)
        a.set_xlabel("iterations (S1: sweep + dual; S3: sweep; PCG: iteration)")
    ax[0].set(ylabel="facet-force error (λ for S1)", title=f"E2 {title}: multiplier forces")
    ax[1].set(ylabel="facet-force error (k·C(q))", title="constitutive forces from positions")
    ax[2].set(ylabel="support-reaction error", title="total support reaction")
    ax[0].legend(fontsize=6)
    save(fig, f"E2_{name}")
