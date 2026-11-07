"""E3 figures: work to 1 % vs size, refinement and stiffness number."""
import numpy as np

from common import load_json, plt, save

R = load_json("E3", "summary")["results"]
FLOPS_PER_POINT_EVAL = 100.0  # stated estimate for converting factor flops into sweep equivalents

fig, ax = plt.subplots(1, 3, figsize=(14, 3.9))
methods = sorted({r["method"] for r in R["height"]}, key=lambda s: (s.startswith("PCG"), s))
for meth in methods:
    rows = [r for r in R["height"] if r["method"] == meth and r["iters_1pct"] is not None]
    h = np.array([r["height_m"] for r in rows])
    it = np.array([r["iters_1pct"] for r in rows])
    ax[0].loglog(h, it, "o-", label=meth)
rows = [r for r in R["height"] if r["method"].startswith("PCG")]
h = np.array([r["height_m"] for r in rows])
eq = np.array([r["factor_flops"] / (FLOPS_PER_POINT_EVAL * r["point_evals_per_sweep"]) for r in rows])
ax[0].loglog(h, eq, "k--", label="direct factorization (sweep equivalents)")
hh = np.array([1, 8.0])
ax[0].loglog(hh, 1254 * hh**4, ":", color="gray", lw=0.8, label="∝ H⁴")
ax[0].loglog(hh, 45 * hh, ":", color="k", lw=0.8, label="∝ H")
ax[0].axhspan(8, 16, color="g", alpha=0.1)
ax[0].set(xlabel="wall height H [m] (width 1 m, cells 0.1 m)", ylabel="iterations to 1 % facet-force error", title="(a) structure grows at fixed cell size", ylim=(1, 1e6))
ax[0].legend(fontsize=6)

for meth in methods:
    rows = [r for r in R["refine"] if r["method"] == meth and r["iters_1pct"] is not None]
    ax[1].loglog([r["cells"] for r in rows], [r["iters_1pct"] for r in rows], "o-", label=meth)
rows = [r for r in R["refine"] if r["method"].startswith("PCG")]
cells = np.array([r["cells"] for r in rows])
ax[1].loglog(cells, [r["factor_flops"] / (FLOPS_PER_POINT_EVAL * 11 * r["cells"]) for r in rows], "k--", label="direct (sweep equivalents)")
ax[1].axhspan(8, 16, color="g", alpha=0.1)
ax[1].set(xlabel="cells (fixed 1 m × 2 m wall, refined)", ylabel="iterations to 1 %", title="(b) refinement of a fixed structure", ylim=(1, 1e6))
ax[1].legend(fontsize=6)

dm = sorted({r["method"] for r in R["dyn"]})
for meth in dm:
    for d, mk in [(0.4, "o-"), (0.1, "x")]:
        rows = [r for r in R["dyn"] if r["method"] == meth and r["dmin"] == d and r["iters_1pct"] is not None]
        lab = meth if d == 0.4 else None
        ax[2].loglog([r["pi_h"] for r in rows], [r["iters_1pct"] for r in rows], mk, label=lab)
ax[2].axhspan(8, 16, color="g", alpha=0.1)
ax[2].set(xlabel="stiffness number π_h = (c h / ℓ)²", ylabel="iterations to 1 % (first step after sudden load)",
          title="(c) dynamic step: 4×8 m (o) and 1×2 m (x) walls collapse")
ax[2].legend(fontsize=6)
save(fig, "E3_scaling")

fig, ax = plt.subplots(figsize=(5.5, 3.8))
for meth in dm:
    rows = [r for r in R["dyn"] if r["method"] == meth and r["dmin"] == 0.4 and r["iters_1pct"] is not None]
    ax.loglog([1 / r["h"] for r in rows], [r["sweeps_per_frame_60Hz_at_1pct"] for r in rows], "o-", label=meth)
ax.set(xlabel="substep rate 1/h [Hz]", ylabel="iterations per 60 Hz frame for 1 %", title="Work per frame vs substep (4×8 m wall, 0.4 m cells)")
ax.legend(fontsize=7)
save(fig, "E3_per_frame")
