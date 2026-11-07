"""E1 figures: error vs iterations against the closed forms."""
import numpy as np

from common import load_csv, load_json, plt, save

S = load_json("E1", "summary")
reg = {r["regime"]: r for r in S["summary"]}

# 1. cold start, shared rho
d = load_csv("E1", "cold_shared")
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
for c in reg["cold_shared"]["configs"]:
    m = d["cfg"] == c["cfg"]
    lab = f"k2/k1={c['ratio']:.0f}, k1/ρ={c['k1_over_rho']:g}  (r={c['r_formula']:.5f})"
    ln, = ax[0].loglog(d["iter"][m], d["force_err_rel_max"][m], label=lab)
    ax[0].loglog(d["iter"][m], np.abs(d["pred_e_lam1_overP"][m]) / np.max(np.abs([c["exact_split"]])), "k:", lw=0.8)
    ax[1].loglog(d["iter"][m], np.abs(d["x_rel_err"][m]), color=ln.get_color())
ax[0].axhline(0.01, color="r", lw=0.8, ls="--")
ax[0].axvspan(8, 16, color="g", alpha=0.08)
ax[0].set(xlabel="ALM iterations (exact primal + dual)", ylabel="max |λᵢ − λᵢ*| / max|λ*|", title="Cold start, shared ρ: force error (dotted: closed form)")
ax[1].set(xlabel="ALM iterations", ylabel="|x − x*| / x*", title="Displacement error (same contraction, smaller amplitude)")
ax[0].legend(fontsize=7)
save(fig, "E1_cold_shared")

# 2. per-spring rho
d = load_csv("E1", "cold_perspring")
fig, ax = plt.subplots(figsize=(5.5, 3.8))
for c in reg["cold_perspring"]["configs"]:
    if c["ratio"] != 2.0:
        continue
    m = d["cfg"] == c["cfg"]
    ax.loglog(d["iter"][m], d["force_err_rel_max"][m], label=f"ρᵢ = {c['rho_over_k']:g}·kᵢ  (r = 1/(1+c) = {c['r']:.4f})")
ax.axhline(0.01, color="r", lw=0.8, ls="--")
ax.set(xlabel="ALM iterations", ylabel="relative force error", title="Per-spring ρᵢ = c·kᵢ (k2/k1 = 2): r depends only on ρ/k")
ax.legend(fontsize=7)
save(fig, "E1_perspring")

# 3. ramp
d = load_csv("E1", "ramp")
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
r_of = {(c["ratio"], c["k1_over_rho"]): c["r_formula"] for c in reg["cold_shared"]["configs"]}
for (ratio, lev), r in r_of.items():
    m = (d["ratio"] == ratio) & (d["k1_over_rho"] == lev)
    kap = np.array([1.0, ratio]) / (1.0 + ratio)
    fac = abs(kap[0] - kap[1]) / 2 / kap.max()  # |ℓᵀκ| / max κ for shared ρ
    x = d["steps"][m] * d["m"][m] * (1 - r)
    ax[0].loglog(x, d["rel_err_final"][m] / fac, "o", ms=3.5, label=f"k2/k1={ratio:.0f}, k1/ρ={lev:g}")
    ax[1].loglog(d["rel_err_pred"][m], d["rel_err_final"][m], "o", ms=3.5)
xx = np.logspace(-3, 4.7, 200)
ax[0].loglog(xx, (1 - np.exp(-xx)) / xx, "k-", lw=0.8, label="r→1 limit (1−e^{−x})/x")
ax[0].set(xlabel="x = n·m·(1−r)  (ramp steps × iterations/step × (1−r))", ylabel="normalised distribution error", title="Linear load ramp with warm start")
ax[0].legend(fontsize=7)
lim = [1e-9, 1]
ax[1].loglog(lim, lim, "k-", lw=0.6)
ax[1].set(xlabel="closed-form prediction", ylabel="measured", title="Ramp: measured vs closed form (all 72 runs)")
save(fig, "E1_ramp")

# 4. step change in k2: policies
d = load_csv("E1", "stepK")
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
for j, ratio in enumerate([2.0, 10.0]):
    for c in reg["stepK"]["configs"]:
        if c["ratio"] != ratio or c["k1_over_rho"] != 1000.0:
            continue
        m = d["cfg"] == c["cfg"]
        ls = {"keep": "-", "rescale": "--", "reset": ":"}[c["policy"]]
        ax[j].loglog(d["iter"][m], d["force_err_rel_max"][m], ls, label=f"k2×{c['k2_factor']}, {c['policy']}")
    ax[j].set(xlabel="ALM iterations after the change", ylabel="relative force error", title=f"Step change of k2 (k2/k1={ratio:.0f}, k1/ρ=1000)")
    ax[j].legend(fontsize=7)
save(fig, "E1_stepK")

# 5. E1-dyn
d = load_csv("E1", "dyn")
fig, ax = plt.subplots(figsize=(5.5, 3.8))
for lev in [10.0, 1000.0]:
    for mm in [1.0, 8.0, 16.0]:
        m = (d["k1_over_mu"] == lev) & (d["m_iters"] == mm)
        ax.semilogy(d["t"][m], np.maximum(d["onestep_split_err"][m], 1e-17), label=f"k1/μ={lev:g}, {mm:.0f} it/step")
ax.axhline(0.01, color="r", lw=0.8, ls="--")
ax.set(xlabel="time [s] (h = 1/480 s, step load at t=0)", ylabel="force error vs exact step from same state", title="E1-dyn: ρ = μ = m/h²")
ax.legend(fontsize=7)
save(fig, "E1_dyn")
