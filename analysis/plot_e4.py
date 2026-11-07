"""E4 figures: E/ν vs γ, objectivity under rigid motion."""
import numpy as np

from common import load_csv, load_json, plt, save

R = load_json("E4", "summary")["results"]

fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
g = np.array([r["gamma"] for r in R["uniaxial"]])
E = np.array([r["E_eff_over_E"] for r in R["uniaxial"]])
nu = np.array([r["nu_eff"] for r in R["uniaxial"]])
gg = np.linspace(0.05, 1, 100)
nv = (1 - gg) / (3 + gg)
ax[0].plot(g, E, "o", label="measured (3 tessellations)")
ax[0].plot(gg, (3 + gg) * (1 - nv**2) / 4, "k-", lw=0.8, label="affine (Voigt) estimate")
ax[0].set(xlabel="γ = k_t/k_n", ylabel="E_eff / E", title="Uniaxial tension: effective modulus")
ax[0].legend()
ax[1].plot(g, nu, "o")
ax[1].plot(gg, nv, "k-", lw=0.8)
ax[1].set(xlabel="γ = k_t/k_n", ylabel="ν_eff", title="Effective Poisson ratio (above Voigt)")

d = load_csv("E4", "rigid")
for tr in [0.0, 1.0, 1e3, 1e6]:
    m = d["translation_m"] == tr
    a = np.maximum(d["alpha_deg"][m], 1e-4)
    ax[2].semilogy(a, np.maximum(d["unloaded_max_force_over_failure"][m], 1e-18), "o-", ms=3, label=f"unloaded, T={tr:g} m")
    ax[2].semilogy(a, np.maximum(d["loaded_max_dev_over_maxforce"][m], 1e-18), "s--", ms=3, label=f"loaded (dev/F_max), T={tr:g} m")
ax[2].set(xlabel="rigid rotation [deg]", ylabel="spurious force (relative)", title="Objectivity: rigid rotation + translation T")
ax[2].legend(fontsize=6, ncol=2)
save(fig, "E4_patch")
