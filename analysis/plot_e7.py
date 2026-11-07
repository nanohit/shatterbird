"""E7 figures: support-removal histories (elastic and with fracture), ledger."""
import numpy as np

from common import load_csv, load_json, plt, save

S = load_json("E7", "summary")
runs = S["runs"]
s = load_csv("E7", "series")


def series(run_id, col):
    m = s["run"] == run_id
    return s["t"][m], s[col][m]


def find(scen, solver, scheme="BDF2", h=1 / 480):
    for r in runs:
        if r["scenario"].startswith(scen) and r["solver"] == solver and r["scheme"] == scheme and abs(r["h"] - h) < 1e-12:
            return r["run"]
    return None


fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
for solver, ls in [("S0", "k-"), ("PerSpring(0.1), 64 sweeps", "--"), ("PerSpring(0.1), 256 sweeps", ":"), ("S1 PerSpring(0.1) + S2(B) 8+8 sweeps", "-.")]:
    rid = find("A-elastic", solver)
    if rid is None:
        continue
    t, r = series(rid, "reaction_y")
    ax[0].plot(t, r / 1e3, ls, label=solver)
ax[0].set(xlabel="t [s]", ylabel="clamp vertical reaction [kN]", title="A-elastic: prop removed at t=0 (BDF2, 1/480)", ylim=(-400, 200))
ax[0].legend(fontsize=6)
for solver, scheme, h, ls in [("S0", "BDF2", 1 / 480, "k-"), ("S0", "BDF2", 1 / 960, "k--"), ("S0", "BDF2", 1 / 1920, "k:"), ("S0", "IE", 1 / 480, "r-"),
                              ("S0 + event sub-stepping", "IE", 1 / 480, "r--"), ("PerSpring(0.1), 64 sweeps", "BDF2", 1 / 480, "b-"),
                              ("PerSpring(0.1), 256 sweeps", "BDF2", 1 / 480, "b--"), ("PerSpring(0.1), 16 sweeps", "BDF2", 1 / 960, "g-")]:
    rid = find("A cantilever prop removal, dmin 0.3", solver, scheme, h)
    if rid is None:
        continue
    t, d = series(rid, "D")
    ax[1].plot(t, d, ls, label=f"{solver}, {scheme} 1/{round(1/h)}")
ax[1].set(xlabel="t [s]", ylabel="fracture dissipation [J]", title="A with fracture: dissipation history", ylim=(0, 160))
ax[1].legend(fontsize=6)
for solver, ls in [("S0", "r-"), ("S0 + event sub-stepping", "r--")]:
    rid = find("A cantilever prop removal, dmin 0.3", solver, "IE", 1 / 480)
    t, bal = series(rid, "balance")
    _, dn = series(rid, "Dnum_IE")
    ax[2].plot(t, bal + dn, ls, label=f"{solver}: unexplained ε")
    ax[2].plot(t, -dn, ls, alpha=0.4, label=f"{solver}: −D_num")
ax[2].axhline(0, color="k", lw=0.5)
ax[2].set(xlabel="t [s]", ylabel="J", title="IE ledger: ΔE + D_frac + D_num = ε")
ax[2].legend(fontsize=6)
save(fig, "E7_dynamic")
