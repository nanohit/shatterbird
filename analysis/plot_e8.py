"""E8 figures: amplification per period, bar reaction under a step load."""
import numpy as np

from common import load_csv, load_json, plt, save

S = load_json("E8", "summary")["results"]
names = ["IE", "BDF2", "Newmark"]
a = load_csv("E8", "amplification")
b = load_csv("E8", "band")
fig, ax = plt.subplots(1, 3, figsize=(13, 3.7))
for si, n in enumerate(names):
    m = a["scheme"] == si
    om = a["Omega"][m]
    per = 2 * np.pi / om
    ln, = ax[0].semilogx(om, a["abs_meas"][m] ** per, "o", ms=3, label=f"{n} (implementation)")
    ax[0].semilogx(om, a["abs_an"][m] ** per, "-", color=ln.get_color(), lw=0.8)
ax[0].set(xlabel="Ω = ω h", ylabel="amplitude kept per physical period", title="Per-period amplitude (lines: closed form)")
ax[0].legend(fontsize=7)
for si, n in enumerate(names):
    for h, ls in [(1 / 480, "-"), (1 / 1920, "--")]:
        m = (b["scheme"] == si) & np.isclose(b["h"], h)
        ax[1].semilogx(b["f_hz"][m], b["amp_per_period"][m], ls, label=f"{n}, h=1/{round(1/h)}")
ax[1].axhline(0.5, color="k", lw=0.5)
ax[1].set(xlabel="mode frequency [Hz]", ylabel="amplitude kept per period", title="Which structural band survives")
ax[1].legend(fontsize=6)
bar = load_csv("E8", "bar")
runs = S["bar"]["runs"]
m = bar["run"] == 0
ax[2].plot(bar["t"][m] * 1e3, bar["reaction_over_F0"][m], "k-", lw=0.8, label="resolved (Newmark, 2 µs)")
for r in runs[1:]:
    if abs(r["h"] - 1 / 480) < 1e-9 or abs(r["h"] - 1 / 3840) < 1e-9:
        mm = bar["run"] == r["run"]
        ax[2].plot(bar["t"][mm] * 1e3, bar["reaction_over_F0"][mm], "o-", ms=2.5, lw=0.8, label=f"{r['scheme']}, h=1/{round(1/r['h'])}")
ax[2].set(xlabel="t [ms]", ylabel="clamp reaction / F0", title="10 m bar (0.5 m cells), step load at the free end")
ax[2].legend(fontsize=6)
save(fig, "E8_time")
