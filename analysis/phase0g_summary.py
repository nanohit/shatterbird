"""Summarize causal-mask runs; no reference data enters the online selector.

Run after target/release/phase0g CASE SEED HE_SCALE [thresholds]. Runtime is
CPU f64 macro advancement only, not GPU time. Diagnostic samples are separate.
"""
import json
import math
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "results" / "phase0g"


def mass_distance(a, b):
    """Mass-weighted Wasserstein distance in log(relative fragment mass)."""
    sa, sb = sum(a), sum(b)
    aa = sorted((math.log(max(m / sa, 1e-30)), m / sa) for m in a)
    bb = sorted((math.log(max(m / sb, 1e-30)), m / sb) for m in b)
    i = j = 0
    wa, wb = aa[0][1], bb[0][1]
    dist = 0.0
    while i < len(aa) and j < len(bb):
        weight = min(wa, wb)
        dist += weight * abs(aa[i][0] - bb[j][0])
        wa -= weight
        wb -= weight
        if wa < 1e-14:
            i += 1
            if i < len(aa):
                wa = aa[i][1]
        if wb < 1e-14:
            j += 1
            if j < len(bb):
                wb = bb[j][1]
    return dist


sources, rows, refs = {}, [], {}
for path in sorted(OUT.glob("*.json")):
    if path.stem == "summary":
        continue
    obj = json.loads(path.read_text())
    if "reference" not in obj:
        continue
    sources[path.stem] = obj["meta"]
    ref = obj["reference"]
    refs[path.stem] = ref
    for item in obj["runs"]:
        r, c = item["result"], item["comparison"]
        s = r["stats"]
        paired = list(zip(r["rows"], ref["rows"]))
        def time_error(column):
            return math.sqrt(sum((a[column] - b[column]) ** 2 for a, b in paired)
                             / max(sum(b[column] ** 2 for _, b in paired), 1e-30))
        rows.append(dict(
            run=path.stem, mode=r["mode"], completed=r["completed"],
            D=r["d"], D_error=c["D_relative_error"],
            force_L2=c["force_time_L2"], local_force_L2=c["near_onset_force_time_L2"],
            pre_onset_force_L2=c.get("pre_onset_force_time_L2", float('nan')),
            first_fracture_force_L2=c.get("first_fracture_force_time_L2", float('nan')),
            damage_Jaccard=c["damage_Jaccard"],
            mean_hot_cells=r["mean_hot_cells"], peak_hot_cells=r["peak_hot_cells"],
            mean_hot_facets=r["mean_hot_facets"],
            nonlinear_point_fraction=s["full_points"] / (s["full_points"] + s["linear_points"]),
            runtime_gain=c["runtime_gain"], runtime_s=r["runtime_s"],
            cold_chi_audit=r["cold_chi_max_audit"], cold_misses_audit=r["cold_over_onset_audit"],
            energy_defect=r["energy_defect"], reference_energy_defect=ref["energy_defect"],
            model_energy_removed=r["model_energy_removed"],
            switch_energy_absolute=s["switch_energy_absolute"],
            geometry_promotions=s["promoted_geometry"], strain_promotions=s["promoted_strain"],
            refresh_points=s.get("refresh_points", 0),
            first_promotion_over_onset=s["promoted_over_onset"],
            KE_time_L2=time_error(5), D_time_L2=time_error(1),
            final_KE=r["rows"][-1][5], reference_final_KE=ref["rows"][-1][5],
            fragment_mass_distance=mass_distance(r["fragment_masses"], ref["fragment_masses"]),
            fragments=len(r["fragment_masses"]), reference_fragments=len(ref["fragment_masses"]),
            onset=r["onset"], reference_onset=ref["onset"],
        ))

refinement = []
for name, coarse in refs.items():
    if not name.endswith("_h1"):
        continue
    fine = refs.get(name[:-1] + "0.5")
    if fine is None:
        continue
    a, b = set(coarse["damage"]), set(fine["damage"])
    refinement.append(dict(
        run=name, D_h=coarse["d"], D_h2=fine["d"],
        reference_D_change=(coarse["d"] - fine["d"]) / max(abs(fine["d"]), 1e-30),
        reference_damage_Jaccard=len(a & b) / max(len(a | b), 1),
        fragment_mass_distance=mass_distance(coarse["fragment_masses"], fine["fragment_masses"]),
        reference_energy_defect_h=coarse["energy_defect"],
        reference_energy_defect_h2=fine["energy_defect"],
    ))

extra_refinement = []
for seed in (1, 2):
    chain = [(scale, refs.get(f"span_s{seed}_h{scale}{suffix}"))
             for scale, suffix in [("1", ""), ("0.5", ""),
                                   ("0.25", "_reference"), ("0.125", "_reference")]]
    chain = [(s, r) for s, r in chain if r is not None]
    for (sa, a), (sb, b) in zip(chain, chain[1:]):
        da, db = set(a["damage"]), set(b["damage"])
        extra_refinement.append(dict(seed=seed, coarse_scale=sa, fine_scale=sb,
            D_coarse=a["d"], D_fine=b["d"], D_change=(a["d"]-b["d"])/b["d"],
            damage_Jaccard=len(da & db)/max(len(da | db), 1),
            fragment_mass_distance=mass_distance(a["fragment_masses"], b["fragment_masses"])))

summary = dict(sources=sources, rows=rows, reference_refinement=refinement,
               extended_span_refinement=extra_refinement,
               scope="All wave DOFs and stable explicit steps retained. CPU measurements, no GPU/FPS/determinism claim. Cold-force audits at recording times are diagnostic only. Hot fractions sampled at macro boundaries; masks are monotone. Runtime excludes diagnostic sampling and startup.")
(OUT / "summary.json").write_text(json.dumps(summary, indent=2))
print("run / mode | D error % | force % | first fracture % | hot cells % | CPU gain | fragment distance")
for r in rows:
    print(f"{r['run']} / {r['mode']} | {100*r['D_error']:.3g} | {100*r['force_L2']:.3g} | "
          f"{100*r['first_fracture_force_L2']:.3g} | {100*r['mean_hot_cells']:.3g} | "
          f"{r['runtime_gain']:.3g} | {r['fragment_mass_distance']:.3g}")
print("Reference refinement:")
for r in refinement:
    print(r)
