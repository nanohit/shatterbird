"""Reproduce comparisons without treating a work estimate as GPU milliseconds.

python3 analysis/phase0f_summary.py
Inputs remain immutable; summary contains each input's own provenance.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "phase0f"
sources = {}


def read(name):
    path = OUT / (name + ".json")
    obj = json.loads(path.read_text())
    sources[name] = obj.get("meta", {})
    return obj["data"]


def relative(a, b):
    return (a - b) / max(abs(b), 1e-30)


def mass_distance(a, b):
    """Mass-weighted 1-Wasserstein distance of log(fragment mass / total mass).

    Independent of cell IDs and fragment ordering; zero for identical masses.
    """
    import math
    sa, sb = sum(a), sum(b)
    aa = sorted((math.log(max(m / sa, 1e-30)), m / sa) for m in a)
    bb = sorted((math.log(max(m / sb, 1e-30)), m / sb) for m in b)
    i = j = 0
    wa, wb = aa[0][1], bb[0][1]
    dist = 0.0
    while i < len(aa) and j < len(bb):
        w = min(wa, wb)
        dist += w * abs(aa[i][0] - bb[j][0])
        wa -= w
        wb -= w
        if wa < 1e-14:
            i += 1
            if i < len(aa):
                wa = aa[i][1]
        if wb < 1e-14:
            j += 1
            if j < len(bb):
                wb = bb[j][1]
    return dist


legacy = read("legacy_wall")["rows"]
lr = legacy[0]
legacy_comparison = [dict(
    mode=r["mode"], D_error=relative(r["D"], lr["D"]),
    work_gain=lr["work"] / r["work"],
    wall_gain=lr["wall_s"] / r["wall_s"],
    mean_fine=r["mean_fine"], completed=r["completed"],
    energy_end_minus_initial=r["energy_end_minus_initial"],
) for r in legacy]
relaxed = read("legacy_relaxed")
legacy_ablation = [dict(r, D_ratio=r["D"] / lr["D"], work_gain=lr["work"] / r["work"])
                   for r in relaxed["rows"]]

wall = read("wall_ports")
wall_rows = []
for item in wall["rows"]:
    r = item["run"]
    wall_rows.append(dict(
        seed=item["seed"], modes=r["modes"],
        representation=r["far_representation"], bridge=r["bridge"],
        zone_half_width=r["half_width"],
        fine_cells=r["fine_cells"], total_cells=r["cells"], dofs=r["dofs"],
        D=r["D"], far_linear_contract_valid=r["far_linear_contract_valid"],
        max_far_trial_chi=r["max_far_trial_chi"],
        D_outside_central_strip=r["D_outside_central_strip"],
        comparison=item.get("comparison", item.get("comparison_to_h")),
        projection_comparison=item.get("comparison_to_same_linear_far_physics"),
        build_work=r["build_work"], build_s=r["build_s"],
        work=r["work"], hot_wall_s=r["hot_wall_s"],
        coefficients_bytes=r["basis_or_sparse_coefficients_bytes"],
        half_step_reference=item.get("half_step_reference", False),
    ))

freeze = read("freeze")["rows"] + read("freeze_refinement")["rows"]
quiet = read("freeze_quiet")["rows"]
freeze_rows = []
for item in freeze + quiet:
    r = item["run"]
    if r["mode"] == "fine":
        continue
    family = quiet if r["mode"].startswith("damped") else freeze
    name = "damped_fine" if r["mode"].startswith("damped") else "fine"
    candidates = [i["run"] for i in family
                  if i["seed"] == item["seed"] and i["run"]["mode"] == name]
    reference = min(candidates, key=lambda a: abs(a["h"] - r["h"]))
    refined = min(candidates, key=lambda a: a["h"])
    freeze_rows.append(dict(
        seed=item["seed"], mode=r["mode"], h=r["h"],
        D=r["D"], same_step_reference_D=reference["D"], refined_reference_D=refined["D"],
        same_step_D_error=relative(r["D"], reference["D"]),
        refined_D_error=relative(r["D"], refined["D"]),
        work_gain=reference["work"] / r["work"],
        peak_frame_gain=reference["peak_frame_work"] / r["peak_frame_work"],
        mean_fine=r["mean_fine_fraction"], peak_fine=r["peak_fine_fraction"],
        min_fine=r["min_fine_fraction"], freezes=r["freezes"],
        fragments=len(r["fragment_masses"]),
        reference_fragments=len(reference["fragment_masses"]),
        fragment_mass_distance=mass_distance(r["fragment_masses"], reference["fragment_masses"]),
        energy_balance=r["energy_balance"], projection_KE_loss=r["projection_KE_loss"],
        damping_loss=r.get("internal_damping_loss", 0),
        wake_contact_validated=False,
    ))

ports = read("ports")
budget = read("budget")
original_wall = read("wall_original")
summary = dict(
    scope="Research controls; CPU f64, not a production architecture acceptance or mobile GPU FPS proof.",
    sources=sources, legacy_original_geometry=legacy_comparison,
    legacy_force_gate_ablation=legacy_ablation,
    sparse_carrier_original_geometry=original_wall,
    wall_scope=wall["scope"], wall=wall_rows, freeze=freeze_rows,
    chain_rows=len(ports["rows"]),
    chain_irf_max_q_error=max(r["full_local_q_l2"] for r in ports["rows"] if r["port"] == "irf"),
    chain_passive_waveguide_max_balance=max(abs(r["linear_energy_balance"]) for r in ports["rows"]
                                          if r["port"] == "waveguide" and not r["damage"]),
    queue=budget["queue"], constitutive_hold=budget["constitutive_hold"],
    instant_split=budget["instant_split"],
    coupled_bundle=[{k: v for k, v in r.items() if k != "trace"} for r in budget["coupled_bundle"]],
    limitations=[
        "Dynamic port/CMS controls use dmin=.1m; S4b rigid controls use .08m. Legacy S3b rerun uses original .04m.",
        "Sparse-carrier original-geometry controls also use .04m. The future-damage-mask localization oracle is deliberately noncausal and is NOT an online activation algorithm.",
        "Frozen components have no new rubble contacts in S4b; wake-on-impact is not validated.",
        "A linear far field crossing chi=1 is outside its strict no-damage contract, regardless of a matching total D.",
        "Late raw force histories have their own substantial h/h2 phase error; early force, peak and fracture-history errors are reported separately.",
        "Damped full/frozen runs share a named new 10ms internal damping model. They cannot be presented as unchanged undamped physics.",
        "Capacity scheduling limits admission, not all scene work or GPU frame time. The 10k-link bundle is coupled but is not a 2D/3D rubble test.",
    ],
)
(OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"legacy": legacy_comparison, "freeze": freeze_rows,
                  "chain_irf_max_q_error": summary["chain_irf_max_q_error"]},
                 ensure_ascii=False, indent=2))
