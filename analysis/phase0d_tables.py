"""Phase 0d: markdown tables from results/phase0d/*/summary_*.json (printed to stdout)."""
import json
import os
import sys

R = os.path.join(os.path.dirname(__file__), '..', 'results', 'phase0d')
if len(sys.argv) > 1:
    R = sys.argv[1]


def load(p):
    p = os.path.join(R, p)
    return json.load(open(p)) if os.path.exists(p) else None


def pc(x):
    return f"{100 * x:+.1f} %"


def s1():
    d = load('s1/summary_s1.json')
    if not d:
        return
    thr = {(t['seed'], t['engine']): t['lambda_crit'] for t in d['thresholds']}
    print('| | ' + ' | '.join(f"seed {r['seed']}" for r in d['results']) + ' |')
    print('|---|' + '---|' * len(d['results']))
    rows = []
    for r in d['results']:
        c = r['criteria']
        s = r['seed']
        rows.append({
            'λ_crit ref / engine': f"{thr[(s, False)]:.3f} / {thr[(s, True)]:.3f} ({pc(c['threshold']['rel'])})" + ('' if c['threshold']['pass'] else ' ✗'),
            'separation, onset, separation time': 'yes' if (c['separation']['pass'] and c['onset']['pass'] and c['sep_t']['pass']) else '✗',
            'crack x ref / engine': f"{c['crack_x']['ref']:.3f} / {c['crack_x']['engine']:.3f} m" + ('' if c['crack_x']['pass'] else ' ✗'),
            'D': pc(c['D']['rel']) + ('' if c['D']['pass'] else ' ✗'),
            'fragment v ref / engine': f"({c['frag_v']['ref'][0]:.3f}, {c['frag_v']['ref'][1]:.3f}) / ({c['frag_v']['engine'][0]:.3f}, {c['frag_v']['engine'][1]:.3f}): {100 * c['frag_v']['rel']:.1f} %" + ('' if c['frag_v']['pass'] else ' ✗'),
            'fragment ω ref / engine': f"{c['frag_omega']['ref']:.3f} / {c['frag_omega']['engine']:.3f}" + ('' if c['frag_omega']['pass'] else ' ✗'),
            'fragment KE': pc(c['frag_ke']['rel']) + ('' if c['frag_ke']['pass'] else ' ✗'),
            'energy creation': f"{c['no_creation']['e_rise']:.2e} J" + ('' if c['no_creation']['pass'] else ' ✗'),
            'numerical dissipation / D_red': f"{c['numerical_dissipation']:.1f} J / {c['d_red']:.1e} J",
            'work / wall-clock vs explicit': f"{c['work_ratio']:.1f}× / {c['wall_ratio']:.1f}×",
            'cells with own DOFs (mean)': f"{100 * c['zone']['mean_frac']:.0f} %",
        })
    for k in rows[0]:
        print(f"| {k} | " + ' | '.join(r[k] for r in rows) + ' |')


def s2():
    d = load('s2/summary_s2.json')
    if not d:
        return
    print('| seed | v_crit ref / engine | factor | all criteria | D | impulse | rebound ref / engine | fragments | num. dissipation / KE₀ | work / wall |')
    print('|---|---|---|---|---|---|---|---|---|---|')
    for r in d['results']:
        t = r['threshold']
        for c in r['consequences']:
            cr = c['criteria']
            ok = all(v['pass'] for v in cr.values() if isinstance(v, dict) and 'pass' in v)
            print(f"| {r['seed']} | {t['ref']:.3f} / {t['engine']:.3f} ({pc(t['rel'])}) | {c['factor']} | {'pass' if ok else '✗'} | {pc(cr['D']['rel'])} | {pc(cr['impulse']['rel'])} | {cr['rebound']['ref']:.4f} / {cr['rebound']['engine']:.4f} | {cr['fragments']['ref']} / {cr['fragments']['engine']} | {100 * cr['numerical_dissipation_over_ke0']:.0f} % | {cr['work_ratio']:.2f} / {cr['wall_ratio']:.2f} |")


def s3(name, key):
    d = load(name)
    if not d:
        return
    print(f'| {key} | cells | consequences | work ratio | wall ratio | cells with own DOFs |')
    print('|---|---|---|---|---|---|')
    for r in d['runs']:
        cr = r['criteria']
        ok = all(v['pass'] for v in cr.values() if isinstance(v, dict) and 'pass' in v)
        print(f"| {r.get('nspan', r.get('width'))} | {r['cells']} | {'pass' if ok else '✗'} | {cr['work_ratio']:.2f} | {cr['wall_ratio']:.2f} | {100 * cr['zone']['mean_frac']:.0f} % |")
    print(f"\nmarginal cost ratio b_engine/b_explicit = {d['scaling']['marginal_ratio']:.2f}")


def s4(name):
    d = load(name)
    if not d:
        return
    print('| | ' + ' | '.join(f"seed {r['seed']}" for r in d['results']) + ' |')
    print('|---|' + '---|' * len(d['results']))
    rows = []
    for r in d['results']:
        c = r['criteria']
        m = lambda L: ', '.join(f"{f['mass']:.0f}" for f in L)
        rows.append({
            'large fragments ref / engine (kg)': f"{c['large_fragments']['ref']} / {c['large_fragments']['engine']} ({m(c['fragments']['ref'])} / {m(c['fragments']['engine'])})",
            'D ref / ref h/2 / engine': f"{c['D']['ref']:.1f} / {c['D']['ref_half']:.1f} / {c['D']['engine']:.1f} J ({pc(c['D']['rel'])}, noise {100 * c['D']['noise']:.1f} %)" + ('' if c['D']['pass'] else ' ✗'),
            'mass distribution distance (noise)': f"{c['mass_cdf_distance']['engine']:.2f} ({c['mass_cdf_distance']['noise']:.2f})" + ('' if c['mass_cdf_distance']['pass'] else ' ✗'),
            'energy creation': f"{c['no_creation']['e_rise']:.1e} J" + ('' if c['no_creation']['pass'] else ' ✗'),
            'num. dissipation / D_red (× KE_max)': f"{100 * c['numerical_dissipation_over_ke_max']:.2f} % / {c['d_red_over_ke_max']:.0e}",
            'cells with own DOFs mean / peak': f"{100 * c['zone']['mean_frac']:.0f} % / {100 * c['zone']['max'] / c['zone']['cells']:.0f} %",
            'work / wall-clock vs explicit': f"{c['work_ratio']:.2f}× / {c['wall_ratio']:.2f}×",
        })
    for k in rows[0]:
        print(f"| {k} | " + ' | '.join(r[k] for r in rows) + ' |')


def s5():
    d = load('s5/summary_s5.json')
    if not d:
        return
    print('| length | cells | separated ref / engine | D | fragment Δv | ω ref / engine | work: explicit / engine (Gflop) | work ratio | wall ratio | cells with own DOFs |')
    print('|---|---|---|---|---|---|---|---|---|---|')
    for r in d['runs']:
        print(f"| {r['len']:.0f} m | {r['cells']} | {r['separated'][0]} / {r['separated'][1]} | {pc(r['D_rel'])} | {100 * r['frag_dv_rel']:.1f} % | {r['omega'][0]:.3f} / {r['omega'][1]:.3f} | {r['work_ref'] / 1e9:.1f} / {r['work_engine'] / 1e9:.2f} | {r['work_ratio']:.1f}× | {r['wall_ratio']:.1f}× | {100 * r['zone_mean']:.0f} % |")


def noise():
    d = load('s1/summary_s1_noise_wide.json')
    if not d:
        return
    from collections import Counter, defaultdict
    br = defaultdict(Counter)
    for r in d['runs']:
        vx = r['fragment']['v'][0]
        b = 'A' if vx > 0.4 else ('B' if vx > 0.2 else 'C')
        br[(r['seed'], r['engine'])][b] += 1
    print('| | seed 1 | seed 2 |')
    print('|---|---|---|')
    for eng in (False, True):
        print(f"| {'engine' if eng else 'reference'} | " + ' | '.join(', '.join(f"{k} ×{v}" for k, v in sorted(br[(s, eng)].items())) for s in (1, 2)) + ' |')


for title, f in [('S1', s1), ('S2', s2), ('S3', lambda: s3('s3/summary_s3_v1.5.json', 'spans')), ('S3b', lambda: s3('s3/summary_s3b_v1.5.json', 'width (m)')), ('S4', lambda: s4('s4/summary_s4.json')), ('S4b', lambda: s4('s4/summary_s4b.json')), ('S5', s5), ('noise', noise)]:
    print(f"\n#### {title}\n")
    f()
