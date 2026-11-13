#!/usr/bin/env python3
"""Phase 2a: compliant pile, every method against the finest explicit run of the same law.
usage: soft_summary.py <dir> <prefix> <reference method> <out.json> <method,...>"""
import json, sys, math, numpy as np
d, pre, ref, outf, names = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5].split(',')
L = lambda m: json.load(open(f'{d}/{pre}_{m}.json'))
def obs(o):
    out = {}
    for ev, r in o['runs'].items():
        l = r['rows'][-1]; k = ev.split(':')
        if k[0] in ('down', 'side'):
            out[f'kick {ev}: kicked sphere'] = l['kicked_abs_D']; out[f'kick {ev}: sum of displacements'] = l['sum_D']; out[f'kick {ev}: bodies > 0.01 D'] = l['over_0p01D']
        elif k[0] == 'remove': out[f'removal {ev}: sum of displacements'] = l['sum_D']; out[f'removal {ev}: bodies > 0.01 D'] = l['over_0p01D']
        elif k[0] == 'slump':
            q = r['rows'][3]; out['slump: centre of mass dx (1 s)'] = q['com_D'][0]; out['slump: x95 (1 s)'] = q['x95_D']; out['slump: bodies > 1 D (1 s)'] = q['over_1D']
        elif k[0] == 'ramp':
            for q in r['rows']:
                if abs(q['tan'] - 0.5) < 1e-9: out['ramp: centre of mass dx at tan 0.5'] = q['com_D'][0]; out['ramp: bodies > 0.1 D at tan 0.5'] = q['over_0p1D']
            out['ramp: bodies > 0.1 D at the end'] = l['over_0p1D']
        elif k[0] == 'control': out['control: sum of displacements'] = l['sum_D']
    return out
R = obs(L(ref)); res = dict(reference=ref, values={ref: R}, error={}, solver={})
cls = [('kicked sphere', lambda k: 'kicked sphere' in k), ('kick: field', lambda k: k.startswith('kick') and 'kicked' not in k), ('removal', lambda k: k.startswith('removal')), ('slump', lambda k: k.startswith('slump')), ('ramp', lambda k: k.startswith('ramp'))]
print(f'{pre}: mean |log ratio| to {ref} per class (counts +1, lengths +0.001 D)')
print('method   ' + ''.join(c.rjust(15) for c, _ in cls) + '   control   iterations/step   certificate max (J)')
for n in names:
    try: o = L(n)
    except FileNotFoundError: print(n, 'missing'); continue
    V = obs(o); e = {}
    for k in R:
        if k.startswith('control') or k not in V: continue
        dd = 1.0 if 'bodies' in k else 0.001; e[k] = abs(math.log((abs(V[k]) + dd) / (abs(R[k]) + dd)))
    res['values'][n] = V; res['error'][n] = e
    sv = [r['solver'] for r in o['runs'].values() if 'solver' in r]; res['solver'][n] = dict(iters_mean=float(np.mean([s['iters_mean'] for s in sv])), cert_max=float(max(s['cert_max_J'] for s in sv))) if sv else None
    print(n.ljust(9) + ''.join(f'{np.mean([v for k, v in e.items() if f(k)]):15.3f}' for _, f in cls) + f"   {V.get('control: sum of displacements', 0):.1e}" + (f"   {res['solver'][n]['iters_mean']:8.1f}   {res['solver'][n]['cert_max']:.1e}" if sv else ''))
json.dump(res, open(outf, 'w'), indent=1)
