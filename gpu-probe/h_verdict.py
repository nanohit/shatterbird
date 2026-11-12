#!/usr/bin/env python3
"""Phase 1h verdict: per resolved observable, error of each variant against the reference, averaged over piles.
error = |log((v + d) / (r + d))|, d = 1 for counts, 0.001 D for shifts. Mean over the evaluation piles.
usage: h_verdict.py <dir> <reference> <out.json> <seeds> <variant,...>"""
import json, sys, math, os
d, ref, out, seeds, names = sys.argv[1], sys.argv[2], sys.argv[3], [int(x) for x in sys.argv[4].split(',')], sys.argv[5].split(',')

def obs(name, seed):
    f = f'{d}/r2_12_{seed}_{name}.json'
    if not os.path.exists(f): return None
    o = {}
    for r in json.load(open(f))['runs']:
        ev, last = r['event'], r['rows'][-1]
        k = ev.split(':')
        if k[0] in ('down', 'side'):
            key = f'{k[0]}{k[1]}'
            o.setdefault(f'kick {key}: bodies > 0.01 D', 0); o[f'kick {key}: bodies > 0.01 D'] += last['over_0p01D']
            o.setdefault(f'kick {key}: bodies > 0.1 D', 0); o[f'kick {key}: bodies > 0.1 D'] += last['over_0p1D']
        elif k[0] == 'slump':
            row = [x for x in r['rows'] if x['t'] == 960][0]
            o['slump: centre of mass dx'] = row['com_dx_D']; o['slump: centre of mass dy'] = -row['com_dy_D']
            o['slump: bodies > 1 D'] = row['moved_over_1D']; o['slump: bodies > 0.1 D'] = row['moved_over_0p1D']
        elif k[0] == 'tilt':
            o[f'tilt {k[1]}: centre of mass dx'] = last['com_shift_D'][0]; o[f'tilt {k[1]}: bodies > 0.1 D'] = last['over_0p1D']
        elif k[0] == 'remove':
            o.setdefault('removal: bodies > 0.01 D', 0); o['removal: bodies > 0.01 D'] += last['over_0p01D']
            o.setdefault('removal: bodies > 0.1 D', 0); o['removal: bodies > 0.1 D'] += last['over_0p1D']
        if k[0] != 'control':
            o.setdefault('events never at exact rest', 0); o['events never at exact rest'] += 1 if r['quiet_at'] < 0 else 0
            o.setdefault('control moved', 0)
        else:
            o['control moved'] = last['sum_D']
    return o

R = {s: obs(ref, s) for s in seeds}
seeds = [s for s in seeds if R[s]]
res = {'reference': ref, 'seeds': seeds, 'error': {}, 'values': {}}
keys = [k for k in R[seeds[0]] if k not in ('events never at exact rest', 'control moved')]
for n in names:
    V = {s: obs(n, s) for s in seeds}
    if any(V[s] is None for s in seeds): print('missing', n, [s for s in seeds if V[s] is None]); continue
    err = {}
    for k in keys:
        dd = 0.001 if 'centre' in k else 1.0
        err[k] = sum(abs(math.log((max(V[s][k], 0) + dd) / (max(R[s][k], 0) + dd))) for s in seeds) / len(seeds)
    res['error'][n] = err
    res['values'][n] = {k: [V[s][k] for s in seeds] for k in list(keys) + ['events never at exact rest', 'control moved']}
res['values'][ref] = {k: [R[s][k] for s in seeds] for k in keys}
json.dump(res, open(out, 'w'), indent=1)
groups = [('kicks', lambda k: k.startswith('kick')), ('slump', lambda k: k.startswith('slump')), ('tilt', lambda k: k.startswith('tilt')), ('removal', lambda k: k.startswith('removal'))]
print(f'mean |log error| against {ref} over piles {seeds} (lower is better); per class the mean over its observables')
print('variant'.ljust(8) + ''.join(g.rjust(9) for g, _ in groups) + '   never at rest   control moved')
for n in res['error']:
    e = res['error'][n]
    print(n.ljust(8) + ''.join(f'{sum(v for k, v in e.items() if f(k)) / max(1, sum(1 for k in e if f(k))):9.3f}' for _, f in groups),
          '  ', sum(res['values'][n]['events never at exact rest']), '            ', max(res['values'][n]['control moved']))
