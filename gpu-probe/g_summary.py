#!/usr/bin/env python3
"""Phase 1g: kick ensemble at equal horizons. Reads the per-variant records written by lab3.cjs (resp / resp2)
from a cache directory and compares every variant with a reference variant kick by kick, substep by substep.
usage: g_summary.py <cache dir> <n> <reference name> [out.json] [name1,name2,...]"""
import json, sys, os, glob, math, struct, array
d, n, ref = sys.argv[1], sys.argv[2], sys.argv[3]
out = sys.argv[4] if len(sys.argv) > 4 else None
EPS = 1e-4
def load(name):
    runs = {}
    for seed in (11, 12, 13):
        for pat, new in ((f'{d}/resp_{n}_{seed}_{name}.json', False), (f'{d}/r2_{n}_{seed}_{name}.json', True)):
            if not os.path.exists(pat): continue
            j = json.load(open(pat))
            for r in j['runs']:
                if new:
                    p = r['event'].split(':')
                    if p[0] not in ('down', 'side', 'up'): continue
                    key = (seed, p[0], float(p[1]), int(p[2])); f = f"{d}/r2_{n}_{seed}_{name}_{r['event'].replace(':', '_')}.f32"
                else:
                    if r['label'] == 'control': continue
                    a, q = r['label'].split('_k'); kind = 'down' if a.startswith('down') else 'side'; key = (seed, kind, float(a[len(kind):]), int(q)); f = f"{d}/resp_{n}_{seed}_{name}_{r['label']}.f32"
                if key in runs: continue
                runs[key] = {'rows': {row['t']: row for row in r['rows']}, 'order': [row['t'] for row in r['rows']], 'f': f, 'quiet_at': r['quiet_at'], 'N': j['N']}
    return runs
def field(run, t):
    k = run['order'].index(t); N = run['N']; a = array.array('f')
    with open(run['f'], 'rb') as fh: fh.seek(k * 12 * N); a.frombytes(fh.read(12 * N))
    return a
def gmean(xs): return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else None
names = sorted({os.path.basename(p).split('_', 3)[3][:-5] for p in glob.glob(f'{d}/resp_{n}_*_*.json') + glob.glob(f'{d}/r2_{n}_*_*.json')})
if len(sys.argv) > 5: names = [x for x in sys.argv[5].split(',')]
R = load(ref); res = {'reference': ref, 'n': int(n), 'kicks_in_reference': len(R), 'variants': {}}
kinds = [('down', 0.25), ('down', 1.0), ('down', 4.0), ('side', 1.0)]
for name in names:
    X = load(name)
    if not X: continue
    v = {}
    for kind, sp in kinds:
        for T in (25, 100, 400, 800):
            keys = [k for k in X if k[1] == kind and k[2] == sp and k in R and T in X[k]['rows'] and T in R[k]['rows']]
            if not keys: continue
            rk = [(X[k]['rows'][T]['kicked_abs_D'] + EPS) / (R[k]['rows'][T]['kicked_abs_D'] + EPS) for k in keys]
            rs = [(X[k]['rows'][T]['sum_D'] + EPS) / (R[k]['rows'][T]['sum_D'] + EPS) for k in keys]
            re = [(X[k]['rows'][T]['ke_frac'] + 1e-12) / (R[k]['rows'][T]['ke_frac'] + 1e-12) for k in keys]
            fd = []
            for k in keys:
                try:
                    a, b = field(X[k], T), field(R[k], T); den = sum(abs(y) for y in b)
                    if den > 0: fd.append(sum(abs(x - y) for x, y in zip(a, b)) / den)
                except Exception as e: pass
            v[f'{kind}{sp:g}@{T}'] = {'kicks': len(keys), 'kicked_gmean_ratio': round(gmean(rk), 3), 'sum_gmean_ratio': round(gmean(rs), 3),
                'kicked_median_D': [round(sorted(X[k]['rows'][T]['kicked_abs_D'] for k in keys)[len(keys) // 2], 4), round(sorted(R[k]['rows'][T]['kicked_abs_D'] for k in keys)[len(keys) // 2], 4)],
                'over_0p1D_total': [sum(X[k]['rows'][T]['over_0p1D'] for k in keys), sum(R[k]['rows'][T]['over_0p1D'] for k in keys)],
                'over_0p01D_total': [sum(X[k]['rows'][T]['over_0p01D'] for k in keys), sum(R[k]['rows'][T]['over_0p01D'] for k in keys)],
                'ke_gmean_ratio': round(gmean(re), 3), 'field_L1_rel_diff_median': round(sorted(fd)[len(fd) // 2], 3) if fd else None,
                'quiet': sum(1 for k in keys if 0 < X[k]['quiet_at'] <= T), 'anchor_impulse_cum_mean': round(sum(X[k]['rows'][T].get('anchor_impulse_cum', 0) for k in keys) / len(keys))}
    res['variants'][name] = v
if out: json.dump(res, open(out, 'w'), indent=1)
for T in (100, 800, 400):
    print(f'\n== substep {T}: geometric-mean ratio to {ref} (kicked sphere | sum of displacements | field L1 diff | bodies > 0.1 D variant/ref)')
    print(f'{"variant":12s}' + ''.join(f'{k + str(s):>34s}' for k, s in kinds))
    for name, v in res['variants'].items():
        line = f'{name:12s}'
        for kind, sp in kinds:
            c = v.get(f'{kind}{sp:g}@{T}')
            line += f'{"-":>34s}' if not c else f"   {c['kicked_gmean_ratio']:5.2f} | {c['sum_gmean_ratio']:5.2f} | {c['field_L1_rel_diff_median'] if c['field_L1_rel_diff_median'] is not None else -1:5.2f} | {c['over_0p1D_total'][0]:2d}/{c['over_0p1D_total'][1]:2d} ({c['kicks']:2d})"
        print(line)
