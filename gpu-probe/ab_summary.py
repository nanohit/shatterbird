#!/usr/bin/env python3
"""Phase 2a, experiment R: the impulse network of the rigid law, state or cache.
usage: ab_summary.py <dir> <out.json>. Compares responses from networks A (reference), A1 (noise of one unit),
B (another exact equilibrium), Z (zero impulses) on the same geometry."""
import json, sys, glob, math, numpy as np
d, outf = sys.argv[1], sys.argv[2]
res = {}
def fdist(x, y): return float(np.abs(x - y).sum() / max(1e-12, 0.5 * (np.abs(x).sum() + np.abs(y).sum())))
for f in sorted(glob.glob(f'{d}/ab_8_*_i*.json')):
    o = json.load(open(f)); N, it, seed = o['N'], o['iters'], o['seed']; keys = o['field_keys']
    F = {s: np.fromfile(f.replace('.json', f'_{s}.f32'), dtype=np.float32).reshape(len(keys), N, 3) for s in 'A A1 B Z'.split()}
    row = res.setdefault(it, {}).setdefault(seed, {'network_L1_vs_A': o['network_L1_vs_A'], 'events': {}})
    for ev in o['states']['A']:
        e = {}
        for s in 'A A1 B Z'.split():
            r = o['states'][s][ev]; l = r['rows'][-1]
            if ev == 'ramp':
                rows = r['rows']; fail = next((x['tan'] for x in rows if x['over_0p1D'] >= 0.05 * N), None); onset = next((x['tan'] for x in rows if x['over_0p1D'] >= 1), None)
                e[s] = dict(tan_first_body_0p1D=onset, tan_5pct_bodies_0p1D=fail, com_dx_at_tan0p4=next(x['com_dx_D'] for x in rows if x['tan'] >= 0.4), com_dx_at_tan0p3=next(x['com_dx_D'] for x in rows if x['tan'] >= 0.3))
            elif ev == 'slump': e[s] = dict(com_dx=l['com_dx_D'], com_dy=l['com_dy_D'], over_1D=l['moved_over_1D'], over_0p1D=l['moved_over_0p1D'], x95=l['x95_D'], quiet_at=r['quiet_at'])
            else: e[s] = dict(kicked=l['kicked_abs_D'], sum=l['sum_D'], max=l['max_D'], over_0p01D=l['over_0p01D'], over_0p1D=l['over_0p1D'], quiet_at=r['quiet_at'])
        ks = [k for k in keys if k == ev or k.startswith(ev + '@')]
        if ks:
            k = keys.index(ks[-1] if ev != 'ramp' else 'ramp@44'); e['field_distance_to_A'] = {s: round(fdist(F['A'][k], F[s][k]), 4) for s in ('A1', 'B', 'Z')}
        row['events'][ev] = e
json.dump(res, open(outf, 'w'), indent=1)
# ---- table: mean over piles and events of a class ----
cls = [('control', ['control']), ('kick 1 m/s', ['down:1:0', 'down:1:1', 'down:1:2', 'side:1:0']), ('kick 4 m/s', ['down:4:0']), ('removal', ['remove:0', 'remove:1']), ('slump', ['slump']), ('ramp (field at tan 0.4)', ['ramp'])]
print('field distance to the response from A  (L1 of the displacement difference / mean L1 of the two responses); mean over piles')
print('iters  class                       A1 (noise)   B (other equilibrium)   Z (amnesia)')
for it in sorted(res):
    for name, evs in cls:
        v = {s: [] for s in ('A1', 'B', 'Z')}
        for seed in res[it]:
            for ev in evs:
                fd = res[it][seed]['events'][ev].get('field_distance_to_A')
                if fd:
                    for s in v: v[s].append(fd[s])
        if name == 'control': continue
        print(f'{it:5d}  {name:26s} ' + '   '.join(f'{np.mean(v[s]):10.3f}' for s in v))
print('\nscalar observables, mean |log ratio to A| (counts +1, lengths +0.001 D)')
print('iters  class        observable      A1      B       Z')
def lr(a, b, dd): return abs(math.log((max(a, 0) + dd) / (max(b, 0) + dd)))
for it in sorted(res):
    for name, evs, obs in [('kicks', ['down:1:0', 'down:1:1', 'down:1:2', 'side:1:0', 'down:4:0'], [('kicked', .001), ('over_0p01D', 1), ('sum', .001)]), ('removal', ['remove:0', 'remove:1'], [('over_0p01D', 1), ('sum', .001)]), ('slump', ['slump'], [('over_0p1D', 1), ('over_1D', 1)])]:
        for ob, dd in obs:
            v = {s: [] for s in ('A1', 'B', 'Z')}
            for seed in res[it]:
                for ev in evs:
                    e = res[it][seed]['events'][ev]
                    for s in v: v[s].append(lr(e[s][ob], e['A'][ob], dd))
            print(f'{it:5d}  {name:10s}  {ob:12s} ' + '  '.join(f'{np.mean(v[s]):6.3f}' for s in v))
print('\nramp: tan at which 5 % of the bodies have moved 0.1 D   (A, A1, B, Z per pile)')
for it in sorted(res):
    print(f'{it:5d} ', ' | '.join(' '.join(str(res[it][seed]['events']['ramp'][s]['tan_5pct_bodies_0p1D']) for s in ('A', 'A1', 'B', 'Z')) for seed in sorted(res[it])))
print('\ncontrol: bodies moved > 0.01 D and time to exact rest from Z (amnesia), per pile')
for it in sorted(res):
    print(f'{it:5d} ', ' | '.join(f"{res[it][seed]['events']['control']['Z']['over_0p01D']} bodies, sum {res[it][seed]['events']['control']['Z']['sum']:.2f} D, quiet {res[it][seed]['events']['control']['Z']['quiet_at']}; B quiet {res[it][seed]['events']['control']['B']['quiet_at']} sum {res[it][seed]['events']['control']['B']['sum']}" for seed in sorted(res[it])))
print('\nnetwork L1 difference to A:', {seed: res[4][seed]['network_L1_vs_A'] for seed in res[4]})
