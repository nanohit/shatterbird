#!/usr/bin/env python3
"""Phase 1g: slump, tilt, support removal and cost (work of incremental stepping, work of change-driven iterations)
per variant. Reads the resp2 records of lab3.cjs. usage: g_events.py <cache dir> <n> <out.json> name1,name2,..."""
import json, sys, os, statistics as st
SEEDS = [int(x) for x in os.environ.get('SEEDS', '11,12,13').split(',')]
d, n, out, names = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4].split(',')
res = {}
def mean(x): return sum(x) / len(x) if x else None
for name in names:
    runs = []
    for seed in SEEDS:
        f = f'{d}/r2_{n}_{seed}_{name}.json'
        if os.path.exists(f): j = json.load(open(f)); runs += [(seed, r, j['N'], j['prm']['iters']) for r in j['runs']]
    if not runs: continue
    v = {'runs': len(runs)}
    N, I = runs[0][2], runs[0][3]
    for kind, sp in (('down', '0.25'), ('down', '1'), ('down', '4'), ('side', '1')):
        rs = [r for _, r, _, _ in runs if r['event'].startswith(f'{kind}:{sp}:')]
        if not rs: continue
        last = [r['rows'][-1] for r in rs]; q = [r['quiet_at'] for r in rs]
        v[f'{kind}{sp}'] = {'kicks': len(rs), 'quiet_within_800': sum(1 for x in q if x > 0), 'quiet_at_median': st.median([x if x > 0 else 801 for x in q]),
            'work_incremental_mean': round(mean([r['work'] for r in last])), 'work_change_driven_mean': round(mean([r.get('lazy_work', 0) for r in last])),
            'anchor_impulse_cum_over_kick_momentum': round(mean([r['anchor_impulse_cum'] for r in last]) / (float(sp) * 5436 * 32 * (1.5 if kind == 'down' else 1)), 3)}
    sl = [r for _, r, _, _ in runs if r['kind'] == 'slump']
    if sl:
        v['slump'] = {str(t): {'x95_D': round(mean([row['x95_D'] for r in sl for row in r['rows'] if row['t'] == t]), 3), 'com_dx_D': round(mean([row['com_dx_D'] for r in sl for row in r['rows'] if row['t'] == t]), 4),
            'com_dy_D': round(mean([row['com_dy_D'] for r in sl for row in r['rows'] if row['t'] == t]), 4),
            'moved_over_1D': [row['moved_over_1D'] for r in sl for row in r['rows'] if row['t'] == t], 'moved_over_0p1D': [row['moved_over_0p1D'] for r in sl for row in r['rows'] if row['t'] == t]} for t in (240, 960, 3840)}
        v['slump']['quiet_at'] = [r['quiet_at'] for r in sl]; v['slump']['work_change_driven'] = [r['rows'][-1].get('lazy_work', 0) for r in sl]; v['slump']['work_incremental'] = [r['rows'][-1]['work'] for r in sl]
    for R in ('0.2', '0.3', '0.4', '0.5'):
        tl = [r for _, r, _, _ in runs if r['event'] == f'tilt:{R}']
        if tl: last = [r['rows'][-1] for r in tl]; v[f'tilt{R}'] = {'com_dx_D': [round(x['com_shift_D'][0], 4) for x in last], 'moved_over_0p1D': [x['over_0p1D'] for x in last], 'moved_over_1D': [x.get('moved_over_1D') for x in last], 'quiet_at': [r['quiet_at'] for r in tl]}
    rm = [r for _, r, _, _ in runs if r['kind'] == 'remove']
    if rm:
        last = [r['rows'][-1] for r in rm]; first = [r['rows'][0] for r in rm]
        v['remove'] = {'events': len(rm), 'sum_D_mean': round(mean([x['sum_D'] for x in last]), 3), 'max_D_median': round(st.median([x['max_D'] for x in last]), 4), 'over_0p1D_total': sum(x['over_0p1D'] for x in last), 'over_0p01D_total': sum(x['over_0p01D'] for x in last),
            'moving_after_6_substeps': sum(1 for x in first if x['changed'] > 0), 'quiet_within_800': sum(1 for r in rm if r['quiet_at'] > 0), 'work_change_driven_mean': round(mean([x.get('lazy_work', 0) for x in last])), 'work_incremental_mean': round(mean([x['work'] for x in last]))}
    res[name] = v
json.dump(res, open(out, 'w'), indent=1)
print('kicks: quiet within 800 | median substep of quiet | work incremental | work change-driven | anchor impulse / kick momentum')
for name, v in res.items():
    print(f'{name:8s}', '  '.join(f"{k}: {c['quiet_within_800']:2d}/{c['kicks']} q{c['quiet_at_median']:5.0f} w{c['work_incremental_mean']/1e6:6.2f}M l{c['work_change_driven_mean']/1e6:6.2f}M a{c['anchor_impulse_cum_over_kick_momentum']:5.2f}" for k, c in v.items() if k[:4] in ('down', 'side')))
print('\nslump at 240 / 960 / 3840: x95 | com dx | moved > 1 D per pile ; quiet_at ; work change-driven (M)')
for name, v in res.items():
    if 'slump' in v: s = v['slump']; print(f'{name:8s}', ' | '.join(f"{s[t]['x95_D']:.3f} {s[t]['com_dx_D']:.4f} {s[t]['moved_over_1D']} {s[t]['moved_over_0p1D']}" for t in ('240', '960', '3840')), s['quiet_at'], [round(x / 1e6, 1) for x in s['work_change_driven']], [round(x / 1e6, 1) for x in s['work_incremental']])
print('\ntilt: com dx per pile | moved > 0.1 D | quiet_at')
for name, v in res.items():
    print(f'{name:8s}', '  '.join(f"{R}: {v['tilt' + R]['com_dx_D']} {v['tilt' + R]['moved_over_0p1D']} {v['tilt' + R]['quiet_at']}" for R in ('0.2', '0.3', '0.4', '0.5') if 'tilt' + R in v))
print('\nsupport removal (18 events): sum D mean | max D median | bodies > 0.1 D | > 0.01 D | moving after 6 substeps | quiet | work change-driven')
for name, v in res.items():
    if 'remove' in v: r = v['remove']; print(f'{name:8s}', r['sum_D_mean'], r['max_D_median'], r['over_0p1D_total'], r['over_0p01D_total'], r['moving_after_6_substeps'], r['quiet_within_800'], r['work_change_driven_mean'], r['work_incremental_mean'])
