#!/usr/bin/env python3
"""Phase 1g: table of GPU time per substep from the xdev2 'time' records. usage: g_time.py <dir> [out.json]"""
import json, sys, glob, os, re
rows = []
for f in sorted(glob.glob(sys.argv[1] + '/time_*.json')):
    j = json.load(open(f))
    if 'time' not in j or not j['time']: print('no data', f, j.get('error')); continue
    r = j['time'][0]; m = re.match(r'time_(v2|v[34]_i(\d+))_(\w+?)_(\w+)\.json', os.path.basename(f)); spec = j['spec']
    rows.append({'device': m.group(4), 'bodies': r['n'], 'law': 'v2 (anchor)' if spec['hand'] == 0 else ('v4' if spec.get('cheb') else 'v3'), 'iterations': spec['iters'],
        'falling_full_ms': r['falling']['full_ms'], 'falling_inc_ms': r['falling']['inc_ms'], 'rest_by_substep': r['rest_by_substep'],
        'rest_full_ms': r['at_rest']['full_ms'], 'rest_inc_ms': r['at_rest']['inc_ms'],
        'settle_inc_ms_total': round(sum(p['inc_ms'] * 100 for p in r['settle']), 1), 'settle_full_ms_total': round(sum(p['full_ms'] * 100 for p in r['settle']), 1),
        'wake_quiet_by': r['wake_quiet_by'], 'wake_full_ms': r['wake_total']['full_ms'], 'wake_inc_ms': r['wake_total']['inc_ms'], 'wake_response': r.get('wake_response'),
        'identical_full_vs_incremental': [r['rest_identical'], r['wake_identical']], 'rest_hash': r['rest_hash'], 'wake_hash': r['wake_hash'], 'bytes_per_body': r['bytes_per_body'], 'anchors_pct_weight_after_wake': r['anchors_pct_weight'], 'anchored_bodies': r['anchored_bodies'], 'passes_per_substep': r['passes_per_substep']})
rows.sort(key=lambda x: (x['bodies'], x['device'], x['law'], x['iterations']))
if len(sys.argv) > 2: json.dump({'note': 'GPU time from timestamp queries, ms per substep unless a total is named; wake: kick of 1 m/s on a surface sphere, windows of 20 substeps until nothing is evaluated', 'rows': rows}, open(sys.argv[2], 'w'), indent=1)
print(f"{'device':10s} {'bodies':>7s} {'law':12s} it | falling full/inc | rest full/inc ms | settle inc total (rest by) | wake: quiet by, full / inc ms | kicked D, >0.01D | ident | hashes")
for x in rows:
    w = x['wake_response'] or {}
    print(f"{x['device']:10s} {x['bodies']:7d} {x['law']:12s} {x['iterations']:2d} | {x['falling_full_ms']:6.2f} {x['falling_inc_ms']:6.2f} | {x['rest_full_ms']:6.2f} {x['rest_inc_ms']:6.3f} | {x['settle_inc_ms_total']:8.0f} ({x['rest_by_substep']}) | {x['wake_quiet_by']:4d} {x['wake_full_ms']:8.1f} {x['wake_inc_ms']:7.2f} | {w.get('kicked_D')} {w.get('over_0p01D')} | {x['identical_full_vs_incremental']} | {x['rest_hash']} {x['wake_hash']} a={x['anchors_pct_weight_after_wake']}")
