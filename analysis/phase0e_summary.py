"""Rebuild the audit summary from completed runs; never compare a failed run.

python3 analysis/phase0e_summary.py
Intermediate controls are kept in results/phase0e for provenance. Only the named
current runs below feed the summary. CPU work counts are not GPU milliseconds.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/phase0e'

def read(name):
    return json.loads((OUT / (name + '.json')).read_text())

def pair(reference, engine):
    r, e = read(reference), read(engine)
    valid = r['failed_at'] is None and e['failed_at'] is None
    result = dict(valid_comparison=valid, reference_source=r.get('label'), engine_source=e.get('label'))
    if not valid:
        return result
    peak = lambda x: max(f['work'] for f in x['frames'])
    result.update(work_gain=r['work_total']/e['work_total'], peak_frame_work_gain=peak(r)/peak(e),
                  D_relative_error=(e['d_end']-r['d_end'])/max(r['d_end'],1e-30),
                  reference_D=r['d_end'], engine_D=e['d_end'], mean_active=e['mean_e_frac'])
    return result

summary = {'scope': 'Phase 0e remediation and controlled experiments; architecture is not approved by these tests'}
summary['source_hash'] = read('regression')['meta']['source_hash']
if (OUT/'verification.json').exists():
    summary['verification'] = read('verification')
summary['s1'] = []
for seed in (1, 2):
    rn, en = f's1_seed{seed}_reference', f's1_seed{seed}_engine'
    p = pair(rn, en)
    r, e = read(rn), read(en)
    if p['valid_comparison']:
        a,b = r['fragment'], e['fragment']
        p.update(seed=seed, reference_omega=a['omega'], engine_omega=b['omega'],
                 omega_relative_error=abs(b['omega']-a['omega'])/abs(a['omega']),
                 velocity_absolute_error=sum((x-y)**2 for x,y in zip(a['v'],b['v']))**0.5)
    summary['s1'].append(p)

summary['fixed_duration_scale'] = [dict(L=L,T=0.12,g_scale=0.5,**pair(f'scale_L{L}_reference',f'scale_L{L}_engine')) for L in (2,4,8)]
summary['collapse'] = []
for mu in (0,0.5):
    modes={k:read(f'collapse_passive_mu{mu}_{k}') for k in ('reference','reference_half','engine')}
    item={'mu':mu,'comparison':pair(f'collapse_passive_mu{mu}_reference',f'collapse_passive_mu{mu}_engine')}
    for name,t in modes.items():
        masses=t['fragment_masses'];ke=max(t['ke_max'],1e-30)
        item[name] = dict(completed=t['failed_at'] is None,
            unclosed_energy_J=t['e_end']-t['e0']+t['ledger']['d_num_r'],
            unclosed_energy_over_ke=(t['e_end']-t['e0']+t['ledger']['d_num_r'])/ke,
            max_energy_rise_over_ke=t['e_max_rise']/ke,
            component_count=len(masses),cells=t['cells'],all_cells_separate=len(masses)==t['cells'],
            largest_mass_fraction=max(masses)/sum(masses),roundoff_accepts=t['work'].get('roundoff_accepts',0))
    summary['collapse'].append(item)

variants=read('s1variants')['data']
d0=variants[0]['dissipation_by_point']
summary['checkpoint_controls']=[dict(mode=v['mode'],omega=v['result']['fragment']['omega'],
    changed_damage_points=sum(abs(a-b)>1e-10 for a,b in zip(d0,v['dissipation_by_point'])),
    D=v['result']['D'],completed=v['result']['completed']) for v in variants]
summary['pulse_2d_projection_only']=read('pulse2d')['data']
summary['pulse_actual_coupler']=read('pulse_engine')['data']
summary['pulse_1d_control']=read('pulse')['data']
num=read('numeric3d')['data']
summary['numeric3d']={'scope':num['scope'],'models':num['models'],'blocks':len(num['local_blocks']),
    'failed_blocks':sum(b['fixed'] is None for b in num['local_blocks']),
    'worst_solution_error':max(b['fixed']['max_solution_error'] for b in num['local_blocks'] if b['fixed'])}
gpu=json.loads((OUT/'gpu/audit.json').read_text())
fh=gpu['float_history']['meshes'][0]['timings'];fx=gpu['fixed']['meshes'][0]['timings']
summary['gpu']={'scope':gpu['scope'],'adapter':gpu['probe']['adapter'],'kernel_sha256':gpu.get('kernel_sha256'),
    'timings':[dict(steps=t['substeps'],fixed_ms=t['gpu_ms']['median'],float_ms=next(x['gpu_ms']['median'] for x in fh if x['variant']=='E-B-H history' and x['substeps']==t['substeps']),
        ratio=t['gpu_ms']['median']/next(x['gpu_ms']['median'] for x in fh if x['variant']=='E-B-H history' and x['substeps']==t['substeps'])) for t in fx],
    'cpu_reference':json.loads((OUT/'gpu/fixed_ref_1k_audit.json').read_text())}
summary['limits']=[
    'No nonlinear fracture run with an accepted new dynamic far-field port yet.',
    'Elastic controls do not validate a material law or certify crack-path accuracy.',
    'Single desktop Apple GPU, 991 cells, existing kernels; not mobile performance or cross-vendor proof.',
    '3D probe is linear elastic only. Integer LDL is a CPU wide-arithmetic oracle, not a GPU solver.',
    'A saturated collapse cannot discriminate fracture fidelity by total dissipation.',
    'CPU work is an estimate. Mean work gains do not impose a worst-frame bound.',
]
(OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
for key in ('s1','fixed_duration_scale','collapse','checkpoint_controls','numeric3d','gpu'):
    print(key, json.dumps(summary[key],ensure_ascii=False))
