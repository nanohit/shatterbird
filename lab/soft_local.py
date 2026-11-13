"""Phase 2a: depth, locality, wake and state-or-cache measurements on the compliant contact (soft.py).
  python3 lab/soft_local.py depth <out.json>            columns of L spheres released at zero overlap: iterations per step against L and ell
  python3 lab/soft_local.py local <pile.npz> <out.json> decay of the response of one implicit step with graph distance; error of solving only a neighbourhood
  python3 lab/soft_local.py wake <pile.npz> <out.json>  bodies above a velocity threshold after a kick, per step
  python3 lab/soft_local.py radius <pile.npz> <out.json> whole events with only the bodies within r hops of the source moving (the rest held)
  python3 lab/soft_local.py state <pile.npz> <out.json> what of the contact state is state: slips forgotten, another history, noise; solver warm start
"""
import sys, os, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from soft import *
from soft_exp import load, save, exposed, deep, observe

def column(L, ell, zeta=0.5):
    x = np.array([[R, R + i * D, R] for i in range(L)]); return World(x, (D, 1e9, D), ell=ell, zeta=zeta)

def hops(W, src):
    n, d, wd = W.geometry(); t = d > 0; adj = [[] for _ in range(W.N)]
    for i, j in zip(W.pi[t], W.pj[t]): adj[i].append(j); adj[j].append(i)
    dist = np.full(W.N, -1); dist[src] = 0; q = [src]
    for a in q:
        for b in adj[a]:
            if dist[b] < 0: dist[b] = dist[a] + 1; q.append(b)
    return dist

if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'depth':
        out = dict(rows=[]); tolJ = 1e-12
        for ell in (1.0, 2.0, 4.0, 8.0):
            for L in (8, 16, 32, 64, 128):
                for meth in ('nest', 'jacobi'):
                    if meth == 'jacobi' and (ell > 2 or L > 64): continue
                    W = column(L, ell); its = []; rest = -1; ymin = []; t0 = time.time()
                    for k in range(1, 2401):
                        r = W.step_implicit(iters=20000, method=meth, tol=tolJ * W.N); its.append(r['iters'])
                        if rest < 0 and W.energy() < 1e-14 * W.N and k > 5: rest = k; break
                    comp = (R + (L - 1) * D - W.x[-1, 1]) / D; ideal = G * L * (L - 1) / 2 / (W.kn / W.m) / D + G * L / (W.kn / W.m) / D
                    row = dict(ell=ell, L=L, method=meth, iters_mean=float(np.mean(its)), iters_max=int(max(its)), iters_p90=float(np.quantile(its, 0.9)), steps_to_rest=rest, compression_D=float(comp), compression_static_D=float(ideal), lam_lo=r['lam_lo'], wall=round(time.time() - t0, 1))
                    out['rows'].append(row); print(json.dumps(row), flush=True)
        # fixed budgets on the 64-column at ell 2: does an unconverged step still come to rest at the right place
        for meth, it in (('nest', 8), ('nest', 16), ('nest', 32), ('jacobi', 16), ('jacobi', 32)):
            W = column(64, 2.0); rest = -1; peak = 0
            for k in range(1, 4801):
                r = W.step_implicit(iters=it, method=meth); peak = max(peak, (R + 63 * D - W.x[-1, 1]) / D)
                if W.energy() < 1e-14 * W.N and k > 5: rest = k; break
            row = dict(ell=2.0, L=64, method=f'{meth} {it}', steps_to_rest=rest, compression_D=float((R + 63 * D - W.x[-1, 1]) / D), peak_compression_D=float(peak), cert_last=r['cert'])
            out['rows'].append(row); print(json.dumps(row), flush=True)
        json.dump(out, open(sys.argv[2], 'w'), indent=1)
    if mode == 'local':
        f, outf = sys.argv[2], sys.argv[3]; out = dict(pile=f, cases=[])
        W0 = load(f); tg = exposed(W0)
        for src, dv in ((tg[0], (0, -0.2, 0)), (tg[1], (0.2, 0, 0)), (deep(W0, tg[0]), (0, -0.2, 0))):
            dist = hops(W0, src); base = load(f); r0 = base.step_implicit(iters=20000, method='nest', tol=1e-22 * base.N); u0 = np.hstack([base.v, base.w])
            W = load(f); W.v[src] = dv; r = W.step_implicit(iters=20000, method='nest', tol=1e-22 * W.N); u = np.hstack([W.v, W.w]); du = u - u0
            Mh = np.array([W.m] * 3 + [W.I] * 3); en = 0.5 * (du ** 2 * Mh).sum(1); tot = en.sum(); shells = []
            for h in range(0, dist.max() + 1):
                m = dist == h
                if m.any(): shells.append(dict(hops=h, bodies=int(m.sum()), energy_share=float(en[m].sum() / tot), max_speed_ratio=float(np.sqrt(en[m].max() / en[src]))))
            kappa = 1 / r['lam_lo']; q = (np.sqrt(kappa) - 1) / (np.sqrt(kappa) + 1)
            # restriction: only bodies within r hops are solved, the rest keep the velocity of the undisturbed step
            restr = []
            for rad in (0, 1, 2, 3, 4, 6, 8):
                Wr = load(f); Wr.v[src] = dv; act = (dist >= 0) & (dist <= rad); Wr.step_implicit(iters=20000, method='nest', tol=1e-22 * Wr.N, active=act)
                ur = np.hstack([Wr.v, Wr.w]); e = 0.5 * ((ur - u) ** 2 * Mh).sum()
                restr.append(dict(radius_hops=rad, bodies_solved=int(act.sum()), energy_error_over_response=float(e / tot), source_speed_error=float(np.linalg.norm(ur[src, :3] - u[src, :3]) / np.linalg.norm(du[src, :3]))))
            c = dict(source=int(src), kick=dv, iterations=r['iters'], kappa_bound=float(kappa), q_bound=float(q), shells=shells, restriction=restr); out['cases'].append(c)
            print('source', src, 'kick', dv, 'kappa', round(kappa, 1), 'q bound', round(q, 3)); print(' shells', [(s['hops'], s['bodies'], '%.1e' % s['energy_share'], '%.1e' % s['max_speed_ratio']) for s in shells[:12]]); print(' restriction', [(s['radius_hops'], s['bodies_solved'], '%.1e' % s['energy_error_over_response'], '%.1e' % s['source_speed_error']) for s in restr], flush=True)
        json.dump(out, open(outf, 'w'), indent=1)
    if mode == 'wake':
        f, outf = sys.argv[2], sys.argv[3]; out = dict(pile=f, runs=[])
        for v, meth in ((1.0, 'i16n'), (4.0, 'i16n'), (1.0, 'e64'), (4.0, 'e64')):
            W = load(f); tg = exposed(W); W.v[tg[0]] = (v / 2, -v, 0); thr = (1e-4, 1e-3, 1e-2); ever = [np.zeros(W.N, bool) for _ in thr]; rows = []; last = [0, 0, 0]
            for k in range(1, 481):
                if meth[0] == 'e':
                    for _ in range(64): W.step_explicit(W.h / 64)
                else: W.step_implicit(iters=16, method='nest')
                sp = np.maximum(np.linalg.norm(W.v, axis=1), np.linalg.norm(W.w, axis=1) * R); cnt = []
                for q, t in enumerate(thr):
                    m = sp > t; ever[q] |= m; cnt.append(int(m.sum()))
                    if m.any(): last[q] = k
                if k in (1, 2, 4, 8, 16, 32, 64, 128, 256, 480): rows.append(dict(step=k, above=cnt, ever=[int(e.sum()) for e in ever]))
            body_steps = None
            out['runs'].append(dict(kick_m_s=v, method=meth, thresholds_m_s=thr, rows=rows, last_step_above=last, N=W.N)); print(v, meth, 'last step above', last, 'ever', [int(e.sum()) for e in ever], 'of', W.N, [(r['step'], r['above']) for r in rows], flush=True)
        json.dump(out, open(outf, 'w'), indent=1)
    if mode == 'state':
        f, outf = sys.argv[2], sys.argv[3]; out = dict(pile=f, variants={})
        A = load(f); tg = exposed(A)
        # B: the same pile after a history: gravity tilted to tan 0.15 for 0.3 s, back, 0.7 s of rest (explicit)
        B = load(f); c = 1 / np.hypot(1, 0.15); B.grav = np.array([G * 0.15 * c, -G * c, 0])
        for _ in range(int(0.3 * 480 * 64)): B.step_explicit(B.h / 64)
        B.grav = np.array([0, -G, 0])
        for _ in range(int(0.7 * 480 * 64)): B.step_explicit(B.h / 64)
        nA, dA, _ = A.geometry(); B.pairs(force=True); A2 = load(f); A2.pairs(force=True); nB, dB, _ = B.geometry(); nA, dA, _ = A2.geometry()
        fnA = np.maximum(0, A.kn * dA); fnB = np.maximum(0, B.kn * dB); common = (len(fnA) == len(fnB)) and bool((A2.pi == B.pi).all() and (A2.pj == B.pj).all())
        out['history_B'] = dict(moved_max_D=float(np.abs(B.x - A.x).max() / D), moved_mean_D=float(np.linalg.norm(B.x - A.x, axis=1).mean() / D), ke=B.energy(), same_pair_list=common,
                                normal_force_L1_diff=float(np.abs(fnA - fnB).sum() / fnA.sum()) if common else None, tangential_L1_diff=float(np.abs(A2.kt * A2.xi - B.kt * B.xi).sum() / np.abs(A2.kt * A2.xi).sum()) if common else None,
                                mobilisation_A=float((np.linalg.norm(A2.kt * A2.xi, axis=1)[fnA > 0] / (A.mu * fnA[fnA > 0])).mean()), mobilisation_B=float((np.linalg.norm(B.kt * B.xi, axis=1)[fnB > 0] / (B.mu * fnB[fnB > 0])).mean()) if common else None)
        print('history B', out['history_B'], flush=True)
        def variant(name):
            W = load(f)
            if name == 'B': W.x = B.x.copy(); W.v = B.v.copy(); W.w = B.w.copy(); W.pi, W.pj, W.xi, W.th, W.wxi, W.wth, W.xref = B.pi.copy(), B.pj.copy(), B.xi.copy(), B.th.copy(), B.wxi.copy(), B.wth.copy(), B.xref.copy()
            if name == 'A1': r = np.random.default_rng(3); W.xi *= 1 + 1e-6 * (2 * r.random(W.xi.shape) - 1)
            if name == 'Z': W.xi[:] = 0; W.wxi[:] = 0; W.th[:] = 0; W.wth[:] = 0
            if name == 'Zslip': W.xi[:] = 0; W.wxi[:] = 0
            return W
        events = ['control', 'down:1:0', 'down:1:1', 'down:4:0', 'side:1:0', 'remove:1', 'tilt:0.4']
        fields = {}
        for name in ('A', 'A1', 'B', 'Z', 'Zslip'):
            res = {}
            for ev in events:
                W = variant(name); x0 = W.x.copy(); k = ev.split(':'); kick = -1
                if k[0] in ('down', 'side'): kick = tg[int(k[2])]; v = float(k[1]); W.v[kick] = (v / 2, -v, 0) if k[0] == 'down' else (v, 0, 0)
                elif k[0] == 'remove': g = deep(W, tg[int(k[1])]); W.alive[g] = False; W.x[g] = (-10 - g, -10, -10); x0 = W.x.copy(); W.pairs(force=True)
                elif k[0] == 'tilt': tn = float(k[1]); c = 1 / np.hypot(1, tn); W.grav = np.array([G * tn * c, -G * c, 0])
                for _ in range(480): W.step_implicit(iters=32, method='nest')
                o = observe(W, x0, kick); res[ev] = o; fields[(name, ev)] = (W.x - x0) / D
                print(name, ev, 'sum', round(o['sum_D'], 4), 'n001', o['over_0p01D'], 'n01', o['over_0p1D'], 'kicked', round(o.get('kicked_abs_D', 0), 4), 'ke', '%.1e' % o['ke'], flush=True)
            out['variants'][name] = res
        fd = lambda x, y: float(np.abs(x - y).sum() / max(1e-12, 0.5 * (np.abs(x).sum() + np.abs(y).sum())))
        out['field_distance_to_A'] = {ev: {n: fd(fields[('A', ev)], fields[(n, ev)]) for n in ('A1', 'B', 'Z', 'Zslip')} for ev in events if ev != 'control'}
        print(json.dumps(out['field_distance_to_A'], indent=0))
        # the solver's warm start is a cache: cold start (from the free velocity) against warm start, converged and at 16 iterations
        rows = []
        for it, tol in ((20000, 1e-14), (16, 0.0)):
            xs = []
            for warm in (True, False):
                W = load(f); W.v[tg[0]] = (2, -4, 0); x0 = W.x.copy(); n_it = 0
                for _ in range(240): n_it += W.step_implicit(iters=it, method='nest', tol=tol * W.N, warm=warm)['iters']
                xs.append((W.x - x0) / D); rows.append(dict(iters=it, warm=warm, iterations_total=n_it, kicked_D=float(np.linalg.norm(xs[-1][tg[0]])), sum_D=float(np.linalg.norm(xs[-1], axis=1).sum())))
            rows[-1]['field_distance_warm_cold'] = fd(xs[0], xs[1])
        out['warm_start'] = rows; print(rows)
        json.dump(out, open(outf, 'w'), indent=1)

    if mode == 'radius':
        f, outf = sys.argv[2], sys.argv[3]; out = dict(pile=f, events={}); W0 = load(f); tg = exposed(W0)
        for ev in ('down:1:0', 'down:4:0', 'down:4:1', 'side:1:0', 'remove:1'):
            k = ev.split(':'); rows = []; ref = None
            for rad in (99, 2, 3, 4, 5, 6, 8):
                W = load(f); kick = -1
                if k[0] in ('down', 'side'): kick = src = tg[int(k[2])]; v = float(k[1]); dist = hops(W, src); W.v[kick] = (v / 2, -v, 0) if k[0] == 'down' else (v, 0, 0)
                else: src = deep(W, tg[int(k[1])]); dist = hops(W, src); W.alive[src] = False; W.x[src] = (-10 - src, -10, -10); W.pairs(force=True)
                x0 = W.x.copy(); act = None if rad == 99 else (dist >= 0) & (dist <= rad)
                for _ in range(480): W.step_implicit(iters=16, method='nest', active=act)
                o = observe(W, x0, kick); d = np.linalg.norm((W.x - x0) / D, axis=1)
                if rad == 99: ref = (W.x - x0) / D; refd = d
                row = dict(radius_hops=rad, bodies_moving=int(W.N if act is None else act.sum()), kicked_D=o.get('kicked_abs_D'), sum_D=o['sum_D'], over_0p01D=o['over_0p01D'], over_0p1D=o['over_0p1D'], max_D=o['max_D'],
                           field_error=float(np.abs((W.x - x0) / D - ref).sum() / np.abs(ref).sum()), outside_share_of_reference=float(0 if act is None else refd[~act].sum() / refd.sum()))
                rows.append(row); print(ev, {a: (round(b, 4) if isinstance(b, float) else b) for a, b in row.items()}, flush=True)
            out['events'][ev] = rows
        json.dump(out, open(outf, 'w'), indent=1)
