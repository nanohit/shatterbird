"""Phase 2a experiments on the compliant contact lab (soft.py).
  python3 lab/soft_exp.py settle <seed> <ell> <out.npz>           pile at rest under the explicit law
  python3 lab/soft_exp.py run <state.npz> <method> <out.json>     events from that state; method: e<sub> | i<iters>[j|c] | iconv
  SB_EVENTS=control,down:1:0,...   SB_T=1.0   SB_PERTURB=1e-9 (noise control)   SB_AMNESIA=xi (forget tangential slips)
"""
import sys, os, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from soft import *

def save(W, f): np.savez(f, x=W.x, v=W.v, w=W.w, pi=W.pi, pj=W.pj, xi=W.xi, wxi=W.wxi, th=W.th, wth=W.wth, box=W.box, ell=W.ell, xref=W.xref, mur=W.mur, zeta=W.zeta)
def load(f, **kw):
    z = np.load(f); W = World(z['x'], z['box'], ell=float(z['ell']), mur=float(z['mur']), zeta=float(z['zeta']), **kw)
    W.v = z['v'].copy(); W.w = z['w'].copy(); W.pi = z['pi'].copy(); W.pj = z['pj'].copy(); W.xi = z['xi'].copy(); W.wxi = z['wxi'].copy(); W.th = z['th'].copy(); W.wth = z['wth'].copy(); W.xref = z['xref'].copy(); return W

def exposed(W):
    """six exposed spheres (nothing above within 60 degrees), nearest to the centre and to five points on a ring."""
    N = W.N; ex = np.ones(N, bool); n, d, wd = W.geometry(); t = d > -0.05 * D
    up = n[:, 1]; ex[W.pj[t & (up > 0.5)]] = False; ex[W.pi[t & (up < -0.5)]] = False
    cx, cz = W.box[0] / 2, W.box[2] / 2; rad = 0.25 * min(W.box[0], W.box[2]); out = []
    for q in range(6):
        tx = cx + (rad * np.cos(q * 2 * np.pi / 5) if q else 0); tz = cz + (rad * np.sin(q * 2 * np.pi / 5) if q else 0)
        sc = W.x[:, 1] - 2 * np.hypot(W.x[:, 0] - tx, W.x[:, 2] - tz); sc[~ex] = -1e9; sc[out] = -1e9; out.append(int(sc.argmax()))
    return out
def deep(W, target):
    top = W.x[:, 1].max(); sc = np.hypot(W.x[:, 0] - W.x[target, 0], W.x[:, 2] - W.x[target, 2]) + 2 * np.abs(W.x[:, 1] - top / 3); return int(sc.argmin())

def observe(W, x0, kick):
    a = W.alive; u = (W.x - x0)[a] / D; d = np.linalg.norm(u, axis=1)
    o = dict(t=round(W.t, 5), sum_D=float(d.sum()), max_D=float(d.max()), over_0p01D=int((d > 0.01).sum()), over_0p1D=int((d > 0.1).sum()), over_1D=int((d > 1).sum()),
             com_D=[float(c) for c in u.mean(0)], ke=float(W.energy()), x95_D=float(np.quantile(W.x[a, 0], 0.95) / D))
    if kick >= 0: o['kicked_D'] = [float(c) for c in (W.x[kick] - x0[kick]) / D]; o['kicked_abs_D'] = float(np.linalg.norm(W.x[kick] - x0[kick]) / D)
    return o

def advance(W, method, T, hook=None):
    """advance by T with the chosen method; returns per-step solver records for the implicit ones."""
    rec = []; t_end = W.t + T - 1e-12
    if method[0] == 'e':
        sub = int(method[1:]); dt = W.h / sub
        while W.t < t_end:
            if hook: hook(W)
            for _ in range(sub): W.step_explicit(dt)
    elif method[0] == 'm':        # implicit midpoint, m<iters>n | mconv
        conv = method == 'mconv'; it = 3000 if conv else int(method[1:].rstrip('n'))
        while W.t < t_end:
            if hook: hook(W)
            r = W.step_midpoint(iters=it, method='nest', tol=1e-12 * W.N if conv else 0.0); rec.append((r['iters'], r['cert']))
    else:
        conv = method == 'iconv'; kind = 'nest' if conv or method.endswith('n') else 'jacobi' if method.endswith('j') else 'cheb'; it = 3000 if conv else int(method[1:].rstrip('jcn'))
        while W.t < t_end:
            if hook: hook(W)
            r = W.step_implicit(iters=it, method=kind, tol=1e-12 * W.N if conv else 0.0); rec.append((r['iters'], r['cert']))
    return rec

def run_events(state, method, events, T):
    W0 = load(state); N = W0.N; targets = exposed(W0); out = dict(state=state, method=method, N=N, ell=W0.ell, mur=W0.mur, targets=targets, runs={})
    pert = float(os.environ.get('SB_PERTURB', 0)); amn = os.environ.get('SB_AMNESIA', '')
    fields = {}
    for ev in events:
        k = ev.split(':'); W = load(state); kick = -1; t0 = time.time(); hook = None
        if pert: W.x += (np.random.default_rng(7).random(W.x.shape) - 0.5) * 2 * pert * D
        if amn == 'xi': W.xi[:] = 0; W.wxi[:] = 0; W.th[:] = 0; W.wth[:] = 0
        x0 = W.x.copy(); TT = T
        if k[0] in ('down', 'side'):
            kick = targets[int(k[2])]; v = float(k[1]); W.v[kick] = (v / 2, -v, 0) if k[0] == 'down' else (v, 0, 0)
        elif k[0] == 'remove':
            g = deep(W, targets[int(k[1])]); W.alive[g] = False; W.x[g] = (-10 - g, -10, -10); W.pairs(force=True)
        elif k[0] == 'slump': W.box[0] = 1e9; TT = 2 * T
        elif k[0] == 'tilt': tn = float(k[1]); c = 1 / np.hypot(1, tn); W.grav = np.array([G * tn * c, -G * c, 0])
        elif k[0] == 'ramp':          # gravity rotates slowly: tan from 0 to 0.8 over 4 s
            TT = 4.0; t_start = W.t
            def hook(Wx):
                tn = 0.8 * (Wx.t - t_start) / 4.0; c = 1 / np.hypot(1, tn); Wx.grav = np.array([G * tn * c, -G * c, 0])
        rows = []; rec = []; nchunk = 8
        for q in range(nchunk):
            rec += advance(W, method, TT / nchunk, hook); o = observe(W, x0, kick)
            if k[0] == 'ramp': o['tan'] = 0.8 * (q + 1) / nchunk
            rows.append(o)
        r = dict(rows=rows, wall_s=round(time.time() - t0, 1), evals=W.evals)
        if rec: it = np.array([a for a, b in rec]); ce = np.array([b for a, b in rec]); r['solver'] = dict(iters_mean=float(it.mean()), iters_max=int(it.max()), cert_max_J=float(ce.max()), cert_mean_J=float(ce.mean()))
        out['runs'][ev] = r; fields[ev] = ((W.x - x0) / D).astype(np.float32)
        l = rows[-1]; print(f"{method} {ev}: kicked={l.get('kicked_abs_D', 0):.4f} sum={l['sum_D']:.3f} max={l['max_D']:.3f} n01={l['over_0p1D']} n001={l['over_0p01D']} comx={l['com_D'][0]:.4f} ke={l['ke']:.2e} wall={r['wall_s']}", flush=True)
    return out, fields

if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'settle':
        seed, ell, f = int(sys.argv[2]), float(sys.argv[3]), sys.argv[4]; nx = int(os.environ.get('SB_NX', 8)); layers = int(os.environ.get('SB_LAYERS', 16))
        p, box = rain(nx, nx, layers, seed=seed); W = World(p, box, ell=ell, mur=float(os.environ.get('SB_MUR', 0.1)), zeta=float(os.environ.get('SB_ZETA', 0.5))); sub = int(os.environ.get('SB_SUB', 64)); k = 0
        while W.t < 8.0:
            W.step_explicit(W.h / sub); k += 1
            if k % (480 * sub // 4) == 0:
                print(f't={W.t:.2f} KE/N={W.energy() / W.N:.3e} vmax={np.abs(W.v).max():.2e}', flush=True)
                if W.t > 1.5 and np.abs(W.v).max() < 1e-7 and np.abs(W.w).max() * R < 1e-7: break
        W.t = 0.0; save(W, f); print('saved', f, 'N', W.N, 'pairs', len(W.pi), 'height/D', W.x[:, 1].max() / D)
    elif mode == 'run':
        state, method, f = sys.argv[2], sys.argv[3], sys.argv[4]
        ev = os.environ.get('SB_EVENTS', 'control,down:1:0,down:1:1,down:1:2,down:4:0,down:4:1,side:1:0,side:1:1,remove:0,remove:1,slump,ramp').split(',')
        out, fields = run_events(state, method, ev, float(os.environ.get('SB_T', 1.0)))
        json.dump(out, open(f, 'w')); np.savez_compressed(f.replace('.json', '_fields.npz'), **{k.replace(':', '_'): v for k, v in fields.items()})
