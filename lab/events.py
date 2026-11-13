"""Phase 2a, experiment C: fracture decisions with a certificate, and the criticality of a bond as a property of the operator.

Structure: triangular lattice plate, unit masses, brittle bonds (stiffness 1, random strain limits), bottom row held,
top row pulled at a constant rate. Implicit step (backward Euler in displacement form):
    A u+ = M (2u - u-)/h^2 + f,     A = M/h^2 + K(broken set),      K = sum_e k_e b_e b_e'
Event rule of the model: after the step is solved, the most overloaded bond above its limit breaks and the step is
solved again; repeat until no bond is above its limit.

Certificate for a decision. For an approximate solution u~ with residual r = b - A u~ the strain error of bond e is
    |eps_e(u~) - eps_e(u)| = |b_e' A^-1 r| / L  <=  |b_e|_{A^-1} |r|_{A^-1} / L  <=  eta_e
with the computable bounds  |r|_{A^-1} <= h |r|_{M^-1}  and  |b_e|^2_{A^-1} <= 1 / (k_e + m / (2 h^2))   (A >= M/h^2 + k_e b_e b_e').
A bond is certainly intact if eps + eta < limit, certainly broken if eps - eta > limit. The solver iterates only until
every decision of the step is certain.

  python3 lab/events.py run <ell> <out.json>
  python3 lab/events.py critical <ell> <out.json>
"""
import sys, json, time, numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
sys.path.insert(0, __import__('os').path.dirname(__file__))
from memory import lattice

class Plate:
    def __init__(s, nx=40, ny=24, seed=1, ell=4.0, rate=2e-4, spread=0.3):
        s.P, s.B, s.dirs, s.L, _ = lattice(nx, ny) if nx * ny <= 400 else lattice_light(nx, ny); s.n = len(s.P); s.nb = len(s.B); s.nx, s.ny = nx, ny; r = np.random.default_rng(seed)
        s.lim = 0.01 * (1 + spread * (2 * r.random(s.nb) - 1)); s.h = ell; s.m = 1.0; s.rate = rate; s.bi = s.B[:, 0]; s.bj = s.B[:, 1]
        s.alive = np.ones(s.nb, bool); s.bot = np.arange(nx); s.top = np.arange((ny - 1) * nx, ny * nx); s.kb = 20.0     # boundary springs
        s.u = np.zeros(2 * s.n); s.um = np.zeros(2 * s.n); s.t = 0.0; s.events = []; s.bnd = np.concatenate([s.bot, s.top])
    def assemble(s, mass=True):
        a = s.alive; i, j, d = s.bi[a], s.bj[a], s.dirs[a]; rows = []; cols = []; vals = []
        for p in range(2):
            for q in range(2):
                v = d[:, p] * d[:, q]
                for (r_, c_, sg) in ((i, i, 1), (j, j, 1), (i, j, -1), (j, i, -1)): rows.append(2 * r_ + p); cols.append(2 * c_ + q); vals.append(sg * v)
        dg = np.zeros(2 * s.n); dg[2 * s.bnd] += s.kb; dg[2 * s.bnd + 1] += s.kb
        if mass: dg += s.m / s.h ** 2
        rows.append(np.arange(2 * s.n)); cols.append(np.arange(2 * s.n)); vals.append(dg)
        return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(2 * s.n, 2 * s.n))
    def rhs(s, t):
        b = s.m * (2 * s.u - s.um) / s.h ** 2; b[2 * s.top + 1] += s.kb * s.rate * t; return b
    def strain(s, u): u2 = u.reshape(s.n, 2); return ((u2[s.bj] - u2[s.bi]) * s.dirs).sum(1) / s.L
    def over(s, u): e = s.strain(u) / s.lim; e[~s.alive] = 0; return e
    def reaction(s, u, t): return float(s.kb * (s.rate * t - u[2 * s.top + 1]).sum())
    def Gdense(s):
        G = np.zeros((s.nb, 2 * s.n))
        for c, ((i, j), d) in enumerate(zip(s.B, s.dirs)): G[c, 2 * i:2 * i + 2] = -d; G[c, 2 * j:2 * j + 2] = d
        return G

def lattice_light(nx, ny):
    P = np.array([[(i + 0.5 * (j % 2)), j * np.sqrt(3) / 2] for j in range(ny) for i in range(nx)]); idx = lambda i, j: j * nx + i; B = []
    for j in range(ny):
        for i in range(nx):
            if i + 1 < nx: B.append((idx(i, j), idx(i + 1, j)))
            if j + 1 < ny:
                B.append((idx(i, j), idx(i, j + 1))); k = i + 1 if j % 2 else i - 1
                if 0 <= k < nx: B.append((idx(i, j), idx(k, j + 1)))
    B = np.array(B); d = P[B[:, 1]] - P[B[:, 0]]; L = np.linalg.norm(d, axis=1); return P, B, d / L[:, None], L, None

def blockjacobi(A, n):
    idx = np.arange(n); Dm = np.zeros((n, 2, 2)); Ad = A.diagonal(); Dm[:, 0, 0] = Ad[0::2]; Dm[:, 1, 1] = Ad[1::2]
    off = np.asarray(A[2 * idx, 2 * idx + 1]).ravel(); Dm[:, 0, 1] = off; Dm[:, 1, 0] = off
    return np.linalg.inv(Dm), Dm

def cheb(A, b, u, Di, lam_lo, lam_hi, stop, kmax=200000):
    """Chebyshev iteration on the block-Jacobi preconditioned system. stop(u, r, k) -> True ends it. Returns u, iterations, node updates."""
    th = (lam_hi + lam_lo) / 2; de = (lam_hi - lam_lo) / 2; sig = th / de; rho = 1 / sig; d = None; k = 0; n = len(b) // 2; work = 0; floor = 1e-13 * (np.abs(b).max() + 1e-300)
    while True:
        r = b - A @ u
        if stop(u, r, k) or k >= kmax: return u, k, work
        z = np.einsum('nij,nj->ni', Di, r.reshape(n, 2)).ravel()
        if k == 0: d = z / th
        else: rn = 1 / (2 * sig - rho); d = rn * rho * d + 2 * rn / de * z; rho = rn
        u = u + d; k += 1; work += int((np.abs(r).reshape(n, 2).max(1) > floor).sum())

def run(ell, policy, steps=260, seed=1, nx=40, ny=24, trace=None):
    """policy: ('exact',) direct solve | ('tol', rel) Chebyshev to a relative residual | ('fixed', k) | ('cert',) certified decisions | ('cert', rel) certified decisions and a floor on the state accuracy.
    trace: a reference run; then the true strain error against it is recorded while the event histories agree."""
    s = Plate(nx, ny, seed, ell); n = s.n; iters = 0; work = 0; solves = 0; ties = 0; t0 = time.time(); peak = 0.0; load = []; rho_sum = 0.0; drift = []
    eb = np.sqrt(1.0 / (1.0 + s.m / (2 * s.h ** 2))) / s.L          # |b_e|_{A^-1} / L for unit stiffness
    for step in range(steps):
        t = s.t + s.h; b = s.rhs(t); u = 2 * s.u - s.um            # extrapolated initial guess
        while True:
            A = s.assemble(); solves += 1
            if policy[0] == 'exact': u = spsolve(A.tocsc(), b)
            else:
                Di, Dm = blockjacobi(A, n); lam_lo = (s.m / s.h ** 2) / np.linalg.eigvalsh(Dm)[:, -1].max(); nb0 = np.linalg.norm(b); last = {}
                if policy[0] == 'tol': stop = lambda uu, r, k: np.linalg.norm(r) <= policy[1] * nb0
                elif policy[0] == 'fixed': stop = lambda uu, r, k: k >= policy[1]
                else:
                    def stop(uu, r, k):
                        rn = np.linalg.norm(r); last['rho'] = s.h * rn / np.sqrt(s.m); eta = eb * last['rho'] / s.lim; o = s.over(uu)
                        if len(policy) > 1 and rn > policy[1] * nb0: return False   # a floor on the accuracy of the state itself
                        if (o + eta).max() < 1: return True                    # no bond can be above its limit: the step is decided
                        c = int(o.argmax()); oth = o + eta; oth[c] = 0         # the model breaks the most overloaded bond: the leader must be certain
                        if o[c] - eta[c] > 1 and o[c] - eta[c] > oth.max(): return True
                        if rn <= 1e-13 * nb0: last['tie'] = 1; return True     # a tie below the resolution of the arithmetic: decided by the values
                        return False
                u, k, w = cheb(A, b, u, Di, lam_lo, 2.0, stop); iters += k; work += w; ties += last.get('tie', 0)
            o = s.over(u); c = int(o.argmax())
            if o[c] > 1.0: s.alive[c] = False; s.events.append((step, c)); continue
            if policy[0] == 'cert': rho_sum += last['rho']
            break
        s.um = s.u; s.u = u; s.t = t; f = s.reaction(u, t); load.append(f); peak = max(peak, f)
        if trace is not None and s.events == [e for e in trace['events'] if e[0] <= step]:
            drift.append(float(np.abs(s.strain(u) - s.strain(trace['U'][step]))[s.alive].max() / 0.01))
        if trace is None and policy[0] == 'exact': s.__dict__.setdefault('U', []).append(u.copy())
    return dict(policy=list(policy), ell=ell, events=s.events, n_events=len(s.events), iterations=iters, node_updates=work, solves=solves, ties=ties, wall=round(time.time() - t0, 1), final_u=s.u, alive=s.alive,
                peak_load=peak, load=load, U=s.__dict__.get('U'), rho_sum=rho_sum, drift_max=max(drift) if drift else None, steps_same_history=len(drift))

def compare(a, ref):
    ea, er = a['events'], ref['events']; k = 0
    while k < min(len(ea), len(er)) and ea[k] == er[k]: k += 1
    sa, sr = set(c for _, c in ea), set(c for _, c in er)
    return dict(identical_sequence=ea == er, first_difference_at_event=None if ea == er else k, broken_same=len(sa & sr), broken_only_here=len(sa - sr), broken_only_ref=len(sr - sa), jaccard=round(len(sa & sr) / max(1, len(sa | sr)), 4))

if __name__ == '__main__':
    mode, ell, outf = sys.argv[1], float(sys.argv[2]), sys.argv[3]
    if mode == 'run':
        out = dict(ell=ell, rows=[]); steps = int(440 * 4 / ell)
        for seed in (1, 2, 3):
            ref = run(ell, ('exact',), steps, seed); print(f'seed {seed}: reference {ref["n_events"]} events, {ref["solves"]} solves, peak load {ref["peak_load"]:.4f}, wall {ref["wall"]}', flush=True)
            for pol in [('cert',), ('tol', 1e-10), ('tol', 1e-6), ('tol', 1e-4), ('tol', 1e-3), ('tol', 1e-2), ('fixed', 64), ('fixed', 32), ('fixed', 16), ('fixed', 8)]:
                r = run(ell, pol, steps, seed, trace=ref); c = compare(r, ref)
                row = dict(seed=seed, policy=' '.join(str(x) for x in pol), events=r['n_events'], ref_events=ref['n_events'], iterations=r['iterations'], iterations_per_solve=round(r['iterations'] / r['solves'], 2), node_updates=r['node_updates'], ties=r['ties'],
                           peak_load_error=abs(r['peak_load'] - ref['peak_load']) / ref['peak_load'], load_curve_error=float(np.abs(np.array(r['load']) - np.array(ref['load'])).max() / ref['peak_load']),
                           strain_drift_max_over_limit=r['drift_max'], steps_with_same_history=r['steps_same_history'], steps=steps, sum_of_step_bounds=r['rho_sum'], **c)
                out['rows'].append(row); print(json.dumps(row), flush=True); json.dump(out, open(outf, 'w'), indent=1)
    if mode == 'critical':
        s = Plate(40, 24, 1, ell); out = dict(ell=ell)
        # damage the plate by running the model for a while, then analyse the operator
        ref = run(ell, ('exact',), int(float(sys.argv[4]) * 4 / ell) if len(sys.argv) > 4 else int(300 * 4 / ell), 1); s.alive = ref['alive']; u = ref['final_u']; nbroken = int((~s.alive).sum()); s.G = s.Gdense()
        for name, A in (('dynamic', s.assemble().toarray()), ('static', s.assemble(mass=False).toarray())):
            Ai = np.linalg.inv(A); Ga = s.G[s.alive]; T = Ga @ Ai @ Ga.T; rho = np.diag(T).copy(); ids = np.flatnonzero(s.alive)
            # exactness of the rank-one update: remove the bond with the largest strain, compare predicted and recomputed strains for the same right-hand side
            b = A @ u; e0 = Ga @ u; c = int(np.argmax(e0)); pred = e0 + T[:, c] * e0[c] / (1 - rho[c]); s2 = Plate(40, 24, 1, ell); s2.alive = s.alive.copy(); s2.alive[ids[c]] = False
            A2 = s2.assemble().toarray() if name == 'dynamic' else s2.assemble(mass=False).toarray(); e1 = Ga @ np.linalg.solve(A2, b); keep = np.arange(len(ids)) != c
            # decay of the influence |T_je| with graph distance between bonds (in bond-midpoint distance, lattice units)
            mid = (s.P[s.B[ids, 0]] + s.P[s.B[ids, 1]]) / 2; rows = []
            sample = np.random.default_rng(0).choice(len(ids), 200, replace=False)
            dd = np.linalg.norm(mid[sample][:, None, :] - mid[None, :, :], axis=2); Tn = np.abs(T[sample]) / np.sqrt(rho[sample][:, None] * rho[None, :])
            for lo, hi in ((0.5, 1.5), (1.5, 3), (3, 5), (5, 8), (8, 12), (12, 18), (18, 30)):
                m = (dd >= lo) & (dd < hi); rows.append(dict(distance=[lo, hi], influence_median=float(np.median(Tn[m])), influence_p95=float(np.quantile(Tn[m], 0.95))))
            out[name] = dict(bonds_alive=len(ids), bonds_broken=nbroken, leverage_mean=float(rho.mean()), leverage_quantiles={q: float(np.quantile(rho, q)) for q in (0.01, 0.1, 0.5, 0.9, 0.99, 0.999)}, leverage_max=float(rho.max()),
                             bonds_with_leverage_above_0p9=int((rho > 0.9).sum()), rank_one_update_error=float(np.abs(pred[keep] - e1[keep]).max() / np.abs(e1).max()), influence_decay=rows)
            print(name, json.dumps(out[name])[:600])
            if name == 'dynamic':
                # leverage from a local solve: only nodes within R of the bond
                acc = []
                for R in (1.5, 2.5, 4, 6, 9):
                    errs = []
                    for c2 in sample[:60]:
                        near = np.flatnonzero(np.linalg.norm(s.P - mid[c2], axis=1) <= R); dof = np.concatenate([2 * near, 2 * near + 1]); g = Ga[c2, dof]
                        errs.append(abs(g @ np.linalg.solve(A[np.ix_(dof, dof)], g) - rho[c2]) / rho[c2])
                    acc.append(dict(radius=R, nodes=int(len(near)), rel_error_median=float(np.median(errs)), rel_error_max=float(np.max(errs))))
                out['local_leverage'] = acc; print('local leverage', acc)
                # does the leverage predict what the next break does? remove each of 40 loaded bonds in turn, measure the growth of the largest overload elsewhere
                o0 = s.over(u)[ids]; cand = np.argsort(-o0)[:40]; rowsp = []
                for c2 in cand:
                    pred = e0 + T[:, c2] * e0[c2] / (1 - rho[c2]); po = pred / s.L[ids] / s.lim[ids]; po[c2] = 0; rowsp.append(dict(leverage=float(rho[c2]), overload_before=float(o0[c2]), max_overload_elsewhere_after=float(po.max()), bonds_pushed_over_limit=int((po > 1).sum())))
                out['removal_consequences'] = rowsp
        json.dump(out, open(outf, 'w'), indent=1)
