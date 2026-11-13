"""Phase 2a, experiment A: how much memory does an elastic body need behind its ports?

Body: triangular lattice of unit masses joined by unit springs (a beam nx x ny), free in the plane, modal damping zeta.
Ports: a few boundary nodes. Everything unilateral or nonlinear (supports, strikers) stays outside and talks to the
body only through port forces. Seen through the ports the body is the linear map force -> motion
    G(s) = sum_i  phi_i(ports) phi_i(ports)' / (s^2 + 2 zeta w_i s + w_i^2)
Reduced models keep the rigid motions exactly and differ in what they keep of the flexible part:
    static        no internal variable: the whole flexible response is an instantaneous compliance (Guyan)
    low(r)        the r lowest flexible modes, the rest dropped                       (what Phase 1b/1c used)
    low+s(r)      the r lowest flexible modes, the rest as an instantaneous compliance
    hankel+s(r)   the r modes with the largest Hankel singular value seen from the ports, the rest as compliance
The Hankel singular values of the port map bound from below the error of ANY linear model with r internal modes.

  python3 lab/memory.py hankel <out.json>          spectra and their scaling with size, mesh and input bandwidth
  python3 lab/memory.py impacts <out.json>         a sequence of strikers on a supported beam, all models
"""
import sys, json, numpy as np
from scipy.linalg import eigh, solve_continuous_lyapunov, svd, cholesky, expm

def lattice(nx, ny, a=1.0):
    P = np.array([[(i + 0.5 * (j % 2)) * a, j * a * np.sqrt(3) / 2] for j in range(ny) for i in range(nx)]); idx = lambda i, j: j * nx + i; B = []
    for j in range(ny):
        for i in range(nx):
            if i + 1 < nx: B.append((idx(i, j), idx(i + 1, j)))
            if j + 1 < ny:
                B.append((idx(i, j), idx(i, j + 1))); k = i + 1 if j % 2 else i - 1
                if 0 <= k < nx: B.append((idx(i, j), idx(k, j + 1)))
    B = np.array(B); n = len(P); K = np.zeros((2 * n, 2 * n)); dirs = P[B[:, 1]] - P[B[:, 0]]; L = np.linalg.norm(dirs, axis=1); dirs /= L[:, None]
    k = 1.0 / a ** 0          # unit spring per bond; refinement rescales below
    for (i, j), d in zip(B, dirs):
        g = np.concatenate([-d, d]); ii = [2 * i, 2 * i + 1, 2 * j, 2 * j + 1]; K[np.ix_(ii, ii)] += k * np.outer(g, g)
    return P, B, dirs, L, K

class Body:
    def __init__(s, nx, ny, zeta=0.02, a=1.0, mass=1.0, kscale=1.0):
        s.P, s.B, s.dirs, s.L, K = lattice(nx, ny, a); s.n = len(s.P); s.nx, s.ny = nx, ny; s.mass = mass
        w2, Phi = eigh(K * kscale); w2 = np.maximum(w2, 0); s.w = np.sqrt(w2) / np.sqrt(mass); s.Phi = Phi / np.sqrt(mass); s.w[:3] = 0; s.zeta = zeta
        s.nf = 2 * s.n - 3
    def node(s, i, j): return j * s.nx + i
    def port_matrix(s, dofs):                       # rows: modes, columns: port dofs
        return s.Phi[dofs, :].T
    def strain_matrix(s, bonds):                    # rows: modes, columns: bonds (strain = elongation / length)
        out = np.zeros((2 * s.n, len(bonds)))
        for c, b in enumerate(bonds):
            i, j = s.B[b]; d = s.dirs[b]; out[:, c] = (s.Phi[2 * j] * d[0] + s.Phi[2 * j + 1] * d[1] - s.Phi[2 * i] * d[0] - s.Phi[2 * i + 1] * d[1]) / s.L[b]
        return out
    def hankel_modal(s, Bp):
        """per flexible mode, force -> velocity at the ports: sigma_i = |b_i|^2 / (4 zeta w_i) (exact for one isolated mode)."""
        b2 = (Bp[3:] ** 2).sum(1); return b2 / (4 * s.zeta * s.w[3:])
    def hankel_exact(s, Bp, wf=None, nmax=None):
        """Hankel singular values of the flexible part, force -> port velocity, by Gramians.
        wf: corner frequency of a second-order low-pass on every input (band-limited port forces)."""
        m = slice(3, 3 + nmax if nmax else None); w = s.w[m]; b = Bp[m]; nm = len(w); p = b.shape[1]
        A = np.zeros((2 * nm, 2 * nm)); A[:nm, nm:] = np.eye(nm); A[nm:, :nm] = -np.diag(w ** 2); A[nm:, nm:] = -np.diag(2 * s.zeta * w)
        Bm = np.zeros((2 * nm, p)); Bm[nm:] = b; C = np.zeros((p, 2 * nm)); C[:, nm:] = b.T
        if wf:
            Af = np.zeros((2 * p, 2 * p)); Af[:p, p:] = np.eye(p); Af[p:, :p] = -wf ** 2 * np.eye(p); Af[p:, p:] = -np.sqrt(2) * wf * np.eye(p)
            Bf = np.zeros((2 * p, p)); Bf[p:] = wf ** 2 * np.eye(p); Cf = np.zeros((p, 2 * p)); Cf[:, :p] = np.eye(p)
            A2 = np.zeros((2 * nm + 2 * p,) * 2); A2[:2 * nm, :2 * nm] = A; A2[:2 * nm, 2 * nm:] = Bm @ Cf; A2[2 * nm:, 2 * nm:] = Af
            B2 = np.zeros((2 * nm + 2 * p, p)); B2[2 * nm:] = Bf; C2 = np.zeros((p, 2 * nm + 2 * p)); C2[:, :2 * nm] = C; A, Bm, C = A2, B2, C2
        Wc = solve_continuous_lyapunov(A, -Bm @ Bm.T); Wo = solve_continuous_lyapunov(A.T, -C.T @ C)
        ev = np.linalg.eigvals(Wc @ Wo); return np.sort(np.sqrt(np.maximum(ev.real, 0)))[::-1]

def count(sv, eps): return int((sv > eps * sv[0]).sum())

# ---------------- time domain: supported beam, sequence of strikers ----------------
def simulate(body, keep, static, ports, strikers, supports, T, dt, grav=0.0, watch=None, ks=1.0, Tpre=0.0):
    """keep: indices of flexible modes kept (into the list of all modes, >= 3). static: add the instantaneous compliance of the dropped ones.
    strikers: list of (port dof, t0, mass, speed, kc). supports: list of dofs (unilateral springs to the ground, stiffness ks).
    The modes are integrated exactly for forces constant over a step. The compliance of the dropped modes is in series with
    the unilateral springs: the port displacements solve a small piecewise-linear system every step (active-set iteration).
    Tpre: a damped lead-in under gravity so that every model starts from its own static equilibrium."""
    modes = np.concatenate([[0, 1, 2], np.array(keep, int)]) if len(keep) else np.array([0, 1, 2]); w = body.w[modes]; z = body.zeta
    pd = sorted(set([d for d, *_ in strikers] + list(supports))); Bp = body.port_matrix(pd); col = {d: c for c, d in enumerate(pd)}; npd = len(pd)
    bk = Bp[modes]; drop = np.setdiff1d(np.arange(3, 2 * body.n), modes)
    Sres = (Bp[drop].T / body.w[drop] ** 2) @ Bp[drop] if static and len(drop) else np.zeros((npd, npd))
    Wm = watch if watch is not None else np.zeros((2 * body.n, 0)); wk = Wm[modes]; Wres = (Wm[drop].T / body.w[drop] ** 2) @ Bp[drop] if static and len(drop) else np.zeros((Wm.shape[1], npd))
    def props(zz, rigid_damp):
        nm = len(w); E = np.zeros((nm, 2, 2)); F = np.zeros((nm, 2))
        for i, wi in enumerate(w):
            A = np.array([[0, 1], [-wi * wi, -2 * zz * wi - (rigid_damp if wi == 0 else 0)]])
            if wi == 0 and not rigid_damp: E[i] = [[1, dt], [0, 1]]; F[i] = [dt * dt / 2, dt]
            else:
                Ei = expm(A * dt); E[i] = Ei
                if wi == 0: F[i] = [dt * dt / 2, dt]
                else: F[i] = np.linalg.solve(A, (Ei - np.eye(2)) @ np.array([0, 1.0]))
        return E, F
    nm = len(w); q = np.zeros(nm); qd = np.zeros(nm); f = np.zeros(npd); gm = np.zeros(nm); gs = np.zeros(npd); gw = np.zeros(Wm.shape[1])
    if grav:
        gvec = np.zeros(2 * body.n); gvec[1::2] = -grav * body.mass; gall = body.Phi.T @ gvec; gm = gall[modes]
        if static and len(drop): gs = (Bp[drop].T / body.w[drop] ** 2) @ gall[drop]; gw = (Wm[drop].T / body.w[drop] ** 2) @ gall[drop]
    ns = len(strikers); ys = np.zeros(ns); vs = np.array([-s[3] for s in strikers]); live = np.zeros(ns, bool); done = np.zeros(ns, bool); reb = np.full(ns, np.nan)
    sup = np.array([col[d] for d in supports]); hist = dict(t=[], up=[], watch=[]); imp = np.zeros(npd); t = -Tpre
    E, F = props(0.7, 4 * body.w[3]) if Tpre else props(z, 0); pre = Tpre > 0; k = 0
    while t < T - 1e-12:
        if pre and t >= 0: E, F = props(z, 0); pre = False
        a = bk.T @ q + gs                              # port displacement without the compliance of the dropped modes under contact forces
        for i, (d, t0, ms, v0, kc) in enumerate(strikers):
            if not live[i] and not done[i] and t >= t0: live[i] = True; ys[i] = (a + Sres @ f)[col[d]]
        # contact forces: f_c = kact_c (target_c - up_c) on active contacts; up = a + Sres f
        up = a + Sres @ f
        for _ in range(6):
            kact = np.zeros(npd); tgt = np.zeros(npd); act = []
            for c in sup:
                if up[c] < 0: kact[c] += ks
            for i, (d, t0, ms, v0, kc) in enumerate(strikers):
                if live[i] and up[col[d]] > ys[i]: tgt[col[d]] = (tgt[col[d]] * kact[col[d]] + kc * ys[i]) / (kact[col[d]] + kc); kact[col[d]] += kc
            upn = np.linalg.solve(np.eye(npd) + Sres * kact[None, :], a + Sres @ (kact * tgt)) if static else a
            if np.allclose(upn, up, rtol=0, atol=1e-15): up = upn; break
            up = upn
        f = np.zeros(npd)
        for c in sup: f[c] += ks * max(0.0, -up[c])
        for i, (d, t0, ms, v0, kc) in enumerate(strikers):
            if live[i]:
                pen = up[col[d]] - ys[i]; fc = kc * max(0.0, pen); f[col[d]] -= fc; vs[i] += dt * (fc / ms - grav); ys[i] += dt * vs[i]
                if pen < 0 and vs[i] > 0 and t > t0 + 4 * dt: live[i] = False; done[i] = True; reb[i] = vs[i]
        if t >= 0: imp += f * dt
        fm = bk @ f + gm
        q, qd = E[:, 0, 0] * q + E[:, 0, 1] * qd + F[:, 0] * fm, E[:, 1, 0] * q + E[:, 1, 1] * qd + F[:, 1] * fm; t += dt; k += 1
        if t >= 0 and k % 4 == 0: hist['t'].append(t); hist['up'].append(bk.T @ q + gs + Sres @ f); hist['watch'].append(wk.T @ q + gw + Wres @ f)
    return dict(t=np.array(hist['t']), up=np.array(hist['up']), watch=np.array(hist['watch']), rebound=reb, impulse=imp, ports=pd)

if __name__ == '__main__':
    mode, outf = sys.argv[1], sys.argv[2]
    if mode == 'hankel':
        out = {}
        def ports_of(b): nx, ny = b.nx, b.ny; nodes = [b.node(nx // 2, ny - 1), b.node(nx // 4, ny - 1), b.node(2, 0), b.node(nx - 3, 0)]; return [2 * n + c for n in nodes for c in (0, 1)]
        # 1. the reference beam: modal estimate against the exact Gramian computation
        b = Body(48, 6); Bp = b.port_matrix(ports_of(b)); hm = np.sort(b.hankel_modal(Bp))[::-1]; he = b.hankel_exact(Bp)
        T1 = 2 * np.pi / b.w[3]; out['beam_48x6'] = dict(flexible_modes=b.nf, period_first_mode=T1, exact=he[:400:2].tolist(), modal=hm[:200].tolist(), counts_exact={str(e): count(he[::2], e) for e in (0.3, 0.1, 0.03, 0.01)}, counts_modal={str(e): count(hm, e) for e in (0.3, 0.1, 0.03, 0.01)})
        print('48x6: modes above eps*sigma1  exact', out['beam_48x6']['counts_exact'], 'modal', out['beam_48x6']['counts_modal'], 'of', b.nf)
        # 2. band-limited inputs: corner frequency in units of the first flexible frequency
        rows = []
        for mult in (1, 3, 10, 30, 100, None):
            he = b.hankel_exact(Bp, wf=mult * b.w[3] if mult else None); rows.append(dict(corner_over_w1=mult, counts={str(e): count(he[::2], e) for e in (0.3, 0.1, 0.03, 0.01)})); print('band', mult, rows[-1]['counts'])
        out['band_limited'] = rows
        # 3. scaling: mesh refinement at fixed size and fixed physical ports/bandwidth; longer beams at fixed mesh
        rows = []
        for f in (1, 2, 3):
            bb = Body(24 * f, 3 * f, a=1.0 / f, mass=1.0 / f ** 2, kscale=1.0); nodes = [bb.node(bb.nx // 2, bb.ny - 1), bb.node(bb.nx // 4, bb.ny - 1), bb.node(f, 0), bb.node(bb.nx - 1 - f, 0)]
            Bq = bb.port_matrix([2 * n + c for n in nodes for c in (0, 1)]); w1 = bb.w[3]
            for mult in (3, 10, 30):
                he = bb.hankel_exact(Bq, wf=mult * w1); rows.append(dict(refine=f, dof=2 * bb.n, w1=w1, corner_over_w1=mult, counts={str(e): count(he[::2], e) for e in (0.1, 0.03, 0.01)})); print('refine', f, 2 * bb.n, mult, rows[-1]['counts'])
        out['refinement'] = rows; rows = []
        for nx in (24, 48, 96):
            bb = Body(nx, 6); Bq = bb.port_matrix(ports_of(bb)); wfix = 10 * Body(48, 6).w[3] if nx != 48 else 10 * bb.w[3]
            he = bb.hankel_exact(Bq, wf=wfix); rows.append(dict(nx=nx, dof=2 * bb.n, counts={str(e): count(he[::2], e) for e in (0.1, 0.03, 0.01)})); print('length', nx, rows[-1]['counts'])
        out['length_at_fixed_corner'] = rows
        json.dump(out, open(outf, 'w'), indent=1)
    if mode == 'impacts':
        b = Body(48, 6); nx, ny = b.nx, b.ny; top = lambda i: 2 * b.node(i, ny - 1) + 1; sup = [2 * b.node(2, 0) + 1, 2 * b.node(nx - 3, 0) + 1]
        T1 = 2 * np.pi / b.w[3]; Mb = b.n * b.mass
        # decision observable: strain of the horizontal bonds of the bottom row over the middle third
        bonds = [c for c, (i, j) in enumerate(b.B) if i < nx and j < nx and nx // 3 <= i < 2 * nx // 3]; Wm = b.strain_matrix(bonds)
        pd_all = [top(nx // 2), top(nx // 4), top(3 * nx // 4)] + sup; Bp = b.port_matrix(pd_all); hk = b.hankel_modal(Bp); order_h = 3 + np.argsort(-hk); order_l = np.arange(3, 2 * b.n)
        out = dict(T1=T1, dof=2 * b.n, cases={})
        ks = 0.5; nu = len(sys.argv) > 3 and sys.argv[3] == 'quick'
        for tc_rel in ((0.3,) if nu else (1.0, 0.3, 0.1, 0.03)):
            tc = tc_rel * T1; ms = 0.05 * Mb; kc = ms * (np.pi / tc) ** 2; v0 = 1e-3; grav = 0.15 * v0 / T1 * 2 * np.pi
            strikers = [(top(nx // 2), 0.0, ms, v0, kc), (top(nx // 4), 0.45 * T1, ms, v0, kc), (top(nx // 2), 1.3 * T1, ms, 2 * v0, kc), (top(3 * nx // 4), 1.55 * T1, ms, v0, kc), (top(nx // 2), 4.2 * T1, ms, v0, kc), (top(nx // 4), 4.3 * T1, ms, 2 * v0, kc)]
            dt = min(tc / 60, 2 * np.pi / b.w[-1] / 12); T = 7 * T1; kw = dict(ports=None, strikers=strikers, supports=sup, T=T, dt=dt, grav=grav, watch=Wm, ks=ks, Tpre=3 * T1)
            ref = simulate(b, order_l, False, **kw); res = {}; pd = ref['ports']; cm = pd.index(top(nx // 2)); cs = [pd.index(d) for d in sup]
            defl = lambda r: r['up'][:, cm] - r['up'][:, cs].mean(1); d0 = defl(ref)
            def err(r):
                m = min(len(d0), len(r['up'])); dd = defl(r)
                return dict(rebound=float(np.nanmean(np.abs(r['rebound'] - ref['rebound']) / np.abs(ref['rebound']))),
                            rebound_energy=float(abs((r['rebound'] ** 2).sum() - (ref['rebound'] ** 2).sum()) / (ref['rebound'] ** 2).sum()),
                            deflection=float(np.linalg.norm(dd[:m] - d0[:m]) / np.linalg.norm(d0[:m] - d0[0])), peak_deflection=float(abs(np.abs(dd - dd[0]).max() - np.abs(d0 - d0[0]).max()) / np.abs(d0 - d0[0]).max()),
                            support_impulse=float(np.abs(r['impulse'][cs] - ref['impulse'][cs]).sum() / np.abs(ref['impulse'][cs]).sum()),
                            peak_strain=float(abs(r['watch'].max() - ref['watch'].max()) / ref['watch'].max()))
            conv = err(simulate(b, order_l[:400], False, **kw)); res['check_400_of_573_modes'] = {400: conv}
            res['rigid'] = {0: err(simulate(b, [], False, **kw))}; res['static'] = {0: err(simulate(b, [], True, **kw))}
            for name, order, st in (('low', order_l, False), ('low+s', order_l, True), ('hankel+s', order_h, True)):
                res[name] = {}
                for r in (1, 2, 4, 8, 16, 32, 64):
                    res[name][r] = err(simulate(b, order[:r], st, **kw))
            out['cases'][tc_rel] = dict(contact_time_over_T1=tc_rel, rebound_ref=(ref['rebound'] / v0).tolist(), peak_strain_ref=float(ref['watch'].max()), static_strain=float(ref['watch'][0].max()), errors=res)
            print(f'--- contact time {tc_rel} T1; rebounds {np.round(ref["rebound"] / v0, 3)}; peak strain / static strain {ref["watch"].max() / ref["watch"][0].max():.2f}', flush=True)
            for name in res:
                for r in res[name]: e = res[name][r]; print(f'{name:9s} r={r:3d}  ' + '  '.join(f'{k} {v:.3f}' for k, v in e.items()), flush=True)
            json.dump(out, open(outf, 'w'), indent=1)
