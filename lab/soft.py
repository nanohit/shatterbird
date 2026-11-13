"""Phase 2a lab: equal spheres with rotation and a declared compliant contact (float64, numpy).

Contact law (pair i, j; n from j to i; overlap d = D - |xi - xj|):
  normal      fn = max(0, kn d - cn vn),          vn = n . vrel  (separation speed)
  tangential  ft = clamp(-kt xi - ct vt, mu fn),  xi = accumulated tangential slip of the contact (state)
  vrel = vi - vj + R n x (wi + wj)   (velocity of i relative to j at the contact point)
  rolling     torque tr = clamp(-kr th - cr wr, mur R fn) on i (opposite on j), wr = wi - wj, th = accumulated
              relative rotation of the contact (state). A spring with a slider, like the tangential law: a purely
              viscous resistance has no static branch and the pile creeps for ever; without any resistance a
              sphere on a flat floor never comes to rest. mur = 0 switches it off.
One friction coefficient mu, one rolling coefficient mur. Walls are the same law against a fixed plane.

Two integrators of the same law:
  explicit   semi-implicit Euler with step dt (the reference as dt -> 0)
  implicit   backward Euler in the velocities over a step h with the contact set and normals frozen:
             minimise  Phi(u) = 1/2 (u-u*)' M (u-u*) + h sum_c psi_c(J u)   (convex for frozen friction bounds)
             solved by a majorised block-Jacobi iteration (6x6 per body), optionally Chebyshev accelerated
             over the certified interval [lam_lo, 1] of the preconditioned Hessian.
Certificate: 1/2 |u - u_opt|_M^2 <= 1/2 g' M^-1 g  (strong convexity of Phi with respect to M).
"""
import numpy as np
from scipy.spatial import cKDTree

D = 0.0942; R = D / 2; G = 9.81; H = 1 / 480

def cross(a, b):
    a, b = np.broadcast_arrays(a, b); o = np.empty(a.shape)
    o[..., 0] = a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1]; o[..., 1] = a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2]; o[..., 2] = a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
    return o

class World:
    def __init__(s, x, box, ell=2.0, zeta=0.5, mu=0.5, kt_ratio=2 / 7, h=H, m=1.0, grav=(0, -G, 0), mur=0.1):
        s.x = np.array(x, float); s.N = len(s.x); s.v = np.zeros((s.N, 3)); s.w = np.zeros((s.N, 3))
        s.box = np.array(box, float)          # x, y, z extents; floor y=0, walls x=0, x=box0, z=0, z=box2 (inf = absent)
        s.m = m; s.I = 0.4 * m * R * R; s.mu = mu; s.mur = mur; s.grav = np.array(grav, float); s.h = h
        s.set_stiffness(ell, zeta, kt_ratio)
        s.skin = 0.15 * D; s.pi = np.zeros(0, int); s.pj = np.zeros(0, int); s.xi = np.zeros((0, 3)); s.th = np.zeros((0, 3)); s.xref = None
        s.wn = np.array([[0, 1, 0], [1, 0, 0], [-1, 0, 0], [0, 0, 1], [0, 0, -1]], float)   # wall normals (into the box)
        s.wxi = np.zeros((s.N, 5, 3)); s.wth = np.zeros((s.N, 5, 3)); s.t = 0.0; s.alive = np.ones(s.N, bool); s.evals = 0
    def set_stiffness(s, ell, zeta, kt_ratio=2 / 7):
        s.ell = ell; s.kn = s.m * (ell / s.h) ** 2; s.kt = kt_ratio * s.kn; s.zeta = zeta
        s.cn = 2 * zeta * np.sqrt(s.kn * s.m / 2); s.ct = 2 * zeta * np.sqrt(s.kt * s.m / 7); s.cr = s.ct * R * R; s.kr = s.kt * R * R
    # ---- neighbour list with a skin; tangential slips are carried by pair key ----
    def pairs(s, force=False):
        if not force and s.xref is not None and np.abs(s.x - s.xref).max() < s.skin / 2: return
        tr = cKDTree(s.x); p = tr.query_pairs(D + s.skin, output_type='ndarray')
        if len(p): p = p[s.alive[p[:, 0]] & s.alive[p[:, 1]]]
        key = p[:, 0] * s.N + p[:, 1] if len(p) else np.zeros(0, int); o = np.argsort(key); p = p[o]; key = key[o]
        old = s.pi * s.N + s.pj; xi = np.zeros((len(p), 3)); th = np.zeros((len(p), 3))
        if len(old) and len(key):
            q = np.searchsorted(old, key); q[q >= len(old)] = len(old) - 1; hit = old[q] == key; xi[hit] = s.xi[q[hit]]; th[hit] = s.th[q[hit]]
        s.pi, s.pj = (p[:, 0], p[:, 1]) if len(p) else (np.zeros(0, int), np.zeros(0, int)); s.xi = xi; s.th = th; s.xref = s.x.copy()
    def geometry(s):
        e = s.x[s.pi] - s.x[s.pj]; L = np.linalg.norm(e, axis=1); n = e / L[:, None]; d = D - L
        wd = np.stack([R - s.x[:, 1], R - s.x[:, 0], R - (s.box[0] - s.x[:, 0]), R - s.x[:, 2], R - (s.box[2] - s.x[:, 2])], 1)   # N x 5 overlaps
        wd[~s.alive] = -1e9
        return n, d, wd
    # ---- the contact law. dts: time over which slip accumulates within the evaluation. Returns forces and new slips ----
    def law(s, n, d0, wd0, xi0, wxi0, th0, wth0, v, w, dts, implicit):
        vr = v[s.pi] - v[s.pj] + R * cross(n, w[s.pi] + w[s.pj]); vn = (vr * n).sum(1); vt = vr - vn[:, None] * n
        d = d0 - dts * vn if implicit else d0
        fn = np.maximum(0.0, s.kn * d - s.cn * vn);
        if not implicit: fn[d0 <= 0] = 0
        xi = xi0 - (xi0 * n).sum(1)[:, None] * n + dts * vt
        tt = -s.kt * xi - s.ct * vt; tn = np.linalg.norm(tt, axis=1); lim = s.mu * fn; sl = tn > lim
        sc = np.ones(len(tn)); sc[sl] = lim[sl] / np.maximum(tn[sl], 1e-300); ft = tt * sc[:, None]
        xin = xi.copy(); xin[sl] = -(ft[sl] + s.ct * vt[sl]) / s.kt; xin[fn <= 0] = 0
        F = fn[:, None] * n + ft
        # walls
        wv = v[:, None, :] + R * cross(s.wn[None], w[:, None, :]); wvn = (wv * s.wn).sum(2); wvt = wv - wvn[..., None] * s.wn
        wdd = wd0 - dts * wvn if implicit else wd0
        wfn = np.maximum(0.0, s.kn * wdd - s.cn * wvn)
        if not implicit: wfn[wd0 <= 0] = 0
        wxi = wxi0 - (wxi0 * s.wn).sum(2)[..., None] * s.wn + dts * wvt
        wtt = -s.kt * wxi - s.ct * wvt; wtn = np.linalg.norm(wtt, axis=2); wlim = s.mu * wfn; wsl = wtn > wlim
        wsc = np.ones(wtn.shape); wsc[wsl] = wlim[wsl] / np.maximum(wtn[wsl], 1e-300); wft = wtt * wsc[..., None]
        wxin = wxi.copy(); wxin[wsl] = -(wft[wsl] + s.ct * wvt[wsl]) / s.kt; wxin[wfn <= 0] = 0
        WF = wfn[..., None] * s.wn + wft
        tq = -R * cross(n, F); N = s.N
        Fb = np.stack([np.bincount(s.pi, F[:, k], N) - np.bincount(s.pj, F[:, k], N) for k in range(3)], 1)
        Tb = np.stack([np.bincount(s.pi, tq[:, k], N) + np.bincount(s.pj, tq[:, k], N) for k in range(3)], 1)
        Fb += WF.sum(1); Tb += (-R * cross(s.wn[None], WF)).sum(1)
        thn, wthn = th0, wth0
        if s.mur > 0:
            wr = w[s.pi] - w[s.pj]; th = th0 + dts * wr; tt = -s.kr * th - s.cr * wr; tm = np.linalg.norm(tt, axis=1); lim = s.mur * R * fn; sl2 = tm > lim
            sc = np.ones(len(tm)); sc[sl2] = lim[sl2] / np.maximum(tm[sl2], 1e-300); tr = tt * sc[:, None]
            thn = th.copy(); thn[sl2] = -(tr[sl2] + s.cr * wr[sl2]) / s.kr; thn[fn <= 0] = 0
            Tb += np.stack([np.bincount(s.pi, tr[:, k], N) - np.bincount(s.pj, tr[:, k], N) for k in range(3)], 1)
            wr = np.broadcast_to(w[:, None, :], (N, 5, 3)); th = wth0 + dts * wr; tt = -s.kr * th - s.cr * wr; tm = np.linalg.norm(tt, axis=2); lim = s.mur * R * wfn; sl2 = tm > lim
            sc = np.ones(tm.shape); sc[sl2] = lim[sl2] / np.maximum(tm[sl2], 1e-300); tr = tt * sc[..., None]
            wthn = th.copy(); wthn[sl2] = -(tr[sl2] + s.cr * wr[sl2]) / s.kr; wthn[wfn <= 0] = 0
            Tb += tr.sum(1)
        s.evals += 1
        return Fb, Tb, xin, wxin, thn, wthn, (fn, ft, sl, wfn, wft, wsl)
    def step_explicit(s, dt):
        s.pairs(); n, d, wd = s.geometry()
        Fb, Tb, s.xi, s.wxi, s.th, s.wth, s.last = s.law(n, d, wd, s.xi, s.wxi, s.th, s.wth, s.v, s.w, dt, False)
        a = s.alive[:, None]
        s.v += dt * (Fb / s.m + s.grav) * a; s.w += dt * Tb / s.I * a; s.x += dt * s.v * a; s.t += dt
    # ---- implicit step ----
    def precond(s, n, d0, wd0, margin):
        gn = s.h * (s.h * s.kn + s.cn); gt = s.h * (s.h * s.kt + s.ct); gr = s.h * (s.h * s.kr + s.cr)
        act = d0 > -margin; wact = wd0 > -margin
        def blocks(nn):      # per contact: Gm (3x3), B = R [n]x
            P = nn[:, :, None] * nn[:, None, :]; Gm = gn * P + gt * (np.eye(3)[None] - P)
            B = np.zeros((len(nn), 3, 3)); B[:, 0, 1] = -nn[:, 2]; B[:, 0, 2] = nn[:, 1]; B[:, 1, 0] = nn[:, 2]; B[:, 1, 2] = -nn[:, 0]; B[:, 2, 0] = -nn[:, 1]; B[:, 2, 1] = nn[:, 0]
            return Gm, R * B
        A = np.zeros((s.N, 6, 6)); A[:, :3, :3] = s.m * np.eye(3); A[:, 3:, 3:] = s.I * np.eye(3)
        Gm, B = blocks(n); Gm = Gm * act[:, None, None]; GB = Gm @ B; BGB = B.transpose(0, 2, 1) @ GB
        for idx, sg in ((s.pi, 1.0), (s.pj, -1.0)):
            np.add.at(A[:, :3, :3], idx, 2 * Gm); np.add.at(A[:, :3, 3:], idx, 2 * sg * GB); np.add.at(A[:, 3:, :3], idx, 2 * sg * GB.transpose(0, 2, 1)); np.add.at(A[:, 3:, 3:], idx, 2 * BGB)
            if s.mur > 0: np.add.at(A[:, 3:, 3:], idx, (2 * gr * act)[:, None, None] * np.eye(3)[None])
        for k in range(5):
            Gw, Bw = blocks(np.repeat(s.wn[k][None], s.N, 0)); Gw = Gw * wact[:, k][:, None, None]; GBw = Gw @ Bw
            A[:, :3, :3] += Gw; A[:, :3, 3:] += GBw; A[:, 3:, :3] += GBw.transpose(0, 2, 1); A[:, 3:, 3:] += Bw.transpose(0, 2, 1) @ GBw   # a wall is one-sided: no factor 2
            if s.mur > 0: A[:, 3:, 3:] += (gr * wact[:, k])[:, None, None] * np.eye(3)[None]
        Ai = np.linalg.inv(A)
        # certified lower bound of the preconditioned Hessian: H >= M, so lam >= min_i lam_min(A_i^-1 M_i)
        Mh = np.array([s.m] * 3 + [s.I] * 3) ** 0.5; lam = 1.0 / np.linalg.eigvalsh(A / Mh[None, :, None] / Mh[None, None, :])[:, -1]
        return Ai, lam
    def grad(s, n, d0, wd0, u, ustar):
        v, w = u[:, :3], u[:, 3:]; Fb, Tb, xin, wxin, thn, wthn, det = s.law(n, d0, wd0, s.xi, s.wxi, s.th, s.wth, v, w, s.h, True)
        g = np.empty_like(u); g[:, :3] = s.m * (v - ustar[:, :3]) - s.h * Fb; g[:, 3:] = s.I * (w - ustar[:, 3:]) - s.h * Tb
        g[~s.alive] = 0
        return g, (xin, wxin, thn, wthn), det
    def cert(s, g): return 0.5 * ((g[:, :3] ** 2).sum() / s.m + (g[:, 3:] ** 2).sum() / s.I)       # energy bound, J
    def step_midpoint(s, **kw):
        """implicit midpoint: the backward-Euler problem over h/2 gives the mid-step velocity; the end velocity is its reflection.
        No numerical damping of the contact oscillation (backward Euler removes the rebound of an impact at ell >= 1)."""
        s.pairs(); h = s.h; v0 = s.v.copy(); w0 = s.w.copy(); x0 = s.x.copy(); t0 = s.t; st0 = (s.xi, s.wxi, s.th, s.wth)
        s.h = h / 2; r = s.step_implicit(**kw); um, wm = s.v.copy(), s.w.copy(); s.h = h
        # slips over the whole step at the mid-step velocity
        s.x = x0; s.xi, s.wxi, s.th, s.wth = st0; n, d0, wd0 = s.geometry()
        _, _, s.xi, s.wxi, s.th, s.wth, s.last = s.law(n, d0, wd0, s.xi, s.wxi, s.th, s.wth, um, wm, h, True)
        a = s.alive; s.v[a] = 2 * um[a] - v0[a]; s.w[a] = 2 * wm[a] - w0[a]; s.x[a] = x0[a] + h * um[a]; s.t = t0 + h
        return r
    def step_implicit(s, iters=16, method='cheb', tol=0.0, warm=True, active=None, log=None):
        """one step h. method: 'jacobi' | 'cheb' | 'nest' (Nesterov momentum with a per-body gradient restart). tol: stop when the certificate (J) is below tol.
        active: boolean mask of bodies that are solved (others keep their current velocity) - domain restriction."""
        s.pairs(); n, d0, wd0 = s.geometry(); h = s.h
        vmax = np.abs(s.v).max() + G * h; margin = 2 * h * vmax + 1e-4 * D
        Ai, lam = s.precond(n, d0, wd0, margin); lam_lo = lam[s.alive].min()
        ustar = np.hstack([s.v + h * s.grav, s.w]); u = np.hstack([s.v, s.w]) if warm else ustar.copy()
        if active is not None: u[~active] = np.hstack([s.v, s.w])[~active]
        lam_lo = min(lam_lo, 0.999); th = (1 + lam_lo) / 2; de = (1 - lam_lo) / 2; sig = th / de; rho = 1 / sig; dstep = None; k = 0; c = None
        sq = np.sqrt(lam_lo); beta = (1 - sq) / (1 + sq); up = u.copy(); restarts = 0
        while True:
            # 'nest': gradient at the extrapolated point y = u + beta (u - up); the momentum of a body is dropped when it points uphill
            y = u + beta * (u - up) if method == 'nest' and k > 0 else u
            g, st, det = s.grad(n, d0, wd0, y, ustar)
            if active is not None: g[~active] = 0
            c = s.cert(g)
            if log is not None: log.append(c)
            if k >= iters or c <= tol: u = y; break
            z = -np.einsum('nij,nj->ni', Ai, g)
            if method == 'nest':
                un = y + z; bad = (g * (un - u)).sum(1) > 0; restarts += int(bad.sum())
                up = u.copy(); up[bad] = un[bad]; u = un; k += 1; continue
            if method == 'jacobi': dstep = z
            elif k == 0: dstep = z / th
            else: rho_n = 1 / (2 * sig - rho); dstep = rho_n * rho * dstep + 2 * rho_n / de * z; rho = rho_n
            u = u + dstep; k += 1
        a = s.alive
        s.v[a] = u[a, :3]; s.w[a] = u[a, 3:]; s.xi, s.wxi, s.th, s.wth = st; s.x[a] += h * s.v[a]; s.t += h; s.last = det
        return dict(iters=k, cert=c, lam_lo=lam_lo)
    # ---- observables ----
    def energy(s):
        a = s.alive; return 0.5 * s.m * (s.v[a] ** 2).sum() + 0.5 * s.I * (s.w[a] ** 2).sum()
    def copy(s):
        import copy; return copy.deepcopy(s)

def rain(nx, nz, layers, seed=1, pitch=1.3):
    """loose jittered lattice of spheres above the floor (disordered after settling). Box of nx x nz diameters."""
    r = np.random.default_rng(seed); pts = []; mx = int(nx / pitch); mz = int(nz / pitch)
    for l in range(layers):
        for i in range(mx):
            for k in range(mz):
                pts.append([(i + 0.5) * nx * D / mx, (l + 0.6) * 1.1 * D, (k + 0.5) * nz * D / mz])
    p = np.array(pts); j = (pitch - 1.02) / 2 * D; p[:, [0, 2]] += (r.random((len(p), 2)) - 0.5) * 2 * j; p[:, 1] += (r.random(len(p)) - 0.5) * 0.08 * D
    return p, (nx * D, 1e9, nz * D)

def settle(W, dt, tmax=6.0, ke_tol=1e-10, check=200):
    k = 0
    while W.t < tmax:
        W.step_explicit(dt); k += 1
        if k % check == 0 and W.energy() / W.N < ke_tol: break
    return W
