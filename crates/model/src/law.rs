//! Interface law SB-IL1 at one facet integration point (docs/00-theory-and-design.md §3.3).
//!
//! Scalar damage driven by the energy-norm equivalent opening
//! δ_eq = sqrt(<δn>+² + γ δt²), linear softening in the opening, secant unloading,
//! undamaged compressive normal branch. With this choice the dissipation to full
//! damage is G_c·A_p for every loading path and the loading tangent is symmetric.

use serde::Serialize;

pub const LAW_VERSION: &str = "SB-IL1/2pt-corot v1";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum LawKind {
    Elastic,
    Damage,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct PointLaw {
    pub kind: LawKind,
    /// Normal and tangential stiffness of this point [N/m].
    pub kn: f64,
    pub kt: f64,
    /// γ = kt/kn (the energy-norm weight).
    pub gamma: f64,
    /// Onset opening σ_eff ℓ/E and final opening 2 G_c/σ_eff [m].
    pub delta0: f64,
    pub wc: f64,
    /// Damage cap (1.0 = exact law). Values < 1 are a named regularisation.
    pub d_max: f64,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct LawEval {
    pub fnorm: f64,
    pub ftan: f64,
    /// Symmetric 2×2 tangent [[dn,dn],[dn,dt],[dt,dt]].
    pub tan: [f64; 3],
    pub psi: f64,
    /// Trial history and damage.
    pub kappa: f64,
    pub d: f64,
    pub loading: bool,
}

impl PointLaw {
    pub fn beta(&self) -> f64 {
        1.0 - self.delta0 / self.wc
    }

    pub fn damage(&self, kappa: f64) -> f64 {
        if self.kind == LawKind::Elastic || kappa <= self.delta0 {
            0.0
        } else if kappa >= self.wc {
            self.d_max
        } else {
            ((kappa - self.delta0) / (kappa * self.beta())).min(self.d_max)
        }
    }

    fn damage_slope(&self, kappa: f64) -> f64 {
        if self.kind == LawKind::Elastic || kappa <= self.delta0 || kappa >= self.wc {
            0.0
        } else {
            let d = (kappa - self.delta0) / (kappa * self.beta());
            if d >= self.d_max {
                0.0
            } else {
                self.delta0 / (self.beta() * kappa * kappa)
            }
        }
    }

    /// Closed-form cumulative dissipation D(κ) of the law [J].
    pub fn dissipation(&self, kappa: f64) -> f64 {
        if self.kind == LawKind::Elastic || kappa <= self.delta0 {
            return 0.0;
        }
        let k = kappa.min(self.wc);
        0.5 * self.kn * self.delta0 * (k - self.delta0) / self.beta()
    }

    /// Dissipation at full separation (= G_c A_p).
    pub fn full_dissipation(&self) -> f64 {
        0.5 * self.kn * self.delta0 * self.wc
    }

    pub fn eval(&self, dn: f64, dt: f64, kappa_c: f64) -> LawEval {
        if self.kind == LawKind::Elastic {
            return LawEval {
                fnorm: self.kn * dn,
                ftan: self.kt * dt,
                tan: [self.kn, 0.0, self.kt],
                psi: 0.5 * (self.kn * dn * dn + self.kt * dt * dt),
                kappa: kappa_c,
                d: 0.0,
                loading: false,
            };
        }
        let dnp = dn.max(0.0);
        let dnm = dn.min(0.0);
        let deq = (dnp * dnp + self.gamma * dt * dt).sqrt();
        let loading = deq > kappa_c;
        let kappa = deq.max(kappa_c);
        let d = self.damage(kappa);
        let om = 1.0 - d;
        let fnorm = om * self.kn * dnp + self.kn * dnm;
        let ftan = om * self.kt * dt;
        let psi = 0.5 * om * (self.kn * dnp * dnp + self.kt * dt * dt) + 0.5 * self.kn * dnm * dnm;
        let mut tan = [if dn > 0.0 { om * self.kn } else { self.kn }, 0.0, om * self.kt];
        if loading {
            let gp = self.damage_slope(kappa);
            if gp > 0.0 && deq > 0.0 {
                let a = self.kn * dnp;
                let b = self.kt * dt;
                let c = gp / (self.kn * deq);
                tan[0] -= c * a * a;
                tan[1] -= c * a * b;
                tan[2] -= c * b * b;
            }
        }
        LawEval { fnorm, ftan, tan, psi, kappa, d, loading }
    }

    /// Trial potential Ψ(δ) within a step (history κ_c frozen): its gradient is the
    /// force of `eval`, so it is the merit function of the incremental problem.
    pub fn potential(&self, dn: f64, dt: f64, kappa_c: f64) -> f64 {
        if self.kind == LawKind::Elastic {
            return 0.5 * (self.kn * dn * dn + self.kt * dt * dt);
        }
        let dnp = dn.max(0.0);
        let dnm = dn.min(0.0);
        let deq = (dnp * dnp + self.gamma * dt * dt).sqrt();
        0.5 * self.kn * dnm * dnm + self.kn * self.f_eq(deq, kappa_c)
    }

    /// F(x) with F' = (1 − g(max(x, κ_c)))·x on loading and the secant below κ_c.
    fn f_eq(&self, x: f64, kc: f64) -> f64 {
        let dc = self.damage(kc);
        if x <= kc {
            return 0.5 * (1.0 - dc) * x * x;
        }
        let base = 0.5 * (1.0 - dc) * kc * kc;
        // h(s) = (1-g(s)) s, piecewise: [0,δ0]: s ; [δ0,s*]: s-(s-δ0)/β ; [s*,∞): (1-d_max) s
        let d0 = self.delta0;
        let beta = self.beta();
        let sstar = d0 / (1.0 - self.d_max * beta);
        let seg = |a: f64, b: f64| -> f64 {
            let mut acc = 0.0;
            let (a1, b1) = (a.max(0.0).min(d0), b.min(d0));
            if b1 > a1 {
                acc += 0.5 * (b1 * b1 - a1 * a1);
            }
            let (a2, b2) = (a.max(d0), b.min(sstar));
            if b2 > a2 {
                let q = 0.5 * (b2 * b2 - a2 * a2);
                acc += q - (q - d0 * (b2 - a2)) / beta;
            }
            let (a3, b3) = (a.max(sstar), b);
            if b3 > a3 {
                acc += (1.0 - self.d_max) * 0.5 * (b3 * b3 - a3 * a3);
            }
            acc
        };
        base + seg(kc, x)
    }

    /// Secant stiffnesses at the trial state (used by S1/S2): (normal, tangential).
    pub fn secant(&self, dn: f64, d: f64) -> (f64, f64) {
        if self.kind == LawKind::Elastic {
            return (self.kn, self.kt);
        }
        let kn = if dn > 0.0 { (1.0 - d) * self.kn } else { self.kn };
        (kn, (1.0 - d) * self.kt)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn law() -> PointLaw {
        PointLaw { kind: LawKind::Damage, kn: 3e8, kt: 3e8, gamma: 1.0, delta0: 1e-5, wc: 6.67e-5, d_max: 1.0 }
    }

    #[test]
    fn dissipation_closed_form_matches_integral() {
        let l = law();
        // ∫ ½ kn κ² d'(κ) dκ by midpoint rule
        let n = 200_000;
        let (a, b) = (l.delta0, l.wc);
        let h = (b - a) / n as f64;
        let mut s = 0.0;
        for i in 0..n {
            let k = a + (i as f64 + 0.5) * h;
            s += 0.5 * l.kn * k * k * l.damage_slope(k) * h;
        }
        assert!((s - l.full_dissipation()).abs() / l.full_dissipation() < 1e-8, "{s}");
        assert!((l.dissipation(l.wc) - l.full_dissipation()).abs() < 1e-12 * l.full_dissipation());
    }

    #[test]
    fn tangent_is_derivative() {
        let l = law();
        for &(dn, dt, kc) in &[(2e-5, 1e-5, 1.5e-5), (3e-5, -2e-5, 0.0), (-1e-5, 2e-5, 0.0), (5e-6, 1e-6, 0.0)] {
            let e = l.eval(dn, dt, kc);
            let h = 1e-11;
            let ep = l.eval(dn + h, dt, kc);
            let em = l.eval(dn - h, dt, kc);
            let tp = l.eval(dn, dt + h, kc);
            let tm = l.eval(dn, dt - h, kc);
            let dfn_dn = (ep.fnorm - em.fnorm) / (2.0 * h);
            let dfn_dt = (tp.fnorm - tm.fnorm) / (2.0 * h);
            let dft_dn = (ep.ftan - em.ftan) / (2.0 * h);
            let dft_dt = (tp.ftan - tm.ftan) / (2.0 * h);
            let sc = l.kn;
            assert!((dfn_dn - e.tan[0]).abs() < 1e-5 * sc, "{dfn_dn} {}", e.tan[0]);
            assert!((dfn_dt - e.tan[1]).abs() < 1e-5 * sc);
            assert!((dft_dn - e.tan[1]).abs() < 1e-5 * sc);
            assert!((dft_dt - e.tan[2]).abs() < 1e-5 * sc);
            // forces are the gradient of the trial potential
            let pdn = (l.potential(dn + h, dt, kc) - l.potential(dn - h, dt, kc)) / (2.0 * h);
            let pdt = (l.potential(dn, dt + h, kc) - l.potential(dn, dt - h, kc)) / (2.0 * h);
            assert!((pdn - e.fnorm).abs() < 1e-6 * (e.fnorm.abs() + 1.0), "{pdn} {}", e.fnorm);
            assert!((pdt - e.ftan).abs() < 1e-6 * (e.ftan.abs() + 1.0), "{pdt} {}", e.ftan);
        }
    }
}
