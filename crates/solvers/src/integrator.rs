//! Variational time integrators (docs §4.4): each step minimises
//! α/(2h²)‖q − q̂‖²_M + Π(q). Velocities/accelerations are integrator state.

use crate::assembly::Inertia;
use model::Model;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scheme {
    ImplicitEuler,
    Bdf2,
    /// Newmark average acceleration (β = 1/4, γ = 1/2).
    Newmark,
}

impl Scheme {
    pub fn name(&self) -> &'static str {
        match self {
            Scheme::ImplicitEuler => "IE",
            Scheme::Bdf2 => "BDF2",
            Scheme::Newmark => "Newmark",
        }
    }
}

#[derive(Clone, Debug)]
pub struct Dyn {
    pub scheme: Scheme,
    pub h: f64,
    pub q_prev: Vec<[f64; 3]>,
    pub v: Vec<[f64; 3]>,
    pub v_prev: Vec<[f64; 3]>,
    pub a: Vec<[f64; 3]>,
    pub steps: u64,
}

impl Dyn {
    pub fn new(scheme: Scheme, h: f64, m: &Model) -> Self {
        let n = m.bodies.len();
        Dyn { scheme, h, q_prev: m.q.clone(), v: vec![[0.0; 3]; n], v_prev: vec![[0.0; 3]; n], a: vec![[0.0; 3]; n], steps: 0 }
    }

    fn effective(&self) -> Scheme {
        if self.scheme == Scheme::Bdf2 && self.steps == 0 {
            Scheme::ImplicitEuler
        } else {
            self.scheme
        }
    }

    /// Inertial term for the step starting at q (current configuration).
    pub fn inertia(&self, q: &[[f64; 3]]) -> Inertia {
        let h = self.h;
        let sc = self.effective();
        let qhat = (0..q.len())
            .map(|i| {
                let mut r = [0.0; 3];
                for k in 0..3 {
                    r[k] = match sc {
                        Scheme::ImplicitEuler => q[i][k] + h * self.v[i][k],
                        Scheme::Bdf2 => {
                            4.0 / 3.0 * q[i][k] - 1.0 / 3.0 * self.q_prev[i][k]
                                + 2.0 * h / 3.0 * (4.0 / 3.0 * self.v[i][k] - 1.0 / 3.0 * self.v_prev[i][k])
                        }
                        Scheme::Newmark => q[i][k] + h * self.v[i][k] + 0.25 * h * h * self.a[i][k],
                    };
                }
                r
            })
            .collect();
        let alpha = match sc {
            Scheme::ImplicitEuler => 1.0,
            Scheme::Bdf2 => 9.0 / 4.0,
            Scheme::Newmark => 4.0,
        };
        Inertia { alpha_h2: alpha / (h * h), qhat }
    }

    /// Acceleration implied by a candidate end-of-step state (used by S2).
    pub fn accel_of(&self, q_old: &[[f64; 3]], q_new: &[[f64; 3]]) -> Vec<[f64; 3]> {
        let ine = self.inertia(q_old);
        (0..q_new.len())
            .map(|i| {
                let mut a = [0.0; 3];
                for k in 0..3 {
                    a[k] = ine.alpha_h2 * (q_new[i][k] - ine.qhat[i][k]);
                }
                a
            })
            .collect()
    }

    /// Advance the integrator state after accepting q_new.
    pub fn finish(&mut self, q_old: &[[f64; 3]], q_new: &[[f64; 3]]) {
        let h = self.h;
        let sc = self.effective();
        let a_new = self.accel_of(q_old, q_new);
        let n = q_new.len();
        let mut v_new = vec![[0.0; 3]; n];
        for i in 0..n {
            for k in 0..3 {
                v_new[i][k] = match sc {
                    Scheme::ImplicitEuler => (q_new[i][k] - q_old[i][k]) / h,
                    Scheme::Bdf2 => 1.5 / h * (q_new[i][k] - 4.0 / 3.0 * q_old[i][k] + 1.0 / 3.0 * self.q_prev[i][k]),
                    Scheme::Newmark => self.v[i][k] + 0.5 * h * (self.a[i][k] + a_new[i][k]),
                };
            }
        }
        self.v_prev = std::mem::replace(&mut self.v, v_new);
        self.a = a_new;
        self.q_prev = q_old.to_vec();
        self.steps += 1;
    }

    pub fn kinetic(&self, m: &Model) -> f64 {
        let mut t = 0.0;
        for (i, b) in m.bodies.iter().enumerate() {
            t += 0.5 * b.mass * (self.v[i][0].powi(2) + self.v[i][1].powi(2)) + 0.5 * b.inertia * self.v[i][2].powi(2);
        }
        t
    }
}
