//! Range log for Q5: min/max |value| per named channel. Thread-local so solvers can
//! record deep inside loops without threading a handle through every call.

use serde::Serialize;
use std::cell::RefCell;
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, Serialize)]
pub struct Range {
    pub min_abs_nonzero: f64,
    pub max_abs: f64,
    pub count: u64,
    pub zeros: u64,
}

impl Range {
    /// log2(max/min): bits of dynamic range needed to hold all observed nonzeros.
    pub fn dyn_bits(&self) -> f64 {
        if self.min_abs_nonzero.is_finite() && self.min_abs_nonzero > 0.0 {
            (self.max_abs / self.min_abs_nonzero).log2()
        } else {
            0.0
        }
    }
}

thread_local! {
    static LOG: RefCell<Option<BTreeMap<String, Range>>> = const { RefCell::new(None) };
}

pub fn enable() {
    LOG.with(|l| *l.borrow_mut() = Some(BTreeMap::new()));
}

pub fn take() -> BTreeMap<String, Range> {
    LOG.with(|l| l.borrow_mut().take().unwrap_or_default())
}

#[inline]
pub fn enabled() -> bool {
    LOG.with(|l| l.borrow().is_some())
}

#[inline]
pub fn rec(name: &str, v: f64) {
    LOG.with(|l| {
        if let Some(m) = l.borrow_mut().as_mut() {
            let a = v.abs();
            let r = m.entry(name.to_string()).or_insert(Range {
                min_abs_nonzero: f64::INFINITY,
                max_abs: 0.0,
                count: 0,
                zeros: 0,
            });
            r.count += 1;
            if a == 0.0 {
                r.zeros += 1;
            } else {
                r.min_abs_nonzero = r.min_abs_nonzero.min(a);
                r.max_abs = r.max_abs.max(a);
            }
        }
    });
}
