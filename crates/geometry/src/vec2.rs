use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub, SubAssign};

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct V2 {
    pub x: f64,
    pub y: f64,
}

impl V2 {
    pub const ZERO: V2 = V2 { x: 0.0, y: 0.0 };

    #[inline]
    pub const fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }
    #[inline]
    pub fn dot(self, o: V2) -> f64 {
        self.x * o.x + self.y * o.y
    }
    /// z-component of the 3D cross product.
    #[inline]
    pub fn cross(self, o: V2) -> f64 {
        self.x * o.y - self.y * o.x
    }
    /// Rotation by +90 degrees (the matrix J = [[0,-1],[1,0]]).
    #[inline]
    pub fn perp(self) -> V2 {
        V2::new(-self.y, self.x)
    }
    #[inline]
    pub fn norm(self) -> f64 {
        self.x.hypot(self.y)
    }
    #[inline]
    pub fn norm2(self) -> f64 {
        self.dot(self)
    }
    #[inline]
    pub fn normalized(self) -> V2 {
        self / self.norm()
    }
    /// Exact rotation by angle `th`.
    #[inline]
    pub fn rot(self, th: f64) -> V2 {
        let (s, c) = th.sin_cos();
        V2::new(c * self.x - s * self.y, s * self.x + c * self.y)
    }
    #[inline]
    pub fn rot_cs(self, c: f64, s: f64) -> V2 {
        V2::new(c * self.x - s * self.y, s * self.x + c * self.y)
    }
    #[inline]
    pub fn dist(self, o: V2) -> f64 {
        (self - o).norm()
    }
}

impl Add for V2 {
    type Output = V2;
    #[inline]
    fn add(self, o: V2) -> V2 {
        V2::new(self.x + o.x, self.y + o.y)
    }
}
impl Sub for V2 {
    type Output = V2;
    #[inline]
    fn sub(self, o: V2) -> V2 {
        V2::new(self.x - o.x, self.y - o.y)
    }
}
impl Mul<f64> for V2 {
    type Output = V2;
    #[inline]
    fn mul(self, s: f64) -> V2 {
        V2::new(self.x * s, self.y * s)
    }
}
impl Mul<V2> for f64 {
    type Output = V2;
    #[inline]
    fn mul(self, v: V2) -> V2 {
        V2::new(self * v.x, self * v.y)
    }
}
impl Div<f64> for V2 {
    type Output = V2;
    #[inline]
    fn div(self, s: f64) -> V2 {
        V2::new(self.x / s, self.y / s)
    }
}
impl Neg for V2 {
    type Output = V2;
    #[inline]
    fn neg(self) -> V2 {
        V2::new(-self.x, -self.y)
    }
}
impl AddAssign for V2 {
    #[inline]
    fn add_assign(&mut self, o: V2) {
        self.x += o.x;
        self.y += o.y;
    }
}
impl SubAssign for V2 {
    #[inline]
    fn sub_assign(&mut self, o: V2) {
        self.x -= o.x;
        self.y -= o.y;
    }
}
