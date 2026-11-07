//! Geometry for Phase 0: deterministic seeding, 2D/3D Voronoi by convex clipping,
//! polygon/polyhedron mass properties. f64 throughout; no fixed point here.

pub mod polygon;
pub mod rng;
pub mod rsa;
pub mod vec2;
pub mod vec3;
pub mod voronoi2d;
pub mod voronoi3d;

pub use vec2::V2;
pub use vec3::V3;
