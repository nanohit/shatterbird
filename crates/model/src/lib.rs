//! Phase 0 mechanics model (2D RBSN, two-point corotated facets, law SB-IL1) and the
//! linearised 3D elastic RBSN used for spectra.

pub mod kin;
pub mod law;
pub mod range;
pub mod rbsn3;
pub mod structure;

pub use law::{LawKind, PointLaw, LAW_VERSION};
pub use structure::{BodyKind, BoundaryAttach, FacetForce, LawOptions, Material, Model};
