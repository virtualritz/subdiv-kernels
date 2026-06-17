/// Rule for sharp/crease/corner vertices. Currently only the OpenSubdiv/DeRose
/// rule-transition behavior is available.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum CornerRule {
    /// OpenSubdiv / DeRose rule-transition behavior.
    #[default]
    OpenSubdivDeRose,
}

/// Edge sharpness propagation policy.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum CreaseComputationMethod {
    /// Integer decrement style.
    #[default]
    Uniform,

    /// Chaikin-inspired smoothing style.
    Chaikin,
}

/// Triangle handling mode for Catmull-Clark on triangle faces.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum TriangleSubdivisionRule {
    /// Catmull-Clark triangle rule.
    #[default]
    CatmullClark,

    /// Smooth-triangle variant.
    SmoothTriangles,
}

/// Boundary interpolation policy for positional evaluation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum BoundaryInterpolation {
    /// No boundary-specific handling.
    Natural,

    /// Interpolate boundary edges only.
    #[default]
    EdgesOnly,

    /// Interpolate boundary edges and pin corners.
    EdgesAndCorners,
}

/// Face-varying (per-corner, seam-capable) interpolation policy, mirroring
/// OpenSubdiv's `Sdc::Options::FVarLinearInterpolation` spectrum from fully
/// linear to fully smooth.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum FaceVaryingInterpolation {
    /// Linear interpolation everywhere (OSD `FVAR_LINEAR_ALL`); the old
    /// RenderMan `facevarying` class.
    Linear,

    /// Smooth interior, linear only at face-varying corners
    /// (OSD `FVAR_LINEAR_CORNERS_ONLY`).
    SmoothWithLinearCorners,

    /// Smooth interior, linear along face-varying boundaries/seams
    /// (OSD `FVAR_LINEAR_BOUNDARIES`). The common default for UV channels:
    /// smooth within an island, pinned at island seams.
    #[default]
    SmoothWithLinearBoundaries,

    /// Smooth subdivision rules everywhere, with seams as smooth boundary
    /// curves (OSD `FVAR_LINEAR_NONE`); the old RenderMan `facevertex` class.
    Smooth,
}

// ── New API types ──────────────────────────────────────────────────────

/// Which subdivision scheme to apply.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Scheme {
    /// Catmull–Clark: quad-based, the film/VFX standard. Accepts any polygons;
    /// every refined face is a quad.
    CatmullClark,
    /// Loop: for triangle meshes. Triangles in, triangles out.
    Loop,
    /// √3 (Kobbelt): for triangle meshes. Adds fewer triangles per step than
    /// Loop and reorients them each level.
    Sqrt3,
    /// Doo–Sabin: corner-cutting; produces a new face around each original
    /// vertex, edge, and face.
    DooSabin,
}

/// Scheme-level options that define subdivision behavior.
///
/// These are set once when creating a [`Refiner`](crate::Refiner) and
/// apply to all refinement calls on that refiner.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SchemeOptions {
    /// Sharp-vertex rule selection.
    pub corner_rule: CornerRule,

    /// Crease propagation mode.
    pub crease_computation: CreaseComputationMethod,

    /// Positional boundary mode.
    pub boundary_interpolation: BoundaryInterpolation,

    /// Triangle handling mode (CC only).
    pub triangle_subdivision_rule: TriangleSubdivisionRule,

    /// Whether to promote vertices with 3+ sharp incident edges.
    pub auto_corner: bool,

    /// Whether crease values are normalized per level.
    pub crease_normalize: bool,

    /// Whether corner values are normalized per level.
    pub corner_normalize: bool,
}

impl Default for SchemeOptions {
    fn default() -> Self {
        Self {
            corner_rule: CornerRule::default(),
            crease_computation: CreaseComputationMethod::default(),
            boundary_interpolation: BoundaryInterpolation::default(),
            triangle_subdivision_rule: TriangleSubdivisionRule::default(),
            auto_corner: false,
            crease_normalize: false,
            corner_normalize: false,
        }
    }
}

use core::num::NonZeroU8;

/// Per-call refinement parameters.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct UniformRefine {
    /// Number of uniform refinement levels. Must be at least 1.
    pub levels: NonZeroU8,

    /// Optional face selection mask. When `Some`, only selected faces are refined.
    pub selected_faces: Option<Vec<bool>>,

    /// Crease weight applied at selection boundaries.
    pub selection_boundary_crease: f32,

    /// Track the refined vertices lying along each input edge.
    ///
    /// When enabled, the refinement result fills in its `edge_polylines`:
    /// for every input edge, the refined vertex indices on it, in order.
    pub edge_polylines: bool,
}

impl Default for UniformRefine {
    fn default() -> Self {
        Self {
            // SAFETY: 1 is non-zero.
            levels: NonZeroU8::new(1).unwrap(),
            selected_faces: None,
            selection_boundary_crease: 0.0,
            edge_polylines: false,
        }
    }
}

impl From<NonZeroU8> for UniformRefine {
    fn from(levels: NonZeroU8) -> Self {
        Self {
            levels,
            ..Self::default()
        }
    }
}
