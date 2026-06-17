//! Bicubic B-spline patches over regular Catmull-Clark quads.
//!
//! After at least one uniform refinement every Catmull-Clark face is a
//! quad. [`RefinementResult::patch_table`] classifies each refined quad
//! ([`QuadClass`]) and extracts one uniform bicubic B-spline patch per
//! *regular* quad: all four corner vertices interior, valence 4, with
//! no vertex sharpness and no sharp incident edge at the refined level
//! (sharp per the rule selection of the [`LimitStencils`] docs: stored
//! crease value `> 0.0` or boundary). Everything else --
//! extraordinary vertices, creases, boundaries -- is
//! [`QuadClass::Feature`] and left to the feature-patch machinery
//! (limit-surface SDF design s2).
//!
//! # Exactness contract
//!
//! Over a regular quad the patch evaluates the *exact* limit surface:
//! its 16 control points are the quad's 4x4 refined-vertex neighborhood
//! (the corners plus their one-rings), and under the regularity
//! conditions every subdivision rule that influences the quad's nested
//! neighborhoods -- the corner vertex rules, the edge rules of the
//! corners' incident edges, and the (sharpness-free) face rules -- is
//! the regular B-spline rule, so Catmull-Clark refinement under the
//! quad coincides with B-spline knot insertion and converges to the
//! B-spline surface. Ring vertices may themselves be boundary, crease,
//! or extraordinary vertices; only their *positions* enter the patch.
//!
//! # Parameterization and control-point layout
//!
//! A patch covers its quad's `[0, 1]^2`: CSR corner `k` of the quad
//! (the order of [`Mesh::face_vertex_indices`]) sits at
//! `(u, v) = (0, 0), (1, 0), (1, 1), (0, 1)` for `k = 0, 1, 2, 3` --
//! `u` runs along the corner-0 -> corner-1 edge and `v` along
//! corner-0 -> corner-3. Control points are row-major in `v` then `u`:
//! entry `4 * j + i` sits at grid position `(i, j)` with `i` along `u`
//! and `j` along `v`, the quad's own corners occupying the interior
//! positions `(1, 1)`, `(2, 1)`, `(2, 2)`, `(1, 2)`.
//!
//! Derivatives from [`PatchTable::eval_with_derivatives`] are with
//! respect to this in-quad parameterization, so against a parent
//! (ptex-style) unit parameterization of the root face they carry an
//! extra factor of `2^level`.
//!
//! Control points are *indices* into the refined vertex order of
//! [`RefinementResult::topology`]; evaluation gathers from whatever
//! positions buffer the caller supplies (CPU-interpolated or read back
//! from the GPU stencil path), so a patch table is built once per
//! topology and re-evaluated across edits.
//!
//! [`LimitStencils`]: crate::LimitStencils

use crate::limit::{validate_refined_quads, vertex_ring};
use crate::{Adjacency, KernelError, Mesh, RefinementResult};

/// Classification of one refined quad at the evaluated level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QuadClass {
    /// All four corners interior, valence-4, sharpness-free: covered
    /// exactly by a bicubic B-spline patch.
    Regular,
    /// Touches an extraordinary vertex, crease, corner, or boundary;
    /// no patch in the table (feature evaluation is s2's job).
    Feature,
}

/// Bicubic B-spline patches over the regular quads of a refined level.
///
/// Built by [`RefinementResult::patch_table`]; see the module docs for
/// the exactness contract and the control-point layout.
#[derive(Debug, Clone, PartialEq)]
pub struct PatchTable {
    /// 16 refined-vertex indices per patch, row-major (v then u), one
    /// entry per regular refined quad; feature quads get no patch here.
    pub control_points: Vec<[u32; 16]>,
    /// The refined face each patch covers (index into face CSR).
    pub faces: Vec<u32>,
    /// Per refined face: its patch index, `u32::MAX` for feature quads.
    face_patches: Vec<u32>,
}

impl RefinementResult {
    /// Classify every refined quad and extract the bicubic B-spline
    /// patches of the regular ones.
    ///
    /// Catmull-Clark only, and the result must come from at least one
    /// full (unselected) refinement -- the same gating as
    /// [`limit_stencils`](Self::limit_stencils), except that open
    /// meshes are accepted under every boundary rule (boundary quads
    /// are always [`QuadClass::Feature`]).
    pub fn patch_table(&self) -> Result<PatchTable, KernelError> {
        build_patch_table(self)
    }
}

impl PatchTable {
    /// Number of patches (regular quads).
    pub fn len(&self) -> usize {
        self.control_points.len()
    }

    /// Whether the refined level has no regular quad at all.
    pub fn is_empty(&self) -> bool {
        self.control_points.is_empty()
    }

    /// Classification of refined face `face`.
    pub fn quad_class(&self, face: u32) -> QuadClass {
        if self.face_patches[face as usize] == u32::MAX {
            QuadClass::Feature
        } else {
            QuadClass::Regular
        }
    }

    /// The patch covering refined face `face`
    /// (`faces[patch] == face`), `None` for feature quads.
    pub fn face_patch(&self, face: u32) -> Option<u32> {
        let patch = self.face_patches[face as usize];
        (patch != u32::MAX).then_some(patch)
    }

    /// Limit position of patch `patch` at in-quad `(u, v)`.
    ///
    /// `control_positions` is indexed by the patch's control-point
    /// entries, i.e. one position per refined vertex. The basis is
    /// accumulated in f64 and rounded once on return.
    pub fn eval(&self, patch: usize, uv: [f32; 2], control_positions: &[[f32; 3]]) -> [f32; 3] {
        let wu = basis(uv[0] as f64);
        let wv = basis(uv[1] as f64);
        tensor(&self.control_points[patch], &wu, &wv, control_positions)
    }

    /// Limit position and first derivatives `(p, dp/du, dp/dv)` of
    /// patch `patch` at in-quad `(u, v)`.
    ///
    /// Derivatives are with respect to the quad's own `[0, 1]^2`
    /// parameterization (see the module docs for the scale);
    /// `dp/du x dp/dv` is the winding-oriented surface normal.
    pub fn eval_with_derivatives(
        &self,
        patch: usize,
        uv: [f32; 2],
        control_positions: &[[f32; 3]],
    ) -> ([f32; 3], [f32; 3], [f32; 3]) {
        let points = &self.control_points[patch];
        let (u, v) = (uv[0] as f64, uv[1] as f64);
        let (wu, wv) = (basis(u), basis(v));
        let (du, dv) = (derivative(u), derivative(v));
        (
            tensor(points, &wu, &wv, control_positions),
            tensor(points, &du, &wv, control_positions),
            tensor(points, &wu, &dv, control_positions),
        )
    }

    /// The 16 `(control-point index, basis weight)` pairs behind
    /// [`eval`](Self::eval) at in-quad `(u, v)` -- the sparse position
    /// row of patch `patch`, f64 weights for downstream stencil
    /// composition (`LimitEvaluator::weights_at`).
    pub(crate) fn position_weights(&self, patch: usize, uv: [f32; 2]) -> [(u32, f64); 16] {
        let points = &self.control_points[patch];
        let wu = basis(uv[0] as f64);
        let wv = basis(uv[1] as f64);
        core::array::from_fn(|slot| (points[slot], wu[slot % 4] * wv[slot / 4]))
    }
}

/// Uniform cubic B-spline basis on the patch's knot interval.
fn basis(t: f64) -> [f64; 4] {
    let s = 1.0 - t;
    [
        s * s * s / 6.0,
        (3.0 * t * t * t - 6.0 * t * t + 4.0) / 6.0,
        (-3.0 * t * t * t + 3.0 * t * t + 3.0 * t + 1.0) / 6.0,
        t * t * t / 6.0,
    ]
}

/// First derivative of [`basis`].
fn derivative(t: f64) -> [f64; 4] {
    let s = 1.0 - t;
    [
        -0.5 * s * s,
        1.5 * t * t - 2.0 * t,
        -1.5 * t * t + t + 0.5,
        0.5 * t * t,
    ]
}

/// Tensor-product accumulation of one patch over a positions buffer.
fn tensor(
    points: &[u32; 16],
    wu: &[f64; 4],
    wv: &[f64; 4],
    control_positions: &[[f32; 3]],
) -> [f32; 3] {
    let mut acc = [0.0f64; 3];
    for (j, &row_weight) in wv.iter().enumerate() {
        for (i, &column_weight) in wu.iter().enumerate() {
            let p = control_positions[points[4 * j + i] as usize];
            let w = column_weight * row_weight;
            acc[0] += w * p[0] as f64;
            acc[1] += w * p[1] as f64;
            acc[2] += w * p[2] as f64;
        }
    }
    [acc[0] as f32, acc[1] as f32, acc[2] as f32]
}

/// Control-grid slot `(i, j)` (`i` along `u`, `j` along `v`) -> index.
const fn grid(i: usize, j: usize) -> usize {
    4 * j + i
}

/// Grid slots filled from each quad corner's rotated ring (slot 0 = the
/// corner's out-edge within the quad): ring neighbors 2 and 3 and ring
/// diagonals 1, 2, 3, in that order. Neighbors 0/1 and diagonal 0 are
/// quad corners and already placed; the overlap between consecutive
/// corners' rings is debug-asserted consistent.
const RING_GRID: [[usize; 5]; 4] = [
    [grid(0, 1), grid(1, 0), grid(0, 2), grid(0, 0), grid(2, 0)],
    [grid(2, 0), grid(3, 1), grid(1, 0), grid(3, 0), grid(3, 2)],
    [grid(3, 2), grid(2, 3), grid(3, 1), grid(3, 3), grid(1, 3)],
    [grid(1, 3), grid(0, 2), grid(2, 3), grid(0, 3), grid(0, 1)],
];

fn build_patch_table(result: &RefinementResult) -> Result<PatchTable, KernelError> {
    validate_refined_quads(result)?;
    let mesh = &result.topology;
    let adjacency = &result.adjacency;

    // Per-vertex regularity: interior, valence 4, no vertex sharpness,
    // and no sharp incident edge at the refined level (the sharpness
    // convention of the limit module docs; boundary edges are sharp).
    let regular_corner: Vec<bool> = (0..mesh.vertex_count as usize)
        .map(|vi| {
            let start = adjacency.vert_edge_offsets[vi] as usize;
            let end = adjacency.vert_edge_offsets[vi + 1] as usize;
            !adjacency.vertex_is_boundary[vi]
                && end - start == 4
                && mesh.vertex_corners[vi] <= 0.0
                && adjacency.vert_edges[start..end].iter().all(|&ei| {
                    mesh.edge_creases[ei as usize] <= 0.0
                        && !adjacency.edge_is_boundary[ei as usize]
                })
        })
        .collect();

    let face_count = mesh.face_vertex_counts.len();
    let mut control_points = Vec::new();
    let mut faces = Vec::new();
    let mut face_patches = vec![u32::MAX; face_count];
    for (face, corners) in mesh.face_vertex_indices.chunks_exact(4).enumerate() {
        if corners.iter().all(|&c| regular_corner[c as usize]) {
            face_patches[face] = control_points.len() as u32;
            control_points.push(extract_control_points(
                face as u32,
                corners,
                mesh,
                adjacency,
            )?);
            faces.push(face as u32);
        }
    }

    Ok(PatchTable {
        control_points,
        faces,
        face_patches,
    })
}

/// The 4x4 control-point neighborhood of one regular quad, from its
/// corners' one-rings (see [`RING_GRID`] for the slot correspondence).
fn extract_control_points(
    face: u32,
    corners: &[u32],
    mesh: &Mesh,
    adjacency: &Adjacency,
) -> Result<[u32; 16], KernelError> {
    let mut points = [u32::MAX; 16];
    let mut place = |slot: usize, vertex: u32| {
        debug_assert!(
            points[slot] == u32::MAX || points[slot] == vertex,
            "control-point slot {slot} of face {face} disagrees between corner rings",
        );
        points[slot] = vertex;
    };
    place(grid(1, 1), corners[0]);
    place(grid(2, 1), corners[1]);
    place(grid(2, 2), corners[2]);
    place(grid(1, 2), corners[3]);

    for (k, &corner) in corners.iter().enumerate() {
        // Regular corners are interior valence-4 vertices, so the ring
        // is a 4-slot cycle and rotation is well defined.
        let ring = vertex_ring(corner as usize, mesh, adjacency)?;
        let slot =
            ring.faces
                .iter()
                .position(|&f| f == face)
                .ok_or(KernelError::InvalidTopology(
                    "quad corner ring does not contain the quad",
                ))?;
        let ring = ring.rotated(slot);
        debug_assert_eq!(
            ring.neighbors[0],
            corners[(k + 1) % 4],
            "rotated ring of corner {k} does not lead with the quad's out-edge",
        );
        debug_assert_eq!(
            ring.diagonals[0],
            corners[(k + 2) % 4],
            "rotated ring of corner {k} does not see the quad's diagonal",
        );
        debug_assert_eq!(
            ring.neighbors[1],
            corners[(k + 3) % 4],
            "rotated ring of corner {k} does not trail into the quad's in-edge",
        );
        let [n2, n3, d1, d2, d3] = RING_GRID[k];
        place(n2, ring.neighbors[2]);
        place(n3, ring.neighbors[3]);
        place(d1, ring.diagonals[1]);
        place(d2, ring.diagonals[2]);
        place(d3, ring.diagonals[3]);
    }
    debug_assert!(
        points.iter().all(|&p| p != u32::MAX),
        "regular quad {face} did not fill its 4x4 neighborhood",
    );
    Ok(points)
}

#[cfg(test)]
mod tests {
    use core::num::NonZeroU8;

    use crate::{KernelError, Mesh, Refiner, Scheme, SchemeOptions, UniformRefine};

    /// A 2x2 quad grid (the geometry gates live in
    /// `tests/patch_table.rs` and `tests/patch_osd_oracle.rs`; these
    /// unit tests cover the error paths only).
    fn grid() -> Mesh {
        Mesh {
            vertex_count: 9,
            face_vertex_counts: vec![4; 4],
            face_vertex_indices: vec![0, 3, 4, 1, 1, 4, 5, 2, 3, 6, 7, 4, 4, 7, 8, 5],
            edge_vertices: Vec::new(),
            edge_creases: Vec::new(),
            vertex_corners: vec![0.0; 9],
        }
    }

    #[test]
    fn non_catmull_clark_scheme_is_rejected() {
        let refiner =
            Refiner::new(grid(), Scheme::DooSabin, SchemeOptions::default()).expect("refiner");
        let result = refiner
            .refine_uniform(&UniformRefine::default())
            .expect("refinement");
        assert!(matches!(
            result.patch_table(),
            Err(KernelError::NotImplemented(_)),
        ));
    }

    #[test]
    fn partial_face_selection_is_rejected() {
        let req = UniformRefine {
            // SAFETY: 1 is non-zero.
            levels: NonZeroU8::new(1).unwrap(),
            selected_faces: Some(vec![true, true, true, false]),
            ..Default::default()
        };
        let refiner =
            Refiner::new(grid(), Scheme::CatmullClark, SchemeOptions::default()).expect("refiner");
        let result = refiner.refine_uniform(&req).expect("refinement");
        assert!(matches!(
            result.patch_table(),
            Err(KernelError::NotImplemented(_)),
        ));
    }
}
