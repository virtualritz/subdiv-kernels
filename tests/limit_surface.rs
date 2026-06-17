//! Limit-stencil gates that need no external oracle.
//!
//! Two properties pin the analytic Catmull-Clark limit masks:
//!
//! - **Planarity**: on a flat open grid every limit position stays in the
//!   plane and `tangent1 x tangent2` normalizes to exactly the winding
//!   normal -- smooth, crease (boundary), and single-face-corner masks
//!   all collapse to the plane.
//! - **Convergence**: the analytic limit of a level-2 vertex must agree
//!   with the position of its repeated vertex-vertex descendant eight
//!   levels deeper, and the analytic normal with the numerically
//!   averaged vertex normal there. Run for a closed cube (smooth
//!   extraordinary vertices), a cube with one infinitely sharp edge
//!   ring (interior crease rule), and a warped open grid (boundary
//!   crease + corner rules).

use std::num::NonZeroU8;
use subdiv_kernels::{
    Refinement, RefinementResult, Refiner, Scheme, SchemeOptions, Adjacency,
    Mesh, UniformRefine, VertexOrigin,
};

const SHALLOW_LEVEL: u8 = 2;
const DEEP_LEVEL: u8 = SHALLOW_LEVEL + 8;
/// Position tolerance: fraction of the cage bounding-box diagonal.
const POSITION_TOLERANCE: f64 = 1e-3;
/// Normal tolerance in degrees vs the deep numerically averaged normal.
const NORMAL_TOLERANCE_DEG: f64 = 1.0;

// -- f64 vector helpers ------------------------------------------------

fn v3(p: [f32; 3]) -> [f64; 3] {
    [p[0] as f64, p[1] as f64, p[2] as f64]
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn length(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}

fn normalize(a: [f64; 3]) -> [f64; 3] {
    let len = length(a);
    assert!(len > 1e-12, "degenerate vector cannot be normalized: {a:?}");
    [a[0] / len, a[1] / len, a[2] / len]
}

fn angle_deg(a: [f64; 3], b: [f64; 3]) -> f64 {
    dot(normalize(a), normalize(b)).clamp(-1.0, 1.0).acos() * 180.0 / std::f64::consts::PI
}

// -- Cage builders -----------------------------------------------------

/// An `n` x `n` quad grid in the y = height(i, j) sheet over the xz
/// plane (the `benches/gpu_eval.rs` `grid_topology` layout). The face
/// winding normal is -y for a flat sheet.
fn quad_grid(n: u32, height: impl Fn(u32, u32) -> f32) -> (Mesh, Vec<[f32; 3]>) {
    let stride = n + 1;
    let height = &height;
    let positions: Vec<[f32; 3]> = (0..stride)
        .flat_map(|i| (0..stride).map(move |j| [i as f32, height(i, j), j as f32]))
        .collect();

    let vid = |i: u32, j: u32| i * stride + j;
    let face_vertex_indices: Vec<u32> = (0..n)
        .flat_map(|i| {
            (0..n).flat_map(move |j| {
                [vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)]
            })
        })
        .collect();

    let topo = Mesh {
        vertex_count: stride * stride,
        face_vertex_counts: vec![4; (n * n) as usize],
        face_vertex_indices,
        edge_vertices: Vec::new(),
        edge_creases: Vec::new(),
        vertex_corners: vec![0.0; (stride * stride) as usize],
    };
    (topo, positions)
}

/// Unit-ish cube with outward (counterclockwise from outside) winding.
/// `creased_top` puts an infinitely sharp crease on the z = +1 rim.
fn cube(creased_top: bool) -> (Mesh, Vec<[f32; 3]>) {
    let positions: Vec<[f32; 3]> = vec![
        [-1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0],
        [1.0, 1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, 1.0],
        [-1.0, 1.0, 1.0],
    ];
    let face_vertex_indices: Vec<u32> = vec![
        0, 3, 2, 1, // z = -1, seen from below.
        4, 5, 6, 7, // z = +1.
        0, 1, 5, 4, // y = -1.
        1, 2, 6, 5, // x = +1.
        2, 3, 7, 6, // y = +1.
        3, 0, 4, 7, // x = -1.
    ];
    let (edge_vertices, edge_creases) = if creased_top {
        (
            vec![[4u32, 5], [5, 6], [6, 7], [4, 7]],
            vec![f32::INFINITY; 4],
        )
    } else {
        (Vec::new(), Vec::new())
    };
    let topo = Mesh {
        vertex_count: 8,
        face_vertex_counts: vec![4; 6],
        face_vertex_indices,
        edge_vertices,
        edge_creases,
        vertex_corners: vec![0.0; 8],
    };
    (topo, positions)
}

// -- Shared evaluation plumbing ----------------------------------------

/// Per-refined-vertex limit position, tangent1, tangent2.
type LimitData = (Vec<[f32; 3]>, Vec<[f32; 3]>, Vec<[f32; 3]>);

/// Refine to `SHALLOW_LEVEL` and evaluate limit position + tangents per
/// refined vertex (tables applied to the refined level's own data).
fn shallow_limit(refiner: &Refiner, positions: &[[f32; 3]]) -> (RefinementResult, LimitData) {
    let result = refiner
        .refine_uniform(&UniformRefine::from(
            NonZeroU8::new(SHALLOW_LEVEL).expect("non-zero"),
        ))
        .expect("shallow refinement");
    let refined = result.interpolate(positions);
    let limit = result.limit_stencils().expect("limit stencils");
    let lim_pos = limit.position.interpolate(&refined);
    let tan1 = limit.tangent1.interpolate(&refined);
    let tan2 = limit.tangent2.interpolate(&refined);
    (result, (lim_pos, tan1, tan2))
}

/// Map every `SHALLOW_LEVEL` vertex to its vertex-vertex descendant at
/// `DEEP_LEVEL` by chaining `VertexOrigin::Vertex` lineage per step.
fn descendants(deep: &Refinement, shallow_count: u32) -> Vec<u32> {
    (SHALLOW_LEVEL..DEEP_LEVEL).fold(
        (0..shallow_count).collect::<Vec<u32>>(),
        |tracked, step| {
            let lineage = deep
                .level_lineage(step as usize)
                .expect("deep refinement has this step");
            // Child index per parent vertex; vertex points are a suffix
            // of the child level so every parent vertex has one.
            let parent_count = tracked.iter().copied().max().unwrap_or(0) as usize + 1;
            let mut child_of = vec![u32::MAX; parent_count];
            lineage
                .vertex_origin
                .iter()
                .enumerate()
                .for_each(|(child, origin)| {
                    if let VertexOrigin::Vertex(parent) = *origin
                        && (parent as usize) < parent_count
                    {
                        child_of[parent as usize] = child as u32;
                    }
                });
            tracked
                .iter()
                .map(|&p| {
                    let c = child_of[p as usize];
                    assert_ne!(c, u32::MAX, "vertex {p} has no vertex-vertex child");
                    c
                })
                .collect()
        },
    )
}

/// Area-weighted average of incident quad normals (quad normal = cross
/// of its diagonals) around vertex `vi` of the deep level.
fn averaged_vertex_normal(
    topo: &Mesh,
    adjacency: &Adjacency,
    positions: &[[f32; 3]],
    vi: usize,
) -> [f64; 3] {
    let start = adjacency.vert_face_offsets[vi] as usize;
    let end = adjacency.vert_face_offsets[vi + 1] as usize;
    let sum = adjacency.vert_faces[start..end].iter().fold(
        [0.0f64; 3],
        |acc, &fi| {
            let off = (fi * 4) as usize;
            let c: Vec<[f64; 3]> = (0..4)
                .map(|k| v3(positions[topo.face_vertex_indices[off + k] as usize]))
                .collect();
            add(acc, cross(sub(c[2], c[0]), sub(c[3], c[1])))
        },
    );
    normalize(sum)
}

/// The convergence gate: analytic limit at `SHALLOW_LEVEL` vs the
/// vertex-vertex descendant at `DEEP_LEVEL`. Normals are skipped where
/// `skip_normal` says so (one-sided crease normals have no numeric
/// average to converge to).
fn assert_limit_converges(
    topo: Mesh,
    positions: &[[f32; 3]],
    skip_normal: impl Fn(&RefinementResult, usize) -> bool,
) {
    let refiner = Refiner::new(topo, Scheme::CatmullClark, SchemeOptions::default())
        .expect("refiner");
    let (result, (lim_pos, tan1, tan2)) = shallow_limit(&refiner, positions);

    let deep = refiner
        .refine_topology(&UniformRefine::from(
            NonZeroU8::new(DEEP_LEVEL).expect("non-zero"),
        ))
        .expect("deep refinement");
    let deep_pos = deep
        .vertex_stencils()
        .iter()
        .fold(positions.to_vec(), |data, table| table.interpolate(&data));
    let descendant = descendants(&deep, result.topology.vertex_count);
    let deep_topo = deep.final_topology();
    let deep_adjacency = deep.adjacency();

    let (lo, hi) = positions.iter().fold(
        ([f64::MAX; 3], [f64::MIN; 3]),
        |(lo, hi), &p| {
            let p = v3(p);
            (
                [lo[0].min(p[0]), lo[1].min(p[1]), lo[2].min(p[2])],
                [hi[0].max(p[0]), hi[1].max(p[1]), hi[2].max(p[2])],
            )
        },
    );
    let position_tolerance = POSITION_TOLERANCE * length(sub(hi, lo));

    for vi in 0..lim_pos.len() {
        let target = v3(deep_pos[descendant[vi] as usize]);
        let distance = length(sub(v3(lim_pos[vi]), target));
        assert!(
            distance <= position_tolerance,
            "vertex {vi}: limit position {:?} is {distance} from its deep descendant {:?} \
             (tolerance {position_tolerance})",
            lim_pos[vi],
            target,
        );

        if skip_normal(&result, vi) {
            continue;
        }
        let analytic = cross(v3(tan1[vi]), v3(tan2[vi]));
        let numeric = averaged_vertex_normal(
            deep_topo,
            deep_adjacency,
            &deep_pos,
            descendant[vi] as usize,
        );
        let angle = angle_deg(analytic, numeric);
        assert!(
            angle <= NORMAL_TOLERANCE_DEG,
            "vertex {vi}: analytic normal {:?} is {angle} deg from the deep averaged normal {:?}",
            normalize(analytic),
            numeric,
        );
    }
}

/// True when the refined vertex touches a sharp (creased) edge -- its
/// limit normal is one-sided, so the two-sided numeric average is not a
/// convergence target for it.
fn touches_crease(result: &RefinementResult, vi: usize) -> bool {
    let start = result.adjacency.vert_edge_offsets[vi] as usize;
    let end = result.adjacency.vert_edge_offsets[vi + 1] as usize;
    result.adjacency.vert_edges[start..end]
        .iter()
        .any(|&ei| result.topology.edge_creases[ei as usize] > 0.0)
}

// -- Tests ---------------------------------------------------------------

#[test]
fn flat_grid_limit_is_planar_with_winding_normal() {
    let (topo, positions) = quad_grid(6, |_, _| 0.0);
    let refiner = Refiner::new(topo, Scheme::CatmullClark, SchemeOptions::default())
        .expect("refiner");
    let (result, (lim_pos, tan1, tan2)) = shallow_limit(&refiner, &positions);

    for vi in 0..lim_pos.len() {
        assert!(
            lim_pos[vi][1].abs() <= 1e-6,
            "vertex {vi}: limit position {:?} left the plane",
            lim_pos[vi],
        );
        let normal = normalize(cross(v3(tan1[vi]), v3(tan2[vi])));
        // Faces wind +x then +z, so the winding normal is exactly -y.
        assert!(
            normal[0].abs() <= 1e-6 && (normal[1] + 1.0).abs() <= 1e-6 && normal[2].abs() <= 1e-6,
            "vertex {vi}: limit normal {normal:?} is not the -y winding normal",
        );
    }

    // The cage -> limit composition evaluates to the same values
    // straight from control points.
    let composed = result
        .compose_limit_stencils(positions.len())
        .expect("composed limit stencils");
    let from_cage = composed.position.interpolate(&positions);
    for vi in 0..lim_pos.len() {
        let distance = length(sub(v3(from_cage[vi]), v3(lim_pos[vi])));
        assert!(
            distance <= 1e-5,
            "vertex {vi}: composed limit position diverges from the chained one by {distance}",
        );
    }
}

#[test]
fn cube_limit_converges_to_deep_refinement() {
    let (topo, positions) = cube(false);
    assert_limit_converges(topo, &positions, |_, _| false);
}

#[test]
fn creased_cube_limit_converges_to_deep_refinement() {
    let (topo, positions) = cube(true);
    assert_limit_converges(topo, &positions, touches_crease);
}

#[test]
fn open_grid_limit_converges_to_deep_refinement() {
    // Warped so boundary tangents are exercised off-plane; boundary
    // normals are one-sided and well-defined, so none are skipped.
    let (topo, positions) = quad_grid(2, |i, j| {
        0.4 * (i as f32).sin() - 0.3 * (1.3 * j as f32).cos()
    });
    assert_limit_converges(topo, &positions, |_, _| false);
}
