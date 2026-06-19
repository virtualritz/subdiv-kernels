//! Sectored limit-stencil gates that need no external oracle (the
//! per-corner OpenSubdiv comparisons live in `tests/limit_osd_oracle.rs`).
//!
//! At an infinitely sharp crease the limit surface has one normal per
//! *sector* -- the fan of faces between consecutive sharp edges around
//! a vertex -- so [`SectoredLimitStencils`] must put a crease vertex's
//! two sides in different tangent rows while corners within one side
//! share theirs. Pinned (corner-rule) vertices get one row per fan.
//! Three gates:
//!
//! - **Creased cube**: at every refined vertex of the sharp rim, top
//!   and side corners land in different sector rows, the flat top row
//!   normalizes to exactly `+z`, and the two rows disagree by roughly
//!   the 90-degree crease dihedral; smooth vertices share one row and
//!   the position table is bit-identical to the per-vertex API's.
//! - **Composition**: the cage-composed tables evaluate to the same
//!   limit data as the refined-level tables.
//! - **Spoked cube**: a pinned vertex (three infinitely sharp spokes
//!   under [`CornerRule::OpenSubdivDeRose`]) splits into one row per
//!   fan -- two single-face sectors and one two-face sector whose
//!   crease-machinery cross-tangent keeps the normal in the fan's
//!   mirror plane (the bounding-edge cross product would degenerate
//!   there; see the module docs of `src/limit.rs`).

use std::num::NonZeroU8;
use subdiv_kernels::{
    CornerRule, RefinementResult, Refiner, Scheme, SchemeOptions, SectoredLimitStencils,
    Mesh, UniformRefine, VertexOrigin,
};

const LEVEL: u8 = 2;

// -- f64 vector helpers ------------------------------------------------

fn v3(p: [f32; 3]) -> [f64; 3] {
    [p[0] as f64, p[1] as f64, p[2] as f64]
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
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

/// Unit cube with outward winding; `creased_top` puts an infinitely
/// sharp crease on the z = +1 rim (the `tests/limit_surface.rs` cube).
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

fn refine(
    topo: Mesh,
    options: SchemeOptions,
    positions: &[[f32; 3]],
) -> (RefinementResult, Vec<[f32; 3]>) {
    let refiner = Refiner::new(topo, Scheme::CatmullClark, options).expect("refiner");
    let result = refiner
        .refine_uniform(&UniformRefine::from(NonZeroU8::new(LEVEL).expect("non-zero")))
        .expect("refinement");
    let refined = result.interpolate(positions);
    (result, refined)
}

// -- Shared inspection plumbing ----------------------------------------

/// Number of sharp (creased) edges incident to refined vertex `vi`.
fn sharp_edge_count(result: &RefinementResult, vi: usize) -> usize {
    let start = result.adjacency.vertex_edge_offsets[vi] as usize;
    let end = result.adjacency.vertex_edge_offsets[vi + 1] as usize;
    result.adjacency.vertex_edges[start..end]
        .iter()
        .filter(|&&ei| result.topology.edge_creases[ei as usize] > 0.0)
        .count()
}

/// `(face, sector row)` per incident face-corner of refined vertex `vi`.
fn corner_rows(
    result: &RefinementResult,
    sectored: &SectoredLimitStencils,
    vi: usize,
) -> Vec<(u32, u32)> {
    let start = result.adjacency.vertex_face_offsets[vi] as usize;
    let end = result.adjacency.vertex_face_offsets[vi + 1] as usize;
    result.adjacency.vertex_faces[start..end]
        .iter()
        .map(|&fi| {
            let off = (fi * 4) as usize;
            let corner = result.topology.face_vertex_indices[off..off + 4]
                .iter()
                .position(|&c| c == vi as u32)
                .expect("incident face contains its vertex");
            (fi, sectored.corner_sector[off + corner])
        })
        .collect()
}

/// Unit normal of sector row `row` from interpolated tangent tables.
fn row_normal(tan1: &[[f32; 3]], tan2: &[[f32; 3]], row: u32) -> [f64; 3] {
    normalize(cross(v3(tan1[row as usize]), v3(tan2[row as usize])))
}

// -- Tests ---------------------------------------------------------------

#[test]
fn creased_cube_corners_split_into_per_side_sectors() {
    let (topo, positions) = cube(true);
    let (result, refined) = refine(topo, SchemeOptions::default(), &positions);
    let sectored = result
        .sectored_limit_stencils()
        .expect("sectored limit stencils");
    let limit = result.limit_stencils().expect("per-vertex limit stencils");

    // The position table is the per-vertex one, bit for bit.
    assert_eq!(sectored.position, limit.position);
    assert_eq!(
        sectored.corner_sector.len(),
        result.topology.face_vertex_indices.len(),
    );

    let tan1 = sectored.tangent1.interpolate(&refined);
    let tan2 = sectored.tangent2.interpolate(&refined);
    assert_eq!(tan1.len(), tan2.len());
    assert!(
        sectored
            .corner_sector
            .iter()
            .all(|&row| (row as usize) < tan1.len()),
        "corner_sector references a row beyond the tangent tables",
    );

    let vertex_count = result.topology.vertex_count as usize;
    let crease_vertices: Vec<usize> = (0..vertex_count)
        .filter(|&vi| sharp_edge_count(&result, vi) == 2)
        .collect();
    // The level-2 rim: 4 base corners + 12 edge descendants.
    assert_eq!(crease_vertices.len(), 16, "unexpected rim vertex count");
    // One row per vertex plus one extra per two-sector crease vertex.
    assert_eq!(tan1.len(), vertex_count + crease_vertices.len());

    // A face is on the flat top sheet iff all its corners stayed at
    // z = +1 (the creased top subdivides within its own plane).
    let face_on_top = |fi: u32| -> bool {
        let off = (fi * 4) as usize;
        result.topology.face_vertex_indices[off..off + 4]
            .iter()
            .all(|&c| (refined[c as usize][2] - 1.0).abs() < 1e-3)
    };

    for vi in 0..vertex_count {
        let rows = corner_rows(&result, &sectored, vi);
        if !crease_vertices.contains(&vi) {
            // Smooth vertices: every corner shares the single sector.
            assert!(
                rows.iter().all(|&(_, row)| row == rows[0].1),
                "smooth vertex {vi} has corners in different sectors: {rows:?}",
            );
            continue;
        }

        let top_rows: Vec<u32> = rows
            .iter()
            .filter(|&&(fi, _)| face_on_top(fi))
            .map(|&(_, row)| row)
            .collect();
        let side_rows: Vec<u32> = rows
            .iter()
            .filter(|&&(fi, _)| !face_on_top(fi))
            .map(|&(_, row)| row)
            .collect();
        assert!(
            !top_rows.is_empty() && !side_rows.is_empty(),
            "rim vertex {vi} is missing a side: {rows:?}",
        );
        // Corners within one side share their row; the sides differ.
        assert!(
            top_rows.iter().all(|&row| row == top_rows[0]),
            "rim vertex {vi} has top corners in different sectors: {top_rows:?}",
        );
        assert!(
            side_rows.iter().all(|&row| row == side_rows[0]),
            "rim vertex {vi} has side corners in different sectors: {side_rows:?}",
        );
        assert_ne!(
            top_rows[0], side_rows[0],
            "rim vertex {vi} shades both sides from one sector row",
        );

        // The flat top's sector normal is exactly +z; the side normal
        // disagrees by roughly the 90-degree crease dihedral.
        let top_normal = row_normal(&tan1, &tan2, top_rows[0]);
        assert!(
            angle_deg(top_normal, [0.0, 0.0, 1.0]) < 0.1,
            "rim vertex {vi}: top sector normal {top_normal:?} is not +z",
        );
        let side_normal = row_normal(&tan1, &tan2, side_rows[0]);
        let dihedral = angle_deg(top_normal, side_normal);
        assert!(
            (60.0..=120.0).contains(&dihedral),
            "rim vertex {vi}: sector normals disagree by {dihedral} degrees, expected ~90",
        );
    }
}

#[test]
fn composed_sectored_tables_evaluate_from_the_cage() {
    let (topo, positions) = cube(true);
    let (result, refined) = refine(topo, SchemeOptions::default(), &positions);
    let sectored = result
        .sectored_limit_stencils()
        .expect("sectored limit stencils");
    let composed = result
        .compose_sectored_limit_stencils(positions.len())
        .expect("composed sectored limit stencils");

    assert_eq!(composed.corner_sector, sectored.corner_sector);

    let chained = [
        sectored.position.interpolate(&refined),
        sectored.tangent1.interpolate(&refined),
        sectored.tangent2.interpolate(&refined),
    ];
    let from_cage = [
        composed.position.interpolate(&positions),
        composed.tangent1.interpolate(&positions),
        composed.tangent2.interpolate(&positions),
    ];
    for (name, (chained, from_cage)) in ["position", "tangent1", "tangent2"]
        .iter()
        .zip(chained.iter().zip(&from_cage))
    {
        assert_eq!(chained.len(), from_cage.len());
        for (row, (a, b)) in chained.iter().zip(from_cage).enumerate() {
            let distance = length(sub(v3(*a), v3(*b)));
            assert!(
                distance <= 1e-4,
                "{name} row {row}: composed value {b:?} diverges from chained {a:?}",
            );
        }
    }
}

/// Once-refined cube with three infinitely sharp spokes at the top
/// face-center: the center is pinned (corner rule) with two single-face
/// sectors and one two-face sector, all interior. Returns the spoked
/// cage, its positions, and the pinned vertex's cage index.
fn spoked_cube() -> (Mesh, Vec<[f32; 3]>, u32) {
    let (topo, positions) = cube(false);
    let refiner =
        Refiner::new(topo, Scheme::CatmullClark, SchemeOptions::default()).expect("refiner");
    let result = refiner
        .refine_uniform(&UniformRefine::default())
        .expect("cage refinement");
    let cage_positions = result.interpolate(&positions);

    let center = result
        .lineage
        .vertex_origin
        .iter()
        .position(|origin| *origin == VertexOrigin::Face(1))
        .expect("face point of base face 1") as u32;
    let estart = result.adjacency.vertex_edge_offsets[center as usize] as usize;
    let neighbors: Vec<u32> = result.adjacency.vertex_edges[estart..estart + 3]
        .iter()
        .map(|&ei| {
            let [a, b] = result.topology.edge_vertices[ei as usize];
            if a == center { b } else { a }
        })
        .collect();
    let edge_vertices: Vec<[u32; 2]> = neighbors.iter().map(|&n| [center, n]).collect();
    let edge_creases = vec![f32::INFINITY; edge_vertices.len()];

    let cage = Mesh {
        vertex_count: result.topology.vertex_count,
        face_vertex_counts: result.topology.face_vertex_counts.clone(),
        face_vertex_indices: result.topology.face_vertex_indices.clone(),
        edge_vertices,
        edge_creases,
        vertex_corners: vec![0.0; result.topology.vertex_count as usize],
    };
    (cage, cage_positions, center)
}

#[test]
fn spoked_cube_pinned_vertex_gets_one_row_per_fan() {
    let (cage, cage_positions, center) = spoked_cube();
    let center_position = v3(cage_positions[center as usize]);
    // OpenSubdiv's corner rule pins the three-sharp-edge vertex.
    let options = SchemeOptions {
        corner_rule: CornerRule::OpenSubdivDeRose,
        ..Default::default()
    };
    let (result, refined) = refine(cage, options, &cage_positions);
    let sectored = result
        .sectored_limit_stencils()
        .expect("sectored limit stencils");
    let tan1 = sectored.tangent1.interpolate(&refined);
    let tan2 = sectored.tangent2.interpolate(&refined);

    // The pinned vertex's descendant sits exactly at the cage position.
    let pinned = (0..result.topology.vertex_count as usize)
        .find(|&vi| length(sub(v3(refined[vi]), center_position)) < 1e-6)
        .expect("pinned vertex descendant");
    assert_eq!(sharp_edge_count(&result, pinned), 3);

    // One row per fan: two single-face sectors and one two-face sector
    // (its row appears on both of that fan's corners).
    let rows = corner_rows(&result, &sectored, pinned);
    assert_eq!(rows.len(), 4, "pinned vertex should have four corners");
    let mut distinct: Vec<u32> = rows.iter().map(|&(_, row)| row).collect();
    distinct.sort_unstable();
    distinct.dedup();
    assert_eq!(
        distinct.len(),
        3,
        "pinned vertex should have three sectors: {rows:?}",
    );
    let multiplicity = |row: u32| rows.iter().filter(|&&(_, r)| r == row).count();
    let mut sizes: Vec<usize> = distinct.iter().map(|&row| multiplicity(row)).collect();
    sizes.sort_unstable();
    assert_eq!(sizes, [1, 1, 2], "fan sizes should be 1, 1, 2: {rows:?}");

    // The spokes run along -x, -y, +x, so the mesh mirrors in x: the
    // two-face sector's crease-machinery normal stays in the y-z
    // mirror plane pointing outward, and the single-face sector
    // normals are each other's x-mirror images.
    let two_face_row = distinct
        .iter()
        .copied()
        .find(|&row| multiplicity(row) == 2)
        .expect("two-face sector row");
    let n2 = row_normal(&tan1, &tan2, two_face_row);
    assert!(
        n2[0].abs() < 1e-4 && n2[1] > 0.0 && n2[2] > 0.0,
        "two-face sector normal {n2:?} should lie in the y-z plane pointing out",
    );
    let singles: Vec<[f64; 3]> = distinct
        .iter()
        .copied()
        .filter(|&row| multiplicity(row) == 1)
        .map(|row| row_normal(&tan1, &tan2, row))
        .collect();
    let mirrored = [-singles[1][0], singles[1][1], singles[1][2]];
    assert!(
        angle_deg(singles[0], mirrored) < 0.1,
        "single-face sector normals {singles:?} should be x-mirror images",
    );
    assert!(
        angle_deg(singles[0], n2) > 10.0,
        "single-face and two-face sector normals should differ: {singles:?} vs {n2:?}",
    );
}
