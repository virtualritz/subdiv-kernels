//! Closest-point gates (limit-surface SDF design s3) -- all in-crate,
//! no OSD: the decisive oracle is a brute-force argmin over a dense
//! level-7 limit-surface point cloud (`tests/common/mod.rs`'s
//! `BruteOracle`; every sample lies exactly on the limit surface, so
//! the brute distance upper-bounds the true one). These deep-refinement
//! suites take minutes in debug, like the other oracle gates.
//!
//! Five properties pin the query:
//!
//! - **Brute-force argmin** (warped grid, cube L1, creased cube L2,
//!   ~120 deterministic queries: surface-near, far-field, inside,
//!   medial-axis, near-EV, near-crease): `closest_point.distance`
//!   never exceeds the brute distance beyond f32 noise, and the foot
//!   point matches the brute argmin within the sampling-resolution
//!   tolerance wherever the minimizer is unique. Medial-axis queries
//!   whose argmin is ambiguous at sampling resolution are skipped with
//!   counted assertions.
//! - **Sign**: on the closed cube/creased cube a known inside grid is
//!   negative and known outside points (beyond the cage hull) are
//!   positive, with `|signed| == distance`; the open grid returns
//!   `None`.
//! - **Seam/walk**: queries constructed to project exactly onto
//!   refined-quad edges (and straddling pairs nudged toward either
//!   side) return the seam point consistently from both sides --
//!   the adjacency walk, not a clamped one-sided minimum.
//! - **Corner/fan**: queries along the limit normal of an
//!   extraordinary vertex project onto the EV limit point (the corner
//!   fan walk), and crease-wedge queries along the rim's pseudonormal
//!   bisector sign correctly on both sides of the surface.
//! - **Determinism**: repeated queries are bit-identical (the cached
//!   index and isolation caches).

use std::num::NonZeroU8;
use subdiv_kernels::{
    ClosestPoint, LimitEvaluator, RefinementResult, Refiner, Scheme, UniformRefine,
};

mod common;

use common::{
    BruteOracle, Case, brute_oracle, cross, cube_case, dot, grid_case, length, normalize, scale,
    sub, v3,
};

/// Brute-oracle refinement depth (design: "level 7-8"; 7 keeps the
/// debug-mode suites in minutes while the cloud spacing stays well
/// below every feature scale of the cases).
const ORACLE_LEVEL: u8 = 7;

/// `closest_point.distance` may exceed the brute distance only by f32
/// evaluation noise -- the foot point itself lies on the limit surface.
const DISTANCE_SLACK: f64 = 1e-4;

const CORNER_UV: [[f32; 2]; 4] = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];

struct EvaluatedCase {
    result: RefinementResult,
    refined: Vec<[f32; 3]>,
}

fn evaluated_case(case: &Case, level: u8) -> EvaluatedCase {
    let refiner =
        Refiner::new(case.topology(), Scheme::CatmullClark, case.options).expect("refiner");
    let result = refiner
        .refine_uniform(&UniformRefine::from(
            NonZeroU8::new(level).expect("non-zero"),
        ))
        .expect("refinement");
    let refined = result.interpolate(&case.positions);
    EvaluatedCase { result, refined }
}

fn warped_grid() -> Case {
    grid_case(4, |i, j| {
        0.25 * (i as f32) * (i as f32) - 0.2 * (j as f32) * (i as f32 + 1.0)
    })
}

/// All 26 nonzero sign patterns of `{-1, 0, 1}^3`.
fn sign_directions() -> Vec<[f32; 3]> {
    const AXIS: [f32; 3] = [-1.0, 0.0, 1.0];
    AXIS.iter()
        .flat_map(|&x| {
            AXIS.iter()
                .flat_map(move |&y| AXIS.iter().map(move |&z| [x, y, z]))
        })
        .filter(|d| d.iter().any(|&c| c != 0.0))
        .collect()
}

// -- Brute-force oracle ------------------------------------------------------

#[derive(Debug, Default)]
struct OracleStats {
    queries: usize,
    position_checked: usize,
    ambiguous_skips: usize,
    max_distance_gap: f64,
    max_position_error: f64,
}

/// The decisive gate: per query, internal consistency, distance never
/// beyond the brute argmin, and -- where the minimizer is unique at
/// sampling resolution -- the foot point within the lateral tolerance
/// of the brute argmin (`sqrt(2 d h)` drift at sample spacing `h`,
/// plus the spacing itself).
fn assert_matches_brute(
    evaluator: &LimitEvaluator,
    oracle: &BruteOracle,
    queries: &[[f32; 3]],
) -> OracleStats {
    let mut stats = OracleStats::default();
    for &query in queries {
        let cp = evaluator.closest_point(query).expect("closest point");
        let q = v3(query);
        stats.queries += 1;

        // The reported distance is the distance to the reported foot.
        let measured = length(sub(v3(cp.position), q));
        assert!(
            (measured - cp.distance as f64).abs() <= 1e-5 * (1.0 + measured),
            "query {query:?}: reported distance {} vs measured {measured}",
            cp.distance,
        );
        if let Some(signed) = cp.signed_distance {
            assert_eq!(
                signed.abs(),
                cp.distance,
                "query {query:?}: |signed| != distance"
            );
        }

        let (brute_pos, brute_dist) = oracle.closest(q);
        let gap = cp.distance as f64 - brute_dist;
        stats.max_distance_gap = stats.max_distance_gap.max(gap);
        assert!(
            gap <= DISTANCE_SLACK,
            "query {query:?}: ours {} is farther than brute {brute_dist}",
            cp.distance,
        );
        assert!(
            cp.distance as f64 >= brute_dist - 2.0 * oracle.max_edge,
            "query {query:?}: ours {} undercuts the brute floor {brute_dist} \
             (max_edge {})",
            cp.distance,
            oracle.max_edge,
        );

        let pos_tol = (2.0 * brute_dist * oracle.max_edge).sqrt() + 2.0 * oracle.max_edge;
        if oracle.ambiguous(q, (brute_pos, brute_dist), oracle.max_edge, 3.0 * pos_tol) {
            stats.ambiguous_skips += 1;
        } else {
            let error = length(sub(v3(cp.position), brute_pos));
            stats.max_position_error = stats.max_position_error.max(error);
            assert!(
                error <= pos_tol,
                "query {query:?}: foot {:?} is {error} from brute argmin {brute_pos:?} \
                 (tolerance {pos_tol})",
                cp.position,
            );
            stats.position_checked += 1;
        }
    }
    assert!(stats.position_checked > 0, "no position was checked at all");
    stats
}

#[test]
fn warped_grid_matches_brute_force_oracle() {
    // 45 fixed queries spanning the sheet (x, z in [0, 4]) and beyond
    // its open boundary, below/near/above in y -- boundary projections
    // exercise the no-neighbor constrained minimum.
    let queries: Vec<[f32; 3]> = [-1.0f32, 0.6, 2.1, 3.4, 5.2]
        .iter()
        .flat_map(|&x| {
            [-2.5f32, 0.4, 3.0]
                .iter()
                .flat_map(move |&y| [-0.8f32, 1.7, 4.6].iter().map(move |&z| [x, y, z]))
        })
        .collect();
    let case = warped_grid();
    let ours = evaluated_case(&case, 2);
    let evaluator = ours
        .result
        .limit_evaluator(&ours.refined)
        .expect("limit evaluator");
    let oracle = brute_oracle(&case, ORACLE_LEVEL);
    let stats = assert_matches_brute(&evaluator, &oracle, &queries);
    println!(
        "warped grid oracle: {stats:?} (max_edge {})",
        oracle.max_edge
    );
    assert!(
        stats.ambiguous_skips <= queries.len() / 5,
        "too many ambiguous skips: {stats:?}",
    );
}

#[test]
fn cube_matches_brute_force_oracle() {
    // Far field (26 directions), surface-near outside along axes and
    // EV diagonals, inside points, and deliberate medial-axis probes
    // (the origin and its axis neighborhood).
    let mut queries: Vec<[f32; 3]> = sign_directions()
        .iter()
        .map(|d| [d[0] * 3.0, d[1] * 3.0, d[2] * 3.0])
        .collect();
    // Near-surface outside: axis face centers and EV corners.
    queries.extend([
        [1.05, 0.0, 0.0],
        [-1.05, 0.0, 0.0],
        [0.0, 1.05, 0.0],
        [0.0, -1.05, 0.0],
        [0.0, 0.0, 1.05],
        [0.0, 0.0, -1.05],
    ]);
    queries.extend(
        sign_directions()
            .iter()
            .filter(|d| d.iter().all(|&c| c != 0.0))
            .map(|d| [d[0] * 0.95, d[1] * 0.95, d[2] * 0.95]),
    );
    // Inside / medial axis: the origin is ambiguous by symmetry.
    queries.extend([
        [0.0, 0.0, 0.0],
        [0.2, 0.0, 0.0],
        [-0.2, 0.0, 0.0],
        [0.0, 0.2, 0.0],
        [0.0, -0.2, 0.0],
        [0.0, 0.0, 0.2],
        [0.0, 0.0, -0.2],
        [0.1, 0.15, -0.05],
        [0.3, 0.3, 0.3],
    ]);
    let case = cube_case(false);
    let ours = evaluated_case(&case, 1);
    let evaluator = ours
        .result
        .limit_evaluator(&ours.refined)
        .expect("limit evaluator");
    let oracle = brute_oracle(&case, ORACLE_LEVEL);
    let stats = assert_matches_brute(&evaluator, &oracle, &queries);
    println!("cube oracle: {stats:?} (max_edge {})", oracle.max_edge);
    assert!(
        stats.ambiguous_skips >= 1,
        "the origin must be ambiguous: {stats:?}"
    );
    assert!(
        stats.ambiguous_skips <= 10,
        "too many ambiguous skips: {stats:?}",
    );
}

#[test]
fn creased_cube_matches_brute_force_oracle() {
    // The infinitely sharp top rim: queries above/outside the crease
    // line and its corners, just inside the rim, plus inside and
    // far-field coverage.
    let mut queries: Vec<[f32; 3]> = vec![
        [3.0, 0.0, 0.0],
        [-3.0, 0.0, 0.0],
        [0.0, 3.0, 0.0],
        [0.0, -3.0, 0.0],
        [0.0, 0.0, 3.0],
        [0.0, 0.0, -3.0],
    ];
    for &x in &[-0.9f32, -0.3, 0.4, 0.9] {
        // Outside the rim wedge on both rim pairs.
        queries.push([x, 1.2, 1.2]);
        queries.push([x, -1.25, 1.15]);
        queries.push([1.2, x, 1.25]);
        // Just inside the rim.
        queries.push([x, 0.8, 0.85]);
    }
    // Rim corners (sharp crease meets EV) and the top face.
    queries.extend([
        [1.2, 1.2, 1.2],
        [-1.15, 1.2, 1.25],
        [1.25, -1.2, 1.15],
        [0.2, 0.0, 1.4],
        [-0.3, 0.1, 1.3],
    ]);
    // Inside / medial axis.
    queries.extend([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.4],
        [0.25, -0.2, 0.1],
        [0.0, 0.5, 0.5],
    ]);
    let case = cube_case(true);
    let ours = evaluated_case(&case, 2);
    let evaluator = ours
        .result
        .limit_evaluator(&ours.refined)
        .expect("limit evaluator");
    let oracle = brute_oracle(&case, ORACLE_LEVEL);
    let stats = assert_matches_brute(&evaluator, &oracle, &queries);
    println!(
        "creased cube oracle: {stats:?} (max_edge {})",
        oracle.max_edge
    );
    assert!(
        stats.ambiguous_skips <= queries.len() / 5,
        "too many ambiguous skips: {stats:?}",
    );
}

// -- Sign --------------------------------------------------------------------

/// Inside grid strictly within the sampled inscribed radius, outside
/// points strictly beyond the cage hull (the limit lies inside the
/// cage's convex hull, so `|q|_inf > 1` is provably outside).
fn assert_sign_classifies(case: &Case, level: u8) {
    let ours = evaluated_case(case, level);
    let evaluator = ours
        .result
        .limit_evaluator(&ours.refined)
        .expect("limit evaluator");
    // A coarse cloud suffices for the inscribed-radius margin.
    let oracle = brute_oracle(case, 5);
    let r_min = oracle.min_radius();
    assert!(r_min > 0.3, "unexpectedly small inscribed radius {r_min}");

    let inner = (0.45 * r_min) as f32;
    let axis = [-1.0f32, 0.0, 1.0];
    for &x in &axis {
        for &y in &axis {
            for &z in &axis {
                let q = [x * inner, y * inner, z * inner];
                let cp = evaluator.closest_point(q).expect("inside query");
                let signed = cp.signed_distance.expect("closed surface sign");
                assert!(signed < 0.0, "inside query {q:?} got signed {signed}");
                assert_eq!(signed.abs(), cp.distance);
            }
        }
    }
    for direction in sign_directions() {
        for &radius in &[1.1f32, 3.0] {
            let q = [
                direction[0] * radius,
                direction[1] * radius,
                direction[2] * radius,
            ];
            let cp = evaluator.closest_point(q).expect("outside query");
            let signed = cp.signed_distance.expect("closed surface sign");
            assert!(signed > 0.0, "outside query {q:?} got signed {signed}");
            assert_eq!(signed.abs(), cp.distance);
        }
    }
}

#[test]
fn cube_signed_distance_classifies_inside_and_outside() {
    assert_sign_classifies(&cube_case(false), 1);
}

#[test]
fn creased_cube_signed_distance_classifies_inside_and_outside() {
    assert_sign_classifies(&cube_case(true), 2);
}

#[test]
fn open_grid_signed_distance_is_none() {
    let case = warped_grid();
    let ours = evaluated_case(&case, 1);
    let evaluator = ours
        .result
        .limit_evaluator(&ours.refined)
        .expect("limit evaluator");
    for q in [[2.0f32, 3.0, 2.0], [2.0, -3.0, 2.0], [-1.0, 0.0, 5.0]] {
        let cp = evaluator.closest_point(q).expect("open-mesh query");
        assert!(
            cp.signed_distance.is_none(),
            "open mesh returned a sign for {q:?}: {:?}",
            cp.signed_distance,
        );
    }
}

// -- Seam/walk ---------------------------------------------------------------

/// `(u, v)` at parameter `x` along CSR edge slot `slot`.
fn edge_uv(slot: usize, x: f32) -> [f32; 2] {
    let (a, b) = (CORNER_UV[slot], CORNER_UV[(slot + 1) % 4]);
    [a[0] + x * (b[0] - a[0]), a[1] + x * (b[1] - a[1])]
}

/// Queries projecting exactly onto interior refined-quad seams: the
/// foot must come back as the seam point (not a one-sided clamp), and
/// straddling pairs nudged tangentially toward either quad must agree.
fn assert_seam_queries_consistent(case: &Case, level: u8) {
    let ours = evaluated_case(case, level);
    let evaluator = ours
        .result
        .limit_evaluator(&ours.refined)
        .expect("limit evaluator");
    let interior: Vec<usize> = ours
        .result
        .adjacency
        .edge_faces
        .iter()
        .enumerate()
        .filter(|&(_, &[fa, fb])| fa != u32::MAX && fb != u32::MAX)
        .map(|(ei, _)| ei)
        .collect();
    assert!(interior.len() >= 8, "case has too few interior seams");

    let stride = interior.len() / 8;
    for &ei in interior.iter().step_by(stride.max(1)).take(8) {
        let [fa, _] = ours.result.adjacency.edge_faces[ei];
        let slot = ours.result.adjacency.face_edges[fa as usize * 4..fa as usize * 4 + 4]
            .iter()
            .position(|&e| e as usize == ei)
            .expect("edge is in its incident face");
        let (p, du, dv) = evaluator
            .eval_with_derivatives(fa, edge_uv(slot, 0.5))
            .expect("seam eval");
        let seam = v3(p);
        let normal = cross(v3(du), v3(dv));
        if length(normal) < 1e-9 {
            // Crease-crease degeneracies have no one-sided normal here.
            continue;
        }
        let normal = normalize(normal);
        // The seam direction and the in-surface perpendicular pointing
        // across it.
        let tangent = normalize(match slot {
            0 => v3(du),
            1 => v3(dv),
            2 => scale(v3(du), -1.0),
            _ => scale(v3(dv), -1.0),
        });
        let across = normalize(cross(normal, tangent));

        for &t in &[0.02f64, 0.1] {
            let q = [
                (seam[0] + t * normal[0]) as f32,
                (seam[1] + t * normal[1]) as f32,
                (seam[2] + t * normal[2]) as f32,
            ];
            let cp = evaluator.closest_point(q).expect("seam query");
            let off = length(sub(v3(cp.position), seam));
            assert!(
                off <= 1e-3,
                "edge {ei} at t = {t}: foot {:?} is {off} from the seam point {seam:?}",
                cp.position,
            );
            assert!(
                (cp.distance as f64 - t).abs() <= 1e-3,
                "edge {ei} at t = {t}: distance {}",
                cp.distance,
            );

            // Straddle: nudge toward either incident quad; both sides
            // must resolve to the same seam neighborhood.
            let nudge = 1e-5;
            let straddle = |side: f64| {
                let q = [
                    (seam[0] + t * normal[0] + side * nudge * across[0]) as f32,
                    (seam[1] + t * normal[1] + side * nudge * across[1]) as f32,
                    (seam[2] + t * normal[2] + side * nudge * across[2]) as f32,
                ];
                evaluator.closest_point(q).expect("straddle query")
            };
            let (left, right) = (straddle(-1.0), straddle(1.0));
            let spread = length(sub(v3(left.position), v3(right.position)));
            assert!(
                spread <= 1e-3,
                "edge {ei} at t = {t}: straddle feet disagree by {spread}",
            );
            assert!(
                (left.distance as f64 - right.distance as f64).abs() <= 1e-4,
                "edge {ei} at t = {t}: straddle distances {} vs {}",
                left.distance,
                right.distance,
            );
        }
    }
}

#[test]
fn cube_seam_queries_agree_from_both_sides() {
    assert_seam_queries_consistent(&cube_case(false), 2);
}

#[test]
fn warped_grid_seam_queries_agree_from_both_sides() {
    assert_seam_queries_consistent(&warped_grid(), 1);
}

#[test]
fn extraordinary_corner_queries_walk_the_vertex_fan() {
    // Queries along the limit normal of a valence-3 EV project onto
    // the EV limit point -- on a seam corner shared by three quads, so
    // any one candidate quad must walk/fan to agree.
    let case = cube_case(false);
    let ours = evaluated_case(&case, 1);
    let evaluator = ours
        .result
        .limit_evaluator(&ours.refined)
        .expect("limit evaluator");
    let limit = ours.result.limit_stencils().expect("limit stencils");
    let positions = limit.position.interpolate(&ours.refined);
    let tan1 = limit.tangent1.interpolate(&ours.refined);
    let tan2 = limit.tangent2.interpolate(&ours.refined);

    let valence = |vi: usize| {
        ours.result.adjacency.vert_edge_offsets[vi + 1]
            - ours.result.adjacency.vert_edge_offsets[vi]
    };
    let ev = (0..ours.result.topology.vertex_count as usize)
        .find(|&vi| valence(vi) == 3)
        .expect("an extraordinary vertex");
    let p = v3(positions[ev]);
    let normal = normalize(cross(v3(tan1[ev]), v3(tan2[ev])));

    for &t in &[0.05f64, 0.3, -0.15] {
        let q = [
            (p[0] + t * normal[0]) as f32,
            (p[1] + t * normal[1]) as f32,
            (p[2] + t * normal[2]) as f32,
        ];
        let cp = evaluator.closest_point(q).expect("EV query");
        let off = length(sub(v3(cp.position), p));
        assert!(
            off <= 2e-3,
            "EV query at t = {t}: foot {:?} is {off} from the EV limit {p:?}",
            cp.position,
        );
        assert!(
            (cp.distance as f64 - t.abs()).abs() <= 2e-3,
            "EV query at t = {t}: distance {}",
            cp.distance,
        );
        let signed = cp.signed_distance.expect("closed surface sign");
        assert_eq!(
            signed < 0.0,
            t < 0.0,
            "EV query at t = {t}: signed {signed}",
        );
    }
}

#[test]
fn crease_wedge_queries_sign_by_pseudonormal() {
    // A rim crease vertex of the creased cube has two sector normals;
    // queries along their bisector are closest to the rim point itself
    // and must sign by the edge pseudonormal: positive outside the
    // wedge, negative on the mirrored inside ray.
    let case = cube_case(true);
    let ours = evaluated_case(&case, 2);
    let evaluator = ours
        .result
        .limit_evaluator(&ours.refined)
        .expect("limit evaluator");
    let sectored = ours
        .result
        .sectored_limit_stencils()
        .expect("sectored limit stencils");
    let limit_positions = sectored.position.interpolate(&ours.refined);
    let tan1 = sectored.tangent1.interpolate(&ours.refined);
    let tan2 = sectored.tangent2.interpolate(&ours.refined);

    let sharp_count = |vi: usize| {
        let start = ours.result.adjacency.vert_edge_offsets[vi] as usize;
        let end = ours.result.adjacency.vert_edge_offsets[vi + 1] as usize;
        ours.result.adjacency.vert_edges[start..end]
            .iter()
            .filter(|&&e| ours.result.topology.edge_creases[e as usize] > 0.0)
            .count()
    };
    let rim = (0..ours.result.topology.vertex_count as usize)
        .find(|&vi| sharp_count(vi) == 2 && valence_of(&ours, vi) == 4)
        .expect("a rim crease vertex");

    // The vertex's two sector normals from its incident face-corners.
    let mut normals: Vec<[f64; 3]> = Vec::new();
    let fstart = ours.result.adjacency.vert_face_offsets[rim] as usize;
    let fend = ours.result.adjacency.vert_face_offsets[rim + 1] as usize;
    for &face in &ours.result.adjacency.vert_faces[fstart..fend] {
        let k = ours.result.topology.face_vertex_indices[face as usize * 4..face as usize * 4 + 4]
            .iter()
            .position(|&c| c as usize == rim)
            .expect("fan face contains the rim vertex");
        let row = sectored.corner_sector[face as usize * 4 + k] as usize;
        let n = normalize(cross(v3(tan1[row]), v3(tan2[row])));
        if !normals.iter().any(|m| dot(*m, n) > 0.999) {
            normals.push(n);
        }
    }
    assert_eq!(
        normals.len(),
        2,
        "rim vertex should have two sector normals"
    );

    let p = v3(limit_positions[rim]);
    let bisector = normalize([
        normals[0][0] + normals[1][0],
        normals[0][1] + normals[1][1],
        normals[0][2] + normals[1][2],
    ]);

    let out = [
        (p[0] + 0.3 * bisector[0]) as f32,
        (p[1] + 0.3 * bisector[1]) as f32,
        (p[2] + 0.3 * bisector[2]) as f32,
    ];
    let cp = evaluator.closest_point(out).expect("wedge query");
    let off = length(sub(v3(cp.position), p));
    assert!(
        off <= 5e-3,
        "wedge query foot {:?} is {off} from the rim point {p:?}",
        cp.position,
    );
    let signed = cp.signed_distance.expect("closed surface sign");
    assert!(signed > 0.0, "outside wedge query got signed {signed}");

    let inside = [
        (p[0] - 0.3 * bisector[0]) as f32,
        (p[1] - 0.3 * bisector[1]) as f32,
        (p[2] - 0.3 * bisector[2]) as f32,
    ];
    let cp = evaluator.closest_point(inside).expect("inside wedge query");
    let signed = cp.signed_distance.expect("closed surface sign");
    assert!(signed < 0.0, "inside wedge query got signed {signed}");
}

fn valence_of(ours: &EvaluatedCase, vi: usize) -> u32 {
    ours.result.adjacency.vert_edge_offsets[vi + 1] - ours.result.adjacency.vert_edge_offsets[vi]
}

// -- Determinism ---------------------------------------------------------------

#[test]
fn repeated_queries_are_bit_identical() {
    let case = cube_case(true);
    let ours = evaluated_case(&case, 2);
    let evaluator = ours
        .result
        .limit_evaluator(&ours.refined)
        .expect("limit evaluator");
    let queries = [
        [3.0f32, 0.2, 0.4],
        [0.0, 1.2, 1.2],
        [0.1, 0.0, 0.0],
        [-0.9, -1.25, 1.15],
        [0.0, 0.0, 1.4],
    ];
    let first: Vec<ClosestPoint> = queries
        .iter()
        .map(|&q| evaluator.closest_point(q).expect("first pass"))
        .collect();
    let second: Vec<ClosestPoint> = queries
        .iter()
        .map(|&q| evaluator.closest_point(q).expect("second pass"))
        .collect();
    assert_eq!(first, second, "cached index changed a repeated query");
}
