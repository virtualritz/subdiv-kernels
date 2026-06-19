//! Closest point and signed distance to the Catmull-Clark limit
//! surface (limit-surface SDF design s3, per design sections 4-4/4-5).
//!
//! [`LimitEvaluator::closest_point`] answers point queries against the
//! exact limit surface the s1/s2 machinery evaluates: a best-first
//! descent of a per-refined-quad AABB hierarchy proposes candidate
//! quads, a safeguarded Gauss-Newton iteration on each candidate's
//! `(u, v)` polishes the foot point -- walking across refined-quad
//! seams through the [`RefinementResult`] adjacency when the minimizer
//! pins to a quad edge or corner -- and an angle-weighted pseudonormal
//! signs the distance on closed surfaces.
//!
//! The query lives on [`LimitEvaluator`] itself rather than a wrapping
//! query struct: the acceleration index depends on exactly the
//! topology + positions pair the evaluator already binds, shares its
//! lifetime and invalidation story with the s2 isolation cache, and
//! the s4 SDF sampler and s5 amendment drag both hold an evaluator
//! anyway. The index is built lazily on the first query (a `OnceCell`
//! next to the isolation `RefCell`) and reused by every later query --
//! like the evaluator, cheap to build per thread but not `Sync`.
//!
//! # Acceleration: conservative per-quad AABBs
//!
//! One axis-aligned box per refined quad, each guaranteed to contain
//! the quad's entire limit patch, so box-distance pruning can never
//! cut off the true minimizer:
//!
//! - **Regular quads**: the box of the patch's 16 B-spline control
//!   points -- the bicubic basis is nonnegative and partitions unity,
//!   so the patch lies in the control points' convex hull.
//! - **Feature quads**: the box of the quad's *support submesh*
//!   vertices (the quad plus every face sharing a vertex with it, the
//!   exact point set s2's recursive isolation refines). Every
//!   Catmull-Clark refinement rule and every limit mask the isolation
//!   applies is a convex combination -- nonnegative weights summing to
//!   one -- and by the s2 support contract each isolation level's
//!   central children read only points derived from the previous
//!   level's support, so by induction the central quad's limit lies in
//!   the convex hull of the support cage, hence in its box.
//!
//! The boxes feed a median-split binary BVH (built once, O(n log n),
//! flat nodes). A BVH over patch boxes was chosen over a point
//! kd-tree/grid because the boxes bound the *continuous* patches --
//! pruning is exact rather than sample-resolution-limited -- and over
//! no index at all because the SDF sampler and amendment drags are
//! many-query consumers (design section 6).
//!
//! # Candidates -> Gauss-Newton, and the seam walk
//!
//! Best-first descent orders nodes by box distance in a min-heap and
//! stops when the nearest unvisited box is no closer than the best
//! foot point so far. Each surviving candidate quad runs a damped
//! Gauss-Newton minimization of `|S(u, v) - q|^2` from five starts
//! (center plus the four corners) using
//! [`eval_with_derivatives`](LimitEvaluator::eval_with_derivatives)'s
//! first derivatives -- the normal-equations step; the curvature term
//! of the true Hessian is unavailable through the s2 interface and
//! unnecessary, since the step-halving line search only ever accepts
//! strictly improving iterates. Steps are clamped to the unit square,
//! and when the full step fails the line search against a pinned
//! bound, the active-set reduction retries with the 1D Gauss-Newton
//! step along the free coordinate (quadratic along edge minimizers
//! where the clamped full step would creep linearly). When an
//! accepted step pins to a quad edge with the unclamped step
//! pointing outside, the iteration *transfers* across the seam to the
//! adjacent refined quad (winding-consistent parameter remap) and
//! continues, and a run that converges pinned to an edge or corner
//! seeds the adjacent quad -- the whole vertex fan at a corner -- so a
//! minimizer on a seam is polished from every side instead of being
//! accepted as a clamped one-sided local minimum. Strict-improvement
//! acceptance rules out non-improving walk cycles; hop and seed caps
//! are backstops only.
//!
//! *Feature lines* -- open boundaries and persistent sharp creases --
//! get special treatment: derivatives evaluated exactly on such a
//! line are depth-cap-degraded (the s2 module docs), so a run pinned
//! against one minimizes along the edge on positions only (a coarse
//! scan plus dyadic resolution doubling), continuing through endpoint
//! corners via the fan seeds; Gauss-Newton never iterates on degraded
//! on-line gradients.
//!
//! Because every evaluated point (all seeds and every accepted
//! iterate) updates the running best, the returned point is never
//! farther than the best brute-force candidate the search encountered.
//!
//! Near the medial axis the box pruning necessarily degrades (many
//! quads tie), so per-query dedup keeps the candidate cost flat:
//! corner starts run once per corner *vertex* (they are shared surface
//! points; skipped repeats are still evaluated and offered), a vertex
//! fan expands once, and a feature edge is slid along once.
//!
//! # Convergence tolerances
//!
//! A run stops when no halved step improves, when it exhausts
//! [`MAX_NEWTON_ITERATIONS`](self), or when an accepted step moves the
//! foot point by less than `1e-6` of the root box diagonal -- positions
//! are f32, so ~`1e-7` relative is the noise floor and the stop
//! criterion sits one decade above it. `(u, v)` quantizes to f32 at
//! evaluation, matching the s1/s2 interfaces.
//!
//! # Sign (design section 4-5): angle-weighted pseudonormal
//!
//! `signed_distance` is `Some` exactly when the refined topology is
//! closed (no boundary edge); open meshes keep the documented-unsigned
//! status quo. The sign tests `query - position` against:
//!
//! - **quad interior**: the sector-correct limit normal `du x dv`.
//! - **on a seam edge** (within `1e-4` in parameter): the two incident
//!   quads' unit normals, equal-weighted. Across a smooth seam they
//!   agree; across an infinitely sharp crease this is the classic
//!   two-face edge pseudonormal.
//! - **on a corner**: the corner vertex's full face fan, each face's
//!   sector normal weighted by its wedge angle (the angle between the
//!   one-sided derivatives along the quad's two corner edges) -- s2's
//!   corner snap supplies per-sector tangents at persistent features.
//!
//! Cone-point caveat: at a multi-sector pinned vertex the limit has no
//! convergent normal, so the fan normals entering the pseudonormal are
//! s2's deterministic per-sector tangent planes -- the sign is
//! deterministic but the pseudonormal is a fan aggregate, not a limit
//! of surface normals. Should the aggregate degenerate outright
//! (antipodal fan normals), the reported normal is zero and the sign
//! falls back to positive.

use std::cmp::Ordering;
use std::collections::{BTreeSet, BinaryHeap};

use crate::limit_eval::persistent_sharp_edge;
use crate::{KernelError, LimitEvaluator};

/// Faces per BVH leaf.
const LEAF_SIZE: usize = 4;
/// Gauss-Newton iterations per run (one seed, transfers included).
const MAX_NEWTON_ITERATIONS: usize = 32;
/// Seam transfers per run.
const MAX_SEAM_HOPS: usize = 8;
/// Seeds per candidate quad (five starts plus walk/fan seeds).
const MAX_SEEDS: usize = 24;
/// Step halvings per line search.
const MAX_HALVINGS: usize = 12;
/// Converged-step threshold relative to the root box diagonal.
const STEP_TOLERANCE_REL: f64 = 1e-6;
/// In-parameter snap classifying a foot point as on-edge/on-corner
/// for the pseudonormal (an interior minimizer this close to a seam
/// is normal-indistinguishable from the seam at f32 resolution).
const EDGE_SNAP: f64 = 1e-4;

/// In-quad `(u, v)` of CSR corner `k`, the `PatchTable` convention.
const CORNER_UV: [[f64; 2]; 4] = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];

/// One f64 limit sample: `(position, dp/du, dp/dv)`.
type SampleF64 = ([f64; 3], [f64; 3], [f64; 3]);

/// An accepted line-search step:
/// `(raw uv, clamped uv, position, dp/du, dp/dv, dist2)`.
type AcceptedStep = ([f64; 2], [f64; 2], [f64; 3], [f64; 3], [f64; 3], f64);

/// Result of [`LimitEvaluator::closest_point`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClosestPoint {
    /// The foot point on the limit surface.
    pub position: [f32; 3],
    /// The refined quad of the closest point.
    pub face: u32,
    /// In-quad `(u, v)` of the closest point
    /// (`position == eval(face, uv)`).
    pub uv: [f32; 2],
    /// Euclidean distance from the query to `position`.
    pub distance: f32,
    /// Signed variant of `distance` when sign is available (closed
    /// surface), negative inside (the anti-winding-normal side).
    pub signed_distance: Option<f32>,
    /// The limit normal at the closest point (sector-correct at
    /// creases): the surface normal in quad interiors, the
    /// angle-weighted pseudonormal on seam edges/corners. Unit length,
    /// or zero if it degenerates (see the module docs).
    pub normal: [f32; 3],
}

// -- The acceleration index --------------------------------------------------

/// Conservative axis-aligned box (f32 bounds are exact min/max of f32
/// points; distances are computed in f64).
#[derive(Debug, Clone, Copy)]
struct Aabb {
    min: [f32; 3],
    max: [f32; 3],
}

impl Aabb {
    const EMPTY: Self = Aabb {
        min: [f32::INFINITY; 3],
        max: [f32::NEG_INFINITY; 3],
    };

    fn add(&mut self, p: [f32; 3]) {
        for (c, &coord) in p.iter().enumerate() {
            self.min[c] = self.min[c].min(coord);
            self.max[c] = self.max[c].max(coord);
        }
    }

    fn union(self, other: Self) -> Self {
        Aabb {
            min: [
                self.min[0].min(other.min[0]),
                self.min[1].min(other.min[1]),
                self.min[2].min(other.min[2]),
            ],
            max: [
                self.max[0].max(other.max[0]),
                self.max[1].max(other.max[1]),
                self.max[2].max(other.max[2]),
            ],
        }
    }

    fn centroid(&self) -> [f64; 3] {
        [
            (self.min[0] as f64 + self.max[0] as f64) * 0.5,
            (self.min[1] as f64 + self.max[1] as f64) * 0.5,
            (self.min[2] as f64 + self.max[2] as f64) * 0.5,
        ]
    }

    /// Squared distance from `q` to the box (zero inside) -- a lower
    /// bound on the squared distance to anything the box bounds.
    fn distance2(&self, q: [f64; 3]) -> f64 {
        (0..3)
            .map(|c| {
                let t = q[c].clamp(self.min[c] as f64, self.max[c] as f64) - q[c];
                t * t
            })
            .sum()
    }

    fn diagonal(&self) -> f64 {
        (0..3)
            .map(|c| ((self.max[c] - self.min[c]) as f64).powi(2))
            .sum::<f64>()
            .sqrt()
    }
}

enum NodeKind {
    Internal {
        left: u32,
        right: u32,
    },
    /// Faces `order[start..start + len]`.
    Leaf {
        start: u32,
        len: u32,
    },
}

struct Node {
    aabb: Aabb,
    kind: NodeKind,
}

/// The per-evaluator closest-point index: a BVH over conservative
/// per-refined-quad boxes, plus the closed-surface flag and the
/// tolerance scale. Built once by [`LimitEvaluator::closest_point`]
/// and cached on the evaluator.
pub(crate) struct SearchIndex {
    nodes: Vec<Node>,
    root: u32,
    /// Refined-face indices permuted into leaf order.
    order: Vec<u32>,
    /// Conservative box per refined face (indexed by face).
    boxes: Vec<Aabb>,
    /// No boundary edge at the refined level: sign is available.
    closed: bool,
    /// Root box diagonal -- the physical tolerance scale.
    diag: f64,
}

fn build_index(evaluator: &LimitEvaluator) -> Result<SearchIndex, KernelError> {
    let mesh = &evaluator.result.topology;
    let adjacency = &evaluator.result.adjacency;
    let face_count = mesh.face_vertex_counts.len();
    if face_count == 0 {
        return Err(KernelError::InvalidTopology(
            "refined level has no faces to run closest-point queries against",
        ));
    }
    let boxes: Vec<Aabb> = (0..face_count as u32)
        .map(|face| face_box(evaluator, face))
        .collect();
    let centroids: Vec<[f64; 3]> = boxes.iter().map(Aabb::centroid).collect();
    let mut faces: Vec<u32> = (0..face_count as u32).collect();
    let mut nodes = Vec::new();
    let mut order = Vec::with_capacity(face_count);
    let root = build_node(&mut faces, &boxes, &centroids, &mut nodes, &mut order);
    let diag = nodes[root as usize].aabb.diagonal();
    Ok(SearchIndex {
        nodes,
        root,
        order,
        boxes,
        closed: !adjacency.edge_is_boundary.iter().any(|&b| b),
        diag,
    })
}

/// Conservative box of one refined quad's limit patch; see the module
/// docs for why each variant bounds the limit.
fn face_box(evaluator: &LimitEvaluator, face: u32) -> Aabb {
    let mut aabb = Aabb::EMPTY;
    match evaluator.table.face_patch(face) {
        Some(patch) => {
            for &cp in &evaluator.table.control_points[patch as usize] {
                aabb.add(evaluator.positions[cp as usize]);
            }
        }
        None => {
            let mesh = &evaluator.result.topology;
            let adjacency = &evaluator.result.adjacency;
            let off = (face * 4) as usize;
            let support: BTreeSet<u32> = mesh.face_vertex_indices[off..off + 4]
                .iter()
                .flat_map(|&corner| {
                    let start = adjacency.vertex_face_offsets[corner as usize] as usize;
                    let end = adjacency.vertex_face_offsets[corner as usize + 1] as usize;
                    adjacency.vertex_faces[start..end].iter().copied()
                })
                .collect();
            for f in support {
                for &v in &mesh.face_vertex_indices[(f * 4) as usize..(f * 4) as usize + 4] {
                    aabb.add(evaluator.positions[v as usize]);
                }
            }
        }
    }
    aabb
}

/// Median-split build over box centroids; returns the node index.
fn build_node(
    faces: &mut [u32],
    boxes: &[Aabb],
    centroids: &[[f64; 3]],
    nodes: &mut Vec<Node>,
    order: &mut Vec<u32>,
) -> u32 {
    let aabb = faces
        .iter()
        .fold(Aabb::EMPTY, |acc, &f| acc.union(boxes[f as usize]));
    let kind = if faces.len() <= LEAF_SIZE {
        let start = order.len() as u32;
        order.extend_from_slice(faces);
        NodeKind::Leaf {
            start,
            len: faces.len() as u32,
        }
    } else {
        // Split the centroid bounds on their widest axis.
        let (lo, hi) = faces.iter().fold(
            ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]),
            |(mut lo, mut hi), &f| {
                for c in 0..3 {
                    lo[c] = lo[c].min(centroids[f as usize][c]);
                    hi[c] = hi[c].max(centroids[f as usize][c]);
                }
                (lo, hi)
            },
        );
        let axis = (0..3).fold(0, |best, c| {
            if hi[c] - lo[c] > hi[best] - lo[best] {
                c
            } else {
                best
            }
        });
        let mid = faces.len() / 2;
        faces.select_nth_unstable_by(mid, |&a, &b| {
            centroids[a as usize][axis].total_cmp(&centroids[b as usize][axis])
        });
        let (left_faces, right_faces) = faces.split_at_mut(mid);
        let left = build_node(left_faces, boxes, centroids, nodes, order);
        let right = build_node(right_faces, boxes, centroids, nodes, order);
        NodeKind::Internal { left, right }
    };
    nodes.push(Node { aabb, kind });
    (nodes.len() - 1) as u32
}

/// Min-heap entry: smallest box distance pops first.
#[derive(Clone, Copy)]
struct HeapEntry {
    d2: f64,
    node: u32,
}

impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for HeapEntry {}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reversed: `BinaryHeap` is a max-heap, we pop the nearest box.
        other
            .d2
            .total_cmp(&self.d2)
            .then_with(|| other.node.cmp(&self.node))
    }
}

// -- The query ----------------------------------------------------------------

/// Running best foot point.
struct Best {
    dist2: f64,
    face: u32,
    uv: [f64; 2],
    position: [f64; 3],
}

impl LimitEvaluator<'_> {
    /// Closest point on the limit surface to `query`, signed on closed
    /// surfaces. See the module docs of `closest_point.rs` for the
    /// search, walk, tolerance, and sign semantics. The acceleration
    /// index is built on the first call and reused afterwards.
    pub fn closest_point(&self, query: [f32; 3]) -> Result<ClosestPoint, KernelError> {
        let index = self.search_index()?;
        let q = v3(query);
        let mut minimizer = Minimizer {
            evaluator: self,
            q,
            tolerance: STEP_TOLERANCE_REL * index.diag,
            best: Best {
                dist2: f64::INFINITY,
                face: 0,
                uv: [0.0; 2],
                position: [0.0; 3],
            },
            seeds: Vec::new(),
            seeded: Vec::new(),
            corner_runs: Vec::new(),
            fanned: Vec::new(),
            polished: Vec::new(),
        };
        let mut heap = BinaryHeap::new();
        heap.push(HeapEntry {
            d2: index.nodes[index.root as usize].aabb.distance2(q),
            node: index.root,
        });
        while let Some(entry) = heap.pop() {
            if entry.d2 >= minimizer.best.dist2 {
                // Min-heap order: every remaining box is farther still.
                break;
            }
            match index.nodes[entry.node as usize].kind {
                NodeKind::Internal { left, right } => {
                    for child in [left, right] {
                        let d2 = index.nodes[child as usize].aabb.distance2(q);
                        if d2 < minimizer.best.dist2 {
                            heap.push(HeapEntry { d2, node: child });
                        }
                    }
                }
                NodeKind::Leaf { start, len } => {
                    for &face in &index.order[start as usize..(start + len) as usize] {
                        if index.boxes[face as usize].distance2(q) < minimizer.best.dist2 {
                            minimizer.search_quad(face)?;
                        }
                    }
                }
            }
        }
        self.finish(q, index, minimizer.best)
    }

    /// The cached index, built on first use.
    fn search_index(&self) -> Result<&SearchIndex, KernelError> {
        if self.search.get().is_none() {
            // Not Sync, so no concurrent set; an existing value is
            // impossible on this path.
            let _ = self.search.set(build_index(self)?);
        }
        // SAFETY: populated just above when it was empty.
        Ok(self.search.get().expect("search index populated"))
    }

    /// f64 view of [`eval_with_derivatives`](Self::eval_with_derivatives).
    fn sample(&self, face: u32, uv: [f64; 2]) -> Result<SampleF64, KernelError> {
        let (p, du, dv) = self.eval_with_derivatives(face, [uv[0] as f32, uv[1] as f32])?;
        Ok((v3(p), v3(du), v3(dv)))
    }

    /// Whether the edge at `slot` of `face` is a *feature line* --
    /// open boundary or persistent sharp crease -- where exactly-on-
    /// line derivatives are depth-cap-degraded (the s2 module docs):
    /// the walk minimizes along such an edge on positions only
    /// instead of Gauss-Newton-ing on it.
    fn feature_line(&self, face: u32, slot: usize) -> bool {
        let adjacency = &self.result.adjacency;
        let edge = adjacency.face_edges[face as usize * 4 + slot] as usize;
        adjacency.edge_is_boundary[edge]
            || persistent_sharp_edge(
                self.result.topology.edge_creases[edge],
                &self.result.options,
            )
    }

    /// The neighbor across edge slot `slot` of `face`, with the
    /// in-edge parameter `x` remapped into the neighbor's frame
    /// (winding-consistent meshes traverse a shared edge oppositely).
    /// `None` on boundary edges or inconsistent adjacency.
    fn across_edge(&self, face: u32, slot: usize, x: f64) -> Option<(u32, [f64; 2])> {
        let adjacency = &self.result.adjacency;
        let edge = adjacency.face_edges[face as usize * 4 + slot];
        let [fa, fb] = adjacency.edge_faces[edge as usize];
        let neighbor = if fa == face { fb } else { fa };
        (neighbor != u32::MAX)
            .then(|| {
                adjacency.face_edges[neighbor as usize * 4..neighbor as usize * 4 + 4]
                    .iter()
                    .position(|&e| e == edge)
                    .map(|t| (neighbor, edge_uv(t, 1.0 - x)))
            })
            .flatten()
    }

    /// Package the best foot point: pseudonormal, sign, f32 narrowing.
    fn finish(
        &self,
        q: [f64; 3],
        index: &SearchIndex,
        best: Best,
    ) -> Result<ClosestPoint, KernelError> {
        let distance = best.dist2.sqrt() as f32;
        let pseudo = self.pseudonormal(best.face, best.uv)?;
        let len = length(pseudo);
        let normal = if len > 1e-12 {
            scale(pseudo, 1.0 / len)
        } else {
            [0.0; 3]
        };
        let signed_distance = index.closed.then(|| {
            if dot(sub(q, best.position), normal) < 0.0 {
                -distance
            } else {
                distance
            }
        });
        Ok(ClosestPoint {
            position: [
                best.position[0] as f32,
                best.position[1] as f32,
                best.position[2] as f32,
            ],
            face: best.face,
            uv: [best.uv[0] as f32, best.uv[1] as f32],
            distance,
            signed_distance,
            normal: [normal[0] as f32, normal[1] as f32, normal[2] as f32],
        })
    }

    /// Unnormalized pseudonormal at a foot point, classified by the
    /// [`EDGE_SNAP`] parameter snap; see the module docs.
    fn pseudonormal(&self, face: u32, uv: [f64; 2]) -> Result<[f64; 3], KernelError> {
        let pin = |t: f64| {
            if t <= EDGE_SNAP {
                Some(false)
            } else if t >= 1.0 - EDGE_SNAP {
                Some(true)
            } else {
                None
            }
        };
        match (pin(uv[0]), pin(uv[1])) {
            (None, None) => self.face_normal(face, uv),
            (Some(u_hi), Some(v_hi)) => {
                let corner = match (u_hi, v_hi) {
                    (false, false) => 0,
                    (true, false) => 1,
                    (true, true) => 2,
                    (false, true) => 3,
                };
                self.corner_pseudonormal(face, corner)
            }
            (u_pin, v_pin) => {
                // Exactly one coordinate pinned: an on-edge foot point.
                let (slot, x) = match (u_pin, v_pin) {
                    (None, Some(false)) => (0, uv[0]),
                    (Some(true), None) => (1, uv[1]),
                    (None, Some(true)) => (2, 1.0 - uv[0]),
                    _ => (3, 1.0 - uv[1]),
                };
                let own = self.face_normal(face, edge_uv(slot, x))?;
                let other = match self.across_edge(face, slot, x) {
                    Some((neighbor, neighbor_uv)) => self.face_normal(neighbor, neighbor_uv)?,
                    None => [0.0; 3],
                };
                Ok(add(own, other))
            }
        }
    }

    /// Unit surface normal `du x dv`, or zero where it degenerates.
    fn face_normal(&self, face: u32, uv: [f64; 2]) -> Result<[f64; 3], KernelError> {
        let (_, du, dv) = self.sample(face, uv)?;
        Ok(normalize_or_zero(cross(du, dv)))
    }

    /// Angle-weighted pseudonormal over the full face fan of the quad
    /// corner's vertex: per incident face, the sector normal weighted
    /// by the wedge angle between the one-sided derivatives along the
    /// quad's two corner edges.
    fn corner_pseudonormal(&self, face: u32, corner: usize) -> Result<[f64; 3], KernelError> {
        let mesh = &self.result.topology;
        let adjacency = &self.result.adjacency;
        let vi = mesh.face_vertex_indices[face as usize * 4 + corner];
        let start = adjacency.vertex_face_offsets[vi as usize] as usize;
        let end = adjacency.vertex_face_offsets[vi as usize + 1] as usize;
        let mut acc = [0.0; 3];
        for &fan_face in &adjacency.vertex_faces[start..end] {
            let off = fan_face as usize * 4;
            let k = mesh.face_vertex_indices[off..off + 4]
                .iter()
                .position(|&c| c == vi)
                .ok_or(KernelError::InvalidTopology(
                    "vertex fan face does not contain the fan vertex",
                ))?;
            let (_, du, dv) = self.sample(fan_face, CORNER_UV[k])?;
            // Wedge directions toward the next/previous CSR corners.
            let (toward_next, toward_prev) = match k {
                0 => (du, dv),
                1 => (dv, neg(du)),
                2 => (neg(du), neg(dv)),
                _ => (neg(dv), du),
            };
            let normal = normalize_or_zero(cross(du, dv));
            acc = add(acc, scale(normal, wedge_angle(toward_next, toward_prev)));
        }
        Ok(acc)
    }
}

/// Per-query Gauss-Newton state: the running best plus the seed queue
/// of the candidate quad being searched.
struct Minimizer<'e, 'a> {
    evaluator: &'e LimitEvaluator<'a>,
    q: [f64; 3],
    /// Physical converged-step threshold.
    tolerance: f64,
    best: Best,
    /// Pending `(face, uv)` seeds of the current candidate quad.
    seeds: Vec<(u32, [f64; 2])>,
    /// `(face, uv bits)` starts already seeded for the current
    /// candidate quad -- keyed by the exact start, so a fan seed at a
    /// different corner of an already-seeded face still runs.
    seeded: Vec<(u32, [u64; 2])>,
    /// Corner vertices already Newton-run this query. Corner seeds are
    /// shared surface points between incident quads; near the medial
    /// axis (where box pruning cannot cut candidates) running each one
    /// once instead of once per quad is the difference between ~2 and
    /// 5 runs per candidate. Skipped seeds are still evaluated and
    /// offered, so the brute-candidate guarantee is untouched.
    corner_runs: Vec<u32>,
    /// Corner vertices whose full fan was already seeded this query
    /// (the fan starts are identical wherever they are triggered from).
    fanned: Vec<u32>,
    /// Feature edges (by refined edge index) already slid along this
    /// query: every seed converging onto the same boundary or crease
    /// edge -- from either side -- shares one polish.
    polished: Vec<u32>,
}

impl Minimizer<'_, '_> {
    /// Multi-start search of one candidate quad: center + corners,
    /// plus whatever walk/fan seeds the runs enqueue. Corner starts
    /// whose vertex already ran this query are evaluated and offered
    /// but not re-run (see [`corner_runs`](Self::corner_runs)).
    fn search_quad(&mut self, face: u32) -> Result<(), KernelError> {
        self.seeds.clear();
        self.seeded.clear();
        self.enqueue(face, [0.5, 0.5]);
        let off = face as usize * 4;
        for (k, &uv) in CORNER_UV.iter().enumerate() {
            let vertex = self.evaluator.result.topology.face_vertex_indices[off + k];
            if self.corner_runs.contains(&vertex) {
                let (p, _, _) = self.evaluator.sample(face, uv)?;
                let dist2 = norm2(sub(p, self.q));
                self.offer(face, uv, p, dist2);
            } else {
                self.corner_runs.push(vertex);
                self.enqueue(face, uv);
            }
        }
        let mut next = 0;
        while next < self.seeds.len() {
            let (seed_face, seed_uv) = self.seeds[next];
            next += 1;
            self.run(seed_face, seed_uv)?;
        }
        Ok(())
    }

    fn offer(&mut self, face: u32, uv: [f64; 2], position: [f64; 3], dist2: f64) {
        if dist2 < self.best.dist2 {
            self.best = Best {
                dist2,
                face,
                uv,
                position,
            };
        }
    }

    /// One damped Gauss-Newton run with in-run seam transfers; pinned
    /// convergence seeds the across-seam neighbors.
    fn run(&mut self, face: u32, uv: [f64; 2]) -> Result<(), KernelError> {
        let (mut face, mut uv) = (face, uv);
        let (mut p, mut du, mut dv) = self.evaluator.sample(face, uv)?;
        let mut dist2 = norm2(sub(p, self.q));
        self.offer(face, uv, p, dist2);
        let mut hops = 0;
        for _ in 0..MAX_NEWTON_ITERATIONS {
            let d = sub(p, self.q);
            let full = gauss_newton_step(d, du, dv);
            // The full step first; when it fails against a pinned
            // bound, the reduced active-set step (1D Newton along the
            // free coordinate) -- the clamped full step would only
            // creep linearly along an edge minimizer.
            let mut accepted = self.line_search(face, uv, full, dist2)?;
            if accepted.is_none()
                && let Some(reduced) = reduced_step(uv, full, d, du, dv)
            {
                accepted = self.line_search(face, uv, reduced, dist2)?;
            }
            let Some((raw, cand, cp, cdu, cdv, cd2)) = accepted else {
                // Local (possibly constrained) minimum: slide along
                // any pinned feature edge (Gauss-Newton cannot -- see
                // `polish_feature_edge`) and polish any pinned seam
                // from the other side too.
                self.polish_pinned_feature_edges(face, uv)?;
                self.enqueue_pinned(face, uv);
                break;
            };
            let moved2 = norm2(sub(cp, p));
            (uv, p, du, dv, dist2) = (cand, cp, cdu, cdv, cd2);
            self.offer(face, uv, p, dist2);
            if moved2 <= self.tolerance * self.tolerance {
                self.polish_pinned_feature_edges(face, uv)?;
                self.enqueue_pinned(face, uv);
                break;
            }
            // Transfer across the seam when the step was clamped at an
            // edge; strict improvement above rules out ping-pong.
            let pin_u = pinned_crossing(uv[0], raw[0]);
            let pin_v = pinned_crossing(uv[1], raw[1]);
            match (pin_u, pin_v) {
                (Some(_), Some(_)) => {
                    self.enqueue_corner_fan(face, uv);
                    break;
                }
                (Some(side), None) | (None, Some(side)) => {
                    let (slot, x) = if pin_u.is_some() {
                        if side { (1, uv[1]) } else { (3, 1.0 - uv[1]) }
                    } else if side {
                        (2, 1.0 - uv[0])
                    } else {
                        (0, uv[0])
                    };
                    if self.evaluator.feature_line(face, slot) {
                        // Boundary or sharp crease: Gauss-Newton on
                        // the line is derivative-degraded -- slide on
                        // positions only. No across-seed: a foot
                        // beyond the crease lives in the neighbor's
                        // interior, which stays an unpruned BVH
                        // candidate of its own, and an on-line seed
                        // would only grind deep snapped evals.
                        self.polish_feature_edge(face, slot)?;
                        break;
                    }
                    hops += 1;
                    if hops > MAX_SEAM_HOPS {
                        break;
                    }
                    match self.evaluator.across_edge(face, slot, x) {
                        Some((neighbor, neighbor_uv)) => {
                            (face, uv) = (neighbor, neighbor_uv);
                            (p, du, dv) = self.evaluator.sample(face, uv)?;
                            dist2 = norm2(sub(p, self.q));
                            self.offer(face, uv, p, dist2);
                        }
                        // Unreachable for manifold seams; stand pat.
                        None => break,
                    }
                }
                (None, None) => {}
            }
        }
        Ok(())
    }

    /// [`polish_feature_edge`](Self::polish_feature_edge) for every
    /// feature edge the converged point pins to (both incident edges
    /// at a pinned corner).
    fn polish_pinned_feature_edges(&mut self, face: u32, uv: [f64; 2]) -> Result<(), KernelError> {
        let pin = |t: f64| (t == 0.0).then_some(false).or((t == 1.0).then_some(true));
        let slots: &[usize] = match (pin(uv[0]), pin(uv[1])) {
            (Some(false), Some(false)) => &[3, 0],
            (Some(true), Some(false)) => &[0, 1],
            (Some(true), Some(true)) => &[1, 2],
            (Some(false), Some(true)) => &[2, 3],
            (Some(false), None) => &[3],
            (Some(true), None) => &[1],
            (None, Some(false)) => &[0],
            (None, Some(true)) => &[2],
            (None, None) => &[],
        };
        for &slot in slots {
            if self.evaluator.feature_line(face, slot) {
                self.polish_feature_edge(face, slot)?;
            }
        }
        Ok(())
    }

    /// Constrained 1D minimization along feature edge `slot` of
    /// `face`, on positions only: *derivatives* exactly on a feature
    /// line are depth-cap-degraded (the s2 module docs), so
    /// Gauss-Newton cannot slide along a boundary or sharp-crease
    /// minimizer. A coarse scan plus resolution-doubling descent on
    /// *dyadic* parameters replaces it -- a dyadic `x` with `k` bits
    /// snaps as an exact depth-`k` corner in the s2 isolation, so the
    /// probes stay shallow and share ancestor nodes, where arbitrary
    /// `x` would build a fresh chain to the depth cap each. An
    /// endpoint minimizer seeds the corner fan so the slide continues
    /// into the next quad along the feature.
    fn polish_feature_edge(&mut self, face: u32, slot: usize) -> Result<(), KernelError> {
        let edge = self.evaluator.result.adjacency.face_edges[face as usize * 4 + slot];
        if self.polished.contains(&edge) {
            return Ok(());
        }
        self.polished.push(edge);
        // Depth-3 scan (the distance along one feature segment need
        // not be unimodal over [0, 1])...
        let mut best = (f64::INFINITY, 0.0f64);
        let mut step = 1.0 / 8.0;
        for k in 0..=8 {
            let x = k as f64 * step;
            let d2 = self.edge_distance2(face, slot, x)?;
            if d2 < best.0 {
                best = (d2, x);
            }
        }
        // ...then double the dyadic resolution around the running
        // argmin down to 2^-18 of the edge (well under the foot-point
        // tolerances; deeper would out-resolve f32 positions anyway).
        for _ in 0..15 {
            step *= 0.5;
            for x in [best.1 - step, best.1 + step] {
                if (0.0..=1.0).contains(&x) {
                    let d2 = self.edge_distance2(face, slot, x)?;
                    if d2 < best.0 {
                        best = (d2, x);
                    }
                }
            }
        }
        if best.1 <= 1e-3 {
            self.enqueue_corner_fan(face, edge_uv(slot, 0.0));
        } else if best.1 >= 1.0 - 1e-3 {
            self.enqueue_corner_fan(face, edge_uv(slot, 1.0));
        }
        Ok(())
    }

    /// One on-line probe of [`polish_feature_edge`]: squared distance
    /// at parameter `x` along edge `slot`, offered to the running best.
    fn edge_distance2(&mut self, face: u32, slot: usize, x: f64) -> Result<f64, KernelError> {
        let uv = edge_uv(slot, x);
        let (p, _, _) = self.evaluator.sample(face, uv)?;
        let d2 = norm2(sub(p, self.q));
        self.offer(face, uv, p, d2);
        Ok(d2)
    }

    /// Clamped backtracking line search: the first halving that
    /// strictly improves, as `(raw, clamped, position, du, dv, dist2)`.
    fn line_search(
        &self,
        face: u32,
        uv: [f64; 2],
        step: [f64; 2],
        dist2: f64,
    ) -> Result<Option<AcceptedStep>, KernelError> {
        let mut alpha = 1.0;
        for _ in 0..MAX_HALVINGS {
            let raw = [uv[0] + alpha * step[0], uv[1] + alpha * step[1]];
            let cand = [raw[0].clamp(0.0, 1.0), raw[1].clamp(0.0, 1.0)];
            if cand != uv {
                let (cp, cdu, cdv) = self.evaluator.sample(face, cand)?;
                let cd2 = norm2(sub(cp, self.q));
                if cd2 < dist2 {
                    return Ok(Some((raw, cand, cp, cdu, cdv, cd2)));
                }
            }
            alpha *= 0.5;
        }
        Ok(None)
    }

    /// Seed the neighbors of an exactly pinned converged point: the
    /// whole vertex fan at a corner, the across-edge quad on a smooth
    /// seam (feature lines are polished instead -- an on-line seed
    /// would start on degraded derivatives).
    fn enqueue_pinned(&mut self, face: u32, uv: [f64; 2]) {
        let pin = |t: f64| (t == 0.0).then_some(false).or((t == 1.0).then_some(true));
        match (pin(uv[0]), pin(uv[1])) {
            (Some(_), Some(_)) => self.enqueue_corner_fan(face, uv),
            (u_pin @ Some(side), None) | (u_pin @ None, Some(side)) => {
                let (slot, x) = if u_pin.is_some() {
                    if side { (1, uv[1]) } else { (3, 1.0 - uv[1]) }
                } else if side {
                    (2, 1.0 - uv[0])
                } else {
                    (0, uv[0])
                };
                if !self.evaluator.feature_line(face, slot)
                    && let Some((neighbor, neighbor_uv)) = self.evaluator.across_edge(face, slot, x)
                {
                    self.enqueue(neighbor, neighbor_uv);
                }
            }
            (None, None) => {}
        }
    }

    /// Seed every face around the corner vertex at its own corner
    /// parameter (the corner walk: the true minimizer may live in any
    /// fan face, including across the diagonal). Each vertex fans at
    /// most once per query -- the starts are identical wherever the
    /// fan is triggered from.
    fn enqueue_corner_fan(&mut self, face: u32, uv: [f64; 2]) {
        let corner = match (uv[0] == 1.0, uv[1] == 1.0) {
            (false, false) => 0,
            (true, false) => 1,
            (true, true) => 2,
            (false, true) => 3,
        };
        let mesh = &self.evaluator.result.topology;
        let adjacency = &self.evaluator.result.adjacency;
        let vi = mesh.face_vertex_indices[face as usize * 4 + corner];
        if self.fanned.contains(&vi) {
            return;
        }
        self.fanned.push(vi);
        let start = adjacency.vertex_face_offsets[vi as usize] as usize;
        let end = adjacency.vertex_face_offsets[vi as usize + 1] as usize;
        for &fan_face in &adjacency.vertex_faces[start..end] {
            let off = fan_face as usize * 4;
            if let Some(k) = mesh.face_vertex_indices[off..off + 4]
                .iter()
                .position(|&c| c == vi)
            {
                self.enqueue(fan_face, CORNER_UV[k]);
            }
        }
    }

    fn enqueue(&mut self, face: u32, uv: [f64; 2]) {
        let key = (face, [uv[0].to_bits(), uv[1].to_bits()]);
        if self.seeds.len() < MAX_SEEDS && !self.seeded.contains(&key) {
            self.seeded.push(key);
            self.seeds.push((face, uv));
        }
    }
}

/// Gauss-Newton step for `|S - q|^2/2` from the first-derivative
/// normal equations; falls back to scaled steepest descent when the
/// tangent frame degenerates (sector-plane corner tangents,
/// crease-crease corners) -- the line search safeguards either way.
fn gauss_newton_step(d: [f64; 3], du: [f64; 3], dv: [f64; 3]) -> [f64; 2] {
    let (a, b, c) = (dot(du, du), dot(dv, dv), dot(du, dv));
    let (gu, gv) = (dot(d, du), dot(d, dv));
    let det = a * b - c * c;
    if det > 1e-12 * a * b {
        [(c * gv - b * gu) / det, (c * gu - a * gv) / det]
    } else {
        let scale = (a + b).max(1e-30);
        [-gu / scale, -gv / scale]
    }
}

/// Active-set reduction when the full step pushes outward through
/// exactly one pinned bound: 1D Gauss-Newton along the free
/// coordinate (quadratic along an edge minimizer where the clamped
/// full step would creep linearly). `None` when nothing is pinned
/// outward, the frame degenerates, or both bounds pin (a corner --
/// the fan seeding owns that case).
fn reduced_step(
    uv: [f64; 2],
    full: [f64; 2],
    d: [f64; 3],
    du: [f64; 3],
    dv: [f64; 3],
) -> Option<[f64; 2]> {
    let outward = |t: f64, s: f64| (t == 0.0 && s < 0.0) || (t == 1.0 && s > 0.0);
    let one_d = |tangent: [f64; 3]| {
        let scale = dot(tangent, tangent);
        (scale > 1e-30).then(|| -dot(d, tangent) / scale)
    };
    match (outward(uv[0], full[0]), outward(uv[1], full[1])) {
        (true, false) => one_d(dv).map(|s| [0.0, s]),
        (false, true) => one_d(du).map(|s| [s, 0.0]),
        _ => None,
    }
}

/// Whether a clamped coordinate is pinned at 0/1 with the raw step
/// strictly outside (the seam-crossing test).
fn pinned_crossing(clamped: f64, raw: f64) -> Option<bool> {
    (clamped == 0.0 && raw < 0.0)
        .then_some(false)
        .or((clamped == 1.0 && raw > 1.0).then_some(true))
}

/// `(u, v)` at parameter `x` along CSR edge slot `slot` (corner `slot`
/// toward corner `slot + 1`).
fn edge_uv(slot: usize, x: f64) -> [f64; 2] {
    match slot {
        0 => [x, 0.0],
        1 => [1.0, x],
        2 => [1.0 - x, 1.0],
        _ => [0.0, 1.0 - x],
    }
}

/// Angle between two wedge tangents; a neutral right angle when either
/// degenerates.
fn wedge_angle(a: [f64; 3], b: [f64; 3]) -> f64 {
    let (la, lb) = (length(a), length(b));
    if la > 1e-12 && lb > 1e-12 {
        (dot(a, b) / (la * lb)).clamp(-1.0, 1.0).acos()
    } else {
        std::f64::consts::FRAC_PI_2
    }
}

// -- f64 vector helpers --------------------------------------------------------

fn v3(p: [f32; 3]) -> [f64; 3] {
    [p[0] as f64, p[1] as f64, p[2] as f64]
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn neg(a: [f64; 3]) -> [f64; 3] {
    [-a[0], -a[1], -a[2]]
}

fn scale(a: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn norm2(a: [f64; 3]) -> f64 {
    dot(a, a)
}

fn length(a: [f64; 3]) -> f64 {
    norm2(a).sqrt()
}

fn normalize_or_zero(a: [f64; 3]) -> [f64; 3] {
    let len = length(a);
    if len > 1e-12 {
        scale(a, 1.0 / len)
    } else {
        [0.0; 3]
    }
}

#[cfg(test)]
mod tests {
    use super::{Aabb, edge_uv, pinned_crossing};

    /// The geometry gates live in `tests/closest_point.rs`; these unit
    /// tests pin the pure helpers.
    #[test]
    fn aabb_distance_is_zero_inside_and_exact_outside() {
        let mut aabb = Aabb::EMPTY;
        aabb.add([0.0, 0.0, 0.0]);
        aabb.add([2.0, 1.0, 3.0]);
        assert_eq!(aabb.distance2([1.0, 0.5, 1.5]), 0.0);
        assert_eq!(aabb.distance2([3.0, 0.5, 1.5]), 1.0);
        assert_eq!(aabb.distance2([-1.0, -1.0, 4.0]), 3.0);
    }

    #[test]
    fn edge_parameterizations_traverse_csr_corners() {
        // Slot `s` runs corner `s` -> corner `s + 1`.
        assert_eq!(edge_uv(0, 0.0), [0.0, 0.0]);
        assert_eq!(edge_uv(0, 1.0), [1.0, 0.0]);
        assert_eq!(edge_uv(1, 0.25), [1.0, 0.25]);
        assert_eq!(edge_uv(2, 0.25), [0.75, 1.0]);
        assert_eq!(edge_uv(3, 0.25), [0.0, 0.75]);
    }

    #[test]
    fn pinned_crossing_requires_clamp_and_overshoot() {
        assert_eq!(pinned_crossing(0.0, -0.5), Some(false));
        assert_eq!(pinned_crossing(1.0, 1.5), Some(true));
        // Pinned but moving along the edge: no crossing.
        assert_eq!(pinned_crossing(0.0, 0.0), None);
        assert_eq!(pinned_crossing(0.5, 0.5), None);
    }
}
