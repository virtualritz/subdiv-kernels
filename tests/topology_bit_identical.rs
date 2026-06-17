//! Characterization guard: a bit-exact fingerprint of refined output (topology
//! + adjacency + per-level stencils + lineage) for every scheme. Pins behaviour
//! so the `edge_key_to_index` / `source_creases` storage swap (BTreeMap ->
//! FxHashMap) is proven output-identical, and guards future accidental changes.

use std::num::NonZeroU8;
use subdiv_kernels::{
    RefinementResult, Refiner, Scheme, SchemeOptions, Mesh, UniformRefine, VertexOrigin,
};

/// Order-sensitive FNV-1a/64 over the raw bytes of the refined result.
struct Fnv(u64);
impl Fnv {
    fn new() -> Self {
        Self(0xcbf2_9ce4_8422_2325)
    }
    fn byte(&mut self, b: u8) {
        self.0 ^= b as u64;
        self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
    }
    fn u32(&mut self, v: u32) {
        v.to_le_bytes().iter().for_each(|&b| self.byte(b));
    }
    fn u32s(&mut self, s: &[u32]) {
        self.u32(s.len() as u32);
        s.iter().for_each(|&v| self.u32(v));
    }
    fn f32s(&mut self, s: &[f32]) {
        self.u32(s.len() as u32);
        s.iter().for_each(|&v| self.u32(v.to_bits()));
    }
    fn finish(&self) -> u64 {
        self.0
    }
}

fn checksum(r: &RefinementResult) -> u64 {
    let mut h = Fnv::new();

    let t = &r.topology;
    h.u32(t.vertex_count);
    h.u32s(&t.face_vertex_counts);
    h.u32s(&t.face_vertex_indices);
    h.u32(t.edge_vertices.len() as u32);
    t.edge_vertices.iter().for_each(|e| {
        h.u32(e[0]);
        h.u32(e[1]);
    });
    h.f32s(&t.edge_creases);
    h.f32s(&t.vertex_corners);

    let a = &r.adjacency;
    h.u32s(&a.face_edges);
    h.u32(a.edge_faces.len() as u32);
    a.edge_faces.iter().for_each(|e| {
        h.u32(e[0]);
        h.u32(e[1]);
    });
    h.u32s(&a.vert_edge_offsets);
    h.u32s(&a.vert_edges);
    h.u32s(&a.vert_face_offsets);
    h.u32s(&a.vert_faces);
    a.edge_is_boundary.iter().for_each(|&b| h.byte(b as u8));
    a.vertex_is_boundary.iter().for_each(|&b| h.byte(b as u8));

    h.u32(r.level_stencils.len() as u32);
    r.level_stencils.iter().for_each(|s| {
        h.u32s(&s.offsets);
        h.u32s(&s.indices);
        h.f32s(&s.weights);
    });

    r.lineage.vertex_origin.iter().for_each(|o| match o {
        VertexOrigin::Vertex(i) => {
            h.byte(0);
            h.u32(*i);
        }
        VertexOrigin::Edge(i) => {
            h.byte(1);
            h.u32(*i);
        }
        VertexOrigin::Face(i) => {
            h.byte(2);
            h.u32(*i);
        }
    });
    h.u32s(&r.lineage.face_parent);
    h.u32s(&r.lineage.edge_parent);

    h.finish()
}

fn edges_from_faces(counts: &[u32], indices: &[u32]) -> Vec<[u32; 2]> {
    let mut seen = std::collections::BTreeSet::new();
    let mut edges = Vec::new();
    let mut cursor = 0usize;
    for &count in counts {
        let n = count as usize;
        for i in 0..n {
            let (a, b) = (indices[cursor + i], indices[cursor + (i + 1) % n]);
            let k = if a <= b { (a, b) } else { (b, a) };
            if seen.insert(k) {
                edges.push([k.0, k.1]);
            }
        }
        cursor += n;
    }
    edges
}

fn mesh(
    vertex_count: u32,
    counts: Vec<u32>,
    indices: Vec<u32>,
    creased_edges: usize,
) -> Mesh {
    let edge_vertices = edges_from_faces(&counts, &indices);
    let mut edge_creases = vec![0.0; edge_vertices.len()];
    (0..creased_edges.min(edge_creases.len())).for_each(|i| edge_creases[i] = 2.0);
    Mesh {
        vertex_count,
        face_vertex_counts: counts,
        face_vertex_indices: indices,
        edge_creases,
        edge_vertices,
        vertex_corners: vec![0.0; vertex_count as usize],
    }
}

fn cube(creased_edges: usize) -> Mesh {
    mesh(
        8,
        vec![4; 6],
        vec![
            0, 1, 3, 2, 2, 3, 5, 4, 4, 5, 7, 6, 6, 7, 1, 0, 1, 7, 5, 3, 6, 0, 2, 4,
        ],
        creased_edges,
    )
}

fn tetra() -> Mesh {
    mesh(4, vec![3; 4], vec![0, 1, 2, 0, 2, 3, 0, 3, 1, 1, 3, 2], 0)
}

fn refine(topo: Mesh, scheme: Scheme, level: u8) -> RefinementResult {
    Refiner::new(topo, scheme, SchemeOptions::default())
        .expect("refiner")
        .refine_uniform(&UniformRefine::from(NonZeroU8::new(level).unwrap()))
        .expect("refine")
}

#[test]
fn refined_output_is_bit_identical() {
    // (label, computed checksum, golden). Golden = 0 is a placeholder; the first
    // run prints the real values to paste in.
    let cases: [(&str, u64, u64); 5] = [
        (
            "cc_cube_l3",
            checksum(&refine(cube(0), Scheme::CatmullClark, 3)),
            0x9225ea1c9d8607df,
        ),
        (
            "cc_creased_cube_l2",
            checksum(&refine(cube(3), Scheme::CatmullClark, 2)),
            0x779f502ec2cea43b,
        ),
        (
            "doosabin_cube_l2",
            checksum(&refine(cube(0), Scheme::DooSabin, 2)),
            0xd75dac856853958d,
        ),
        (
            "loop_tetra_l3",
            checksum(&refine(tetra(), Scheme::Loop, 3)),
            0x1b232b8359e2e283,
        ),
        (
            "sqrt3_tetra_l2",
            checksum(&refine(tetra(), Scheme::Sqrt3, 2)),
            0x13d28b2d117a438b,
        ),
    ];

    let mut mismatches = Vec::new();
    for (label, got, golden) in cases {
        eprintln!("{label}: 0x{got:016x}");
        if got != golden {
            mismatches.push(format!("{label}: got 0x{got:016x}, golden 0x{golden:016x}"));
        }
    }
    assert!(
        mismatches.is_empty(),
        "checksum mismatches:\n{}",
        mismatches.join("\n")
    );
}
