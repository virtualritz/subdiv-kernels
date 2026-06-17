# `subdiv-kernels`

Standalone subdivision surface kernel crate. Topology-first, stencil-based,
generic over data types.

## Design

Topology is refined separately from data interpolation. The kernel produces
refined topology + per-level stencil tables. Stencil tables can then
interpolate any user type that implements `Interpolatable`.

```text
TopologyMesh ──► Refiner ──► RefinementResult
                                 │
                  positions ─────┼──► result.interpolate(&positions)
                  uvs ───────────┼──► result.interpolate(&uvs)
                  colors ────────┼──► result.interpolate(&colors)
                  custom ────────┘──► result.interpolate(&custom)
```

### Performance model

- **One-shot subdivision**: `result.interpolate()` chains per-level stencil
  application. Same algorithmic cost as direct subdivision — no exponential
  stencil growth. Each level's stencils are small (4-8 entries per vertex).

- **Animation (static topology, changing data)**: `result.compose_stencils()`
  precomputes a single stencil table mapping original vertices directly to
  final refined vertices. One `StencilTable::interpolate()` call per frame.
  Composition is O(output x entries^2) but amortized over many frames.

- **Multiple buffers**: `interpolate()` can be called once per buffer — all
  share the same topology computation.

- **Generic over T**: any type implementing `Interpolatable` (a single
  `add_with_weight` method + `Default + Clone`) can be subdivided. Positions,
  UVs, colors, normals, scalar weights, custom attributes.

## Usage

```rust
use subdiv_kernels::{Refiner, Scheme, SchemeOptions, TopologyMesh, UniformRefine};
use core::num::NonZeroU8;

let topology = TopologyMesh {
    vertex_count: 4,
    face_vertex_counts: vec![4],
    face_vertex_indices: vec![0, 1, 2, 3],
    edge_vertices: vec![[0, 1], [1, 2], [2, 3], [0, 3]],
    edge_creases: vec![0.0; 4],
    vertex_corners: vec![0.0; 4],
};

let positions: Vec<[f32; 3]> = vec![
    [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
    [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
];

let refiner = Refiner::new(topology, Scheme::CatmullClark, SchemeOptions::default())
    .expect("valid topology");

// SAFETY: 2 is non-zero.
let req = UniformRefine::from(NonZeroU8::new(2).unwrap());
let result = refiner.refine_uniform(&req).expect("refinement succeeds");

// One-shot: chain per-level stencils (fast, no composition overhead)
let refined_positions = result.interpolate(&positions);

// Animation: precompute composed stencils (reuse across frames)
let composed = result.compose_stencils(positions.len());
let frame_1_positions = composed.interpolate(&positions);
let frame_2_positions = composed.interpolate(&new_frame_positions);
```

## Supported schemes

| Scheme | Vertex stencils | Face-varying stencils |
|---|---|---|
| Catmull-Clark | All options (creases, corners, boundary modes, triangle rules) | AllLinear, Boundaries |
| Loop | All options (creases, corners, boundary modes) | Not yet |

## Interpolatable trait

Implement for any type you want to subdivide:

```rust
use subdiv_kernels::Interpolatable;

#[derive(Default, Clone)]
struct Color { r: f32, g: f32, b: f32, a: f32 }

impl Interpolatable for Color {
    fn add_with_weight(&mut self, src: &Self, weight: f32) {
        self.r += src.r * weight;
        self.g += src.g * weight;
        self.b += src.b * weight;
        self.a += src.a * weight;
    }
}
```

Built-in implementations: `f32`, `f64`, `[f32; 2..4]`, `[f64; 2..4]`.
