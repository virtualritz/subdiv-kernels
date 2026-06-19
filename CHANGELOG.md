# Changelog

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

First public release — subdivision-surface kernels, topology-first and
stencil-based, generic over the data you subdivide.

### Schemes

- Catmull–Clark, Loop, √3 (Kobbelt), and Doo–Sabin.

### Evaluation

- Per-level stencil tables for one-shot refinement, a composed table for
  animation (static topology, changing data), and sparse-edit queries.
- Generic over any `Interpolatable` type (positions, UVs, colors, …); built-in
  impls for `f32`, `f64`, and `[f32; N]` / `[f64; N]`.
- Limit-surface position, tangent, and normal stencils.

### Features

- Creases and corners (OpenSubdiv/DeRose rules), boundary-interpolation modes,
  and face-varying (UV-seam) channels for every scheme.
- Optional `wgpu` GPU compute path for stencil evaluation.
- Interactive Bevy + egui example (`cargo run --example bevy --features bevy`).
