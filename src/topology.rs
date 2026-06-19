//! Mesh topology types: the control cage [`Mesh`], refined [`Adjacency`], and
//! face-varying channels.

use crate::KernelError;

/// The control mesh: faces, edges, and optional crease/corner sharpness — but
/// **no positions**.
///
/// This is the input to [`Refiner`](crate::Refiner). Positions and any other
/// per-vertex data are carried separately and applied with a
/// [`StencilTable`](crate::StencilTable), so one refinement serves every
/// attribute.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Mesh {
    /// Number of vertices (needed since there is no positions array).
    pub vertex_count: u32,

    /// Number of corners for each face.
    pub face_vertex_counts: Vec<u32>,

    /// Flat face-vertex index list.
    pub face_vertex_indices: Vec<u32>,

    /// Canonical undirected edge endpoints.
    pub edge_vertices: Vec<[u32; 2]>,

    /// Per-edge crease values aligned with `edge_vertices`.
    pub edge_creases: Vec<f32>,

    /// Per-vertex corner values aligned by vertex index.
    pub vertex_corners: Vec<f32>,
}

impl Mesh {
    /// Validate basic array consistency.
    pub fn validate(&self) -> Result<(), KernelError> {
        let corner_count: usize = self.face_vertex_counts.iter().map(|v| *v as usize).sum();

        (corner_count == self.face_vertex_indices.len())
            .then_some(())
            .ok_or(KernelError::InvalidTopology(
                "face corner count does not match index buffer length",
            ))?;

        (self.edge_creases.len() == self.edge_vertices.len())
            .then_some(())
            .ok_or(KernelError::InvalidTopology(
                "edge crease count does not match edge buffer length",
            ))?;

        (self.vertex_corners.len() == self.vertex_count as usize)
            .then_some(())
            .ok_or(KernelError::InvalidTopology(
                "vertex corner count does not match vertex count",
            ))?;

        self.face_vertex_indices
            .iter()
            .all(|&idx| idx < self.vertex_count)
            .then_some(())
            .ok_or(KernelError::InvalidTopology("face index out of bounds"))?;

        self.edge_vertices
            .iter()
            .flat_map(|e| e.iter())
            .all(|&idx| idx < self.vertex_count)
            .then_some(())
            .ok_or(KernelError::InvalidTopology("edge endpoint out of bounds"))
    }
}

/// Pre-built adjacency arrays in CSR format.
///
/// Produced by refinement and consumed by adapter-side mesh construction
/// to avoid redundant topology analysis.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct Adjacency {
    /// Per face-corner → edge index (same layout as `face_vertex_indices`).
    pub face_edges: Vec<u32>,
    /// Two incident faces per edge. `u32::MAX` = boundary.
    pub edge_faces: Vec<[u32; 2]>,
    /// CSR vertex → edge adjacency: per-vertex start offsets into `vertex_edges`.
    pub vertex_edge_offsets: Vec<u32>,
    /// Flattened incident-edge indices (sliced by `vertex_edge_offsets`).
    pub vertex_edges: Vec<u32>,
    /// CSR vertex → face adjacency: per-vertex start offsets into `vertex_faces`.
    pub vertex_face_offsets: Vec<u32>,
    /// Flattened incident-face indices (sliced by `vertex_face_offsets`).
    pub vertex_faces: Vec<u32>,
    /// Per-edge boundary flag.
    pub edge_is_boundary: Vec<bool>,
    /// Per-vertex boundary flag.
    pub vertex_is_boundary: Vec<bool>,
}

/// Per-channel face-varying value-index topology.
///
/// At UV seams, face-corners sharing a geometric vertex can reference
/// different FVar values. This struct holds the per-face-corner indices
/// into the channel's own value array.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FaceVaryingChannel {
    /// Per face-corner index into the channel's value array.
    /// Same length as `Mesh::face_vertex_indices`.
    pub indices: Vec<u32>,

    /// Number of distinct values in this channel.
    pub value_count: u32,
}
