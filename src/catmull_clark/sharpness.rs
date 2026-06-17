//! Crease and corner sharpness helpers for CC stencil extraction.

use crate::options::{CornerRule, CreaseComputationMethod};

use super::topology::Topology;

pub(super) const AUTO_CORNER_MIN_CREASE: f32 = 0.01;
const OSD_INFINITE_SHARPNESS: f32 = 10.0;

pub(super) fn child_edge_creases(
    edge_index: usize,
    topology: &Topology,
    effective_edge_creases: &[f32],
    crease_normalize: bool,
    crease_computation: CreaseComputationMethod,
    corner_rule: CornerRule,
) -> (f32, f32) {
    let parent = effective_edge_creases[edge_index];

    if parent <= 0.0 {
        return (0.0, 0.0);
    }

    if crease_normalize {
        return (parent, parent);
    }

    match crease_computation {
        CreaseComputationMethod::Uniform => {
            let child = match corner_rule {
                CornerRule::OpenSubdivDeRose => osd_decrement_sharpness(parent),
            };
            (child, child)
        }
        CreaseComputationMethod::Chaikin => {
            let [a, b] = topology.edge_vertices[edge_index];
            let avg_a = mean_incident_creases(a as usize, topology, effective_edge_creases);
            let avg_b = mean_incident_creases(b as usize, topology, effective_edge_creases);
            (
                chaikin_subdivide_sharpness(parent, avg_a),
                chaikin_subdivide_sharpness(parent, avg_b),
            )
        }
    }
}

fn mean_incident_creases(vertex_index: usize, topology: &Topology, edge_creases: &[f32]) -> f32 {
    let incident = topology.vertex_edges.row(vertex_index);

    if incident.is_empty() {
        return 0.0;
    }

    incident
        .iter()
        .map(|&edge_index| edge_creases[edge_index as usize])
        .sum::<f32>()
        / incident.len() as f32
}

fn decay_sharpness(sharpness: f32) -> f32 {
    (sharpness - 1.0).max(0.0)
}

fn chaikin_subdivide_sharpness(parent_sharpness: f32, vertex_avg_sharpness: f32) -> f32 {
    ((parent_sharpness + vertex_avg_sharpness) * 0.5 - 1.0).max(0.0)
}

fn osd_decrement_sharpness(sharpness: f32) -> f32 {
    if sharpness >= OSD_INFINITE_SHARPNESS {
        OSD_INFINITE_SHARPNESS
    } else {
        decay_sharpness(sharpness)
    }
}

pub(super) fn decay_corner_sharpness(sharpness: f32, corner_rule: CornerRule) -> f32 {
    if sharpness <= 0.0 {
        return 0.0;
    }

    match corner_rule {
        CornerRule::OpenSubdivDeRose => osd_decrement_sharpness(sharpness),
    }
}

pub(super) fn auto_corner_sharpness_from_creases(creases: &[f32]) -> f32 {
    let (sum, count) = creases
        .iter()
        .copied()
        .filter(|crease| crease.is_finite() && *crease > AUTO_CORNER_MIN_CREASE)
        .fold((0.0f32, 0u32), |(sum, count), crease| {
            (sum + crease, count + 1)
        });

    if count >= 3 { sum / count as f32 } else { 0.0 }
}

pub(super) fn sharpness_to_blend(sharpness: f32) -> f32 {
    if sharpness <= 0.0 || sharpness.is_nan() {
        0.0
    } else if !sharpness.is_finite() {
        1.0
    } else {
        sharpness.clamp(0.0, 1.0)
    }
}
