//! Shared crease and corner sharpness helpers.
//!
//! Used by CC, Loop, and Sqrt3 stencil modules.

pub(crate) const AUTO_CORNER_MIN_CREASE: f32 = 0.01;

/// Decay crease sharpness by one level.
pub(crate) fn decay_sharpness(sharpness: f32) -> f32 {
    (sharpness - 1.0).max(0.0)
}

/// Map sharpness to a blend factor in [0, 1].
pub(crate) fn sharpness_to_blend(sharpness: f32) -> f32 {
    if sharpness <= 0.0 || sharpness.is_nan() {
        0.0
    } else if !sharpness.is_finite() {
        1.0
    } else {
        sharpness.clamp(0.0, 1.0)
    }
}

/// Compute implicit corner sharpness from incident crease values.
///
/// Returns the average sharpness if 3+ creases exceed the threshold,
/// otherwise 0.0.
pub(crate) fn auto_corner_sharpness(creases: &[f32]) -> f32 {
    let (sum, count) = creases
        .iter()
        .copied()
        .filter(|c| c.is_finite() && *c > AUTO_CORNER_MIN_CREASE)
        .fold((0.0f32, 0u32), |(sum, count), c| (sum + c, count + 1));

    if count >= 3 { sum / count as f32 } else { 0.0 }
}

const OSD_INFINITE_SHARPNESS: f32 = 10.0;

/// One level of sharpness decrement (OpenSubdiv/DeRose), capping the infinite
/// sentinel.
fn osd_decrement(s: f32) -> f32 {
    if s >= OSD_INFINITE_SHARPNESS {
        OSD_INFINITE_SHARPNESS
    } else {
        decay_sharpness(s)
    }
}

/// Crease vertex point: `6/8·v + 1/8·n0 + 1/8·n1`.
fn crease_stencil(vi: u32, n0: u32, n1: u32) -> Vec<(u32, f32)> {
    Vec::from([(vi, 6.0 / 8.0), (n0, 1.0 / 8.0), (n1, 1.0 / 8.0)])
}

/// First and last index satisfying `is_sharp` (the two outermost sharp edges),
/// or `None` if fewer than two qualify.
fn edge_pair_indices(sharp_creases: &[f32], is_sharp: impl Fn(f32) -> bool) -> Option<[usize; 2]> {
    let first = sharp_creases.iter().position(|&c| is_sharp(c))?;
    let last = sharp_creases.iter().rposition(|&c| is_sharp(c))?;
    (first != last).then_some([first, last])
}

/// Sharp/crease/corner vertex stencil, OpenSubdiv/DeRose rule.
///
/// Classifies the vertex by its sharp-edge count — smooth (0–1), crease (2,
/// `6/8·v + 1/8·n + 1/8·n`), or corner (3+, pinned to `v`) — and blends the
/// parent rule toward the post-decrement child rule by the transitioning
/// sharpness, so fractional creases interpolate across levels. `corner` is the
/// vertex's own (explicit or auto) corner sharpness; `smooth` is the scheme's
/// ordinary smooth stencil; `original` is `[(vi, 1.0)]`. Shared by all schemes.
pub(crate) fn sharp_vertex_stencil_osd(
    vi: u32,
    smooth: &[(u32, f32)],
    original: &[(u32, f32)],
    corner: f32,
    sharp_neighbors: &[u32],
    sharp_creases: &[f32],
) -> Vec<(u32, f32)> {
    let parent_sharp_count = sharp_creases.iter().filter(|&&c| c > 0.0).count();
    let parent_rule = if corner > 0.0 {
        3
    } else {
        parent_sharp_count.min(3)
    };

    let rule_pos = |rule: usize, is_sharp: &dyn Fn(f32) -> bool| -> Vec<(u32, f32)> {
        match rule {
            3 => original.to_vec(),
            2 => edge_pair_indices(sharp_creases, is_sharp)
                .map(|[i, j]| crease_stencil(vi, sharp_neighbors[i], sharp_neighbors[j]))
                .unwrap_or_else(|| smooth.to_vec()),
            _ => smooth.to_vec(),
        }
    };

    let parent_pos = rule_pos(parent_rule, &|c| c > 0.0);

    let child_corner = osd_decrement(corner);
    let child_sharp_count = sharp_creases
        .iter()
        .filter(|&&c| osd_decrement(c) > 0.0)
        .count();
    let child_rule = if child_corner > 0.0 {
        3
    } else {
        child_sharp_count.min(3)
    };

    if child_rule == parent_rule {
        return parent_pos;
    }
    let child_pos = rule_pos(child_rule, &|c| osd_decrement(c) > 0.0);

    // Blend parent→child by the sharpness that decays to zero this level.
    let mut transition_sum = 0.0f32;
    let mut transition_count = 0usize;
    if corner > 0.0 && child_corner <= 0.0 {
        transition_sum += corner;
        transition_count += 1;
    }
    sharp_creases.iter().for_each(|&pc| {
        if pc > 0.0 && osd_decrement(pc) <= 0.0 {
            transition_sum += pc;
            transition_count += 1;
        }
    });
    let parent_weight = if transition_count == 0 {
        0.0
    } else {
        (transition_sum / transition_count as f32).clamp(0.0, 1.0)
    };

    let mut result = Vec::new();
    merge(&mut result, &child_pos, 1.0 - parent_weight);
    merge(&mut result, &parent_pos, parent_weight);
    result
}

/// Accumulate `source` scaled by `w` into `stencil`.
pub(crate) fn merge(stencil: &mut Vec<(u32, f32)>, source: &[(u32, f32)], w: f32) {
    source.iter().for_each(|&(idx, sw)| {
        if let Some(entry) = stencil.iter_mut().find(|(i, _)| *i == idx) {
            entry.1 += sw * w;
        } else {
            stencil.push((idx, sw * w));
        }
    });
}
