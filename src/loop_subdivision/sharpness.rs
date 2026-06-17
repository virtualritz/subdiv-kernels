//! Crease and corner sharpness helpers for Loop stencil extraction.

pub(super) const AUTO_CORNER_MIN_CREASE: f32 = 0.01;

pub(super) fn auto_corner_sharpness_from_creases(creases: impl IntoIterator<Item = f32>) -> f32 {
    let (sharp_sum, sharp_count) = creases
        .into_iter()
        .filter(|&crease| crease.is_finite() && crease > AUTO_CORNER_MIN_CREASE)
        .fold((0.0f32, 0u32), |(sum, count), crease| {
            (sum + crease, count + 1)
        });

    if sharp_count >= 3 {
        sharp_sum / sharp_count as f32
    } else {
        0.0
    }
}

pub(super) fn decay_sharpness(sharpness: f32) -> f32 {
    (sharpness - 1.0).max(0.0)
}
