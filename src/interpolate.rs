//! Interpolation trait for subdivision-compatible data.

use core::ops::AddAssign;

/// Trait for data that can be interpolated by subdivision rules.
///
/// Implement this for any type you want to subdivide: positions, UVs,
/// colors, normals, scalar weights, custom attributes.
///
/// Follows the OpenSubdiv `Clear()` / `AddWithWeight()` pattern.
/// Using a local trait avoids orphan-rule problems that arise with
/// `Mul<f32>` on foreign array types.
///
/// # Example
///
/// ```
/// use subdiv_kernels::Interpolatable;
///
/// #[derive(Default, Clone)]
/// struct Color { r: f32, g: f32, b: f32, a: f32 }
///
/// impl Interpolatable for Color {
///     fn add_with_weight(&mut self, src: &Self, weight: f32) {
///         self.r += src.r * weight;
///         self.g += src.g * weight;
///         self.b += src.b * weight;
///         self.a += src.a * weight;
///     }
/// }
/// ```
pub trait Interpolatable: Default + Clone {
    /// Accumulate `src` scaled by `weight` into `self`.
    fn add_with_weight(&mut self, src: &Self, weight: f32);
}

// ── Blanket impls for scalars ──────────────────────────────────────────

impl Interpolatable for f32 {
    #[inline]
    fn add_with_weight(&mut self, src: &Self, weight: f32) {
        *self += src * weight;
    }
}

impl Interpolatable for f64 {
    #[inline]
    fn add_with_weight(&mut self, src: &Self, weight: f32) {
        *self += *src * weight as f64;
    }
}

// ── Blanket impls for fixed-size float arrays ──────────────────────────
//
// Const-generic over the length, so any `[f32; N]` / `[f64; N]` works
// (`[1.0; 3]` positions, `[_; 2]` UVs, `[_; 4]` colors, `[_; 8]` skin weights,
// …). The `where [_; N]: Default` bound is required because the standard
// library only implements `Default` for arrays up to length 32.

impl<const N: usize> Interpolatable for [f32; N]
where
    [f32; N]: Default,
{
    #[inline]
    fn add_with_weight(&mut self, src: &Self, weight: f32) {
        self.iter_mut()
            .zip(src.iter())
            .for_each(|(dst, s)| dst.add_assign(s * weight));
    }
}

impl<const N: usize> Interpolatable for [f64; N]
where
    [f64; N]: Default,
{
    #[inline]
    fn add_with_weight(&mut self, src: &Self, weight: f32) {
        let w = weight as f64;
        self.iter_mut()
            .zip(src.iter())
            .for_each(|(dst, s)| dst.add_assign(s * w));
    }
}
