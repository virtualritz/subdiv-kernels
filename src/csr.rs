//! Compressed Sparse Row (CSR) vector — a flat, cache-friendly
//! representation of variable-length per-element adjacency lists.

/// CSR-packed adjacency list. Row `i` spans
/// `values[offsets[i]..offsets[i+1]]`.
///
/// # Invariants (maintained by all constructors)
///
/// - `offsets.len() >= 1` (empty CSR has `offsets == [0]`).
/// - `offsets` is monotonically non-decreasing.
/// - `*offsets.last() as usize <= values.len()`.
///
/// These invariants make all `get_unchecked` calls in accessors sound.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub(crate) struct CsrVec {
    offsets: Vec<u32>,
    values: Vec<u32>,
}

impl CsrVec {
    /// Number of rows.
    #[inline]
    pub fn len(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    /// Slice of values for row `i`.
    ///
    /// # Panics
    ///
    /// Debug-asserts `i < self.len()`.
    #[inline]
    pub fn row(&self, i: usize) -> &[u32] {
        debug_assert!(i < self.len());
        // SAFETY: Constructors guarantee offsets[i] <= offsets[i+1] <= values.len().
        // Caller must pass i < self.len(), enforced by debug_assert above.
        unsafe {
            let s = *self.offsets.get_unchecked(i) as usize;
            let e = *self.offsets.get_unchecked(i + 1) as usize;
            self.values.get_unchecked(s..e)
        }
    }

    /// Number of entries in row `i`.
    #[inline]
    pub fn row_len(&self, i: usize) -> usize {
        debug_assert!(i < self.len());
        // SAFETY: Same as `row`.
        unsafe { (*self.offsets.get_unchecked(i + 1) - *self.offsets.get_unchecked(i)) as usize }
    }

    /// Element at `(row, col)` within that row's slice.
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> u32 {
        debug_assert!(col < self.row_len(row));
        // SAFETY: `row()` returns a valid slice; col < row_len is
        // enforced by debug_assert.
        unsafe { *self.row(row).get_unchecked(col) }
    }

    /// Consume into (offsets, values) pair for moving into
    /// [`Adjacency`](crate::Adjacency).
    #[inline]
    pub fn into_parts(self) -> (Vec<u32>, Vec<u32>) {
        (self.offsets, self.values)
    }

    /// Build directly from prebuilt `offsets`/`values`. Caller guarantees the
    /// CSR invariants (offsets non-empty, non-decreasing, last <= values.len()).
    /// Used by analytic builders that size and fill rows by index.
    #[inline]
    pub fn from_parts(offsets: Vec<u32>, values: Vec<u32>) -> Self {
        debug_assert!(!offsets.is_empty());
        debug_assert!(*offsets.last().unwrap() as usize <= values.len());
        Self { offsets, values }
    }

    /// Build from jagged rows (any `AsRef<[usize]>`, e.g. `Vec` or `SmallVec`),
    /// casting values to `u32`.
    pub fn from_jagged<R: AsRef<[usize]>>(jagged: &[R]) -> Self {
        let offsets: Vec<u32> = std::iter::once(0)
            .chain(jagged.iter().scan(0u32, |acc, v| {
                *acc += v.as_ref().len() as u32;
                Some(*acc)
            }))
            .collect();
        let values: Vec<u32> = jagged
            .iter()
            .flat_map(|v| v.as_ref().iter().map(|&x| x as u32))
            .collect();
        Self { offsets, values }
    }

    /// Build from jagged rows (any `AsRef<[u32]>`, e.g. `Vec` or `SmallVec`).
    pub fn from_jagged_u32<R: AsRef<[u32]>>(jagged: &[R]) -> Self {
        let offsets: Vec<u32> = std::iter::once(0)
            .chain(jagged.iter().scan(0u32, |acc, v| {
                *acc += v.as_ref().len() as u32;
                Some(*acc)
            }))
            .collect();
        let values: Vec<u32> = jagged
            .iter()
            .flat_map(|v| v.as_ref().iter().copied())
            .collect();
        Self { offsets, values }
    }
}
