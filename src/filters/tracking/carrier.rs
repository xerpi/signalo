// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Carrier phase and frequency recovery.

use num_complex::Complex;
use num_traits::Float;

use crate::filters::iir::loop_filter::LoopFilter;

/// A carrier-recovery phase-locked loop.
///
/// The loop keeps a numerically-controlled-oscillator phase estimate. A caller
/// derotates each sample with [`derotate`](Self::derotate), measures the
/// residual phase error against a reference (a known symbol for data-aided
/// operation, or the sample's hard decision for decision-directed operation),
/// and feeds it to [`advance`](Self::advance). The second-order
/// [`LoopFilter`] tracks both a constant phase and a constant frequency
/// offset.
#[derive(Clone, Debug)]
pub struct CarrierSynchronizer<T = f32> {
    filter: LoopFilter<T>,
    phase: T,
}

impl<T> CarrierSynchronizer<T>
where
    T: Float + core::fmt::Debug,
{
    /// Creates a carrier synchronizer with the given normalized loop
    /// bandwidth (`Bn * T`) and damping factor.
    ///
    /// # Panics
    ///
    /// Panics if `loop_bandwidth` or `damping` is not finite or is not
    /// positive (see [`LoopFilter::new`]).
    #[must_use]
    pub fn new(loop_bandwidth: T, damping: T) -> Self {
        Self {
            filter: LoopFilter::new(loop_bandwidth, damping),
            phase: T::zero(),
        }
    }

    /// Resets the oscillator phase and loop gains for an independent burst.
    pub fn reset(&mut self, loop_bandwidth: T, damping: T) {
        self.filter = LoopFilter::new(loop_bandwidth, damping);
        self.phase = T::zero();
    }

    /// Derotates `sample` by the current phase estimate.
    #[must_use]
    pub fn derotate(&self, sample: Complex<T>) -> Complex<T> {
        sample * Complex::cis(-self.phase)
    }

    /// Advances the oscillator phase from a residual `phase_error` (radians).
    pub fn advance(&mut self, phase_error: T) {
        self.phase = self.phase + self.filter.update(phase_error);
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    use super::*;

    #[test]
    fn derotation_tracks_a_constant_phase_offset() {
        let offset = 0.5_f32;
        let mut synchronizer = CarrierSynchronizer::new(0.05, 0.707);
        let reference = Complex::new(1.0_f32, 0.0);
        let mut residual = f32::NAN;
        for _ in 0..200 {
            let received = reference * Complex::cis(offset);
            let derotated = synchronizer.derotate(received);
            residual = (derotated * reference.conj()).arg();
            synchronizer.advance(residual);
        }
        assert_abs_diff_eq!(residual, 0.0, epsilon = 1e-3);
    }

    #[test]
    fn reset_restores_the_zero_phase() {
        let mut synchronizer = CarrierSynchronizer::new(0.05_f32, 0.707);
        synchronizer.advance(1.0);
        synchronizer.advance(1.0);
        synchronizer.reset(0.05, 0.707);
        let sample = Complex::new(0.0_f32, 1.0);
        let derotated = synchronizer.derotate(sample);
        assert_abs_diff_eq!(derotated.re, sample.re, epsilon = 1e-6);
        assert_abs_diff_eq!(derotated.im, sample.im, epsilon = 1e-6);
    }
}
