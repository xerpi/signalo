// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Fractional-delay polyphase filter bank for sub-sample interpolation.
//!
//! One low-pass prototype, designed at the upsampled rate `num_phases`, is
//! split into `num_phases` phase branches by
//! [`PolyphaseFilterBankVec`]. Branch `q` implements a fractional delay of
//! `q / num_phases` of a sample, so evaluating the branch nearest a
//! fractional timing instant reconstructs the signal there. A second bank
//! holds the prototype's derivative, giving `dy` at the strobe for a
//! maximum-likelihood timing detector.

use num_traits::Float;

use crate::filters::fir::design::kaiser_order;
use crate::filters::fir::design::windowed_sinc::kaiser;
use crate::filters::fir::polyphase::filter_bank::PolyphaseFilterBankVec;

/// Rounds `value` up to the next odd number, leaving odd values unchanged.
const fn make_odd(value: usize) -> usize {
    if value.is_multiple_of(2) {
        value + 1
    } else {
        value
    }
}

/// Per-phase subfilter length for a fractional-delay bank whose prototype
/// must span at least `min_len` taps across `num_phases` branches. Forced odd
/// so the centre phase is a pure integer delay (a linear-phase, symmetric
/// subfilter).
const fn taps_per_phase(min_len: usize, num_phases: usize) -> usize {
    make_odd((min_len - 1).div_ceil(num_phases) + 1)
}

/// A fractional-delay polyphase interpolator and its matched derivative bank.
#[derive(Clone, Debug)]
pub struct FractionalDelayPfb<K = f32> {
    num_phases: usize,
    taps_per_phase: usize,
    /// Interpolation bank (phase-major).
    interpolation: PolyphaseFilterBankVec<K>,
    /// Derivative bank, matching layout.
    derivative: PolyphaseFilterBankVec<K>,
}

impl<K> FractionalDelayPfb<K>
where
    K: Float + core::fmt::Debug,
{
    /// Designs a `num_phases`-branch bank with a Kaiser prototype at the given
    /// stopband attenuation (dB). The prototype is a unit-gain interpolation
    /// low-pass at the input Nyquist, so the branch for phase 0 is (near) an
    /// identity and higher phases are sub-sample delays.
    ///
    /// # Panics
    ///
    /// Panics if `num_phases` is zero or a design parameter is not
    /// representable in `K`.
    #[must_use]
    pub fn new(num_phases: usize, stopband_atten: K) -> Self {
        assert!(num_phases > 0, "fractional-delay bank needs phase branches");
        let phases = K::from(num_phases).expect("phase count is representable");
        // Cutoff at the input Nyquist, expressed at the upsampled
        // (`num_phases`) rate. The transition band is as wide as the cutoff,
        // which sets a modest per-phase tap count.
        let cutoff = K::from(0.5).expect("0.5 is representable") / phases;
        let order = kaiser_order(stopband_atten, cutoff);
        let taps_per_phase = taps_per_phase(order.num_taps, num_phases);
        // A prototype length that is a multiple of `num_phases` recovers
        // exactly `taps_per_phase` per branch; only the leading
        // fractional-delay span is filled, the tail stays zero.
        let length = num_phases * (taps_per_phase - 1) + 1;

        let mut prototype = alloc::vec![K::zero(); num_phases * taps_per_phase];
        kaiser::lowpass_with_beta(&mut prototype[..length], order.beta, cutoff);
        // Unit passband gain per phase: the prototype sums to one, so scale to
        // `num_phases`.
        for tap in &mut prototype[..length] {
            *tap = *tap * phases;
        }

        // Central-difference derivative of the prototype, scaled to
        // per-input-sample units.
        let scale = phases / K::from(2.0).expect("2.0 is representable");
        let mut derivative = alloc::vec![K::zero(); num_phases * taps_per_phase];
        derivative[0] = scale * prototype[1];
        for n in 1..length - 1 {
            derivative[n] = scale * (prototype[n + 1] - prototype[n - 1]);
        }
        derivative[length - 1] = -scale * prototype[length - 2];

        Self {
            num_phases,
            taps_per_phase,
            interpolation: PolyphaseFilterBankVec::from_prototype_taps(num_phases, &prototype),
            derivative: PolyphaseFilterBankVec::from_prototype_taps(num_phases, &derivative),
        }
    }
}

impl<K> FractionalDelayPfb<K> {
    /// Number of phase branches (timing resolution).
    #[must_use]
    pub fn num_phases(&self) -> usize {
        self.num_phases
    }

    /// Per-phase subfilter length; also the number of window samples each
    /// evaluation reads.
    #[must_use]
    pub fn taps_per_phase(&self) -> usize {
        self.taps_per_phase
    }

    /// Integer group delay in input samples: the window is centred this many
    /// samples back.
    #[must_use]
    pub fn group_delay(&self) -> usize {
        (self.taps_per_phase - 1) / 2
    }

    /// Interpolates at branch `phase` over the `taps_per_phase`-sample
    /// `window` (oldest first).
    pub fn interpolate<'a, T, I>(&self, phase: usize, window: I) -> T
    where
        T: 'a
            + Clone
            + num_traits::Zero
            + core::ops::Add<Output = T>
            + core::ops::Mul<K, Output = T>,
        K: Clone,
        I: IntoIterator<Item = &'a T>,
    {
        self.interpolation.execute(phase, window)
    }

    /// Evaluates the derivative bank at branch `phase`, giving `dy` for a
    /// maximum-likelihood timing detector.
    pub fn derivative<'a, T, I>(&self, phase: usize, window: I) -> T
    where
        T: 'a
            + Clone
            + num_traits::Zero
            + core::ops::Add<Output = T>
            + core::ops::Mul<K, Output = T>,
        K: Clone,
        I: IntoIterator<Item = &'a T>,
    {
        self.derivative.execute(phase, window)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn constant(value: f32, count: usize) -> alloc::vec::Vec<f32> {
        alloc::vec![value; count]
    }

    /// Phase 0 is an integer delay, so a constant window returns that constant
    /// (unit passband gain).
    #[test]
    fn phase_zero_passes_a_constant() {
        let pfb = FractionalDelayPfb::<f32>::new(32, 60.0);
        let window = constant(2.5, pfb.taps_per_phase());
        let out: f32 = pfb.interpolate(0, window.iter());
        assert!((out - 2.5).abs() < 1e-3, "got {out}");
    }

    /// Every phase has unit DC gain, so a constant window is preserved at any
    /// fractional offset.
    #[test]
    fn all_phases_preserve_dc() {
        let pfb = FractionalDelayPfb::<f32>::new(32, 60.0);
        let window = constant(1.0, pfb.taps_per_phase());
        for phase in 0..pfb.num_phases() {
            let out: f32 = pfb.interpolate(phase, window.iter());
            assert!((out - 1.0).abs() < 1e-2, "phase {phase} got {out}");
        }
    }

    /// On a linear ramp the derivative bank reports the slope per input
    /// sample (within the truncated-window bias), and DC has zero derivative.
    #[test]
    fn derivative_reports_the_ramp_slope() {
        let pfb = FractionalDelayPfb::<f32>::new(32, 60.0);
        #[allow(clippy::cast_precision_loss)]
        let ramp: alloc::vec::Vec<f32> = (0..pfb.taps_per_phase()).map(|k| k as f32).collect();
        let slope: f32 = pfb.derivative(0, ramp.iter());
        assert!((slope - 1.0).abs() < 0.05, "got {slope}");
        let flat = constant(3.0, pfb.taps_per_phase());
        let zero: f32 = pfb.derivative(0, flat.iter());
        assert!(zero.abs() < 1e-2, "got {zero}");
    }
}
