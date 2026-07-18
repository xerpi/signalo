// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Timing-error detectors for symbol synchronization.
//!
//! Each detector estimates the fractional timing error that a loop filter
//! drives toward zero. They share the [`TimingErrorDetector`] interface:
//! interpolated strobes are fed in with
//! [`consume_strobe`](TimingErrorDetector::consume_strobe), and data-aided or
//! decision-directed detectors take an observed/reference pair through
//! [`apply_feedback`](TimingErrorDetector::apply_feedback).
//!
//! # When to use which detector
//!
//! | Detector              | Feedback        | Strobe rate      |
//! | --------------------- | --------------- | ---------------- |
//! | [`Gardner`]           | none            | twice per symbol |
//! | [`MaximumLikelihood`] | known or sliced | once per symbol  |
//! | [`MuellerMuller`]     | known or sliced | once per symbol  |
//!
//! - **Gardner** is non-data-aided and runs at twice the symbol rate (a mid
//!   strobe between decisions); it needs no feedback and tolerates an
//!   uncorrected carrier rotation.
//! - **`MaximumLikelihood`** runs at the symbol rate from the derivative
//!   matched-filter output and a data-aided (known symbol) or
//!   decision-directed (slicer output) reference.
//! - **`MuellerMuller`** runs at the symbol rate from consecutive
//!   observed/reference pairs and needs no derivative filter.
//!
//! # Error polarity
//!
//! Every detector reports a positive error when the strobe is late, so one
//! loop convention (`position -= filter(error)`) serves them all.

use num_complex::Complex;
use num_traits::{Float, FloatConst};

/// A symbol-timing error detector.
pub trait TimingErrorDetector<T> {
    /// The number of interpolated strobes consumed per symbol interval.
    const STROBE_RATE: usize;

    /// Consumes one interpolated `strobe` and its optional matched-filter
    /// `derivative`, returning a non-data-aided timing error when enough
    /// strobe history is available.
    fn consume_strobe(&mut self, strobe: Complex<T>, derivative: Option<Complex<T>>) -> Option<T>;

    /// Computes a data-aided or decision-directed timing error from the
    /// `observed` symbol and its `reference` (a known or sliced symbol), or
    /// `None` when the detector needs no feedback or has no history yet.
    fn apply_feedback(&mut self, observed: Complex<T>, reference: Complex<T>) -> Option<T>;

    /// Clears the strobe and feedback history for a new burst, keeping the
    /// detector configuration.
    fn reset(&mut self);
}

/// Non-data-aided Gardner timing-error detector.
///
/// Error: `e = Re{mid * conj(y_cur - y_prev)}`, using the mid strobe between
/// two decision strobes. It runs at twice the symbol rate: strobes alternate
/// decision, mid, decision, mid, and only the decision strobes are visible.
#[derive(Clone, Copy, Debug)]
pub struct Gardner<T = f32> {
    previous: Option<Complex<T>>,
    mid: Option<Complex<T>>,
    expecting_mid: bool,
}

impl<T> Default for Gardner<T> {
    fn default() -> Self {
        Self {
            previous: None,
            mid: None,
            expecting_mid: false,
        }
    }
}

impl<T> TimingErrorDetector<T> for Gardner<T>
where
    T: Float,
{
    const STROBE_RATE: usize = 2;

    fn consume_strobe(&mut self, strobe: Complex<T>, _derivative: Option<Complex<T>>) -> Option<T> {
        if self.expecting_mid {
            self.mid = Some(strobe);
            self.expecting_mid = false;
            return None;
        }
        let error = match (self.previous, self.mid) {
            (Some(previous), Some(mid)) => Some((mid * (strobe - previous).conj()).re),
            _ => None,
        };
        self.previous = Some(strobe);
        self.expecting_mid = true;
        error
    }

    fn apply_feedback(&mut self, _observed: Complex<T>, _reference: Complex<T>) -> Option<T> {
        None
    }

    fn reset(&mut self) {
        *self = Self::default();
    }
}

/// Data-aided or decision-directed maximum-likelihood timing-error detector.
///
/// Error: `e = -Re{conj(reference) * derivative} / gain`, where `derivative`
/// is the derivative matched-filter output captured at the strobe. The
/// negation makes a positive error mean the strobe is late, the polarity
/// [`Gardner`] reports (the derivative itself is positive on the pulse's
/// rising edge, before the peak). Dividing by the detector gain keeps the
/// loop filter's bandwidth independent of the pulse shape. It runs at the
/// symbol rate.
#[derive(Clone, Copy, Debug)]
pub struct MaximumLikelihood<T = f32> {
    derivative: Option<Complex<T>>,
    gain: T,
}

impl<T> Default for MaximumLikelihood<T>
where
    T: Float,
{
    fn default() -> Self {
        Self {
            derivative: None,
            gain: T::one(),
        }
    }
}

impl<T> MaximumLikelihood<T>
where
    T: Float + FloatConst + core::fmt::Debug,
{
    /// Returns the detector gain `Kp` for a root-raised-cosine
    /// transmit/matched-filter pair with the given `rolloff` in `(0, 1]`, in
    /// per-symbol-period units.
    ///
    /// # Panics
    ///
    /// Panics if `rolloff` is outside `(0, 1]` or is not finite.
    #[must_use]
    pub fn detector_gain(rolloff: T) -> T {
        assert!(
            rolloff.is_finite() && rolloff > T::zero() && rolloff <= T::one(),
            "RRC rolloff must be finite and in (0, 1] (got {rolloff:?})"
        );
        let three = T::from(3.0).expect("3.0 is representable");
        let eight = T::from(8.0).expect("8.0 is representable");
        let rolloff_squared = rolloff * rolloff;
        let pi_squared = T::PI() * T::PI();
        (-pi_squared * (T::one() + three * rolloff_squared) / three + eight * rolloff_squared).abs()
    }

    /// Creates a detector that normalizes its error by `gain` (typically
    /// [`Self::detector_gain`]), so the timing loop's bandwidth is set by the
    /// loop filter alone.
    ///
    /// # Panics
    ///
    /// Panics if `gain` is not finite and strictly positive.
    #[must_use]
    pub fn with_gain(gain: T) -> Self {
        assert!(
            gain.is_finite() && gain > T::zero(),
            "maximum-likelihood timing-detector gain must be finite and > 0 (got {gain:?})"
        );
        Self {
            derivative: None,
            gain,
        }
    }
}

impl<T> TimingErrorDetector<T> for MaximumLikelihood<T>
where
    T: Float,
{
    const STROBE_RATE: usize = 1;

    fn consume_strobe(&mut self, _strobe: Complex<T>, derivative: Option<Complex<T>>) -> Option<T> {
        self.derivative = derivative;
        None
    }

    fn apply_feedback(&mut self, _observed: Complex<T>, reference: Complex<T>) -> Option<T> {
        self.derivative
            .map(|derivative| -(reference.conj() * derivative).re / self.gain)
    }

    fn reset(&mut self) {
        self.derivative = None;
    }
}

/// Data-aided or decision-directed Mueller & Muller timing-error detector.
///
/// Error: `e = Re{conj(reference_cur) * y_prev} - Re{conj(reference_prev) * y_cur}`,
/// from the current and previous observed symbols and their references. The
/// term order makes a positive error mean the strobe is late, the polarity
/// [`Gardner`] reports. It runs at the symbol rate.
#[derive(Clone, Copy, Debug)]
pub struct MuellerMuller<T = f32> {
    previous_symbol: Option<Complex<T>>,
    previous_reference: Option<Complex<T>>,
}

impl<T> Default for MuellerMuller<T> {
    fn default() -> Self {
        Self {
            previous_symbol: None,
            previous_reference: None,
        }
    }
}

impl<T> TimingErrorDetector<T> for MuellerMuller<T>
where
    T: Float,
{
    const STROBE_RATE: usize = 1;

    fn consume_strobe(
        &mut self,
        _strobe: Complex<T>,
        _derivative: Option<Complex<T>>,
    ) -> Option<T> {
        None
    }

    fn apply_feedback(&mut self, observed: Complex<T>, reference: Complex<T>) -> Option<T> {
        let error = match (self.previous_symbol, self.previous_reference) {
            (Some(previous_symbol), Some(previous_reference)) => Some(
                (reference.conj() * previous_symbol).re - (previous_reference.conj() * observed).re,
            ),
            _ => None,
        };
        self.previous_symbol = Some(observed);
        self.previous_reference = Some(reference);
        error
    }

    fn reset(&mut self) {
        *self = Self::default();
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    use super::*;

    fn c(re: f32, im: f32) -> Complex<f32> {
        Complex::new(re, im)
    }

    #[test]
    fn gardner_computes_error_from_mid_and_decision_strobes() {
        let mut ted = Gardner::default();
        assert!(ted.consume_strobe(c(1.0, 0.0), None).is_none());
        assert!(ted.consume_strobe(c(0.5, 0.0), None).is_none());
        let error = ted.consume_strobe(c(3.0, 0.0), None).unwrap_or(f32::NAN);
        assert_abs_diff_eq!(error, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn maximum_likelihood_reports_positive_when_late() {
        let mut ted = MaximumLikelihood::default();
        assert!(ted.consume_strobe(c(1.0, 0.0), Some(c(2.0, 1.0))).is_none());
        // A positive derivative (rising pulse, early strobe) yields a negative error.
        let error = ted
            .apply_feedback(c(2.0, 0.0), c(1.0, 0.0))
            .unwrap_or(f32::NAN);
        assert_abs_diff_eq!(error, -2.0, epsilon = 1e-6);
    }

    #[test]
    fn maximum_likelihood_detector_gain_is_positive() {
        assert!(MaximumLikelihood::detector_gain(0.35_f32) > 0.0);
    }

    #[test]
    #[should_panic(expected = "maximum-likelihood timing-detector gain")]
    fn maximum_likelihood_rejects_an_invalid_gain() {
        let _ = MaximumLikelihood::with_gain(0.0_f32);
    }

    #[test]
    fn mueller_muller_needs_two_symbols() {
        let mut ted = MuellerMuller::default();
        assert!(ted.apply_feedback(c(1.0, 0.0), c(1.0, 0.0)).is_none());
        let error = ted
            .apply_feedback(c(2.0, 0.0), c(1.0, 0.0))
            .unwrap_or(f32::NAN);
        assert_abs_diff_eq!(error, -1.0, epsilon = 1e-6);
    }

    #[test]
    fn reset_clears_history_and_keeps_configuration() {
        let mut ted = MaximumLikelihood::with_gain(2.0_f32);
        assert!(ted.consume_strobe(c(1.0, 0.0), Some(c(2.0, 0.0))).is_none());
        ted.reset();
        // No derivative after reset, and the gain still divides the next error.
        assert!(ted.apply_feedback(c(1.0, 0.0), c(1.0, 0.0)).is_none());
        assert!(ted.consume_strobe(c(1.0, 0.0), Some(c(2.0, 0.0))).is_none());
        let error = ted
            .apply_feedback(c(1.0, 0.0), c(1.0, 0.0))
            .unwrap_or(f32::NAN);
        assert_abs_diff_eq!(error, -1.0, epsilon = 1e-6);
    }
}
