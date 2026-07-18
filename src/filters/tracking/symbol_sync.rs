// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Symbol-timing recovery for oversampled linear-modulation streams.
//!
//! A timing-error detector drives a second-order loop filter that steers a
//! fractional-delay polyphase interpolator, recovering one sample per symbol
//! at the optimal instant. This resolves the sub-sample timing that an
//! integer-sample burst detector leaves.
//!
//! [`SymbolSynchronizer`] is generic over the [`TimingErrorDetector`] and
//! streaming:
//! full-rate matched-filter samples are fed one at a time with
//! [`push`](SymbolSynchronizer::push), and recovered symbols are pulled with
//! [`next_symbol`](SymbolSynchronizer::next_symbol). A non-data-aided
//! detector (Gardner) produces its error from the strobes alone. A data-aided
//! or decision-directed detector (maximum likelihood, Mueller & Muller)
//! instead takes a reference symbol through
//! [`apply_feedback`](SymbolSynchronizer::apply_feedback) after each pulled
//! symbol. It owns the loop filter, detector, interpolator, and timing
//! position, and keeps only the samples the interpolator needs in a fixed
//! ring sized at construction, so it neither holds the whole stream nor
//! allocates while running.

use num_complex::Complex;
use num_traits::Float;

use crate::filters::fir::polyphase::fractional_delay::FractionalDelayPfb;
use crate::filters::iir::loop_filter::LoopFilter;
use crate::filters::tracking::ted::{Gardner, TimingErrorDetector};

/// Phase branches in the timing interpolator, setting the sub-sample timing
/// resolution.
const TIMING_PHASES: usize = 32;
/// Stopband attenuation (dB) of the timing interpolator's Kaiser prototype.
const TIMING_STOPBAND_ATTEN: f64 = 60.0;

/// The strobe the synchronizer will interpolate next.
#[derive(Clone, Copy, Debug)]
enum Stage {
    /// Priming decision strobe one symbol before the first output (twice-rate
    /// detectors only).
    PrimeDecision,
    /// Priming mid strobe half a symbol before the first output (twice-rate
    /// detectors only).
    PrimeMid,
    /// Decision strobe: the recovered, visible symbol.
    Decision,
    /// Mid strobe between two decisions, consumed internally (twice-rate
    /// detectors only).
    Mid,
}

/// Generic symbol-timing synchronizer, parameterized by its timing-error
/// detector.
#[derive(Clone, Debug)]
pub struct SymbolSynchronizer<D = Gardner<f32>, T = f32> {
    sps: T,
    half_symbol: T,
    filter: LoopFilter<T>,
    detector: D,
    /// Fractional-delay interpolation bank.
    pfb: FractionalDelayPfb<T>,
    /// Recent pushed samples, indexed by absolute sample index modulo the
    /// buffer capacity.
    history: alloc::vec::Vec<Complex<T>>,
    /// Number of samples pushed so far (the absolute index of the next one).
    pushed: usize,
    /// Absolute index of the next decision strobe.
    position: T,
    /// Absolute index of the first decision strobe.
    start: T,
    stage: Stage,
}

impl<D, T> SymbolSynchronizer<D, T>
where
    D: TimingErrorDetector<T> + Default,
    T: Float + core::fmt::Debug,
{
    /// Creates a synchronizer for `sps` samples per symbol whose first
    /// recovered symbol lands near the fractional sample `start`, with the
    /// given normalized loop bandwidth (`Bn * T`) and damping factor.
    ///
    /// # Panics
    ///
    /// Panics if `sps` is zero.
    #[must_use]
    pub fn new(sps: usize, start: T, loop_bandwidth: T, damping: T) -> Self {
        Self::with_detector(sps, start, loop_bandwidth, damping, D::default())
    }
}

impl<D, T> SymbolSynchronizer<D, T>
where
    D: TimingErrorDetector<T>,
    T: Float + core::fmt::Debug,
{
    /// Creates a synchronizer using an explicitly-configured detector,
    /// otherwise as [`Self::new`].
    ///
    /// # Panics
    ///
    /// Panics if `sps` is zero.
    #[must_use]
    pub fn with_detector(sps: usize, start: T, loop_bandwidth: T, damping: T, detector: D) -> Self {
        assert!(sps > 0, "samples per symbol must be nonzero");
        let sps = T::from(sps).expect("samples per symbol is representable");
        let two = T::from(2.0).expect("2.0 is representable");
        let atten = T::from(TIMING_STOPBAND_ATTEN).expect("attenuation is representable");
        let pfb = FractionalDelayPfb::new(TIMING_PHASES, atten);
        let cap = pfb.taps_per_phase() + 2;
        Self {
            sps,
            half_symbol: sps / two,
            filter: LoopFilter::new(loop_bandwidth, damping),
            detector,
            pfb,
            history: alloc::vec![Complex::new(T::zero(), T::zero()); cap],
            pushed: 0,
            position: start,
            start,
            stage: if D::STROBE_RATE == 2 {
                Stage::PrimeDecision
            } else {
                Stage::Decision
            },
        }
    }

    /// Resets the synchronizer to re-run from a new `start` with fresh loop
    /// parameters, reusing the already-allocated buffers (so it is
    /// allocation-free: a streaming demodulator calls this once per burst,
    /// and picks up any runtime change to the loop bandwidth or damping).
    pub fn reset(&mut self, start: T, loop_bandwidth: T, damping: T) {
        self.filter = LoopFilter::new(loop_bandwidth, damping);
        self.detector.reset();
        self.pushed = 0;
        self.position = start;
        self.start = start;
        self.stage = if D::STROBE_RATE == 2 {
            Stage::PrimeDecision
        } else {
            Stage::Decision
        };
    }

    /// Feeds one full-rate matched-filter sample. Pull completed symbols with
    /// [`Self::next_symbol`].
    pub fn push(&mut self, sample: Complex<T>) {
        let capacity = self.history.len();
        self.history[self.pushed % capacity] = sample;
        self.pushed += 1;
    }

    /// The fractional (sub-sample) timing offset the loop has settled on, in
    /// samples (`[0, 1)`).
    #[must_use]
    pub fn fractional_offset(&self) -> T {
        self.position - self.position.floor()
    }

    /// The timing interpolator's group delay in samples: a strobe at position
    /// `p` interpolates the window `p - group_delay ..= p + group_delay`, so
    /// the first strobe needs this much lead-in.
    #[must_use]
    pub fn group_delay(&self) -> usize {
        self.pfb.group_delay()
    }

    /// Advances the interpolator and returns the next recovered symbol, or
    /// `None` until enough samples have been pushed to complete its strobe. A
    /// data-aided or decision-directed detector needs
    /// [`Self::apply_feedback`] called with the symbol's reference before the
    /// next call.
    ///
    /// # Panics
    ///
    /// Panics if the strobe position exceeds the range representable as an
    /// index.
    pub fn next_symbol(&mut self) -> Option<Complex<T>> {
        let phases = self.pfb.num_phases();
        let taps = self.pfb.taps_per_phase();
        let delay = self.pfb.group_delay();
        loop {
            let position = self.strobe_position();
            // The interpolation window is centred on `index`, spanning
            // `index - delay ..= index + delay`; clamp so it never runs before
            // sample zero.
            let index = position
                .max(T::one())
                .floor()
                .to_usize()
                .expect("strobe position is representable")
                .max(delay);
            // Wait until the newest window sample, `index + delay`, has been
            // pushed.
            if index + delay + 1 > self.pushed {
                return None;
            }

            let frac = position - T::from(index).expect("window index is representable");
            // Branch `q` interpolates at window-coord `delay + q/P`; with the
            // window based at `index - delay` that lands at `index + q/P`, so
            // `q = round(frac * P)`.
            let phase = (frac * T::from(phases).expect("phase count is representable"))
                .round()
                .max(T::zero())
                .to_usize()
                .expect("phase index is representable")
                .min(phases - 1);
            let base = index - delay;
            let strobe = self.pfb.interpolate(
                phase,
                (0..taps).map(|tap| &self.history[(base + tap) % self.history.len()]),
            );
            let derivative = self.pfb.derivative(
                phase,
                (0..taps).map(|tap| &self.history[(base + tap) % self.history.len()]),
            );
            let timing_error = self.detector.consume_strobe(strobe, Some(derivative));

            match self.stage {
                // Priming and mid strobes seed the detector; only decision
                // strobes are emitted.
                Stage::PrimeDecision => self.stage = Stage::PrimeMid,
                Stage::PrimeMid | Stage::Mid => self.stage = Stage::Decision,
                Stage::Decision => {
                    self.position = self.position + self.sps;
                    // A non-data-aided detector reports its error here;
                    // data-aided ones defer to `apply_feedback` and report
                    // `None`.
                    if let Some(error) = timing_error {
                        self.apply_timing_correction(error);
                    }
                    self.stage = if D::STROBE_RATE == 2 {
                        Stage::Mid
                    } else {
                        Stage::Decision
                    };
                    return Some(strobe);
                }
            }
        }
    }

    /// Supplies the `reference` (a known or sliced symbol) for the last
    /// symbol from [`Self::next_symbol`], letting a data-aided or
    /// decision-directed detector steer the loop. A non-data-aided detector
    /// ignores it.
    pub fn apply_feedback(&mut self, observed: Complex<T>, reference: Complex<T>) {
        if let Some(error) = self.detector.apply_feedback(observed, reference) {
            self.apply_timing_correction(error);
        }
    }

    /// Absolute fractional index of the strobe the current stage interpolates.
    fn strobe_position(&self) -> T {
        match self.stage {
            Stage::PrimeDecision => self.start - self.sps,
            Stage::PrimeMid => self.start - self.half_symbol,
            Stage::Decision => self.position,
            Stage::Mid => self.position - self.half_symbol,
        }
    }

    /// Nudges the next strobe position by the loop-filtered timing error.
    fn apply_timing_correction(&mut self, error: T) {
        self.position = self.position - self.filter.update(error);
    }
}

#[cfg(test)]
mod tests {
    use num_complex::Complex32;

    use super::SymbolSynchronizer;
    use crate::filters::fir::design::root_raised_cosine;
    use crate::filters::tracking::ted::{Gardner, MaximumLikelihood, MuellerMuller};

    /// A known unit-magnitude QPSK symbol stream of `count` symbols.
    fn qpsk_symbols(count: usize) -> alloc::vec::Vec<Complex32> {
        (0..count)
            .map(|k| match k % 4 {
                0 => Complex32::new(1.0, 0.0),
                1 => Complex32::new(0.0, 1.0),
                2 => Complex32::new(-1.0, 0.0),
                _ => Complex32::new(0.0, -1.0),
            })
            .collect()
    }

    /// Shapes a QPSK stream with a root-raised-cosine pulse and matched-filters
    /// it so symbol `k` lands at matched index `k * sps`.
    fn matched_qpsk(
        symbols: &[Complex32],
        span: usize,
        sps: usize,
        rolloff: f32,
    ) -> alloc::vec::Vec<Complex32> {
        let taps = root_raised_cosine::taps_vec(span, sps, rolloff);
        let upsampled_len = symbols.len() * sps + taps.len() - 1;
        let mut shaped = alloc::vec![Complex32::new(0.0, 0.0); upsampled_len];
        for (index, &symbol) in symbols.iter().enumerate() {
            for (offset, &tap) in taps.iter().enumerate() {
                shaped[index * sps + offset] += symbol * tap;
            }
        }
        let mut matched = alloc::vec![Complex32::new(0.0, 0.0); shaped.len() - taps.len() + 1];
        for (index, out) in matched.iter_mut().enumerate() {
            *out = shaped[index..index + taps.len()]
                .iter()
                .zip(taps.iter().rev())
                .map(|(&sample, &tap)| sample * tap)
                .sum();
        }
        // The cascade peaks `taps.len() - 1` samples in; drop the lead so
        // symbol `k` sits at index `k * sps`.
        matched.drain(..taps.len() - 1);
        matched
    }

    /// Runs a synchronizer over the stream, feeding data-aided references when
    /// `references` is set, and asserts the recovered symbols converge.
    fn assert_pulls_in<D>(data_aided: bool)
    where
        D: super::TimingErrorDetector<f32> + Default,
    {
        let sps = 8;
        let symbols = qpsk_symbols(40);
        let matched = matched_qpsk(&symbols, 8, sps, 0.35);

        // Start several symbols in, deliberately off by 0.35 samples.
        let first = 6;
        #[allow(clippy::cast_precision_loss)]
        let start = (first * sps) as f32 + 0.35;
        let mut synchronizer =
            SymbolSynchronizer::<D>::new(sps, start, 0.05, core::f32::consts::FRAC_1_SQRT_2);
        let mut recovered = alloc::vec::Vec::with_capacity(24);
        for &sample in &matched {
            synchronizer.push(sample);
            while let Some(symbol) = synchronizer.next_symbol() {
                if recovered.len() < 24 {
                    if data_aided {
                        let reference = symbols[first + recovered.len()];
                        synchronizer.apply_feedback(symbol, reference);
                    }
                    recovered.push(symbol);
                }
            }
        }
        assert_eq!(recovered.len(), 24, "expected 24 recovered symbols");

        // After the loop locks (skip the first few symbols), each strobe must
        // match its symbol within the truncated-RRC inter-symbol residual.
        for (index, strobe) in recovered.iter().enumerate().skip(6) {
            let expected = symbols[first + index];
            assert!(
                (strobe - expected).norm() < 0.15,
                "symbol {index} = {strobe} differs from {expected}"
            );
        }
    }

    #[test]
    fn gardner_loop_pulls_in_a_fractional_timing_offset() {
        assert_pulls_in::<Gardner>(false);
    }

    #[test]
    fn maximum_likelihood_loop_pulls_in_a_fractional_timing_offset() {
        assert_pulls_in::<MaximumLikelihood>(true);
    }

    #[test]
    fn mueller_muller_loop_pulls_in_a_fractional_timing_offset() {
        assert_pulls_in::<MuellerMuller>(true);
    }

    #[test]
    #[should_panic(expected = "samples per symbol must be nonzero")]
    fn rejects_zero_samples_per_symbol() {
        let _ = SymbolSynchronizer::<Gardner>::new(0, 0.0, 0.01, 1.0);
    }
}
