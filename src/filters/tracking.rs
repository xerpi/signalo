// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Synchronization-loop error detectors.
//!
//! Error detectors measure the residual timing (or phase) error of a
//! synchronization loop from the received samples. They feed a loop filter
//! such as [`LoopFilter`](crate::filters::iir::loop_filter::LoopFilter),
//! whose output steers the loop's interpolator or oscillator.

pub mod ted;
