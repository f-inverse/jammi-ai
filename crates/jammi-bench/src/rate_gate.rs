//! The throughput rate-regression gate: a generic same-box baseline guard.
//!
//! A *rate* (throughput, QPS, pairs/s) is not portable the way the recall
//! fraction in [`crate::recall`] is — it is a property of the box that produced
//! it, so a committed rate baseline is a *same-box* reference, refreshed by hand
//! when the emit box changes, not a number a different machine can re-derive.
//! What stays portable is the *shape* of the gate: a measured rate must not fall
//! more than a fixed fraction below the committed baseline. That shape mirrors
//! the recall floor's `measured >= floor` precedent — a `>=`, never an equality
//! and never a bit-compare — but gates a rate against a relative threshold
//! rather than an absolute fraction against an absolute floor.
//!
//! ## Why a relative threshold, and why a generous one
//!
//! The gate runs on a shared runner whose throughput jitters with co-tenant
//! load, CPU governor state, and allocator warmth. An absolute floor would
//! either flap (set too high) or never bite (set too low). A *relative* drop
//! threshold tracks the baseline as the box's own reference and asks only "did
//! this run regress by more than X% against what the box committed?". The
//! threshold is set generously ([`DEFAULT_REGRESSION_THRESHOLD`], 30%) because
//! the load-bearing failure this gate must catch is a *structural* regression —
//! an algorithm that went quadratic, a lock that serialized a parallel path, a
//! dropped fast path — which collapses throughput by far more than a third. A
//! tighter threshold would trade that real signal for false alarms on runner
//! noise; the recall floor makes the same trade with its safety-margin headroom.
//!
//! ## Reusable across tiers
//!
//! The gate is a pure function over `(measured, baseline, threshold)`: any tier
//! that measures a throughput — the training tier here, and the inference /
//! embedding tiers that follow — commits its own baseline rate and calls
//! [`RateGate::evaluate`] with it. The mechanism names no tier and carries no
//! tier-specific knob, so a new tier reuses it by committing a baseline and one
//! call, never by copying this logic.

/// The default relative drop a measured rate may fall below its baseline before
/// the gate fails: 30%. Generous on purpose — see the module docs. A tier may
/// pass its own threshold to [`RateGate::evaluate`] when it has a measured
/// reason to be tighter or looser, but the default is the one shared-runner
/// noise was sized against.
pub const DEFAULT_REGRESSION_THRESHOLD: f64 = 0.30;

/// The verdict of one rate-regression check: the measured rate, the baseline it
/// was gated against, the threshold applied, and the derived floor and pass
/// flag. Every input travels with the verdict so a failing gate prints the full
/// arithmetic rather than a bare boolean.
#[derive(Debug, Clone, Copy)]
pub struct RateGate {
    /// The rate this run measured.
    pub measured: f64,
    /// The committed same-box baseline rate the measurement is gated against.
    pub baseline: f64,
    /// The relative drop fraction applied: the measured rate must stay at or
    /// above `baseline * (1 - threshold)`.
    pub threshold: f64,
    /// The absolute floor the threshold derived from the baseline:
    /// `baseline * (1 - threshold)`.
    pub floor: f64,
    /// Whether the gate held: `measured >= floor`.
    pub passed: bool,
}

impl RateGate {
    /// Evaluate a measured rate against a committed baseline at `threshold`
    /// relative drop.
    ///
    /// Fails when `measured < baseline * (1 - threshold)`. A non-finite or
    /// non-positive baseline cannot anchor a relative gate — it is the absence
    /// of a real reference, so the gate fails closed (floor `0`, `passed`
    /// `false`) rather than vacuously passing against a meaningless baseline.
    /// `threshold` outside `[0, 1)` is clamped into range so a `1 - threshold`
    /// floor stays a fraction of the baseline.
    pub fn evaluate(measured: f64, baseline: f64, threshold: f64) -> Self {
        if !baseline.is_finite() || baseline <= 0.0 {
            return Self {
                measured,
                baseline,
                threshold,
                floor: 0.0,
                passed: false,
            };
        }
        let threshold = threshold.clamp(0.0, 1.0 - f64::EPSILON);
        let floor = baseline * (1.0 - threshold);
        Self {
            measured,
            baseline,
            threshold,
            floor,
            passed: measured >= floor,
        }
    }

    /// A human-readable one-line summary of the verdict, with the full
    /// arithmetic so a failure surfaces the numbers, not just a boolean.
    pub fn detail(&self) -> String {
        format!(
            "measured {:.1} vs baseline {:.1} ({}; floor {:.1} = baseline·(1−{:.2}))",
            self.measured,
            self.baseline,
            if self.passed { "PASS" } else { "REGRESSED" },
            self.floor,
            self.threshold,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A rate at the baseline, above it, and just inside the floor all pass; the
    /// gate is a `>=` against the derived floor, not an equality.
    #[test]
    fn at_or_above_floor_passes() {
        let g = RateGate::evaluate(100.0, 100.0, 0.30);
        assert!(g.passed);
        assert_eq!(g.floor, 70.0);

        // Above the baseline (a faster run) passes.
        assert!(RateGate::evaluate(120.0, 100.0, 0.30).passed);
        // Exactly on the floor passes — the boundary is inclusive.
        assert!(RateGate::evaluate(70.0, 100.0, 0.30).passed);
    }

    /// A rate that fell more than the threshold below the baseline fails — the
    /// gate has teeth (RC1: an assertion must be able to fail).
    #[test]
    fn below_floor_fails() {
        let g = RateGate::evaluate(69.9, 100.0, 0.30);
        assert!(!g.passed, "a >30% drop must fail the gate");
        assert_eq!(g.floor, 70.0);
    }

    /// A non-positive or non-finite baseline cannot anchor a relative gate, so
    /// the gate fails closed rather than passing vacuously.
    #[test]
    fn degenerate_baseline_fails_closed() {
        assert!(!RateGate::evaluate(1000.0, 0.0, 0.30).passed);
        assert!(!RateGate::evaluate(1000.0, -5.0, 0.30).passed);
        assert!(!RateGate::evaluate(1000.0, f64::NAN, 0.30).passed);
    }
}
