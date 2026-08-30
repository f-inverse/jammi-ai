//! Does an error message name an out-of-memory condition? The single home
//! for that question in the crate, shared by the inference-side
//! batch-halving retry (`crate::inference::runner::InferenceRunner::is_oom_error`)
//! and the training-side OOM guidance classifier
//! (`crate::fine_tune::worker::classify_training_oom`) — neither module
//! reaches into the other's private classifier; both import from here.
//!
//! One spelling table ([`OOM_SPELLINGS`]), one strictness column, so the two
//! predicates below cannot drift silently apart:
//!
//! - [`is_oom_message`] (the **retry** predicate) matches every table entry.
//!   A false positive here costs the inference retry one wasted
//!   batch-halving attempt: bounded, self-correcting (the next attempt
//!   either succeeds smaller or hits the real failure again unchanged). It
//!   is never attached to a durable, caller-facing message, so it also
//!   carries the bare `oom` token (#319: only a genuine OOM should take the
//!   retry, but a false-positive retry is cheap — the token's ambiguity is
//!   an acceptable cost here specifically because it's bounded).
//! - [`is_definite_oom_message`] (the **strict** predicate) matches only the
//!   table entries marked `strict: true` — the long, unambiguous spellings.
//!   Its caller's output (the training classifier's rewritten message) lands
//!   in a job's durable `error_message`, read directly by `jammi train
//!   status` / a Python `job.status()`. The bare `oom` token is deliberately
//!   EXCLUDED from strict: a 3-byte token adjacent to a path/version
//!   separator (`/data/oom/train.csv`, `train-oom-sample.csv`,
//!   `acme/oom-detector`, a column literally named `"oom"`, a `--oom` flag)
//!   is not safe evidence inside user-controlled text, and no word-boundary
//!   check rescues it — `/data/oom/train.csv` has separators on both sides
//!   of `oom` and still names nothing about memory. Every real driver/
//!   framework OOM string carries one of the long spellings instead, so
//!   dropping the bare token costs nothing here. All strict patterns are
//!   matched as plain (non-word-boundary) substrings — being long, none of
//!   them are at meaningful risk of appearing as a fragment of an unrelated
//!   identifier the way a 3-byte token is.
//!
//! Every pattern here is lower-case ASCII; both predicates require the
//! caller to lower-case `msg_lower` first (non-ASCII case-adjacency doesn't
//! matter once the bare token — the only entry short enough for adjacency to
//! be a real question — is gone).
//!
//! **Known residual (not silent):** cuBLAS surfaces a workspace-allocation
//! failure as `CublasError(CUBLAS_STATUS_ALLOC_FAILED)`, which names no
//! "out of memory" spelling at all. Neither predicate here recognizes it —
//! the test `cublas_alloc_failed_is_a_known_uncovered_residual` pins this
//! gap executably rather than leaving it merely documented. Widening either
//! predicate to catch it is a future table entry, not implied by this doc.

/// One entry in the shared OOM spelling table.
struct OomSpelling {
    /// The lower-case substring to match. The caller lower-cases the target
    /// message; every pattern here is already lower-case.
    pattern: &'static str,
    /// Included in the strict (training-classifier) predicate
    /// ([`is_definite_oom_message`])? Every entry is included in the retry
    /// predicate ([`is_oom_message`]) regardless of this flag.
    strict: bool,
}

/// The single spelling table both predicates below read. Add a new spelling
/// here — never as a fourth hand-written pattern list elsewhere.
const OOM_SPELLINGS: &[OomSpelling] = &[
    // Long, unambiguous spellings: safe as plain substrings for both
    // predicates. Covers CUDA's `CUDA_ERROR_OUT_OF_MEMORY` /
    // `cudaOutOfMemory`, HIP's `hipErrorOutOfMemory`, Metal's
    // `MPS_ERROR_OUT_OF_MEMORY`, PyTorch's `torch.cuda.OutOfMemoryError`,
    // and candle's `OutOfMemory`, among others, via one of the three
    // separator styles (space / underscore / none).
    OomSpelling {
        pattern: "out of memory",
        strict: true,
    },
    OomSpelling {
        pattern: "out_of_memory",
        strict: true,
    },
    OomSpelling {
        pattern: "outofmemory",
        strict: true,
    },
    // Short and ambiguous: retry-only (see the module doc for why a bare
    // "oom" token is unsafe evidence for the strict predicate even with a
    // word-boundary check).
    OomSpelling {
        pattern: "oom",
        strict: false,
    },
];

/// The retry predicate: matches every table entry as a plain substring. See
/// the module doc for why this is safe here (bounded, self-correcting, never
/// caller-facing) but not for [`is_definite_oom_message`].
///
/// `msg_lower` must already be lower-cased by the caller.
pub(crate) fn is_oom_message(msg_lower: &str) -> bool {
    OOM_SPELLINGS.iter().any(|s| msg_lower.contains(s.pattern))
}

/// The strict predicate: matches only the table entries marked `strict:
/// true` — the long, unambiguous spellings — as plain substrings. See the
/// module doc for why the bare `oom` token is excluded entirely rather than
/// word-boundary-checked.
///
/// `msg_lower` must already be lower-cased by the caller.
pub(crate) fn is_definite_oom_message(msg_lower: &str) -> bool {
    OOM_SPELLINGS
        .iter()
        .filter(|s| s.strict)
        .any(|s| msg_lower.contains(s.pattern))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Ties the table's `strict` column directly to both predicates'
    /// behavior on each entry in isolation, so the two predicates cannot
    /// drift silently apart from the table that is supposed to be their only
    /// source of truth.
    #[test]
    fn every_table_entry_matches_its_declared_predicates_in_isolation() {
        for spelling in OOM_SPELLINGS {
            assert!(
                is_oom_message(spelling.pattern),
                "retry predicate must match its own table entry {:?}",
                spelling.pattern
            );
            assert_eq!(
                is_definite_oom_message(spelling.pattern),
                spelling.strict,
                "strict predicate disagrees with the table's strict flag for {:?}",
                spelling.pattern
            );
        }
    }

    /// Table-driven divergence test over the two adversarial sets an audit
    /// exercised: false positives (a bare "oom" fragment embedded in
    /// user-controlled text) and false negatives (real long driver/framework
    /// spellings). Asserts BOTH predicates' answers for every string, so a
    /// regression in either predicate — or the two drifting apart — fails
    /// here directly.
    #[test]
    fn oom_predicates_diverge_exactly_on_the_documented_adversarial_sets() {
        // Strict must reject every one of these (no "out of memory" /
        // "out_of_memory" / "outofmemory" spelling present — only a bare
        // "oom" fragment, some separator-adjacent on both sides). Retry
        // still matches all of them via the bare "oom" table entry.
        let false_positives = [
            "/data/oom/train.csv",
            "train-oom-sample.csv",
            "acme/oom-detector",
            "column \"oom\"",
            "--oom",
            "cannot locate 'bigscience/bloomz-560m' in hf hub cache",
            "/data/classroom-corpus/train.csv missing",
        ];
        for m in false_positives {
            assert!(
                !is_definite_oom_message(m),
                "strict predicate must reject a bare 'oom' fragment in user text: {m}"
            );
            assert!(
                is_oom_message(m),
                "retry predicate still matches the bare 'oom' fragment: {m}"
            );
        }

        // Real long spellings strict must catch as plain substrings — no
        // word-boundary check needed or used.
        let real_long_spellings = [
            "torch.cuda.outofmemoryerror",
            "cudaoutofmemory",
            "hiperroroutofmemory",
            "mps_error_out_of_memory",
            "error_out_of_memory",
            "my_cuda_error_out_of_memory",
        ];
        for m in real_long_spellings {
            assert!(
                is_definite_oom_message(m),
                "strict predicate must catch the real long spelling: {m}"
            );
            assert!(is_oom_message(m), "retry predicate must catch: {m}");
        }
    }

    /// Documents a currently NOT-covered spelling (cuBLAS workspace
    /// allocation failure) — the module doc's residual, executable rather
    /// than silent. If this starts failing (because a future change widens
    /// a pattern to catch it), update the module doc's residual paragraph to
    /// match — don't just delete this test.
    #[test]
    fn cublas_alloc_failed_is_a_known_uncovered_residual() {
        let m = "cublaserror(cublas_status_alloc_failed)";
        assert!(
            !is_definite_oom_message(m),
            "known gap: cuBLAS's alloc-failed spelling names no OOM pattern in the table"
        );
        assert!(
            !is_oom_message(m),
            "known gap: cuBLAS's alloc-failed spelling names no OOM pattern in the table"
        );
    }
}
