//! CI-only dump of `jammi_kernels::admission::PROBED_OPS`'s own registry
//! and report keys as JSON — see this crate's own `Cargo.toml` description
//! for why it exists (`ci/scripts/perf/test_finetune_ab_disable_op_keys.py`'s
//! sole consumer).
//!
//! Usage: `cargo run --release -p probed-ops-index` — JSON on stdout,
//! nothing else (safe to pipe straight into `json.load`). No arguments:
//! this tool has exactly one table to read, unlike `ci/tools/symbol-index`
//! (which takes root directories to scan) — `PROBED_OPS` is a single,
//! already-linked-in constant, not something to walk a directory tree for.
//!
//! Output shape: `{"registry_keys": [...], "report_keys": [...]}`, both
//! sorted, deduplicated string arrays. `registry_keys` is
//! [`jammi_kernels::admission::ProbedOp::all_registry_keys`] over every
//! [`jammi_kernels::admission::PROBED_OPS`] row (the exact set
//! `JAMMI_EAGER_DISABLE_OP_KEYS` in `ci/scripts/perf/finetune_ab.sh` draws
//! its own "live standalone" subset from, minus that script's own
//! `KNOWN_NON_STANDALONE_REGISTRY_KEYS`); `report_keys` is each row's own
//! dtype-neutral `report_key`, included for any future consumer that needs
//! the OTHER key space this same table carries (unused by
//! `test_finetune_ab_disable_op_keys.py` today, but cheap to emit
//! alongside `registry_keys` from the one process that already has the
//! table linked in).

use std::collections::BTreeSet;

use jammi_kernels::admission::PROBED_OPS;

fn main() {
    let registry_keys: BTreeSet<&str> = PROBED_OPS
        .iter()
        .flat_map(|op| op.all_registry_keys())
        .collect();
    let report_keys: BTreeSet<&str> = PROBED_OPS.iter().map(|op| op.report_key()).collect();
    let out = serde_json::json!({
        "registry_keys": registry_keys,
        "report_keys": report_keys,
    });
    println!("{out}");
}
