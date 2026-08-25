//! The in-crate half of the `flash-attn` feature-isolation guard.
//!
//! `flash-attn` (the vendored FlashAttention-2 build — CUTLASS, a
//! minute of nvcc, a static archive) DEPENDS on `cuda` but must never be
//! IMPLIED by it: the `cuda` lane is what the release image and every
//! downstream `--features cuda` build enable. This test reads this crate's
//! own `Cargo.toml` and asserts the two edges that could leak it —
//! `default` and `cuda` — do not name `flash-attn`. The cross-crate
//! closure (no workspace crate's feature graph reaching
//! `jammi-kernels/flash-attn` from a consumer's features) is
//! `ci/scripts/check_flash_attn_closure.py`, which walks `cargo metadata`;
//! this file is the cheap, dependency-free cell that runs in every
//! default `cargo test -p jammi-kernels`.
//!
//! Default features (no `cuda`, no nvcc) — runs everywhere.

use std::path::Path;

/// The `[features]` table of this crate's manifest as `(name, deps)` pairs,
/// parsed structurally enough to be exact for the shape Cargo accepts
/// (`name = [ "a", "b/c", "dep:d" ]` on one line per feature, comments and
/// blank lines interleaved) and to fail loudly on anything else.
fn feature_table(manifest: &str) -> Vec<(String, Vec<String>)> {
    let mut in_features = false;
    let mut out = Vec::new();
    for raw in manifest.lines() {
        let line = raw.trim();
        if line.starts_with('[') {
            in_features = line == "[features]";
            continue;
        }
        if !in_features || line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (name, rhs) = line
            .split_once('=')
            .unwrap_or_else(|| panic!("unparsed [features] line: {raw:?}"));
        let rhs = rhs.trim();
        assert!(
            rhs.starts_with('[') && rhs.ends_with(']'),
            "feature {name:?}: expected a one-line array, got {rhs:?}"
        );
        let deps: Vec<String> = rhs[1..rhs.len() - 1]
            .split(',')
            .map(|s| s.trim().trim_matches('"').to_string())
            .filter(|s| !s.is_empty())
            .collect();
        out.push((name.trim().to_string(), deps));
    }
    assert!(!out.is_empty(), "no [features] table found");
    out
}

fn own_features() -> Vec<(String, Vec<String>)> {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml");
    let text = std::fs::read_to_string(&manifest)
        .unwrap_or_else(|e| panic!("read {}: {e}", manifest.display()));
    feature_table(&text)
}

#[test]
fn flash_attn_feature_exists_and_depends_on_cuda() {
    let table = own_features();
    let (_, deps) = table
        .iter()
        .find(|(n, _)| n == "flash-attn")
        .expect("`flash-attn` feature is declared");
    assert!(
        deps.iter().any(|d| d == "cuda"),
        "`flash-attn` must imply `cuda` (it links against candle's CUDA device); got {deps:?}"
    );
}

#[test]
fn neither_default_nor_cuda_implies_flash_attn() {
    let table = own_features();
    for edge in ["default", "cuda"] {
        let (_, deps) = table
            .iter()
            .find(|(n, _)| n == edge)
            .unwrap_or_else(|| panic!("feature `{edge}` is declared"));
        assert!(
            !deps
                .iter()
                .any(|d| d == "flash-attn" || d.starts_with("flash-attn/")),
            "feature `{edge}` must not enable `flash-attn`; got {deps:?}"
        );
    }
    // `default` is empty by construction (every fused path is opt-in).
    let (_, default) = table.iter().find(|(n, _)| n == "default").unwrap();
    assert!(
        default.is_empty(),
        "default features must stay empty, got {default:?}"
    );
}

#[test]
fn parser_detects_a_leaked_edge() {
    // Red control for the parser: the assertion above is only as good as
    // the parse, so prove the parse sees a leaked edge in a synthetic
    // manifest of the same shape.
    let leaked = r#"
[package]
name = "x"

[features]
default = []
# a comment
cuda = ["dep:bindgen_cuda", "flash-attn"]
flash-attn = ["cuda"]

[[test]]
name = "y"
"#;
    let table = feature_table(leaked);
    let (_, cuda) = table.iter().find(|(n, _)| n == "cuda").unwrap();
    assert!(cuda.iter().any(|d| d == "flash-attn"));
    assert_eq!(table.len(), 3);
}
