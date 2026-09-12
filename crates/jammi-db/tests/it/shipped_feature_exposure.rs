//! The set of cargo features a shipped `jammi-server` artifact carries is DERIVED here and
//! gated against `deny.toml`, never hand-maintained as prose that drifts from the release
//! lanes. Four rounds of this exact unit blocked on the same class: a hand-written sentence
//! in `deny.toml` naming "the release image and the CUDA lanes" as the features-exposed set,
//! silently understating it every time a lane was added or a literal `--features` list moved.
//!
//! Six artifact families ship a `jammi-server` build. Three (the CUDA tarball, wheel and
//! image) already read `ci/release-feature-manifest.json`'s `lanes.<key>.cargo_features` at
//! build time; this test reads the same manifest for those three, so their feature list can
//! never duplicate what the release workflows themselves already treat as authoritative. The
//! other three (the CPU PyPI wheel, the CPU tarball, the CPU container image) carry a literal
//! `--features`/`cargo_features:` list in their own workflow/Dockerfile — `ci/release-feature-
//! manifest.json` is NOT the home for that data: `check_release_manifest.py` (human-amend-only,
//! `SWARM_GATE_TOUCHED`-guarded) requires every entry under the manifest's `lanes` object to
//! carry a `capabilities` block byte-identical to every other lane's, a property scoped
//! explicitly to the three CUDA capability-proof lanes ("the manifest's three CUDA release
//! lanes describe ONE shipped capability surface" — that file's own module doc). Adding a
//! non-CUDA lane there would either fail that check (a non-identical `capabilities` block) or
//! falsely claim a CPU-only build carries CUDA capabilities it does not have. So the three CPU
//! families' `cargo_features` and their one literal build-site each are declared here instead —
//! still exactly one place per family, just not the same file as the CUDA three.
//!
//! This lives as an it-test rather than a `ci/scripts/check_*.py` gate for the same reason
//! `bans_names_resolve.rs` does: gate scripts are a human-amend-only surface
//! (`SWARM_GATE_TOUCHED`), `toml`/`serde_json` are already jammi-db dependencies, and the
//! hermetic `cargo test --workspace` lane already runs this binary on every PR.
//!
//! What this test asserts, all fail-closed (never a silent skip):
//!   1. Every CPU family's literal build site still carries its expected feature-list template,
//!      at its expected occurrence count in its file (drift on a release lane's own `--features`
//!      line is caught here instead of at release time).
//!   2. Every `.lanes["<key>"]` string any workflow reads resolves to a real lane in
//!      `ci/release-feature-manifest.json` (a misspelled lane key is a `jq: null` failure that
//!      `release-binaries.yml` cannot surface pre-merge — it has no `pull_request` trigger — so
//!      this test is the only pre-merge catch for it).
//!   3. `deny.toml` carries a GENERATED, marker-delimited exposure line for every advisory whose
//!      reason names a cargo-gated feature, byte-equal to what today's family data computes.
//!      `exposure = [...]` as a structured key inside `[[advisories.ignore]]` is not an option:
//!      measured on cargo-deny 0.20.2, it is `error[unexpected-keys]` and rejects the WHOLE
//!      config. This test never parses `deny.toml`'s prose — it generates the expected line and
//!      checks the file contains that exact line, byte for byte.
//!   4. A COMPLETENESS sweep: every line across `.github/workflows/**` and `Dockerfile` that
//!      declares a literal (non-templated) cargo-feature list for a `jammi-server` build is
//!      either one of the six families' declared sites, a manifest read (assertion 2), or an
//!      explicit, reasoned `NOT_SHIPPED` allowlist entry. This is the half that actually closes
//!      the class: assertions 1-3 alone only catch a family whose exposure UNDER-states a lane
//!      already enumerated; they cannot catch a real shipped lane nobody enumerated at all. A
//!      unit test below (`completeness_sweep_flags_an_unaccounted_feature_site`) proves the
//!      sweep's classifier catches a synthetic unaccounted site without depending on today's
//!      tree ever regressing.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// The workspace root, walked up from this crate's manifest directory
/// (`<root>/crates/jammi-db`). Panics rather than skipping if the layout ever changes, so this
/// test cannot silently stop checking anything (same pattern `bans_names_resolve.rs` uses).
fn workspace_root() -> PathBuf {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let root = manifest
        .ancestors()
        .find(|dir| dir.join("deny.toml").is_file() && dir.join("Cargo.lock").is_file())
        .unwrap_or_else(|| {
            panic!(
                "no ancestor of {} holds both deny.toml and Cargo.lock",
                manifest.display()
            )
        });
    root.to_path_buf()
}

fn read_to_string(path: &Path) -> String {
    std::fs::read_to_string(path).unwrap_or_else(|e| panic!("reading {}: {e}", path.display()))
}

// ---------------------------------------------------------------------------
// The six shipped `jammi-server` artifact families, in the order the exposure
// line lists them.
// ---------------------------------------------------------------------------

const FAMILY_ORDER: &[&str] = &[
    "cpu-wheel",
    "cpu-tarball",
    "cpu-image",
    "cu12-tarball",
    "cu12-wheel",
    "cu12-image",
];

/// One literal (non-templated) build site: `template` is the exact substring expected on a
/// build line in `file`, `count` is how many times it must occur, and `bare_value` is the bare
/// comma-separated feature list the completeness sweep's classifier extracts from that same
/// line (so a declared site and the sweep agree on what "covered" means).
struct LiteralSite {
    file: &'static str,
    template: &'static str,
    count: usize,
    bare_value: &'static str,
}

struct CpuFamily {
    key: &'static str,
    cargo_features: &'static [&'static str],
    site: LiteralSite,
}

const CPU_FAMILIES: &[CpuFamily] = &[
    CpuFamily {
        key: "cpu-wheel",
        cargo_features: &["jetstream-broker", "storage-cloud"],
        site: LiteralSite {
            file: ".github/workflows/pypi-server.yml",
            template: "cargo_features: jetstream-broker,storage-cloud",
            count: 2, // x86_64 leg + aarch64 leg
            bare_value: "jetstream-broker,storage-cloud",
        },
    },
    CpuFamily {
        key: "cpu-tarball",
        cargo_features: &["jetstream-broker", "storage-cloud"],
        site: LiteralSite {
            file: ".github/workflows/release-binaries.yml",
            template: "--features jetstream-broker,storage-cloud",
            count: 1, // one templated `run:` line, matrixed over both targets
            bare_value: "jetstream-broker,storage-cloud",
        },
    },
    CpuFamily {
        key: "cpu-image",
        cargo_features: &["jetstream-broker", "storage-cloud"],
        site: LiteralSite {
            file: "Dockerfile",
            template: "--features jammi-server/jetstream-broker,jammi-server/storage-cloud",
            count: 1,
            bare_value: "jammi-server/jetstream-broker,jammi-server/storage-cloud",
        },
    },
];

/// The three CUDA families' manifest lane keys. Their `cargo_features` are read from
/// `ci/release-feature-manifest.json` at test time, never duplicated here.
const CUDA_FAMILIES: &[&str] = &["cu12-tarball", "cu12-wheel", "cu12-image"];

/// Every advisory this test derives an exposure line for, and the single cargo feature that
/// gates the code path the advisory is in. Lives here (not in `ci/release-feature-
/// manifest.json`, which has no advisory-facing section) because nothing besides this test and
/// `deny.toml` needs it, and it is exactly one place either way.
const GATED_ADVISORIES: &[(&str, &str)] = &[
    ("RUSTSEC-2026-0194", "storage-cloud"),
    ("RUSTSEC-2026-0195", "storage-cloud"),
];

/// Build sites that carry a literal `jammi-server` feature list but are NOT a shipped artifact.
/// Matched by (file, substring-of-the-classifier's-extracted-value). Every entry must carry a
/// reason; an unreasoned allowlist is exactly the thing this test exists to prevent.
struct NotShipped {
    file: &'static str,
    literal: &'static str,
    #[allow(dead_code)] // documentation only; not read by the assertions, kept for the reviewer
    reason: &'static str,
}

const NOT_SHIPPED: &[NotShipped] = &[
    NotShipped {
        file: ".github/workflows/distributed.yml",
        literal: "storage-s3",
        reason: "a CI test build for the distributed harness (cargo build -p jammi-server, \
                  never packaged or published) -- not a shipped artifact",
    },
    NotShipped {
        file: ".github/workflows/ci.yml",
        literal: "test-hooks",
        reason: "the gated-test-surfaces clippy step's compile-only feature closure (cargo \
                  clippy -p jammi-server --tests --features test-hooks -- -D warnings) -- lints \
                  that `jammi-server`'s test targets (and the `CARGO_BIN_EXE_jammi-server` they \
                  spawn) still compile with the engine's rendezvous hooks the `jammi-ai = { \
                  features = [\"test-hooks\"] }` dev-dependency union pulls in; `--tests` never \
                  builds the shipped `jammi-server` binary, so nothing here reaches a release \
                  artifact",
    },
    NotShipped {
        file: "Dockerfile",
        literal: "cuda,jetstream-broker,storage-cloud",
        reason: "the builder-cuda stage's `ARG CARGO_FEATURES` default, used only by a bare \
                  `docker build --build-arg RUNTIME_VARIANT=runtime-cuda` with no CARGO_FEATURES \
                  override; every server-image.yml job that actually publishes or PR-verifies \
                  the CUDA image (build-and-push-cu12, build-cuda-pr) passes CARGO_FEATURES \
                  explicitly from the cu12-image manifest lane, so this default never determines \
                  what ships",
    },
];

fn manifest_json(root: &Path) -> serde_json::Value {
    let text = read_to_string(&root.join("ci/release-feature-manifest.json"));
    serde_json::from_str(&text)
        .unwrap_or_else(|e| panic!("ci/release-feature-manifest.json does not parse as JSON: {e}"))
}

fn manifest_lane_features(manifest: &serde_json::Value, lane: &str) -> Vec<String> {
    manifest
        .get("lanes")
        .and_then(|l| l.get(lane))
        .and_then(|l| l.get("cargo_features"))
        .and_then(|f| f.as_array())
        .unwrap_or_else(|| {
            panic!("ci/release-feature-manifest.json has no lanes.{lane}.cargo_features array")
        })
        .iter()
        .map(|v| {
            v.as_str()
                .unwrap_or_else(|| panic!("lanes.{lane}.cargo_features has a non-string entry"))
                .to_string()
        })
        .collect()
}

/// Every family (of the six) that carries `feature`, in `FAMILY_ORDER`.
fn families_carrying(
    feature: &str,
    cuda_features: &BTreeMap<&str, Vec<String>>,
) -> Vec<&'static str> {
    let mut carrying: Vec<&'static str> = Vec::new();
    for fam in CPU_FAMILIES {
        if fam.cargo_features.contains(&feature) {
            carrying.push(fam.key);
        }
    }
    for key in CUDA_FAMILIES {
        if cuda_features
            .get(key)
            .is_some_and(|feats| feats.iter().any(|f| f == feature))
        {
            carrying.push(*key);
        }
    }
    FAMILY_ORDER
        .iter()
        .filter(|k| carrying.contains(k))
        .copied()
        .collect()
}

fn generated_exposure_line(advisory_id: &str, families: &[&str]) -> String {
    format!("# EXPOSURE[{advisory_id}] = {}", families.join(", "))
}

#[test]
fn cpu_family_literal_sites_match_their_expected_template_and_count() {
    let root = workspace_root();
    for fam in CPU_FAMILIES {
        let content = read_to_string(&root.join(fam.site.file));
        let occurrences = content.matches(fam.site.template).count();
        assert_eq!(
            occurrences, fam.site.count,
            "family `{}`: expected `{}` to occur {} time(s) in {}, found {} -- a release \
             lane's own --features/cargo_features line moved without this test's declared \
             site moving with it",
            fam.key, fam.site.template, fam.site.count, fam.site.file, occurrences
        );
    }
}

#[test]
fn every_dot_lanes_key_in_a_workflow_resolves_in_the_manifest() {
    let root = workspace_root();
    let manifest = manifest_json(&root);
    let lanes = manifest
        .get("lanes")
        .and_then(|l| l.as_object())
        .expect("manifest has a `lanes` object");

    let workflows_dir = root.join(".github/workflows");
    let mut unresolved: Vec<String> = Vec::new();
    for entry in std::fs::read_dir(&workflows_dir)
        .unwrap_or_else(|e| panic!("reading {}: {e}", workflows_dir.display()))
    {
        let entry = entry.expect("dir entry");
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("yml") {
            continue;
        }
        let content = read_to_string(&path);
        for key in extract_lane_keys(&content) {
            if !lanes.contains_key(&key) {
                unresolved.push(format!(
                    "{}: .lanes[\"{key}\"] resolves to no lane in ci/release-feature-manifest.json",
                    path.display()
                ));
            }
        }
    }
    assert!(
        unresolved.is_empty(),
        "unresolved manifest lane key(s), each a misspelling that would only surface at \
         release time (release-binaries.yml has no pull_request trigger): {unresolved:#?}"
    );
}

/// Every `.lanes["<key>"]` substring in `content`, in order of appearance.
fn extract_lane_keys(content: &str) -> Vec<String> {
    let marker = ".lanes[\"";
    let mut keys = Vec::new();
    let mut cursor = 0usize;
    while let Some(rel) = content[cursor..].find(marker) {
        let start = cursor + rel + marker.len();
        match content[start..].find('"') {
            Some(rel_end) => {
                let end = start + rel_end;
                keys.push(content[start..end].to_string());
                cursor = end + 1;
            }
            None => break,
        }
    }
    keys
}

#[test]
fn deny_toml_carries_the_generated_exposure_line_for_every_gated_advisory() {
    let root = workspace_root();
    let manifest = manifest_json(&root);
    let cuda_features: BTreeMap<&str, Vec<String>> = CUDA_FAMILIES
        .iter()
        .map(|lane| (*lane, manifest_lane_features(&manifest, lane)))
        .collect();

    let deny_text = read_to_string(&root.join("deny.toml"));

    let mut missing: Vec<String> = Vec::new();
    for (advisory_id, feature) in GATED_ADVISORIES {
        let families = families_carrying(feature, &cuda_features);
        assert!(
            !families.is_empty(),
            "advisory {advisory_id} names gated feature `{feature}`, but no shipped family's \
             cargo_features carries it -- either the advisory map or a family's feature list is \
             stale"
        );
        let expected = generated_exposure_line(advisory_id, &families);
        if !deny_text.lines().any(|l| l.trim() == expected) {
            missing.push(expected);
        }
    }
    assert!(
        missing.is_empty(),
        "deny.toml is missing (or has a stale) GENERATED exposure line -- add these exact \
         lines verbatim inside the `[[advisories]] ignore` array's reason block (never a \
         structured `exposure = [...]` key: measured on cargo-deny 0.20.2 that rejects the \
         WHOLE config with `error[unexpected-keys]`):\n{}",
        missing.join("\n")
    );
}

// ---------------------------------------------------------------------------
// The completeness sweep (R4(c)): every literal jammi-server feature-building
// site across `.github/workflows/**` and `Dockerfile` is accounted for.
// ---------------------------------------------------------------------------

/// True if `line` names a jammi-server build (`-p jammi-server`, `--package jammi-server`, or a
/// `jammi-server/`-prefixed feature).
fn is_jammi_server_context(line: &str) -> bool {
    line.contains("-p jammi-server")
        || line.contains("--package jammi-server")
        || line.contains("jammi-server/")
}

/// True if `value` is a concrete (non-templated) feature-list token: only identifier
/// characters, `-`, `_`, `,` and `/` -- never `$`, `{`, `(`, which mark a shell/GitHub-Actions
/// expression that resolves elsewhere (already covered by the manifest-key check above, or by a
/// literal at its own origin site).
fn is_literal_value(value: &str) -> bool {
    !value.is_empty()
        && value
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | ',' | '/'))
}

fn first_token(s: &str) -> &str {
    s.split_whitespace()
        .next()
        .unwrap_or("")
        .trim_matches(|c| c == '"' || c == '\\')
}

/// If `line` declares a concrete jammi-server cargo-feature list, return that bare value.
/// Pure and file-content-agnostic so it can be unit-tested against synthetic input.
fn literal_feature_value(line: &str) -> Option<String> {
    let t = line.trim();

    if let Some(idx) = t.find("--features") {
        let rest = &t[idx + "--features".len()..];
        let value = first_token(rest);
        if is_jammi_server_context(t) && is_literal_value(value) {
            return Some(value.to_string());
        }
    }
    if let Some(rest) = t.strip_prefix("cargo_features:") {
        let value = rest.trim();
        if is_literal_value(value) {
            return Some(value.to_string());
        }
    }
    if let Some(rest) = t.strip_prefix("ARG CARGO_FEATURES=") {
        let value = rest.trim();
        if is_literal_value(value) {
            return Some(value.to_string());
        }
    }
    None
}

/// Every declared literal site's (file, bare_value), used to recognize a line as "already
/// accounted for by a declared family site".
fn declared_sites() -> Vec<(&'static str, &'static str)> {
    CPU_FAMILIES
        .iter()
        .map(|f| (f.site.file, f.site.bare_value))
        .collect()
}

/// The sweep's classifier, applied to one file's content. Returns one finding string per
/// unaccounted line. Pure (file path + content in, no disk access), so it is exercised both
/// against the real tree and against synthetic content in the unit test below.
fn sweep_unaccounted(
    file: &str,
    content: &str,
    sites: &[(&str, &str)],
    not_shipped: &[NotShipped],
) -> Vec<String> {
    let mut findings = Vec::new();
    for (lineno, line) in content.lines().enumerate() {
        let Some(value) = literal_feature_value(line) else {
            continue;
        };
        let covered_by_site = sites.iter().any(|(f, v)| *f == file && *v == value);
        let covered_by_allowlist = not_shipped
            .iter()
            .any(|ns| ns.file == file && value.contains(ns.literal));
        if !covered_by_site && !covered_by_allowlist {
            findings.push(format!(
                "{file}:{}: unaccounted jammi-server feature site `{value}` -- not a declared \
                 family literal_site, not a manifest read, not in NOT_SHIPPED",
                lineno + 1
            ));
        }
    }
    findings
}

#[test]
fn completeness_sweep_over_workflows_and_dockerfile_has_no_unaccounted_site() {
    let root = workspace_root();
    let sites = declared_sites();

    let mut findings: Vec<String> = Vec::new();

    let workflows_dir = root.join(".github/workflows");
    for entry in std::fs::read_dir(&workflows_dir)
        .unwrap_or_else(|e| panic!("reading {}: {e}", workflows_dir.display()))
    {
        let entry = entry.expect("dir entry");
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("yml") {
            continue;
        }
        let rel = format!(
            ".github/workflows/{}",
            path.file_name().unwrap().to_string_lossy()
        );
        let content = read_to_string(&path);
        findings.extend(sweep_unaccounted(&rel, &content, &sites, NOT_SHIPPED));
    }

    let dockerfile = read_to_string(&root.join("Dockerfile"));
    findings.extend(sweep_unaccounted(
        "Dockerfile",
        &dockerfile,
        &sites,
        NOT_SHIPPED,
    ));

    assert!(
        findings.is_empty(),
        "a jammi-server feature-building site exists that this test does not account for -- add \
         it to a family's `site` (if shipped) or to `NOT_SHIPPED` (with a reason, if not):\n{}",
        findings.join("\n")
    );
}

#[test]
fn completeness_sweep_flags_an_unaccounted_feature_site() {
    // A synthetic new lane -- the shape of all four prior BLOCKs on this unit: a real shipped
    // build gains a feature that nothing enumerates. Never depends on today's tree, so this
    // stays a permanent proof that the sweep's classifier actually catches the failure mode,
    // not merely today's already-known sites.
    let synthetic = "        run: cargo build --release -p jammi-server --bin jammi-server --features storage-cloud,new-cloud-tier\n";
    let findings = sweep_unaccounted(
        ".github/workflows/synthetic-lane.yml",
        synthetic,
        &declared_sites(),
        NOT_SHIPPED,
    );
    assert!(
        !findings.is_empty(),
        "the completeness sweep failed to flag a synthetic jammi-server feature site that no \
         declared family site or NOT_SHIPPED entry covers -- this is exactly the failure mode \
         (a real shipped lane nobody enumerated) all four prior rounds on this unit blocked on"
    );
}

#[test]
fn completeness_sweep_does_not_flag_a_pure_template_reference() {
    // `${{ }}`/`$(...)` plumbing that resolves via a manifest read or a caller input must never
    // be flagged -- it carries no literal of its own to drift.
    let synthetic = "        run: cargo build --release -p jammi-server --bin jammi-server --features ${{ steps.manifest.outputs.cargo_features }}\n";
    let findings = sweep_unaccounted(
        ".github/workflows/synthetic-lane.yml",
        synthetic,
        &declared_sites(),
        NOT_SHIPPED,
    );
    assert!(
        findings.is_empty(),
        "a templated (non-literal) feature reference must never be flagged by the completeness \
         sweep: {findings:#?}"
    );
}

#[test]
fn generated_exposure_line_lists_every_family_carrying_the_gated_feature() {
    // Self-contained proof of the generator's own logic: a feature carried by every family
    // must produce a line naming every family, in FAMILY_ORDER.
    let cuda_features: BTreeMap<&str, Vec<String>> = CUDA_FAMILIES
        .iter()
        .map(|lane| (*lane, vec!["storage-cloud".to_string()]))
        .collect();
    let families = families_carrying("storage-cloud", &cuda_features);
    assert_eq!(
        families,
        vec![
            "cpu-wheel",
            "cpu-tarball",
            "cpu-image",
            "cu12-tarball",
            "cu12-wheel",
            "cu12-image",
        ]
    );
    assert_eq!(
        generated_exposure_line("RUSTSEC-TEST-0000", &families),
        "# EXPOSURE[RUSTSEC-TEST-0000] = cpu-wheel, cpu-tarball, cpu-image, cu12-tarball, cu12-wheel, cu12-image"
    );
}
