//! `docs_toml_fences_parse_under_the_real_loader`: every ```toml fence under
//! `docs/guide/src` whose first non-blank, non-comment line names a
//! top-level `JammiConfig`
//! field parses under the REAL loader (`JammiConfig::parse_from`), not a
//! hand-copied fixture. A guide example that stops matching the config
//! shape it documents fails here, naming `file:line`, instead of silently
//! rotting until a reader copy-pastes it and hits a load-time error the
//! guide never warned about.
//!
//! # Selection
//!
//! A fence is selected when its first non-blank, non-comment line (leading
//! `#` lines stripped) is `[<section>…]` or `<key> = …` and `<section>`/`<key>`
//! is one of `JammiConfig`'s own top-level field names. This is what keeps a
//! `[dependencies]` Cargo fence (`installation.md`, `cloud-storage.md`) out
//! of the walk while still catching `cloud-storage.md`'s `[storage.cloud.*]`
//! fences and `configuration.md`'s bare `broker = "in_memory"` /
//! `signing_key = "env"` fences.
//!
//! # `${NAME}` and `{ file = "…" }`
//!
//! Every `${NAME}` in a selected fence is resolved from a placeholder env
//! map (`NAME` -> `"x"`). A `{ file = "…" }` secret form
//! is real at deserialization time — `Secret::deserialize` reads the named
//! file eagerly (see `crate::config::secret`), which a guide's illustrative
//! path (`/run/secrets/pg-url`, `/etc/jammi/sa.json`, …) never resolves on a
//! test runner. Rather than forbid the shape our own docs use (and MUST use,
//! since it is the shape deployers copy), this test neutralizes it: every
//! `file = "…"` path is rewritten to one shared placeholder file this test
//! creates, before `parse_from` ever sees the fence. The doc's own text is
//! never touched; only the string handed to the loader in-memory is.
//!
//! # `{{#include …}}` fences
//!
//! A fence whose body is a single `{{#include <relpath>}}` line (mdbook's
//! include syntax) is resolved before selection: `<relpath>` is joined
//! against the `.md` file's own directory and that file's contents replace
//! the fence body, so a manifest included this way is walked and parsed
//! exactly like a fence typed directly into the guide. `file`/`line` still
//! point at the fence in the guide (not the included file), so a failure
//! still names the guide location a reader would look at first; the
//! resolved include path is appended to the failure message.
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use jammi_db::config::JammiConfig;

/// Mirrors `jammi_db::config::env_map::TOP_LEVEL_FIELDS` (crate-private).
/// Duplicated here, as a literal list rather than a re-export, on purpose:
/// this is the doc-selection oracle walking the PUBLIC contract, and
/// `top_level_fields_matches_jammi_config` in `jammi-db`'s own
/// `config::tests` independently pins the private list against
/// `JammiConfig`'s actual fields, so a drift between the two lists is caught
/// there, not smuggled in here by a shared `pub(crate)` constant.
const TOP_LEVEL_FIELDS: &[&str] = &[
    "artifact_dir",
    "engine",
    "gpu",
    "inference",
    "embedding",
    "fine_tuning",
    "lease",
    "worker",
    "jobs",
    "cache",
    "server",
    "logging",
    "observability",
    "catalog",
    "broker",
    "signing_key",
    "storage",
    "models",
];

fn guide_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../docs/guide/src")
}

/// One ```toml fence extracted from a guide page.
struct Fence {
    file: PathBuf,
    /// 1-based line number of the fence's first BODY line (the line after
    /// the opening ```toml marker), so a reported failure points a reader
    /// straight at the offending TOML, not the marker line.
    line: usize,
    body: String,
}

fn markdown_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let entries = fs::read_dir(dir).unwrap_or_else(|e| panic!("reading {}: {e}", dir.display()));
    for entry in entries {
        let entry = entry.unwrap();
        let path = entry.path();
        if path.is_dir() {
            markdown_files(&path, out);
        } else if path.extension().and_then(|e| e.to_str()) == Some("md") {
            out.push(path);
        }
    }
}

fn extract_toml_fences(path: &Path) -> Vec<Fence> {
    let text =
        fs::read_to_string(path).unwrap_or_else(|e| panic!("reading {}: {e}", path.display()));
    let mut fences = Vec::new();
    let mut lines = text.lines().enumerate();
    while let Some((i, line)) = lines.next() {
        if line.trim_start() != "```toml" {
            continue;
        }
        let start_line = i + 2; // 1-based line number of the first body line
        let mut body = String::new();
        for (_, l) in lines.by_ref() {
            if l.trim_start() == "```" {
                break;
            }
            body.push_str(l);
            body.push('\n');
        }
        fences.push(Fence {
            file: path.to_path_buf(),
            line: start_line,
            body,
        });
    }
    fences
}

/// If `body`'s only non-blank line is `{{#include <relpath>}}` (mdbook's
/// include syntax), resolve `<relpath>` against `md_file`'s own directory,
/// read that file, and return its contents in place of `body`, along with
/// the resolved path (for the failure message). Otherwise return `body`
/// unchanged and `None` — a fence typed directly into the guide is never
/// touched.
fn resolve_include(md_file: &Path, body: &str) -> (String, Option<PathBuf>) {
    let non_blank: Vec<&str> = body.lines().filter(|l| !l.trim().is_empty()).collect();
    let [line] = non_blank[..] else {
        return (body.to_string(), None);
    };
    let line = line.trim();
    let Some(relpath) = line
        .strip_prefix("{{#include ")
        .and_then(|rest| rest.strip_suffix("}}"))
    else {
        return (body.to_string(), None);
    };
    let relpath = relpath.trim();
    let dir = md_file.parent().unwrap_or_else(|| Path::new("."));
    let resolved = dir.join(relpath);
    let included = fs::read_to_string(&resolved).unwrap_or_else(|e| {
        panic!(
            "resolving {{{{#include {relpath}}}}} from {}: {e}",
            md_file.display()
        )
    });
    (included, Some(resolved))
}

/// Whether `body`'s first non-blank, non-comment line names a top-level
/// `JammiConfig` field, as `[<section>…]` or `<key> = …`.
fn is_selected(body: &str) -> bool {
    for raw in body.lines() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let name = if let Some(rest) = line.strip_prefix('[') {
            rest.trim_start_matches('[')
                .split(['.', ']'])
                .next()
                .unwrap_or("")
                .trim()
        } else if let Some(eq) = line.find('=') {
            line[..eq].trim()
        } else {
            ""
        };
        return TOP_LEVEL_FIELDS.contains(&name);
    }
    false
}

/// Rewrite every `file = "…"` secret path in `body` to `placeholder`
/// (a real file this test created) so `Secret::deserialize`'s eager file
/// read succeeds regardless of the illustrative path the guide shows.
fn neutralize_secret_files(body: &str, placeholder: &str) -> String {
    const MARK: &str = "file = \"";
    let mut out = String::with_capacity(body.len());
    let mut rest = body;
    loop {
        match rest.find(MARK) {
            Some(idx) => {
                out.push_str(&rest[..idx]);
                out.push_str(MARK);
                let after = &rest[idx + MARK.len()..];
                let end = after
                    .find('"')
                    .unwrap_or_else(|| panic!("unterminated file = \"...\" in fence:\n{body}"));
                out.push_str(placeholder);
                out.push('"');
                rest = &after[end + 1..];
            }
            None => {
                out.push_str(rest);
                break;
            }
        }
    }
    out
}

/// Build the placeholder env map the module docs describe: every `${NAME}` in
/// `body` maps to `"x"`.
fn placeholder_env(body: &str) -> BTreeMap<String, String> {
    let mut env = BTreeMap::new();
    let bytes = body.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'$' && bytes.get(i + 1) == Some(&b'{') {
            if let Some(off) = body[i + 2..].find('}') {
                let name = &body[i + 2..i + 2 + off];
                if !name.is_empty() && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
                    env.insert(name.to_string(), "x".to_string());
                }
                i += 2 + off + 1;
                continue;
            }
        }
        i += 1;
    }
    env
}

#[test]
fn docs_toml_fences_parse_under_the_real_loader() {
    let dir = tempfile::tempdir().unwrap();
    let placeholder = dir.path().join("placeholder-secret");
    fs::write(&placeholder, "placeholder-secret\n").unwrap();
    let placeholder = placeholder.to_str().unwrap().to_string();

    let mut files = Vec::new();
    markdown_files(&guide_root(), &mut files);
    files.sort();
    assert!(
        !files.is_empty(),
        "no markdown files found under {}",
        guide_root().display()
    );

    let mut selected = 0usize;
    let mut failures = Vec::new();
    for file in &files {
        for fence in extract_toml_fences(file) {
            let (resolved_body, included_from) = resolve_include(&fence.file, &fence.body);
            if !is_selected(&resolved_body) {
                continue;
            }
            selected += 1;
            let neutralized = neutralize_secret_files(&resolved_body, &placeholder);
            let env = placeholder_env(&neutralized);
            if let Err(e) = JammiConfig::parse_from(&neutralized, env) {
                let mut msg = format!("{}:{}: {e}", fence.file.display(), fence.line);
                if let Some(included_from) = &included_from {
                    msg.push_str(&format!(" (included from {})", included_from.display()));
                }
                failures.push(msg);
            }
        }
    }

    // Pinned, not just `> 0`: a silent DROP in the selected count (the
    // selection rule drifting away from what the guide actually writes) is
    // just as much a coverage regression as selecting zero fences, and
    // `> 0` alone would stay green through it. Bump this number in the same
    // commit that adds (or removes) a `JammiConfig`-shaped ```toml fence
    // under `docs/guide/src` -- including one that arrives only as a
    // resolved `{{#include}}`. 27 direct fences + 2 includes
    // (`deploy/kubernetes/base/jammi.toml`, `deploy/kubernetes/overlays/shape-d/jammi-compute.toml`)
    // = 29; `deploy/kubernetes/overlays/ci/jammi.toml` is not included by
    // the guide, so it is not counted here.
    assert_eq!(
        selected,
        29,
        "selected {selected} config fence(s) under {} -- expected exactly 29; if you \
         added or removed a JammiConfig-shaped ```toml fence (directly or via {{{{#include}}}}), \
         update this pinned count",
        guide_root().display()
    );
    assert!(
        failures.is_empty(),
        "config fence(s) failed to parse under the real loader:\n{}",
        failures.join("\n")
    );
}

/// Pinned negative control: a fence with an unknown key fails, naming it —
/// proves the walk above is not vacuously green because every failure mode
/// is swallowed somewhere upstream of `parse_from`.
#[test]
fn unknown_key_in_a_docs_shaped_fence_fails_naming_it() {
    let toml = "[catalog.sqlite]\nbogus_field = 1\n";
    let err = JammiConfig::parse_from(toml, std::iter::empty::<(String, String)>())
        .expect_err("an unknown key under a known section must be refused");
    let msg = err.to_string();
    assert!(
        msg.contains("bogus_field"),
        "error did not name the unknown key: {msg}"
    );
}

#[test]
fn fence_selection_skips_a_cargo_dependencies_fence() {
    let body = "[dependencies]\njammi-db = { version = \"0.5\" }\n";
    assert!(
        !is_selected(body),
        "a Cargo [dependencies] fence must not be selected as a config fence"
    );
}
