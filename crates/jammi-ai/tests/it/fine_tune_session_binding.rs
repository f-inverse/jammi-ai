//! No code under `crates/jammi-ai/src/fine_tune/` binds a name into the
//! shared `SessionContext`.
//!
//! A relation registered on the shared session under a per-job or per-spec
//! token collides under reclaim: two overlapping materializations of ONE job
//! id (a lease lost mid-sampling, then reclaimed by a second worker) bind the
//! same name, because the resource is scoped to one CALL while the name is
//! unique only per JOB. The fine-tune arm therefore reads through nameless
//! providers and the store's own registration path, never by binding a name
//! itself.
//!
//! **Why this is a source scan and not a type.** `fine_tune/` holds the
//! session (`self.context()`) for its SQL reads, and DataFusion's
//! registration verbs are inherent methods on that very `SessionContext` —
//! no type in this workspace can take them away from one module while
//! leaving it the reads. So the property is stated over the source, with
//! ZERO tolerance and NO allow list: the target count is zero,
//! unconditionally, and a future direct call under `fine_tune/` fails this
//! test rather than earning a reviewed entry.
//!
//! The scan walks the REAL syntax tree (`syn`), never source text: a verb or
//! a DDL keyword in a comment, a doc comment or an attribute is never a
//! hit, a call inside a macro invocation's tokens (`assert!`,
//! `tokio::select!`, `format!`) is. The universe is every git-TRACKED `.rs`
//! file under the directory — `git ls-files` recurses on its own — and a
//! tracked file this process cannot read or parse is a hard failure naming
//! it, never a silent skip.

use std::path::{Path, PathBuf};
use std::process::Command;

use syn::visit::Visit;

/// The directory the property quantifies over, relative to the repo root.
const FINE_TUNE_DIR: &str = "crates/jammi-ai/src/fine_tune";

/// DataFusion verbs with BOTH a `register_`/`deregister_` form, each binding
/// the shared session under a caller-chosen token. Seven are
/// `SessionContext` methods; `schema` is a `CatalogProvider` method reached
/// through `ctx.catalog(..)` — the scan matches the call SHAPE, not one
/// receiver type.
const PAIRED_VERBS: &[&str] = &[
    "table",
    "object_store",
    "udtf",
    "udf",
    "udaf",
    "udwf",
    "higher_order_function",
    "schema",
];

/// `SessionContext` verbs with only a `register_` form, each a wrapper that
/// binds a table under a caller-chosen `TableReference` one or two calls
/// removed from `register_table` — the same collision surface under a
/// different spelling.
const UNPAIRED_VERBS: &[&str] = &[
    "batch",
    "csv",
    "json",
    "parquet",
    "avro",
    "listing_table",
    "arrow",
    "catalog",
];

/// Every method name the scan treats as a session binding.
fn binding_verbs() -> Vec<String> {
    PAIRED_VERBS
        .iter()
        .flat_map(|verb| [format!("register_{verb}"), format!("deregister_{verb}")])
        .chain(UNPAIRED_VERBS.iter().map(|verb| format!("register_{verb}")))
        .collect()
}

/// Whether `text` is a DDL statement that binds a relation into a session's
/// catalog when handed to `SessionContext::sql` — `CREATE [OR REPLACE]
/// {VIEW | TABLE | EXTERNAL TABLE | SCHEMA}`, matched case-insensitively
/// over identifier tokens (so `create_view_something` is one identifier,
/// never the two words). `CREATE OR REPLACE` is deregister-then-register on
/// the same token and so binds exactly like the bare form.
fn binds_a_relation(text: &str) -> bool {
    let lower = text.to_ascii_lowercase();
    let tokens: Vec<&str> = lower
        .split(|c: char| !c.is_ascii_alphanumeric() && c != '_')
        .filter(|s| !s.is_empty())
        .collect();
    tokens.iter().enumerate().any(|(i, tok)| {
        if *tok != "create" {
            return false;
        }
        let mut j = i + 1;
        if tokens.get(j) == Some(&"or") && tokens.get(j + 1) == Some(&"replace") {
            j += 2;
        }
        match tokens.get(j) {
            Some(&"view") | Some(&"table") | Some(&"schema") => true,
            Some(&"external") => tokens.get(j + 1) == Some(&"table"),
            _ => false,
        }
    })
}

/// One binding found in a file: which verb or DDL keyword, and where.
#[derive(Debug, PartialEq, Eq)]
struct Hit {
    line: usize,
    what: String,
}

/// The syntax-tree walk: a method call or path call whose name is a binding
/// verb, a string literal in expression position that binds a relation, and
/// the same two shapes inside any macro invocation's token stream.
/// Attributes (including the `#[doc = ".."]` a doc comment becomes) are
/// never descended into.
struct BindingScan {
    verbs: Vec<String>,
    hits: Vec<Hit>,
}

impl BindingScan {
    fn is_verb(&self, ident: &syn::Ident) -> bool {
        self.verbs.iter().any(|v| ident == v)
    }

    fn scan_tokens(&mut self, tokens: proc_macro2::TokenStream) {
        let mut iter = tokens.into_iter().peekable();
        while let Some(tree) = iter.next() {
            match tree {
                proc_macro2::TokenTree::Ident(ident) => {
                    let called = matches!(
                        iter.peek(),
                        Some(proc_macro2::TokenTree::Group(g))
                            if g.delimiter() == proc_macro2::Delimiter::Parenthesis
                    );
                    if called && self.is_verb(&ident) {
                        self.hits.push(Hit {
                            line: ident.span().start().line,
                            what: ident.to_string(),
                        });
                    }
                }
                proc_macro2::TokenTree::Literal(lit) => {
                    if let Ok(syn::Lit::Str(s)) = syn::parse_str::<syn::Lit>(&lit.to_string()) {
                        if binds_a_relation(&s.value()) {
                            self.hits.push(Hit {
                                line: lit.span().start().line,
                                what: "DDL".to_string(),
                            });
                        }
                    }
                }
                proc_macro2::TokenTree::Group(group) => self.scan_tokens(group.stream()),
                proc_macro2::TokenTree::Punct(_) => {}
            }
        }
    }
}

impl<'ast> Visit<'ast> for BindingScan {
    fn visit_attribute(&mut self, _: &'ast syn::Attribute) {}

    fn visit_expr_method_call(&mut self, call: &'ast syn::ExprMethodCall) {
        if self.is_verb(&call.method) {
            self.hits.push(Hit {
                line: call.method.span().start().line,
                what: call.method.to_string(),
            });
        }
        syn::visit::visit_expr_method_call(self, call);
    }

    fn visit_expr_call(&mut self, call: &'ast syn::ExprCall) {
        if let syn::Expr::Path(path) = call.func.as_ref() {
            if let Some(last) = path.path.segments.last() {
                if self.is_verb(&last.ident) {
                    self.hits.push(Hit {
                        line: last.ident.span().start().line,
                        what: last.ident.to_string(),
                    });
                }
            }
        }
        syn::visit::visit_expr_call(self, call);
    }

    fn visit_lit_str(&mut self, lit: &'ast syn::LitStr) {
        if binds_a_relation(&lit.value()) {
            self.hits.push(Hit {
                line: lit.span().start().line,
                what: "DDL".to_string(),
            });
        }
    }

    fn visit_macro(&mut self, mac: &'ast syn::Macro) {
        self.scan_tokens(mac.tokens.clone());
    }
}

/// Every session binding in one Rust source text.
fn bindings_in(source: &str) -> Vec<Hit> {
    let file = syn::parse_file(source).expect("the source parses as Rust");
    let mut scan = BindingScan {
        verbs: binding_verbs(),
        hits: Vec::new(),
    };
    scan.visit_file(&file);
    scan.hits
}

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crates/jammi-ai has two ancestors: crates/, then the repo root")
        .to_path_buf()
}

/// Every git-tracked `.rs` file under [`FINE_TUNE_DIR`], as
/// `(repo-relative path, source)`; a tracked file that cannot be read is a
/// hard failure naming it.
fn fine_tune_sources() -> Vec<(String, String)> {
    let root = repo_root();
    let output = Command::new("git")
        .args([
            "-C",
            root.to_str().expect("utf8 repo root"),
            "ls-files",
            "--",
            FINE_TUNE_DIR,
        ])
        .output()
        .expect("spawn git ls-files");
    assert!(
        output.status.success(),
        "git ls-files -- {FINE_TUNE_DIR} failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let files: Vec<(String, String)> = String::from_utf8(output.stdout)
        .expect("utf8 git ls-files output")
        .lines()
        .filter(|line| line.ends_with(".rs"))
        .map(|rel| {
            let text = std::fs::read_to_string(root.join(rel))
                .unwrap_or_else(|e| panic!("{rel} is git-tracked but could not be read ({e})"));
            (rel.to_string(), text)
        })
        .collect();
    assert!(
        !files.is_empty(),
        "git ls-files -- {FINE_TUNE_DIR} returned no tracked .rs files — the repo root or the \
         pathspec is wrong, which would make this test vacuous"
    );
    files
}

#[test]
fn fine_tune_never_binds_a_name_on_the_shared_session() {
    let hits: Vec<(String, Hit)> = fine_tune_sources()
        .into_iter()
        .flat_map(|(file, text)| {
            bindings_in(&text)
                .into_iter()
                .map(move |hit| (file.clone(), hit))
        })
        .collect();
    assert!(
        hits.is_empty(),
        "no file under {FINE_TUNE_DIR} may call a session-binding verb (a \
         `SessionContext`/`CatalogProvider` register_/deregister_ verb) or hand a DDL literal \
         that binds a relation (CREATE VIEW/TABLE/EXTERNAL TABLE/SCHEMA) to the session — the \
         shared session is not a per-call namespace, and any name that is not unique per CALL \
         collides under reclaim. Hits: {hits:?}"
    );
}

#[test]
fn falsification_a_direct_verb_call_is_a_hit_and_a_comment_is_not() {
    let source = r#"
        /// A doc comment that spells ctx.register_table(..) is prose.
        // So is a line comment: ctx.deregister_table("x").
        fn bind(ctx: &SessionContext) {
            ctx.register_table("jobs", provider).unwrap();
            ctx.catalog("datafusion").unwrap().register_schema("s", schema);
            SessionContext::register_udf(ctx, udf);
        }
    "#;
    let hits = bindings_in(source);
    assert_eq!(
        hits.iter().map(|h| h.what.as_str()).collect::<Vec<_>>(),
        vec!["register_table", "register_schema", "register_udf"],
        "{hits:?}"
    );
}

#[test]
fn falsification_a_verb_with_no_caller_chosen_token_is_not_a_hit() {
    let source = r#"
        fn configure(ctx: &SessionContext) {
            ctx.register_variable(VarType::System, provider);
            ctx.register_relation_planner(planner);
            ctx.register_catalog_list(list);
            ctx.register_table_options_extension(ext);
        }
    "#;
    assert!(bindings_in(source).is_empty());
}

#[test]
fn falsification_a_ddl_literal_is_a_hit_in_an_expression_and_inside_a_macro() {
    let source = r#"
        /// Prose mentioning CREATE TABLE is not a hit.
        async fn bind(ctx: &SessionContext, t: &str) {
            ctx.sql("create or replace view v as select 1").await.unwrap();
            let ddl = format!("CREATE EXTERNAL TABLE {t} STORED AS PARQUET LOCATION 'x'");
            assert!(ctx.sql(&format!("CREATE SCHEMA {t}")).await.is_ok());
            let plain = "SELECT * FROM t WHERE create_view_count > 1";
        }
    "#;
    let hits = bindings_in(source);
    assert_eq!(
        hits.iter().map(|h| h.what.as_str()).collect::<Vec<_>>(),
        vec!["DDL", "DDL", "DDL"],
        "{hits:?}"
    );
}

#[test]
fn falsification_a_verb_inside_a_macro_invocation_is_a_hit() {
    let source = r#"
        fn bind(ctx: &SessionContext) {
            assert!(ctx.register_parquet("t", "x.parquet", opts).is_ok());
        }
    "#;
    let hits = bindings_in(source);
    assert_eq!(hits.len(), 1, "{hits:?}");
    assert_eq!(hits[0].what, "register_parquet");
}
